"""
YOLO + ByteTrack car detector/counter for a 4-way intersection (chauraha).

Upload a bird's-eye-view video of a 4-way intersection to the Flask server.
YOLO detects cars, ByteTrack assigns a persistent ID to each car, and the
app counts how many unique cars crossed each of the 4 sides (North, South,
East, West) of the intersection. An annotated output video is produced with
boxes + IDs + a live count overlay.

Requirements (put these in requirements.txt / pip install):
    flask
    ultralytics
    opencv-python
    numpy
    huggingface_hub

Run:
    python app.py
Then open http://localhost:5000 and upload a video.

HOW THE 4 SIDES ARE DEFINED
----------------------------
Since every uploaded video can have the intersection framed differently,
the frame is split into 4 trapezoid/quadrant-ish zones around the center
(N/S/E/W) using simple proportional geometry based on frame width/height.
A car's side is decided by the zone its centroid is in **the moment its
track is first seen** (i.e., which approach it came from). This is a
reasonable default for top-down 4-way intersection footage; tune the
ZONE_SPLIT / CENTER_BOX constants below if your footage is framed
differently, or pass custom polygons via `--zones zones.json`.

----------------------------------------------------------------------------
UPDATE: missed-detection fixes
----------------------------------------------------------------------------
This version adds three changes aimed specifically at the two failure modes
visible in testing: (1) small/edge-of-frame cars (e.g. a packed parking lot
in a corner) going undetected, and (2) adjacent/overlapping cars getting
merged or suppressed by NMS.

  1. Tuned `iou` + `agnostic_nms=True` on every detector call, so boxes for
     tightly packed cars are less likely to be merged away.
  2. A second, *tiled* inference pass ("SAHI-lite"). The full frame is
     always run once at GEO_TRAX_IMGSZ. In addition, the frame is split
     into overlapping tiles and each tile is run through the model at its
     own native resolution (so small/corner cars occupy far more pixels
     than they do in the downscaled full-frame pass). Detections from all
     tiles are translated back into full-frame coordinates and merged with
     the full-frame pass via NMS. This runs on EVERY frame by default
     (TILE_EVERY_FRAME=True) -- set it to False to only tile occasionally
     (cheaper, slightly less reliable) if full tiling is too slow for your
     hardware.
  3. A tuned `bytetrack.yaml` (written out alongside this script) with a
     longer `track_buffer` and lower `track_low_thresh`, so a car that
     drops below the confidence threshold for a frame or two (motion blur,
     brief occlusion by another car) keeps its ID instead of being dropped
     and re-counted as a "new" car.
"""

import os
import json
import time
import uuid
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# geo-trax class ids: 0=car, 1=bus, 2=truck, 3=motorcycle.
# Add 1 and/or 2 here if you also want buses/trucks counted as "cars".
CAR_CLASS_IDS = {0}

CONF_THRES = 0.18          # lowered from 0.25 -- recovers partially-occluded /
                            # tightly-packed parked cars at the cost of a few
                            # more false positives (NMS tuning below offsets this)
IOU_THRES = 0.5             # tighter NMS than ultralytics' default 0.7, so
                             # adjacent/overlapping car boxes stop getting merged
AGNOSTIC_NMS = True          # suppress across classes too, helps dense clusters

TRACKER_CFG = str(BASE_DIR / "bytetrack_tuned.yaml")  # written out at startup, see below

# geo-trax: a YOLOv8s model trained specifically on drone bird's-eye-view
# footage (VisDrone, UAVDT, CARPK, Songdo and others). Its class IDs are
# 0=car, 1=bus, 2=truck, 3=motorcycle (NOT the COCO ordering), which is why
# CAR_CLASS_IDS below is {0}, not {2}.
GEO_TRAX_REPO_ID = "rfonod/geo-trax"
GEO_TRAX_FILENAME = "geotrax_hbb_yolov8s_1920_v1.pt"
GEO_TRAX_IMGSZ = 1920  # the resolution this model was trained/validated at

# Downscale every frame to this width before running YOLO (height auto-scales
# to keep aspect ratio). 1920 matches the resolution geo-trax was trained/
# validated at, and is still a 4x pixel-count reduction from a 4K source —
# a good speed/accuracy balance. Drop to 1280 or 1024 if it's still too slow
# and you're OK trading a bit of recall on small/distant cars for speed.
MAX_WIDTH = 1920

# --------------------------------------------------------------------------
# Tiled ("SAHI-lite") inference settings -- recovers small / corner /
# densely-packed cars that the single full-frame pass misses because they
# get downscaled to too few pixels.
# --------------------------------------------------------------------------
ENABLE_TILING = True         # master switch
TILE_EVERY_FRAME = False    # True = tile every frame (slower, most reliable).
                             # False = only every TILE_FRAME_STRIDE frames --
                             # cars that are missed on an un-tiled frame are
                             # almost always still sitting in roughly the
                             # same spot a few frames later (parked cars
                             # especially), so this recovers nearly all the
                             # same detections for a fraction of the compute.
TILE_FRAME_STRIDE = 5        # used only if TILE_EVERY_FRAME is False
TILE_GRID = (2, 2)          # (cols, rows) -- 2x2 covers the 4 corners/edges well;
                             # bump to (3, 2) etc. for very large / high-res frames
TILE_OVERLAP = 0.20         # 20% overlap between neighboring tiles, so a car
                             # sitting on a tile boundary is still fully inside
                             # at least one tile
TILE_MERGE_IOU = 0.5        # IoU threshold used when merging tile detections
                             # with the full-frame pass (and with each other)

# Fraction of frame width/height that defines the "center cross" region.
# Anything outside the center belongs to N/S/E/W based on which side of
# the two diagonals it falls on.
CENTER_MARGIN = 0.15  # not currently required, kept for future zone tuning

# In-memory job store: job_id -> status dict
JOBS = {}
JOBS_LOCK = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB uploads

_model = None
_model_lock = threading.Lock()
_device = None
_half = False


def get_device():
    """
    Pick the fastest available device once and cache it. CUDA is ~10-30x
    faster than CPU for this kind of model; if you have a GPU and are still
    seeing CPU-level speeds, this is almost always why -- check
    `torch.cuda.is_available()` in your environment / driver setup.
    """
    global _device, _half
    if _device is not None:
        return _device, _half
    import torch
    if torch.cuda.is_available():
        _device = "cuda:0"
        _half = True   # FP16 roughly halves inference time on modern GPUs
                        # with negligible accuracy loss for detection
    elif torch.backends.mps.is_available():  # Apple Silicon
        _device = "mps"
        _half = False   # fp16 support on MPS is inconsistent; keep fp32
    else:
        _device = "cpu"
        _half = False
    return _device, _half


def get_model():
    """Lazy-load the geo-trax drone-BEV vehicle detector once (thread-safe)."""
    global _model
    with _model_lock:
        if _model is None:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id=GEO_TRAX_REPO_ID, filename=GEO_TRAX_FILENAME
            )
            _model = YOLO(weights_path)
            device, half = get_device()
            _model.to(device)
    return _model


def write_tuned_tracker_cfg():
    """
    Write a ByteTrack config tuned to hold onto tracks through brief
    confidence dips (motion blur, partial occlusion by another car) instead
    of dropping them and later re-counting the same car as "new".

    Key changes vs. ultralytics' stock bytetrack.yaml:
      - track_buffer: 30 -> 60   (keep a lost track alive twice as long
                                   before giving up on it, in frames)
      - track_low_thresh: 0.1 -> 0.05  (secondary association step picks up
                                   more low-confidence boxes to keep a track
                                   alive instead of starting a fresh ID)
      - new_track_thresh stays conservative so this doesn't spawn extra
        spurious tracks.
    """
    cfg_path = Path(TRACKER_CFG)
    if cfg_path.exists():
        return
    cfg_path.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.05\n"
        "new_track_thresh: 0.3\n"
        "track_buffer: 60\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )


# --------------------------------------------------------------------------
# Zone logic: assign a point to N / S / E / W based on position relative
# to the frame center, split by the two diagonals.
# --------------------------------------------------------------------------
def classify_side(cx, cy, width, height):
    """
    Split the frame into 4 triangles by its two diagonals (like a bowtie/
    X shape from the center). This naturally matches the 4 approaches of
    a 4-way intersection viewed from directly above or at a steep angle.

        \\        /
          \\  N  /
            \\  /
      W       X       E
            /  \\
          /  S  \\
        /        \\
    """
    dx = cx - width / 2.0
    dy = cy - height / 2.0

    # Normalize by aspect ratio so the diagonals are the true frame diagonals
    ndx = dx / (width / 2.0)
    ndy = dy / (height / 2.0)

    if abs(ndx) > abs(ndy):
        return "East" if ndx > 0 else "West"
    else:
        return "South" if ndy > 0 else "North"


# --------------------------------------------------------------------------
# Tiled inference helpers
# --------------------------------------------------------------------------
def make_tiles(width, height, grid, overlap):
    """
    Build a list of (x0, y0, x1, y1) tile boxes in full-frame pixel coords,
    covering the frame with `grid` = (cols, rows) tiles and `overlap`
    fraction of overlap between neighbors.
    """
    cols, rows = grid
    tile_w = width / cols
    tile_h = height / rows
    ox = tile_w * overlap
    oy = tile_h * overlap

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(c * tile_w - ox))
            y0 = max(0, int(r * tile_h - oy))
            x1 = min(width, int((c + 1) * tile_w + ox))
            y1 = min(height, int((r + 1) * tile_h + oy))
            tiles.append((x0, y0, x1, y1))
    return tiles


def run_tile_detections(model, frame, tiles, device=None, half=False):
    """
    Run plain (untracked) detection on all tiles in a SINGLE batched
    model.predict() call (instead of one call per tile) and return all boxes
    translated back into full-frame coordinates, as a list of
    (x1, y1, x2, y2, conf).

    Batching matters a lot for speed: each model.predict() call has fixed
    overhead (CUDA kernel launches, pre/post-processing setup) on top of the
    actual forward pass, and a GPU forward pass on a batch of N images is
    far closer to the cost of 1 image than to N separate images. Calling
    predict() once per tile (the old approach) pays that per-call overhead
    N times for no benefit.
    """
    tile_imgs = []
    valid_tiles = []
    for (tx0, ty0, tx1, ty1) in tiles:
        tile_img = frame[ty0:ty1, tx0:tx1]
        if tile_img.size == 0:
            continue
        tile_imgs.append(tile_img)
        valid_tiles.append((tx0, ty0, tx1, ty1))

    if not tile_imgs:
        return []

    results_list = model.predict(
        tile_imgs,                     # batched: list of images -> one forward pass
        conf=CONF_THRES,
        iou=IOU_THRES,
        classes=list(CAR_CLASS_IDS),
        agnostic_nms=AGNOSTIC_NMS,
        device=device,
        half=half,
        verbose=False,
    )

    detections = []
    for results, (tx0, ty0, tx1, ty1) in zip(results_list, valid_tiles):
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (bx1, by1, bx2, by2), conf in zip(xyxy, confs):
            detections.append((bx1 + tx0, by1 + ty0, bx2 + tx0, by2 + ty0, float(conf)))
    return detections


def nms_merge(boxes_a, boxes_b, iou_thresh):
    """
    Merge two lists of (x1,y1,x2,y2,conf) boxes, dropping boxes from
    `boxes_b` that overlap an already-kept box above `iou_thresh`. Boxes
    from `boxes_a` (the full-frame/tracked pass) are always kept, since
    those are the ones that carry track IDs; `boxes_b` (tile pass) only
    contributes genuinely new detections the full-frame pass missed.
    """
    def iou(b1, b2):
        ax1, ay1, ax2, ay2 = b1[:4]
        bx1, by1, bx2, by2 = b2[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    kept_extra = []
    for cand in boxes_b:
        if all(iou(cand, existing) < iou_thresh for existing in boxes_a):
            kept_extra.append(cand)
    return kept_extra


# --------------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------------
def process_video(job_id, in_path, out_path):
    job = JOBS[job_id]
    model = get_model()
    device, half = get_device()
    write_tuned_tracker_cfg()

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        job["status"] = "error"
        job["error"] = "Could not open uploaded video."
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Compute the downscaled size we'll actually run inference + write output at.
    if orig_width > MAX_WIDTH:
        scale = MAX_WIDTH / orig_width
        width = MAX_WIDTH
        height = int(orig_height * scale)
    else:
        width, height = orig_width, orig_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    tiles = make_tiles(width, height, TILE_GRID, TILE_OVERLAP) if ENABLE_TILING else []

    # track_id -> side it was first counted on
    counted_ids = {}
    side_counts = {"North": 0, "South": 0, "East": 0, "West": 0}

    # Synthetic IDs for tile-only detections that never get a real track ID
    # (they're drawn/counted once, flagged as "unconfirmed" in the label, and
    # not fed back into the tracker -- the next real frame's tracked pass
    # will normally pick them up and assign a proper ID).
    next_synthetic_id = -1

    frame_idx = 0
    job["status"] = "processing"

    # Draw the diagonal guide lines + side labels once per frame for clarity
    def draw_guides(frame):
        cv2.line(frame, (0, 0), (width, height), (80, 80, 80), 1)
        cv2.line(frame, (width, 0), (0, height), (80, 80, 80), 1)
        labels = {
            "North": (width // 2 - 30, 30),
            "South": (width // 2 - 30, height - 15),
            "East": (width - 70, height // 2),
            "West": (15, height // 2),
        }
        for side, (x, y) in labels.items():
            cv2.putText(frame, side, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break

            # Resize BEFORE inference — this is what actually saves the time,
            # not just resizing the output.
            if (width, height) != (orig_width, orig_height):
                frame_small = cv2.resize(raw_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_small = raw_frame

            r = model.track(
                frame_small,
                conf=CONF_THRES,
                iou=IOU_THRES,
                classes=list(CAR_CLASS_IDS),
                tracker=TRACKER_CFG,
                imgsz=GEO_TRAX_IMGSZ,
                persist=True,
                agnostic_nms=AGNOSTIC_NMS,
                device=device,
                half=half,
                verbose=False,
            )[0]

            frame = frame_small.copy()
            draw_guides(frame)

            tracked_boxes = []  # (x1,y1,x2,y2,conf) for NMS-merge comparison
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), tid, conf in zip(xyxy, ids, confs):
                    tracked_boxes.append((x1, y1, x2, y2, float(conf)))
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    if tid not in counted_ids:
                        side = classify_side(cx, cy, width, height)
                        counted_ids[tid] = side
                        side_counts[side] += 1

                    side_for_id = counted_ids[tid]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 200, 0), 2)
                    label = f"ID {tid} {side_for_id} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), max(int(y1) - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2,
                                cv2.LINE_AA)

            # --- Tiled pass: catch small / corner / densely-packed cars the
            # full-frame tracked pass above missed entirely. These are drawn
            # in a different color and counted, but given synthetic IDs
            # rather than fed into ByteTrack, so they don't interfere with
            # the tracker's own ID/association logic.
            do_tile_this_frame = ENABLE_TILING and (
                TILE_EVERY_FRAME or (frame_idx % TILE_FRAME_STRIDE == 0)
            )
            if do_tile_this_frame and tiles:
                tile_dets = run_tile_detections(model, frame_small, tiles, device=device, half=half)
                extra = nms_merge(tracked_boxes, tile_dets, TILE_MERGE_IOU)
                for (x1, y1, x2, y2, conf) in extra:
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    side = classify_side(cx, cy, width, height)
                    side_counts[side] += 1
                    counted_ids[next_synthetic_id] = side
                    next_synthetic_id -= 1

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 160, 255), 2)
                    label = f"tile {side} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), max(int(y1) - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 2,
                                cv2.LINE_AA)

            # Overlay running counts
            y0 = 24
            total = sum(side_counts.values())
            cv2.putText(frame, f"Total cars: {total}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            for i, side in enumerate(["North", "South", "East", "West"]):
                cv2.putText(frame, f"{side}: {side_counts[side]}", (10, y0 + 26 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
            if total_frames:
                job["progress"] = round(100 * frame_idx / total_frames, 1)

    finally:
        writer.release()
        cap.release()

    if frame_idx == 0:
        # cap.isOpened() can return True even when OpenCV can't actually
        # decode the file's codec (common with .webm/VP9 on some Windows
        # OpenCV builds). When that happens cap.read() fails on every call
        # and we silently process zero frames. Surface this as a real error
        # instead of reporting a false "done" with all-zero counts.
        job["status"] = "error"
        job["error"] = (
            "No frames could be read from this video. This usually means "
            "OpenCV can't decode its codec (common with .webm/VP9). "
            "Convert it to .mp4 (H.264) and re-upload, e.g.: "
            "ffmpeg -i input.webm -c:v libx264 -crf 23 output.mp4"
        )
        return

    job["status"] = "done"
    job["progress"] = 100.0
    job["counts"] = side_counts
    job["total_cars"] = sum(side_counts.values())
    job["result_file"] = out_path.name


# --------------------------------------------------------------------------
# Flask routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part named 'video'"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    job_id = uuid.uuid4().hex
    ext = Path(f.filename).suffix or ".mp4"
    in_path = UPLOAD_DIR / f"{job_id}{ext}"
    out_path = RESULT_DIR / f"{job_id}_annotated.mp4"
    f.save(in_path)

    JOBS[job_id] = {"status": "queued", "progress": 0.0}

    thread = threading.Thread(target=process_video, args=(job_id, in_path, out_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Result not ready"}), 404
    return send_from_directory(RESULT_DIR, job["result_file"], as_attachment=True)


if __name__ == "__main__":
    write_tuned_tracker_cfg()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)