# AI Based Smart Traffic Model

*4-Way Intersection Car Counter using YOLO + ByteTrack*

A Flask web app that takes a bird's-eye (drone / top-down) video of a 4-way intersection (*chauraha*), detects every car, tracks each one with a persistent ID, and counts how many unique cars approached from each side: **North, South, East and West**. It returns an annotated video with bounding boxes, IDs and a live count overlay.

## Team

| Name | Role |
|------|------|
| Ayush Kumar Bariyar | Developer |
| Aditi Prajapati | Developer |
| Ananya Kumari | Developer |
| S.K. Mohapatra | Supervisor |

## Features

- **Drone-view vehicle detection** using the `rfonod/geo-trax` YOLOv8s model (trained on VisDrone, UAVDT, CARPK, Songdo and others), downloaded automatically from Hugging Face on first run.
- **Multi-object tracking** with ByteTrack, tuned (`bytetrack_tuned.yaml`) to keep IDs through brief occlusion and motion blur so the same car is not counted twice.
- **Tiled inference ("SAHI-lite")** that runs the model on overlapping tiles as well as the full frame, recovering small, corner or densely packed cars that get lost when the frame is downscaled.
- **Per-side counting**: a car is assigned to a side based on where its track is *first seen*, using the frame's two diagonals to split it into North / South / East / West.
- **Web interface** with upload, live progress bar, per-side counts and a download link for the annotated video.
- **Automatic device selection**: CUDA (FP16) → Apple MPS → CPU.

## How it works

1. The video is uploaded through the web page and processed in a background thread.
2. Each frame is downscaled to at most 1920 px wide, then run through YOLO + ByteTrack.
3. Every 5th frame (configurable), the frame is also split into a 2×2 grid of overlapping tiles and run through the detector in a single batched call. Detections not already covered by the tracked pass are merged in via NMS.
4. Each new track ID is classified into a side by its first position and added to that side's count.
5. The annotated frame is written to the output video, and the browser polls `/status/<job_id>` until processing is done.

## Project structure

```
AI-based-smart-traffic-model/
├── app.py                  # Flask server, detection, tracking, tiling, counting
├── debug.py                # Diagnostic: prints raw detections for a few video frames
├── bytetrack_tuned.yaml    # Tuned ByteTrack configuration
├── templates/
│   └── index.html          # Upload page and results UI
├── uploads/                # Uploaded videos (created automatically)
└── results/                # Annotated output videos (created automatically)
```

## Requirements

- Python 3.9+
- A GPU is strongly recommended (CPU works but is slow at 1920 px with tiling)
- Python packages: `flask`, `ultralytics`, `opencv-python`, `numpy`, `huggingface_hub`, `torch`

## Installation

```bash
git clone https://github.com/ayushbariyar/AI-based-smart-traffic-model.git
cd AI-based-smart-traffic-model

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install flask ultralytics opencv-python numpy huggingface_hub torch
```

On first run the detector weights (`geotrax_hbb_yolov8s_1920_v1.pt`) are downloaded from Hugging Face, so an internet connection is needed once.

## Usage

```bash
python app.py
```

Open <http://localhost:5000>, choose a bird's-eye video of a 4-way intersection, and click **Upload & Process**. When processing finishes you will see the count for each side, the total, and a link to download the annotated video.

### API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Upload page |
| POST | `/upload` | Upload a video (form field `video`); returns `{ "job_id": ... }` |
| GET | `/status/<job_id>` | Job status, progress, and counts when done |
| GET | `/download/<job_id>` | Download the annotated video |

### Debugging detections

```bash
python debug.py path/to/your_video.mp4
```

## Configuration

Tune these constants at the top of `app.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONF_THRES` | `0.18` | Detection confidence threshold |
| `IOU_THRES` | `0.5` | NMS IoU, tighter so adjacent cars are not merged |
| `CAR_CLASS_IDS` | `{0}` | Classes counted (0=car, 1=bus, 2=truck, 3=motorcycle) |
| `MAX_WIDTH` | `1920` | Frames are downscaled to this width before inference |
| `ENABLE_TILING` | `True` | Master switch for tiled inference |
| `TILE_EVERY_FRAME` | `False` | Tile every frame or every N frames |
| `TILE_FRAME_STRIDE` | `5` | Tiling interval when `TILE_EVERY_FRAME` is `False` |
| `TILE_GRID` | `(2, 2)` | Tile layout (columns, rows) |
| `TILE_OVERLAP` | `0.20` | Overlap between neighbouring tiles |

If processing is too slow, lower `MAX_WIDTH` (e.g. 1280), increase `TILE_FRAME_STRIDE`, or set `ENABLE_TILING = False`.

## Limitations

- Side assignment uses simple diagonal geometry and assumes the intersection is centred in the frame.
- Cars found only by the tiled pass are not fed into the tracker and get synthetic IDs, so they can occasionally be counted more than once.
- The job store is in memory, so job results are lost when the server restarts.
- OpenCV may not decode some codecs (e.g. `.webm`/VP9). Convert first with `ffmpeg -i input.webm -c:v libx264 -crf 23 output.mp4`.

## Acknowledgements

- Supervised by **S.K. Mohapatra**
- [geo-trax](https://huggingface.co/rfonod/geo-trax) drone-view vehicle detector by rfonod
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and ByteTrack
- OpenCV and Flask
