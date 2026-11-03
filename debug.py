"""
Quick diagnostic: run YOLO on a handful of frames from your video and print
every single detection (any class, any confidence) so we can see what the
model is actually seeing, before any car-class or confidence filtering.

Run:
    python debug_detect.py path\to\your_video.mp4
"""
import sys
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python debug_detect.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
weights_path = hf_hub_download(
    repo_id="rfonod/geo-trax", filename="geotrax_hbb_yolov8s_1920_v1.pt"
)
model = YOLO(weights_path)  # class ids: 0=car, 1=bus, 2=truck, 3=motorcycle

cap = cv2.VideoCapture(video_path)
print("Opened:", cap.isOpened())
print("Total frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Check a handful of frames spread through the video
frame_positions = [0, 50, 150, 300]

for pos in frame_positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, frame = cap.read()
    if not ok:
        print(f"Frame {pos}: could not read")
        continue

    h, w = frame.shape[:2]
    if w > 1920:
        scale = 1920 / w
        frame = cv2.resize(frame, (1920, int(h * scale)))

    # conf=0.05 and NO class filter — show literally everything the model sees
    results = model.predict(frame, conf=0.05, imgsz=1920, verbose=False)[0]

    print(f"\n--- Frame {pos} ---")
    if results.boxes is None or len(results.boxes) == 0:
        print("  No detections at all (any class, conf>=0.05)")
    else:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            print(f"  class={cls_name:12s} conf={conf:.2f}")

cap.release()