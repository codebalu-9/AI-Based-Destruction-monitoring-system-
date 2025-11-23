import argparse
import time
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def open_link_on_mac(url: str, app: str | None = None):
    """
    Open a URL using macOS 'open'. Optionally choose a specific app (e.g., 'Safari', 'Google Chrome', 'VLC').
    """
    try:
        if app:
            subprocess.run(["open", "-a", app, url], check=False)
        else:
            subprocess.run(["open", url], check=False)
    except Exception as e:
        # Fallback to webbrowser if needed
        import webbrowser
        webbrowser.open(url)
        print(f"Used webbrowser fallback. Reason: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Detect a smartphone with webcam and auto-play a video on macOS.")
    p.add_argument("--url", "-u", required=True, help="Video URL to open when a phone is detected.")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default: 0).")
    p.add_argument("--model", "-m", default="yolov8n.pt", help="YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt).")
    p.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold (0-1).")
    p.add_argument("--frames", type=int, default=6, help="Consecutive frames required to confirm detection.")
    p.add_argument("--cooldown", type=float, default=10.0, help="Cooldown seconds before re-triggering.")
    p.add_argument("--open-with", default=None, help="App to open the URL with (e.g., 'Safari', 'Google Chrome', 'VLC').")
    p.add_argument("--show", action="store_true", help="Show webcam window with detections.")
    return p.parse_args()


def main():
    args = parse_args()
    video_url = args.url

    # Load YOLO model (downloads weights on first run)
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Resolve the relevant class ids for 'cell phone' in the model labels
    name_map = model.names if hasattr(model, "names") else {}
    phone_ids = [i for i, n in name_map.items() if str(n).lower().strip().replace("_", " ") in
                 {"cell phone", "cellphone", "mobile phone", "smartphone", "telephone"}]
    if not phone_ids:
        # COCO usually uses 'cell phone' with id 67
        phone_ids = [67]

    # Initialize camera (macOS-friendly backend, fallback if needed)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Try a different index with --camera or check permissions.")
        sys.exit(1)

    # Optional: set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' in the preview window to quit (if --show was used).")
    print("Waiting for phone detection...")

    consecutive_hits = 0
    detected_stable = False
    last_trigger_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: Failed to read frame from camera.")
                break

            # Run inference
            results = model(frame, conf=args.conf, verbose=False)[0]
            boxes = results.boxes

            phone_present_this_frame = False
            if boxes is not None and len(boxes) > 0:
                # Parse detections
                for box in boxes:
                    # Class, confidence, and bbox
                    cls_id = int(float(box.cls[0])) if hasattr(box.cls, "__len__") else int(float(box.cls))
                    conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    x1, y1, x2, y2 = xyxy

                    if cls_id in phone_ids and conf >= args.conf:
                        phone_present_this_frame = True

                        if args.show:
                            label = f"{name_map.get(cls_id, 'phone')} {conf:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                    else:
                        if args.show:
                            # Optionally draw other boxes faintly (comment out if you want cleaner view)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

            # Debounce: require N consecutive frames with a phone
            if phone_present_this_frame:
                consecutive_hits += 1
            else:
                consecutive_hits = max(0, consecutive_hits - 1)

            now = time.time()
            became_stable = (not detected_stable) and (consecutive_hits >= args.frames)
            if became_stable and (now - last_trigger_time) >= args.cooldown:
                print("Phone detected. Opening your video...")
                open_link_on_mac(video_url, app=args.open_with)
                last_trigger_time = now
                detected_stable = True

            # Reset stable flag when phone is gone for a couple frames
            if detected_stable and consecutive_hits == 0:
                detected_stable = False

            if args.show:
                status_text = f"Phone frames: {consecutive_hits}/{args.frames} | Stable: {detected_stable}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Phone detector (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()