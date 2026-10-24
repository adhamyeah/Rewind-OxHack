"""Rewind Phase 1: change-gated camera detection.

The capture loop runs at full webcam speed. YOLO runs only when the scene
changes, so one object move = one detection. This is the same shape as the
real pipeline (change detection -> settling -> keyframe), with the "save
keyframe" step stubbed out as an appeared/left-view log.

Run:  python phase1_camera_gate.py
Quit: Q
"""

import time
from datetime import datetime

import cv2
from ultralytics import YOLOWorld

ITEMS = ["keys", "glasses", "wallet", "phone", "remote control", "mug", "headphones", "pen"]
CONFIDENCE = 0.15

# --- YOLO knobs: the real speed levers (tune these first) ---
MODEL = "yolov8s-worldv2.pt"   # smallest YOLO-World model: no nano variant exists (only s/m/l/x)
# The weights are NOT auto-downloaded. If yolov8s-worldv2.pt is missing, fetch it once:
#   curl -L -o yolov8s-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt
IMGSZ = 480                    # default is 640; 480 is fine for a table and ~2x faster
DEVICE = "cpu"                 # "0" if you have an NVIDIA GPU

# --- change gate knobs ---
GATE_W, GATE_H = 480, 360   # diff runs on this tiny greyscale image, not the full frame
BLUR = 9                    # odd kernel; kills sensor noise
DIFF_THRESHOLD = 5.0        # mean absdiff that counts as a change; tune with the on-screen number
DRIFT = 0.02                # baseline blend per frame; stops slow lighting changes retriggering
MIN_SECONDS_BETWEEN_CHECKS = 1.0   # while the hand keeps moving, don't re-run every frame

model = YOLOWorld(MODEL)
model.set_classes(ITEMS)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise SystemExit("No webcam found. Try VideoCapture(1).")

visible: set[str] = set()
baseline = None                 # greyscale gate image of the last inspected scene
annotated = None
last_check = 0.0
last_detect_ms = 0.0
fps = 0.0
fps_frames, fps_t0 = 0, time.monotonic()

while True:
    ok, frame = cam.read()
    if not ok:
        break

    fps_frames += 1
    now = time.monotonic()
    if now - fps_t0 >= 1.0:
        fps = fps_frames / (now - fps_t0)
        fps_frames, fps_t0 = 0, now

    # tiny greyscale version of the frame, used only to detect change
    gate = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gate = cv2.resize(gate, (GATE_W, GATE_H), interpolation=cv2.INTER_AREA)
    gate = cv2.GaussianBlur(gate, (BLUR, BLUR), 0)

    diff = 0.0
    if baseline is None:
        baseline = gate            # first frame: nothing to compare against yet
    else:
        diff = float(cv2.absdiff(baseline, gate).mean())
        if diff > DIFF_THRESHOLD and now - last_check >= MIN_SECONDS_BETWEEN_CHECKS:
            last_check = now
            start = time.monotonic()
            result = model.predict(frame, conf=CONFIDENCE, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
            last_detect_ms = (time.monotonic() - start) * 1000

            seen = {ITEMS[int(c)] for c in result.boxes.cls}
            stamp = datetime.now().strftime("%H:%M:%S")
            for item in sorted(seen - visible):
                print(f"{stamp}  {item} appeared")
            for item in sorted(visible - seen):
                print(f"{stamp}  {item} left view")
            visible = seen
            annotated = result.plot()
            baseline = gate         # snap: this frame is the new "still" scene
        else:
            # gentle drift blend so slow lighting changes don't retrigger
            baseline = cv2.addWeighted(gate, DRIFT, baseline, 1 - DRIFT, 0)

    overlay = annotated if annotated is not None else frame.copy()
    status = f"in {fps:.0f} fps | diff {diff:.1f}/{DIFF_THRESHOLD:.1f}"
    if last_detect_ms:
        status += f" | yolo {last_detect_ms:.0f} ms"
    cv2.putText(overlay, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Rewind test (Q to quit)", overlay)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cam.release()
cv2.destroyAllWindows()
