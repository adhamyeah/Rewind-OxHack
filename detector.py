"""Rewind detector: watches the camera and logs where items are put down.

The capture loop runs at full webcam speed. YOLO only runs once the scene has
changed *and then gone still* (the hand has left), plus a slow periodic
re-check, so one object move = one detection. Each detection pass goes to
memory.py, which writes placed/gone rows to rewind.db for server.py to read.

Run:  python detector.py                   (live window, Q to quit)
      python detector.py --source 1        (second camera)
      python detector.py --no-window --seconds 20
      python detector.py --source test.mp4 --items "keys,glasses"   (video file, other item names)
"""

import argparse
import time

import cv2
import torch
from ultralytics import YOLOWorld

import db
from camera import open_camera, parse_source
from memory import Memory, delete_old_snapshots
from zones import draw_zones, load_zones

ITEMS = ["keys", "glasses", "wallet", "phone", "remote control", "mug", "headphones", "pen"]
# per-item confidence: small things like keys need a lower bar
CONFIDENCE = {"keys": 0.08, "glasses": 0.1, "pen": 0.1}
DEFAULT_CONFIDENCE = 0.15
MAX_BOX_FRACTION = 0.3      # drop an "item" covering more than this share of the frame

# --- YOLO knobs: the real speed levers (tune these first) ---
MODEL = "yolov8s-worldv2.pt"   # smallest YOLO-World model: no nano variant exists (only s/m/l/x)
# The weights are NOT auto-downloaded. If yolov8s-worldv2.pt is missing, fetch it once:
#   curl -L -o yolov8s-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt

# --- change gate knobs (tune with the on-screen numbers) ---
GATE_W, GATE_H = 320, 240   # change detection runs on this small greyscale image, not the full frame
BLUR = 9                    # odd kernel; kills sensor noise
PIXEL_DIFF = 25             # grey levels a pixel must change by to count as changed
CHANGE_PCT = 0.3            # % of pixels changed since the last look that means "something happened"
STILL_PCT = 0.2             # % of pixels changing frame-to-frame below which the scene is still
SETTLE_SECONDS = 0.6        # still this long = the hand has gone, look now
DRIFT = 0.02                # baseline blend per frame; stops slow lighting changes retriggering
RECHECK_SECONDS = 1.0       # look again this soon when memory is unsure (an item went missing)
HEARTBEAT_SECONDS = 10.0    # look anyway this often, to catch anything the gate missed


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_gate(frame):
    gate = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gate = cv2.resize(gate, (GATE_W, GATE_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gate, (BLUR, BLUR), 0)


def changed_pct(a, b):
    return float((cv2.absdiff(a, b) > PIXEL_DIFF).mean() * 100)


def detect(model, frame, names, imgsz, device):
    result = model.predict(frame, conf=min(DEFAULT_CONFIDENCE, *CONFIDENCE.values()),
                           imgsz=imgsz, device=device, verbose=False)[0]
    frame_area = frame.shape[0] * frame.shape[1]
    dets = []
    for (x1, y1, x2, y2), c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                         result.boxes.conf.tolist()):
        name = names[int(c)]
        if name != "person":
            if conf < CONFIDENCE.get(name, DEFAULT_CONFIDENCE):
                continue
            if (x2 - x1) * (y2 - y1) > MAX_BOX_FRACTION * frame_area:
                continue
        dets.append((int(x1), int(y1), int(x2), int(y2), name, float(conf)))
    return dets


def draw(frame, dets, zones, status):
    draw_zones(frame, zones)
    for x1, y1, x2, y2, name, conf in dets:
        colour = (160, 160, 160) if name == "person" else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(frame, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=parse_source, default=0, help="camera index or video file")
    parser.add_argument("--items", help="comma-separated item names to look for, instead of ITEMS")
    parser.add_argument("--imgsz", type=int, help="YOLO input size (default 480 on CPU, 640 on GPU)")
    parser.add_argument("--no-window", action="store_true", help="run without the video window")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    args = parser.parse_args()

    device = pick_device()
    imgsz = args.imgsz or (480 if device == "cpu" else 640)
    items = [i.strip() for i in args.items.split(",")] if args.items else ITEMS
    names = items + ["person"]          # person: for blurring snapshots and "picked up"
    model = YOLOWorld(MODEL)
    model.set_classes(names)

    cam = open_camera(args.source)
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("Camera opened but gave no frame. Try --source 1.")
    model.predict(frame, imgsz=imgsz, device=device, verbose=False)   # warm-up: first run is slow
    print(f"device={device} imgsz={imgsz} items={', '.join(items)}")
    is_file = not isinstance(args.source, int)
    frame_gap = 1 / (cam.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0

    delete_old_snapshots()
    zones = load_zones()
    if not zones:
        print("no zones.json yet: places will be left/middle/right. Run zone_editor.py to name them.")
    memory = Memory(db.open_db(), zones)

    dets = []
    baseline = prev = None
    pending = True                      # look once the first scene is still
    still_since = next_check = time.monotonic()
    detect_ms = change = 0.0
    fps, fps_frames, fps_t0 = 0.0, 0, time.monotonic()
    started = last_frame = time.monotonic()

    while True:
        ok, frame = cam.read()
        if not ok:
            print("end of video" if is_file else "camera stopped giving frames")
            break
        if is_file:                     # play files at their real speed, so settle timing holds
            time.sleep(max(0.0, frame_gap - (time.monotonic() - last_frame)))
        now = last_frame = time.monotonic()
        fps_frames += 1
        if now - fps_t0 >= 1.0:
            fps, fps_frames, fps_t0 = fps_frames / (now - fps_t0), 0, now

        gate = to_gate(frame)
        if baseline is None:
            baseline = prev = gate
        if changed_pct(prev, gate) > STILL_PCT:
            still_since = now           # something is moving: wait for it to stop
        prev = gate
        change = changed_pct(baseline, gate)
        if change > CHANGE_PCT:
            pending = True

        still = now - still_since >= SETTLE_SECONDS
        if still and (pending or now >= next_check):
            t0 = time.monotonic()
            dets = detect(model, frame, names, imgsz, device)
            detect_ms = (time.monotonic() - t0) * 1000
            unsure = memory.update(frame, dets)
            baseline, pending = gate, False
            next_check = time.monotonic() + (RECHECK_SECONDS if unsure else HEARTBEAT_SECONDS)
        elif not pending:
            # gentle drift blend so slow lighting changes don't retrigger
            baseline = cv2.addWeighted(gate, DRIFT, baseline, 1 - DRIFT, 0)

        if not args.no_window:
            state = "waiting for still" if pending else "watching"
            status = f"{fps:.0f} fps | change {change:.1f}/{CHANGE_PCT}% | {state}"
            if detect_ms:
                status += f" | yolo {detect_ms:.0f} ms"
            cv2.imshow("Rewind (Q to quit)", draw(frame, dets, zones, status))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty("Rewind (Q to quit)",
                                                                     cv2.WND_PROP_VISIBLE) < 1:
                break
        if args.seconds and now - started >= args.seconds:
            break

    print(f"video {fps:.0f} fps, last yolo {detect_ms:.0f} ms, seeing: "
          + (", ".join(d[4] for d in dets) or "nothing"))
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
