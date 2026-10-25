"""
Rewind - smooth live detection
==============================

Why the old version lagged: it grabbed a frame, ran YOLO on it, showed it,
then grabbed the next one. So the video could only ever be as fast as YOLO,
and the webcam's frame queue filled up, adding delay on top.

This version runs three things at once:
  Camera thread   -> keeps grabbing frames, only ever keeps the NEWEST one
  Detector thread -> runs YOLO on the newest frame in the background
  Main loop       -> shows video at full camera speed, draws the latest boxes

Video stays at ~30 fps even if YOLO only manages 5-10 fps. For Rewind that's
fine: things on a table don't move, so boxes a fraction of a second old are OK.

Run:
  python rewind_fast.py                     # default webcam, normal YOLO (COCO)
  python rewind_fast.py --source 1          # second camera (e.g. phone webcam)
  python rewind_fast.py --source demo.mp4   # a video file
  python rewind_fast.py --open-vocab        # YOLO-World: keys, glasses, wallet...
  python rewind_fast.py --no-window --seconds 10   # just print speed numbers
Press q (or close the window) to quit.
"""

import argparse
import os
import platform
import threading
import time

import cv2

# Things the normal COCO model already knows that are useful for Rewind.
# (COCO has no "keys", "glasses" or "wallet" - use --open-vocab for those.)
COCO_ITEMS = [
    "person", "cell phone", "remote", "cup", "bottle", "book", "scissors",
    "handbag", "backpack", "laptop", "mouse", "umbrella", "teddy bear",
]

# Free-text item names for YOLO-World (--open-vocab). Edit freely.
OPEN_VOCAB_ITEMS = [
    "keys", "glasses", "wallet", "phone", "remote control", "mug",
    "pill bottle", "person",
]

WINDOW = "Rewind"


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
def pick_device():
    """Use the GPU if there is one: NVIDIA (cuda), Apple Silicon (mps), else CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# Camera: background reader that only keeps the newest frame
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, source, width=1280, height=720):
        self.is_file = not str(source).isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            index = int(source)
            # DirectShow opens much faster than the Windows default backend.
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            # MJPG lets most USB webcams do 30 fps at 720p (raw YUYV often can't).
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue old frames

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera or video: {source!r}")

        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.ended = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:  # video finished
                    self.ended = True
                    break
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            if self.is_file:  # play files at real speed, not as fast as possible
                time.sleep(1.0 / self.file_fps)

    def latest(self):
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------
# Detector: runs YOLO on the newest frame, in the background
# --------------------------------------------------------------------------
class Detector:
    def __init__(self, camera, open_vocab, imgsz, conf, max_fps, device):
        import torch
        from ultralytics import YOLO, YOLOWorld

        self.camera = camera
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.min_gap = 1.0 / max_fps if max_fps > 0 else 0.0

        if device == "cpu":
            # Leave a core free so the video loop never gets starved.
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

        if open_vocab:
            self.model = YOLOWorld("yolov8s-worldv2.pt")
            self.model.set_classes(OPEN_VOCAB_ITEMS)
            self.classes = None
        else:
            self.model = YOLO("yolo11n.pt")  # n = nano, the fastest
            wanted = set(COCO_ITEMS)
            self.classes = [i for i, n in self.model.names.items() if n in wanted]

        self.lock = threading.Lock()
        self.detections = []  # list of (x1, y1, x2, y2, name, confidence)
        self.fps = 0.0
        self.running = True

        self._warm_up()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _predict(self, frame):
        return self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            classes=self.classes,
            verbose=False,  # printing every frame is surprisingly slow
        )[0]

    def _warm_up(self):
        """The first YOLO call is slow (loading kernels). Do it before we start."""
        import numpy as np

        self._predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def _loop(self):
        last_id = -1
        last_done = None
        while self.running:
            frame_id, frame = self.camera.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id

            start = time.perf_counter()
            result = self._predict(frame)
            names = result.names
            dets = [
                (int(x1), int(y1), int(x2), int(y2), names[int(k)], float(c))
                for (x1, y1, x2, y2), c, k in zip(
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            ]
            with self.lock:
                self.detections = dets

            took = time.perf_counter() - start
            if took < self.min_gap:  # cap the rate: things on a table don't move fast
                time.sleep(self.min_gap - took)

            # Real detections per second (including the cap), smoothed.
            done = time.perf_counter()
            if last_done is not None:
                rate = 1 / max(done - last_done, 1e-6)
                self.fps = 0.9 * self.fps + 0.1 * rate if self.fps else rate
            last_done = done

    def latest(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def colour_for(name):
    h = sum(ord(ch) * 31 for ch in name)
    return (h * 67 % 200 + 55, h * 131 % 200 + 55, h * 197 % 200 + 55)


def draw(frame, detections, video_fps, detect_fps):
    for x1, y1, x2, y2, name, conf in detections:
        col = colour_for(name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    status = f"video {video_fps:4.1f} fps | detect {detect_fps:4.1f} fps | {len(detections)} objects"
    cv2.rectangle(frame, (0, 0), (520, 32), (0, 0, 0), -1)
    cv2.putText(frame, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Rewind - smooth live detection")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file path")
    ap.add_argument("--open-vocab", action="store_true", help="use YOLO-World with OPEN_VOCAB_ITEMS")
    ap.add_argument("--imgsz", type=int, default=None, help="YOLO input size: 320/416/480/640 (smaller = faster)")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum confidence to show a box")
    ap.add_argument("--max-detect-fps", type=float, default=10, help="cap on YOLO runs per second (0 = no cap)")
    ap.add_argument("--no-window", action="store_true", help="don't show video, just print speeds")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = run until q)")
    args = ap.parse_args()

    device = pick_device()
    imgsz = args.imgsz or (640 if device != "cpu" else 416)
    print(f"[rewind] device={device}  imgsz={imgsz}  model={'YOLO-World' if args.open_vocab else 'yolo11n'}")

    cam = Camera(args.source)
    det = Detector(cam, args.open_vocab, imgsz, args.conf, args.max_detect_fps, device)

    shown, video_fps = -1, 0.0
    started = time.perf_counter()
    last_print = started
    window_start, window_frames = started, 0  # for measuring video fps
    try:
        while True:
            frame_id, frame = cam.latest()
            if frame is None or frame_id == shown:
                if cam.ended:
                    break
                if not args.no_window and cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                time.sleep(0.002)
            else:
                shown = frame_id
                now = time.perf_counter()
                window_frames += 1
                if now - window_start >= 1.0:  # frames shown in the last second
                    video_fps = window_frames / (now - window_start)
                    window_start, window_frames = now, 0

                dets = det.latest()
                if args.no_window:
                    if now - last_print > 1:
                        names = sorted({d[4] for d in dets})
                        print(f"video {video_fps:5.1f} fps | detect {det.fps:5.1f} fps | seen: {', '.join(names) or '-'}")
                        last_print = now
                else:
                    view = draw(frame.copy(), dets, video_fps, det.fps)
                    cv2.imshow(WINDOW, view)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    finally:
        det.stop()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()