"""
phase1_camera_gate.py - Rewind's eyes: the camera and the fast item detector.

This is the detection from the `detection` branch, unchanged in how it sees:
  - YOLO-World MEDIUM model at size 640 (finds ~40% more items than small/480 on
    128 labelled photos), one label per object (agnostic NMS: no "wallet" AND
    "phone" on one box)
  - people from their own model (yolo11n, bar 0.5), background crowd ignored
  - the change gate: the models only re-run when the picture changes (plus a check
    every 2 s), so the video stays smooth
  - "settled": once the scene is still for 1.5 s after movement, on_settle fires
    (memory.py saves a rewind moment and Gemini scans the table)

rewind.py runs the whole app (web page, memory, voice). To just check the camera
and the boxes:
  python phase1_camera_gate.py               # default webcam
  python phase1_camera_gate.py --source 1    # second camera (e.g. phone webcam)
Press q to quit.
"""

import argparse
import os
import platform
import threading
import time
from datetime import datetime

import cv2

# --- what to look for (from phase1: works on real glasses) ---
ITEMS = ["keys", "glasses", "wallet", "phone", "mug", "headphones", "pen", "water bottle"]
CONFIDENCE = 0.15          # bar for items (0.2 lost a quarter of the items in testing)

# --- item model ---
MODEL = "yolov8m-worldv2.pt"   # m = medium. "yolov8s-worldv2.pt" is ~2x faster but finds fewer items
IMGSZ = 640                    # 480 is ~1.5x faster but misses small things

# --- people (separate model, made for people) ---
PERSON_MODEL = "yolo11n.pt"
PERSON_CONFIDENCE = 0.5
MIN_PERSON_AREA = 0.03     # ignore people smaller than 3% of the picture (background crowd)

# --- camera ---
CAMERA_W, CAMERA_H = 640, 480  # like phase1: items look bigger than in a wide 1280x720 view
USE_MAC_GPU = False            # True = try Apple "mps" (faster, but can give worse results)

# --- change gate knobs (from phase1) ---
GATE_W, GATE_H = 480, 360  # the diff runs on this small greyscale image, not the full frame
BLUR = 9                   # odd kernel; kills sensor noise
DIFF_THRESHOLD = 5.0       # mean difference that counts as a change; tune with the on-screen number
DRIFT = 0.1                # baseline blend per pass, so slow lighting changes don't retrigger
RECHECK_SECONDS = 2.0      # run the models at least this often, even if nothing changed
PASSES_PER_SECOND = 5      # how often the detector looks at a new frame
SETTLE_SECONDS = 1.5       # still this long after movement = "settled": Gemini scans the whole scene

WINDOW = "Rewind camera check (q to quit)"


# --------------------------------------------------------------------------
# Hardware and models
# --------------------------------------------------------------------------
def pick_device():
    """NVIDIA GPU if there is one; Apple GPU only if USE_MAC_GPU; else CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if USE_MAC_GPU and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_item_model():
    from ultralytics import YOLOWorld

    model = YOLOWorld(MODEL)
    model.set_classes(ITEMS)  # items only: people come from their own model
    return model


def load_person_model():
    from ultralytics import YOLO

    return YOLO(PERSON_MODEL)


# --------------------------------------------------------------------------
# Camera: background reader that only keeps the newest frame
# --------------------------------------------------------------------------
def available_cameras(max_index=4):
    """Camera indexes OpenCV can open right now (only used to write a helpful error message)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class Camera:
    def __init__(self, source, width=CAMERA_W, height=CAMERA_H):
        self.is_file = not str(source).isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            index = int(source)
            # DirectShow opens much faster than the Windows default backend.
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            # MJPG lets most USB webcams do 30 fps (raw YUYV often can't).
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue old frames

        if not self.cap.isOpened():
            self.cap.release()
            if self.is_file:
                raise RuntimeError(f"Could not open the video file {source!r}. Check the path.")
            found = available_cameras()
            raise RuntimeError(
                f"Could not open camera {source!r}. "
                + (f"Cameras that can be opened right now: {found}. Try --source {found[0]}."
                   if found else "No camera can be opened right now.")
                + " Phone cameras need the phone unlocked and nearby; also check camera permission for your terminal.")

        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.ended = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
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
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------
# Detector: change gate + item model + person model, in the background
# --------------------------------------------------------------------------
class Detector:
    def __init__(self, camera, device, on_pass=None, on_settle=None):
        import torch

        self.camera = camera
        self.device = device
        self.on_pass = on_pass      # e.g. memory.update(frame, detections), called every pass
        self.on_settle = on_settle  # e.g. memory.settled(frame, detections), once each time the scene settles
        if device == "cpu":
            # Leave a core free so the video loop never gets starved.
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

        self.item_model = load_item_model()
        self.person_model = load_person_model()
        self.lock = threading.Lock()
        self.detections = []  # list of (x1, y1, x2, y2, name, confidence)
        self.visible = set()
        self.baseline = None  # greyscale gate image of the last inspected scene
        self.diff = 0.0
        self.item_ms = 0.0
        self.person_ms = 0.0
        self.scene_still = False
        self.model_runs = 0
        self.passes = 0
        self.running = True

        self._warm_up()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _warm_up(self):
        """The first call is slow (loading kernels). Do it before we start."""
        import numpy as np

        blank = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        try:
            self._detect(blank)
        except Exception as e:  # e.g. an op the Mac GPU doesn't support
            if self.device == "cpu":
                raise
            print(f"[rewind] GPU ({self.device}) failed: {e}\n[rewind] falling back to CPU")
            self.device = "cpu"
            self._detect(blank)
        self.model_runs = 0

    @staticmethod
    def _gate_image(frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (GATE_W, GATE_H), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (BLUR, BLUR), 0)

    def _detect(self, frame):
        h, w = frame.shape[:2]
        dets = []

        start = time.perf_counter()
        r = self.item_model.predict(frame, conf=CONFIDENCE, imgsz=IMGSZ, device=self.device,
                                    agnostic_nms=True,  # one label per object: no "wallet" AND "phone" on one box
                                    verbose=False)[0]
        self.item_ms = (time.perf_counter() - start) * 1000
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append((int(x1), int(y1), int(x2), int(y2), r.names[int(k)], float(c)))

        start = time.perf_counter()
        r = self.person_model.predict(frame, conf=PERSON_CONFIDENCE, imgsz=IMGSZ, classes=[0],
                                      device=self.device, verbose=False)[0]
        self.person_ms = (time.perf_counter() - start) * 1000
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if (x2 - x1) * (y2 - y1) >= MIN_PERSON_AREA * w * h:  # skip the background crowd
                dets.append((int(x1), int(y1), int(x2), int(y2), "person", float(c)))

        self.model_runs += 1
        return dets

    def _log_changes(self, dets):
        """phase1's appeared / left-view log (items only)."""
        seen = {d[4] for d in dets if d[4] != "person"}
        stamp = datetime.now().strftime("%H:%M:%S")
        for item in sorted(seen - self.visible):
            print(f"{stamp}  {item} appeared")
        for item in sorted(self.visible - seen):
            print(f"{stamp}  {item} left view")
        self.visible = seen

    def _loop(self):
        last_id, last_run = -1, 0.0
        last_change, settled = time.perf_counter(), False  # the first settle scans the room at startup
        gap = 1.0 / PASSES_PER_SECOND
        while self.running:
            frame_id, frame = self.camera.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id
            start = time.perf_counter()

            gate = self._gate_image(frame)
            if self.baseline is None:
                changed, self.diff = True, 0.0  # first frame: always look
            else:
                self.diff = float(cv2.absdiff(self.baseline, gate).mean())
                changed = self.diff > DIFF_THRESHOLD

            if changed or start - last_run > RECHECK_SECONDS:
                dets = self._detect(frame)
                last_run = start
                self.baseline = gate  # snap: this frame is the new "still" scene
                self._log_changes(dets)
                with self.lock:
                    self.detections = dets
                self.scene_still = not changed
            else:
                # Nothing moved: the last boxes are still right, reuse them.
                self.baseline = cv2.addWeighted(gate, DRIFT, self.baseline, 1 - DRIFT, 0)
                dets = self.latest()
                self.scene_still = True

            self.passes += 1
            if self.on_pass:
                self.on_pass(frame, dets)

            # Settled = no change for SETTLE_SECONDS after something moved: fire once.
            if changed:
                last_change, settled = start, False
            elif not settled and start - last_change >= SETTLE_SECONDS:
                settled = True
                if self.on_settle:
                    self.on_settle(frame, dets)

            took = time.perf_counter() - start
            if took < gap:
                time.sleep(gap - took)

    def latest(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


# --------------------------------------------------------------------------
# "Is someone holding it?"
# --------------------------------------------------------------------------
def held_by_person(box, people):
    """True if most of the item's box sits inside a person's box (held or worn).

    Simple on purpose: from a side-on camera, an item on the table behind
    someone can look "held" too.
    """
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for px1, py1, px2, py2 in people:
        ix = max(0, min(x2, px2) - max(x1, px1))
        iy = max(0, min(y2, py2) - max(y1, py1))
        if ix * iy / area > 0.6:
            return True
    return False


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def colour_for(name):
    h = sum(ord(ch) * 31 for ch in name)
    return (h * 67 % 200 + 55, h * 131 % 200 + 55, h * 197 % 200 + 55)


def label_box(frame, x1, y1, text, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw(frame, detections, status):
    """Boxes on the live frame: people thin and grey, items in colour, "(on person)" when held or worn."""
    people = [d[:4] for d in detections if d[4] == "person"]
    for x1, y1, x2, y2 in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
        label_box(frame, x1, y1, "person", (160, 160, 160))
    for x1, y1, x2, y2, name, conf in detections:
        if name == "person":
            continue
        col = colour_for(name)
        held = held_by_person((x1, y1, x2, y2), people)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4 if held else 2)
        label_box(frame, x1, y1, f"{name} {conf:.2f}" + (" (on person)" if held else ""), col)
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (0, 0), (tw + 12, 24), (0, 0, 0), -1)
    cv2.putText(frame, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# --------------------------------------------------------------------------
# Quick camera + detector check (the real app is rewind.py)
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Rewind camera + detector check (the app itself is rewind.py)")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file path")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = run until q)")
    args = ap.parse_args()

    device = pick_device()
    print(f"[rewind] device={device}  imgsz={IMGSZ}  items={MODEL}  people={PERSON_MODEL}")
    print("[rewind] loading models (first run downloads them, give it a minute)")
    cam = Camera(args.source)
    det = Detector(cam, device)
    cam.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    shown, started = -1, time.perf_counter()
    try:
        while not cam.ended:
            frame_id, frame = cam.latest()
            if frame is not None and frame_id != shown:
                shown = frame_id
                state = "still" if det.scene_still else "moving"
                status = (f"items {det.item_ms:.0f} ms | people {det.person_ms:.0f} ms"
                          f" | diff {det.diff:.1f}/{DIFF_THRESHOLD:.1f} {state}")
                cv2.imshow(WINDOW, draw(frame.copy(), det.latest(), status))
            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    finally:
        det.stop()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
