"""
LiveDetector, two jobs on one background thread:

  Job 1 (live boxes): YOLO-World boxes for the LIVE view, only re-run when the scene
        changes (plus a check every RECHECK_SECONDS). Same change gate as phase1.
  Job 2 (keyframes):  notice "something moved, then the scene went still" and call
        on_keyframe(frame, dets) ONCE. Movement = fraction of pixels that changed between
        two passes; "still" = no movement for SETTLE_SECONDS.
"""

import os
import threading
import time

import cv2
import numpy as np

import config


def pick_device():
    """NVIDIA GPU if there is one; Apple GPU only if USE_MAC_GPU; else CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if config.USE_MAC_GPU and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gate_image(frame):
    """Small blurred greyscale copy: the diff runs on this, not the full frame."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (config.GATE_W, config.GATE_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (config.BLUR, config.BLUR), 0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def one_label_per_object(dets):
    """Two different labels on (almost) the same box ("wallet" AND "phone") are one object:
    keep the more confident label. agnostic_nms already does most of this; this catches the rest."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[5]):  # strongest first
        if not any(k[4] != d[4] and iou(k[:4], d[:4]) > config.SAME_OBJECT_IOU for k in keep):
            keep.append(d)
    return keep


def should_fire_keyframe(dirty, still_for, dirty_for):
    """A keyframe is due when the scene changed AND (it has gone still, OR it has been changing so
    long that waiting is pointless: a moving face never gives a still second)."""
    return dirty and (still_for >= config.SETTLE_SECONDS or dirty_for >= config.MAX_MOTION_WAIT)


def motion_fraction(prev_gate, gate):
    """Share of pixels (0..1) that changed by more than MOTION_PIXEL_DIFF between two gate images."""
    return float((cv2.absdiff(prev_gate, gate) > config.MOTION_PIXEL_DIFF).mean())


class LiveDetector:
    def __init__(self, camera, device="cpu", on_keyframe=None):
        self.camera = camera
        self.device = device
        self.on_keyframe = on_keyframe
        if device == "cpu":
            import torch

            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))  # keep a core free for video

        from ultralytics import YOLO, YOLOWorld

        self.item_model = YOLOWorld(config.MODEL)
        self.item_model.set_classes(config.YOLO_ITEMS)
        self.person_model = YOLO(config.PERSON_MODEL) if config.USE_PERSON_MODEL else None

        self.lock = threading.Lock()
        self.detections = []
        self.baseline = None       # gate image of the last inspected scene (live boxes)
        self.prev_gate = None      # gate image of the previous pass (motion)
        self.diff = 0.0
        self.motion = 0.0
        self.still_for = 0.0
        self.dirty = True          # True = scene changed since the last keyframe (first scene counts)
        self.dirty_since = time.perf_counter()  # when it became dirty (for MAX_MOTION_WAIT)
        self.last_motion = time.perf_counter()
        self.item_ms = 0.0
        self.keyframes_fired = 0
        self.passes = 0
        self.running = True
        self._warm_up()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _warm_up(self):
        """The first call is slow (loading kernels). Do it before we start."""
        blank = np.zeros((config.CAMERA_H, config.CAMERA_W, 3), dtype=np.uint8)
        try:
            self.detect(blank)
        except Exception as e:
            if self.device == "cpu":
                raise
            print(f"[rewind] GPU ({self.device}) failed: {e}\n[rewind] falling back to CPU")
            self.device = "cpu"
            self.detect(blank)

    def detect(self, frame):
        """Run YOLO-World (and the person model) on one frame -> list of detection tuples."""
        h, w = frame.shape[:2]
        dets = []
        start = time.perf_counter()
        r = self.item_model.predict(frame, conf=config.CONFIDENCE, imgsz=config.IMGSZ, device=self.device,
                                    agnostic_nms=True,  # one label per object: no "wallet" AND "phone" on one box
                                    verbose=False)[0]
        self.item_ms = (time.perf_counter() - start) * 1000
        items = [(int(x1), int(y1), int(x2), int(y2), r.names[int(k)], float(c))
                 for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        dets.extend(one_label_per_object(items))
        if self.person_model is not None:
            r = self.person_model.predict(frame, conf=config.PERSON_CONFIDENCE, imgsz=config.IMGSZ,
                                          classes=[0], device=self.device, verbose=False)[0]
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                if (x2 - x1) * (y2 - y1) >= config.MIN_PERSON_AREA * w * h:  # skip the background crowd
                    dets.append((int(x1), int(y1), int(x2), int(y2), "person", float(c)))
        return dets

    def _loop(self):
        last_id, last_run = -1, 0.0
        gap = 1.0 / config.PASSES_PER_SECOND
        while self.running:
            frame_id, frame = self.camera.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id
            now = time.perf_counter()
            gate = gate_image(frame)

            # --- Job 2: motion and settling ---
            if self.prev_gate is not None:
                self.motion = motion_fraction(self.prev_gate, gate)
                if self.motion > config.MOTION_FRACTION:
                    if not self.dirty:
                        self.dirty_since = now
                    self.dirty = True
                    self.last_motion = now
            self.prev_gate = gate
            self.still_for = now - self.last_motion

            # --- Job 1: live boxes behind the change gate ---
            if self.baseline is None:
                changed, self.diff = True, 0.0
            else:
                self.diff = float(cv2.absdiff(self.baseline, gate).mean())
                changed = self.diff > config.DIFF_THRESHOLD
            if changed or now - last_run > config.RECHECK_SECONDS:
                dets = self.detect(frame)
                last_run = now
                self.baseline = gate
                with self.lock:
                    self.detections = dets
            else:
                self.baseline = cv2.addWeighted(gate, config.DRIFT, self.baseline, 1 - config.DRIFT, 0)

            # --- fire one keyframe once the scene has settled ---
            if should_fire_keyframe(self.dirty, self.still_for, now - self.dirty_since):
                self.dirty = False
                dets = self.detect(frame)  # fresh boxes on the settled scene
                with self.lock:
                    self.detections = dets
                self.keyframes_fired += 1
                if self.on_keyframe:
                    self.on_keyframe(frame.copy(), dets)

            self.passes += 1
            took = time.perf_counter() - now
            if took < gap:
                time.sleep(gap - took)

    def force_keyframe(self):
        """Ask for one more keyframe of the (still) scene without waiting for motion."""
        if not self.dirty:
            self.dirty_since = time.perf_counter()
        self.dirty = True

    def latest(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=3)
