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
from vocab import VOCAB


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
        self.vocab = VOCAB if config.OPEN_VOCAB else None
        self.vocab_version = -1
        self.item_names = []       # what YOLO-World is looking for right now
        self._refresh_classes()
        self.person_model = YOLO(config.PERSON_MODEL) if config.USE_PERSON_MODEL else None

        self.lock = threading.Lock()
        self.detections = []
        self.baseline = None       # gate image of the last inspected scene (live boxes)
        self.prev_gate = None      # gate image of the previous pass (motion)
        self.diff = 0.0
        self.motion = 0.0
        self.still_for = 0.0
        self.dirty = True          # True = scene changed since the last keyframe (first scene counts)
        self.last_motion = time.perf_counter()
        self.last_keyframe = 0.0   # when the last keyframe fired
        self.look_now = False      # SPACE in the window: take a keyframe right away
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

    def _refresh_classes(self):
        """Tell YOLO-World what to look for: exactly the names Gemini has predicted so far.

        Until Gemini's first keyframe that's nothing, so the live view shows only people.
        (Without open vocabulary it's the fixed config.ITEMS, as originally.)
        """
        if self.vocab is None:
            if self.vocab_version == -1:
                self.item_model.set_classes(config.ITEMS)
                self.item_names, self.vocab_version = list(config.ITEMS), 0
            return
        if self.vocab.version != self.vocab_version:
            self.vocab_version = self.vocab.version
            names = self.vocab.names()
            if names:
                self.item_model.set_classes(names)
                print(f"[live] YOLO now tracks what Gemini predicted: {', '.join(names)}")
            self.item_names = names

    def detect(self, frame):
        """Run YOLO-World (and the person model) on one frame -> list of detection tuples."""
        self._refresh_classes()
        h, w = frame.shape[:2]
        dets = []
        start = time.perf_counter()
        if self.item_names:                  # nothing predicted yet: no item names to look for
            r = self.item_model.predict(frame, conf=config.CONFIDENCE, imgsz=config.IMGSZ,
                                        device=self.device, verbose=False)[0]
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append((int(x1), int(y1), int(x2), int(y2), r.names[int(k)], float(c)))
        self.item_ms = (time.perf_counter() - start) * 1000
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
            # ...or once it has kept changing for MAX_KEYFRAME_WAIT: someone in front of the
            # camera, or holding a thing up to it, never looks "still", and without this the
            # new thing was never shown to Gemini at all.
            settled = self.still_for >= config.SETTLE_SECONDS
            overdue = now - self.last_keyframe >= config.MAX_KEYFRAME_WAIT
            if self.look_now or (self.dirty and (settled or overdue)):
                self.dirty = self.look_now = False
                self.last_keyframe = now
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
        self.dirty = True

    def request_look(self):
        """Take a keyframe on the next pass, moving or not (the SPACE key)."""
        self.look_now = True

    def latest(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=3)
