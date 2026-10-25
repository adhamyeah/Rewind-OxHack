"""
KeyframeWorker: turns each settled keyframe into memory, off the main thread.

Gemini is slow (seconds), so it runs here. The queue holds ONE job: if a newer keyframe
arrives while Gemini is busy, the older waiting one is dropped (stale anyway).
If Gemini fails or is off, we fall back to the live YOLO boxes (people removed).
"""

import os
import threading

import cv2

import config
from gemini_detector import gemini_detect


class KeyframeWorker:
    def __init__(self, memory, store=None, use_gemini=True, detect_fn=gemini_detect, on_done=None):
        self.memory = memory
        self.store = store
        self.use_gemini = use_gemini
        self.detect_fn = detect_fn
        self.on_done = on_done       # on_done(keyframe, events), e.g. to refresh the trackbar
        self.cond = threading.Condition()
        self.job = None              # (frame, yolo_dets) or None
        self.running = True
        self.status = "idle"
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame, dets):
        """Queue a keyframe, replacing any older one still waiting."""
        with self.cond:
            self.job = (frame, dets)
            self.cond.notify()

    def _detections_for(self, frame, yolo_dets):
        if self.use_gemini:
            try:
                self.status = "gemini"
                return self.detect_fn(frame)
            except Exception as e:
                print(f"[worker] Gemini failed ({e}); using YOLO boxes instead")
        self.status = "yolo"
        return [d for d in yolo_dets if d[4] != "person"]

    def _loop(self):
        while True:
            with self.cond:
                while self.job is None and self.running:
                    self.cond.wait(timeout=0.5)
                if not self.running:
                    return
                frame, yolo_dets = self.job
                self.job = None
            dets = self._detections_for(frame, yolo_dets)
            kf, events = self.memory.add_keyframe(dets, image=frame)
            for ev in events:
                print(f"[memory] {ev.label} {ev.kind}")
            self._save(kf, frame, events)
            self.status = "idle"
            if self.on_done:
                self.on_done(kf, events)

    def _save(self, kf, frame, events):
        try:
            os.makedirs(config.KEYFRAME_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(config.KEYFRAME_DIR, f"kf_{kf.index:04d}.jpg"), frame)
        except Exception as e:
            print(f"[worker] could not save keyframe image: {e}")
        if self.store is not None:
            self.store.save(kf, events)  # store handles its own errors

    def stop(self):
        with self.cond:
            self.running = False
            self.cond.notify()
        self.thread.join(timeout=2)

