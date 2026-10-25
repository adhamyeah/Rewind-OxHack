"""
KeyframeWorker: turns each settled keyframe into memory, off the main thread.

Gemini is slow (seconds), so it runs here. The queue holds ONE job: if a newer keyframe
arrives while Gemini is busy, the older waiting one is dropped (stale anyway).
If Gemini fails or is off, we fall back to the live YOLO boxes (people removed).
"""

import os
import threading
import time

import cv2

import config
from gemini_detector import gemini_detect


class KeyframeWorker:
    def __init__(self, memory, store=None, use_gemini=True, detect_fn=gemini_detect, on_done=None,
                 on_pending=None):
        self.memory = memory
        self.store = store
        self.use_gemini = use_gemini
        self.detect_fn = detect_fn
        self.on_pending = on_pending  # called when an item is missing but not yet confirmed gone
        self.on_done = on_done       # on_done(keyframe, events), e.g. to refresh the trackbar
        self.cond = threading.Condition()
        self.job = None              # (frame, yolo_dets) or None
        self.running = True
        self.status = "idle"
        self.gemini_off_until = 0.0  # time.time() before which we do not even try Gemini
        self.last_source = "-"       # which detector made the last keyframe, and how long it took
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @property
    def busy(self):
        """True while a keyframe is waiting or being processed (memory is about to change)."""
        return self.job is not None or self.status != "idle"

    def submit(self, frame, dets):
        """Queue a keyframe, replacing any older one still waiting."""
        with self.cond:
            self.job = (frame, dets)
            self.cond.notify()

    def _detections_for(self, frame, yolo_dets):
        if self.use_gemini and time.time() >= self.gemini_off_until:
            try:
                self.status = "gemini"
                return self.detect_fn(frame)
            except Exception as e:
                quota = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
                reason = "quota used up" if quota else str(e)[:100]
                # a used-up quota won't come back soon; a network blip or busy server usually
                # does, and while Gemini is off nothing new can be learned (YOLO only knows
                # names Gemini already predicted), so retry those quickly
                wait = config.GEMINI_COOLDOWN if quota else config.GEMINI_RETRY_SECONDS
                self.gemini_off_until = time.time() + wait
                print(f"[worker] Gemini failed ({reason}); using YOLO boxes for the next "
                      + (f"{wait / 60:.0f} min" if wait >= 60 else f"{wait:.0f} s"))
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
            started = time.perf_counter()
            dets = self._detections_for(frame, yolo_dets)
            self.last_source = f"{self.status} {time.perf_counter() - started:.1f}s"
            print(f"[worker] keyframe via {self.last_source}")
            kf, events = self.memory.add_keyframe(dets, image=frame)
            for ev in events:
                print(f"[memory] {ev.label} {ev.kind}")
            self._save(kf, frame, events)
            self.status = "idle"
            if self.on_pending and self.memory.pending_misses():
                self.on_pending()
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

