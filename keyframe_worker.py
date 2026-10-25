"""
KeyframeWorker: turns each settled keyframe into memory, off the main thread.

Gemini is slow (seconds), so it runs here. The queue holds ONE job: if a newer keyframe
arrives while Gemini is busy, the older waiting one is dropped (stale anyway).

What one keyframe's detections are made of (capture pipeline from Adham's detection branch):
  1. Gemini SCAN of the whole frame, with people blurred out (Gemini never sees faces).
     It lists every item it can see that is NOT being held or worn.
  2. Items in someone's hand or being worn, from the local YOLO model (the frame never
     leaves the laptop for these). Memory marks them "with you".
  3. Items YOLO saw that the scan didn't list: Gemini CHECKS each one in an orange box.
     YES = add it, NONE = "that isn't a pen" (ignored at that spot for 2 min),
     HELD = in a hand (look again in 8 s). Verdicts are remembered while the item stays put,
     so a still table costs no extra calls.
If Gemini fails or is off, we fall back to the live YOLO boxes (people removed).
"""

import os
import threading
import time

import cv2

import config
from gemini_detector import blur_people, gemini_check, gemini_detect
from memory import is_held
from ui import overlap


def _far_apart(a, b, width):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > config.CHECK_MOVE_FRACTION * width


class KeyframeWorker:
    def __init__(self, memory, store=None, use_gemini=True, detect_fn=gemini_detect, on_done=None,
                 on_pending=None, check_fn=gemini_check):
        self.memory = memory
        self.store = store
        self.use_gemini = use_gemini
        self.detect_fn = detect_fn
        self.check_fn = check_fn     # check_fn(frame, box, item) -> "ok" / "none" / "held" / None
        self.on_pending = on_pending  # called when the next keyframe has work to do (item missing, checks left)
        self.on_done = on_done       # on_done(keyframe, events), e.g. to refresh the trackbar
        self.cond = threading.Condition()
        self.job = None              # (frame, yolo_dets) or None
        self.running = True
        self.status = "idle"
        self.gemini_off_until = 0.0  # time.time() before which we do not even try Gemini
        self.checks_off_until = 0.0  # same, for the checks (they can use a different model and quota)
        self.verdicts = {}           # item -> (verdict, box, valid until) from Gemini's checks
        self.more_to_check = False   # YOLO items left over because of MAX_CHECKS_PER_KEYFRAME
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
        items = [d for d in yolo_dets if d[4] != "person"]
        people = [d[:4] for d in yolo_dets if d[4] == "person"]
        self.more_to_check = False
        if self.use_gemini and time.time() >= self.gemini_off_until:
            try:
                self.status = "gemini"
                private = blur_people(frame.copy(), people)  # Gemini never sees faces
                dets = list(self.detect_fn(private))
            except Exception as e:
                self._cool_down("Gemini", e, "gemini_off_until")
            else:
                dets += self._held_by_yolo(items, people, dets)
                dets += self._checked_extras(private, items, people, dets)
                return dets
        self.status = "yolo"
        return items

    def _held_by_yolo(self, items, people, scanned):
        """Items in someone's hand or being worn. Gemini was told to skip these (and can't see
        them through the blur), so they come from the local YOLO model."""
        have = {d[4] for d in scanned}
        best = {}
        for d in items:
            if d[4] not in have and is_held(d[:4], people) and (d[4] not in best or d[5] > best[d[4]][5]):
                best[d[4]] = d
        return list(best.values())

    def _checked_extras(self, private, items, people, scanned):
        """YOLO items on the table that Gemini's scan didn't list: ask Gemini to check each one."""
        have = {d[4] for d in scanned}
        width, now = private.shape[1], time.time()
        best = {}  # most confident YOLO box per item
        for d in items:
            if d[4] in have or is_held(d[:4], people):
                continue
            if any(overlap(d[:4], s[:4]) >= 0.5 for s in scanned):
                continue  # Gemini already looked at that object and named it something else
            if d[4] not in best or d[5] > best[d[4]][5]:
                best[d[4]] = d
        extras, checks = [], 0
        for item, d in sorted(best.items(), key=lambda kv: -kv[1][5]):
            box = d[:4]
            cached = self.verdicts.get(item)
            if cached and not _far_apart(cached[1], box, width) and now < cached[2]:
                if cached[0] == "ok":
                    extras.append(d)
                continue
            if now < self.checks_off_until:
                continue
            if checks >= config.MAX_CHECKS_PER_KEYFRAME:
                self.more_to_check = True
                continue
            checks += 1
            try:
                verdict = self.check_fn(private, box, item)
            except Exception as e:
                self._cool_down("Gemini check", e, "checks_off_until")
                continue
            print(f"[worker] check: YOLO's {item} -> {verdict or 'unclear'}")
            if verdict == "ok":
                self.verdicts[item] = ("ok", box, float("inf"))  # holds while the item stays put
                extras.append(d)
            elif verdict == "none":
                self.verdicts[item] = ("none", box, now + config.VETO_NONE_SECONDS)
            else:  # "held", or a reply we couldn't read: look again shortly
                self.verdicts[item] = (verdict or "unclear", box, now + config.VETO_HELD_SECONDS)
        return extras

    def _cool_down(self, what, e, attr):
        reason = "quota used up" if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) else str(e)[:100]
        setattr(self, attr, time.time() + config.GEMINI_COOLDOWN)
        fallback = "using YOLO boxes" if attr == "gemini_off_until" else "skipping checks"
        print(f"[worker] {what} failed ({reason}); {fallback} for the next {config.GEMINI_COOLDOWN / 60:.0f} min")

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
            people = [d[:4] for d in yolo_dets if d[4] == "person"]
            kf, events = self.memory.add_keyframe(dets, image=frame, people=people)
            for ev in events:
                print(f"[memory] {ev.label} {ev.kind}")
            self._save(kf, frame, events)
            self.status = "idle"
            if self.on_pending and (self.memory.pending_misses() or self.more_to_check):
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
