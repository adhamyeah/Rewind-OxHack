"""Decides when an item has settled somewhere or gone, and logs it to rewind.db.

The detector only calls update() once the scene has gone still after a change
(plus a periodic re-check), so hands moving through the shot never reach here.
On top of that:
  Placed: seen in the same zone on CONFIRM_PASSES passes in a row.
  Gone:   missing on MISSES_FOR_GONE passes in a row, so one missed detection
          doesn't count. "Picked up" if a person touched it or the spot
          where it sat now looks different (something physically took it).
"""

import glob
import os
import time

import cv2

import db
from zones import zone_for

CONFIRM_PASSES = 1          # scenes are already still when we look; raise if items flicker in
MISSES_FOR_GONE = 2         # the detector re-checks ~1 s after a miss to confirm
SPOT_CHANGED = 25.0         # mean grey difference where the item sat that means it really left
SNAP_DIR = "snapshots"
KEEP_SNAPSHOTS_HOURS = 24


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def grey_crop(frame, box):
    x1, y1, x2, y2 = box
    return cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)


def delete_old_snapshots(hours=KEEP_SNAPSHOTS_HOURS):
    """Privacy: item photos never outlive a day."""
    for f in glob.glob(f"{SNAP_DIR}/*.jpg"):
        if time.time() - os.path.getmtime(f) > hours * 3600:
            os.remove(f)


class Memory:
    def __init__(self, con, zones):
        self.db, self.zones, self.items = con, zones, {}
        os.makedirs(SNAP_DIR, exist_ok=True)

    def update(self, frame, detections, now=None):
        """detections: [(x1, y1, x2, y2, name, conf)]. Returns True if it wants a re-check soon."""
        now = now or time.time()
        width = frame.shape[1]
        people = [d[:4] for d in detections if d[4] == "person"]

        best = {}  # strongest box per item this pass
        for x1, y1, x2, y2, name, conf in detections:
            if name != "person" and conf > best.get(name, (0, None))[0]:
                best[name] = (conf, (x1, y1, x2, y2))

        unsure = False
        for name, (conf, box) in best.items():
            s = self.items.setdefault(name, dict(cand=None, count=0, zone=None, present=False,
                                                 misses=0, box=None, crop=None, touched=False))
            zone = zone_for(box, self.zones, width)
            s["count"] = s["count"] + 1 if zone == s["cand"] else 1
            s["cand"], s["misses"] = zone, 0
            settled_here = s["count"] >= CONFIRM_PASSES
            if s["present"] and zone == s["zone"]:
                s["box"], s["crop"] = box, grey_crop(frame, box)   # same spot: keep tracking
            elif settled_here:
                s.update(zone=zone, present=True, box=box, crop=grey_crop(frame, box), touched=False)
                self._log(name, "placed", zone, box, now, self._snapshot(frame, name, box, people, now))
            else:
                unsure = True

        for name, s in self.items.items():
            if name in best:
                continue
            if not s["present"]:
                s["cand"], s["count"] = None, 0
                continue
            if any(overlaps(p, s["box"]) for p in people):
                s["touched"] = True
            s["misses"] += 1
            if s["misses"] < MISSES_FOR_GONE:
                unsure = True
                continue
            picked_up = s["touched"] or self._spot_changed(frame, s)
            s.update(present=False, cand=None, count=0, misses=0)
            self._log(name, "gone", s["zone"], s["box"], now, None, picked_up)
        return unsure

    def _spot_changed(self, frame, s):
        now_crop = grey_crop(frame, s["box"])
        if s["crop"] is None or now_crop.shape != s["crop"].shape or now_crop.size == 0:
            return False
        return float(cv2.absdiff(now_crop, s["crop"]).mean()) > SPOT_CHANGED

    def _snapshot(self, frame, name, box, people, now):
        img = frame.copy()
        for x1, y1, x2, y2 in people:   # privacy: blur anyone in the shot
            x1, y1 = max(0, x1), max(0, y1)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 0)
        cv2.rectangle(img, box[:2], box[2:], (0, 200, 255), 4)
        path = f"{SNAP_DIR}/{name.replace(' ', '_')}_{int(now * 1000)}.jpg"
        cv2.imwrite(path, img)
        return path

    def _log(self, item, event, zone, box, at, image, picked_up=False):
        db.log_event(self.db, item, event, zone, box, at, image, picked_up)
        print(f"{time.strftime('%H:%M:%S', time.localtime(at))}  {item} {event} {zone}"
              + (" (picked up)" if picked_up else ""))
