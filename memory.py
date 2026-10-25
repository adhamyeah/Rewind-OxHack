"""
memory.py - remembers where each item is and when it was last seen (Stage 4).

Every pass from the detector comes in here. For each item:
  - in someone's hand (or being worn)        -> "you have it with you"; not put down yet
  - sitting STILL in one spot for ~1.5 s     -> PUT DOWN: snapshot + short clip + database row
  - still there on later passes              -> LAST SEEN time keeps updating
  - picked up and put back in the same place -> a fresh put-down (so times stay right)
  - unseen for 4 seconds                     -> GONE ("picked up" if it was in a hand)

Gemini helps in two ways (describe.py, Stage 7):
  - it CHECKS each put-down the detector reports ("that isn't a pen" / "it's in a
    hand" cancels it) and DESCRIBES the spot in words
  - when the scene SETTLES after movement, it SCANS the whole picture and reports
    every item it can see. It recognises objects better than the fast detector,
    so anything the detector missed or mislabelled still gets logged.

"Where" combines the ZONE (zones.py, Stage 3: fixed names you drew) with
Gemini's phrase ("on the desk, next to the laptop"), or the items nearby
("next to the mug") when Gemini has nothing to add.

MOMENTS (the rewind slider on the web page): every time the scene settles, and
every time something is put down or taken, the table is saved as a moment: a
still picture plus the few seconds leading up to it (the little rewind gif).
Answers point at the moment an item was put down (or taken), so the page can
replay it. People are blurred in every saved picture. Everything starts fresh
each time Rewind is started.
"""

import os
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2

from describe import Describer
from zones import load_zones, zone_for

CONFIRM_PASSES = 8       # still for 8 passes in a row (~1.5 s at 5 passes/s) = put down
STILL_FRACTION = 0.03    # "still" = the box centre moves less than 3% of the picture's width
MOVE_FRACTION = 0.08     # moved more than 8% of the picture's width (or to another zone) = a new spot
LEFT_AFTER = 3.0         # away from its spot this long = it was moved; putting it back logs a fresh put-down
GONE_AFTER = 4.0         # unseen this many seconds = it has left
HELD_RECENT = 3.0        # seen in a hand this recently = "you have it with you"
SAME_OBJECT_IOU = 0.7    # two labels on (almost) the same box = one object; keep the stronger label
VETO_NONE_SECONDS = 120  # Gemini said "that's not a pen": ignore that spot for 2 minutes
VETO_HELD_SECONDS = 8    # Gemini said "it's in a hand": look again after 8 seconds
LAST_SEEN_WRITE_EVERY = 5.0
CLIP_FRAMES = 20         # ~4 s of frames before each moment: the rewind gif
MAX_CLIP_FRAMES = 30     # a moment refreshed when the scene settles can grow to this (~6 s)
CLIP_SIZE = (320, 240)
MOMENT_GAP = 3.0         # something happening this soon after the last moment belongs to that moment
MAX_REPLAYS = 300        # keep the gifs of the latest 300 moments (older ones keep their still picture)
NEAR_FRACTION = 0.3      # items closer than this (fraction of the picture's width) are "next to" each other
SNAP_DIR = "snapshots"
NO_ZONE = "in front of the camera"
PREPOSITIONS = ("on", "in", "at", "by", "next", "near", "under", "beside", "behind", "inside",
                "between", "beneath", "underneath", "above", "against", "outside", "in front")


def open_db(path="rewind.db"):
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")  # lets the Gemini thread and the window read/write alongside
    db.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY,
        item TEXT, event TEXT,            -- 'placed', 'gone', or 'rejected' (Gemini vetoed it)
        zone TEXT, x1 INT, y1 INT, x2 INT, y2 INT,
        at REAL, image TEXT, picked_up INT DEFAULT 0,
        description TEXT)                 -- Gemini's phrase""")
    have = [row[1] for row in db.execute("PRAGMA table_info(events)")]
    for column, kind in (("description", "TEXT"), ("last_seen", "REAL"), ("clip", "TEXT"), ("source", "TEXT"),
                         ("keyframe", "INT")):
        if column not in have:  # database from an older version
            db.execute(f"ALTER TABLE events ADD COLUMN {column} {kind}")
    # Each run of Rewind is a session. Answers and the timeline only use the current one;
    # older sessions stay in the file but are ignored.
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, started REAL)")
    db.commit()
    return db


def latest_session_start(db_path="rewind.db"):
    """When the most recent run of Rewind started (0 if unknown), e.g. for opening the timeline on its own."""
    try:
        db = sqlite3.connect(db_path)
        row = db.execute("SELECT MAX(started) FROM sessions").fetchone()
        db.close()
        return row[0] or 0
    except sqlite3.Error:
        return 0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def place_phrase(zone):
    """Make a zone name read well after "Your keys are ...": 'table at the back' -> 'on the table at the back'."""
    if not zone:
        return NO_ZONE
    low = zone.lower()
    if low.startswith(PREPOSITIONS) or low == NO_ZONE:
        return zone
    return f"on {zone}" if low.startswith(("the ", "my ", "your ")) else f"on the {zone}"


def plural(item):
    return item.endswith("s")  # keys, glasses, headphones


def ago(t):
    secs = time.time() - t
    if secs < 60:
        return "just now"
    mins = int(secs // 60)
    if mins == 1:
        return "a minute ago"
    if mins < 60:
        return f"{mins} minutes ago"
    hours = mins // 60
    return "about an hour ago" if hours == 1 else f"about {hours} hours ago"


def held_by_person(box, people):
    """True if most of the item's box sits inside a person's box (held or worn)."""
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for px1, py1, px2, py2 in people:
        ix = max(0, min(x2, px2) - max(x1, px1))
        iy = max(0, min(y2, py2) - max(y1, py1))
        if ix * iy / area > 0.6:
            return True
    return False


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def far_apart(a, b, frame_w, fraction=MOVE_FRACTION):
    if a is None or b is None:
        return False
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > fraction * frame_w


def blur_people(img, people, scale=1.0):
    for x1, y1, x2, y2 in people:
        x1, y1, x2, y2 = (max(0, int(v * scale)) for v in (x1, y1, x2, y2))
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (31, 31), 0)
    return img


def jpeg(img, quality=80):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def next_to(box, others, frame_w):
    """'next to the mug' / 'next to the mug and the pen' / None. `others` = [(x1, y1, x2, y2, name), ...]."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    near = sorted(((((o[0] + o[2]) / 2 - cx) ** 2 + ((o[1] + o[3]) / 2 - cy) ** 2) ** 0.5, o[4]) for o in others)
    names = [name for d, name in near if d <= NEAR_FRACTION * frame_w][:2]
    return "next to " + " and ".join(f"the {n}" for n in names) if names else None


@dataclass
class Keyframe:
    """One saved moment of the table - a step on the web page's rewind slider."""
    index: int
    time: float
    still: bytes = b""                          # JPEG of the whole picture, people blurred
    clip: list = field(default_factory=list)    # small JPEGs of the seconds before it: the rewind gif
    items: list = field(default_factory=list)   # [(x1, y1, x2, y2, item, 1.0)] what memory knew was there
    people: list = field(default_factory=list)  # [(x1, y1, x2, y2)]
    last_pass: int = 0                          # newest detector pass in the clip


# --------------------------------------------------------------------------
class Memory:
    def __init__(self, db_path="rewind.db", use_gemini=True, on_confirmed=None, items=(), store=None):
        self.db_path = db_path
        self.db = open_db(db_path)
        self.started = time.time()  # this run's history starts now; earlier runs are ignored
        self.db.execute("INSERT INTO sessions (started) VALUES (?)", (self.started,))
        self.db.commit()
        self.zones = load_zones()
        self.item_names = list(items)
        self.items = {}
        self.lock = threading.Lock()
        self.notes = OrderedDict()   # item -> latest line, for the on-screen banner
        self.inbox = []              # Gemini results, applied on the detector thread
        self.history = deque(maxlen=MAX_CLIP_FRAMES)  # recent small frames for the rewind gifs
        self.passes = 0
        self.frame_w, self.frame_h = 640, 480
        self.scan_frames = {}
        self.scan_id = 0
        self.scan_keyframe = {}      # scan id -> the moment it belongs to
        self.first_scan = True       # the startup scan logs what's already there, silently
        self.on_confirmed = on_confirmed  # called as on_confirmed(item, phrase) once Gemini agrees
        self.store = store           # optional Tiger Data copy of moments and events (store.py)
        self.view_lock = threading.RLock()  # the web page reads moments/events from its own threads
        self.keyframes = []          # Keyframe, oldest first (this run only)
        self.events = []             # dicts for the page: put down / put back / picked up / out of view / seen
        os.makedirs(SNAP_DIR, exist_ok=True)
        print(f"[memory] zones: {', '.join(self.zones) or 'none yet - draw them on the web page (Zones)'}")
        self.describer = (Describer(db_path, on_result=self._gemini_result, on_scan=self._gemini_scan,
                                    items=self.item_names) if use_gemini else None)

    def _state(self, name):
        return self.items.setdefault(name, dict(
            cand=None, anchor=None, count=0, zone=None, box=None, present=False, row=None, prev=None,
            seen=0, place_seen=0, last_written=0, touch=0, held_seen=0, handled=False,
            veto_box=None, veto_until=0, veto_none=False, description=None,
            live=None,       # (zone, box, frame width, time) of the latest sighting NOT in someone's hand
            tracked=False,   # has the fast detector seen it at its spot? If not, only Gemini's scans can
            misses=0))       # Gemini scans in a row that didn't find it (for items only Gemini can see)

    # ----------------------------------------------------------------------
    def update(self, frame, detections):
        """Called on every detector pass."""
        now = time.time()
        h, w = frame.shape[:2]
        self.frame_w, self.frame_h = w, h
        self.passes += 1
        people = [d[:4] for d in detections if d[4] == "person"]
        self.history.append((self.passes, cv2.resize(frame, CLIP_SIZE), people, w))
        self._apply_inbox(now)

        best = {}  # strongest box per item this pass
        for x1, y1, x2, y2, name, conf in detections:
            if name != "person" and conf > best.get(name, (0, None))[0]:
                best[name] = (conf, (x1, y1, x2, y2))
        # One object, two labels (e.g. "wallet" AND "phone" on the same box): keep the stronger one.
        for a in list(best):
            for b in list(best):
                if a != b and a in best and b in best and iou(best[a][1], best[b][1]) > SAME_OBJECT_IOU:
                    del best[a if best[a][0] < best[b][0] else b]

        for name, (conf, box) in best.items():
            s = self._state(name)
            s["seen"] = now
            if held_by_person(box, people):
                # In someone's hand (or being worn): not put down; "you have it with you".
                s["touch"] = s["held_seen"] = now
                s["handled"] = s["handled"] or s["present"]
                s["cand"], s["count"] = None, 0
                continue

            zone = zone_for(box, self.zones, w, h) or NO_ZONE
            s["live"] = (zone, box, w, now)
            at_its_spot = s["present"] and zone == s["zone"] and not far_apart(s["box"], box, w)
            if at_its_spot:
                s["place_seen"] = now  # still where it was put down
                s["tracked"] = True
                self._write_last_seen(s, now)
            elif s["present"]:
                s["handled"] = True    # it's been moved from its spot

            # Must stay STILL: compared with where this run of sightings started.
            if zone == s["cand"] and not far_apart(s["anchor"], box, w, STILL_FRACTION):
                s["count"] += 1
            else:
                s["cand"], s["anchor"], s["count"] = zone, box, 1
            if s["count"] < CONFIRM_PASSES:
                continue
            vetoed = now < s["veto_until"] and not far_apart(s["veto_box"], box, w)
            if not at_its_spot and not vetoed:
                self._place(frame, name, zone, box, people, now)
            elif at_its_spot and s["handled"]:
                # Picked up and put back in the same place: fresh time, same description, no re-check.
                self._place(frame, name, zone, box, people, now, description=s["description"],
                            source="put back", check=False, announce=False)

        for name, s in self.items.items():
            if not s["present"] or name in best:
                continue
            if any(overlaps(p, s["box"]) for p in people):
                s["touch"] = now  # someone's hand/body is over it
            if not s["tracked"]:
                continue  # only Gemini can see this one: its scans decide when it has gone
            unseen = now - s["seen"]
            if unseen > LEFT_AFTER:
                s["handled"] = True
            if unseen > GONE_AFTER:
                s["present"], s["cand"], s["count"] = False, None, 0
                self._gone(name, s, now, picked_up=now - s["touch"] < GONE_AFTER + 2, frame=frame)

    def settled(self, frame, detections):
        """Called by the detector when the scene goes still after movement: save a moment, Gemini scans it."""
        people = [d[:4] for d in detections if d[4] == "person"]
        moment = self._moment(frame, time.time(), people, refresh=True)
        if not self._checking():
            return
        self.scan_id += 1
        self.scan_keyframe = {self.scan_id: moment}
        self.scan_frames = {self.scan_id: (frame.copy(), people)}  # only the newest scan matters
        image = blur_people(frame.copy(), people)                   # Gemini never sees faces
        self.describer.scan(self.scan_id, cv2.imencode(".jpg", image)[1].tobytes(), frame.shape[1], frame.shape[0])

    # ----------------------------------------------------------------------
    def _place(self, frame, item, zone, box, people, now, description=None, source="yolo",
               check=True, announce=True, moment=None, kind=None):
        s = self._state(item)
        image = self._snapshot(frame, item, box, people, now)
        s["prev"] = (s["present"], s["zone"], s["box"], s["row"], s["description"],
                     s["place_seen"])  # to undo if Gemini disagrees
        s.update(zone=zone, box=box, present=True, place_seen=now, last_written=now,
                 handled=False, cand=None, count=0, description=description,
                 tracked=source != "gemini", misses=0)
        if moment is None:
            moment = self._moment(frame, now, people)
        row = self.db.execute(
            "INSERT INTO events (item, event, zone, x1, y1, x2, y2, at, image, description, last_seen, source, keyframe) "
            "VALUES (?, 'placed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item, zone, *box, now, image, description, now, source, moment)).lastrowid
        self.db.commit()
        s["row"] = row
        kind = kind or ("put back" if source == "put back" else "put down")
        self._event(row, now, item, kind, moment, box, self._where(item, zone, description, box))
        where = description or place_phrase(zone)
        print(f"{datetime.now():%H:%M:%S}  {item} {'put back' if source == 'put back' else 'put down'} {where}"
              + ("  [found by Gemini]" if source == "gemini" else ""))
        if description:
            self._note(item, f"{item}: {description}")
            if announce and self.on_confirmed:
                self.on_confirmed(item, description)
        elif check and self._checking():
            self._note(item, f"{item}: {place_phrase(zone)} (checking...)")
            self.describer.request(row, image, item, None if zone == NO_ZONE else zone)
        else:
            self._note(item, f"{item}: {place_phrase(zone)}")

    def _gone(self, item, s, now, picked_up, frame=None):
        moment = self._moment(frame, now) if frame is not None else self._latest_moment()
        row = self.db.execute(
            "INSERT INTO events (item, event, zone, x1, y1, x2, y2, at, picked_up, description, last_seen, keyframe) "
            "VALUES (?, 'gone', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item, s["zone"], *s["box"], now, int(picked_up), s["description"], s["place_seen"], moment)).lastrowid
        self.db.execute("UPDATE events SET last_seen = ? WHERE id = ?", (s["place_seen"], s["row"]))
        self.db.commit()
        what = "picked up" if picked_up else "out of view"
        self._event(row, now, item, what, moment, s["box"], f"was {s['description'] or place_phrase(s['zone'])}")
        print(f"{datetime.now():%H:%M:%S}  {item} {what} (was {s['description'] or place_phrase(s['zone'])})")
        self._note(item, f"{item}: {what}")

    def _write_last_seen(self, s, now):
        if s["row"] and now - s["last_written"] > LAST_SEEN_WRITE_EVERY:
            self.db.execute("UPDATE events SET last_seen = ? WHERE id = ?", (now, s["row"]))
            self.db.commit()
            s["last_written"] = now

    def _checking(self):
        return self.describer is not None and self.describer.enabled

    # ----------------------------------------------------------------------
    # Gemini results arrive on Gemini's thread; they're queued and applied here.
    def _gemini_result(self, row_id, item, verdict, phrase):
        with self.lock:
            self.inbox.append(("check", row_id, item, verdict, phrase))

    def _gemini_scan(self, scan_id, found):
        with self.lock:
            self.inbox.append(("scan", scan_id, found))

    def _apply_inbox(self, now):
        with self.lock:
            inbox, self.inbox = self.inbox, []
        for job in inbox:
            if job[0] == "check":
                self._apply_check(now, *job[1:])
            else:
                self._apply_scan(now, *job[1:])

    def _apply_check(self, now, row_id, item, verdict, phrase):
        s = self.items.get(item)
        if not s or s["row"] != row_id:  # an older put-down; a newer one has replaced it
            return
        if verdict == "ok":
            s["description"] = phrase
            self._update_event(row_id, where=phrase)
            if s["present"]:
                self._note(item, f"{item}: {phrase}")
                if self.on_confirmed:
                    self.on_confirmed(item, phrase)
            return
        # Gemini vetoed it: undo the put-down and ignore that spot for a while.
        removed = self._update_event(row_id, remove=True)
        s["veto_box"] = s["box"]
        s["veto_until"] = now + (VETO_NONE_SECONDS if verdict == "none" else VETO_HELD_SECONDS)
        s["veto_none"] = verdict == "none"
        if s["prev"]:
            s["present"], s["zone"], s["box"], s["row"], s["description"], s["place_seen"] = s["prev"]
        else:
            s["present"] = False
        s["cand"], s["count"] = None, 0
        self._refresh_items(removed["keyframe"] if removed else None)
        if verdict == "held":
            s["held_seen"] = now
            self._note(item, f"{item}: in someone's hand")
        else:
            with self.lock:
                self.notes.pop(item, None)

    def _apply_scan(self, now, scan_id, found):
        frame, people = self.scan_frames.pop(scan_id, (None, None))
        moment = self.scan_keyframe.pop(scan_id, None)
        if frame is None:
            return
        first, self.first_scan = self.first_scan, False
        h, w = frame.shape[:2]
        # Items only Gemini can see: two scans in a row without them = gone.
        for name, s in self.items.items():
            if s["present"] and not s["tracked"] and name not in [f[0] for f in found]:
                s["misses"] += 1
                if s["misses"] >= 2:
                    s["present"], s["cand"], s["count"] = False, None, 0
                    self._gone(name, s, now, picked_up=now - s["touch"] < 30 or now - s["held_seen"] < 30,
                               frame=frame)
        for item, box, where in found:
            s = self._state(item)
            zone = zone_for(box, self.zones, w, h) or NO_ZONE
            s["seen"] = now
            s["misses"] = 0
            if s["present"] and zone == s["zone"] and not far_apart(s["box"], box, w):
                s["place_seen"] = now  # Gemini agrees it's still there
                if where and not s["description"]:
                    s["description"] = where
                    self.db.execute("UPDATE events SET description = ? WHERE id = ?", (where, s["row"]))
                    self.db.commit()
                    self._update_event(s["row"], where=where)
                    self._note(item, f"{item}: {where}")
                continue
            self._place(frame, item, zone, box, people, now, description=where or None,
                        source="gemini", check=False, announce=not first, moment=moment,
                        kind="seen" if first else None)
        self._refresh_items(moment)

    # ----------------------------------------------------------------------
    def _note(self, item, line):
        with self.lock:
            self.notes.pop(item, None)
            self.notes[item] = line

    def note_answer(self, text):
        self._note("?", f"> {text}")  # the latest spoken answer, shown under the item lines

    def recent_notes(self, n=3):
        with self.lock:
            return list(self.notes.values())[-n:]

    def answer(self, item):
        """A spoken answer to "Where is my <item>?" - live state first, then the database."""
        return self.explain(item)[0]

    def explain(self, item):
        """-> (answer text, jump). jump = {"keyframe", "box", "label", "kind"} - the moment to replay - or None."""
        text, row_id = self._answer(item)
        jump = None
        if row_id is not None:
            with self.view_lock:
                ev = next((e for e in reversed(self.events) if e["row"] == row_id), None)
            if ev and ev["keyframe"] is not None:
                jump = {"keyframe": ev["keyframe"], "box": list(ev["box"]), "label": item, "kind": ev["kind"]}
        return text, jump

    def _answer(self, item):
        """-> (text, id of the database row the answer is about, or None)."""
        now = time.time()
        are, were, they, them = ("are", "were", "they", "them") if plural(item) else ("is", "was", "it", "it")
        s = self.items.get(item)
        back_on_its_spot = s and s["present"] and s["place_seen"] >= s["held_seen"]
        if s and now - s["held_seen"] < HELD_RECENT and not back_on_its_spot:
            return f"It looks like you have your {item} with you right now.", None
        # Sitting somewhere new right now (moved, but not logged yet - or Gemini doubted it):
        # say what the camera sees, not the old spot.
        live = s and s["live"]
        if live and now - live[3] < 2 and s["cand"] == live[0] and s["count"] >= 3:
            zone, box, w, _ = live
            ruled_out = s["veto_none"] and now < s["veto_until"] and not far_apart(s["veto_box"], box, w)
            at_its_spot = s["present"] and zone == s["zone"] and not far_apart(s["box"], box, w)
            if not ruled_out and not at_its_spot:
                return f"Your {item} {are} {self._where(item, zone, None, box)}. I can see {them} there right now.", None

        db = sqlite3.connect(self.db_path)  # own connection: called from the web / voice threads
        # Only this run counts: nothing from before Rewind was started.
        last = db.execute("SELECT id, event, at, picked_up, last_seen FROM events WHERE item = ? "
                          "AND event IN ('placed', 'gone') AND at >= ? ORDER BY id DESC LIMIT 1",
                          (item, self.started)).fetchone()
        placed = db.execute("SELECT id, zone, description, at, last_seen, x1, y1, x2, y2 FROM events WHERE item = ? "
                            "AND event = 'placed' AND at >= ? ORDER BY id DESC LIMIT 1",
                            (item, self.started)).fetchone()
        db.close()
        if not last or not placed:
            return f"Sorry, I haven't seen your {item} since I started watching.", None
        row_id, zone, description, placed_at, last_seen, *box = placed
        where = self._where(item, zone, description, box)
        last_id, event, at, picked_up, _ = last
        if event == "placed":
            if s and s["present"] and s["row"] == row_id:
                last_seen = s["place_seen"]  # live, more up to date than the database
            last_seen = last_seen or placed_at
            if now - last_seen < 5:
                return f"Your {item} {are} {where}. I can see {them} there right now.", row_id
            return f"Your {item} {are} {where}. I last saw {them} there {ago(last_seen)}.", row_id
        if picked_up:
            return (f"{ago(at).capitalize()}, your {item} {were} picked up. Before that, {they} {were} {where}. "
                    f"You might have {them} with you."), last_id
        return f"I last saw your {item} {ago(last_seen or at)}, {where}. Then {they} went out of view.", last_id

    def _where(self, item, zone, description, box):
        """Gemini's phrase if there is one, else the zone plus what's next to it ("on the desk, next to the mug")."""
        if description:
            return description
        where = place_phrase(zone)
        others = [k for k in self.known_items() if k[4] != item]
        near = next_to(box, others, self.frame_w) if box else None
        return f"{where}, {near}" if near else where

    # ----------------------------------------------------------------------
    # Moments (the rewind slider) and the event list, for the web page
    # ----------------------------------------------------------------------
    def known_items(self):
        """[(x1, y1, x2, y2, item, 1.0)] for items memory believes are sitting at their spot right now."""
        return [(*s["box"], name, 1.0) for name, s in list(self.items.items()) if s["present"] and s["box"]]

    def gemini_only_items(self):
        """Known items the fast detector can't see (only Gemini found them): drawn from memory on the live view."""
        return [(*s["box"], name, 1.0) for name, s in list(self.items.items())
                if s["present"] and s["box"] and not s["tracked"]]

    def _latest_moment(self):
        with self.view_lock:
            return self.keyframes[-1].index if self.keyframes else None

    def _moment(self, frame, now, people=None, refresh=False):
        """The moment for something happening now: the latest one if it is recent, else a new one. -> index."""
        with self.view_lock:
            last = self.keyframes[-1] if self.keyframes else None
            if last and now - last.time < MOMENT_GAP:
                if refresh:  # the scene just settled: a calmer final picture, and the gif runs on to it
                    self._fill(last, frame, people)
                else:
                    last.items = self.known_items()
                return last.index
            kf = Keyframe(len(self.keyframes), now)
            self._fill(kf, frame, people)
            self.keyframes.append(kf)
            if len(self.keyframes) > MAX_REPLAYS:  # keep memory use flat on long runs
                self.keyframes[-MAX_REPLAYS - 1].clip = []
        if self.store is not None:
            self.store.save_keyframe(kf.index, kf.time, kf.items)
        return kf.index

    def _fill(self, kf, frame, people=None):
        """Picture, gif frames and item boxes for a moment, from the frame and the recent history."""
        if people is None:
            people = self.history[-1][2] if self.history else []
        kf.people = [tuple(p) for p in people]
        kf.still = jpeg(blur_people(frame.copy(), people), 85)
        fresh = [h for h in self.history if h[0] > kf.last_pass]
        if not kf.last_pass:
            fresh = fresh[-CLIP_FRAMES:]  # a new moment: the ~4 s before it
        for _, small, ppl, w in fresh:
            kf.clip.append(jpeg(blur_people(small.copy(), ppl, CLIP_SIZE[0] / w), 75))
        del kf.clip[:-MAX_CLIP_FRAMES]
        if fresh:
            kf.last_pass = fresh[-1][0]
        kf.items = self.known_items()

    def _refresh_items(self, index=None):
        """Memory changed its mind (Gemini): redraw the boxes of that moment (default: the latest)."""
        with self.view_lock:
            if index is None:
                index = self.keyframes[-1].index if self.keyframes else None
            if index is not None and index < len(self.keyframes):
                self.keyframes[index].items = self.known_items()

    def _event(self, row, now, item, kind, moment, box, where):
        ev = {"row": row, "t": now, "item": item, "kind": kind, "keyframe": moment,
              "box": tuple(int(v) for v in box), "where": where}
        with self.view_lock:
            self.events.append(ev)
        if self.store is not None:
            self.store.save_event(now, item, kind, moment, ev["box"])

    def _update_event(self, row, where=None, remove=False):
        with self.view_lock:
            for i, ev in enumerate(self.events):
                if ev["row"] == row:
                    if remove:
                        del self.events[i]
                    elif where:
                        ev["where"] = where
                    return ev

    def keyframe(self, index):
        with self.view_lock:
            return self.keyframes[index] if 0 <= index < len(self.keyframes) else None

    def page_state(self, n_events=12):
        """What the web page shows: the moments, what's on the table, the latest events."""
        with self.view_lock:
            moments = [{"i": k.index, "t": k.time, "since": k.time - self.started,
                        "labels": sorted({d[4] for d in k.items})} for k in self.keyframes]
            events = [dict(ev, box=list(ev["box"])) for ev in self.events[-n_events:]]
        present = sorted(name for name, s in list(self.items.items()) if s["present"])
        return {"started": self.started, "keyframes": moments, "present": present, "events": events}

    def reload_zones(self):
        self.zones = load_zones()

    # ----------------------------------------------------------------------
    def _snapshot(self, frame, item, box, people, now):
        img = blur_people(frame.copy(), people)                 # privacy: Gemini sees this blurred copy
        cv2.rectangle(img, box[:2], box[2:], (0, 140, 255), 3)  # orange box: Gemini is told to look here
        path = f"{SNAP_DIR}/{item.replace(' ', '_')}_{int(now * 1000)}.jpg"
        cv2.imwrite(path, img)
        return path
