"""Rewind: watches the webcam, says what it sees, and remembers where things are put down.

Gemini does all the seeing. There's no local detection model: each "look" sends
one cleaned-up frame to Gemini, which returns every object and person with a box,
a name and where it is ("next to the blue mug").

Looks cost quota (free keys get a few hundred a day), so Rewind only looks when
the scene has changed and gone still again (a hand put something down and left),
plus a slow check every minute. Between looks the boxes stay where they were.

- Everything captured is cleaned first (cleaning.py): frames lose grain and get
  better light without blurring, and blurry/dark frames aren't sent; speech loses
  hum and room noise; Gemini's answers are checked, tidied and de-duplicated.
- Each object gets a type (electronics, kitchenware, furniture...), shown on its box.
- Says the name and type of each thing as it comes into view ("mug, kitchenware").
- Listens to the microphone: ask "where is my phone?" or "what can you see?".
- Items that settle somewhere are logged to rewind.db with a photo (people
  blurred) for the web page.

Run:  python main.py                          (Q quit, E toggle enhanced view, SPACE look now)
      python main.py --source 1               (another camera)
      python main.py --source clip.mp4        (a video file)
      python main.py --no-voice --no-mic
      python main.py --verbose                (also print what it sees and does)
"""

import argparse
import glob
import io
import json
import os
import queue
import sys
import threading
import time
import wave

import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

import db
from cleaning import TYPES, Enhancer, FrameCheck, clean_audio, clean_detections, noise_profile

load_dotenv()                       # GEMINI_API_KEY (and optional GEMINI_MODEL) from .env
cv2.setNumThreads(4)
VERBOSE = False                     # --verbose: print every look/placed/gone/heard line, not just speak names


def log(text):
    if VERBOSE:
        print(text)


# --- Gemini ---
# flash-lite: ~2 s per look and a much bigger free daily quota than gemini-3.5-flash (20 a day)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
LOOK_WIDTH = 1024                   # frames are sent this wide (JPEG): plenty for Gemini, small upload
LOOK_PROMPT = (
    "Find every distinct object and every person in this image. For each give: a short common "
    "lowercase name; its type (one of: {types}); its box as [ymin, xmin, ymax, xmax] scaled "
    "0-1000; where it is relative to "
    "nearby things in under 12 words, starting with 'on', 'in' or 'next to'; and movable: true if "
    "it's a small thing someone could put down and lose (phone, keys, mug, glasses, remote), false "
    "for people, furniture, walls, lights and fixtures. {reuse}")
LOOK_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "label": {"type": "string"},
    "type": {"type": "string", "enum": TYPES},
    "box_2d": {"type": "array", "items": {"type": "integer"}},
    "place": {"type": "string"},
    "movable": {"type": "boolean"}},
    "required": ["label", "type", "box_2d", "place", "movable"]}}
QUESTION_PROMPT = (
    "This is a recording of someone talking to a home camera that finds objects. "
    "Things it has seen: {known}. In view right now: {visible}. "
    'Reply as JSON: {{"heard": "<what they said>", "item": "<if they ask where something is: '
    "the matching name from the seen list, or the plain name of the thing if it isn't there; "
    'EVERYTHING if they ask what you can see; otherwise NONE>"}}')

# --- when to look ---
GATE_W, GATE_H = 320, 180           # change detection runs on this small greyscale image
PIXEL_DIFF = 25                     # grey levels a pixel must change by to count as changed
CHANGE_PCT = 0.5                    # % of pixels changed since the last look = "something happened"
STILL_PCT = 0.3                     # % changing frame-to-frame below which the scene is still
SETTLE_SECONDS = 0.8                # still this long = the hand has gone, look now
MIN_GAP_SECONDS = 4.0               # never look more often than this
RECHECK_SECONDS = 6.0               # look again this soon when an item has just gone missing
HEARTBEAT_SECONDS = 60.0            # look anyway this often
QUOTA_WAIT_SECONDS = 60.0           # after a "too many requests" reply, wait this long

# --- camera ---
WORK_WIDTH = 1280                   # full 720p: shown and cleaned at the camera's own resolution
CAMERA_BRIGHTNESS = (64, 192)       # range the auto-brightness may move the camera's own setting in

# --- memory rules ---
MISSES_FOR_GONE = 2                 # missing on 2 looks in a row = gone (one look can miss things)
SNAP_DIR = "snapshots"
KEEP_SNAPSHOTS_HOURS = 24

# --- voice ---
ANNOUNCE_COOLDOWN = 20.0            # don't repeat a name sooner than this
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
MIN_SPEECH_SECONDS = 0.4
MAX_SPEECH_SECONDS = 8.0
END_SILENCE_SECONDS = 0.8
VOICE_GAP_SECONDS = 3.0             # at most one voice question per this many seconds (saves quota)


# ---------- camera ----------

def open_camera(source):
    if not isinstance(source, int):                      # a video file
        cam = cv2.VideoCapture(source)
        if not cam.isOpened():
            raise SystemExit(f"Could not open video file {source}.")
        return cam
    # Windows: Media Foundation gave 30 fps at 720p where DirectShow only managed 10
    cam = cv2.VideoCapture(source, cv2.CAP_MSMF) if sys.platform == "win32" else None
    if cam is None or not cam.isOpened():
        cam = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(source)
    if not cam.isOpened():
        raise SystemExit(
            f"Could not open camera {source}.\n"
            "- Try --source 1, and close Zoom/Teams.\n"
            "- Windows: Settings > Privacy & security > Camera > let desktop apps use it,\n"
            "  and check the laptop's camera key / privacy shutter.")
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # 30 fps on most webcams
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)                  # always read the newest frame
    for _ in range(10):                                  # first frames are often dark
        cam.read()
    return cam


def to_work_size(frame):
    h, w = frame.shape[:2]
    if w <= WORK_WIDTH:
        return frame
    return cv2.resize(frame, (WORK_WIDTH, int(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)


class AutoBrightness:
    """Nudges the camera's own brightness so the picture isn't washed out or murky.

    Your camera ignores exposure settings, but its brightness setting works:
    lowering it cut blown-out highlights from 0.7% to 0.1% in testing.
    The original value is put back when Rewind quits.
    """

    def __init__(self, cam):
        self.cam, self.original = cam, cam.get(cv2.CAP_PROP_BRIGHTNESS)
        self.value, self.next = self.original, 0.0
        self.enabled = CAMERA_BRIGHTNESS[0] <= self.original <= CAMERA_BRIGHTNESS[1]

    def update(self, frame, now):
        if not self.enabled or now < self.next:
            return
        self.next = now + 0.5
        grey = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        step = 0
        if grey.mean() > 150 or (grey >= 250).mean() > 0.02:
            step = -8                                    # washed out
        elif grey.mean() < 90:
            step = 8                                     # murky
        if step:
            self.value = min(max(self.value + step, CAMERA_BRIGHTNESS[0]), CAMERA_BRIGHTNESS[1])
            self.cam.set(cv2.CAP_PROP_BRIGHTNESS, self.value)

    def restore(self):
        if self.enabled:
            self.cam.set(cv2.CAP_PROP_BRIGHTNESS, self.original)


class ChangeGate:
    """Decides when a look is worth a Gemini request: something changed, then went still."""

    def __init__(self):
        self.baseline = self.prev = None
        self.still_since = time.monotonic()
        self.pending = True                              # look once the first scene is still
        self.change = 0.0

    def update(self, frame, now):
        gate = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (GATE_W, GATE_H)), cv2.COLOR_BGR2GRAY), (9, 9), 0)
        if self.baseline is None:
            self.baseline = self.prev = gate
        if changed_pct(self.prev, gate) > STILL_PCT:
            self.still_since = now                       # something is moving: wait for it to stop
        self.prev = gate
        self.change = changed_pct(self.baseline, gate)
        if self.change > CHANGE_PCT:
            self.pending = True
        elif not self.pending:
            self.baseline = cv2.addWeighted(gate, 0.02, self.baseline, 0.98, 0)   # follow slow light changes

    def settled(self, now):
        return now - self.still_since >= SETTLE_SECONDS

    def looked(self):
        self.baseline, self.pending = self.prev, False


def changed_pct(a, b):
    return float((cv2.absdiff(a, b) > PIXEL_DIFF).mean() * 100)


# ---------- seeing (Gemini) ----------

def gemini_error(e):
    if isinstance(e, errors.ClientError):
        if e.code in (400, 401, 403) and "key" in str(e.message).lower():
            return "the API key in .env was rejected"
        if e.code == 429:
            daily = "PerDay" in str(e.details)
            return "daily limit reached" if daily else "too many requests this minute"
        return f"request error {e.code} {e.message}"
    if isinstance(e, errors.ServerError):
        return f"server error {e.code}"
    return f"couldn't reach Gemini ({type(e).__name__})"   # network errors and anything else


def look(client, frame, known, previous):
    """One Gemini request: every object and person in the frame, cleaned, in frame pixels."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (LOOK_WIDTH, int(h * LOOK_WIDTH / w)), interpolation=cv2.INTER_AREA)
    jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    reuse = f"Use these names again for the same kinds of thing: {', '.join(known)}." if known else ""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=jpg, mime_type="image/jpeg"),
                  LOOK_PROMPT.format(types=", ".join(TYPES), reuse=reuse)],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=LOOK_SCHEMA))
    try:
        raw = json.loads(response.text or "[]")
    except json.JSONDecodeError:
        raw = []                                         # cut-off or garbled reply: treat as nothing seen
    return clean_detections(raw, w, h, previous)


class Looker(threading.Thread):
    """Sends a frame to Gemini whenever the main loop asks for a look, one at a time."""

    def __init__(self, client, memory, announcer):
        super().__init__(daemon=True)
        self.client, self.memory, self.announcer = client, memory, announcer
        self.frame, self.wanted = None, threading.Event()
        self.objects, self.known = [], []                # latest look; every name seen so far
        self.busy, self.last, self.count = False, 0.0, 0
        self.blocked_until, self.problem, self.unsure = 0.0, "", False

    def ready(self, now):
        return not self.busy and now >= self.blocked_until and now - self.last >= MIN_GAP_SECONDS

    def request(self, frame):
        self.frame, self.busy = frame, True
        self.wanted.set()

    def run(self):
        while True:
            self.wanted.wait()
            self.wanted.clear()
            frame, t0 = self.frame, time.monotonic()
            try:
                objects = look(self.client, frame, self.known[-40:], self.objects)
                self.problem = ""
            except Exception as e:
                self.problem = gemini_error(e)
                log(f"Gemini: {self.problem}")
                if "daily" in self.problem or "minute" in self.problem:
                    self.blocked_until = time.monotonic() + QUOTA_WAIT_SECONDS
                self.last, self.busy = time.monotonic(), False
                continue
            self.objects, self.count = objects, self.count + 1
            for o in objects:
                if o["name"] not in self.known:
                    self.known.append(o["name"])
            log(f"look {self.count} ({time.monotonic() - t0:.1f} s): "
                + (", ".join(f"{o['name']} ({o['type']})" for o in objects) or "nothing"))
            now = time.time()
            self.announcer.update(objects)
            self.unsure = self.memory.update(frame, objects, now)
            self.last, self.busy = time.monotonic(), False


# ---------- voice out ----------

class Speaker(threading.Thread):
    """Speaks queued sentences one at a time with the computer's own voice (works offline)."""

    def __init__(self, enabled):
        super().__init__(daemon=True)
        self.enabled, self.queue, self.speaking_until = enabled, queue.Queue(), 0.0

    def say(self, text):
        log(f"  says: {text}")
        if self.enabled:
            self.queue.put(text)

    def busy(self):
        """True while talking (and just after), so the microphone ignores our own voice."""
        return not self.queue.empty() or time.monotonic() < self.speaking_until

    def run(self):
        if sys.platform == "win32":
            import comtypes                              # speech engine needs COM in this thread
            comtypes.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        while True:
            text = self.queue.get()
            self.speaking_until = time.monotonic() + 60
            engine.say(text)
            engine.runAndWait()
            self.speaking_until = time.monotonic() + 0.5


class Announcer:
    """Says a thing's name when it comes into view; not again until it has left and come back."""

    def __init__(self, speaker):
        self.speaker, self.in_view, self.said = speaker, set(), {}

    def update(self, objects):
        now = time.monotonic()
        kinds = {o["name"]: o["type"] for o in objects}
        for name in sorted(set(kinds) - self.in_view):   # new since the last look
            if now - self.said.get(name, -1e9) > ANNOUNCE_COOLDOWN:
                self.said[name] = now
                self.speaker.say(name if kinds[name] in ("person", "other") else f"{name}, {kinds[name]}")
        self.in_view = set(kinds)


# ---------- voice in ----------

def spoken_by_type(objects):
    """[mug, phone, laptop, person] -> "a phone and a laptop (electronics), a mug (kitchenware) and a person"."""
    groups = {}
    for o in objects:
        groups.setdefault(o["type"], [])
        if o["name"] not in groups[o["type"]]:
            groups[o["type"]].append(o["name"])
    parts = [spoken_list(names) + ("" if kind in ("person", "other") else f" ({kind})")
             for kind, names in groups.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def spoken_list(names):
    """["cup", "apple", "person"] -> "a cup, an apple and a person"."""
    words = [("an " if n[0] in "aeiou" else "a ") + n for n in names]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


class Listener(threading.Thread):
    """Waits for someone to speak, then asks Gemini what they want and answers out loud."""

    def __init__(self, gemini, speaker, looker, threshold):
        super().__init__(daemon=True)
        self.gemini, self.speaker, self.looker = gemini, speaker, looker
        self.min_threshold, self.noise, self.level = threshold, 30.0, 0
        self.last_ask = 0.0
        self.quiet = []                                  # recent blocks with nobody talking: the room's noise

    def run(self):
        block = int(SAMPLE_RATE * BLOCK_SECONDS)
        chunks, silent, talking = [], 0.0, False
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block)
        except Exception as e:
            print(f"Microphone: can't open it ({e}); voice questions are off")
            return
        with stream:
            while True:
                try:
                    audio, _ = stream.read(block)
                except sd.PortAudioError:                # microphone closed (program exiting)
                    return
                self.level = int(np.sqrt((audio.astype(np.float32) ** 2).mean()))
                if self.speaker.busy():                  # don't listen to our own voice
                    chunks, talking = [], False
                    continue
                threshold = max(self.min_threshold, self.noise * 4)
                if self.level > threshold:
                    talking, silent = True, 0.0
                    chunks.append(audio.copy())
                elif talking:
                    chunks.append(audio.copy())
                    silent += BLOCK_SECONDS
                else:
                    self.noise = 0.95 * self.noise + 0.05 * self.level   # learn the room's hum
                    self.quiet = (self.quiet + [audio.copy()])[-15:]     # last 1.5 s of quiet
                length = len(chunks) * BLOCK_SECONDS
                if talking and (silent >= END_SILENCE_SECONDS or length >= MAX_SPEECH_SECONDS):
                    if length - silent >= MIN_SPEECH_SECONDS and \
                            time.monotonic() - self.last_ask >= VOICE_GAP_SECONDS:
                        self.last_ask = time.monotonic()
                        recording = np.concatenate(chunks)
                        threading.Thread(target=self.handle, args=(recording,), daemon=True).start()
                    chunks, silent, talking = [], 0.0, False

    def handle(self, recording):
        quiet = np.concatenate(self.quiet)[:, 0] if self.quiet else None
        recording = clean_audio(recording[:, 0] if recording.ndim > 1 else recording,
                                SAMPLE_RATE, noise_profile(quiet))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(recording.tobytes())
        visible = sorted({o["name"] for o in self.looker.objects})
        seen_now = list(self.looker.objects)
        known = [n for n in self.looker.known if n != "person"]
        prompt = QUESTION_PROMPT.format(known=", ".join(known) or "nothing yet",
                                        visible=", ".join(visible) or "nothing")
        try:
            response = self.gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            reply = json.loads(response.text or "{}")
        except Exception as e:
            log(f"Gemini (voice): {gemini_error(e)}")
            return
        heard, item = reply.get("heard", ""), str(reply.get("item", "NONE")).strip().lower()
        log(f"  heard: {heard!r} -> {item}")
        if item == "everything":
            self.speaker.say(f"I can see {spoken_by_type(seen_now)}." if seen_now else "I can't see anything right now.")
        elif item and item != "none":
            self.speaker.say(self.where_is(item))

    def where_is(self, item):
        now_seen = next((o for o in self.looker.objects if o["name"] == item), None)
        if now_seen:
            kind = "" if now_seen["type"] in ("person", "other") else f" ({now_seen['type']})"
            return f"Your {item}{kind} is {now_seen['place']}. I can see it right now."
        con = db.open_db()
        row = next((r for r in db.latest_rows(con) if r["item"] == item), None)
        con.close()
        return db.answer(row) if row else f"I haven't seen a {item} yet."


# ---------- memory: placed / gone ----------

def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def mostly_inside(box, outer, share=0.8):
    """True if most of box lies within outer: a phone in a hand, earphones on a head."""
    w = min(box[2], outer[2]) - max(box[0], outer[0])
    h = min(box[3], outer[3]) - max(box[1], outer[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return w > 0 and h > 0 and area > 0 and w * h / area >= share


def rough_place(box, width):
    """Which third of the view: used to notice an item moving to a new spot."""
    x = (box[0] + box[2]) / 2 / width
    if x < 1 / 3:
        return "on the left of the camera view"
    if x < 2 / 3:
        return "in the middle of the camera view"
    return "on the right of the camera view"


def delete_old_snapshots():
    """Privacy: item photos never outlive a day."""
    for f in glob.glob(f"{SNAP_DIR}/*.jpg"):
        if time.time() - os.path.getmtime(f) > KEEP_SNAPSHOTS_HOURS * 3600:
            os.remove(f)


class Memory:
    """Logs movable things that settle somewhere ('placed') and that disappear ('gone')."""

    def __init__(self, con):
        self.db, self.items = con, {}
        os.makedirs(SNAP_DIR, exist_ok=True)

    def update(self, frame, objects, now):
        """One look. Returns True if an item just went missing and a re-check would help."""
        people = [o["box"] for o in objects if o["name"] == "person"]
        here = {}
        for o in objects:
            if o["movable"] and not any(mostly_inside(o["box"], p) for p in people):   # held items aren't put down
                here.setdefault(o["name"], o)

        for name, o in here.items():
            s = self.items.setdefault(name, dict(present=False, zone=None, box=None, misses=0, touched=False))
            zone = rough_place(o["box"], frame.shape[1])
            s["misses"] = 0
            if s["present"] and zone == s["zone"]:
                s["box"] = o["box"]
                continue
            s.update(present=True, zone=zone, box=o["box"], touched=False, type=o["type"])
            event_id = self._log(name, "placed", zone, o["box"], now,
                                 self._snapshot(frame, name, o["box"], people, now), kind=o["type"])
            if o["place"]:
                db.set_description(self.db, event_id, o["place"])

        unsure = False
        for name, s in self.items.items():
            if name in here or not s["present"]:
                continue
            if any(overlaps(p, s["box"]) for p in people):
                s["touched"] = True
            s["misses"] += 1
            if s["misses"] < MISSES_FOR_GONE:
                unsure = True
                continue
            s.update(present=False, misses=0)
            self._log(name, "gone", s["zone"], s["box"], now, None, s["touched"], s.get("type"))
        return unsure

    def _snapshot(self, frame, name, box, people, now):
        img = frame.copy()
        for x1, y1, x2, y2 in people:                    # privacy: blur anyone in the shot
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 0)
        cv2.rectangle(img, box[:2], box[2:], (0, 165, 255), 4)
        path = f"{SNAP_DIR}/{name.replace(' ', '_')}_{int(now * 1000)}.jpg"
        cv2.imwrite(path, img)
        return path

    def _log(self, item, event, zone, box, at, image, picked_up=False, kind=None):
        event_id = db.log_event(self.db, item, event, zone, box, at, image, picked_up, kind)
        log(f"{time.strftime('%H:%M:%S', time.localtime(at))}  {item} {event}"
            + (" (picked up)" if picked_up else ""))
        return event_id


# ---------- window ----------

def label(frame, text, x, y, colour):
    """Smooth text on a solid tag, readable on any background."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    y = max(h + base + 4, y)
    cv2.rectangle(frame, (x, y - h - base - 4), (x + w + 8, y), colour, -1)
    cv2.putText(frame, text, (x + 4, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)


TYPE_COLOURS = {                    # BGR, one colour per type so kinds stand apart at a glance
    "person": (80, 200, 80), "animal": (60, 140, 220), "electronics": (230, 160, 40),
    "kitchenware": (0, 165, 255), "food & drink": (60, 200, 250), "furniture": (150, 120, 90),
    "clothing & accessories": (200, 100, 220), "personal care": (180, 180, 250),
    "stationery & books": (240, 220, 120), "toys & games": (100, 100, 255), "tools": (160, 160, 160),
    "plants": (60, 170, 60), "decor": (200, 200, 120), "fixtures": (190, 190, 190),
    "vehicles": (90, 90, 200), "other": (220, 220, 220)}


def draw(frame, objects, status):
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        colour = TYPE_COLOURS.get(o["type"], (220, 220, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        label(frame, o["name"] if o["type"] == "person" else f"{o['name']} | {o['type']}", x1, y1, colour)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera number or video file")
    parser.add_argument("--no-voice", action="store_true", help="don't speak")
    parser.add_argument("--no-mic", action="store_true", help="don't listen to the microphone")
    parser.add_argument("--mic-threshold", type=int, default=100,
                        help="how loud speech must be (watch the mic number in the window)")
    parser.add_argument("--no-window", action="store_true", help="run without the video window")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    parser.add_argument("--verbose", action="store_true", help="print what happens, not just speak names")
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    source = int(args.source) if args.source.isdigit() else args.source

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Rewind needs Gemini to see: put GEMINI_API_KEY=... in .env")
    # retry a busy server (503) a couple of times; quota errors are handled by waiting instead
    retry = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1.0,
                                                                   http_status_codes=[500, 502, 503, 504]))
    gemini = genai.Client(api_key=key, http_options=retry)
    print(f"Gemini: {GEMINI_MODEL} does the seeing")

    cam = open_camera(source)
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("Camera opened but gave no frame. Try --source 1.")

    delete_old_snapshots()
    speaker = Speaker(enabled=not args.no_voice)
    speaker.start()
    looker = Looker(gemini, Memory(db.open_db()), Announcer(speaker))
    looker.start()
    listener = None
    if not args.no_mic:
        listener = Listener(gemini, speaker, looker, args.mic_threshold)
        listener.start()
        print("Microphone: on. Ask \"where is my phone?\" or \"what can you see?\"")

    is_file = not isinstance(source, int)
    brightness = AutoBrightness(cam) if not is_file else None
    print("Running. Rewind looks when something changes and says what it sees. Q in the window to quit.")
    enhancer, gate, check = Enhancer(), ChangeGate(), FrameCheck()
    usable, frame_problem = True, ""
    frame_gap = 1 / (cam.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
    show_enhanced, look_now = True, False
    fps, fps_frames, fps_t0 = 0.0, 0, time.monotonic()
    started = last_frame = time.monotonic()
    window = "Rewind (Q quit, E enhanced view, SPACE look now)"

    while True:
        ok, frame = cam.read()
        if not ok:
            log("end of video" if is_file else "camera stopped giving frames")
            break
        if is_file:                                      # play files at real speed
            time.sleep(max(0.0, frame_gap - (time.monotonic() - last_frame)))
        now = last_frame = time.monotonic()
        fps_frames += 1
        if now - fps_t0 >= 1.0:
            fps, fps_frames, fps_t0 = fps_frames / (now - fps_t0), 0, now

        raw = to_work_size(frame)
        if brightness:
            brightness.update(raw, now)
        clean = enhancer(raw)
        gate.update(clean, now)

        due = (gate.pending or look_now
               or (looker.unsure and now - looker.last >= RECHECK_SECONDS)
               or now - looker.last >= HEARTBEAT_SECONDS)
        if due and gate.settled(now) and looker.ready(now):
            usable, frame_problem = check(raw)           # don't spend a look on a bad frame
            if usable:
                looker.request(clean.copy())
                gate.looked()
                looker.unsure = look_now = False

        if not args.no_window:
            if looker.problem:
                state = f"Gemini: {looker.problem}"
            elif not usable and due:
                state = f"waiting: {frame_problem}"
            elif looker.busy:
                state = "looking..."
            elif gate.pending:
                state = "waiting for still"
            else:
                state = f"last look {now - looker.last:.0f}s ago" if looker.count else "starting"
            status = f"video {fps:.0f} fps | {looker.count} looks | {state} | {'enhanced' if show_enhanced else 'raw'}"
            if listener:
                status += f" | mic {listener.level}"
            view = (clean if show_enhanced else raw).copy()
            cv2.imshow(window, draw(view, looker.objects, status))
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed in (ord("e"), ord("E")):
                show_enhanced = not show_enhanced
            elif key_pressed == ord(" "):
                look_now = True
            if key_pressed in (ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        if args.seconds and now - started >= args.seconds:
            break

    log(f"video {fps:.0f} fps, {looker.count} looks, seeing: "
        + (", ".join(sorted({o['name'] for o in looker.objects})) or "nothing"))
    if brightness:
        brightness.restore()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
