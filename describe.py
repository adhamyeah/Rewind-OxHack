"""
describe.py - everything Rewind asks Gemini (Stage 7), in the background.

Three jobs, all off the camera's thread so the video never waits:
  CHECK + DESCRIBE  When the fast detector reports a put-down, Gemini looks at the
                    snapshot (item in an orange box, people blurred). "NONE" (that
                    isn't a pen) or "HELD" (it's still in a hand) cancels it;
                    otherwise it says where it is: "on the desk, next to the laptop".
  SCAN              When the scene settles after movement, Gemini looks at the whole
                    picture and lists every item it can see, with a box and a phrase.
                    It recognises objects much better than the fast detector, which
                    can't think - so phones, bottles etc. it missed still get logged.
  UNDERSTAND        Spoken questions the simple word-matching can't place ("where's
                    the thing I read with?") are mapped to an item.
With no key, no internet, or a Gemini error, Rewind keeps working without it.

Setup (once):
  pip install -U google-genai        # new "AQ." keys need an up-to-date library
  put this line in .env (git-ignored, never commit it):   GEMINI_API_KEY=your-key-here
Check it works:   python describe.py
"""

import base64
import json
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict

MODEL = "gemini-3.5-flash-lite"   # quick checks + descriptions: fastest and cheapest
SCAN_MODEL = "gemini-3.8-flash"   # whole-scene object finding: Google's docs use this for detection
MIN_SECONDS_BETWEEN_CALLS = 4.0   # stays well under free-tier per-minute limits
MIN_SECONDS_BETWEEN_SCANS = 8.0
TIMEOUT_SECONDS = 20


def load_key():
    """GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment or a .env file."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), ".env"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    name, sep, value = line.strip().partition("=")
                    value = value.strip().strip('"').strip("'")
                    if sep and name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value:
                        os.environ.setdefault(name.strip(), value)
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


# --------------------------------------------------------------------------
# Prompts and reading the replies
# --------------------------------------------------------------------------
def build_prompt(item, zone):
    are = "are" if item.endswith("s") else "is"
    hint = f' The family calls this area "{zone}".' if zone else ""
    return (
        f"This photo is from a camera in someone's home. A detector thinks there is a {item} "
        f"inside the orange box.{hint} Reply in exactly one of these three ways:\n"
        f"NONE - if the orange box does not contain a {item}.\n"
        f"HELD - if the {item} is in someone's hand or being worn.\n"
        f'Otherwise ONE short phrase, under 15 words, that completes the sentence "Your {item} {are} ...", '
        "naming one or two nearby objects or pieces of furniture you can see, so an elderly person "
        "with memory problems could find it. Describe only what is visible. No quotes."
    )


def read_reply(text, item):
    """Gemini's reply -> ("ok", phrase), ("none", None), ("held", None), or (None, None) if unusable."""
    first = (text or "").strip().split()
    word = first[0].strip(".,:;-!\"'").upper() if first else ""
    if word == "NONE":
        return "none", None
    if word == "HELD":
        return "held", None
    phrase = clean(text, item)
    return ("ok", phrase) if phrase else (None, None)


def clean(text, item):
    """Tidy Gemini's reply into a phrase that fits after 'Your keys are ...'."""
    if not text:
        return None
    text = text.strip().splitlines()[0].strip().strip('"').strip("'").rstrip(".").strip()
    for are in ("are", "is"):
        prefix = f"your {item} {are} "
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    text = text.rstrip(".").strip()
    if len(text) > 1 and text[0].isupper() and text[1].islower():  # "On the..." -> "on the...", keep "TV"
        text = text[0].lower() + text[1:]
    return text[:120] or None


def build_scan_prompt(items):
    names = ", ".join(f'"{name}"' for name in items)
    return (
        "This is a still photo from a camera in someone's home. Find each of these objects that is "
        f"clearly visible: {names}. Skip anything that is being held, carried or worn by a person.\n"
        "Return ONLY a JSON list, no other text. Each entry: "
        '{"label": one of the names above exactly, '
        '"box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000, '
        '"where": a short phrase, under 12 words, saying where it is, like "on the desk, next to the laptop"}. '
        "Return [] if none of them are visible."
    )


def parse_detections(text, items, width, height):
    """Gemini's JSON -> [(item, (x1, y1, x2, y2) in pixels, where-phrase)]. Bad entries are skipped."""
    if not text:
        return []
    match = re.search(r"[\[{].*[\]}]", text, re.S)  # tolerate ```json fences and stray words
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("boxes") or data.get("objects") or data.get("items") or []
    found, wanted = [], {name.lower(): name for name in items}
    for entry in data if isinstance(data, list) else []:
        try:
            label = wanted.get(str(entry["label"]).strip().lower())
            ymin, xmin, ymax, xmax = (float(v) for v in entry["box_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        if not label or not (0 <= xmin < xmax <= 1000 and 0 <= ymin < ymax <= 1000):
            continue
        box = (int(xmin / 1000 * width), int(ymin / 1000 * height), int(xmax / 1000 * width), int(ymax / 1000 * height))
        where = clean(entry.get("where"), label) if isinstance(entry.get("where"), str) else None
        found.append((label, box, where))
    return found


def build_question_prompt(question, items):
    return (f'Someone asked a home assistant: "{question}". Which one of these things are they asking '
            f"about: {', '.join(items)}? Reply with just that name exactly as written, or NONE.")


# --------------------------------------------------------------------------
class Gemini:
    """Thin wrapper: (picture +) prompt in, text out."""

    def __init__(self, key, model=MODEL):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000))
        self.model = model
        # Google's newest API is "Interactions"; older libraries only have generate_content.
        self.use_interactions = hasattr(self.client, "interactions")

    def ask(self, image_bytes, prompt, model=None):
        model = model or self.model
        if self.use_interactions:
            try:
                parts = [{"type": "text", "text": prompt}]
                if image_bytes:
                    parts.append({"type": "image", "data": base64.b64encode(image_bytes).decode("utf-8"),
                                  "mime_type": "image/jpeg"})
                reply = self.client.interactions.create(model=model, input=parts)
                if reply.output_text:
                    return reply.output_text
            except Exception as e:  # fall back to generate_content and stick with it if it works
                print(f"[gemini] Interactions API failed ({type(e).__name__}); trying generate_content")
            self.use_interactions = False

        from google.genai import types

        contents = [types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt] if image_bytes else prompt
        return self.client.models.generate_content(model=model, contents=contents).text

    def ask_text(self, prompt, model=None):
        return self.ask(None, prompt, model)


class Describer:
    """Queues Gemini jobs and runs them one at a time in the background."""

    def __init__(self, db_path="rewind.db", on_result=None, on_scan=None, items=()):
        self.db_path = db_path
        self.on_result = on_result  # on_result(row_id, item, verdict, phrase); verdict "ok" / "none" / "held"
        self.on_scan = on_scan      # on_scan(scan_id, [(item, box, where), ...])
        self.items = list(items)
        self.gemini = None
        self.errors = 0

        key = load_key()
        if not key:
            print("[gemini] no GEMINI_API_KEY found (.env file or environment) - using zone names only")
            return
        try:
            self.gemini = Gemini(key)
        except ImportError:
            print("[gemini] google-genai isn't installed: pip install -U google-genai - using zone names only")
            return

        self.pending = OrderedDict()  # newest job per item (+ newest scan); older ones are dropped
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[gemini] on ({MODEL} for checks, {SCAN_MODEL} for scans)")

    @property
    def enabled(self):
        return self.gemini is not None

    def request(self, row_id, image_path, item, zone):
        self._queue(item, ("check", row_id, image_path, item, zone))

    def scan(self, scan_id, jpeg, width, height):
        self._queue("__scan__", ("scan", scan_id, jpeg, width, height))

    def _queue(self, key, job):
        if not self.enabled:
            return
        with self.lock:
            self.pending.pop(key, None)
            self.pending[key] = job
        self.wake.set()

    def match_item(self, question):
        """Which item is this spoken question about? Called straight away (the person is waiting)."""
        if not self.enabled:
            return None
        try:
            reply = (self.gemini.ask_text(build_question_prompt(question, self.items)) or "").strip().strip(".").lower()
        except Exception as e:
            print(f"[gemini] couldn't read the question ({type(e).__name__})")
            return None
        for name in self.items:
            if reply == name.lower() or reply.endswith(name.lower()):
                return name
        return None

    def check(self, image_path, item, zone):
        """-> (verdict, phrase): ("ok", "on the desk, by the lamp"), ("none", None), ("held", None) or (None, None)."""
        try:
            with open(image_path, "rb") as f:
                image = f.read()
            started = time.perf_counter()
            verdict, phrase = read_reply(self.gemini.ask(image, build_prompt(item, zone)), item)
            took = time.perf_counter() - started
            if verdict == "none":
                print(f"[gemini] says that's not a {item} - ignoring it  ({took:.1f}s)")
            elif verdict == "held":
                print(f"[gemini] says the {item} is in someone's hand - not put down yet  ({took:.1f}s)")
            elif verdict == "ok":
                print(f"[gemini] {item}: {phrase}  ({took:.1f}s)")
            return verdict, phrase
        except Exception as e:
            self._error(e)
            return None, None

    def scan_now(self, jpeg, width, height):
        try:
            started = time.perf_counter()
            found = parse_detections(self.gemini.ask(jpeg, build_scan_prompt(self.items), SCAN_MODEL),
                                     self.items, width, height)
            print(f"[gemini] scan: {', '.join(item for item, _, _ in found) or 'nothing'}  "
                  f"({time.perf_counter() - started:.1f}s)")
            return found
        except Exception as e:
            self._error(e)
            return None

    def _error(self, e):
        self.errors += 1
        if self.errors <= 3:  # don't flood the terminal
            print(f"[gemini] failed ({type(e).__name__}): {str(e)[:300]}")

    def _loop(self):
        db = sqlite3.connect(self.db_path)  # its own connection: this runs on its own thread
        last_call = last_scan = 0.0
        while True:
            self.wake.wait()
            with self.lock:
                job = None
                for key, queued in self.pending.items():
                    # checks first; a scan waits until scans are allowed again
                    if queued[0] == "check" or time.time() - last_scan >= MIN_SECONDS_BETWEEN_SCANS:
                        job = self.pending.pop(key)
                        break
                if job is None:
                    if not self.pending:
                        self.wake.clear()
                        continue
            if job is None:
                time.sleep(0.5)
                continue
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()

            if job[0] == "scan":
                _, scan_id, jpeg, width, height = job
                last_scan = time.time()
                found = self.scan_now(jpeg, width, height)
                if found is not None and self.on_scan:
                    self.on_scan(scan_id, found)
                continue

            _, row_id, image_path, item, zone = job
            verdict, phrase = self.check(image_path, item, zone)
            if verdict == "ok":
                db.execute("UPDATE events SET description = ? WHERE id = ?", (phrase, row_id))
            elif verdict in ("none", "held"):
                # Gemini vetoed it: keep the row for the record, but answers ignore 'rejected'.
                db.execute("UPDATE events SET event = 'rejected', description = ? WHERE id = ?",
                           (f"gemini: {verdict}", row_id))
            db.commit()
            if verdict and self.on_result:
                self.on_result(row_id, item, verdict, phrase)


if __name__ == "__main__":
    # Self-test: checks your key and sends Gemini one tiny made-up picture.
    import cv2
    import numpy as np

    key = load_key()
    if not key:
        raise SystemExit("No key found. Create a .env file next to this script with:  GEMINI_API_KEY=your-key")
    picture = np.full((240, 320, 3), 255, np.uint8)
    cv2.circle(picture, (160, 120), 60, (0, 0, 255), -1)  # a red circle
    jpeg = cv2.imencode(".jpg", picture)[1].tobytes()
    try:
        gemini = Gemini(key)
    except ImportError:
        raise SystemExit("Install the library first:  pip install -U google-genai")
    for model in (MODEL, SCAN_MODEL):
        started = time.perf_counter()
        try:
            answer = gemini.ask(jpeg, "What colour is the circle in this picture? Answer with one word.", model)
        except Exception as e:
            msg = str(e)
            print(f"{model} FAILED: {type(e).__name__}: {msg[:400]}")
            if "401" in msg or "403" in msg or "API key" in msg or "UNAUTHENTICATED" in msg:
                print("-> Key problem. Run  pip install -U google-genai  (new 'AQ.' keys need it), check the key "
                      "was copied fully, and that it's a Gemini API key from Google AI Studio.")
            elif "404" in msg or "not found" in msg.lower():
                print(f"-> Model '{model}' isn't available to you: change it at the top of describe.py.")
            elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print("-> Rate limit: wait a minute and try again.")
            continue
        api = "Interactions API" if gemini.use_interactions else "generate_content"
        print(f"OK: {model} says '{answer.strip()}' in {time.perf_counter() - started:.1f}s ({api})")
