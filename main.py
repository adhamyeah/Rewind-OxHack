"""Rewind: watches the webcam, says what it sees, and remembers where things are put down.

- Live window with boxes around every object and person YOLO knows (80 everyday
  things: person, cell phone, cup, bottle, laptop, book, remote...).
- Each frame is cleaned up before detection: averaged over time to remove
  sensor noise, smoothed, brightened if dark, and contrast-boosted.
- Says the name of each thing as it comes into view ("cup", "person").
- Listens to the microphone: ask "where is my phone?" or "what can you see?".
- When an item settles somewhere it's logged to rewind.db with a photo
  (people blurred), and Gemini describes the spot ("next to the blue mug").

Run:  python main.py                          (Q quit, E toggle enhanced view)
      python main.py --source 1               (another camera)
      python main.py --source clip.mp4        (a video file)
      python main.py --items "keys,glasses"   (things outside the library, via YOLO-World)
      python main.py --no-voice --no-mic
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
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from ultralytics import YOLO, YOLOWorld

import db

load_dotenv()                       # GEMINI_API_KEY (and optional GEMINI_MODEL) from .env

# --- detection ---
MODEL = "yolo11s.pt"                # standard YOLO "library": 80 everyday things; downloads once (19 MB)
OPEN_VOCAB_MODEL = "yolov8s-worldv2.pt"   # only with --items, for names outside the library
CONFIDENCE = 0.45                   # below this a box is ignored (one-off mistakes were ~35-50%)
MAX_DETECT_FPS = 8                  # leaves CPU for smooth video
# seen and announced, but never logged as "put down somewhere"
NOT_ITEMS = {"person", "dining table", "chair", "couch", "bed", "tv", "toilet", "sink",
             "refrigerator", "oven", "bench", "potted plant"}

# --- frame clean-up before detection ---
WORK_WIDTH = 960                    # frames are resized to at most this wide
TEMPORAL = 0.5                      # weight of the newest frame in the running average
TARGET_BRIGHTNESS = 120             # darker frames get their shadows lifted

# --- memory rules ---
CONFIRM_PASSES = 4                  # same place this many detections in a row = put down
GONE_AFTER = 4.0                    # unseen this many seconds = gone
SNAP_DIR = "snapshots"
KEEP_SNAPSHOTS_HOURS = 24

# --- voice ---
ANNOUNCE_COOLDOWN = 20.0            # don't repeat a name sooner than this
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
MIN_SPEECH_SECONDS = 0.4
MAX_SPEECH_SECONDS = 8.0
END_SILENCE_SECONDS = 0.8
VOICE_GAP_SECONDS = 3.0           # at most one voice question per this many seconds (free Gemini keys allow ~10 a minute)

# --- Gemini ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")   # override in .env
DESCRIBE_PROMPT = (
    "The {item} is inside the orange box. In under 15 words, say where it is relative to "
    "nearby objects, as you would to an elderly person. Start with a word like 'on', 'in' "
    "or 'next to'. Reply with the phrase only.")
QUESTION_PROMPT = (
    "This is a recording of someone talking to a home camera that finds objects. "
    "Objects it knows: {known}. Objects in view right now: {visible}. "
    'Reply as JSON: {{"heard": "<what they said>", "item": "<one exact name from the '
    'known list if they ask where something is, EVERYTHING if they ask what you can see, '
    'otherwise NONE>"}}')


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


class Enhancer:
    """Cleans each frame before detection: less noise, better light and contrast."""

    def __init__(self):
        self.avg = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, frame):
        f = frame.astype(np.float32)
        if self.avg is None or self.avg.shape != f.shape:
            self.avg = f
        else:
            cv2.accumulateWeighted(f, self.avg, TEMPORAL)   # averages out flickering sensor grain
        img = cv2.convertScaleAbs(self.avg)
        img = cv2.bilateralFilter(img, 5, 40, 40)            # smooths what's left, keeps edges sharp
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        mean = float(l.mean())
        if mean < TARGET_BRIGHTNESS - 20:                    # too dark: lift the shadows
            gamma = np.log(TARGET_BRIGHTNESS / 255) / np.log(max(mean, 1.0) / 255)
            l = cv2.LUT(l, (((np.arange(256) / 255) ** gamma) * 255).astype(np.uint8))
        l = self.clahe.apply(l)                              # local contrast: edges stand out
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_work_size(frame):
    h, w = frame.shape[:2]
    if w <= WORK_WIDTH:
        return frame
    return cv2.resize(frame, (WORK_WIDTH, int(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)


# ---------- detection ----------

def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(items):
    """Standard YOLO knows 80 things; YOLO-World takes any names but is less reliable."""
    if items:
        model = YOLOWorld(OPEN_VOCAB_MODEL)
        names = items + ["person"]
        model.set_classes(names)
        return model, names
    model = YOLO(MODEL)
    return model, [model.names[i] for i in sorted(model.names)]


def detect(model, frame, names, imgsz, device):
    result = model.predict(frame, conf=CONFIDENCE, imgsz=imgsz, device=device, verbose=False)[0]
    return [(int(x1), int(y1), int(x2), int(y2), names[int(c)], float(conf))
            for (x1, y1, x2, y2), c, conf in zip(result.boxes.xyxy.tolist(),
                                                 result.boxes.cls.tolist(),
                                                 result.boxes.conf.tolist())]


def rough_place(box, width):
    """Fallback place when Gemini isn't available: left / middle / right of the view."""
    x = (box[0] + box[2]) / 2 / width
    if x < 1 / 3:
        return "on the left of the camera view"
    if x < 2 / 3:
        return "in the middle of the camera view"
    return "on the right of the camera view"


# ---------- voice out ----------

class Speaker(threading.Thread):
    """Speaks queued sentences one at a time with the computer's own voice (works offline)."""

    def __init__(self, enabled):
        super().__init__(daemon=True)
        self.enabled, self.queue, self.speaking_until = enabled, queue.Queue(), 0.0

    def say(self, text):
        print(f"  says: {text}")
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
    """Says a thing's name when it comes into view, not again until it has left and come back."""

    def __init__(self, speaker):
        self.speaker, self.last_seen, self.streak, self.said = speaker, {}, {}, {}

    def update(self, names, now):
        for name in set(names):
            back = now - self.last_seen.get(name, -1e9) < 1.5
            self.streak[name] = self.streak.get(name, 0) + 1 if back else 1
            self.last_seen[name] = now
            # 3 detections in a row (~0.4 s) filters one-off mistakes
            if self.streak[name] == 3 and now - self.said.get(name, -1e9) > ANNOUNCE_COOLDOWN:
                self.said[name] = now
                self.speaker.say(name)


# ---------- Gemini ----------

def describe(client, image_path, item):
    """One plain-English phrase about where the item is, from its snapshot. None on failure."""
    with open(image_path, "rb") as f:
        image = types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=[image, DESCRIBE_PROMPT.format(item=item)])
    except Exception as e:
        print(f"Gemini: {gemini_error(e)}; using the rough place")
        return None
    text = (response.text or "").strip().rstrip(".")   # empty if Gemini blocked the image
    return text[:1].lower() + text[1:] or None         # it goes mid-sentence: "Your keys are on..."


def gemini_error(e):
    if isinstance(e, errors.ClientError):
        if e.code in (400, 401, 403) and "key" in str(e.message).lower():
            return "the API key in .env was rejected"
        if e.code == 429:
            return "rate limited"
        return f"request error {e.code} {e.message}"
    if isinstance(e, errors.ServerError):
        return f"server error {e.code}"
    return f"couldn't reach Gemini ({type(e).__name__})"   # network errors and anything else


def describe_in_background(client, event_id, image_path, item):
    def run():
        phrase = describe(client, image_path, item)
        if phrase:
            con = db.open_db()               # own connection: sqlite objects stay in their thread
            db.set_description(con, event_id, phrase)
            con.close()
            print(f"            Gemini: {item} {phrase}")
    threading.Thread(target=run, daemon=True).start()


# ---------- voice in ----------

def spoken_list(names):
    """["cup", "apple", "person"] -> "a cup, an apple and a person"."""
    words = [("an " if n[0] in "aeiou" else "a ") + n for n in names]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


class Listener(threading.Thread):
    """Waits for someone to speak, then asks Gemini what they want and answers out loud."""

    def __init__(self, gemini, speaker, known, detector, threshold):
        super().__init__(daemon=True)
        self.gemini, self.speaker, self.known, self.detector = gemini, speaker, known, detector
        self.min_threshold, self.noise, self.level = threshold, 30.0, 0
        self.last_ask = 0.0

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
                length = len(chunks) * BLOCK_SECONDS
                if talking and (silent >= END_SILENCE_SECONDS or length >= MAX_SPEECH_SECONDS):
                    if length - silent >= MIN_SPEECH_SECONDS and                             time.monotonic() - self.last_ask >= VOICE_GAP_SECONDS:
                        self.last_ask = time.monotonic()
                        recording = np.concatenate(chunks)
                        threading.Thread(target=self.handle, args=(recording,), daemon=True).start()
                    chunks, silent, talking = [], 0.0, False

    def handle(self, recording):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(recording.tobytes())
        visible = sorted({d[4] for d in self.detector.dets})
        prompt = QUESTION_PROMPT.format(known=", ".join(self.known), visible=", ".join(visible) or "nothing")
        try:
            response = self.gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"), prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            reply = json.loads(response.text or "{}")
        except Exception as e:
            print(f"Gemini (voice): {gemini_error(e)}")
            return
        heard, item = reply.get("heard", ""), str(reply.get("item", "NONE")).strip()
        print(f"  heard: {heard!r} -> {item}")
        if item.upper() == "EVERYTHING":
            self.speaker.say(f"I can see {spoken_list(visible)}." if visible else "I can't see anything right now.")
        elif item in self.known:
            self.speaker.say(self.where_is(item, visible))

    def where_is(self, item, visible):
        con = db.open_db()
        row = next((r for r in db.latest_rows(con) if r["item"] == item), None)
        con.close()
        if row and row["event"] == "placed":
            return db.answer(row)
        if item in visible:
            return f"I can see the {item} right now."
        if row:
            return db.answer(row)
        return f"I haven't seen a {item} yet."


# ---------- memory: placed / gone ----------

def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def mostly_inside(box, outer, share=0.8):
    """True if most of box lies within outer: a tie on a shirt, a "cat" that's really hair."""
    w = min(box[2], outer[2]) - max(box[0], outer[0])
    h = min(box[3], outer[3]) - max(box[1], outer[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return w > 0 and h > 0 and area > 0 and w * h / area >= share


def delete_old_snapshots():
    """Privacy: item photos never outlive a day."""
    for f in glob.glob(f"{SNAP_DIR}/*.jpg"):
        if time.time() - os.path.getmtime(f) > KEEP_SNAPSHOTS_HOURS * 3600:
            os.remove(f)


class Memory:
    def __init__(self, con, gemini):
        self.db, self.gemini, self.items = con, gemini, {}
        os.makedirs(SNAP_DIR, exist_ok=True)

    def update(self, frame, dets, now):
        people = [d[:4] for d in dets if d[4] == "person"]
        best = {}                                        # strongest box per item
        for x1, y1, x2, y2, name, conf in dets:
            box = (x1, y1, x2, y2)
            if name in NOT_ITEMS or any(mostly_inside(box, p) for p in people):
                continue                                 # worn or held, not put down
            if conf > best.get(name, (0, None))[0]:
                best[name] = (conf, box)

        for name, (conf, box) in best.items():
            s = self.items.setdefault(name, dict(cand=None, count=0, zone=None, present=False,
                                                 seen=0.0, box=None, touch=0.0))
            zone = rough_place(box, frame.shape[1])
            s["count"] = s["count"] + 1 if zone == s["cand"] else 1
            s["cand"], s["seen"], s["box"] = zone, now, box
            if s["count"] >= CONFIRM_PASSES and (not s["present"] or zone != s["zone"]):
                s["zone"], s["present"] = zone, True
                image = self._snapshot(frame, name, box, people, now)
                event_id = self._log(name, "placed", zone, box, now, image)
                if self.gemini:
                    describe_in_background(self.gemini, event_id, image, name)

        for name, s in self.items.items():
            if not s["present"]:
                continue
            if any(overlaps(p, s["box"]) for p in people):
                s["touch"] = now                         # someone's hand or body is on it
            if name not in best and now - s["seen"] > GONE_AFTER:
                s["present"], s["cand"], s["count"] = False, None, 0
                picked_up = now - s["touch"] < GONE_AFTER + 2
                self._log(name, "gone", s["zone"], s["box"], now, None, picked_up)

    def _snapshot(self, frame, name, box, people, now):
        img = frame.copy()
        for x1, y1, x2, y2 in people:                    # privacy: blur anyone in the shot
            x1, y1 = max(0, x1), max(0, y1)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 0)
        cv2.rectangle(img, box[:2], box[2:], (0, 165, 255), 4)   # the orange box Gemini is told about
        path = f"{SNAP_DIR}/{name.replace(' ', '_')}_{int(now * 1000)}.jpg"
        cv2.imwrite(path, img)
        return path

    def _log(self, item, event, zone, box, at, image, picked_up=False):
        event_id = db.log_event(self.db, item, event, zone, box, at, image, picked_up)
        print(f"{time.strftime('%H:%M:%S', time.localtime(at))}  {item} {event} {zone}"
              + (" (picked up)" if picked_up else ""))
        return event_id


# ---------- detection thread ----------

class Detector(threading.Thread):
    """Runs YOLO on the newest cleaned-up frame, as often as the CPU allows (capped)."""

    def __init__(self, model, names, imgsz, device, memory, announcer):
        super().__init__(daemon=True)
        self.model, self.names, self.imgsz, self.device = model, names, imgsz, device
        self.memory, self.announcer = memory, announcer
        self.frame, self.new = None, threading.Event()
        self.dets, self.ms, self.fps = [], 0.0, 0.0

    def submit(self, frame):
        self.frame = frame
        self.new.set()

    def run(self):
        while True:
            self.new.wait()
            self.new.clear()
            frame, t0 = self.frame, time.monotonic()
            dets = detect(self.model, frame, self.names, self.imgsz, self.device)
            self.dets, self.ms = dets, (time.monotonic() - t0) * 1000
            now = time.time()
            self.announcer.update([d[4] for d in dets], now)
            self.memory.update(frame, dets, now)
            spent = time.monotonic() - t0
            self.fps = 1 / max(spent, 1 / MAX_DETECT_FPS)
            time.sleep(max(0.0, 1 / MAX_DETECT_FPS - spent))


# ---------- window ----------

def draw(frame, dets, status):
    for x1, y1, x2, y2, name, conf in dets:
        colour = (80, 200, 80) if name == "person" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{name} {conf:.0%}", (x1 + 2, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera number or video file")
    parser.add_argument("--items", help="comma-separated names outside the library (uses YOLO-World)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--no-voice", action="store_true", help="don't speak")
    parser.add_argument("--no-mic", action="store_true", help="don't listen to the microphone")
    parser.add_argument("--mic-threshold", type=int, default=100,
                        help="how loud speech must be (watch the mic number in the window)")
    parser.add_argument("--no-window", action="store_true", help="run without the video window")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    key = os.getenv("GEMINI_API_KEY", "").strip()
    # retry busy (429/503) replies a few times: they're usually gone a second later
    retry = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0))
    gemini = genai.Client(api_key=key, http_options=retry) if key else None
    print(f"Gemini: on ({GEMINI_MODEL})" if gemini else
          "Gemini: off (put GEMINI_API_KEY in .env for place descriptions and voice questions)")

    device = pick_device()
    items = [i.strip() for i in args.items.split(",")] if args.items else None
    model, names = load_model(items)
    print(f"device={device} imgsz={args.imgsz} knows {len(names)} things: {', '.join(names[:12])}...")

    cam = open_camera(source)
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("Camera opened but gave no frame. Try --source 1.")
    enhancer = Enhancer()
    detect(model, enhancer(to_work_size(frame)), names, args.imgsz, device)   # warm-up: first run is slow

    delete_old_snapshots()
    speaker = Speaker(enabled=not args.no_voice)
    speaker.start()
    detector = Detector(model, names, args.imgsz, device, Memory(db.open_db(), gemini), Announcer(speaker))
    detector.start()
    listener = None
    if not args.no_mic:
        if gemini:
            listener = Listener(gemini, speaker, [n for n in names if n != "person"], detector,
                                args.mic_threshold)
            listener.start()
            print("Microphone: on. Ask \"where is my phone?\" or \"what can you see?\"")
        else:
            print("Microphone: off (needs GEMINI_API_KEY in .env to understand speech)")

    is_file = not isinstance(source, int)
    frame_gap = 1 / (cam.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
    show_enhanced = True
    fps, fps_frames, fps_t0 = 0.0, 0, time.monotonic()
    started = last_frame = time.monotonic()
    window = "Rewind (Q quit, E enhanced view)"

    while True:
        ok, frame = cam.read()
        if not ok:
            print("end of video" if is_file else "camera stopped giving frames")
            break
        if is_file:                                      # play files at real speed
            time.sleep(max(0.0, frame_gap - (time.monotonic() - last_frame)))
        now = last_frame = time.monotonic()
        fps_frames += 1
        if now - fps_t0 >= 1.0:
            fps, fps_frames, fps_t0 = fps_frames / (now - fps_t0), 0, now

        raw = to_work_size(frame)
        clean = enhancer(raw)
        detector.submit(clean)

        if not args.no_window:
            status = (f"video {fps:.0f} fps | detect {detector.fps:.0f} fps ({detector.ms:.0f} ms) | "
                      f"{'enhanced' if show_enhanced else 'raw'} view")
            if listener:
                status += f" | mic {listener.level}"
            view = (clean if show_enhanced else raw).copy()
            cv2.imshow(window, draw(view, detector.dets, status))
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed in (ord("e"), ord("E")):
                show_enhanced = not show_enhanced
            if key_pressed in (ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        if args.seconds and now - started >= args.seconds:
            break

    print(f"video {fps:.0f} fps, detect {detector.fps:.0f} fps, seeing: "
          + (", ".join(sorted({d[4] for d in detector.dets})) or "nothing"))
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
