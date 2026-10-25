"""
Rewind - live camera (v7: voice questions, last-seen memory, rewind timeline)
=============================================================================

The whole "where did I leave it?" pipeline:
  zones.py      named areas of the view ("on the kitchen table")     - Stage 3
  memory.py     put down / picked up / held, and when last seen       - Stage 4
  listen.py     spoken questions (ElevenLabs speech-to-text)          - Stage 5
  speak.py      spoken answers (ElevenLabs voice)                     - Stage 5
  timeline.py   where an item has been + replay of the put-down       - Stage 6
  describe.py   Gemini: checks put-downs, scans settled scenes,       - Stage 7
                describes spots, understands odd questions
Everything is saved in rewind.db (+ snapshots/ and clips/).

ASK: just say "where are my glasses?" (or press 1-8 in the video window).
REWIND: say "show me where my glasses were", or press r after asking.
Once Gemini confirms a put-down, Rewind says it out loud too ("Got it. Your
glasses are on the desk...") - turn that off with --quiet.

Setup once: python zone_editor.py (draw zones), and a .env file with
GEMINI_API_KEY=... and ELEVENLABS_API_KEY=... (the key needs Text to Speech
and Speech to Text access). Rewind still runs without any of them: no zones
-> "in front of the camera"; no Gemini key -> no checking or scans; no
ElevenLabs key -> answers are printed and the microphone is off.

Detection (unchanged from v4), measured on 128 labelled photos:

  setup (YOLO-World)              items found   time per run*
  small model, size 480 (v3)        28 of 99       161 ms
  small model, size 640             29 of 99       268 ms
  medium model, size 480            31 of 99       386 ms
  MEDIUM MODEL, SIZE 640 (v4)       40 of 99       606 ms
  large model, size 640             38 of 99      1208 ms
  * on a slow 2-core test machine; your laptop will be faster.

  1. The medium model at size 640 finds ~40% more items. Each detection is
     slower, but detection runs in the background, so the VIDEO stays smooth.
     Only the boxes update less often - fine for things on a table.
  2. People come from a separate small model made for people (yolo11n),
     with a higher bar (0.5), and far-away people in the background are
     ignored. Fewer junk person boxes than asking YOLO-World for "person".
  3. The camera runs at 640x480 like phase1: a narrower view, so items take
     up more of the picture and fewer background people get in.
  4. Mac GPU is off by default (it can silently give worse results with
     YOLO-World); NVIDIA GPUs are used automatically.

Kept from before: smooth threads, live-frame drawing, phase1's change gate
(the model only re-runs when the scene changes, plus a check every 2 s),
the appeared / left-view log, and the "(on person)" label.

Run:
  python phase1_camera_gate.py                     # default webcam
  python phase1_camera_gate.py --source 1          # second camera (e.g. phone webcam)
  python phase1_camera_gate.py --source demo.mp4   # a video file
  python phase1_camera_gate.py --no-gemini         # skip Gemini (offline / save quota)
  python phase1_camera_gate.py --no-voice          # print answers instead of speaking them
  python phase1_camera_gate.py --quiet             # speak only when asked, not on every put-down
  python phase1_camera_gate.py --no-listen         # no microphone (number keys still work)
  python phase1_camera_gate.py --wake              # noisy room: only answer questions starting with "Rewind"
  python phase1_camera_gate.py --no-window --seconds 10   # just print numbers
Keys in the video window: 1-8 ask, r rewind, x close the timeline, q quit.
"""

import argparse
import os
import platform
import threading
import time
from datetime import datetime

import cv2

# --- what to look for (from phase1: works on real glasses) ---
ITEMS = ["keys", "glasses", "wallet", "phone", "mug", "headphones", "pen", "water bottle"]
CONFIDENCE = 0.15          # bar for items (0.2 lost a quarter of the items in testing)

# --- item model ---
MODEL = "yolov8m-worldv2.pt"   # m = medium. "yolov8s-worldv2.pt" is ~2x faster but finds fewer items
IMGSZ = 640                    # 480 is ~1.5x faster but misses small things

# --- people (separate model, made for people) ---
PERSON_MODEL = "yolo11n.pt"
PERSON_CONFIDENCE = 0.5
MIN_PERSON_AREA = 0.03     # ignore people smaller than 3% of the picture (background crowd)

# --- camera ---
CAMERA_W, CAMERA_H = 640, 480  # like phase1: items look bigger than in a wide 1280x720 view
USE_MAC_GPU = False            # True = try Apple "mps" (faster, but can give worse results)

# --- change gate knobs (from phase1) ---
GATE_W, GATE_H = 480, 360  # the diff runs on this small greyscale image, not the full frame
BLUR = 9                   # odd kernel; kills sensor noise
DIFF_THRESHOLD = 5.0       # mean difference that counts as a change; tune with the on-screen number
DRIFT = 0.1                # baseline blend per pass, so slow lighting changes don't retrigger
RECHECK_SECONDS = 2.0      # run the models at least this often, even if nothing changed
PASSES_PER_SECOND = 5      # how often the detector looks at a new frame
SETTLE_SECONDS = 1.5       # still this long after movement = "settled": Gemini scans the whole scene

WINDOW = "Rewind (q to quit)"


# --------------------------------------------------------------------------
# Hardware and models
# --------------------------------------------------------------------------
def pick_device():
    """NVIDIA GPU if there is one; Apple GPU only if USE_MAC_GPU; else CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if USE_MAC_GPU and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_item_model():
    from ultralytics import YOLOWorld

    model = YOLOWorld(MODEL)
    model.set_classes(ITEMS)  # items only: people come from their own model
    return model


def load_person_model():
    from ultralytics import YOLO

    return YOLO(PERSON_MODEL)


# --------------------------------------------------------------------------
# Camera: background reader that only keeps the newest frame
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, source, width=CAMERA_W, height=CAMERA_H):
        self.is_file = not str(source).isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            index = int(source)
            # DirectShow opens much faster than the Windows default backend.
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            # MJPG lets most USB webcams do 30 fps (raw YUYV often can't).
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue old frames

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera or video: {source!r}. Try --source 1")

        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.ended = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:  # video finished
                    self.ended = True
                    break
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            if self.is_file:  # play files at real speed, not as fast as possible
                time.sleep(1.0 / self.file_fps)

    def latest(self):
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.cap.release()


# --------------------------------------------------------------------------
# Detector: change gate + item model + person model, in the background
# --------------------------------------------------------------------------
class Detector:
    def __init__(self, camera, device, on_pass=None, on_settle=None):
        import torch

        self.camera = camera
        self.device = device
        self.on_pass = on_pass      # e.g. memory.update(frame, detections), called every pass
        self.on_settle = on_settle  # e.g. memory.settled(frame, detections), once each time the scene settles
        if device == "cpu":
            # Leave a core free so the video loop never gets starved.
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

        self.item_model = load_item_model()
        self.person_model = load_person_model()
        self.lock = threading.Lock()
        self.detections = []  # list of (x1, y1, x2, y2, name, confidence)
        self.visible = set()
        self.baseline = None  # greyscale gate image of the last inspected scene
        self.diff = 0.0
        self.item_ms = 0.0
        self.person_ms = 0.0
        self.scene_still = False
        self.model_runs = 0
        self.passes = 0
        self.running = True

        self._warm_up()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _warm_up(self):
        """The first call is slow (loading kernels). Do it before we start."""
        import numpy as np

        blank = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        try:
            self._detect(blank)
        except Exception as e:  # e.g. an op the Mac GPU doesn't support
            if self.device == "cpu":
                raise
            print(f"[rewind] GPU ({self.device}) failed: {e}\n[rewind] falling back to CPU")
            self.device = "cpu"
            self._detect(blank)
        self.model_runs = 0

    @staticmethod
    def _gate_image(frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (GATE_W, GATE_H), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (BLUR, BLUR), 0)

    def _detect(self, frame):
        h, w = frame.shape[:2]
        dets = []

        start = time.perf_counter()
        r = self.item_model.predict(frame, conf=CONFIDENCE, imgsz=IMGSZ, device=self.device,
                                    agnostic_nms=True,  # one label per object: no "wallet" AND "phone" on one box
                                    verbose=False)[0]
        self.item_ms = (time.perf_counter() - start) * 1000
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append((int(x1), int(y1), int(x2), int(y2), r.names[int(k)], float(c)))

        start = time.perf_counter()
        r = self.person_model.predict(frame, conf=PERSON_CONFIDENCE, imgsz=IMGSZ, classes=[0],
                                      device=self.device, verbose=False)[0]
        self.person_ms = (time.perf_counter() - start) * 1000
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if (x2 - x1) * (y2 - y1) >= MIN_PERSON_AREA * w * h:  # skip the background crowd
                dets.append((int(x1), int(y1), int(x2), int(y2), "person", float(c)))

        self.model_runs += 1
        return dets

    def _log_changes(self, dets):
        """phase1's appeared / left-view log (items only)."""
        seen = {d[4] for d in dets if d[4] != "person"}
        stamp = datetime.now().strftime("%H:%M:%S")
        for item in sorted(seen - self.visible):
            print(f"{stamp}  {item} appeared")
        for item in sorted(self.visible - seen):
            print(f"{stamp}  {item} left view")
        self.visible = seen

    def _loop(self):
        last_id, last_run = -1, 0.0
        last_change, settled = time.perf_counter(), False  # the first settle scans the room at startup
        gap = 1.0 / PASSES_PER_SECOND
        while self.running:
            frame_id, frame = self.camera.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id
            start = time.perf_counter()

            gate = self._gate_image(frame)
            if self.baseline is None:
                changed, self.diff = True, 0.0  # first frame: always look
            else:
                self.diff = float(cv2.absdiff(self.baseline, gate).mean())
                changed = self.diff > DIFF_THRESHOLD

            if changed or start - last_run > RECHECK_SECONDS:
                dets = self._detect(frame)
                last_run = start
                self.baseline = gate  # snap: this frame is the new "still" scene
                self._log_changes(dets)
                with self.lock:
                    self.detections = dets
                self.scene_still = not changed
            else:
                # Nothing moved: the last boxes are still right, reuse them.
                self.baseline = cv2.addWeighted(gate, DRIFT, self.baseline, 1 - DRIFT, 0)
                dets = self.latest()
                self.scene_still = True

            self.passes += 1
            if self.on_pass:
                self.on_pass(frame, dets)

            # Settled = no change for SETTLE_SECONDS after something moved: fire once.
            if changed:
                last_change, settled = start, False
            elif not settled and start - last_change >= SETTLE_SECONDS:
                settled = True
                if self.on_settle:
                    self.on_settle(frame, dets)

            took = time.perf_counter() - start
            if took < gap:
                time.sleep(gap - took)

    def latest(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


# --------------------------------------------------------------------------
# "Is someone holding it?"
# --------------------------------------------------------------------------
def held_by_person(box, people):
    """True if most of the item's box sits inside a person's box (held or worn).

    Simple on purpose: from a side-on camera, an item on the table behind
    someone can look "held" too.
    """
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for px1, py1, px2, py2 in people:
        ix = max(0, min(x2, px2) - max(x1, px1))
        iy = max(0, min(y2, py2) - max(y1, py1))
        if ix * iy / area > 0.6:
            return True
    return False


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def colour_for(name):
    h = sum(ord(ch) * 31 for ch in name)
    return (h * 67 % 200 + 55, h * 131 % 200 + 55, h * 197 % 200 + 55)


def label_box(frame, x1, y1, text, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


ASK_LEGEND = ("ask: " + "  ".join(f"{i} {name}" for i, name in enumerate(ITEMS, start=1))
              + "  |  r rewind  |  or just ask out loud")


class Questions:
    """Answers "where is my X?" from a number key or a spoken question, and can open the rewind timeline."""

    def __init__(self, memory, voice, wake_word=False):
        self.memory, self.voice, self.wake_word = memory, voice, wake_word
        self.last_item = None
        self.rewind_request = None  # an item whose timeline the window should open (set from any thread)

    def ask(self, item, rewind=False):
        self.last_item = item
        answer = self.memory.answer(item)
        self.voice.say(answer)
        self.memory.note_answer(answer)
        if rewind:
            self.rewind_request = item

    def key(self, index):
        if 0 <= index < len(ITEMS):
            self.ask(ITEMS[index])

    def rewind(self):
        """r key: the timeline of the last item asked about, or else the last one put down."""
        placed = [(s["row"] or 0, name) for name, s in self.memory.items.items() if s["row"]]
        item = self.last_item or (max(placed)[1] if placed else None)
        if item:
            self.rewind_request = item
        else:
            self.voice.say("Ask me about something first, then I can rewind it.")

    def heard(self, text):
        """A transcript from the microphone (background thread)."""
        import re

        from listen import parse_question

        if self.wake_word and "rewind" not in text.lower():
            return
        # "Rewind, where are..." / "Hey Rewind ..." is just calling its name; "rewind my wallet" is a request
        body = re.sub(r"^\W*((hey|ok|okay)\s+rewind\b\W*|rewind\s*[,.!?:]+\s*)", "", text, flags=re.I)
        item, question, rewind = parse_question(body, ITEMS)
        short = len(body.split()) <= 3  # just "glasses?" counts as a question
        describer = self.memory.describer
        if not item and (question or rewind) and describer and describer.enabled:
            item = describer.match_item(body)  # odd phrasing: let Gemini work out the item
        if item and (question or rewind or short):
            self.ask(item, rewind)
        elif question and any(w in body.lower() for w in ("where", "find", "lost")):
            self.voice.say(f"Sorry, I can only look for your {', '.join(ITEMS[:-1])}, or {ITEMS[-1]}.")


def draw(frame, detections, status, zones=None, notes=None, legend=None):
    people = [d[:4] for d in detections if d[4] == "person"]

    if zones:
        from zones import draw_zones

        draw_zones(frame, zones)

    # People first, thin and grey, so items stay the focus.
    for x1, y1, x2, y2 in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
        label_box(frame, x1, y1, "person", (160, 160, 160))

    for x1, y1, x2, y2, name, conf in detections:
        if name == "person":
            continue
        col = colour_for(name)
        text = f"{name} {conf:.2f}"
        thick = 2
        if held_by_person((x1, y1, x2, y2), people):
            text += " (on person)"  # held in a hand, or glasses being worn
            thick = 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        label_box(frame, x1, y1, text, col)

    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (0, 0), (tw + 12, 24), (0, 0, 0), -1)
    cv2.putText(frame, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if legend:
        scale = 0.42
        (lw, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if lw > frame.shape[1] - 12:  # shrink to fit the window width
            scale *= (frame.shape[1] - 12) / lw
            (lw, _), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(frame, (0, 24), (lw + 12, 44), (0, 0, 0), -1)
        cv2.putText(frame, legend, (6, 39), cv2.FONT_HERSHEY_SIMPLEX, scale, (180, 220, 255), 1)

    if notes:  # latest "where is it" lines from memory / Gemini, newest at the bottom
        import textwrap

        h, w = frame.shape[:2]
        width = max(30, int(w / 9.5))  # characters that fit across the frame at this font size
        lines = [part for line in notes for part in textwrap.wrap(line, width)]
        top = h - 22 * len(lines) - 8
        cv2.rectangle(frame, (0, top), (w, h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8, top + 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 255), 1)
    return frame


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Rewind - live camera")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file path")
    ap.add_argument("--no-window", action="store_true", help="don't show video, just print numbers")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = run until q)")
    ap.add_argument("--no-gemini", action="store_true", help="don't send snapshots to Gemini")
    ap.add_argument("--no-voice", action="store_true", help="print answers instead of speaking them")
    ap.add_argument("--quiet", action="store_true", help="only speak when asked, not on every put-down")
    ap.add_argument("--no-listen", action="store_true", help="don't use the microphone (number keys still work)")
    ap.add_argument("--wake", action="store_true", help='only answer spoken questions that start with "Rewind"')
    args = ap.parse_args()

    from listen import Listener          # microphone + ElevenLabs speech-to-text
    from memory import Memory, plural    # zones + put-down log + Gemini
    from speak import Voice              # ElevenLabs voice
    from timeline import WINDOW as TIMELINE_WINDOW, TimelineView

    device = pick_device()
    print(f"[rewind] device={device}  imgsz={IMGSZ}  items={MODEL}  people={PERSON_MODEL}")
    if args.no_voice:
        voice = type("PrintOnly", (), {"say": staticmethod(lambda text: print(f"[voice] {text}")),
                                       "busy": staticmethod(lambda: False)})()
    else:
        voice = Voice()

    def announce(item, phrase):  # Gemini confirmed a put-down
        voice.say(f"Got it. Your {item} {'are' if plural(item) else 'is'} {phrase}.")

    memory = Memory(use_gemini=not args.no_gemini, on_confirmed=None if args.quiet else announce, items=ITEMS)
    questions = Questions(memory, voice, wake_word=args.wake)
    listener = None if args.no_listen else Listener(questions.heard, is_busy=voice.busy)
    print(f"[rewind] {ASK_LEGEND}   (keys work in the video window)")
    print("[rewind] loading models (first run downloads them, give it a minute)")

    timeline = [None]  # the open TimelineView, if any

    def on_key(key):
        """True = quit. Number keys ask where an item is; r opens the rewind timeline; x closes it."""
        if key in (ord("q"), ord("Q")):
            return True
        if ord("1") <= key <= ord("9"):
            questions.key(key - ord("1"))
        elif key in (ord("r"), ord("R")):
            questions.rewind()
        elif key in (ord("x"), ord("X"), 27) and timeline[0]:
            timeline[0] = None
            cv2.destroyWindow(TIMELINE_WINDOW)
        return False

    cam = Camera(args.source)
    det = Detector(cam, device, on_pass=memory.update, on_settle=memory.settled)
    cam.start()  # only start reading once the models are ready
    if listener:
        listener.start()

    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # drag the corner to make it bigger

    shown, video_fps = -1, 0.0
    started = time.perf_counter()
    last_print = started
    window_start, window_frames = started, 0  # for measuring video fps
    try:
        while True:
            frame_id, frame = cam.latest()
            if frame is None or frame_id == shown:
                if cam.ended:
                    break
                if not args.no_window and on_key(cv2.waitKey(5) & 0xFF):
                    break
                time.sleep(0.002)
            else:
                shown = frame_id
                now = time.perf_counter()
                window_frames += 1
                if now - window_start >= 1.0:  # NEW pictures shown in the last second
                    video_fps = window_frames / (now - window_start)
                    window_start, window_frames = now, 0

                dets = det.latest()
                state = "still" if det.scene_still else "moving"
                status = (f"video {video_fps:.0f} fps | items {det.item_ms:.0f} ms | people {det.person_ms:.0f} ms"
                          f" | diff {det.diff:.1f}/{DIFF_THRESHOLD:.1f} {state}")
                if args.no_window:
                    if now - last_print > 1:
                        seen = ", ".join(f"{d[4]} {d[5]:.2f}" for d in dets) or "-"
                        print(f"{status} | model runs {det.model_runs}/{det.passes} passes | seen: {seen}")
                        last_print = now
                else:
                    view = draw(frame.copy(), dets, status, memory.zones, memory.recent_notes(), ASK_LEGEND)
                    cv2.imshow(WINDOW, view)  # always the LIVE frame

                    if questions.rewind_request:  # opened by voice ("show me my glasses") or the r key
                        timeline[0] = TimelineView(questions.rewind_request, since=memory.started)
                        questions.rewind_request = None
                    if timeline[0]:
                        board = timeline[0].frame()
                        if board is None:  # timed out
                            timeline[0] = None
                            cv2.destroyWindow(TIMELINE_WINDOW)
                        else:
                            cv2.imshow(TIMELINE_WINDOW, board)

                    if on_key(cv2.waitKey(1) & 0xFF) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    finally:
        det.stop()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
