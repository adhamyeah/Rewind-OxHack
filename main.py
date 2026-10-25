"""Rewind: watches the webcam and remembers where things are put down.

Live window shows the camera with boxes. YOLO-World only runs once the scene
has changed and then gone still (the hand has left), plus a slow re-check.
When an item settles somewhere, a row goes into rewind.db with a photo
(people blurred), and Gemini describes the spot in the background
("next to the blue mug") if GEMINI_API_KEY is set in .env.

Run:  python main.py                         (Q in the window to quit)
      python main.py --source 1              (another camera)
      python main.py --source clip.mp4       (a video file)
      python main.py --items "keys,glasses"  (other item names)
"""

import argparse
import glob
import os
import sys
import threading
import time

import cv2
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from ultralytics import YOLOWorld

import db

load_dotenv()                       # GEMINI_API_KEY (and optional GEMINI_MODEL) from .env

# --- what to look for ---
ITEMS = ["keys", "glasses", "wallet", "phone", "remote control", "mug", "headphones", "pen"]
CONFIDENCE = {"keys": 0.08, "glasses": 0.1, "pen": 0.1}   # small things need a lower bar
DEFAULT_CONFIDENCE = 0.15
MAX_BOX_FRACTION = 0.3      # drop an "item" covering more than this share of the frame
MODEL = "yolov8s-worldv2.pt"

# --- when to look ---
GATE_W, GATE_H = 320, 240   # change detection runs on this small greyscale image
PIXEL_DIFF = 25             # grey levels a pixel must change by to count as changed
CHANGE_PCT = 0.3            # % of pixels changed since the last look = "something happened"
STILL_PCT = 0.2             # % of pixels changing frame-to-frame below which the scene is still
SETTLE_SECONDS = 0.6        # still this long = the hand has gone, look now
DRIFT = 0.02                # baseline blend per frame; stops slow lighting changes retriggering
RECHECK_SECONDS = 1.0       # look again this soon after an item goes missing
HEARTBEAT_SECONDS = 10.0    # look anyway this often

# --- memory rules ---
MISSES_FOR_GONE = 2         # missing on 2 looks in a row = gone (one miss is often YOLO, not reality)
SPOT_CHANGED = 25.0         # grey difference where the item sat that means it really left
SNAP_DIR = "snapshots"
KEEP_SNAPSHOTS_HOURS = 24

# --- Gemini ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")   # override in .env
DESCRIBE_PROMPT = (
    "The {item} is inside the orange box. In under 15 words, say where it is relative to "
    "nearby objects, as you would to an elderly person. Start with a word like 'on', 'in' "
    "or 'next to'. Reply with the phrase only.")


# ---------- camera ----------

def open_camera(source):
    if not isinstance(source, int):                      # a video file
        cam = cv2.VideoCapture(source)
        if not cam.isOpened():
            raise SystemExit(f"Could not open video file {source}.")
        return cam
    cam = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform == "win32" else None
    if cam is None or not cam.isOpened():
        cam = cv2.VideoCapture(source)
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


def to_gate(frame):
    gate = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gate = cv2.resize(gate, (GATE_W, GATE_H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gate, (9, 9), 0)


def changed_pct(a, b):
    return float((cv2.absdiff(a, b) > PIXEL_DIFF).mean() * 100)


# ---------- detection ----------

def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect(model, frame, names, imgsz, device):
    result = model.predict(frame, conf=min(DEFAULT_CONFIDENCE, *CONFIDENCE.values()),
                           imgsz=imgsz, device=device, verbose=False)[0]
    frame_area = frame.shape[0] * frame.shape[1]
    dets = []
    for (x1, y1, x2, y2), c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                         result.boxes.conf.tolist()):
        name = names[int(c)]
        if name != "person":
            if conf < CONFIDENCE.get(name, DEFAULT_CONFIDENCE):
                continue
            if (x2 - x1) * (y2 - y1) > MAX_BOX_FRACTION * frame_area:
                continue
        dets.append((int(x1), int(y1), int(x2), int(y2), name, float(conf)))
    return dets


def rough_place(box, width):
    """Fallback place when Gemini isn't available: left / middle / right of the view."""
    x = (box[0] + box[2]) / 2 / width
    if x < 1 / 3:
        return "on the left of the camera view"
    if x < 2 / 3:
        return "in the middle of the camera view"
    return "on the right of the camera view"


# ---------- Gemini ----------

def describe(client, image_path, item):
    """One plain-English phrase about where the item is, from its snapshot. None on failure."""
    with open(image_path, "rb") as f:
        image = types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=[image, DESCRIBE_PROMPT.format(item=item)])
    except errors.ClientError as e:
        if e.code in (401, 403):
            print("Gemini: the API key in .env was rejected; using rough places instead")
        elif e.code == 429:
            print("Gemini: rate limited; using the rough place for this one")
        else:
            print(f"Gemini: request error {e.code} {e.message}; using the rough place")
        return None
    except errors.ServerError as e:
        print(f"Gemini: server error {e.code}; using the rough place for this one")
        return None
    except Exception as e:                               # network errors (httpx) and anything else
        print(f"Gemini: couldn't get a description ({type(e).__name__}); using the rough place")
        return None
    text = (response.text or "").strip().rstrip(".")   # empty if Gemini blocked the image
    return text[:1].lower() + text[1:] or None         # it goes mid-sentence: "Your keys are on..."


def describe_in_background(client, event_id, image_path, item):
    def run():
        phrase = describe(client, image_path, item)
        if phrase:
            con = db.open_db()               # own connection: sqlite objects stay in their thread
            db.set_description(con, event_id, phrase)
            con.close()
            print(f"            Gemini: {item} {phrase}")
    threading.Thread(target=run, daemon=True).start()


# ---------- memory: placed / gone ----------

def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def grey_crop(frame, box):
    x1, y1, x2, y2 = box
    return cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)


def delete_old_snapshots():
    """Privacy: item photos never outlive a day."""
    for f in glob.glob(f"{SNAP_DIR}/*.jpg"):
        if time.time() - os.path.getmtime(f) > KEEP_SNAPSHOTS_HOURS * 3600:
            os.remove(f)


class Memory:
    def __init__(self, con, gemini):
        self.db, self.gemini, self.items = con, gemini, {}
        os.makedirs(SNAP_DIR, exist_ok=True)

    def update(self, frame, dets):
        """One look at a still scene. Returns True if it wants another look soon."""
        now = time.time()
        people = [d[:4] for d in dets if d[4] == "person"]
        best = {}                                        # strongest box per item
        for x1, y1, x2, y2, name, conf in dets:
            if name != "person" and conf > best.get(name, (0, None))[0]:
                best[name] = (conf, (x1, y1, x2, y2))

        for name, (conf, box) in best.items():
            s = self.items.setdefault(name, dict(present=False, zone=None, box=None,
                                                 crop=None, misses=0, touched=False))
            zone = rough_place(box, frame.shape[1])
            s["misses"] = 0
            if s["present"] and zone == s["zone"]:
                s["box"], s["crop"] = box, grey_crop(frame, box)   # same spot: keep tracking
                continue
            s.update(present=True, zone=zone, box=box, crop=grey_crop(frame, box), touched=False)
            image = self._snapshot(frame, name, box, people, now)
            event_id = self._log(name, "placed", zone, box, now, image)
            if self.gemini:
                describe_in_background(self.gemini, event_id, image, name)

        unsure = False
        for name, s in self.items.items():
            if name in best or not s["present"]:
                continue
            if any(overlaps(p, s["box"]) for p in people):
                s["touched"] = True
            s["misses"] += 1
            if s["misses"] < MISSES_FOR_GONE:
                unsure = True
                continue
            picked_up = s["touched"] or self._spot_changed(frame, s)
            s.update(present=False, misses=0)
            self._log(name, "gone", s["zone"], s["box"], now, None, picked_up)
        return unsure

    def _spot_changed(self, frame, s):
        now_crop = grey_crop(frame, s["box"])
        if s["crop"] is None or now_crop.shape != s["crop"].shape or now_crop.size == 0:
            return False
        return float(cv2.absdiff(now_crop, s["crop"]).mean()) > SPOT_CHANGED

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


# ---------- window ----------

def draw(frame, dets, status):
    for x1, y1, x2, y2, name, conf in dets:
        colour = (160, 160, 160) if name == "person" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(frame, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera number or video file")
    parser.add_argument("--items", help="comma-separated item names, instead of ITEMS")
    parser.add_argument("--imgsz", type=int, help="YOLO input size (default 480 on CPU, 640 on GPU)")
    parser.add_argument("--no-window", action="store_true", help="run without the video window")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    key = os.getenv("GEMINI_API_KEY", "").strip()
    # retry busy (429/503) replies a few times: they're usually gone a second later
    retry = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0))
    gemini = genai.Client(api_key=key, http_options=retry) if key else None
    print(f"Gemini descriptions: on ({GEMINI_MODEL})" if gemini else
          "Gemini descriptions: off (put GEMINI_API_KEY in .env to turn on)")

    device = pick_device()
    imgsz = args.imgsz or (480 if device == "cpu" else 640)
    items = [i.strip() for i in args.items.split(",")] if args.items else ITEMS
    names = items + ["person"]                           # person: for blurring and "picked up"
    model = YOLOWorld(MODEL)
    model.set_classes(names)

    cam = open_camera(source)
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("Camera opened but gave no frame. Try --source 1.")
    model.predict(frame, imgsz=imgsz, device=device, verbose=False)   # warm-up: first run is slow
    print(f"device={device} imgsz={imgsz} looking for: {', '.join(items)}")

    delete_old_snapshots()
    memory = Memory(db.open_db(), gemini)
    is_file = not isinstance(source, int)
    frame_gap = 1 / (cam.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0

    dets = []
    baseline = prev = None
    pending = True                                       # look once the first scene is still
    still_since = next_check = time.monotonic()
    detect_ms = change = 0.0
    fps, fps_frames, fps_t0 = 0.0, 0, time.monotonic()
    started = last_frame = time.monotonic()
    window = "Rewind (Q to quit)"

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

        gate = to_gate(frame)
        if baseline is None:
            baseline = prev = gate
        if changed_pct(prev, gate) > STILL_PCT:
            still_since = now                            # something is moving: wait
        prev = gate
        change = changed_pct(baseline, gate)
        if change > CHANGE_PCT:
            pending = True

        if now - still_since >= SETTLE_SECONDS and (pending or now >= next_check):
            t0 = time.monotonic()
            dets = detect(model, frame, names, imgsz, device)
            detect_ms = (time.monotonic() - t0) * 1000
            unsure = memory.update(frame, dets)
            baseline, pending = gate, False
            next_check = time.monotonic() + (RECHECK_SECONDS if unsure else HEARTBEAT_SECONDS)
        elif not pending:
            baseline = cv2.addWeighted(gate, DRIFT, baseline, 1 - DRIFT, 0)

        if not args.no_window:
            state = "waiting for still" if pending else "watching"
            status = f"{fps:.0f} fps | change {change:.1f}/{CHANGE_PCT}% | {state}"
            if detect_ms:
                status += f" | yolo {detect_ms:.0f} ms"
            cv2.imshow(window, draw(frame, dets, status))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")) or \
                    cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        if args.seconds and now - started >= args.seconds:
            break

    print(f"video {fps:.0f} fps, last yolo {detect_ms:.0f} ms, seeing: "
          + (", ".join(d[4] for d in dets) or "nothing"))
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
