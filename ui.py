"""
ui.py - drawing for the web page, and the small notes other threads leave for it.

  live_boxes()   which boxes the LIVE view draws (YOLO's fast boxes, named the way memory names them)
  draw_*()       boxes, labels and the yellow "this one" highlight
  replay_gif()   a saved moment as the little rewind gif: the seconds leading up to it, then it
  UIState        "jump to this moment", "this was the last answer", "someone is holding Talk"
"""

import io
import threading
import time

import cv2
import numpy as np

GIF_SIZE = (480, 360)       # the gif is drawn at this size (the page scales it to fit)
GIF_FRAME_MS = 150          # ~6.5 frames a second, like the old timeline window
GIF_HOLD_MS = 2500          # how long the gif rests on the final picture before it loops
REWIND_COLOUR = (0, 165, 255)   # orange (BGR)
HIGHLIGHT = (0, 255, 255)       # yellow (BGR)
END_LABELS = {"put down": "put down here", "put back": "put back here", "seen": "here",
              "picked up": "picked up from here", "out of view": "last seen here"}


def colour_for(name):
    h = sum(ord(ch) * 31 for ch in name)
    return (h * 67 % 200 + 55, h * 131 % 200 + 55, h * 197 % 200 + 55)


def label_box(frame, x1, y1, text, colour, scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)


def held_by_person(box, people):
    """Most of the item's box inside a person's box = held in a hand, or worn."""
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for px1, py1, px2, py2 in people:
        ix = max(0, min(x2, px2) - max(x1, px1))
        iy = max(0, min(y2, py2) - max(y1, py1))
        if ix * iy / area > 0.6:
            return True
    return False


def draw_detections(frame, detections, people=None):
    """Boxes + labels on `frame` (in place). People are thin and grey so items stay the focus.
    Items in someone's hand (or worn) get a thick box and "(on person)"."""
    if people is None:
        people = [d[:4] for d in detections if d[4] == "person"]
    for x1, y1, x2, y2 in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
    for x1, y1, x2, y2, name, conf in detections:
        if name == "person":
            continue
        col = colour_for(name)
        held = held_by_person((x1, y1, x2, y2), people)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4 if held else 2)
        text = f"{name} {conf:.2f}" if conf < 1.0 else name
        label_box(frame, x1, y1, text + (" (on person)" if held else ""), col)
    return frame


def draw_highlight(frame, box, label, scale=1.0):
    """Thick yellow box around the item the person asked about."""
    x1, y1, x2, y2 = (int(v * scale) for v in box)
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), HIGHLIGHT, 4)
    label_box(frame, max(0, x1 - 4), max(20, y1 - 4), label, HIGHLIGHT)
    return frame


def overlap(a, b):
    """How much of the smaller box lies inside the other (0..1). Gemini and YOLO draw the same
    object with slightly different boxes, so plain IoU is too strict."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    return ix * iy / smaller


def live_boxes(yolo_dets, known=(), memory_only=(), ruled_out=(), min_overlap=0.5):
    """Which boxes the LIVE view draws.

    yolo_dets    the fast detector's boxes (follow movement quickly)
    known        items memory has at their spot: a YOLO box on one takes memory's name, so the
                 live view never calls the water bottle a "mug" when the answers say "water bottle"
    memory_only  items only Gemini found (the fast detector can't see them): drawn from memory
    ruled_out    boxes Gemini said are NOT that item: YOLO boxes there are hidden
    """
    out = [d for d in yolo_dets if d[4] == "person"]
    named = {}  # known index -> best YOLO box on it
    for d in yolo_dets:
        if d[4] == "person":
            continue
        score, i = max(((overlap(d[:4], k[:4]), i) for i, k in enumerate(known)), default=(0, None))
        if i is not None and score >= min_overlap:
            if i not in named or d[5] > named[i][5]:
                named[i] = d
        elif not any(r[4] == d[4] and overlap(d[:4], r[:4]) >= min_overlap for r in ruled_out):
            out.append(d)
    out += [tuple(d[:4]) + (known[i][4], d[5]) for i, d in named.items()]
    shown = {d[4] for d in out}
    out += [tuple(m) for m in memory_only if m[4] not in shown]
    return out


def decode(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def moment_picture(kf, highlight="", box=None):
    """A saved moment as a still picture with its boxes (and the yellow highlight)."""
    image = decode(kf.still)
    draw_detections(image, list(kf.items), people=list(kf.people))
    if highlight:
        hl = box or next((d[:4] for d in kf.items if d[4] == highlight), None)
        if hl:
            draw_highlight(image, hl, highlight)
    return image


def replay_gif(kf, highlight="", box=None, kind=""):
    """A saved moment as the rewind gif: the seconds before it, then the moment itself (looping)."""
    from PIL import Image

    w, h = GIF_SIZE
    frames = []
    for jpg in kf.clip:
        small = cv2.resize(decode(jpg), GIF_SIZE, interpolation=cv2.INTER_LINEAR)
        cv2.putText(small, "<< REWIND", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(small, "<< REWIND", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, REWIND_COLOUR, 2)
        frames.append(small)

    final = decode(kf.still)
    scale = w / final.shape[1]
    final = cv2.resize(final, GIF_SIZE, interpolation=cv2.INTER_AREA)
    hl = box or (next((d[:4] for d in kf.items if d[4] == highlight), None) if highlight else None)
    if highlight and hl:
        draw_highlight(final, hl, highlight, scale)
        end = END_LABELS.get(kind, "here")
    else:
        for x1, y1, x2, y2, name, _ in kf.items:
            x1, y1, x2, y2 = (int(v * scale) for v in (x1, y1, x2, y2))
            cv2.rectangle(final, (x1, y1), (x2, y2), colour_for(name), 2)
            label_box(final, x1, y1, name, colour_for(name), 0.5)
        end = ""
    if end:
        cv2.putText(final, end, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(final, end, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HIGHLIGHT, 2)
    frames.append(final)

    # FASTOCTREE: ~25x faster than Pillow's default colour reduction, and looks the same here
    pictures = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(256, method=Image.Quantize.FASTOCTREE)
                for f in frames]
    buf = io.BytesIO()
    pictures[0].save(buf, format="GIF", save_all=True, append_images=pictures[1:], loop=0,
                     duration=[GIF_FRAME_MS] * (len(pictures) - 1) + [GIF_HOLD_MS])
    return buf.getvalue()


class UIState:
    """Other threads (the answerer, the microphone) leave notes here; the web page reads them."""

    def __init__(self):
        self.lock = threading.Lock()
        self._jump = None        # {"keyframe", "box", "label", "kind", "caption"} the page should replay
        self.answer = None       # {"id", "text", "heard", "source", "t"} the latest answer, however it was asked
        self._answers = 0
        self._talk_until = 0.0   # someone is holding the page's Talk button until about then

    def request_jump(self, jump, caption=""):
        with self.lock:
            self._jump = dict(jump, caption=caption)

    def take_web_jump(self):
        """The pending jump once, then None."""
        with self.lock:
            j, self._jump = self._jump, None
            return j

    def set_answer(self, text, heard=None, source="typed"):
        with self.lock:
            self._answers += 1
            self.answer = {"id": self._answers, "text": text, "heard": heard, "source": source, "t": time.time()}

    def talk(self, on):
        """Talk button held (on) or let go: the always-on microphone ignores the room meanwhile."""
        self._talk_until = time.time() + 20 if on else 0.0

    def talking(self):
        return time.time() < self._talk_until
