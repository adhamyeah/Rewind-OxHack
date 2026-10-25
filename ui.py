"""Drawing and the thread-safe requests other threads make of the window."""

import threading

import cv2

import config


def colour_for(name):
    h = sum(ord(ch) * 31 for ch in name)
    return (h * 67 % 200 + 55, h * 131 % 200 + 55, h * 197 % 200 + 55)


def label_box(frame, x1, y1, text, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_detections(frame, detections):
    """Boxes + labels on `frame` (in place). People are thin and grey so items stay the focus."""
    for x1, y1, x2, y2, name, conf in detections:
        if name == "person":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            continue
        col = colour_for(name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label_box(frame, x1, y1, f"{name} {conf:.2f}" if conf < 1.0 else name, col)
    return frame


def draw_highlight(frame, box, label):
    """Thick yellow box around the item the user asked about."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 255), 4)
    label_box(frame, x1, y1 - 4, label, (0, 255, 255))
    return frame


def draw_text_bar(frame, text, top=True, colour=(255, 255, 255)):
    """A readable strip of text across the top or bottom of the frame."""
    h, w = frame.shape[:2]
    y0 = 0 if top else h - 26
    cv2.rectangle(frame, (0, y0), (w, y0 + 26), (0, 0, 0), -1)
    cv2.putText(frame, text[:110], (6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return frame


class UIState:
    """Other threads (answerer, knob) ask the window to do things; the main loop reads these.
    OpenCV window calls must stay on the main thread, so threads only ever set flags here."""

    def __init__(self):
        self.lock = threading.Lock()
        self._jump = None        # keyframe index the window should jump to
        self._highlight = None   # (box, label) to draw on the shown keyframe
        self.caption = ""

    def request_jump(self, keyframe, box=None, label="", caption=""):
        with self.lock:
            self._jump = keyframe
            self._highlight = (box, label) if box else None
            self.caption = caption

    def take_jump(self):
        """Main loop: returns the pending keyframe index once, then None."""
        with self.lock:
            j, self._jump = self._jump, None
            return j

    def take_web_jump(self):
        """Web page: the pending jump as a dict (keyframe, label, caption) once, then None."""
        with self.lock:
            j, self._jump = self._jump, None
            if j is None:
                return None
            label = self._highlight[1] if self._highlight else ""
            return {"keyframe": j, "label": label, "caption": self.caption}

    @property
    def highlight(self):
        with self.lock:
            return self._highlight

    def clear(self):
        with self.lock:
            self._highlight, self.caption = None, ""


def overlap(a, b):
    """How much of the smaller box lies inside the other (0..1). Gemini and YOLO draw the same
    object with slightly different boxes, so plain IoU is too strict."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    return ix * iy / smaller


def relabel_with_memory(yolo_dets, memory_dets, min_overlap=0.5):
    """YOLO's fast boxes with memory's (Gemini's) names. A YOLO item box that overlaps a known item
    takes its name; one that matches nothing is dropped, because that is where YOLO's wrong guesses
    ("remote control" for a water bottle) come from. People pass through. One box per known item."""
    out = [d for d in yolo_dets if d[4] == "person"]
    best = {}  # index of the memory box -> best (highest-confidence) YOLO box matching it
    for d in yolo_dets:
        if d[4] == "person":
            continue
        scored = [(overlap(d[:4], m[:4]), i) for i, m in enumerate(memory_dets)]
        score, i = max(scored, default=(0, None))
        if i is not None and score >= min_overlap and (i not in best or d[5] > best[i][5]):
            best[i] = d
    for i, d in best.items():
        out.append(d[:4] + (memory_dets[i][4], d[5]))
    return out


def pick_live_boxes(yolo_dets, last_keyframe_dets, scene_changed, memory_busy):
    """Which boxes the LIVE view draws.

    Settled and memory caught up: memory's boxes (Gemini's names).
    Moving or memory still thinking: YOLO's boxes, which follow movement quickly, but renamed by
    memory so the live view never shows a name the answers would not use.
    Before the first keyframe there is no memory yet, so YOLO's raw boxes are all we have.
    """
    if last_keyframe_dets is None:
        return yolo_dets
    if scene_changed or memory_busy:
        return relabel_with_memory(yolo_dets, last_keyframe_dets)
    # keep YOLO's grey person boxes; items come from memory
    return [d for d in yolo_dets if d[4] == "person"] + list(last_keyframe_dets)
