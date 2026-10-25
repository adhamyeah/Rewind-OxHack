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

    @property
    def highlight(self):
        with self.lock:
            return self._highlight

    def clear(self):
        with self.lock:
            self._highlight, self.caption = None, ""
