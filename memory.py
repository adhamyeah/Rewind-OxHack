"""
Memory: the history of the table.

Each keyframe is one settled snapshot with its detections. For each item we track
where it was last seen and turn changes into "appeared / moved / left" events.
An item only counts as "left" after MISSES_TO_LEAVE keyframes in a row without it,
because detectors blink: one missed frame is not the item leaving.
"""

import threading
import time
from dataclasses import dataclass, field

import config


@dataclass
class Keyframe:
    index: int
    time: float
    image: object            # BGR frame, or None in tests
    detections: list         # [(x1, y1, x2, y2, label, conf)]


@dataclass
class ItemState:
    box: tuple               # last seen (x1, y1, x2, y2)
    keyframe: int            # index of the keyframe where it was last seen
    time: float
    present: bool = True
    misses: int = 0
    left_time: float = 0.0


@dataclass
class Event:
    time: float
    label: str
    kind: str                # "appeared" | "moved" | "left"
    keyframe: int
    box: tuple


@dataclass
class Answer:
    text: str
    keyframe: int = None     # keyframe the window should jump to (None = nothing to show)
    box: tuple = None


def describe_position(box, width, height):
    """'top left', 'middle', 'bottom right'... by splitting the frame into thirds."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    col = min(2, int(cx / width * 3))
    row = min(2, int(cy / height * 3))
    rows, cols = ["top", "middle", "bottom"], ["left", "middle", "right"]
    if row == 1 and col == 1:
        return "middle"
    if row == 1:
        return cols[col]
    if col == 1:
        return rows[row]
    return f"{rows[row]} {cols[col]}"


def ago(seconds):
    seconds = int(seconds)
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    return f"{hours} hour{'s' if hours != 1 else ''} ago"


class Memory:
    def __init__(self, width=config.CAMERA_W, height=config.CAMERA_H, clock=time.time):
        self.width, self.height = width, height
        self.clock = clock
        self.lock = threading.Lock()
        self.keyframes = []
        self.items = {}      # label -> ItemState
        self.events = []

    def add_keyframe(self, detections, image=None, when=None):
        """Record a keyframe. Returns (Keyframe, [new Events])."""
        when = self.clock() if when is None else when
        with self.lock:
            kf = Keyframe(len(self.keyframes), when, image, list(detections))
            self.keyframes.append(kf)
            new_events = []

            # One box per label: two identical items collapse into the most confident one.
            best = {}
            for d in detections:
                if d[4] not in best or d[5] > best[d[4]][5]:
                    best[d[4]] = d

            for label, d in best.items():
                box = d[:4]
                old = self.items.get(label)
                if old is None or not old.present:
                    new_events.append(Event(when, label, "appeared", kf.index, box))
                elif self._shift(old.box, box) > config.MOVE_FRACTION * self.width:
                    new_events.append(Event(when, label, "moved", kf.index, box))
                self.items[label] = ItemState(box, kf.index, when)

            for label, st in self.items.items():
                if label in best or not st.present:
                    continue
                st.misses += 1
                if st.misses >= config.MISSES_TO_LEAVE:
                    st.present, st.left_time = False, when
                    new_events.append(Event(when, label, "left", st.keyframe, st.box))

            self.events.extend(new_events)
            return kf, new_events

    @staticmethod
    def _shift(a, b):
        return ((((a[0] + a[2]) - (b[0] + b[2])) / 2) ** 2 + (((a[1] + a[3]) - (b[1] + b[3])) / 2) ** 2) ** 0.5

    def where(self, label):
        """Template answer for 'where is X?'. Never generates free text."""
        verb = "were" if label in config.PLURAL_ITEMS else "was"
        with self.lock:
            st = self.items.get(label)
            if st is None:
                return Answer(f"I haven't seen the {label} yet.")
            place = describe_position(st.box, self.width, self.height)
            if st.present:
                text = f"The {label} {'are' if verb == 'were' else 'is'} on the table, {place}."
                return Answer(text, st.keyframe, st.box)
            when = ago(self.clock() - st.time)
            text = f"The {label} {verb} last seen {when}, {place}. It's been taken off the table."
            return Answer(text, st.keyframe, st.box)

    def present_items(self):
        with self.lock:
            return [l for l, s in self.items.items() if s.present]

    def recent_events(self, n=3):
        with self.lock:
            return list(self.events[-n:])
