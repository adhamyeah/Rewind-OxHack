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


def describe_surroundings(box, others, width):
    """'next to the mug' / 'next to the mug and the pen' / 'on its own'.

    `others` are the other detections in the same keyframe. We name the (up to two)
    closest ones within NEAR_FRACTION of the frame width, nearest first. No coordinates
    in the wording: people think "next to the mug", not "bottom right".
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    near = []
    for o in others:
        d = (((o[0] + o[2]) / 2 - cx) ** 2 + ((o[1] + o[3]) / 2 - cy) ** 2) ** 0.5
        if d <= config.NEAR_FRACTION * width:
            near.append((d, o[4]))
    names = [label for _, label in sorted(near)[:2]]
    if not names:
        return "on its own"
    return "next to " + " and ".join(f"the {n}" for n in names)


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
            seen_with = {d[4]: d for d in self.keyframes[st.keyframe].detections if d[4] != label}
            place = describe_surroundings(st.box, seen_with.values(), self.width)
            if st.present:
                place = "on the table" if place == "on its own" else f"on the table, {place}"
                text = f"The {label} {'are' if verb == 'were' else 'is'} {place}."
                return Answer(text, st.keyframe, st.box)
            when = ago(self.clock() - st.time)
            has = "They've" if verb == "were" else "It's"
            text = f"The {label} {verb} last seen {when}{'' if place == 'on its own' else ' ' + place}. {has} been taken off the table."
            return Answer(text, st.keyframe, st.box)

    def pending_misses(self):
        """True if some item is missing from the latest keyframe but not yet counted as gone.
        The next keyframe will settle it, so the app asks for one instead of waiting for motion."""
        with self.lock:
            return any(st.present and st.misses > 0 for st in self.items.values())

    def present_items(self):
        with self.lock:
            return [l for l, s in self.items.items() if s.present]

    def recent_events(self, n=3):
        with self.lock:
            return list(self.events[-n:])
