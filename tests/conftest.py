import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import project modules from tests/

ITEMS = ["keys", "glasses", "wallet", "phone", "mug", "headphones", "pen", "water bottle"]


class Clock:
    """A fake time.time() the tests move forward by hand."""

    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    import memory

    c = Clock()
    monkeypatch.setattr(memory.time, "time", c)
    return c


@pytest.fixture()
def mem(tmp_path, monkeypatch, clock):
    """A fresh Memory in an empty folder (its own rewind.db, no zones, no Gemini)."""
    monkeypatch.chdir(tmp_path)
    from memory import Memory

    return Memory(db_path=str(tmp_path / "rewind.db"), use_gemini=False, items=ITEMS)


def frame(level=90):
    return np.full((480, 640, 3), level, np.uint8)


def run(mem, clock, dets, seconds, level=90):
    """Feed the memory 5 detector passes a second, like the real detector."""
    f = frame(level)
    for _ in range(int(seconds * 5)):
        mem.update(f, dets)
        clock.t += 0.2
