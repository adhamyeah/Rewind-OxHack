import time

import answer as answer_mod
from answer import Answerer
from memory import Memory
from ui import UIState


def make(ready=lambda: True):
    m = Memory(640, 480)
    m.add_keyframe([(100, 100, 140, 140, "mug", 1.0)])
    spoken = []
    return m, UIState(), spoken, Answerer(m, UIState(), speak=spoken.append, pick_label=lambda q: None, ready=ready)


def test_question_waits_until_memory_is_ready(monkeypatch):
    monkeypatch.setattr(answer_mod.config, "ASK_WAIT_SECONDS", 2)
    state = {"calls": 0}

    def ready():
        state["calls"] += 1
        return state["calls"] > 5

    _, _, _, a = make(ready)
    start = time.time()
    assert "mug" in a.ask("where is the mug")
    assert 0.2 < time.time() - start < 1.5


def test_question_does_not_wait_forever(monkeypatch):
    monkeypatch.setattr(answer_mod.config, "ASK_WAIT_SECONDS", 0.3)
    _, _, _, a = make(lambda: False)
    start = time.time()
    a.ask("where is the mug")
    assert time.time() - start < 1.5
