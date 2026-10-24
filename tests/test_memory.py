import numpy as np
import pytest

import db
import memory
from memory import Memory

TABLE = {"on the kitchen table": np.array([[0, 0], [300, 0], [300, 240], [0, 240]], np.int32)}
KEYS_TABLE = (100, 100, 140, 130, "keys", 0.6)
KEYS_SHELF = (400, 100, 440, 130, "keys", 0.6)     # right of every zone -> fallback name


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Memory(db.open_db(), TABLE)


def frame(value=200):
    return np.full((240, 480, 3), value, np.uint8)


def events(mem):
    return [(r["item"], r["event"], r["zone"], r["picked_up"])
            for r in mem.db.execute("SELECT * FROM events ORDER BY id")]


def test_placed_logs_zone_and_snapshot(mem, tmp_path):
    mem.update(frame(), [KEYS_TABLE], now=1)
    assert events(mem) == [("keys", "placed", "on the kitchen table", 0)]
    image = mem.db.execute("SELECT image FROM events").fetchone()["image"]
    assert (tmp_path / image).exists()


def test_seeing_it_again_in_same_place_logs_nothing(mem):
    for t in range(5):
        mem.update(frame(), [KEYS_TABLE], now=t)
    assert len(events(mem)) == 1


def test_moving_zone_logs_new_place(mem):
    mem.update(frame(), [KEYS_TABLE], now=1)
    mem.update(frame(), [KEYS_SHELF], now=2)
    assert events(mem)[-1] == ("keys", "placed", "on the right of the camera view", 0)


def test_one_missed_detection_is_not_gone(mem):
    mem.update(frame(), [KEYS_TABLE], now=1)
    assert mem.update(frame(), [], now=2) is True    # asks for a re-check
    mem.update(frame(), [KEYS_TABLE], now=3)
    assert [e[1] for e in events(mem)] == ["placed"]


def test_gone_and_picked_up_when_spot_changes(mem):
    mem.update(frame(200), [KEYS_TABLE], now=1)
    mem.update(frame(50), [], now=2)
    mem.update(frame(50), [], now=3)
    assert events(mem)[-1] == ("keys", "gone", "on the kitchen table", 1)


def test_gone_but_not_picked_up_when_spot_looks_the_same(mem):
    mem.update(frame(), [KEYS_TABLE], now=1)
    mem.update(frame(), [], now=2)
    mem.update(frame(), [], now=3)
    assert events(mem)[-1] == ("keys", "gone", "on the kitchen table", 0)


def test_person_touching_marks_picked_up(mem):
    hand = (90, 90, 200, 200, "person", 0.9)
    mem.update(frame(), [KEYS_TABLE], now=1)
    mem.update(frame(), [hand], now=2)
    mem.update(frame(), [], now=3)
    assert events(mem)[-1][3] == 1


def test_people_are_not_items(mem):
    mem.update(frame(), [(0, 0, 100, 200, "person", 0.9)], now=1)
    assert events(mem) == []


def test_old_snapshots_deleted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "snapshots").mkdir()
    old, new = tmp_path / "snapshots/old.jpg", tmp_path / "snapshots/new.jpg"
    old.write_bytes(b""), new.write_bytes(b"")
    import os, time
    os.utime(old, (time.time() - 25 * 3600,) * 2)
    memory.delete_old_snapshots()
    assert not old.exists() and new.exists()
