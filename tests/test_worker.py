import time

import numpy as np

import keyframe_worker as kw
from memory import Memory

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
YOLO = [(10, 10, 60, 60, "keys", 0.4), (0, 0, 300, 400, "person", 0.9)]


def wait_for(cond, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_worker(monkeypatch, tmp_path, detect_fn, use_gemini=True):
    monkeypatch.setattr(kw.config, "KEYFRAME_DIR", str(tmp_path))
    mem = Memory()
    return mem, kw.KeyframeWorker(mem, store=None, use_gemini=use_gemini, detect_fn=detect_fn)


def test_gemini_error_falls_back_to_yolo_without_people(monkeypatch, tmp_path):
    def boom(frame):
        raise RuntimeError("quota")

    mem, w = make_worker(monkeypatch, tmp_path, boom)
    w.submit(FRAME, YOLO)
    assert wait_for(lambda: len(mem.keyframes) == 1)
    labels = [d[4] for d in mem.keyframes[0].detections]
    assert labels == ["keys"]  # person filtered out
    w.stop()


def test_gemini_result_used_when_it_works(monkeypatch, tmp_path):
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [(1, 1, 9, 9, "mug", 1.0)])
    w.submit(FRAME, YOLO)
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert [d[4] for d in mem.keyframes[0].detections] == ["mug"]
    w.stop()


def test_queue_keeps_only_newest_job(monkeypatch, tmp_path):
    seen = []

    def slow(frame):
        seen.append(int(frame[0, 0, 0]))
        time.sleep(0.3)
        return []

    mem, w = make_worker(monkeypatch, tmp_path, slow)
    for value in (1, 2, 3, 4):
        f = FRAME.copy()
        f[0, 0, 0] = value
        w.submit(f, [])
        time.sleep(0.05)
    assert wait_for(lambda: len(mem.keyframes) >= 2, timeout=5)
    # job 1 started immediately; 2 and 3 were replaced while it ran; only 4 followed
    assert seen == [1, 4]
    w.stop()


def test_gemini_failure_triggers_cooldown_so_it_is_not_retried(monkeypatch, tmp_path):
    calls = []

    def boom(frame):
        calls.append(1)
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    mem, w = make_worker(monkeypatch, tmp_path, boom)
    w.submit(FRAME, YOLO)
    assert wait_for(lambda: len(mem.keyframes) == 1)
    w.submit(FRAME, YOLO)
    assert wait_for(lambda: len(mem.keyframes) == 2)
    assert len(calls) == 1  # second keyframe skipped Gemini during the cooldown
    w.stop()


def test_missing_item_asks_for_a_confirming_keyframe(monkeypatch, tmp_path):
    asked = []
    results = [[(1, 1, 9, 9, "mug", 1.0)], [], []]
    monkeypatch.setattr(kw.config, "KEYFRAME_DIR", str(tmp_path))
    mem = Memory()
    w = kw.KeyframeWorker(mem, use_gemini=True, detect_fn=lambda f: results.pop(0),
                          on_pending=lambda: asked.append(1))
    w.submit(FRAME, [])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert asked == []                       # mug seen: nothing to confirm
    w.submit(FRAME, [])
    assert wait_for(lambda: len(mem.keyframes) == 2)
    assert wait_for(lambda: asked == [1])    # mug missing once: asks for another keyframe
    w.submit(FRAME, [])
    assert wait_for(lambda: len(mem.keyframes) == 3)
    time.sleep(0.2)
    assert asked == [1]                      # now confirmed gone: no more requests
    w.stop()


def test_person_boxes_are_passed_to_memory_for_held_detection(monkeypatch, tmp_path):
    glasses = (280, 100, 360, 130, "glasses", 1.0)
    person = (200, 50, 440, 480, "person", 0.9)
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [glasses])
    w.submit(FRAME, [person])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert mem.keyframes[0].people == [(200, 50, 440, 480)]
    assert mem.items["glasses"].held
    w.stop()
