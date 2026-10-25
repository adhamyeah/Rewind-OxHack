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


def make_worker(monkeypatch, tmp_path, detect_fn, use_gemini=True, check_fn=lambda f, box, item: "none",
                on_pending=None):
    monkeypatch.setattr(kw.config, "KEYFRAME_DIR", str(tmp_path))
    mem = Memory()
    return mem, kw.KeyframeWorker(mem, store=None, use_gemini=use_gemini, detect_fn=detect_fn,
                                  check_fn=check_fn, on_pending=on_pending)


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
    # YOLO's keys are on the table (outside the person) and Gemini's check says they aren't keys
    yolo = [(500, 400, 550, 450, "keys", 0.4), (0, 0, 300, 400, "person", 0.9)]
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [(1, 1, 9, 9, "mug", 1.0)])
    w.submit(FRAME, yolo)
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert [d[4] for d in mem.keyframes[0].detections] == ["mug"]
    w.stop()


def test_item_in_a_hand_comes_from_yolo_and_is_held(monkeypatch, tmp_path):
    checked = []
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [(1, 1, 9, 9, "mug", 1.0)],
                         check_fn=lambda f, box, item: checked.append(item) or "ok")
    w.submit(FRAME, YOLO)  # YOLO's keys sit in the person's upper body
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert sorted(d[4] for d in mem.keyframes[0].detections) == ["keys", "mug"]
    assert mem.items["keys"].held
    assert checked == []  # held items are never sent to Gemini
    w.stop()


def test_people_are_blurred_before_gemini_sees_the_frame(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    seen = []
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: seen.append(f.copy()) or [])
    w.submit(frame, [(100, 100, 300, 400, "person", 0.9)])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    sent = seen[0]
    assert not np.array_equal(sent[100:400, 100:300], frame[100:400, 100:300])  # person blurred
    assert np.array_equal(sent[:, 320:], frame[:, 320:])                        # the rest untouched
    assert np.array_equal(mem.keyframes[0].image, frame)                         # memory keeps the real frame
    w.stop()


def test_yolo_item_the_scan_missed_is_checked_once_and_remembered(monkeypatch, tmp_path):
    calls = []
    keys = (500, 400, 550, 450, "keys", 0.4)
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [],
                         check_fn=lambda f, box, item: calls.append((item, box)) or "ok")
    w.submit(FRAME, [keys])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert [d[4] for d in mem.keyframes[0].detections] == ["keys"]
    assert calls == [("keys", keys[:4])]
    w.submit(FRAME, [(503, 402, 552, 452, "keys", 0.5)])  # same spot, a slightly different box
    assert wait_for(lambda: len(mem.keyframes) == 2)
    assert [d[4] for d in mem.keyframes[1].detections] == ["keys"]
    assert len(calls) == 1  # verdict reused: no second Gemini call
    w.stop()


def test_vetoed_item_is_ignored_at_that_spot_but_rechecked_when_moved(monkeypatch, tmp_path):
    calls = []
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [],
                         check_fn=lambda f, box, item: calls.append(box) or "none")
    w.submit(FRAME, [(500, 400, 550, 450, "pen", 0.3)])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    w.submit(FRAME, [(500, 400, 550, 450, "pen", 0.3)])
    assert wait_for(lambda: len(mem.keyframes) == 2)
    assert mem.keyframes[1].detections == [] and len(calls) == 1
    w.submit(FRAME, [(100, 400, 150, 450, "pen", 0.3)])  # somewhere else entirely
    assert wait_for(lambda: len(mem.keyframes) == 3)
    assert len(calls) == 2
    w.stop()


def test_yolo_box_on_an_object_gemini_already_named_is_not_checked(monkeypatch, tmp_path):
    calls = []
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [(100, 100, 200, 200, "wallet", 1.0)],
                         check_fn=lambda f, box, item: calls.append(item) or "ok")
    w.submit(FRAME, [(105, 105, 195, 195, "phone", 0.3)])  # YOLO calls Gemini's wallet a phone
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert [d[4] for d in mem.keyframes[0].detections] == ["wallet"]
    assert calls == []
    w.stop()


def test_checks_are_capped_per_keyframe_and_the_rest_ask_for_another(monkeypatch, tmp_path):
    monkeypatch.setattr(kw.config, "MAX_CHECKS_PER_KEYFRAME", 2)
    calls, asked = [], []
    yolo = [(i * 100, 400, i * 100 + 40, 440, name, 0.5) for i, name in enumerate(["keys", "pen", "mug"])]
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [],
                         check_fn=lambda f, box, item: calls.append(item) or "ok",
                         on_pending=lambda: asked.append(1))
    w.submit(FRAME, yolo)
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert len(calls) == 2 and len(mem.keyframes[0].detections) == 2
    assert wait_for(lambda: asked == [1])  # one left: asks for another keyframe
    w.submit(FRAME, yolo)
    assert wait_for(lambda: len(mem.keyframes) == 2)
    assert len(calls) == 3 and len(mem.keyframes[1].detections) == 3
    w.stop()


def test_check_failure_keeps_the_scan_and_stops_checking(monkeypatch, tmp_path):
    calls = []

    def boom(f, box, item):
        calls.append(item)
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [(1, 1, 9, 9, "mug", 1.0)], check_fn=boom)
    w.submit(FRAME, [(500, 400, 550, 450, "keys", 0.4)])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert [d[4] for d in mem.keyframes[0].detections] == ["mug"]  # scan still used
    w.submit(FRAME, [(500, 400, 550, 450, "keys", 0.4)])
    assert wait_for(lambda: len(mem.keyframes) == 2)
    assert wait_for(lambda: w.status == "idle")  # the worker sets this just after adding the keyframe
    assert len(calls) == 1  # checks cooling down; scans carry on
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


def test_gemini_held_verdict_adds_the_item(monkeypatch, tmp_path):
    keys = (500, 400, 550, 450, "keys", 0.4)
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: [], check_fn=lambda f, box, item: "held")
    w.submit(FRAME, [keys])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    assert [d[4] for d in mem.keyframes[0].detections] == ["keys"]  # in a hand still counts as seen
    w.stop()


def test_scan_prompt_no_longer_skips_held_items():
    assert "Skip anything" not in kw.config.GEMINI_PROMPT and "held in a hand" in kw.config.GEMINI_PROMPT


def test_only_the_head_is_blurred_before_gemini(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    seen = []
    mem, w = make_worker(monkeypatch, tmp_path, lambda f: seen.append(f.copy()) or [])
    w.submit(frame, [(100, 100, 300, 400, "person", 0.9)])
    assert wait_for(lambda: len(mem.keyframes) == 1)
    head_end = 100 + int(300 * kw.config.PERSON_BLUR_FRACTION)
    assert not np.array_equal(seen[0][100:head_end, 100:300], frame[100:head_end, 100:300])
    assert np.array_equal(seen[0][head_end:, :], frame[head_end:, :])  # hands and table in front stay visible
    w.stop()
