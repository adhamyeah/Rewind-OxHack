import json
from types import SimpleNamespace

import numpy as np

import gemini_detector as gd
from gemini_detector import gemini_detect, parse_gemini_json


def j(objs):
    return json.dumps(objs)


def test_box_is_y_first_and_scaled_to_pixels():
    # [ymin, xmin, ymax, xmax] = [100, 200, 300, 400] on a 0-1000 grid, 1000x500 image
    dets = parse_gemini_json(j([{"label": "keys", "box_2d": [100, 200, 300, 400]}]), 1000, 500)
    assert dets == [(200, 50, 400, 150, "keys", 1.0)]


def test_clamped_to_0_1000():
    dets = parse_gemini_json(j([{"label": "mug", "box_2d": [-50, -10, 1200, 1500]}]), 1000, 1000)
    assert dets == [(0, 0, 1000, 1000, "mug", 1.0)]


def test_swapped_min_max_are_fixed():
    dets = parse_gemini_json(j([{"label": "pen", "box_2d": [300, 400, 100, 200]}]), 1000, 1000)
    assert dets == [(200, 100, 400, 300, "pen", 1.0)]


def test_wrapped_objects_json():
    text = j({"objects": [{"label": "wallet", "box_2d": [0, 0, 500, 500]}]})
    assert parse_gemini_json(text, 100, 100) == [(0, 0, 50, 50, "wallet", 1.0)]


def test_unknown_labels_dropped_and_synonyms_normalised():
    text = j([{"label": "banana", "box_2d": [0, 0, 10, 10]},
              {"label": "specs", "box_2d": [0, 0, 500, 500]}])
    dets = parse_gemini_json(text, 100, 100)
    assert [d[4] for d in dets] == ["glasses"]


def test_garbage_does_not_raise():
    assert parse_gemini_json("not json", 100, 100) == []
    assert parse_gemini_json(None, 100, 100) == []
    assert parse_gemini_json(j([{"label": "keys"}, {"label": "keys", "box_2d": "x"}]), 100, 100) == []
    assert parse_gemini_json(j([{"label": "keys", "box_2d": [5, 5, 5, 5]}]), 100, 100) == []  # zero area


def test_gemini_detect_uses_client_and_converts(monkeypatch):
    monkeypatch.setattr(gd, "_image_part", lambda jpg: "IMAGE")
    monkeypatch.setattr(gd.config, "GEMINI_MIN_GAP", 0)
    calls = []

    class FakeModels:
        def generate_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(text=j([{"label": "keys", "box_2d": [0, 0, 500, 500]}]))

    client = SimpleNamespace(models=FakeModels())
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    assert gemini_detect(frame, client=client) == [(0, 0, 200, 100, "keys", 1.0)]
    assert calls[0]["contents"][0] == "IMAGE"


def test_json_inside_code_fences_or_chatter():
    text = 'Here you go:\n```json\n[{"label": "keys", "box_2d": [0, 0, 500, 500]}]\n```'
    assert parse_gemini_json(text, 100, 100) == [(0, 0, 50, 50, "keys", 1.0)]


def test_scan_uses_the_scan_model(monkeypatch):
    monkeypatch.setattr(gd, "_image_part", lambda jpg: "IMAGE")
    monkeypatch.setattr(gd.config, "GEMINI_MIN_GAP", 0)
    calls = []

    class FakeModels:
        def generate_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(text="[]")

    gemini_detect(np.zeros((10, 10, 3), dtype=np.uint8), client=SimpleNamespace(models=FakeModels()))
    assert calls[0]["model"] == gd.config.GEMINI_SCAN_MODEL
    assert "held" in calls[0]["contents"][1]  # told to skip held / worn items


def test_read_verdict():
    assert gd.read_verdict("YES") == "ok"
    assert gd.read_verdict("yes.") == "ok"
    assert gd.read_verdict("NONE - there is no pen") == "none"
    assert gd.read_verdict("**HELD**") == "held"
    assert gd.read_verdict("I think maybe") is None
    assert gd.read_verdict(None) is None


def test_check_draws_the_orange_box_and_names_the_item(monkeypatch):
    sent = []
    monkeypatch.setattr(gd, "_image_part", lambda jpg: sent.append(jpg) or "IMAGE")
    monkeypatch.setattr(gd.config, "GEMINI_MIN_GAP", 0)
    calls = []

    class FakeModels:
        def generate_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(text="NONE")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    verdict = gd.gemini_check(frame, (20, 20, 60, 60), "pen", client=SimpleNamespace(models=FakeModels()))
    assert verdict == "none"
    assert calls[0]["model"] == gd.config.GEMINI_CHECK_MODEL
    assert "pen inside the orange box" in calls[0]["contents"][1]
    import cv2
    img = cv2.imdecode(np.frombuffer(sent[0], np.uint8), cv2.IMREAD_COLOR)
    assert img[20, 40, 2] > 200 and img[50, 50].sum() < 30  # orange edge drawn, inside left alone
    assert frame.sum() == 0  # the caller's frame is not drawn on


def test_blur_people_only_blurs_the_head_area():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = gd.blur_people(img.copy(), [(10, 10, 50, 90), (-5, -5, 5, 5)])  # boxes may poke off the edge
    head_end = 10 + int(80 * gd.config.PERSON_BLUR_FRACTION)
    assert not np.array_equal(out[10:head_end, 10:50], img[10:head_end, 10:50])  # head blurred
    assert np.array_equal(out[head_end:, :], img[head_end:, :])                  # hands and table untouched
    assert np.array_equal(out[:, 60:], img[:, 60:])                              # outside the person untouched


def test_scan_asks_for_low_thinking_and_survives_a_model_that_refuses_it(monkeypatch):
    monkeypatch.setattr(gd, "_image_part", lambda jpg: "IMAGE")
    monkeypatch.setattr(gd.config, "GEMINI_MIN_GAP", 0)
    monkeypatch.setattr(gd.config, "GEMINI_THINKING", "low")
    seen = []

    class Models:
        def generate_content(self, **kw):
            seen.append(kw["config"].copy())
            if "thinking_config" in kw["config"] and len(seen) == 1:
                raise RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported")
            return SimpleNamespace(text=j([{"label": "keys", "box_2d": [0, 0, 500, 500]}]))

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert gemini_detect(frame, client=SimpleNamespace(models=Models())) == [(0, 0, 50, 50, "keys", 1.0)]
    assert seen[0]["thinking_config"] == {"thinking_level": "low"}
    assert "thinking_config" not in seen[1]
