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
