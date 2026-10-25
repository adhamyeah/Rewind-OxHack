import json
from types import SimpleNamespace

import numpy as np
import pytest

import gemini_detector as gd
from answer import Answerer
from gemini_detector import gemini_detect, gemini_pick_label, parse_gemini_json
from memory import Memory
from ui import UIState
from vocab import Vocabulary, tidy


@pytest.fixture
def v():
    return Vocabulary(max_learned=3)


@pytest.mark.parametrize("label,expected", [
    ("coffee cup", "mug"),                 # synonym of a starting item
    ("Blue Mug.", "mug"),                  # "<colour> <name>" of a known name
    ("car keys", "keys"),
    ("the TV remote", "remote control"),
    ("A Phone Charger", "phone charger"),  # something new: tidied, kept
    ("stapler", "stapler"),
    ("pencil case", "pencil case"),        # not "pen": only whole words at the end count
    ("person", None),                      # people and the room are never items
    ("curly hair", None),                  # nor body parts
    ("object", None),                      # nor names too vague to mean anything
    ("black puffer jacket", "black puffer jacket"),   # clothing is an object too
    ("wooden table", "wooden table"),    # furniture is named and tracked
    ("white wall", None),                # the room itself is not
    ("", None),
    (None, None),
])
def test_canonical(v, label, expected):
    assert v.canonical(label) == expected


def test_tidy():
    assert tidy("  My  Phone-Charger!! ") == "phone-charger"


def test_learned_names_become_canonical_and_findable(v):
    assert v.find_in_question("where's my stapler?") is None
    assert v.names() == []                                  # nothing assumed before Gemini looks
    assert v.learn(["stapler", "mug"]) == ["stapler", "mug"]  # every predicted name is learned
    assert v.canonical("red stapler") == "stapler"
    assert v.find_in_question("where's my red stapler?") == "stapler"
    assert v.find_in_question("where did the staplers go") == "stapler"


def test_learning_bumps_version_only_on_change(v):
    v.learn(["stapler"])
    first = v.version
    v.learn(["stapler"])
    assert v.version == first
    v.learn(["charger"])
    assert v.version == first + 1


def test_only_most_recent_learned_names_are_kept(v):
    v.learn(["a1", "a2", "a3"])
    v.learn(["a1"])                       # seen again: most recent
    v.learn(["a4"])
    assert v.learned == ["a3", "a1", "a4"]
    assert v.names() == ["a3", "a1", "a4"]     # YOLO-World only gets predicted names


def test_parse_open_vocab_keeps_new_things_and_drops_people(v):
    text = json.dumps([{"label": "banana", "box_2d": [0, 0, 500, 500]},
                       {"label": "specs", "box_2d": [0, 0, 500, 500]},
                       {"label": "person", "box_2d": [0, 0, 900, 900]}])
    assert [d[4] for d in parse_gemini_json(text, 100, 100, v)] == ["banana", "glasses"]


def fake_client(reply):
    calls = []

    class Models:
        def generate_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(text=reply)

    return SimpleNamespace(models=Models()), calls


def test_gemini_detect_learns_new_names_from_an_open_prompt(monkeypatch, v):
    monkeypatch.setattr(gd, "_image_part", lambda jpg: "IMAGE")
    monkeypatch.setattr(gd.config, "GEMINI_MIN_GAP", 0)
    client, calls = fake_client(json.dumps([{"label": "Stapler", "box_2d": [0, 0, 500, 500]}]))
    dets = gemini_detect(np.zeros((100, 100, 3), np.uint8), client=client, vocab=v)
    assert [d[4] for d in dets] == ["stapler"]
    assert "stapler" in v.learned
    prompt = calls[0]["contents"][1]
    assert "EVERY object" in prompt
    assert calls[0]["config"]["response_schema"]   # reply shape is enforced
    assert "keys" not in prompt        # no list of names to squeeze things into


def test_pick_label_names_unseen_things(v):
    client, _ = fake_client("Stapler")
    assert gemini_pick_label("have you seen the thing I staple paper with?", client=client, vocab=v) == "stapler"
    client, _ = fake_client("NONE")
    assert gemini_pick_label("what time is it?", client=client, vocab=v) is None


def test_answer_for_a_learned_thing_and_an_unseen_one(v):
    v.learn(["stapler"])
    m = Memory(640, 480)
    m.add_keyframe([(100, 100, 140, 140, "stapler", 1.0)])
    spoken = []
    a = Answerer(m, UIState(), speak=spoken.append, pick_label=lambda q: "charger", vocab=v)
    assert "stapler" in a.ask("where is my stapler?")
    assert a.ask("where's my charger") == "I haven't seen the charger yet."


def test_quota_on_one_model_switches_to_the_next(monkeypatch):
    monkeypatch.setattr(gd.config, "GEMINI_MODELS", ["model-a", "model-b"])
    monkeypatch.setattr(gd, "_used_up", set())
    tried = []

    class Models:
        def generate_content(self, model, **kw):
            tried.append(model)
            if model == "model-a":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(text="ok")

    client = SimpleNamespace(models=Models())
    assert gd._generate(client, contents="x").text == "ok"
    assert gd._generate(client, contents="x").text == "ok"
    assert tried == ["model-a", "model-b", "model-b"]   # a used-up model isn't tried again


def test_all_models_used_up_raises_a_quota_error(monkeypatch):
    monkeypatch.setattr(gd.config, "GEMINI_MODELS", ["model-a"])
    monkeypatch.setattr(gd, "_used_up", set())

    class Models:
        def generate_content(self, model, **kw):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError) as err:
        gd._generate(SimpleNamespace(models=Models()), contents="x")
    assert gd.is_quota_error(err.value)


def test_other_errors_are_not_treated_as_quota(monkeypatch):
    monkeypatch.setattr(gd.config, "GEMINI_MODELS", ["model-a", "model-b"])
    monkeypatch.setattr(gd, "_used_up", set())

    class Models:
        def generate_content(self, model, **kw):
            raise RuntimeError("503 server busy")

    with pytest.raises(RuntimeError, match="503"):
        gd._generate(SimpleNamespace(models=Models()), contents="x")
    assert gd._used_up == set()                          # a busy server doesn't burn a model
