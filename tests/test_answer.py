from answer import Answerer, strip_wake_word
from conftest import ITEMS, run
from listen import find_item
from ui import UIState

KEYS = (100, 300, 160, 340, "keys", 0.5)


class Voice:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)

    def busy(self):
        return False


def make(mem):
    voice, state = Voice(), UIState()
    return Answerer(mem, voice, state, ITEMS), voice, state


def test_typed_question_answers_speaks_and_jumps(mem, clock):
    run(mem, clock, [KEYS], 2.5)
    a, voice, state = make(mem)
    reply = a.ask("where are my keys?")
    assert reply["text"].startswith("Your keys are") and voice.said == [reply["text"]]
    assert state.take_web_jump()["keyframe"] == 0
    assert state.answer["text"] == reply["text"]


def test_unknown_item(mem):
    a, voice, _ = make(mem)
    assert a.ask("where is my umbrella")["text"].startswith("Sorry, I can only look for your keys")


def test_room_chatter_is_ignored_by_the_always_on_mic(mem):
    a, voice, _ = make(mem)
    a.heard("I put the kettle on earlier")
    a.heard("lovely weather today")
    assert voice.said == []
    a.heard("glasses?")
    assert len(voice.said) == 1 and "glasses" in voice.said[0]


def test_wake_word_and_synonyms():
    assert strip_wake_word("Rewind, where are my specs?") == "where are my specs?"
    assert strip_wake_word("rewind my wallet") == "rewind my wallet"
    assert find_item("where are my specs", ITEMS) == "glasses"
    assert find_item("where's my bottle", ITEMS) == "water bottle"
    assert find_item("open the door", ITEMS) is None          # "open" is not a pen
