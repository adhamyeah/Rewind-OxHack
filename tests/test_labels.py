import pytest

from gemini_detector import normalise_label


@pytest.mark.parametrize("text,expected", [
    ("where are my keys?", "keys"),
    ("keys", "keys"),
    ("where are my car keys", "keys"),
    ("have you seen my specs", "glasses"),
    ("my glasses", "glasses"),
    ("where are my airpods", "headphones"),
    ("headphones", "headphones"),
    ("where did I leave my cup", "mug"),
    ("the mug", "mug"),
    ("my phone", "phone"),
    ("where's my mobile", "phone"),
    ("the remote", "remote control"),
    ("remote control", "remote control"),
    ("a pen", "pen"),
    ("WALLET", "wallet"),
])
def test_known_items(text, expected):
    assert normalise_label(text) == expected


@pytest.mark.parametrize("text", ["open the door", "happen", "penguin", "what time is it", "", None])
def test_word_boundaries_and_unrelated(text):
    # "open" contains "pen", "penguin" starts with it: neither is the item.
    assert normalise_label(text) is None
