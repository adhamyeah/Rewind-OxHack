from memory import Memory, describe_position

W, H = 640, 480
KEYS = (100, 100, 160, 140, "keys", 1.0)
MUG = (400, 300, 460, 360, "mug", 1.0)


def kinds(events):
    return [(e.label, e.kind) for e in events]


def test_item_appears():
    m = Memory(W, H)
    _, ev = m.add_keyframe([KEYS])
    assert kinds(ev) == [("keys", "appeared")]


def test_one_missed_keyframe_is_not_left():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    _, ev = m.add_keyframe([])
    assert ev == [] and m.items["keys"].present


def test_two_missed_keyframes_is_left():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    m.add_keyframe([])
    _, ev = m.add_keyframe([])
    assert kinds(ev) == [("keys", "left")]
    assert not m.items["keys"].present


def test_miss_counter_resets_when_seen_again():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    m.add_keyframe([])
    m.add_keyframe([KEYS])
    _, ev = m.add_keyframe([])
    assert ev == []


def test_moved_over_threshold():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    far = (300, 100, 360, 140, "keys", 1.0)
    _, ev = m.add_keyframe([far])
    assert kinds(ev) == [("keys", "moved")]


def test_small_jitter_is_not_moved():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    jitter = (104, 102, 164, 142, "keys", 1.0)
    _, ev = m.add_keyframe([jitter])
    assert ev == []


def test_reappearing_after_left_is_appeared():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    m.add_keyframe([])
    m.add_keyframe([])
    _, ev = m.add_keyframe([KEYS])
    assert kinds(ev) == [("keys", "appeared")]


def test_where_present():
    m = Memory(W, H)
    m.add_keyframe([MUG])
    a = m.where("mug")
    assert "on the table" in a.text and "bottom right" in a.text
    assert a.keyframe == 0 and a.box == MUG[:4]


def test_where_gone():
    t = [1000.0]
    m = Memory(W, H, clock=lambda: t[0])
    m.add_keyframe([KEYS])
    m.add_keyframe([])
    m.add_keyframe([])
    t[0] = 1000.0 + 120
    a = m.where("keys")
    assert "last seen 2 minutes ago" in a.text and "taken off the table" in a.text
    assert a.keyframe == 0


def test_where_never_seen():
    a = Memory(W, H).where("wallet")
    assert "haven't seen" in a.text and a.keyframe is None


def test_plural_grammar():
    m = Memory(W, H)
    m.add_keyframe([KEYS, (10, 10, 50, 50, "wallet", 1.0)])
    assert "The keys are" in m.where("keys").text
    assert "The wallet is" in m.where("wallet").text
    m.add_keyframe([])
    m.add_keyframe([])
    assert "keys were last seen" in m.where("keys").text
    assert "wallet was last seen" in m.where("wallet").text


def test_describe_position_thirds():
    assert describe_position((0, 0, 20, 20), W, H) == "top left"
    assert describe_position((310, 0, 330, 20), W, H) == "top"
    assert describe_position((620, 0, 640, 20), W, H) == "top right"
    assert describe_position((0, 230, 20, 250), W, H) == "left"
    assert describe_position((310, 230, 330, 250), W, H) == "middle"
    assert describe_position((620, 230, 640, 250), W, H) == "right"
    assert describe_position((0, 460, 20, 480), W, H) == "bottom left"
    assert describe_position((310, 460, 330, 480), W, H) == "bottom"
    assert describe_position((620, 460, 640, 480), W, H) == "bottom right"


def test_duplicate_labels_collapse_to_one():
    m = Memory(W, H)
    _, ev = m.add_keyframe([(0, 0, 10, 10, "pen", 0.3), (200, 200, 220, 220, "pen", 0.9)])
    assert kinds(ev) == [("pen", "appeared")]
    assert m.items["pen"].box == (200, 200, 220, 220)
