from memory import Memory, describe_surroundings

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
    assert a.text == "The mug is on the table."
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


def test_surroundings_names_nearest_items_within_range():
    box = (100, 100, 140, 140, "x", 1.0)[:4]
    pen = (150, 100, 170, 140, "pen", 1.0)        # very close
    mug = (260, 100, 300, 140, "mug", 1.0)        # ~160 px away, inside 0.3 * 640 = 192
    far = (600, 400, 620, 440, "wallet", 1.0)     # far away
    assert describe_surroundings(box, [mug, pen, far], W) == "next to the pen and the mug"
    assert describe_surroundings(box, [far], W) == "on its own"
    assert describe_surroundings(box, [], W) == "on its own"


def test_surroundings_only_two_names():
    box = (100, 100, 140, 140)
    others = [(150 + i * 10, 100, 160 + i * 10, 140, l, 1.0) for i, l in enumerate(["pen", "mug", "wallet"])]
    assert describe_surroundings(box, others, W) == "next to the pen and the mug"


def test_where_names_neighbours_from_that_keyframe():
    m = Memory(W, H)
    m.add_keyframe([KEYS, (170, 100, 210, 140, "mug", 1.0)])
    assert m.where("keys").text == "The keys are on the table, next to the mug."


def test_where_gone_uses_neighbours_from_when_last_seen():
    m = Memory(W, H)
    m.add_keyframe([KEYS, (170, 100, 210, 140, "mug", 1.0)])
    m.add_keyframe([(170, 100, 210, 140, "mug", 1.0)])
    m.add_keyframe([(170, 100, 210, 140, "mug", 1.0)])
    text = m.where("keys").text
    assert "keys were last seen" in text and "next to the mug" in text and "They've been taken off" in text


def test_duplicate_labels_collapse_to_one():
    m = Memory(W, H)
    _, ev = m.add_keyframe([(0, 0, 10, 10, "pen", 0.3), (200, 200, 220, 220, "pen", 0.9)])
    assert kinds(ev) == [("pen", "appeared")]
    assert m.items["pen"].box == (200, 200, 220, 220)


def test_pending_misses_until_confirmed_gone():
    m = Memory(W, H)
    m.add_keyframe([KEYS])
    assert not m.pending_misses()
    m.add_keyframe([])               # missing once: not gone yet, but a recheck is wanted
    assert m.pending_misses()
    m.add_keyframe([])               # second miss: now it has left
    assert not m.pending_misses()
