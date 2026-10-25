import sqlite3

from conftest import ITEMS, frame, run
from memory import Memory, next_to

KEYS = (100, 300, 160, 340, "keys", 0.5)
MUG = (190, 290, 250, 350, "mug", 0.6)
PERSON = (60, 100, 400, 479, "person", 0.9)
KEYS_IN_HAND = (150, 250, 200, 290, "keys", 0.5)   # inside the person's box


def kinds(mem):
    return [(e["item"], e["kind"]) for e in mem.events]


def test_put_down_after_staying_still(mem, clock):
    run(mem, clock, [KEYS], 1.0)
    assert kinds(mem) == []                       # not still long enough yet
    run(mem, clock, [KEYS], 1.5)
    assert kinds(mem) == [("keys", "put down")]
    text, jump = mem.explain("keys")
    assert text.startswith("Your keys are in front of the camera") and "right now" in text
    assert jump["keyframe"] == 0 and jump["kind"] == "put down" and jump["box"] == list(KEYS[:4])


def test_each_moment_has_a_picture_and_a_gif(mem, clock):
    run(mem, clock, [KEYS], 2.5)
    kf = mem.keyframes[0]
    assert kf.still[:2] == b"\xff\xd8"            # a JPEG
    assert 5 <= len(kf.clip) <= 20                # the seconds before the put-down
    assert [d[4] for d in kf.items] == ["keys"]


def test_settling_right_after_a_put_down_refreshes_that_moment(mem, clock):
    run(mem, clock, [KEYS], 2.5)
    clips = len(mem.keyframes[0].clip)
    run(mem, clock, [KEYS], 1.0)
    mem.settled(frame(), [KEYS])                  # within MOMENT_GAP: same moment, longer gif
    assert len(mem.keyframes) == 1 and len(mem.keyframes[0].clip) > clips
    clock.t += 10
    mem.settled(frame(), [KEYS])                  # much later: a new moment
    assert len(mem.keyframes) == 2


def test_held_item_is_with_you_and_not_put_down(mem, clock):
    run(mem, clock, [PERSON, KEYS_IN_HAND], 3)
    assert kinds(mem) == []
    text, jump = mem.explain("keys")
    assert text == "It looks like you have your keys with you right now." and jump is None


def test_picked_up_and_carried_away(mem, clock):
    run(mem, clock, [KEYS], 2.5)
    run(mem, clock, [PERSON, KEYS_IN_HAND], 1)
    run(mem, clock, [], 6)
    assert kinds(mem) == [("keys", "put down"), ("keys", "picked up")]
    clock.t += 120
    text, jump = mem.explain("keys")
    assert text.startswith("2 minutes ago, your keys were picked up") and "You might have them with you" in text
    assert jump["kind"] == "picked up" and jump["box"] == list(KEYS[:4])


def test_put_back_in_the_same_place(mem, clock):
    run(mem, clock, [KEYS], 2.5)
    run(mem, clock, [PERSON, KEYS_IN_HAND], 1)
    run(mem, clock, [KEYS], 2.5)
    assert kinds(mem)[-1] == ("keys", "put back")
    assert "right now" in mem.answer("keys")


def test_neighbours_in_the_answer_when_gemini_is_off(mem, clock):
    run(mem, clock, [KEYS, MUG], 2.5)
    assert mem.answer("keys").startswith("Your keys are in front of the camera, next to the mug.")


def test_next_to_wording():
    box = (100, 100, 140, 140)
    pen, mug, far = (150, 100, 170, 140, "pen"), (260, 100, 300, 140, "mug"), (600, 400, 620, 440, "wallet")
    assert next_to(box, [mug, pen, far], 640) == "next to the pen and the mug"
    assert next_to(box, [far], 640) is None


def test_gemini_veto_removes_the_put_down(mem, clock):
    run(mem, clock, [KEYS], 2.5)
    row = mem.items["keys"]["row"]
    sqlite3.connect(mem.db_path).execute("UPDATE events SET event='rejected' WHERE id=?", (row,)).connection.commit()
    mem._gemini_result(row, "keys", "none", None)
    run(mem, clock, [], 0.2)
    assert kinds(mem) == [] and not mem.items["keys"]["present"]
    assert mem.keyframes[0].items == []           # the moment no longer shows the keys


def test_every_run_starts_fresh(tmp_path, monkeypatch, clock):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "rewind.db")
    old = Memory(db_path=db, use_gemini=False, items=ITEMS)
    run(old, clock, [KEYS], 2.5)
    old.db.close()
    clock.t += 3600
    new = Memory(db_path=db, use_gemini=False, items=ITEMS)
    assert new.answer("keys") == "Sorry, I haven't seen your keys since I started watching."
    assert new.keyframes == [] and new.events == []


def test_page_state(mem, clock):
    run(mem, clock, [KEYS, MUG], 2.5)
    state = mem.page_state()
    assert state["present"] == ["keys", "mug"]
    assert state["keyframes"][0]["labels"] == ["keys", "mug"] and state["keyframes"][0]["since"] > 0
    assert {e["kind"] for e in state["events"]} == {"put down"}
