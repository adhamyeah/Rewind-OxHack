import numpy as np

import ui
from memory import Keyframe, jpeg
from ui import UIState, live_boxes, replay_gif

KNOWN = [(0, 0, 10, 10, "phone", 1.0)]


def test_yolo_box_on_a_known_item_takes_memorys_name():
    boxes = live_boxes([(0, 0, 10, 10, "mug", 0.6), (50, 0, 90, 90, "person", 0.9)], KNOWN)
    assert sorted(b[4] for b in boxes) == ["person", "phone"]


def test_new_items_still_show_before_memory_confirms_them():
    boxes = live_boxes([(300, 300, 340, 340, "keys", 0.4)], KNOWN)
    assert [b[4] for b in boxes] == ["keys"]


def test_gemini_only_items_are_drawn_from_memory():
    bottle = (100, 100, 160, 300, "water bottle", 1.0)
    assert live_boxes([], KNOWN, [bottle]) == [bottle]


def test_boxes_gemini_ruled_out_are_hidden():
    stray = [(300, 300, 340, 340, "pen", 0.3)]
    assert live_boxes(stray, [], [], ruled_out=[(300, 300, 340, 340, "pen")]) == []


def test_two_yolo_boxes_on_one_known_item_give_one_box():
    memory_box = [(100, 100, 160, 300, "water bottle", 1.0)]
    yolo = [(105, 110, 150, 290, "mug", 0.3), (100, 100, 160, 300, "phone", 0.6)]
    assert live_boxes(yolo, memory_box) == [(100, 100, 160, 300, "water bottle", 0.6)]


def moment(n_clip=6):
    img = np.full((480, 640, 3), 80, np.uint8)
    clip = [jpeg(np.full((240, 320, 3), 30 + 25 * i, np.uint8)) for i in range(n_clip)]  # each frame different
    return Keyframe(0, 0.0, still=jpeg(img), clip=clip,
                    items=[(100, 100, 160, 160, "keys", 1.0)], people=[])


def test_replay_gif_plays_the_clip_then_holds_the_moment():
    from PIL import Image
    import io

    gif = replay_gif(moment(6), "keys", None, "put down")
    assert gif[:6] in (b"GIF87a", b"GIF89a")
    im = Image.open(io.BytesIO(gif))
    assert im.n_frames == 7 and im.size == ui.GIF_SIZE


def test_gif_for_a_moment_without_a_clip_is_just_the_picture():
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(replay_gif(moment(0)))).n_frames == 1


def test_jump_is_handed_over_once():
    s = UIState()
    s.request_jump({"keyframe": 2, "box": [1, 2, 3, 4], "label": "keys", "kind": "put down"}, "Your keys are...")
    assert s.take_web_jump()["caption"] == "Your keys are..."
    assert s.take_web_jump() is None


def test_talk_button_pauses_the_always_on_mic():
    s = UIState()
    s.talk(True)
    assert s.talking()
    s.talk(False)
    assert not s.talking()
