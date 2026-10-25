from ui import pick_live_boxes

YOLO = [(0, 0, 10, 10, "remote control", 0.6), (50, 0, 90, 90, "person", 0.9)]
MEM = [(0, 0, 10, 10, "phone", 1.0)]


def test_moving_scene_shows_yolo_boxes():
    assert pick_live_boxes(YOLO, MEM, scene_changed=True, memory_busy=False) == YOLO


def test_memory_busy_shows_yolo_boxes():
    assert pick_live_boxes(YOLO, MEM, scene_changed=False, memory_busy=True) == YOLO


def test_no_keyframe_yet_shows_yolo_boxes():
    assert pick_live_boxes(YOLO, None, scene_changed=False, memory_busy=False) == YOLO


def test_settled_scene_shows_gemini_labels_and_keeps_people():
    boxes = pick_live_boxes(YOLO, MEM, scene_changed=False, memory_busy=False)
    assert [b[4] for b in boxes] == ["person", "phone"]


# --- open vocabulary: Gemini's boxes stay on screen even while things move ---
from types import SimpleNamespace  # noqa: E402

from ui import pick_live_boxes_open  # noqa: E402


def test_open_vocab_shows_recent_gemini_boxes_plus_people_and_other_yolo_items():
    kf = SimpleNamespace(time=100.0, detections=[(0, 0, 10, 10, "phone", 1.0), (20, 20, 30, 30, "remote control", 1.0)])
    yolo = YOLO + [(5, 5, 9, 9, "phone", 0.3), (60, 60, 70, 70, "mug", 0.4)]
    boxes = pick_live_boxes_open(yolo, kf, now=104.0)
    # people and things Gemini's last look didn't include come from YOLO; the rest from Gemini
    assert [b[4] for b in boxes] == ["person", "mug", "phone", "remote control"]


def test_open_vocab_drops_old_gemini_boxes():
    kf = SimpleNamespace(time=100.0, detections=MEM)
    assert pick_live_boxes_open(YOLO, kf, now=200.0) == YOLO


def test_open_vocab_before_first_keyframe_shows_yolo():
    assert pick_live_boxes_open(YOLO, None, now=0.0) == YOLO
