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
