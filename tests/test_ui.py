from ui import pick_live_boxes, relabel_with_memory

YOLO = [(0, 0, 10, 10, "remote control", 0.6), (50, 0, 90, 90, "person", 0.9)]
MEM = [(0, 0, 10, 10, "phone", 1.0)]


def test_moving_scene_shows_yolo_boxes_with_memory_names():
    boxes = pick_live_boxes(YOLO, MEM, scene_changed=True, memory_busy=False)
    assert sorted(b[4] for b in boxes) == ["person", "phone"]  # "remote control" renamed by memory


def test_memory_busy_also_renames():
    boxes = pick_live_boxes(YOLO, MEM, scene_changed=False, memory_busy=True)
    assert "remote control" not in [b[4] for b in boxes]


def test_no_keyframe_yet_shows_yolo_boxes():
    assert pick_live_boxes(YOLO, None, scene_changed=False, memory_busy=False) == YOLO


def test_settled_scene_shows_gemini_labels_and_keeps_people():
    boxes = pick_live_boxes(YOLO, MEM, scene_changed=False, memory_busy=False)
    assert [b[4] for b in boxes] == ["person", "phone"]


def test_yolo_box_matching_nothing_known_is_dropped_while_moving():
    stray = [(300, 300, 340, 340, "remote control", 0.65)]
    assert relabel_with_memory(stray, MEM) == []


def test_slightly_different_box_still_matches():
    bottle_memory = [(100, 100, 160, 300, "water bottle", 1.0)]
    yolo = [(105, 110, 150, 290, "remote control", 0.5)]   # inside, a bit smaller
    assert relabel_with_memory(yolo, bottle_memory) == [(105, 110, 150, 290, "water bottle", 0.5)]


def test_two_yolo_boxes_on_one_known_item_give_one_box():
    bottle_memory = [(100, 100, 160, 300, "water bottle", 1.0)]
    yolo = [(105, 110, 150, 290, "remote control", 0.3), (100, 100, 160, 300, "mug", 0.6)]
    boxes = relabel_with_memory(yolo, bottle_memory)
    assert boxes == [(100, 100, 160, 300, "water bottle", 0.6)]
