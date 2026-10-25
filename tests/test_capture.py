import config
from live_detector import iou, one_label_per_object


def test_yolo_does_not_look_for_remote_controls_but_gemini_does():
    assert "remote control" not in config.YOLO_ITEMS
    assert "remote control" in config.ITEMS
    assert "remote control" in config.GEMINI_PROMPT
    assert set(config.YOLO_ITEMS) < set(config.ITEMS)


def test_two_labels_on_one_box_keep_the_stronger():
    dets = [(0, 0, 100, 100, "wallet", 0.3), (2, 2, 101, 99, "phone", 0.6), (300, 300, 340, 340, "pen", 0.2)]
    assert sorted(d[4] for d in one_label_per_object(dets)) == ["pen", "phone"]


def test_separate_objects_are_all_kept():
    dets = [(0, 0, 100, 100, "wallet", 0.3), (60, 0, 160, 100, "phone", 0.6)]  # overlapping, but not the same box
    assert len(one_label_per_object(dets)) == 2


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_keyframe_fires_when_still_or_after_a_long_unsettled_spell():
    from live_detector import should_fire_keyframe
    import config

    assert not should_fire_keyframe(False, 99, 99)                         # nothing changed: never
    assert not should_fire_keyframe(True, 0.2, 1.0)                        # moving, only just started
    assert should_fire_keyframe(True, config.SETTLE_SECONDS, 1.0)          # went still
    assert should_fire_keyframe(True, 0.1, config.MAX_MOTION_WAIT)         # never still (moving face): fire anyway


def test_yolo_does_not_guess_wallets_but_gemini_still_looks_for_them():
    assert "wallet" not in config.YOLO_ITEMS
    assert "wallet" in config.ITEMS and "wallet" in config.GEMINI_PROMPT
