import json

import pytest

from zones import load_zones, zone_for


@pytest.fixture
def zones(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"by the kettle": [[0, 0], [200, 0], [200, 200], [0, 200]]}))
    return load_zones(path)


def test_uses_bottom_centre_of_box(zones):
    # box top is outside the zone but the item rests (bottom-centre) inside it
    assert zone_for((50, 150, 150, 190), zones, 600) == "by the kettle"


@pytest.mark.parametrize("x, place", [(250, "in the middle"), (500, "on the right"), (10, "on the left")])
def test_falls_back_to_left_middle_right(zones, x, place):
    assert zone_for((x, 300, x + 20, 350), zones, 600) == f"{place} of the camera view"


def test_missing_file_means_no_zones(tmp_path):
    assert load_zones(tmp_path / "nope.json") == {}
