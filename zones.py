"""Named areas of the camera view, so Rewind can say where in words.

Name each zone as a phrase that finishes "Your keys are...", e.g. "next to the fruit bowl".
"""

import json
import os

import cv2
import numpy as np

ZONES_PATH = "zones.json"


def load_zones(path=ZONES_PATH):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {name: np.array(pts, dtype=np.int32) for name, pts in json.load(f).items()}


def zone_for(box, zones, frame_width):
    x1, y1, x2, y2 = box
    point = ((x1 + x2) / 2, y2)              # bottom-centre: where the item rests
    for name, poly in zones.items():
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            return name
    # no zone drawn there: fall back to left / middle / right of the view
    third = point[0] / frame_width
    if third < 1 / 3:
        return "on the left of the camera view"
    if third < 2 / 3:
        return "in the middle of the camera view"
    return "on the right of the camera view"


def draw_zones(img, zones):
    for name, poly in zones.items():
        cv2.polylines(img, [poly], True, (0, 255, 0), 2)
        cv2.putText(img, name, tuple(int(v) for v in poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
