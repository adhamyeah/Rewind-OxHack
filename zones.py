"""
zones.py - named areas of the camera view ("on the kitchen table").

Zones are saved in zones.json as fractions of the picture (0.0-1.0), not
pixels, so they still line up if the camera resolution changes. They do NOT
survive the camera itself moving: tape it down, and redraw the zones with
zone_editor.py if it gets bumped.
"""

import json
import os

import cv2
import numpy as np

ZONES_FILE = "zones.json"


def load_zones(path=ZONES_FILE):
    """{zone name: polygon as fractions}. Empty if no zones have been drawn yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {name: np.array(points, dtype=np.float32) for name, points in data.get("zones", {}).items()}


def save_zones(zones, path=ZONES_FILE):
    data = {"zones": {name: np.asarray(poly, dtype=float).round(4).tolist() for name, poly in zones.items()}}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def zone_for(box, zones, frame_w, frame_h):
    """Name of the zone an item is in, or None.

    Uses the bottom-centre of the box: the point where the item rests on the
    surface. With a tilted camera that's more accurate than the box's middle.
    """
    x1, y1, x2, y2 = box
    point = ((x1 + x2) / 2 / frame_w, y2 / frame_h)
    for name, poly in zones.items():
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            return name
    return None


def to_pixels(poly, frame_w, frame_h):
    return (np.asarray(poly) * [frame_w, frame_h]).astype(np.int32)


def draw_zones(frame, zones, alpha=0.15):
    """Light tinted areas with their names, so you can see where zones are."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for poly in zones.values():
        cv2.fillPoly(overlay, [to_pixels(poly, w, h)], (255, 200, 0))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for name, poly in zones.items():
        pts = to_pixels(poly, w, h)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 1)
        x, y = (int(v) for v in pts.min(axis=0))
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y), (x + tw + 8, y + th + 8), (0, 0, 0), -1)
        cv2.putText(frame, name, (x + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 120), 1)
    return frame
