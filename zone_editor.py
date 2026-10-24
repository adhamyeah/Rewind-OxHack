"""Draw named zones on a still from the camera and save them to zones.json.

Run:  python zone_editor.py [--source 1]
Click the corners of an area, press N and type its name in the terminal.
U undoes the last point, S saves, Q quits. Tape the camera down first:
if it moves, every zone is wrong.
"""

import argparse
import json

import cv2
import numpy as np

from camera import open_camera
from zones import ZONES_PATH, draw_zones, load_zones

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0)
args = parser.parse_args()

cam = open_camera(args.source)
ok, frame = cam.read()
cam.release()
if not ok:
    raise SystemExit("Camera opened but gave no frame.")

zones = {name: poly.tolist() for name, poly in load_zones().items()}   # edit existing ones
points = []


def click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


cv2.namedWindow("zones")
cv2.setMouseCallback("zones", click)
print(__doc__)
while True:
    view = frame.copy()
    draw_zones(view, {n: np.array(p, dtype=np.int32) for n, p in zones.items()})
    for p in points:
        cv2.circle(view, tuple(p), 5, (0, 0, 255), -1)
    if len(points) > 1:
        cv2.polylines(view, [np.array(points, dtype=np.int32)], False, (0, 0, 255), 1)
    cv2.imshow("zones", view)
    key = cv2.waitKey(20) & 0xFF
    if key in (ord("n"), ord("N")) and len(points) >= 3:
        name = input('Zone name (finishes "Your keys are..."): ').strip()
        if name:
            zones[name] = points
        points = []
    elif key in (ord("u"), ord("U")) and points:
        points.pop()
    elif key in (ord("s"), ord("S")):
        with open(ZONES_PATH, "w") as f:
            json.dump(zones, f, indent=2)
        print(f"saved {len(zones)} zones to {ZONES_PATH}")
    elif key in (ord("q"), ord("Q")) or cv2.getWindowProperty("zones", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
