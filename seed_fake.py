"""Fills rewind.db with fake events so the server and UI work without a camera.

Run:  python seed_fake.py
"""

import os
import time

import cv2
import numpy as np

import db

FAKE = [  # item, event, zone, minutes ago, picked_up
    ("keys", "placed", "on the kitchen table", 2, False),
    ("glasses", "placed", "next to the fruit bowl", 25, False),
    ("wallet", "placed", "on the top shelf", 140, False),
    ("phone", "gone", "on the kitchen table", 7, True),
]


def fake_snapshot(name, label):
    os.makedirs("snapshots", exist_ok=True)
    img = np.full((360, 480, 3), (215, 230, 240), np.uint8)
    cv2.rectangle(img, (170, 140), (310, 240), (0, 200, 255), 4)
    cv2.putText(img, label, (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    path = f"snapshots/fake_{name}.jpg"
    cv2.imwrite(path, img)
    return path


con = db.open_db()
now = time.time()
for item, event, zone, mins, picked_up in FAKE:
    at = now - mins * 60
    if event == "gone":      # a gone item still needs an earlier placed photo
        db.log_event(con, item, "placed", zone, (170, 140, 310, 240), at - 600,
                     fake_snapshot(item, f"{item} {zone}"))
        db.log_event(con, item, "gone", zone, (170, 140, 310, 240), at, None, picked_up)
    else:
        db.log_event(con, item, "placed", zone, (170, 140, 310, 240), at,
                     fake_snapshot(item, f"{item} {zone}"))
print(f"seeded {len(FAKE)} items into {db.DB_PATH}")
