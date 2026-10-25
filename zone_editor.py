"""
zone_editor.py - draw named zones on the live camera view.

Put the camera in its final position first (tape it down!), then:

  left-click    add a corner of the area
  right-click   undo the last corner
  Enter         finish the shape, then type its name in this window
                (Backspace deletes a letter, Enter saves the name, Esc cancels)
  d             delete the last zone
  q             save to zones.json and quit
  Esc           quit WITHOUT saving

Name each zone as a phrase that finishes "Your keys are ...", e.g.
"on the kitchen table", "by the kettle", "on the top shelf".
Rewind drops the name straight into its answers.

Run:  python zone_editor.py            (or --source 1 for a second camera)
"""

import argparse
import time

import cv2
import numpy as np

from phase1_camera_gate import Camera
from zones import draw_zones, load_zones, save_zones, to_pixels

WINDOW = "Zone editor"
HELP = "click corners | right-click undo | Enter name it | d delete last | q save+quit | Esc quit"


def main():
    ap = argparse.ArgumentParser(description="Draw Rewind zones")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video/image file")
    args = ap.parse_args()

    cam = Camera(args.source)  # same size and settings as phase1_camera_gate.py
    cam.start()

    zones = load_zones()       # keep editing what's already there
    state = {"points": [], "naming": None, "size": (1, 1)}

    def on_mouse(event, x, y, flags, param):
        if state["naming"] is not None:
            return
        w, h = state["size"]
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append([x / w, y / h])  # stored as fractions of the picture
        elif event == cv2.EVENT_RBUTTONDOWN and state["points"]:
            state["points"].pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"[zones] {len(zones)} existing zone(s) loaded. {HELP}")

    last_frame = None
    while True:
        _, frame = cam.latest()
        if frame is None:
            frame = last_frame
        if frame is None:  # camera still starting
            time.sleep(0.05)
            continue
        last_frame = frame
        h, w = frame.shape[:2]
        state["size"] = (w, h)

        view = draw_zones(frame.copy(), zones, alpha=0.25)
        pts = state["points"]
        if pts:
            px = to_pixels(np.array(pts, dtype=np.float32), w, h)
            cv2.polylines(view, [px], len(pts) >= 3, (0, 0, 255), 2)
            for p in px:
                cv2.circle(view, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

        cv2.rectangle(view, (0, h - 24), (w, h), (0, 0, 0), -1)
        cv2.putText(view, HELP, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        if state["naming"] is not None:
            cv2.rectangle(view, (0, 0), (w, 58), (0, 0, 0), -1)
            cv2.putText(view, "Name this zone, e.g. 'on the kitchen table'  (Enter = save, Esc = cancel)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(view, state["naming"] + "_", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF

        if state["naming"] is not None:  # typing a name
            if key == 13:  # Enter
                name = state["naming"].strip()
                if name:
                    zones[name] = np.array(pts, dtype=np.float32)
                    state["points"] = []
                    print(f"[zones] added '{name}'")
                state["naming"] = None
            elif key == 27:  # Esc
                state["naming"] = None
            elif key == 8:  # Backspace
                state["naming"] = state["naming"][:-1]
            elif 32 <= key <= 126:
                state["naming"] += chr(key)
            continue

        if key == 13 and len(pts) >= 3:
            state["naming"] = ""
        elif key == 13:
            print("[zones] click at least 3 corners first")
        elif key == ord("d") and zones:
            removed = list(zones)[-1]
            del zones[removed]
            print(f"[zones] deleted '{removed}'")
        elif key in (ord("q"), ord("Q")):
            save_zones(zones)
            print(f"[zones] saved {len(zones)} zone(s) to zones.json: {', '.join(zones) or '-'}")
            break
        elif key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            print("[zones] quit without saving")
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
