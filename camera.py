"""Opens the webcam (or a video file) the fast way on each OS. Shared by detector.py and zone_editor.py."""

import sys

import cv2

NO_CAMERA_HELP = """Could not open camera {source}.
- Try --source 1 (or 2) for another camera.
- Close Zoom, Teams or anything else using the camera.
- Windows: Settings > Privacy & security > Camera > let desktop apps use the camera,
  and check the laptop's camera switch / privacy shutter.
- Mac: allow camera access for Terminal or VS Code in System Settings > Privacy & Security."""


def parse_source(text):
    """'0' -> camera 0, anything else -> a video file path."""
    return int(text) if str(text).isdigit() else text


def open_camera(source=0, width=1280, height=720):
    if not isinstance(source, int):                      # a video file
        cam = cv2.VideoCapture(source)
        if not cam.isOpened():
            raise SystemExit(f"Could not open video file {source}.")
        return cam
    cam = None
    if sys.platform == "win32":
        cam = cv2.VideoCapture(source, cv2.CAP_DSHOW)    # DirectShow opens much faster than MSMF
    if cam is None or not cam.isOpened():
        cam = cv2.VideoCapture(source)                   # fall back to the default backend
    if not cam.isOpened():
        raise SystemExit(NO_CAMERA_HELP.format(source=source))
    # many webcams only manage 5-10 fps in their default raw format; MJPG gets 30
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)                  # always read the newest frame
    for _ in range(10):                                  # first frames are often dark
        cam.read()
    return cam
