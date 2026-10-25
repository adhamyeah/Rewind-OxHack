"""Camera: a background thread that keeps only the newest frame."""

import platform
import threading
import time

import cv2

import config


BLACK = 3.0  # average brightness (0-255) below this: the camera is giving us nothing


def _open(index, backend, width, height):
    cap = cv2.VideoCapture(index, backend)
    # MJPG lets most USB webcams do 30 fps (raw YUYV often can't).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue old frames
    return cap


def _gives_picture(cap, tries=20):
    """True once a real (not all-black) frame arrives. The first frames are often dark."""
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > BLACK:
            return True
    return False


class Camera:
    def __init__(self, source, width=config.CAMERA_W, height=config.CAMERA_H):
        self.is_file = not str(source).isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video: {source!r}")
        else:
            index = int(source)
            # DirectShow opens much faster than the Windows default backend; Media Foundation
            # is the fallback. When another program holds the camera, Windows often "opens" it
            # anyway and hands out black frames, so we check for a real picture, not just open().
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if platform.system() == "Windows" else [cv2.CAP_ANY]
            self.cap = None
            for backend in backends:
                cap = _open(index, backend, width, height)
                if cap.isOpened() and _gives_picture(cap):
                    self.cap = cap
                    break
                cap.release()
            if self.cap is None:
                raise RuntimeError(
                    f"Camera {index} gives no picture (only black frames).\n"
                    "  - Is Rewind (or Zoom/Teams/the Camera app) already running? Close it: Windows\n"
                    "    lets only one program use the camera at a time.\n"
                    "  - Is the lens covered (privacy shutter or camera key)?\n"
                    "  - Another camera? Try --source 1")

        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.ended = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:  # video finished
                    self.ended = True
                    break
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
            if self.is_file:  # play files at real speed, not as fast as possible
                time.sleep(1.0 / self.file_fps)

    def latest(self):
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.cap.release()
