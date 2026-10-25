"""Camera: a background thread that keeps only the newest frame."""

import platform
import threading
import time

import cv2

import config


def available_cameras(max_index=4):
    """Camera indexes OpenCV can open right now (only used to write a helpful error message)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class Camera:
    def __init__(self, source, width=config.CAMERA_W, height=config.CAMERA_H):
        self.is_file = not str(source).isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            index = int(source)
            # DirectShow opens much faster than the Windows default backend.
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            # MJPG lets most USB webcams do 30 fps (raw YUYV often can't).
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue old frames

        if not self.cap.isOpened():
            self.cap.release()
            if self.is_file:
                raise RuntimeError(f"Could not open the video file {source!r}. Check the path.")
            found = available_cameras()
            raise RuntimeError(
                f"Could not open camera {source!r}. "
                + (f"Cameras that can be opened right now: {found}. Try --source {found[0]}."
                   if found else "No camera can be opened right now.")
                + " Phone cameras need the phone unlocked and nearby; also check camera permission for your terminal.")

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
