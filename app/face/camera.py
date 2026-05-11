import threading
import time

import cv2
import numpy as np

from app.face.face_tools import FrameBuffer

CAM_W = 640
CAM_H = 480
FALLBACK_W = 320
FALLBACK_H = 240


def _open_picamera2(w, h):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (w, h), "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    frame = cam.capture_array()
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError("invalid frame from picamera2")
    return cam


def _open_cv2_cam(w, h):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera device found at index 0.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Camera:
    def __init__(self, w=CAM_W, h=CAM_H):
        self._w = w
        self._h = h
        self._running = True
        self._error = None
        self._cam = None
        self._kind = None
        self._init_camera()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _init_camera(self):
        try:
            self._cam = _open_picamera2(self._w, self._h)
            self._kind = "picamera2"
        except Exception:
            try:
                self._cam = _open_cv2_cam(self._w, self._h)
                self._kind = "cv2"
            except Exception as e:
                self._error = str(e)

    def _loop(self):
        fb = FrameBuffer()
        while self._running:
            try:
                if self._kind == "picamera2":
                    frame = self._cam.capture_array()
                elif self._kind == "cv2":
                    self._cam.grab()
                    ok, frame = self._cam.retrieve()
                    if not ok:
                        continue
                else:
                    time.sleep(0.1)
                    continue
                fb.update(frame)
            except Exception as e:
                pass

    def release(self):
        self._running = False
        if self._kind == "picamera2":
            try:
                self._cam.stop()
            except Exception:
                pass
        elif self._kind == "cv2":
            try:
                self._cam.release()
            except Exception:
                pass
