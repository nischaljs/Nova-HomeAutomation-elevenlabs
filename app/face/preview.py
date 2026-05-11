import os
import threading
import time

import cv2

from app.face.face_tools import FrameBuffer

from face_recognition_system.detector import detect_faces

GREEN = (0, 220, 0)
RED = (0, 60, 240)
YELLOW = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

DETECT_SCALE = 0.5
SCALE_BACK = 1 / DETECT_SCALE
DETECT_INTERVAL_S = 0.15

# How long to keep displaying the last recognized name after the recognizer
# starts returning unknown/low-quality/None. Stops the box from flickering
# Nischal → UNKNOWN → Nischal as the head turns or a single frame goes blurry.
LABEL_HOLD_S = 3.0


class CameraPreview:
    def __init__(self, window_name="Nova Camera"):
        self._window_name = window_name
        self._running = False
        self._thread = None
        self._face_info = None
        self._lock = threading.Lock()
        self._fps = 0.0
        self._enabled = os.environ.get("NOVA_DEBUG", "1") == "1"
        self._last_label = None
        self._lost_since: float | None = None

    def update_face(self, face_info: dict | None):
        """Identity result from the recognition thread (name, confidence)."""
        has_name = (
            face_info is not None
            and face_info.get("name")
            and face_info.get("name") != "unknown"
            and not face_info.get("unknown")
        )
        with self._lock:
            if has_name:
                self._face_info = face_info
                self._lost_since = None
                return
            if self._face_info is None or not self._face_info.get("name"):
                self._face_info = face_info
                return
            if self._lost_since is None:
                self._lost_since = time.time()
            if time.time() - self._lost_since < LABEL_HOLD_S:
                return
            self._face_info = face_info
            self._lost_since = None

    def start(self):
        if not self._enabled:
            print("[PREVIEW] Disabled (NOVA_DEBUG=0)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[PREVIEW] Window '{self._window_name}' started")

    def stop(self):
        self._running = False
        if self._enabled:
            cv2.destroyAllWindows()
        print("[PREVIEW] Stopped")

    def _detect(self, frame):
        try:
            t0 = time.time()
            small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
            faces = detect_faces(small) or []
            return faces
        except Exception:
            return []

    def _label_for(self, face: dict | None) -> tuple[str, tuple[int, int, int]]:
        if face and face.get("name") and face["name"] != "unknown":
            name = face["name"].title()
            conf = float(face.get("confidence", 0))
            return f"{name} {conf:.0%}", GREEN
        return "UNKNOWN", RED

    def _draw_box(self, display, bbox, text, color):
        x, y, w, h = (int(v * SCALE_BACK) for v in bbox)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
        cv2.rectangle(display, (x, max(0, y - th - 8)), (x + tw + 4, y), color, -1)
        cv2.putText(display, text, (x + 2, max(th, y - 4)), FONT, 0.65, (255, 255, 255), 2)

    def _loop(self):
        fb = FrameBuffer()
        t_fps = time.time()
        frame_count = 0
        last_detect = 0.0
        cached_faces: list = []

        print("[PREVIEW] Loop started")

        while self._running:
            frame = fb.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_count += 1
            display = frame.copy()
            now = time.time()

            if now - last_detect >= DETECT_INTERVAL_S:
                cached_faces = self._detect(frame)
                last_detect = now
                if frame_count % 100 == 0:
                    print(f"[PREVIEW] Detected {len(cached_faces)} faces in frame {frame_count}")

            with self._lock:
                face = self._face_info

            text, color = self._label_for(face)
            if text != self._last_label:
                print(f"[PREVIEW] Box label: {text}")
                self._last_label = text

            for f in cached_faces:
                bbox = f.get("bbox")
                if bbox is None:
                    continue
                self._draw_box(display, bbox, text, color)

            if frame_count % 30 == 0:
                self._fps = 30 / max(now - t_fps, 0.001)
                t_fps = now
                if frame_count % 300 == 0:
                    print(f"[PREVIEW] FPS={self._fps:.0f} frame={frame_count} faces_cached={len(cached_faces)}")

            cv2.putText(display, f"FPS {self._fps:.0f}", (8, 22), FONT, 0.6, (200, 200, 200), 1)
            cv2.imshow(self._window_name, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[PREVIEW] User pressed 'q' — stopping")
                break
