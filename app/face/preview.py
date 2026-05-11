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

# Cap preview at ~30 FPS — the camera + Qt imshow path was spinning at
# 80–100 FPS on the Pi, burning CPU that should go to face recognition.
# 30 FPS is plenty smooth for a live preview and frees ~60% of one core.
TARGET_FPS = 30
TARGET_FRAME_INTERVAL_S = 1.0 / TARGET_FPS

# How long to keep displaying a recognized name after the recognizer stops
# returning it. Stops box labels from flickering Name → UNKNOWN → Name as
# the head turns or a frame goes blurry.
LABEL_HOLD_S = 3.0


class CameraPreview:
    def __init__(self, window_name="Nova Camera"):
        self._window_name = window_name
        self._running = False
        self._thread = None
        self._faces_by_id: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._fps = 0.0
        self._enabled = os.environ.get("NOVA_DEBUG", "1") == "1"

    def update_faces(self, recognized: list[dict]):
        """List of recognize_all() results — one entry per detected face."""
        now = time.time()
        with self._lock:
            for f in recognized or []:
                if f.get("unknown") or not f.get("name") or not f.get("id"):
                    continue
                self._faces_by_id[f["id"]] = {
                    "id": f["id"],
                    "name": f["name"],
                    "confidence": f.get("confidence", 0.0),
                    "bbox": f.get("face_bbox"),
                    "last_seen": now,
                }
            stale = [
                fid
                for fid, info in self._faces_by_id.items()
                if now - info["last_seen"] > LABEL_HOLD_S
            ]
            for fid in stale:
                del self._faces_by_id[fid]

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
            small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
            return detect_faces(small) or []
        except Exception:
            return []

    def _match_label(self, det_bbox):
        """Find the recognized face whose stored bbox is closest to det_bbox.
        Returns (label_text, color)."""
        if det_bbox is None:
            return "UNKNOWN", RED

        dx, dy, dw, dh = det_bbox
        dcx, dcy = dx + dw / 2, dy + dh / 2
        radius_sq = (max(dw, dh) * 0.6) ** 2

        best = None
        best_dist = float("inf")
        with self._lock:
            entries = list(self._faces_by_id.values())
        for info in entries:
            rbbox = info.get("bbox")
            if rbbox is None:
                continue
            rx, ry, rw, rh = rbbox
            rcx, rcy = rx + rw / 2, ry + rh / 2
            dist = (dcx - rcx) ** 2 + (dcy - rcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = info

        if best is not None and best_dist <= radius_sq:
            name = str(best["name"]).strip() or "Unknown"
            conf = float(best.get("confidence", 0.0))
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

            for f in cached_faces:
                bbox = f.get("bbox")
                if bbox is None:
                    continue
                text, color = self._match_label(bbox)
                self._draw_box(display, bbox, text, color)

            if frame_count % 30 == 0:
                self._fps = 30 / max(now - t_fps, 0.001)
                t_fps = now
                if frame_count % 300 == 0:
                    print(f"[PREVIEW] FPS={self._fps:.0f} frame={frame_count} faces={len(cached_faces)}")

            cv2.putText(display, f"FPS {self._fps:.0f}", (8, 22), FONT, 0.6, (200, 200, 200), 1)
            cv2.imshow(self._window_name, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[PREVIEW] User pressed 'q' — stopping")
                break

            # Cap frame rate to TARGET_FPS so we don't hog the CPU
            sleep_time = TARGET_FRAME_INTERVAL_S - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)
