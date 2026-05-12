import os
import threading
import time

import cv2

from app.face.face_tools import FrameBuffer

GREEN = (0, 220, 0)
RED = (0, 60, 240)
YELLOW = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Cap preview at ~30 FPS — the camera + Qt imshow path was spinning at
# 80–100 FPS on the Pi, burning CPU that should go to face recognition.
# 30 FPS is plenty smooth for a live preview and frees ~60% of one core.
TARGET_FPS = 30
TARGET_FRAME_INTERVAL_S = 1.0 / TARGET_FPS

# How long to keep displaying a recognized name after the recognizer stops
# returning it. Stops box labels from flickering Name → UNKNOWN → Name as
# the head turns or a frame goes blurry.
LABEL_HOLD_S = 3.0


def _display_available() -> bool:
    """True when a display server is actually reachable.

    On headless Pi neither DISPLAY nor WAYLAND_DISPLAY is set, and trying
    to open a Qt window aborts the whole process. Checking both env vars
    is the cheapest pre-check. NOVA_HEADLESS=1 forces disabled even when
    a display exists (useful in systemd units that inherit DISPLAY but
    don't actually have one).
    """
    if os.environ.get("NOVA_HEADLESS") == "1":
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class CameraPreview:
    """Debug preview window. Consumes LatestVision — no detection of its own.

    Previously this class ran a parallel detector at ~6 Hz so it could
    draw bounding boxes. That doubled the detection cost on the Pi for
    no benefit (the FaceMonitor needs the same boxes). After the
    VisionPipeline refactor, the preview just reads detected boxes from
    LatestVision and overlays them on the live camera frame.
    """

    def __init__(self, window_name="Nova Camera"):
        self._window_name = window_name
        self._running = False
        self._thread = None
        self._faces_by_id: dict[str, dict] = {}
        self._raw_boxes: list[dict] = []
        self._scale_to_source: float = 1.0
        self._lock = threading.Lock()
        self._fps = 0.0
        wants_preview = os.environ.get("NOVA_DEBUG", "1") == "1"
        self._enabled = wants_preview and _display_available()
        self._gui_dead = False

    def update_faces(self, faces_raw: list[dict], recognized: list[dict],
                     scale_to_source: float):
        """Push the latest vision snapshot. Called by FaceMonitor's poll
        loop — same data the recognizer just consumed, so the labels and
        boxes are guaranteed to be in sync."""
        now = time.time()
        with self._lock:
            self._raw_boxes = faces_raw or []
            self._scale_to_source = scale_to_source
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
                fid for fid, info in self._faces_by_id.items()
                if now - info["last_seen"] > LABEL_HOLD_S
            ]
            for fid in stale:
                del self._faces_by_id[fid]

    def start(self):
        if not self._enabled:
            reason = (
                "NOVA_HEADLESS=1" if os.environ.get("NOVA_HEADLESS") == "1"
                else "no DISPLAY/WAYLAND_DISPLAY" if not _display_available()
                else "NOVA_DEBUG=0"
            )
            print(f"[PREVIEW] Disabled ({reason})")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[PREVIEW] Window '{self._window_name}' started")

    def stop(self):
        self._running = False
        if self._enabled and not self._gui_dead:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[PREVIEW] destroyAllWindows failed (ignored): {type(e).__name__}: {e}")
        print("[PREVIEW] Stopped")

    def _match_label(self, det_bbox):
        """Find the recognized face whose stored bbox is closest to det_bbox.
        Returns (label_text, color). bbox here is already in *detection*
        coordinates (480×360), as is every stored face_bbox.
        """
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

    def _draw_box(self, display, bbox, text, color, scale):
        try:
            if bbox is None or len(bbox) < 4:
                return
            x = int(float(bbox[0]) * scale)
            y = int(float(bbox[1]) * scale)
            w = int(float(bbox[2]) * scale)
            h = int(float(bbox[3]) * scale)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(display, pt1, pt2, color, 2)
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
            cv2.rectangle(display, (x, max(0, y - th - 8)), (x + tw + 4, y), color, -1)
            cv2.putText(display, text, (x + 2, max(th, y - 4)), FONT, 0.65, (255, 255, 255), 2)
        except Exception as e:
            print(f"[PREVIEW] draw_box error (ignored): {type(e).__name__}: {e}")

    def _loop(self):
        fb = FrameBuffer()
        t_fps = time.time()
        frame_count = 0

        print("[PREVIEW] Loop started")

        while self._running:
            try:
                frame = fb.get_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue

                frame_count += 1
                display = frame.copy()
                now = time.time()

                with self._lock:
                    raw = list(self._raw_boxes)
                    scale = self._scale_to_source

                for f in raw:
                    bbox = f.get("bbox")
                    if bbox is None:
                        continue
                    text, color = self._match_label(bbox)
                    self._draw_box(display, bbox, text, color, scale)

                if frame_count % 30 == 0:
                    self._fps = 30 / max(now - t_fps, 0.001)
                    t_fps = now
                    if frame_count % 300 == 0:
                        print(f"[PREVIEW] FPS={self._fps:.0f} frame={frame_count} faces={len(raw)}")

                cv2.putText(display, f"FPS {self._fps:.0f}", (8, 22), FONT, 0.6, (200, 200, 200), 1)
                try:
                    cv2.imshow(self._window_name, display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[PREVIEW] User pressed 'q' — stopping")
                        break
                except cv2.error as e:
                    # No display reachable (Qt couldn't open a window).
                    # Switch off permanently rather than spam errors every
                    # frame; the rest of Nova keeps running fine headless.
                    print(f"[PREVIEW] imshow failed — disabling preview "
                          f"({type(e).__name__}: {e})")
                    self._gui_dead = True
                    self._running = False
                    break

                sleep_time = TARGET_FRAME_INTERVAL_S - (time.time() - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                print(f"[PREVIEW] loop error (ignored): {type(e).__name__}: {e}")
                time.sleep(0.1)
