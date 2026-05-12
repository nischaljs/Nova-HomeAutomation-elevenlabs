import os
import threading
import time
from pathlib import Path

import cv2

from app.face.face_tools import FrameBuffer

CAM_W = 640
CAM_H = 480
MAX_PROBE_INDEX = 10

# Cache the working V4L2 device index between boots. After a USB replug
# or reboot, V4L2 indices for the bcm2835 codec/ISP nodes can shift in
# front of the actual USB webcam — and probing 0..MAX_PROBE_INDEX every
# boot costs 1–2 s of latency before the first frame. With a cache the
# happy path skips straight to the known-good index.
CACHE_DIR = Path(os.environ.get("NOVA_CACHE_DIR",
                                str(Path.home() / ".cache" / "nova")))
INDEX_CACHE_PATH = CACHE_DIR / "camera.idx"

# Backoff schedule when the camera goes away mid-run (USB unplug, kernel
# eject). Reopens at increasing intervals so we don't peg one CPU
# spinning on cv2.VideoCapture() while the kernel is still working out
# what happened.
RECONNECT_BACKOFF_S = [2.0, 5.0, 10.0, 30.0]


def _read_cached_index() -> int | None:
    try:
        return int(INDEX_CACHE_PATH.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _write_cached_index(idx: int):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_CACHE_PATH.write_text(str(idx))
    except OSError as e:
        print(f"[CAMERA] cache write failed (ignored): {e}")


def _open_picamera2(w: int, h: int):
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


def _try_open_v4l2_index(idx: int, w: int, h: int):
    """Open one specific /dev/videoN. Returns the cap on success, None
    on any failure (which the caller treats as 'try the next index')."""
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            cap.release()
            return None
        return cap, frame
    except Exception:
        return None


def _open_cv2_cam(w: int, h: int) -> tuple[cv2.VideoCapture, int]:
    """Open a USB webcam via V4L2.

    Strategy: try the cached index from the last successful run first.
    If that fails (or there's no cache), probe /dev/video0..N in order
    and cache the winner. Saves 1–2 s of probing on every subsequent
    boot — a meaningful chunk of Nova's startup time."""
    cached = _read_cached_index()
    last_err: Exception | None = None

    indices_to_try: list[int] = []
    if cached is not None:
        indices_to_try.append(cached)
    indices_to_try.extend(i for i in range(MAX_PROBE_INDEX + 1) if i != cached)

    for idx in indices_to_try:
        result = _try_open_v4l2_index(idx, w, h)
        if result is None:
            continue
        cap, frame = result
        source = "cached" if idx == cached else "probed"
        print(f"[CAMERA] Opened /dev/video{idx} via OpenCV ({source}, "
              f"frame={frame.shape[1]}x{frame.shape[0]})")
        if idx != cached:
            _write_cached_index(idx)
        return cap, idx

    raise RuntimeError(
        f"No usable V4L2 capture device found in /dev/video0..{MAX_PROBE_INDEX} "
        f"(last error: {last_err})"
    )


class Camera:
    """Background camera thread feeding the singleton FrameBuffer.

    Tries Picamera2 first, then V4L2 USB. If the camera disappears
    mid-run (USB unplug, kernel reset), the thread enters a reconnect
    backoff loop instead of silently dying — so re-inserting the USB
    cable will bring the pipeline back without restarting Nova.
    """

    def __init__(self, w: int = CAM_W, h: int = CAM_H):
        self._w = w
        self._h = h
        self._running = True
        self._cam = None
        self._kind: str | None = None
        self._v4l2_index: int | None = None
        self._last_frame_at: float = 0.0
        self._reconnect_streak = 0

        self._init_camera()
        if self._kind is None:
            print(f"[CAMERA] ⚠ no camera available — running blind. "
                  f"Will retry every {RECONNECT_BACKOFF_S[0]}s in background.")
        else:
            print(f"[CAMERA] Backend: {self._kind}")

        t = threading.Thread(target=self._loop, daemon=True, name="camera-loop")
        t.start()

    def _init_camera(self) -> bool:
        """Try Picamera2 first (lower CPU on a Pi camera module), then
        fall back to V4L2/USB. Sets self._kind on success."""
        try:
            self._cam = _open_picamera2(self._w, self._h)
            self._kind = "picamera2"
            return True
        except Exception:
            pass
        try:
            self._cam, self._v4l2_index = _open_cv2_cam(self._w, self._h)
            self._kind = "cv2"
            return True
        except Exception as e:
            print(f"[CAMERA] initialization failed: {e}")
            self._cam = None
            self._kind = None
            return False

    def _grab_frame(self):
        """Pull one frame from whichever backend is live, or raise."""
        if self._kind == "picamera2":
            return self._cam.capture_array()
        if self._kind == "cv2":
            self._cam.grab()
            ok, frame = self._cam.retrieve()
            if not ok:
                raise RuntimeError("v4l2 retrieve returned ok=False")
            return frame
        raise RuntimeError("no camera backend")

    def _teardown_camera(self):
        """Best-effort release of whatever we currently hold."""
        try:
            if self._kind == "picamera2" and self._cam is not None:
                self._cam.stop()
            elif self._kind == "cv2" and self._cam is not None:
                self._cam.release()
        except Exception as e:
            print(f"[CAMERA] teardown ignored {type(e).__name__}: {e}")
        self._cam = None
        self._kind = None

    def _reconnect_with_backoff(self):
        wait = RECONNECT_BACKOFF_S[min(self._reconnect_streak,
                                       len(RECONNECT_BACKOFF_S) - 1)]
        print(f"[CAMERA] reconnect attempt #{self._reconnect_streak + 1} "
              f"after {wait}s sleep")
        end = time.monotonic() + wait
        while self._running and time.monotonic() < end:
            time.sleep(0.2)
        if not self._running:
            return
        if self._init_camera():
            print("[CAMERA] reconnect succeeded")
            self._reconnect_streak = 0
        else:
            self._reconnect_streak += 1

    def _loop(self):
        fb = FrameBuffer()
        consecutive_errors = 0
        frames_since_stats = 0
        stats_since = time.monotonic()
        while self._running:
            if self._kind is None:
                self._reconnect_with_backoff()
                # Reset stats window so a long reconnect doesn't show as 0fps.
                stats_since = time.monotonic()
                frames_since_stats = 0
                continue
            try:
                frame = self._grab_frame()
                fb.update(frame)
                self._last_frame_at = time.monotonic()
                consecutive_errors = 0
                frames_since_stats += 1

                # Periodic FPS log so the user can see at a glance
                # whether the camera is healthy. A typical USB webcam
                # at 640x480 reads ~30 fps; Picamera2 closer to 30-60.
                # Drops well below that during heavy detect/recognize
                # are normal under CPU pressure.
                now = time.monotonic()
                if now - stats_since >= 30.0:
                    fps = frames_since_stats / (now - stats_since)
                    print(f"[CAMERA] {self._kind} capture FPS={fps:.1f} "
                          f"(last 30s)")
                    stats_since = now
                    frames_since_stats = 0
            except Exception as e:
                consecutive_errors += 1
                # Burst of errors → assume the device is gone, tear down
                # and trigger a clean reconnect. 5-in-a-row is enough to
                # paper over single transient bad reads (USB hiccup).
                if consecutive_errors >= 5:
                    print(f"[CAMERA] capture failing repeatedly "
                          f"({type(e).__name__}: {e}) — reconnecting")
                    self._teardown_camera()
                    consecutive_errors = 0
                else:
                    time.sleep(0.05)

    def release(self):
        self._running = False
        self._teardown_camera()

    @property
    def last_frame_age_s(self) -> float:
        if self._last_frame_at <= 0:
            return float("inf")
        return time.monotonic() - self._last_frame_at

    @property
    def kind(self) -> str | None:
        return self._kind
