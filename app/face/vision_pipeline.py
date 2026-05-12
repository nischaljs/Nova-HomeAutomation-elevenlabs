"""Two-thread detect-and-recognize pipeline.

The two stages have different characteristics — detection is fast and
*has to* run at a steady rate to keep engagement signals fresh, while
recognition is slower and only needs to run a couple of times per second.
Running them on one thread means a slow recognize tick (e.g. the user
turned their head, model has to look harder) delays the next detect tick
and engagement updates stall.

Splitting them lets recognition of frame N happen *concurrently* with
detection of frame N+1 — the detector thread keeps publishing
engagement updates while the recognizer is busy. On a Pi 4 under load
this is the difference between engagement-feels-instant and
engagement-feels-laggy.

  detect_thread (~6.6 Hz)         recognize_thread (~2 Hz)
   │                                │
   │ small = resize(frame)          │ wait for new detection
   │ faces_raw = YuNet(small)       │ result = bridge.recognize_in(...)
   │ → engagement.update(...)       │ → LatestVision.publish(recognized=...)
   │ → LatestVision.publish(raw=...)│
   └────────────┬───────────────────┘
                ▼
         LatestVision (one lock, three published fields)

Side benefits of consolidating both stages here:
  * Detection runs on a 480 × 360 input (was 320 × 240), so the same
    MIN_FACE_PX threshold corresponds to a ~85 px face in the source —
    recognizable from ~2–2.5 m instead of ~1 m.
  * Engagement is computed in the detect thread using YuNet's
    landmarks, so the SpeechGate sees an `is_engaged` update every
    ~150 ms — completely independent of how long recognition takes.
"""

import os
import threading
import time

import cv2
import numpy as np

from face_recognition_system.detector import detect_faces

from app.face.face_tools import FrameBuffer, get_bridge
from app.face.lip_motion import LipMotionTracker
from app.orchestration.engagement import ENGAGED_MAX_POSE_ASYM


DETECT_TARGET_W = int(os.environ.get("NOVA_DETECT_W", "480"))
DETECT_TARGET_H = int(os.environ.get("NOVA_DETECT_H", "360"))
DETECT_INTERVAL_S = float(os.environ.get("NOVA_DETECT_INTERVAL_S", "0.15"))
RECOGNIZE_INTERVAL_S = float(os.environ.get("NOVA_RECOGNIZE_INTERVAL_S", "0.5"))
# Idle-mode throttling. When nobody's been in front of the camera for
# IDLE_AFTER_S seconds, drop detection from 6.6 Hz to ~1 Hz. That's a
# ~70 % CPU cut during empty-room time without sacrificing wake-up
# responsiveness — the moment a face appears in any of those 1-Hz
# detect ticks, we drop back to the fast rate immediately. Important
# for battery operation on a mobile robot platform.
IDLE_AFTER_S = float(os.environ.get("NOVA_IDLE_AFTER_S", "60.0"))
IDLE_DETECT_INTERVAL_S = float(os.environ.get("NOVA_IDLE_DETECT_INTERVAL_S", "1.0"))
# Minimum sleep between ticks even if everything was instant — stops a
# loop from pinning a core when there's no work to do.
MIN_SLEEP_S = 0.02


def _face_pose_asym(face: dict) -> float | None:
    """Eye-to-nose horizontal asymmetry from YuNet landmarks.

    Same metric as the face-recognition quality gate's MAX_POSE_ASYM
    check; we lift it here so engagement can react to head pose
    independently of whether the face also passes recognition quality.
    Returns None when landmarks are missing.
    """
    landmarks = face.get("landmarks")
    if landmarks is None or len(landmarks) < 3:
        return None
    right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_to_nose_r = abs(float(right_eye[0]) - float(nose[0]))
    eye_to_nose_l = abs(float(left_eye[0]) - float(nose[0]))
    max_dist = max(eye_to_nose_r, eye_to_nose_l)
    if max_dist <= 0:
        return None
    return abs(eye_to_nose_r - eye_to_nose_l) / max_dist


class LatestVision:
    """Triple-published snapshot — last detect frame + last recognize frame.

    Holds:
      * faces_raw: list of detect dicts (with bbox, score, landmarks) in
        DETECT_TARGET_W × DETECT_TARGET_H coordinates
      * scale_to_source: float multiplier to map those bboxes back to
        the original camera frame so the preview can draw correct boxes
      * recognized: list of recognition results (id, name, confidence,
        face_bbox in detect coords)
      * detect_ts / recognize_ts: monotonic timestamps for staleness
        checks (so a consumer can skip drawing 1 s-old boxes)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._faces_raw: list[dict] = []
        self._recognized: list[dict] = []
        self._detect_ts: float = 0.0
        self._recognize_ts: float = 0.0
        self._scale_to_source: float = 1.0
        # Detection thread publishes (frame_for_recognize, faces_raw,
        # detect_ts) here so the recognize thread doesn't have to
        # re-detect. Holds the *small* (480×360) frame the detector
        # already saw, so SFace embeds the exact pixels YuNet picked.
        self._pending_for_recognize: tuple[np.ndarray, list[dict], float] | None = None

    def publish_detection(
        self,
        *,
        faces_raw: list[dict],
        scale_to_source: float,
        detect_ts: float,
        small_frame: np.ndarray | None = None,
    ):
        with self._lock:
            self._faces_raw = faces_raw
            self._scale_to_source = scale_to_source
            self._detect_ts = detect_ts
            # Recognize thread will pick this up next iteration. Skip if
            # no faces to recognize — recognize thread can sleep.
            if faces_raw and small_frame is not None:
                self._pending_for_recognize = (small_frame, faces_raw, detect_ts)

    def publish_recognition(self, recognized: list[dict], recognize_ts: float):
        with self._lock:
            self._recognized = recognized
            self._recognize_ts = recognize_ts

    def take_pending_for_recognize(self):
        with self._lock:
            p = self._pending_for_recognize
            self._pending_for_recognize = None
            return p

    def snapshot(self) -> tuple[list[dict], list[dict], float, float, float]:
        with self._lock:
            return (
                list(self._faces_raw),
                list(self._recognized),
                self._scale_to_source,
                self._detect_ts,
                self._recognize_ts,
            )


class VisionPipeline:
    def __init__(self, latest: LatestVision, engagement):
        self._latest = latest
        self._engagement = engagement
        self._bridge = get_bridge()
        self._fb = FrameBuffer()
        self._lip_motion = LipMotionTracker()
        self._running = False
        self._detect_thread: threading.Thread | None = None
        self._recognize_thread: threading.Thread | None = None

        # Stats — track both threads independently so a stalled detect
        # or stalled recognize is immediately obvious from the log.
        self._stats_at = 0.0
        self._stats_detects = 0
        self._stats_recogs = 0
        self._stats_detect_ms_max = 0.0
        self._stats_recog_ms_max = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="vision-detect"
        )
        self._recognize_thread = threading.Thread(
            target=self._recognize_loop, daemon=True, name="vision-recognize"
        )
        self._detect_thread.start()
        self._recognize_thread.start()
        print(f"[VISION] pipeline started — detect thread @ "
              f"{DETECT_TARGET_W}x{DETECT_TARGET_H} every {DETECT_INTERVAL_S}s, "
              f"recognize thread every {RECOGNIZE_INTERVAL_S}s")

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # detect thread — fast, drives engagement
    # ------------------------------------------------------------------

    def _detect_loop(self):
        # Cached so the per-tick idle-mode log doesn't fire every loop.
        # Flipping between fast/idle is reported once per transition,
        # not per tick — same pattern as the ENGAGE TRUE/FALSE log.
        was_idle = False
        while self._running:
            t0 = time.monotonic()
            frame = self._fb.get_frame()
            if frame is None:
                time.sleep(0.04)
                continue

            # Decide this tick's pacing. presence_lost_for_s comes from
            # EngagementState — it counts up while no face is visible
            # and resets to ~0 the moment one appears. So as soon as
            # someone walks in, the very next iteration uses the fast
            # interval (the one this tick is sleeping on doesn't matter
            # — we always wake at the slower rate, see a face, then
            # publish the next detection at the fast rate).
            is_idle = self._engagement.presence_lost_for_s() >= IDLE_AFTER_S
            current_interval = (
                IDLE_DETECT_INTERVAL_S if is_idle else DETECT_INTERVAL_S
            )
            if is_idle != was_idle:
                if is_idle:
                    print(f"[VISION] entering idle mode (no face seen for "
                          f"{IDLE_AFTER_S:.0f}s) — detect rate "
                          f"{1/IDLE_DETECT_INTERVAL_S:.1f}Hz")
                else:
                    print(f"[VISION] leaving idle mode — back to "
                          f"{1/DETECT_INTERVAL_S:.1f}Hz detect rate")
                was_idle = is_idle

            try:
                src_h, src_w = frame.shape[:2]
                if src_w != DETECT_TARGET_W or src_h != DETECT_TARGET_H:
                    small = cv2.resize(frame, (DETECT_TARGET_W, DETECT_TARGET_H))
                    scale_to_source = src_w / float(DETECT_TARGET_W)
                else:
                    small = frame
                    scale_to_source = 1.0

                t_detect_start = time.monotonic()
                faces_raw = detect_faces(small) or []
                detect_ms = (time.monotonic() - t_detect_start) * 1000.0
                self._stats_detect_ms_max = max(self._stats_detect_ms_max, detect_ms)

                detect_ts = time.monotonic()
                self._stats_detects += 1

                # Lip-motion check first — it needs the same small
                # frame we just detected on. Result feeds engagement
                # so the SpeechGate can open faster when lips move.
                lips_moving = self._lip_motion.update(small, faces_raw)

                # Engagement signal is updated on the detect thread —
                # this is what makes the audio gate / session lifecycle
                # responsive within ~150 ms of someone walking up,
                # regardless of how busy the recognize thread is.
                self._update_engagement(faces_raw, lips_moving)

                # Publish detection (and the small frame) so the
                # recognize thread can pick it up.
                self._latest.publish_detection(
                    faces_raw=faces_raw,
                    scale_to_source=scale_to_source,
                    detect_ts=detect_ts,
                    small_frame=small,
                )

                # If no faces, also clear stale recognition so the
                # tracker can publish face_lost. (Doing it here means
                # we don't depend on the recognize thread's cadence.)
                if not faces_raw:
                    self._latest.publish_recognition([], detect_ts)

                self._maybe_log_stats()
            except Exception as e:
                print(f"[VISION] detect loop error (ignored): "
                      f"{type(e).__name__}: {e}")

            elapsed = time.monotonic() - t0
            sleep_left = max(MIN_SLEEP_S, current_interval - elapsed)
            time.sleep(sleep_left)

    # ------------------------------------------------------------------
    # recognize thread — slower, drives identity events
    # ------------------------------------------------------------------

    def _recognize_loop(self):
        last_recognize = 0.0
        while self._running:
            t0 = time.monotonic()
            if not self._bridge.models_ready:
                # Models still downloading — recognize thread is a
                # no-op until they're ready, but we still tick every
                # 0.5 s so we don't oversleep the moment they finish.
                time.sleep(0.5)
                continue

            if time.monotonic() - last_recognize < RECOGNIZE_INTERVAL_S:
                time.sleep(MIN_SLEEP_S)
                continue

            pending = self._latest.take_pending_for_recognize()
            if pending is None:
                # Nothing to recognize — sleep a short tick.
                time.sleep(MIN_SLEEP_S)
                continue

            small, faces_raw, _detect_ts = pending
            try:
                t_recog_start = time.monotonic()
                recognized = self._bridge.recognize_in(small, faces_raw)
                recog_ms = (time.monotonic() - t_recog_start) * 1000.0
                self._stats_recog_ms_max = max(self._stats_recog_ms_max, recog_ms)
                self._stats_recogs += 1
                self._latest.publish_recognition(recognized, time.monotonic())
                last_recognize = time.monotonic()
            except Exception as e:
                print(f"[VISION] recognize loop error (ignored): "
                      f"{type(e).__name__}: {e}")
                last_recognize = time.monotonic()

            elapsed = time.monotonic() - t0
            sleep_left = max(MIN_SLEEP_S, RECOGNIZE_INTERVAL_S - elapsed)
            time.sleep(sleep_left)

    # ------------------------------------------------------------------
    # engagement + stats
    # ------------------------------------------------------------------

    def _update_engagement(self, faces_raw: list[dict], lips_moving: bool):
        if not faces_raw:
            self._engagement.update(
                present=False, engaged=False, asym=None, lips_moving=False
            )
            return
        # We pass the *minimum* asym across all faces — that's the
        # face most likely to be looking at the camera, and the one
        # that drives the "engaged?" decision. Sending the min also
        # gives the diagnostic log a meaningful number when the engaged
        # decision flips ("oh, asym dropped to 0.18 — that's why").
        best_asym: float | None = None
        engaged = False
        for f in faces_raw:
            asym = _face_pose_asym(f)
            if asym is None:
                continue
            if best_asym is None or asym < best_asym:
                best_asym = asym
            if asym <= ENGAGED_MAX_POSE_ASYM:
                engaged = True
        self._engagement.update(
            present=True,
            engaged=engaged,
            asym=best_asym,
            lips_moving=lips_moving and engaged,
        )

    def _maybe_log_stats(self):
        now = time.monotonic()
        if now - self._stats_at < 30.0:
            return
        if self._stats_at == 0.0:
            self._stats_at = now
            self._stats_detects = 0
            self._stats_recogs = 0
            self._stats_detect_ms_max = 0.0
            self._stats_recog_ms_max = 0.0
            return
        dt = now - self._stats_at
        det_hz = self._stats_detects / dt
        rec_hz = self._stats_recogs / dt
        print(f"[VISION] stats: detect={det_hz:.1f}Hz "
              f"(peak={self._stats_detect_ms_max:.0f}ms) "
              f"recognize={rec_hz:.1f}Hz "
              f"(peak={self._stats_recog_ms_max:.0f}ms) over {dt:.0f}s")
        self._stats_at = now
        self._stats_detects = 0
        self._stats_recogs = 0
        self._stats_detect_ms_max = 0.0
        self._stats_recog_ms_max = 0.0


LATEST_VISION: LatestVision | None = None


def get_latest() -> LatestVision:
    global LATEST_VISION
    if LATEST_VISION is None:
        LATEST_VISION = LatestVision()
    return LATEST_VISION
