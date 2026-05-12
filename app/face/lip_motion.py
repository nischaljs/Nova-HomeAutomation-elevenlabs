"""Lightweight lip-motion detector.

Why this exists: the engagement check ("is a face looking at the
camera?") is good but coarse — it says yes the entire time a user is
standing in front of Nova, even when they're just thinking. The
SpeechGate then has to wait the full idle threshold (250 ms voiced)
before opening, which costs us snappiness.

Lip motion is the missing signal. When we can *visually confirm* the
engaged user's lips are moving, we know the incoming audio is from
them and we can let the gate open in ~80 ms instead of 250 ms — a
~3× faster response.

How it works (kept dead simple to stay cheap on Pi 4):

  1. YuNet's per-face landmarks include the two mouth corners
     (indices 3 and 4 in our detector's output).
  2. We crop a small luminance ROI centered on the mouth, normalized
     to a fixed 40 × 24 grid so the signal is invariant to how close
     the user is to the camera.
  3. We track the mean absolute pixel diff between consecutive ROIs.
     When lips move, those pixels change a lot frame-to-frame; when
     the face is static, they don't.
  4. A running 30th-percentile of the recent diffs gives us an
     adaptive baseline (essentially the "noise floor" of static
     mouth pixels) — the threshold scales with each user's lighting
     and skin tone instead of needing a global constant.
  5. We declare "lips moving" when current diff > baseline × 2.5
     for at least one frame within the last LIP_MOTION_STICKY_S.

Cost on Pi 4: one absdiff on a 40 × 24 image + one mean() — about
0.5 ms per detect tick. Memory: ~40-frame history × 1 float each.
The whole thing is well below the noise floor of the rest of the
pipeline.

Failure modes (intentionally lenient):
  * No landmarks → tracker says "no lip motion" (safer than yes).
  * ROI out of frame after a face moves quickly → tracker resets,
    next frame establishes a new baseline.
  * Multiple faces → we track each by face_id (matched to recognition
    where available, falling back to bbox-center proximity).
"""

import collections
import os
import time

import cv2
import numpy as np


# Motion-threshold ratio: how many times the running baseline a frame's
# mouth-pixel diff must exceed before we call it "lips moving". Lower
# = more sensitive (catches softer speech but also chewing/laughing
# false positives), higher = stricter.
LIP_MOTION_RATIO = float(os.environ.get("NOVA_LIP_MOTION_RATIO", "2.5"))
# How long a "lips were moving" reading stays true after the last
# moving frame. Acts as a debounce so the gate doesn't flap between
# fast-open and slow-open every detect tick.
LIP_MOTION_STICKY_S = float(os.environ.get("NOVA_LIP_MOTION_STICKY_S", "0.6"))
# Smoothed ROI grid size. 40×24 ≈ mouth aspect ratio, big enough to
# capture lip shape changes, small enough that absdiff costs nothing.
ROI_W = 40
ROI_H = 24
# Minimum baseline diff before we trust the ratio test. When the
# baseline is essentially zero (static frames, locked face) the ratio
# divides into noise and we'd false-positive. 0.5 is "barely above
# webcam sensor noise".
MIN_BASELINE = 0.5
HISTORY_LEN = 40


class _PerFaceState:
    __slots__ = ("prev_roi", "diffs", "last_moving_at")

    def __init__(self):
        self.prev_roi: np.ndarray | None = None
        self.diffs: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self.last_moving_at: float = 0.0


class LipMotionTracker:
    """Tracks per-face lip motion. Lookup is bbox-center based — we
    don't need true face-id tracking, just spatial continuity from
    one detect tick to the next.

    Returns the per-frame "is any face currently moving its lips?"
    boolean. Stale state for faces that disappeared is reaped
    periodically so multi-visitor sessions don't grow this map.
    """

    def __init__(self):
        # Keyed by face-bbox-center quantized to a 32 px grid — coarse
        # enough that one user wobbling doesn't fragment into multiple
        # tracks, fine enough that two users 50 cm apart map distinctly.
        self._tracks: dict[tuple[int, int], _PerFaceState] = {}
        self._last_reap_at = 0.0
        self._enabled = os.environ.get("NOVA_LIP_MOTION", "1") == "1"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, frame_small: np.ndarray, faces: list[dict]) -> bool:
        """Process one detect tick. Returns True if any tracked face's
        lips are currently moving (within LIP_MOTION_STICKY_S)."""
        if not self._enabled or not faces:
            self._maybe_reap()
            return self._any_recently_moving()

        any_moving_now = False
        now = time.monotonic()
        for face in faces:
            moved = self._update_one(frame_small, face, now)
            if moved:
                any_moving_now = True

        self._maybe_reap()
        return any_moving_now or self._any_recently_moving()

    def _update_one(self, frame_small: np.ndarray, face: dict, now: float) -> bool:
        landmarks = face.get("landmarks")
        if landmarks is None or len(landmarks) < 5:
            return False
        bbox = face.get("bbox")
        if bbox is None:
            return False
        try:
            r_mouth = landmarks[3]
            l_mouth = landmarks[4]
            mx = int((float(r_mouth[0]) + float(l_mouth[0])) / 2.0)
            my = int((float(r_mouth[1]) + float(l_mouth[1])) / 2.0)
        except (IndexError, TypeError, ValueError):
            return False

        # ROI extent scales with face bbox so the same fraction of
        # the mouth is captured at any distance.
        bw = max(20, int(float(bbox[2]) * 0.45))
        bh = max(14, int(float(bbox[3]) * 0.22))
        h_img, w_img = frame_small.shape[:2]
        x0 = max(0, mx - bw // 2)
        y0 = max(0, my - bh // 2)
        x1 = min(w_img, x0 + bw)
        y1 = min(h_img, y0 + bh)
        if x1 - x0 < 8 or y1 - y0 < 6:
            return False

        roi = frame_small[y0:y1, x0:x1]
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            std = cv2.resize(gray, (ROI_W, ROI_H))
        except cv2.error:
            return False

        # Key by quantized bbox center so the same user across ticks
        # ends up in the same track even when their face moves a bit.
        key = (mx // 32, my // 32)
        track = self._tracks.get(key)
        if track is None:
            # Try to find an adjacent (within 1 cell) existing track so
            # a small drift doesn't reset the baseline every tick.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    adj = (key[0] + dx, key[1] + dy)
                    if adj in self._tracks:
                        track = self._tracks.pop(adj)
                        break
                if track is not None:
                    break
            if track is None:
                track = _PerFaceState()
            self._tracks[key] = track

        if track.prev_roi is None or track.prev_roi.shape != std.shape:
            track.prev_roi = std
            return False

        diff = cv2.absdiff(std, track.prev_roi)
        diff_mean = float(diff.mean())
        track.diffs.append(diff_mean)
        track.prev_roi = std

        if len(track.diffs) < 8:
            return False

        # 30th-percentile baseline — cheap to compute, robust to
        # occasional outliers (a single big motion frame doesn't
        # dominate). The remaining 70% includes both static and
        # moving frames so the threshold lives well above static noise.
        sorted_diffs = sorted(track.diffs)
        baseline = sorted_diffs[len(sorted_diffs) // 3]
        if baseline < MIN_BASELINE:
            baseline = MIN_BASELINE

        is_moving = diff_mean > (baseline * LIP_MOTION_RATIO)
        if is_moving:
            track.last_moving_at = now
        return is_moving

    def _any_recently_moving(self) -> bool:
        now = time.monotonic()
        for track in self._tracks.values():
            if now - track.last_moving_at <= LIP_MOTION_STICKY_S:
                return True
        return False

    def _maybe_reap(self):
        now = time.monotonic()
        if now - self._last_reap_at < 5.0:
            return
        self._last_reap_at = now
        cutoff = now - 10.0
        dead = [k for k, t in self._tracks.items() if t.last_moving_at < cutoff and not t.diffs]
        for k in dead:
            del self._tracks[k]
        # Also cap total tracks at 16 to bound memory in weird scenes.
        if len(self._tracks) > 16:
            # Remove the tracks with the longest time-since-moving.
            sorted_keys = sorted(
                self._tracks.keys(),
                key=lambda k: self._tracks[k].last_moving_at,
            )
            for k in sorted_keys[: len(self._tracks) - 16]:
                del self._tracks[k]
