"""LipMotionTracker — pure numpy, no camera needed.

We hand-craft "frames" as numpy arrays and "faces" as dicts that match
YuNet's output schema. Lets us prove that the tracker correctly
distinguishes "lips moving" from "lips static" without ever opening
a camera.

Run with: pytest tests/test_lip_motion.py
"""

import numpy as np
import pytest

from app.face.lip_motion import LipMotionTracker


def _make_frame(brightness: int = 64) -> np.ndarray:
    """480×360 BGR frame at the given uniform brightness."""
    return np.full((360, 480, 3), brightness, dtype=np.uint8)


def _face_dict(
    cx: int = 240, cy: int = 180, w: int = 120, h: int = 150,
    mouth_y: int = 230,
) -> dict:
    """Synthetic YuNet-style face dict centered at (cx, cy)."""
    x = cx - w // 2
    y = cy - h // 2
    return {
        "bbox": (x, y, w, h),
        "score": 0.95,
        "landmarks": [
            (cx - 25, cy - 20),  # R eye
            (cx + 25, cy - 20),  # L eye
            (cx, cy + 5),         # nose
            (cx - 20, mouth_y),  # R mouth
            (cx + 20, mouth_y),  # L mouth
        ],
        "raw": None,
    }


def test_no_landmarks_returns_not_moving():
    tracker = LipMotionTracker()
    frame = _make_frame()
    face = {"bbox": (100, 100, 80, 80), "score": 0.9, "landmarks": None, "raw": None}
    result = tracker.update(frame, [face])
    assert result is False


def test_static_frames_no_motion_detected():
    """Feed identical frames — the tracker's baseline rises to match
    the (zero) per-frame diff and the threshold never trips."""
    tracker = LipMotionTracker()
    frame = _make_frame()
    face = _face_dict()
    # Need at least 8 samples in the diff history before the tracker
    # even considers reporting motion.
    saw_motion = False
    for _ in range(20):
        if tracker.update(frame, [face]):
            saw_motion = True
    assert saw_motion is False, (
        "static frames should never report lip motion"
    )


def test_changing_mouth_region_triggers_motion():
    """Burn baseline on a static frame, then introduce a hot mouth ROI
    that's wildly different. The tracker should latch onto that as
    motion since current_diff >> baseline."""
    tracker = LipMotionTracker()
    static = _make_frame(brightness=64)
    face = _face_dict()

    # Burn 12 static frames to establish baseline.
    for _ in range(12):
        tracker.update(static, [face])

    # Now alternate hot/cold mouth regions to simulate lips moving.
    saw_motion = False
    for i in range(15):
        f = static.copy()
        # Paint a bright/dark patch over the mouth area.
        mouth_y = 230
        intensity = 220 if i % 2 == 0 else 30
        f[mouth_y - 12:mouth_y + 12, 200:280] = intensity
        if tracker.update(f, [face]):
            saw_motion = True
    assert saw_motion is True, (
        "alternating hot/cold mouth ROI should trip the motion threshold"
    )


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("NOVA_LIP_MOTION", "0")
    tracker = LipMotionTracker()
    assert tracker.enabled is False
    # Even with valid motion-inducing input, disabled tracker returns False.
    static = _make_frame()
    face = _face_dict()
    for _ in range(15):
        result = tracker.update(static, [face])
    assert result is False


def test_no_faces_returns_false():
    tracker = LipMotionTracker()
    frame = _make_frame()
    assert tracker.update(frame, []) is False


def test_track_reaping_caps_memory():
    """Many spurious face positions over time shouldn't grow tracks
    unboundedly — internal reaper caps at 16."""
    tracker = LipMotionTracker()
    frame = _make_frame()
    # Generate 40 different positions — should all create new tracks
    # but the reaper should keep total under 16.
    for i in range(40):
        face = _face_dict(cx=50 + i * 8, cy=180, mouth_y=230)
        # Make sure each face has unique-enough grid coords to start a
        # new track each time. The 8-px stride > 32-px grid only every
        # 4 calls, but that's fine for the test.
        tracker.update(frame, [face])
    # Force a reap.
    tracker._last_reap_at = 0.0
    tracker._maybe_reap()
    assert len(tracker._tracks) <= 16
