"""SpeechGate behavior — offline tests that simulate audio frames.

We don't need a real microphone to test the gate's *logic*: pump
hand-crafted PCM frames at it and observe which ones get yielded.
That covers the parts that have historically broken (calibration
window, threshold math, lip-motion boost, agent-speaking switch).

Run with: pytest tests/test_speech_gate.py
"""

import math

import numpy as np
import pytest

from app.elevenlabs.speech_gate import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SDK_RATE,
    SpeechGate,
)


class _FakeEngagement:
    """Stub for the EngagementState methods SpeechGate consults.
    Lets us flip is_engaged / is_lips_moving without instantiating a
    real EngagementState."""

    def __init__(self, engaged=True, lips=False):
        self.engaged = engaged
        self.lips = lips

    def is_engaged(self):
        return self.engaged

    def is_lips_moving(self):
        return self.lips


def _silent_frame() -> bytes:
    return b"\x00" * FRAME_BYTES


def _voiced_frame(amp: int = 8000) -> bytes:
    """One 20 ms frame of a 200 Hz tone — high enough to pass webrtcvad
    *and* well above any reasonable noise floor."""
    t = np.arange(FRAME_SAMPLES, dtype=np.float32) / SDK_RATE
    sig = (np.sin(2 * np.pi * 200 * t) * amp).astype(np.int16)
    return sig.tobytes()


def _drain(gate: SpeechGate, frame: bytes) -> list[bytes]:
    return list(gate.feed(frame))


def test_calibration_window_drops_audio(monkeypatch):
    """For the first NOVA_GATE_CALIB_MS, the gate should yield nothing
    even on loud frames — that's how we learn the noise floor."""
    monkeypatch.setenv("NOVA_GATE_CALIB_MS", "200")  # 10 frames @ 20 ms
    gate = SpeechGate(engagement=_FakeEngagement(engaged=True))
    yields = []
    for _ in range(8):
        yields.extend(_drain(gate, _voiced_frame()))
    assert yields == [], (
        "during calibration we MUST drop frames even when they're voiced"
    )


def test_post_calibration_voiced_frames_open_gate(monkeypatch):
    monkeypatch.setenv("NOVA_GATE_CALIB_MS", "60")     # ~3 frames
    monkeypatch.setenv("NOVA_GATE_IDLE_MS", "60")      # ~3 voiced frames to open
    monkeypatch.setenv("NOVA_GATE_PREROLL_MS", "40")
    gate = SpeechGate(engagement=_FakeEngagement(engaged=True))

    # Burn the calibration window with silent frames first so the
    # floor stays low and our voiced frames clearly exceed it.
    for _ in range(5):
        _drain(gate, _silent_frame())

    # Now feed voiced frames — gate should accumulate min_open_ms then
    # start yielding (with preroll, so the first yield contains
    # multiple frames worth).
    total_out = []
    for _ in range(12):
        total_out.extend(_drain(gate, _voiced_frame()))
    assert len(total_out) > 0, "gate should open within ~3 voiced frames"


def test_engagement_false_keeps_gate_closed(monkeypatch):
    monkeypatch.setenv("NOVA_GATE_CALIB_MS", "20")
    gate = SpeechGate(engagement=_FakeEngagement(engaged=False))
    for _ in range(20):
        out = _drain(gate, _voiced_frame())
    # With engagement disabled, no frames should ever flow through.
    # We can't assert the running total in a single line (each feed
    # returns its own list) so we rebuild by accumulation.
    yields = []
    for _ in range(20):
        yields.extend(_drain(gate, _voiced_frame()))
    assert yields == [], (
        "audio should not flow when no engaged face is present"
    )


def test_lip_motion_speeds_up_open(monkeypatch):
    """With lip-motion confirmation, the gate should open with FEWER
    voiced frames than the default threshold."""
    monkeypatch.setenv("NOVA_GATE_CALIB_MS", "20")
    monkeypatch.setenv("NOVA_GATE_IDLE_MS", "300")    # default-ish (15 frames)
    monkeypatch.setenv("NOVA_GATE_LIPS_SPEEDUP", "5")  # → 60 ms ≈ 3 frames
    monkeypatch.setenv("NOVA_GATE_PREROLL_MS", "0")

    gate_with_lips = SpeechGate(
        engagement=_FakeEngagement(engaged=True, lips=True)
    )
    gate_no_lips = SpeechGate(
        engagement=_FakeEngagement(engaged=True, lips=False)
    )

    # Burn calibration on both.
    for g in (gate_with_lips, gate_no_lips):
        for _ in range(2):
            _drain(g, _silent_frame())

    # Pump 5 voiced frames. With lips_speedup=5, the lips gate should
    # have opened by now and yielded at least one frame; the no-lips
    # gate should still be ramping (idle_min_ms=300, needs ~15).
    yields_lips = []
    yields_no_lips = []
    for _ in range(5):
        yields_lips.extend(_drain(gate_with_lips, _voiced_frame()))
        yields_no_lips.extend(_drain(gate_no_lips, _voiced_frame()))

    assert len(yields_lips) > len(yields_no_lips), (
        f"lip-motion should open the gate sooner — got "
        f"{len(yields_lips)} with lips vs {len(yields_no_lips)} without"
    )


def test_agent_speaking_switches_to_barge_in_threshold(monkeypatch):
    """While the agent is speaking, the gate uses NOVA_GATE_BARGE_MS
    instead of NOVA_GATE_IDLE_MS — a 6× higher threshold by default.
    A short voiced burst that would normally open the gate must NOT
    open it during agent speech."""
    monkeypatch.setenv("NOVA_GATE_CALIB_MS", "20")
    monkeypatch.setenv("NOVA_GATE_IDLE_MS", "60")      # ~3 voiced frames
    monkeypatch.setenv("NOVA_GATE_BARGE_MS", "600")    # ~30 voiced frames
    monkeypatch.setenv("NOVA_GATE_PREROLL_MS", "0")
    gate = SpeechGate(engagement=_FakeEngagement(engaged=True))

    # Calibrate.
    for _ in range(2):
        _drain(gate, _silent_frame())

    # Flag agent-speaking and feed a brief burst.
    gate.notify_agent_output()
    yields = []
    for _ in range(5):
        yields.extend(_drain(gate, _voiced_frame()))

    assert yields == [], (
        "brief audio burst during agent speech must NOT open the gate "
        "(barge-in threshold is much higher than idle)"
    )


def test_reset_returns_to_calibration(monkeypatch):
    monkeypatch.setenv("NOVA_GATE_CALIB_MS", "60")
    gate = SpeechGate(engagement=_FakeEngagement(engaged=True))
    # Burn calibration + open.
    for _ in range(20):
        _drain(gate, _voiced_frame())
    gate.reset()
    # Immediately after reset, calibration window restarts → first
    # voiced frame should be dropped (along with the next ones until
    # calibration completes).
    yielded = _drain(gate, _voiced_frame())
    assert yielded == [], (
        "after reset(), the gate should re-enter calibration and drop "
        "the next voiced frame"
    )
