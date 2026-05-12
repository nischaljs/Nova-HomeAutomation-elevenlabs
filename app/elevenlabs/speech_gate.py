"""Pi-side speech gate.

ElevenLabs runs its own VAD in the cloud and barges in on *anything* the
mic forwards. The vanilla DefaultAudioInterface forwards every PCM frame
as long as the WebSocket is open — so a breath, a sneeze, a fan, or a
side conversation across the room all cut Nova off mid-sentence.

This gate sits between the local microphone and the SDK, and only lets
audio through when it looks like real speech from the person standing in
front of the camera. It is intentionally mic-agnostic: every threshold
is expressed as a ratio over an adaptive noise floor, so the same code
behaves sanely on a quiet type-C earphone mic and on a hot condenser
podcast mic.

How it works (16 kHz / 20 ms frames):

  1. Every frame's RMS feeds an EMA noise-floor estimator that tracks
     the bottom 30 % of recent levels. After a 3 s calibration window
     at session start, this gives us a per-device "what is silence"
     reference.
  2. WebRTC VAD runs on the same frame. Aggressiveness 2 catches voice
     by spectral shape regardless of mic gain.
  3. A frame is "voiced" only if BOTH the VAD says yes AND its RMS is
     ≥ floor × gate_ratio.
  4. The gate doesn't *open* until enough consecutive voiced frames
     accumulate — `idle_min_ms` when Nova is silent, `barge_min_ms`
     when Nova is actively speaking (so short barge-in noises don't
     interrupt her). 1.5 s of clearly-voiced speech is the configured
     threshold for intentional barge-in.
  5. When the gate opens, a 250 ms pre-roll buffer is flushed first so
     the leading phoneme isn't lost.
  6. Once open, the gate stays open while voiced frames arrive and for
     a 200 ms hangover after they stop, so word endings aren't clipped.

Engagement gating: the SpeechGate consults a shared EngagementState. If
no face has been engaged with the camera for `disengage_grace_ms`, the
gate refuses to open at all — Nova doesn't react to talk that isn't
addressed to her.

Env tunables (all optional):
  NOVA_GATE_RATIO_IDLE   — multiplier over floor, idle mode (default 3.0)
  NOVA_GATE_RATIO_BARGE  — multiplier over floor, agent-speaking (default 5.0)
  NOVA_GATE_IDLE_MS      — voiced ms needed to open in idle (default 250)
  NOVA_GATE_BARGE_MS     — voiced ms needed to open while agent speaks (default 1500)
  NOVA_GATE_LIPS_SPEEDUP — divide the open-threshold by this when lip
                            motion is confirmed (default 3 — 250→83ms idle, 1500→500ms barge)
  NOVA_GATE_HANGOVER_MS  — trailing voiced-out frames kept open (default 200)
  NOVA_GATE_PREROLL_MS   — leading audio kept before gate opens (default 250)
  NOVA_GATE_CALIB_MS     — silent calibration window per session (default 3000)
  NOVA_MIC_PROFILE       — auto|sensitive|loud|quiet override (default auto)
  NOVA_GATE_DEBUG        — set to 1 to log gate decisions verbosely

Important: lip motion is a SPEED-UP, never a REQUIREMENT. If we never
see lips moving, the gate still opens — just on the default thresholds.
There is no scenario where a non-moving mouth blocks audio. The boost
only kicks in when we have visual *confirmation* the engaged user is
talking, in which case we trust the audio more aggressively.
"""

import collections
import math
import os
import time
from typing import Iterable, Optional

import numpy as np

SDK_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SDK_RATE * FRAME_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2  # int16

# Floor tracking — exponential moving "low-percentile" estimator. Each
# 20 ms frame's RMS is mixed into the running floor by a different alpha
# depending on whether it's above or below the current estimate, so the
# floor settles on the quiet samples instead of the loud ones.
FLOOR_DOWN_ALPHA = 0.05   # pulls floor down fast on quiet frames
FLOOR_UP_ALPHA = 0.001    # pulls floor up only very slowly on loud ones
MIN_FLOOR_RMS = 30.0      # absolute lower bound — int16 RMS rarely sane below this

_MIC_PROFILES = {
    # name: (idle_ratio, barge_ratio, idle_ms, barge_ms)
    "auto": None,
    "sensitive": (4.0, 6.0, 280, 1600),   # hot mic → higher ratio so we don't false-trigger
    "loud":      (4.0, 6.0, 280, 1600),
    "insensitive": (2.4, 3.5, 220, 1400),  # weak mic → lower ratio so we still pick up speech
    "quiet":     (2.4, 3.5, 220, 1400),
}


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


class _RingBytes:
    """Fixed-capacity 20 ms-frame ring buffer for pre-roll audio."""

    def __init__(self, max_frames: int):
        self._dq: collections.deque[bytes] = collections.deque(maxlen=max_frames)

    def push(self, frame_bytes: bytes):
        self._dq.append(frame_bytes)

    def drain(self) -> list[bytes]:
        out = list(self._dq)
        self._dq.clear()
        return out

    def clear(self):
        self._dq.clear()


class SpeechGate:
    """Block non-speech audio from reaching ElevenLabs.

    Construction is cheap (no native init). webrtcvad is imported lazily
    so the rest of the app loads on machines without it (the gate then
    falls back to RMS-only mode and prints a one-time warning).

    Usage:
        gate = SpeechGate(engagement=engagement_state)
        for chunk in gate.feed(pcm_16k_int16_bytes):
            forward_to_elevenlabs(chunk)
    """

    def __init__(self, engagement=None):
        self._engagement = engagement
        profile_name = os.environ.get("NOVA_MIC_PROFILE", "auto").lower()
        profile = _MIC_PROFILES.get(profile_name)

        if profile is None:
            self._idle_ratio = _envf("NOVA_GATE_RATIO_IDLE", 3.0)
            self._barge_ratio = _envf("NOVA_GATE_RATIO_BARGE", 5.0)
            self._idle_min_ms = _envi("NOVA_GATE_IDLE_MS", 250)
            self._barge_min_ms = _envi("NOVA_GATE_BARGE_MS", 1500)
        else:
            i_r, b_r, i_ms, b_ms = profile
            self._idle_ratio = _envf("NOVA_GATE_RATIO_IDLE", i_r)
            self._barge_ratio = _envf("NOVA_GATE_RATIO_BARGE", b_r)
            self._idle_min_ms = _envi("NOVA_GATE_IDLE_MS", i_ms)
            self._barge_min_ms = _envi("NOVA_GATE_BARGE_MS", b_ms)

        self._hangover_ms = _envi("NOVA_GATE_HANGOVER_MS", 200)
        self._preroll_ms = _envi("NOVA_GATE_PREROLL_MS", 250)
        self._calib_ms = _envi("NOVA_GATE_CALIB_MS", 3000)
        self._lips_speedup = max(1, _envi("NOVA_GATE_LIPS_SPEEDUP", 3))
        self._debug = os.environ.get("NOVA_GATE_DEBUG", "0") == "1"

        self._preroll = _RingBytes(self._preroll_ms // FRAME_MS)
        self._partial = b""

        self._vad = self._make_vad()
        self._floor = float(MIN_FLOOR_RMS)
        self._frames_seen = 0
        self._calib_frames_needed = self._calib_ms // FRAME_MS
        self._calibrated = False

        self._voiced_run_ms = 0
        self._hangover_left_ms = 0
        self._is_open = False

        self._is_agent_speaking = False
        self._last_agent_output_at = 0.0
        # If the SDK hasn't called output() in the last AGENT_QUIET_AFTER_MS,
        # we assume Nova has stopped speaking. ElevenLabs streams audio
        # in small chunks so output() fires roughly every 60–100 ms while
        # she's speaking.
        self._agent_quiet_after_s = 0.35

        # Stats for periodic log lines so we can debug "she stopped
        # responding" in the field without re-enabling NOVA_GATE_DEBUG.
        self._frames_total = 0
        self._frames_passed = 0
        self._opens = 0
        self._opens_with_lips = 0
        self._last_stats_at = time.monotonic()

        print(f"[GATE] profile={profile_name} idle_ratio={self._idle_ratio:.1f} "
              f"barge_ratio={self._barge_ratio:.1f} idle_min={self._idle_min_ms}ms "
              f"barge_min={self._barge_min_ms}ms preroll={self._preroll_ms}ms "
              f"hangover={self._hangover_ms}ms calib={self._calib_ms}ms")

    def _make_vad(self):
        try:
            import webrtcvad
            v = webrtcvad.Vad(2)
            print("[GATE] webrtcvad loaded (aggressiveness=2)")
            return v
        except ImportError:
            print("[GATE] webrtcvad missing — RMS-only fallback "
                  "(install with: pip install webrtcvad)")
            return None

    def notify_agent_output(self):
        """Called by the audio interface every time the SDK writes a
        speaker chunk. Tracks whether Nova is currently speaking so the
        gate can apply the stricter barge-in threshold."""
        self._last_agent_output_at = time.monotonic()
        if not self._is_agent_speaking:
            self._is_agent_speaking = True
            if self._debug:
                print("[GATE] agent started speaking — barge-in mode")

    def _refresh_agent_state(self):
        if not self._is_agent_speaking:
            return
        if time.monotonic() - self._last_agent_output_at > self._agent_quiet_after_s:
            self._is_agent_speaking = False
            if self._debug:
                print("[GATE] agent quiet — back to idle gate")

    def _is_engaged(self) -> bool:
        if self._engagement is None:
            return True
        try:
            return bool(self._engagement.is_engaged())
        except Exception:
            return True

    def _lips_moving_now(self) -> bool:
        """Visual confirmation that the engaged user's lips moved
        recently. Used to lower the gate's open-threshold — pure
        speed-up, never a requirement. Falls back to False if the
        engagement object doesn't expose lip-motion (older field
        deployments or tests with stubbed engagement)."""
        if self._engagement is None:
            return False
        try:
            return bool(self._engagement.is_lips_moving())
        except Exception:
            return False

    def feed(self, pcm: bytes) -> Iterable[bytes]:
        """Push raw 16-bit mono 16 kHz PCM in any chunk size. Yields the
        chunks that should be forwarded to the SDK (already aligned to
        20 ms frames). Yields nothing while the gate is closed."""
        self._refresh_agent_state()

        # Re-frame to exact 20 ms boundaries so webrtcvad is happy.
        buf = self._partial + pcm
        n_full = len(buf) // FRAME_BYTES
        if n_full == 0:
            self._partial = buf
            return
        consumed = n_full * FRAME_BYTES
        self._partial = buf[consumed:]
        frames_bytes = buf[:consumed]

        for i in range(n_full):
            frame = frames_bytes[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]
            yield from self._process_frame(frame)

    def _process_frame(self, frame: bytes):
        self._frames_seen += 1
        self._frames_total += 1
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        # RMS in int16 units (0..32767). Add a tiny epsilon so log/ratio
        # math never divides by zero when the input is hard-zero.
        rms = float(math.sqrt(float(np.mean(samples * samples)) + 1.0))
        self._update_floor(rms)

        if not self._calibrated:
            if self._frames_seen >= self._calib_frames_needed:
                self._calibrated = True
                print(f"[GATE] calibrated: noise_floor≈{self._floor:.0f} "
                      f"(int16 RMS) after {self._frames_seen} frames "
                      f"({self._frames_seen * FRAME_MS}ms)")
            # During calibration we keep the gate closed regardless —
            # this also gives the user time to stop adjusting their
            # mic/chair without yelling at Nova.
            self._preroll.push(frame)
            return

        if not self._is_engaged():
            # Nobody facing the camera — never open the gate. Still
            # update the floor so we re-calibrate during quiet periods.
            self._preroll.push(frame)
            if self._is_open:
                self._is_open = False
                self._voiced_run_ms = 0
                self._hangover_left_ms = 0
            return

        vad_voiced = self._vad_says_voiced(frame)
        ratio = self._current_ratio()
        rms_voiced = rms >= self._floor * ratio
        is_voiced = vad_voiced and rms_voiced

        # Lip-motion boost — when we visually confirm the engaged user's
        # lips are moving, trust the audio enough to open the gate
        # ~3× faster. This is the difference between a "feels instant"
        # 80 ms idle open and a "noticeable" 250 ms one, and between a
        # 500 ms barge-in and a 1500 ms one. Pure speed-up: no lip
        # motion → fall back to default thresholds, gate still opens.
        base_min = self._barge_min_ms if self._is_agent_speaking else self._idle_min_ms
        lips_boost = self._lips_moving_now()
        min_open_ms = base_min // self._lips_speedup if lips_boost else base_min

        if is_voiced:
            self._voiced_run_ms += FRAME_MS
            self._hangover_left_ms = self._hangover_ms

            if not self._is_open and self._voiced_run_ms >= min_open_ms:
                self._is_open = True
                self._opens += 1
                if lips_boost:
                    self._opens_with_lips += 1
                if self._debug:
                    print(f"[GATE] open (voiced_run={self._voiced_run_ms}ms "
                          f"min={min_open_ms}ms lips_boost={lips_boost} "
                          f"agent_speaking={self._is_agent_speaking} "
                          f"rms={rms:.0f} floor={self._floor:.0f} ratio={ratio:.1f})")
                for chunk in self._preroll.drain():
                    yield chunk
                yield frame
                self._maybe_log_stats()
                return

            if self._is_open:
                self._frames_passed += 1
                yield frame
                self._maybe_log_stats()
                return

            # Building toward opening — keep frame in pre-roll so we
            # have a smooth onset when the gate finally opens.
            self._preroll.push(frame)
            return

        # Not voiced this frame.
        if self._is_open:
            self._hangover_left_ms -= FRAME_MS
            if self._hangover_left_ms > 0:
                self._frames_passed += 1
                yield frame
                self._maybe_log_stats()
                return
            self._is_open = False
            self._voiced_run_ms = 0
            if self._debug:
                print("[GATE] close (hangover elapsed)")
            self._maybe_log_stats()
            # fall through — frame is non-voiced and gate closed, stash it
        else:
            # Decay the running voiced counter so a single voiced frame
            # surrounded by silence doesn't accumulate toward open.
            self._voiced_run_ms = max(0, self._voiced_run_ms - FRAME_MS)

        self._preroll.push(frame)

    def _vad_says_voiced(self, frame: bytes) -> bool:
        if self._vad is None:
            return True  # RMS-only mode: floor check does the work
        try:
            return self._vad.is_speech(frame, SDK_RATE)
        except Exception as e:
            print(f"[GATE] webrtcvad.is_speech raised {type(e).__name__}: {e} "
                  f"— falling back to RMS-only for this frame")
            return True

    def _current_ratio(self) -> float:
        return self._barge_ratio if self._is_agent_speaking else self._idle_ratio

    def _update_floor(self, rms: float):
        """Asymmetric EMA — falls quickly, rises slowly, so the floor
        settles on the quiet samples even in a room with intermittent
        bursts of speech."""
        if rms < self._floor:
            self._floor = (1 - FLOOR_DOWN_ALPHA) * self._floor + FLOOR_DOWN_ALPHA * rms
        else:
            self._floor = (1 - FLOOR_UP_ALPHA) * self._floor + FLOOR_UP_ALPHA * rms
        if self._floor < MIN_FLOOR_RMS:
            self._floor = MIN_FLOOR_RMS

    def _maybe_log_stats(self):
        now = time.monotonic()
        if now - self._last_stats_at < 30.0:
            return
        elapsed_s = now - self._last_stats_at
        total = max(1, self._frames_total)
        pass_pct = 100.0 * self._frames_passed / total
        print(f"[GATE] stats: {pass_pct:.0f}% frames passed, "
              f"{self._opens} opens ({self._opens_with_lips} with lip-boost) "
              f"in {elapsed_s:.0f}s, "
              f"floor≈{self._floor:.0f}, agent_speaking={self._is_agent_speaking}")
        self._frames_total = 0
        self._frames_passed = 0
        self._opens = 0
        self._opens_with_lips = 0
        self._last_stats_at = now

    def reset(self):
        """Drop in-flight state — call when a new session opens."""
        self._partial = b""
        self._preroll.clear()
        self._voiced_run_ms = 0
        self._hangover_left_ms = 0
        self._is_open = False
        self._frames_seen = 0
        self._calibrated = False
        self._floor = float(MIN_FLOOR_RMS)
        print("[GATE] reset — re-entering calibration window")
