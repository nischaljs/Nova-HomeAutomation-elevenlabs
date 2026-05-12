"""Cross-subsystem 'is someone engaged with the camera?' signal.

Three independent threads need to know whether the person in front of
the camera is actively engaging with Nova:

  * the SpeechGate (audio thread, ~50 calls/sec) — refuses to forward
    audio when nobody's there
  * the orchestrator (asyncio main loop) — opens/closes the ElevenLabs
    session based on engaged-for / disengaged-for windows
  * the FaceMonitor (asyncio task, ~2 calls/sec) — the *producer*; it
    publishes the signal after each face recognition poll

Reading this signal must be cheap and lock-free (the audio callback
hates locks), so the state is held as two atomic-ish floats — last seen
timestamps for "engaged" and "any face visible". Readers compute
booleans from them. Writers always overwrite, never read-modify-write,
so we don't need the GIL to be a real lock for this to be correct
enough.

Definitions:
  engaged    — at least one face is visible AND it's roughly looking at
               the camera (pose_asym ≤ ENGAGED_MAX_POSE_ASYM)
  present    — at least one face is visible at any angle
  is_engaged() — True while *engaged* has been seen within the last
               STICKY_S seconds, so brief look-aways don't immediately
               close the gate
  engaged_for_s() — how long the engaged state has been continuous
  disengaged_for_s() — inverse, for the orchestrator's idle-close timer
"""

import os
import time

# Pose-asymmetry threshold from the face quality gate at which we say
# "they're looking at me." Looser than the face-rec gate so we accept
# casual eye contact, not just stare-into-camera.
ENGAGED_MAX_POSE_ASYM = float(os.environ.get("NOVA_ENGAGED_MAX_ASYM", "0.45"))
# How long a "look away" can last before the audio gate stops counting
# us as engaged. A short value would chop Nova off when the user
# glances down to think — 2.5 s is enough for casual thinking pauses
# without keeping audio open for someone who's clearly turned away.
STICKY_S = float(os.environ.get("NOVA_ENGAGED_STICKY_S", "2.5"))


class EngagementState:
    def __init__(self):
        self._t_last_engaged: float = 0.0
        self._t_last_present: float = 0.0
        self._t_last_lips_moving: float = 0.0
        self._t_engaged_since: float = 0.0
        # Cached so we don't recompute "engaged?" on every audio frame.
        self._was_engaged: bool = False
        # Periodic stats: how much of the last window we were present /
        # engaged. Counted in detect ticks, not seconds, so 100 ticks of
        # presence over 30 s reads as 100/200 = 50% if VisionPipeline
        # is running at the default 6.6 Hz.
        self._ticks_total = 0
        self._ticks_present = 0
        self._ticks_engaged = 0
        self._ticks_lips_moving = 0
        self._flips = 0
        self._last_stats_at: float = 0.0
        self._verbose = os.environ.get("NOVA_ENGAGE_DEBUG", "0") == "1"
        # Most recent asym from the *best* face this tick — kept around
        # purely for diagnostic logging, not used in the engagement
        # decision. Useful for "why did this flip just now?".
        self._last_asym: float | None = None

    def update(
        self,
        *,
        present: bool,
        engaged: bool,
        asym: float | None = None,
        lips_moving: bool = False,
    ):
        """Called by VisionPipeline after each detect cycle.

        `present`: at least one face visible at any angle.
        `engaged`: at least one face roughly looking at the camera.
        `asym`: best (lowest) pose-asymmetry value seen this tick, or
                None if no face has landmarks. Diagnostic-only — does
                not affect the engagement decision.
        """
        now = time.monotonic()
        self._ticks_total += 1
        if present:
            self._t_last_present = now
            self._ticks_present += 1
        if asym is not None:
            self._last_asym = asym
        if lips_moving:
            self._t_last_lips_moving = now
            self._ticks_lips_moving += 1
        if engaged:
            self._t_last_engaged = now
            self._ticks_engaged += 1
            if not self._was_engaged:
                self._t_engaged_since = now
                self._was_engaged = True
                self._flips += 1
                print(f"[ENGAGE] engaged TRUE  "
                      f"(asym={asym if asym is not None else 'n/a'}, "
                      f"wake-up clock starts)")
        else:
            if self._was_engaged and (now - self._t_last_engaged) > STICKY_S:
                self._was_engaged = False
                self._flips += 1
                eng_for = now - self._t_engaged_since
                print(f"[ENGAGE] engaged FALSE "
                      f"(asym={asym if asym is not None else 'n/a'}, "
                      f"was engaged for {eng_for:.1f}s)")
        if self._verbose and present:
            asym_s = f"{asym:.2f}" if asym is not None else "n/a"
            print(f"[ENGAGE] tick: present={present} engaged={engaged} asym={asym_s} "
                  f"sticky_engaged={self._was_engaged}")
        self._maybe_log_stats(now)

    def _maybe_log_stats(self, now: float):
        if self._last_stats_at == 0.0:
            self._last_stats_at = now
            return
        if now - self._last_stats_at < 30.0:
            return
        elapsed = now - self._last_stats_at
        total = max(1, self._ticks_total)
        pres_pct = 100.0 * self._ticks_present / total
        eng_pct = 100.0 * self._ticks_engaged / total
        lip_pct = 100.0 * self._ticks_lips_moving / total
        asym_s = f"{self._last_asym:.2f}" if self._last_asym is not None else "n/a"
        print(f"[ENGAGE] stats: present={pres_pct:.0f}% engaged={eng_pct:.0f}% "
              f"lips_moving={lip_pct:.0f}% flips={self._flips} ticks={total} "
              f"last_asym={asym_s} over {elapsed:.0f}s")
        self._last_stats_at = now
        self._ticks_total = 0
        self._ticks_present = 0
        self._ticks_engaged = 0
        self._ticks_lips_moving = 0
        self._flips = 0

    def is_engaged(self) -> bool:
        """Hot path — called from the audio callback. Keep cheap."""
        if self._t_last_engaged <= 0:
            return False
        return (time.monotonic() - self._t_last_engaged) <= STICKY_S

    def is_present(self) -> bool:
        if self._t_last_present <= 0:
            return False
        return (time.monotonic() - self._t_last_present) <= STICKY_S

    def engaged_for_s(self) -> float:
        if not self.is_engaged():
            return 0.0
        return time.monotonic() - self._t_engaged_since

    def presence_lost_for_s(self) -> float:
        """Time since *any* face was last seen in frame (engaged or not).

        This is the timer that fires when the user actually walks away.
        A 9-second silent pause with the user still visible reads ~0.15 s
        here, not 9 — every camera detection tick refreshes the
        timestamp regardless of head pose.

        Returns +inf at boot so the lifecycle code reads 'long enough,
        keep session closed' until at least one face has been seen.
        """
        if self._t_last_present <= 0:
            return float("inf")
        return time.monotonic() - self._t_last_present

    def disengaged_for_s(self) -> float:
        """Time since we last saw an *engaged* face (looking at camera).

        Distinct from presence_lost_for_s: this counts up when the user
        is in frame but has been turned away for a while. We use it as
        a fallback close trigger so a session doesn't stay live forever
        billing minutes when someone's just sitting in the room with
        their face visible but never engaging with Nova.
        """
        if self._t_last_engaged <= 0:
            return float("inf")
        return time.monotonic() - self._t_last_engaged

    def is_lips_moving(self) -> bool:
        """True if the visual lip-motion tracker observed motion in the
        last ~0.6 s. Used by the SpeechGate to open ~3× faster when we
        can visually confirm the engaged user is speaking — a much
        stronger signal than 'audio energy is up' alone."""
        if self._t_last_lips_moving <= 0:
            return False
        from app.face.lip_motion import LIP_MOTION_STICKY_S
        return (time.monotonic() - self._t_last_lips_moving) <= LIP_MOTION_STICKY_S


ENGAGEMENT: EngagementState | None = None


def get_engagement() -> EngagementState:
    global ENGAGEMENT
    if ENGAGEMENT is None:
        ENGAGEMENT = EngagementState()
    return ENGAGEMENT
