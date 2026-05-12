import asyncio
import os
import threading
import time

from app.elevenlabs.adaptive_audio import AdaptiveDefaultAudioInterface
from app.elevenlabs.context import face_info_to_context_text, face_state_to_context_text
from app.elevenlabs.fallback import ensure_fallback_wav, play_fallback_blocking
from app.elevenlabs.tools import build_client_tools
from app.orchestration.net_probe import get_net_stats


# How long to debounce duplicate contextual updates so we don't spam
# ElevenLabs every poll cycle when the face state hasn't actually changed.
CONTEXT_THROTTLE_S = 2.0

# Backoff used when an active session unexpectedly dies (network blip,
# WS reset). Engagement-gated mode reopens it as long as a face is still
# present.
RESTART_DELAY_S = 1.5
RESTART_BACKOFF_MAX_S = 30.0
EARLY_FAIL_THRESHOLD_S = 3.0

# Engagement-driven open/close thresholds.
#
# Open: how long an engaged face must persist before we spin up a
# session. Low enough that walking up feels instant, high enough that
# someone walking past doesn't accidentally open a session.
ENGAGE_OPEN_AFTER_S = float(os.environ.get("NOVA_ENGAGE_OPEN_S", "2.0"))
#
# Two close conditions, either triggers a tear-down:
#
#   PRESENCE_CLOSE_AFTER_S — face has been entirely gone from frame for
#     this long. Covers "user walked away" plus brief occlusions
#     (bending down, hand-on-face, single missed detection). 12 s is
#     well above the natural "stand back to think" pause and well
#     below "they're clearly gone".
#
#   DISENGAGE_CLOSE_AFTER_S — face is *in* frame but persistently
#     turned away for this long. Covers "user sat down at their desk
#     and is just working" — we don't want to bill ElevenLabs minutes
#     forever for someone who is never going to talk to us. 60 s is
#     generous enough that a real conversation pause doesn't trip it.
PRESENCE_CLOSE_AFTER_S = float(os.environ.get("NOVA_PRESENCE_CLOSE_S", "12.0"))
DISENGAGE_CLOSE_AFTER_S = float(os.environ.get("NOVA_DISENGAGE_CLOSE_S", "60.0"))

# Seconds of failed reconnect (during an engaged session) before we
# play the offline "be right back" WAV. Below this we say nothing —
# typical brief WS resets reconnect in <2 s and the user wouldn't
# notice; the fallback is for *visible* outages.
FALLBACK_AFTER_S = float(os.environ.get("NOVA_FALLBACK_AFTER_S", "5.0"))

# Loop period for the lifecycle thread. Fast enough that wake-up feels
# instant from the user's perspective; slow enough that we're not
# burning a core.
LIFECYCLE_TICK_S = 0.25


class ElevenLabsAgent:
    """Engagement-gated ElevenLabs Conversation wrapper.

    Earlier design opened one session at boot and held it forever,
    auto-restarting on idle close. That meant Nova was billed for
    ElevenLabs minutes 24/7 and that every random noise in the room was
    a potential interruption.

    New design:
      * No session at boot. The lifecycle thread waits for the
        EngagementState to report a face that's been engaged for
        ENGAGE_OPEN_AFTER_S, then opens a session.
      * Session closes after ENGAGE_CLOSE_AFTER_S of no present face
        (allowing for brief look-aways via EngagementState's stickiness).
      * Queued contextual updates flush automatically once the session
        opens, so the agent knows who's in front of it before saying its
        first word.
      * Auth/quota errors (401/403/429) short-circuit the retry loop —
        no point burning attempts when the API is going to keep saying
        no.
    """

    def __init__(self, api_key: str, agent_id: str, engagement=None):
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY missing — set it in .env")
        if not agent_id:
            raise RuntimeError("ELEVENLABS_AGENT_ID missing — set it in .env")
        self._api_key = api_key
        self._agent_id = agent_id
        self._engagement = engagement
        self._audio_interface: AdaptiveDefaultAudioInterface | None = None

        self._conversation = None
        self._conv_lock = threading.Lock()

        self._lifecycle_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()

        self._session_count = 0
        self._session_started_at: float = 0.0
        self._restart_backoff = RESTART_DELAY_S
        # Tracks when the most recent live session ended — used to
        # decide whether to play the offline fallback WAV during a
        # protracted reconnect. Set on _on_session_died, cleared once
        # the next session opens.
        self._session_ended_at: float | None = None
        self._fallback_played_for_outage = False
        # Pre-generate the offline WAV on startup so the first outage
        # doesn't have to wait for espeak. Returns None if espeak-ng
        # isn't installed — we just won't have a fallback then.
        self._fallback_wav = ensure_fallback_wav()

        # Outgoing context — last text + when we last sent it, used to
        # throttle duplicate updates. Pending texts are flushed the
        # moment a session becomes available.
        self._last_context_text: str = ""
        self._last_context_at: float = 0.0
        self._pending_context: str = ""
        self._context_lock = threading.Lock()

        # Permanent-failure flag — set when we hit 401/403/429 etc.
        # Stops the lifecycle loop from burning attempts in a fast loop
        # against an API that will keep saying no.
        self._permanent_failure_reason: str | None = None

    # ------------------------------------------------------------------
    # public lifecycle (called from Orchestrator)
    # ------------------------------------------------------------------

    async def start(self):
        """Kick off the lifecycle thread. Does NOT open a session yet —
        the thread will, the moment the EngagementState reports an
        engaged face."""
        if self._engagement is None:
            print("[AGENT] starting in always-on mode (no engagement signal)")
        else:
            print(f"[AGENT] starting in engagement-gated mode "
                  f"(open after {ENGAGE_OPEN_AFTER_S}s engaged, "
                  f"close after {PRESENCE_CLOSE_AFTER_S}s no face in frame "
                  f"or {DISENGAGE_CLOSE_AFTER_S}s in-frame-but-turned-away)")
        self._stop_requested.clear()
        self._lifecycle_thread = threading.Thread(
            target=self._lifecycle_loop, daemon=True, name="agent-lifecycle"
        )
        self._lifecycle_thread.start()

    async def stop(self):
        print("[AGENT] stop requested — closing any active session")
        self._stop_requested.set()
        await asyncio.to_thread(self._end_session_blocking)
        if self._lifecycle_thread is not None:
            await asyncio.to_thread(self._lifecycle_thread.join, 5.0)
        print("[AGENT] stopped")

    # ------------------------------------------------------------------
    # context updates (called from FaceMonitor / Orchestrator)
    # ------------------------------------------------------------------

    def push_face_context(self, face_info: dict | None):
        text = face_info_to_context_text(face_info)
        self._send_context(text)

    def push_face_state(self, state: dict):
        text = face_state_to_context_text(state)
        self._send_context(text)

    def _send_context(self, text: str):
        if not text:
            return
        now = time.monotonic()
        if text == self._last_context_text and (now - self._last_context_at) < CONTEXT_THROTTLE_S:
            return
        self._last_context_text = text
        self._last_context_at = now

        with self._conv_lock:
            conv = self._conversation

        if conv is None:
            # Queue for after-session-open. We only keep the *latest*
            # — face state is point-in-time, so an older pending text
            # would be misleading by the time the session opens.
            with self._context_lock:
                self._pending_context = text
            print(f"[AGENT] no active session — context queued: {text[:120]}")
            return

        try:
            conv.send_contextual_update(text)
            print(f"[AGENT] contextual_update sent: {text[:120]}")
        except Exception as e:
            print(f"[AGENT] contextual_update failed: {type(e).__name__}: {e}")

    def _flush_pending_context(self):
        with self._context_lock:
            pending = self._pending_context
            self._pending_context = ""
        if not pending:
            return
        with self._conv_lock:
            conv = self._conversation
        if conv is None:
            # Race — session went away between flush and acquire. Re-queue.
            with self._context_lock:
                if not self._pending_context:
                    self._pending_context = pending
            return
        try:
            conv.send_contextual_update(pending)
            print(f"[AGENT] flushed queued context: {pending[:120]}")
        except Exception as e:
            print(f"[AGENT] flush context failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # session open / close (run on the lifecycle thread)
    # ------------------------------------------------------------------

    def _should_be_open(self) -> bool:
        """Engagement policy. Returns True while a face has been engaged
        long enough to justify having a live ElevenLabs session.

        Two independent close conditions, either fires:
          * presence_lost_for_s ≥ PRESENCE_CLOSE_AFTER_S — face has been
            completely gone from frame too long (user walked away)
          * disengaged_for_s ≥ DISENGAGE_CLOSE_AFTER_S — face is visible
            but persistently turned away (user is in the room but not
            talking to us). Generous default (60 s) so a real
            conversational pause doesn't trip it.
        """
        if self._engagement is None:
            return True  # always-on fallback
        with self._conv_lock:
            is_open = self._conversation is not None
        if is_open:
            if self._engagement.presence_lost_for_s() >= PRESENCE_CLOSE_AFTER_S:
                return False
            if self._engagement.disengaged_for_s() >= DISENGAGE_CLOSE_AFTER_S:
                return False
            return True
        return self._engagement.engaged_for_s() >= ENGAGE_OPEN_AFTER_S

    def _lifecycle_loop(self):
        """One loop, three states:
            * closed + engagement says open → open it
            * open + WS still alive → wait
            * open + WS died → handle backoff
            * open + engagement says close → close it
        """
        while not self._stop_requested.is_set():
            if self._permanent_failure_reason:
                # Auth/quota error — sleep and retry once a minute so a
                # transient outage isn't permanently fatal, but we don't
                # burn the user's quota in a tight loop.
                time.sleep(60.0)
                self._permanent_failure_reason = None
                continue

            want_open = self._should_be_open()
            have_open = self._conversation is not None

            if want_open and not have_open:
                # If we're reopening after an unexpected close AND the
                # user is still engaged AND it's been >5 s, play the
                # offline fallback so they know something's happening.
                # Only once per outage — _fallback_played_for_outage
                # resets when a session actually comes back up.
                self._maybe_play_fallback()

                if self._engagement is not None:
                    eng_for = self._engagement.engaged_for_s()
                    print(f"[AGENT] engage policy → opening session "
                          f"(engaged_for={eng_for:.1f}s)")
                self._try_open_session()
                # Fall through to next iteration — wait for WS to either
                # die or close-policy to fire.

            elif want_open and have_open:
                # WS is supposed to be live; check whether it's actually
                # still going. wait_for_session_end blocks until the
                # SDK's monitor thread reports the WS is closed, so we
                # don't poll.
                self._wait_for_ws_or_close_policy()

            elif have_open and not want_open:
                # Log which of the two close timers fired so we can
                # tune them in the field without instrumenting deeper.
                if self._engagement is not None:
                    presence = self._engagement.presence_lost_for_s()
                    disengage = self._engagement.disengaged_for_s()
                    if presence >= PRESENCE_CLOSE_AFTER_S:
                        reason = f"no face in frame for {presence:.0f}s"
                    elif disengage >= DISENGAGE_CLOSE_AFTER_S:
                        reason = f"turned away for {disengage:.0f}s"
                    else:
                        reason = "engagement policy"
                else:
                    reason = "stop requested"
                print(f"[AGENT] closing session #{self._session_count} — {reason}")
                self._end_session_blocking()

            else:
                # closed and policy says stay closed — short sleep, then
                # re-check engagement.
                time.sleep(LIFECYCLE_TICK_S)

        # On stop: make sure WS is closed before the thread exits.
        if self._conversation is not None:
            self._end_session_blocking()

    def _wait_for_ws_or_close_policy(self):
        """Wait either for the WS to die OR for the disengage policy to
        fire. We can't actually block on both at once, so we poll
        engagement every LIFECYCLE_TICK_S while watching for the WS to
        end via the SDK. In practice the WS rarely dies on its own
        within the disengage window."""
        deadline_check = time.monotonic()
        while not self._stop_requested.is_set():
            conv = self._conversation
            if conv is None:
                return  # already closed somehow

            # The SDK's wait_for_session_end is blocking-with-no-timeout
            # — we can't peek. So we test policy every tick and only
            # block briefly. If the SDK closes the WS during the sleep,
            # the next loop iteration will see _conversation is None
            # (because the close handler nulls it).
            now = time.monotonic()
            if now - deadline_check >= LIFECYCLE_TICK_S:
                deadline_check = now
                if not self._should_be_open():
                    return  # policy says close — outer loop will handle

            # Cheap "is WS dead" check: ask the SDK's monitor thread
            # state. The SDK doesn't expose this cleanly, so we just
            # sleep and let the outer loop notice on the next pass.
            time.sleep(LIFECYCLE_TICK_S)

            # Detect dead WS via the conversation's internal state if
            # exposed; otherwise rely on the SDK's wait_for_session_end
            # being callable from another short-lived thread. To keep
            # the implementation portable across SDK minors, we just
            # poll a single attribute below.
            if not self._is_conversation_alive(conv):
                self._on_session_died(unexpected=True)
                return

    def _is_conversation_alive(self, conv) -> bool:
        """Best-effort liveness probe. The SDK doesn't expose this in a
        stable way across versions, so we check several attributes and
        fall back to assuming alive when we can't tell."""
        for attr in ("_session_ended", "session_ended", "_closed", "is_closed"):
            try:
                v = getattr(conv, attr, None)
                if callable(v):
                    v = v()
                if isinstance(v, bool) and v:
                    return False
            except Exception:
                continue
        return True

    def _try_open_session(self):
        try:
            self._start_session_blocking()
            self._restart_backoff = RESTART_DELAY_S
            # Clean up any active-outage state — we're back online.
            self._session_ended_at = None
            self._fallback_played_for_outage = False
            # First thing after open: flush any queued face context so
            # Nova knows who she's looking at before she greets them.
            self._flush_pending_context()
        except Exception as e:
            classified = self._classify_error(e)
            if classified == "permanent":
                print(f"[AGENT] permanent failure: {type(e).__name__}: {e} "
                      f"— will retry in 60 s")
                self._permanent_failure_reason = str(e)
            else:
                self._restart_backoff = min(
                    self._restart_backoff * 2, RESTART_BACKOFF_MAX_S
                )
                print(f"[AGENT] open failed: {type(e).__name__}: {e} "
                      f"— next attempt in {self._restart_backoff:.1f}s")
                time.sleep(self._restart_backoff)

    def _on_session_died(self, unexpected: bool):
        if self._stop_requested.is_set():
            return
        session_duration = time.monotonic() - self._session_started_at
        if unexpected and session_duration < EARLY_FAIL_THRESHOLD_S:
            self._restart_backoff = min(
                self._restart_backoff * 2, RESTART_BACKOFF_MAX_S
            )
            print(f"[AGENT] session died after only {session_duration:.1f}s "
                  f"— back off, next attempt in {self._restart_backoff:.1f}s")
            time.sleep(self._restart_backoff)
        else:
            print(f"[AGENT] session ended after {session_duration:.0f}s")
        with self._conv_lock:
            self._conversation = None
        # Arm the fallback timer — if it takes >FALLBACK_AFTER_S to
        # reopen and the user is still engaged, the next lifecycle
        # tick will play the WAV.
        self._session_ended_at = time.monotonic()
        self._fallback_played_for_outage = False

    def _maybe_play_fallback(self):
        if self._fallback_wav is None or self._fallback_played_for_outage:
            return
        if self._session_ended_at is None:
            return
        # Only play if the outage has lasted long enough that the user
        # would actually be confused by the silence, and only while the
        # user is still engaged (no point talking to an empty room).
        outage_s = time.monotonic() - self._session_ended_at
        if outage_s < FALLBACK_AFTER_S:
            return
        if self._engagement is not None and not self._engagement.is_engaged():
            return
        print(f"[AGENT] outage {outage_s:.0f}s with engaged user — "
              f"playing offline fallback WAV")
        try:
            ok = play_fallback_blocking(self._fallback_wav)
            self._fallback_played_for_outage = True
            if not ok:
                print("[AGENT] fallback playback returned False")
        except Exception as e:
            print(f"[AGENT] fallback playback raised: "
                  f"{type(e).__name__}: {e}")

    def _classify_error(self, e: Exception) -> str:
        msg = str(e).lower()
        if any(s in msg for s in ("401", "403", "429", "unauthorized",
                                  "forbidden", "quota", "rate limit")):
            return "permanent"
        return "transient"

    def _start_session_blocking(self):
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation

        client = ElevenLabs(api_key=self._api_key)
        print(f"[AGENT] opening session #{self._session_count + 1} to ElevenLabs ...")
        # New AdaptiveDefaultAudioInterface per session so the SpeechGate
        # gets a fresh calibration window for each fresh conversation.
        # The old interface (if any) is torn down by _end_session_blocking
        # before we get here.
        self._audio_interface = AdaptiveDefaultAudioInterface(
            engagement=self._engagement
        )
        net_stats = get_net_stats()

        def _on_user_transcript(text: str):
            # Arming first-audio latency capture — the next audio chunk
            # we receive from ElevenLabs will be the agent's reply
            # opening, so the gap is end-to-end STT + LLM + TTS-first-
            # byte. Reported in the periodic [NET] log line.
            net_stats.note_user_transcript()
            print(f"[USER] {text}")

        with self._conv_lock:
            self._conversation = Conversation(
                client,
                self._agent_id,
                requires_auth=True,
                audio_interface=self._audio_interface,
                client_tools=build_client_tools(),
                callback_agent_response=lambda r: print(f"[AGENT] {r}"),
                callback_user_transcript=_on_user_transcript,
            )
            self._conversation.start_session()
            self._session_count += 1
            self._session_started_at = time.monotonic()
        print(f"[AGENT] ✓ session #{self._session_count} live")

    def _end_session_blocking(self):
        with self._conv_lock:
            conv = self._conversation
            self._conversation = None
        if conv is None:
            return
        try:
            conv.end_session()
        except Exception as e:
            print(f"[AGENT] end_session error (ignored): {type(e).__name__}: {e}")
        # Reset the gate so the next session starts with a fresh
        # calibration window — important because the user may have
        # walked off (mic floor dropped) or a new person walked up
        # (different voice levels).
        try:
            if self._audio_interface is not None:
                self._audio_interface.gate.reset()
        except Exception as e:
            print(f"[AGENT] gate reset failed (ignored): {type(e).__name__}: {e}")
