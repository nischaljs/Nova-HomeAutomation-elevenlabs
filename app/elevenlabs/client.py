import asyncio
import threading
import time

from app.elevenlabs.adaptive_audio import AdaptiveDefaultAudioInterface
from app.elevenlabs.context import face_info_to_context_text, face_state_to_context_text
from app.elevenlabs.tools import build_client_tools


CONTEXT_THROTTLE_S = 2.0
RESTART_DELAY_S = 1.5
RESTART_BACKOFF_MAX_S = 30.0
# If a session ends within this many seconds of starting, treat it as a
# hard failure (audio device missing, etc.) rather than a normal idle
# timeout. Triggers exponential backoff so we don't burn ElevenLabs
# connection attempts looping at 1.5s intervals.
EARLY_FAIL_THRESHOLD_S = 3.0


class ElevenLabsAgent:
    """Persistent ElevenLabs Conversation wrapper.

    The agent runs forever — when ElevenLabs hangs up the session (silence
    timeout, max duration, etc.), a monitor thread starts a fresh session
    automatically. The app outside only ever sees one `ElevenLabsAgent`
    instance with `start()` / `stop()` / `push_face_context()`.
    """

    def __init__(self, api_key: str, agent_id: str):
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY missing — set it in .env")
        if not agent_id:
            raise RuntimeError("ELEVENLABS_AGENT_ID missing — set it in .env")
        self._api_key = api_key
        self._agent_id = agent_id
        self._conversation = None
        self._conv_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._session_count = 0
        self._last_context_text: str = ""
        self._last_context_at: float = 0.0

    async def start(self):
        print(f"[AGENT] connecting to ElevenLabs (agent_id={self._agent_id})...")
        await asyncio.to_thread(self._start_session_blocking)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="agent-monitor"
        )
        self._monitor_thread.start()
        print("[AGENT] persistent monitor running — sessions will auto-restart on idle close")

    def _start_session_blocking(self):
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation

        client = ElevenLabs(api_key=self._api_key)
        print("[AGENT] building Conversation + audio interface ...")
        with self._conv_lock:
            self._conversation = Conversation(
                client,
                self._agent_id,
                requires_auth=True,
                audio_interface=AdaptiveDefaultAudioInterface(),
                client_tools=build_client_tools(),
                callback_agent_response=lambda r: print(f"[AGENT] {r}"),
                callback_user_transcript=lambda t: print(f"[USER] {t}"),
            )
            print("[AGENT] opening WebSocket session to ElevenLabs ...")
            self._conversation.start_session()
            self._session_count += 1
        print(f"[AGENT] ✓ session #{self._session_count} live "
              f"— mic open, speaker ready, contextual updates can flow")

    def _monitor_loop(self):
        backoff = RESTART_DELAY_S
        while not self._stop_requested.is_set():
            session_started_at = time.monotonic()
            try:
                conv = self._conversation
                if conv is None:
                    time.sleep(0.5)
                    continue
                conv.wait_for_session_end()
            except Exception as e:
                print(f"[AGENT] wait_for_session_end raised (ignored): {type(e).__name__}: {e}")

            if self._stop_requested.is_set():
                break

            session_duration = time.monotonic() - session_started_at
            if session_duration < EARLY_FAIL_THRESHOLD_S:
                backoff = min(backoff * 2, RESTART_BACKOFF_MAX_S)
                print(f"[AGENT] session died after only {session_duration:.1f}s — "
                      f"likely a hardware/config problem (check the audio device). "
                      f"Backing off, next attempt in {backoff:.1f}s")
            else:
                backoff = RESTART_DELAY_S
                print(f"[AGENT] session ended after {session_duration:.0f}s — "
                      f"restarting in {backoff:.1f}s")

            time.sleep(backoff)
            if self._stop_requested.is_set():
                break

            try:
                self._start_session_blocking()
            except Exception as e:
                backoff = min(backoff * 2, RESTART_BACKOFF_MAX_S)
                print(f"[AGENT] restart failed: {type(e).__name__}: {e} — "
                      f"next attempt in {backoff:.1f}s")
                time.sleep(backoff)

    async def stop(self):
        print("[AGENT] stopping (no auto-restart)...")
        self._stop_requested.set()
        await asyncio.to_thread(self._end_session_blocking)
        if self._monitor_thread is not None:
            await asyncio.to_thread(self._monitor_thread.join, 5.0)
        print("[AGENT] stopped")

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

    def push_face_context(self, face_info: dict | None):
        text = face_info_to_context_text(face_info)
        self._send_context(text)

    def push_face_state(self, state: dict):
        """Multi-person aware update — pass the FacePresenceTracker.current_state()."""
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
            print("[AGENT] no active session — contextual update dropped (will pick up on next session)")
            return

        try:
            conv.send_contextual_update(text)
            print(f"[AGENT] contextual_update sent: {text[:120]}")
        except Exception as e:
            print(f"[AGENT] contextual_update failed: {type(e).__name__}: {e}")
