import threading

from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface


class RobustDefaultAudioInterface(DefaultAudioInterface):
    """`DefaultAudioInterface` with idempotent, fault-tolerant `stop()`.

    The vanilla SDK calls `stop()` from multiple threads when ElevenLabs closes
    the WebSocket — both the WS receive thread and the PyAudio input callback
    race to call `end_session()`. On Linux this manifests as `pthread_join`
    failures and `[Errno -9999] Unanticipated host error` from `stop_stream`,
    which kills the worker thread before cleanup finishes, leaking PyAudio
    streams and breaking restart.

    This subclass:
      * Guards `stop()` so the first caller wins; later callers no-op.
      * Wraps each teardown step in its own try/except so one failure doesn't
        prevent later steps from running.
    """

    def __init__(self):
        super().__init__()
        self._stop_lock = threading.Lock()
        self._stopped = False

    def stop(self):
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        steps = [
            ("should_stop.set", lambda: self.should_stop.set()),
            ("output_thread.join", lambda: self.output_thread.join(timeout=2.0)),
            ("in_stream.stop_stream", lambda: self.in_stream.stop_stream()),
            ("in_stream.close", lambda: self.in_stream.close()),
            ("out_stream.stop_stream", lambda: self.out_stream.stop_stream()),
            ("out_stream.close", lambda: self.out_stream.close()),
            ("pyaudio.terminate", lambda: self.p.terminate()),
        ]
        for name, action in steps:
            try:
                action()
            except Exception as e:
                print(f"[AUDIO] cleanup {name} failed (ignored): {type(e).__name__}: {e}")
