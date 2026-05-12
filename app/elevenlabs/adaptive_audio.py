import queue
import threading

import numpy as np

from elevenlabs.conversational_ai.conversation import AudioInterface

from app.elevenlabs.speech_gate import SpeechGate
from app.orchestration.net_probe import get_net_stats


SDK_RATE = 16000
INPUT_BUFFER_MS = 250
OUTPUT_BUFFER_MS = 62.5
PROBE_RATES = [16000, 48000, 44100, 32000, 24000, 22050, 8000]


class AdaptiveDefaultAudioInterface(AudioInterface):
    """ElevenLabs `AudioInterface` that works on any sane sound card.

    The vanilla `DefaultAudioInterface` opens both streams at 16 kHz —
    the rate the ElevenLabs backend expects. On a USB webcam mic that
    only supports 32/44.1/48 kHz (like the Logitech C920e), or on a
    Pi without ALSA's `plug` plugin wired as default, PortAudio rejects
    the open with `Invalid sample rate (-9997)` and the session dies.

    This subclass:
      * Probes the default input & output devices for a sample rate they
        actually support (16 kHz first — no resampling needed if the
        device offers it — then 48k, 44.1k, ...).
      * Opens each stream at its device's preferred rate.
      * Resamples in/out to/from 16 kHz using `soxr` (libsoxr-backed,
        fast on ARM) so the ElevenLabs side never knows what rate the
        physical device is running at.

    Also makes stop() idempotent and fault-tolerant — see
    RobustDefaultAudioInterface for why.
    """

    def __init__(self, engagement=None):
        try:
            import pyaudio  # noqa: F401
        except ImportError as e:
            raise ImportError("AdaptiveDefaultAudioInterface needs pyaudio") from e
        try:
            import soxr  # noqa: F401
        except ImportError as e:
            raise ImportError("AdaptiveDefaultAudioInterface needs soxr — pip install soxr") from e
        self._stop_lock = threading.Lock()
        self._stopped = False
        # The gate is what makes Nova ignore breaths, sneezes, side
        # chatter, and her own speaker echo. Constructed eagerly so the
        # noise-floor calibration starts on the first frame, not on the
        # first frame *after* an initial burst of audio we'd otherwise
        # ship straight to ElevenLabs.
        self._gate = SpeechGate(engagement=engagement)

    def _probe_supported_rate(self, p, device_index, is_input: bool) -> int:
        import pyaudio
        for rate in PROBE_RATES:
            try:
                ok = p.is_format_supported(
                    rate,
                    input_device=device_index if is_input else None,
                    input_channels=1 if is_input else None,
                    input_format=pyaudio.paInt16 if is_input else None,
                    output_device=device_index if not is_input else None,
                    output_channels=1 if not is_input else None,
                    output_format=pyaudio.paInt16 if not is_input else None,
                )
                if ok:
                    return rate
            except (ValueError, Exception):
                continue
        raise RuntimeError(
            f"No supported sample rate found for "
            f"{'input' if is_input else 'output'} device {device_index}"
        )

    def start(self, input_callback):
        import pyaudio
        import soxr

        self._soxr = soxr
        self.input_callback = input_callback
        self.output_queue: queue.Queue[bytes] = queue.Queue()
        self.should_stop = threading.Event()

        self.p = pyaudio.PyAudio()
        in_idx = self.p.get_default_input_device_info()["index"]
        out_idx = self.p.get_default_output_device_info()["index"]

        self.input_rate = self._probe_supported_rate(self.p, in_idx, is_input=True)
        self.output_rate = self._probe_supported_rate(self.p, out_idx, is_input=False)
        print(f"[AUDIO] adaptive: input device #{in_idx} @ {self.input_rate}Hz, "
              f"output device #{out_idx} @ {self.output_rate}Hz, SDK expects {SDK_RATE}Hz")

        in_frames = max(1, int(self.input_rate * INPUT_BUFFER_MS / 1000))
        out_frames = max(1, int(self.output_rate * OUTPUT_BUFFER_MS / 1000))

        self.in_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.input_rate,
            input=True,
            input_device_index=in_idx,
            stream_callback=self._in_callback,
            frames_per_buffer=in_frames,
            start=True,
        )
        self.out_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.output_rate,
            output=True,
            output_device_index=out_idx,
            frames_per_buffer=out_frames,
            start=True,
        )

        self.output_thread = threading.Thread(target=self._output_thread, daemon=True)
        self.output_thread.start()

    def _in_callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        try:
            if self.input_rate == SDK_RATE:
                pcm_16k = in_data
            else:
                samples = np.frombuffer(in_data, dtype=np.int16)
                resampled = self._soxr.resample(samples, self.input_rate, SDK_RATE)
                pcm_16k = resampled.astype(np.int16).tobytes()
            # SpeechGate filters out non-speech, near-silence, breath
            # bursts, side conversations, and Nova's own speaker echo
            # — yielding only the 20 ms frames that look like real
            # engaged speech. Anything not yielded simply never reaches
            # ElevenLabs, so the server-side VAD never sees it.
            for chunk in self._gate.feed(pcm_16k):
                self.input_callback(chunk)
        except Exception as e:
            print(f"[AUDIO] in_callback error (ignored): {type(e).__name__}: {e}")
        return (None, pyaudio.paContinue)

    @property
    def gate(self) -> SpeechGate:
        return self._gate

    def is_input_alive(self) -> bool:
        """Best-effort check that the USB mic is still there. Used by
        the agent lifecycle to detect hot-unplug — if the user pulls
        out the USB mic mid-conversation, the input stream silently
        stops producing frames but the session's WebSocket happily
        stays open, leaving Nova mute. Returns False when we can
        confirm the stream is dead so the lifecycle thread can rebuild
        the audio interface (which re-probes the default device).
        """
        s = getattr(self, "in_stream", None)
        if s is None:
            return False
        try:
            return bool(s.is_active())
        except Exception:
            return False

    def output(self, audio: bytes):
        # Tell the gate Nova just produced speaker audio. The gate uses
        # this to switch to the stricter barge-in threshold (1.5 s of
        # clear speech required to interrupt) instead of the idle one
        # (250 ms — fine when she's silent). Without this, a breath
        # during her sentence would cut her off.
        self._gate.notify_agent_output()
        # Track downstream bandwidth + first-audio latency. Both feed
        # the [NET] log line. note_first_audio_chunk only fires for
        # the first chunk after each user_transcript — the lifecycle
        # thread arms it via NetStats.note_user_transcript().
        try:
            ns = get_net_stats()
            ns.note_downstream_bytes(len(audio))
            ns.note_first_audio_chunk()
        except Exception:
            pass
        self.output_queue.put(audio)

    def interrupt(self):
        try:
            while True:
                _ = self.output_queue.get(block=False)
        except queue.Empty:
            pass

    def _output_thread(self):
        while not self.should_stop.is_set():
            try:
                audio = self.output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if self.output_rate == SDK_RATE:
                    self.out_stream.write(audio)
                else:
                    samples = np.frombuffer(audio, dtype=np.int16)
                    resampled = self._soxr.resample(samples, SDK_RATE, self.output_rate)
                    self.out_stream.write(resampled.astype(np.int16).tobytes())
            except Exception as e:
                print(f"[AUDIO] output_thread write error (ignored): {type(e).__name__}: {e}")

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
