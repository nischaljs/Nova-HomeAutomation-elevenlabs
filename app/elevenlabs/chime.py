"""Short 'I'm listening' chime played when a session opens.

The engagement-gated lifecycle means Nova "wakes up" the moment you
walk up — but until she speaks her first word, the user has no audible
confirmation she actually heard them step into frame. A 500 ms two-tone
chime through the local speaker fills that gap: walk up → 2 s engaged
check → chime → her greeting.

The WAV is synthesized at first boot (one short numpy call) and cached
in NOVA_CACHE_DIR so subsequent boots load it instantly. Playback uses
a fresh PyAudio stream because the session's own audio interface isn't
fully wired up at the exact moment we want the chime to land — we want
it BEFORE the SDK starts pushing TTS, so opening our own brief stream
is cleaner than coordinating with the SDK's.

Design notes:
  * Two ascending tones (440 Hz + 660 Hz, slightly delayed) with an
    exponential decay so the chime sounds like a soft "ding-ding" rather
    than a beep. Tested on cheap 3.5 mm Pi speakers — recognizable
    without being annoying.
  * 16 kHz mono int16 to match what the rest of Nova's audio path uses.
  * Total duration is 500 ms (you bumped this from 100 ms). Long enough
    to register, short enough not to delay the actual greeting.
"""

import os
import wave
from pathlib import Path

CACHE_DIR = Path(os.environ.get("NOVA_CACHE_DIR",
                                str(Path.home() / ".cache" / "nova")))
WAV_PATH = CACHE_DIR / "wake_chime.wav"

# Chime params — tunable but the defaults sound nice on Pi 3.5mm out.
SAMPLE_RATE = 16000
DURATION_S = float(os.environ.get("NOVA_CHIME_DURATION_S", "0.5"))
TONE_A_HZ = float(os.environ.get("NOVA_CHIME_TONE_A_HZ", "440.0"))
TONE_B_HZ = float(os.environ.get("NOVA_CHIME_TONE_B_HZ", "660.0"))
# Headroom factor — 0.4 = ~−8 dBFS, gentle, won't startle in a quiet room.
VOLUME = float(os.environ.get("NOVA_CHIME_VOLUME", "0.4"))


def ensure_chime_wav() -> Path | None:
    """Generate the chime WAV if it doesn't already exist. Returns the
    path on success, None on any failure (in which case the agent
    lifecycle just won't play the chime — non-fatal)."""
    if WAV_PATH.exists() and WAV_PATH.stat().st_size > 1024:
        return WAV_PATH
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        n = int(SAMPLE_RATE * DURATION_S)
        t = np.linspace(0.0, DURATION_S, n, endpoint=False, dtype=np.float32)
        # Two tones with a small offset so the second one feels like a
        # follow-up. Each gets its own decay envelope.
        env_a = np.exp(-4.5 * t)
        # Tone B starts 60 ms in — gives that "ding-DING" feel.
        offset = int(SAMPLE_RATE * 0.06)
        env_b = np.zeros_like(t)
        if offset < n:
            tail = t[: n - offset]
            env_b[offset:] = np.exp(-4.5 * tail)
        tone_a = np.sin(2.0 * np.pi * TONE_A_HZ * t) * env_a * 0.7
        tone_b = np.sin(2.0 * np.pi * TONE_B_HZ * t) * env_b
        # Soft 5 ms attack on both so we don't click on the first sample.
        attack_n = int(SAMPLE_RATE * 0.005)
        if attack_n > 0:
            ramp = np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
            tone_a[:attack_n] *= ramp
        mix = (tone_a + tone_b) * VOLUME
        # Hard clip to [-1, 1] just in case env_a + env_b additions
        # exceed the headroom on the first few samples.
        np.clip(mix, -1.0, 1.0, out=mix)
        pcm = (mix * 32767.0).astype(np.int16)

        with wave.open(str(WAV_PATH), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        print(f"[CHIME] generated wake chime → {WAV_PATH} "
              f"({WAV_PATH.stat().st_size:,} bytes, {DURATION_S*1000:.0f}ms, "
              f"{TONE_A_HZ:.0f}+{TONE_B_HZ:.0f}Hz)")
        return WAV_PATH
    except Exception as e:
        print(f"[CHIME] generation failed (ignored): "
              f"{type(e).__name__}: {e}")
        return None


def play_chime_blocking(path: Path | None = None) -> bool:
    """Open a fresh PyAudio output stream, play the chime, close. Blocking
    for ~500 ms which is fine — fired once per session open, on the
    lifecycle thread.

    Why a fresh stream instead of the SDK's audio interface: the SDK's
    interface is constructed inside Conversation() and doesn't expose a
    clean way to inject pre-session audio. Opening our own brief stream
    is 3 lines and avoids stepping on the SDK's state machine."""
    target = path or WAV_PATH
    if not target.exists():
        return False
    try:
        import pyaudio
        import wave as _wave
    except ImportError:
        return False
    p = None
    stream = None
    wf = None
    try:
        wf = _wave.open(str(target), "rb")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        return True
    except Exception as e:
        print(f"[CHIME] playback failed (ignored): "
              f"{type(e).__name__}: {e}")
        return False
    finally:
        for closer, obj in (
            (lambda s: s.stop_stream(), stream),
            (lambda s: s.close(), stream),
            (lambda p: p.terminate(), p),
            (lambda w: w.close(), wf),
        ):
            if obj is not None:
                try:
                    closer(obj)
                except Exception:
                    pass
