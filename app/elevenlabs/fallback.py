"""Offline 'I'll be right back' WAV for network outages.

When ElevenLabs' WebSocket dies mid-conversation, today the user
experiences silence until the next session reopens (1.5 – 30 s depending
on backoff). That's bad — they don't know if Nova heard them, if she's
thinking, or if something broke.

This module gives Nova a small bridge: a pre-generated WAV that plays
through the local speaker after ~5 s of reconnect failure, just so the
user gets a clear "I'm here, just a moment" rather than dead air.

Design constraints:
  * Offline by default — generates the WAV at first boot using
    espeak-ng (apt-installable, runs locally, no API call). The WAV is
    cached in NOVA_CACHE_DIR so this only happens once.
  * Skips silently if espeak-ng isn't installed. The fallback is a
    nice-to-have, not a hard dependency — we don't want to fail boot
    over it. The startup log makes it obvious whether it's wired up.
  * Plays through a *separate* PyAudio output stream (not the
    AdaptiveDefaultAudioInterface) because by the time we want to
    play it, that interface has been torn down by the SDK's WS-close
    handler. Opening a fresh PyAudio for 2 s of playback is cheap.
  * Bilingual-ish: espeak-ng can do Nepali but the quality is rough.
    We use a short, mostly-neutral English phrase that won't sound
    weirder to a Nepali speaker than the alternative — Nepali via
    espeak sounds robotic to a degree that's worse than English.
"""

import os
import shutil
import subprocess
import wave
from pathlib import Path


CACHE_DIR = Path(os.environ.get("NOVA_CACHE_DIR",
                                str(Path.home() / ".cache" / "nova")))
WAV_PATH = CACHE_DIR / "be_right_back.wav"

# Customizable phrase, in case someone wants to swap it for their own.
# Keep it short — every second of fallback is a second the user is
# waiting for the actual agent to come back online.
DEFAULT_PHRASE = os.environ.get(
    "NOVA_FALLBACK_PHRASE",
    "Sorry — just a moment, I'll be right back."
)

# espeak-ng voice + speed. en+f3 is a passable English female voice;
# rate 145 wpm is slightly slower than default for clarity over a
# possibly-degraded speaker.
ESPEAK_VOICE = os.environ.get("NOVA_FALLBACK_ESPEAK_VOICE", "en+f3")
ESPEAK_RATE = int(os.environ.get("NOVA_FALLBACK_ESPEAK_RATE", "145"))


def ensure_fallback_wav() -> Path | None:
    """Make sure a fallback WAV exists on disk, generating it if needed.

    Returns the path on success, None if generation failed (espeak-ng
    missing, write error, etc.). Callers should treat None as "don't
    try to play anything" — silence is a worse-but-acceptable fallback.
    """
    if WAV_PATH.exists() and WAV_PATH.stat().st_size > 1024:
        return WAV_PATH

    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        print("[FALLBACK] espeak-ng not installed — offline 'be right back' "
              "audio disabled. Install with: sudo apt install espeak-ng")
        return None

    binary = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                binary,
                "-v", ESPEAK_VOICE,
                "-s", str(ESPEAK_RATE),
                "-w", str(WAV_PATH),
                DEFAULT_PHRASE,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
        size = WAV_PATH.stat().st_size
        print(f"[FALLBACK] generated {WAV_PATH} ({size:,} bytes) via {binary}")
        return WAV_PATH
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[FALLBACK] espeak generation failed: {type(e).__name__}: {e}")
        try:
            if WAV_PATH.exists():
                WAV_PATH.unlink()
        except OSError:
            pass
        return None


def play_fallback_blocking(path: Path | None = None) -> bool:
    """Play the fallback WAV synchronously. Returns True on success.

    Opens its own PyAudio output stream — we can't reuse the SDK's
    because by the time this is called the WS has died and the SDK
    has torn its audio interface down. ~2 s of blocking is fine in
    the lifecycle thread; this only fires after 5 s of failed
    reconnect anyway.
    """
    target = path or WAV_PATH
    if not target.exists():
        return False
    try:
        import pyaudio
    except ImportError:
        print("[FALLBACK] pyaudio missing — can't play fallback")
        return False

    p = None
    stream = None
    wf = None
    try:
        wf = wave.open(str(target), "rb")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        chunk = 1024
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        return True
    except Exception as e:
        print(f"[FALLBACK] playback failed: {type(e).__name__}: {e}")
        return False
    finally:
        for closer, target_obj in (
            (lambda s: s.stop_stream(), stream),
            (lambda s: s.close(), stream),
            (lambda p: p.terminate(), p),
            (lambda w: w.close(), wf),
        ):
            if target_obj is not None:
                try:
                    closer(target_obj)
                except Exception:
                    pass
