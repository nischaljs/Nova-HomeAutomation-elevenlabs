"""Network health probe for ElevenLabs.

A real speedtest takes 10+ seconds and burns several megabytes of
traffic — not something you want running every 30 seconds in a robot
that's already streaming audio over the same link. Instead we probe two
cheap signals that, taken together, give a very good picture of "is
the connection to ElevenLabs healthy right now":

  * DNS resolution time for api.elevenlabs.io — pure DNS hop
  * TCP-connect time to api.elevenlabs.io:443 — TCP RTT to the actual
    endpoint we care about. This catches both network congestion AND
    AP/router/upstream problems that ping(8) can miss because ICMP is
    deprioritized on many networks.

Both probes run in a worker thread (network I/O off the asyncio loop)
and publish their results to a shared `NetStats` snapshot. The watchdog
reads the snapshot on its own cadence (every 30 s) and emits the
periodic `[NET]` log line.

Optional: while an ElevenLabs WS session is live, the audio_interface's
`output()` callback can also count incoming TTS bytes. That gives us
real downstream-bandwidth numbers without any extra requests. Wired up
in adaptive_audio.py via NetStats.note_downstream_bytes().
"""

import os
import socket
import threading
import time


# Tunables — kept conservative so the probe doesn't itself become a
# noticeable load. Defaults can be overridden if a future deployment
# wants tighter/looser monitoring.
PROBE_INTERVAL_S = float(os.environ.get("NOVA_NET_PROBE_S", "30.0"))
PROBE_HOST = os.environ.get("NOVA_NET_HOST", "api.elevenlabs.io")
PROBE_PORT = int(os.environ.get("NOVA_NET_PORT", "443"))
CONNECT_TIMEOUT_S = float(os.environ.get("NOVA_NET_TIMEOUT_S", "3.0"))


class NetStats:
    """Lock-free-ish shared snapshot of the most recent network probe.

    Writers always overwrite individual floats (Python integer/float
    assignment is atomic under the GIL for our purposes), readers
    grab a quick lock to copy a consistent set. Hot-path callers
    (audio_interface.output) only do counters, which we reset on read.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.last_dns_ms: float | None = None
        self.last_tcp_ms: float | None = None
        self.last_probe_at: float = 0.0
        self.consecutive_failures = 0
        # Downstream bytes (TTS chunks coming in from ElevenLabs)
        # accumulated since the last log emit. Reset every read.
        self._downstream_bytes_since_log: int = 0
        self._downstream_window_started_at: float = time.monotonic()
        # Cumulative — useful if someone wants to know "how much TTS
        # have I been billed for visually" since boot.
        self.total_downstream_bytes: int = 0
        # First-audio latency: time from user_transcript callback to
        # the next output() chunk. Captures end-to-end RTT through
        # STT + LLM + TTS first-byte. Most interesting metric for
        # "does this feel snappy?".
        self.last_first_audio_ms: float | None = None
        self._user_transcript_at: float | None = None

    def update_probe(self, *, dns_ms: float | None, tcp_ms: float | None):
        with self._lock:
            self.last_dns_ms = dns_ms
            self.last_tcp_ms = tcp_ms
            self.last_probe_at = time.monotonic()
            if dns_ms is None or tcp_ms is None:
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0

    def note_downstream_bytes(self, n: int):
        # Hot path — called for every TTS chunk. Avoid acquiring the
        # lock for the read counters; integer += under GIL is atomic
        # enough for byte-counting purposes (we don't care about
        # exactly-accurate sums, just orders of magnitude).
        self._downstream_bytes_since_log += n
        self.total_downstream_bytes += n

    def note_user_transcript(self):
        self._user_transcript_at = time.monotonic()

    def note_first_audio_chunk(self):
        """Called when audio_interface.output() fires for the first
        chunk after the most recent user_transcript. The lifecycle
        thread arms this by setting user_transcript_at; we only
        compute latency on the *first* chunk after."""
        if self._user_transcript_at is None:
            return
        ms = (time.monotonic() - self._user_transcript_at) * 1000.0
        with self._lock:
            self.last_first_audio_ms = ms
        # Disarm — wait for the next user_transcript.
        self._user_transcript_at = None

    def drain_downstream_kbps(self) -> tuple[float, float]:
        """Return (kbps_since_last_drain, seconds_of_window)."""
        now = time.monotonic()
        with self._lock:
            window = now - self._downstream_window_started_at
            bytes_in_window = self._downstream_bytes_since_log
            self._downstream_bytes_since_log = 0
            self._downstream_window_started_at = now
        if window <= 0:
            return 0.0, 0.0
        kbps = (bytes_in_window * 8.0 / 1024.0) / window
        return kbps, window

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "dns_ms": self.last_dns_ms,
                "tcp_ms": self.last_tcp_ms,
                "last_probe_at": self.last_probe_at,
                "consecutive_failures": self.consecutive_failures,
                "first_audio_ms": self.last_first_audio_ms,
                "total_downstream_bytes": self.total_downstream_bytes,
            }


def _probe_once() -> tuple[float | None, float | None]:
    """One DNS + TCP round-trip. Returns (dns_ms, tcp_ms), with None
    in place of any leg that failed."""
    dns_ms: float | None = None
    tcp_ms: float | None = None
    addr: str | None = None

    t0 = time.monotonic()
    try:
        infos = socket.getaddrinfo(PROBE_HOST, PROBE_PORT,
                                    type=socket.SOCK_STREAM)
        dns_ms = (time.monotonic() - t0) * 1000.0
        addr = infos[0][4][0] if infos else None
    except Exception:
        return None, None

    if addr is None:
        return dns_ms, None

    t1 = time.monotonic()
    sock = None
    try:
        sock = socket.create_connection(
            (addr, PROBE_PORT), timeout=CONNECT_TIMEOUT_S
        )
        tcp_ms = (time.monotonic() - t1) * 1000.0
    except Exception:
        tcp_ms = None
    finally:
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass

    return dns_ms, tcp_ms


class NetProbe:
    """Background thread running _probe_once() every PROBE_INTERVAL_S."""

    def __init__(self, stats: NetStats):
        self._stats = stats
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="net-probe"
        )
        self._thread.start()
        print(f"[NET] probe started — {PROBE_HOST}:{PROBE_PORT} "
              f"every {PROBE_INTERVAL_S}s")

    def stop(self):
        self._running = False

    def _loop(self):
        # First probe runs ~2 s after start so the boot log isn't
        # interleaved with health output before the orchestrator has
        # printed its banner.
        time.sleep(2.0)
        while self._running:
            try:
                dns_ms, tcp_ms = _probe_once()
                self._stats.update_probe(dns_ms=dns_ms, tcp_ms=tcp_ms)
            except Exception as e:
                print(f"[NET] probe error (ignored): "
                      f"{type(e).__name__}: {e}")
            # Sleep in short chunks so stop() is responsive.
            slept = 0.0
            while self._running and slept < PROBE_INTERVAL_S:
                time.sleep(min(1.0, PROBE_INTERVAL_S - slept))
                slept += 1.0


NET_STATS: NetStats | None = None
NET_PROBE: NetProbe | None = None


def get_net_stats() -> NetStats:
    global NET_STATS
    if NET_STATS is None:
        NET_STATS = NetStats()
    return NET_STATS


def get_net_probe() -> NetProbe:
    global NET_PROBE
    if NET_PROBE is None:
        NET_PROBE = NetProbe(get_net_stats())
    return NET_PROBE
