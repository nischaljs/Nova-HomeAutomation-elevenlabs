import asyncio
import os
import subprocess
import time

from app.elevenlabs.client import ElevenLabsAgent
from app.orchestration.cooldown import CooldownManager
from app.orchestration.engagement import get_engagement
from app.orchestration.event_bus import EventBus
from app.orchestration.net_probe import get_net_probe, get_net_stats
from app.orchestration.state import ConversationStateMachine
from app.platform_detect import IS_PI
from config.config import ELEVENLABS_AGENT_ID, ELEVENLABS_API_KEY


# Watchdog parameters. The orchestrator polls subsystem health every
# WATCHDOG_TICK_S; if the camera hasn't delivered a frame in this long,
# we log + ask the camera thread to reconnect. The face-recognition
# stage has its own reconnect logic (download retries), so we don't
# poke it here.
WATCHDOG_TICK_S = 5.0
CAMERA_STALE_S = 10.0
# System health (Pi temp, throttling, process RSS) is logged at a slower
# cadence — every 30 s — so a long terminal session doesn't drown in
# health pings. Throttle flags are logged loudly the first time they
# appear because they tend to cause audio glitches.
HEALTH_LOG_INTERVAL_S = 30.0


def _read_pi_throttle() -> str | None:
    """Decode `vcgencmd get_throttled` into a human string, or None if
    the command isn't available (not on a Pi, or vcgencmd missing).

    The bitfield from vcgencmd (e.g. 0x50000) is famously cryptic, so
    we turn it into "OK" or a list of what's currently wrong + a count
    of past events. See https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd
    """
    if not IS_PI:
        return None
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"], stderr=subprocess.DEVNULL, timeout=2.0
        )
        text = out.decode().strip()  # e.g. "throttled=0x50000"
        raw = int(text.split("=")[1], 16)
    except Exception:
        return None
    if raw == 0:
        return "OK"
    flags_now = []
    flags_past = []
    bits = {
        0: ("undervoltage", "now"),
        1: ("arm_freq_capped", "now"),
        2: ("currently_throttled", "now"),
        3: ("soft_temp_limit", "now"),
        16: ("undervoltage", "past"),
        17: ("arm_freq_capped", "past"),
        18: ("throttling", "past"),
        19: ("soft_temp_limit", "past"),
    }
    for bit, (name, when) in bits.items():
        if raw & (1 << bit):
            (flags_now if when == "now" else flags_past).append(name)
    parts = []
    if flags_now:
        parts.append("NOW: " + ",".join(flags_now))
    if flags_past:
        parts.append("PAST: " + ",".join(flags_past))
    return " | ".join(parts) if parts else "OK"


def _read_pi_temp() -> float | None:
    """CPU temperature in °C, or None if not readable."""
    if not IS_PI:
        return None
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, timeout=2.0
        )
        # "temp=51.0'C"
        text = out.decode().strip()
        return float(text.split("=")[1].split("'")[0])
    except Exception:
        # Sysfs fallback works on most distros.
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return None


def _read_rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None


class Orchestrator:
    """Wires camera → vision pipeline → face monitor → engagement →
    ElevenLabs agent.

    Boot is parallelized: camera, face-model warm-up, and the agent
    lifecycle thread all come up concurrently. Nothing blocks on the
    SFace model download — it happens in the background and the rest of
    the stack runs as soon as each piece is ready. On the very first
    boot the user can see the camera preview and engagement before the
    agent itself is recognizing faces.
    """

    def __init__(self):
        self.event_bus = EventBus()
        self.state = ConversationStateMachine(self.event_bus)
        self.cooldown = CooldownManager()
        self.engagement = get_engagement()
        self.net_stats = get_net_stats()
        self.net_probe = get_net_probe()
        self.camera = None
        self.preview = None
        self.vision = None
        self.latest_vision = None
        self.face_monitor = None
        self._memory = None
        self._background_tasks: list[asyncio.Task] = []
        self._running = False
        self._skip_face = os.getenv("NOVA_SKIP_FACE") == "1"
        # Agent is constructed lazily so we can pass the EngagementState
        # in and avoid two import paths racing on get_engagement().
        self.agent = ElevenLabsAgent(
            ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID, engagement=self.engagement
        )

    async def start(self):
        self._running = True
        print("[ORCH] ━━━ Nova starting up ━━━")
        self._print_startup_banner()
        # Net probe runs from its own daemon thread; cheap (DNS + TCP
        # connect every 30 s). Start it before anything else so the
        # first [NET] line lands within ~5 s of boot.
        self.net_probe.start()

        if self._skip_face:
            print("[ORCH] NOVA_SKIP_FACE=1 — skipping face pipeline")
            await self.agent.start()
            print("[ORCH] startup complete (agent only, no face pipeline)")
            return

        # Wire face event subscriptions before any face-monitor poll so
        # the very first face_recognized event can land in _on_face_event.
        self.event_bus.subscribe("face_recognized", self._on_face_event)
        self.event_bus.subscribe("face_unknown", self._on_face_event)
        self.event_bus.subscribe("face_lost", self._on_face_event)

        # Boot the slow stuff in parallel. The agent lifecycle thread
        # starts immediately but won't actually open a WS until
        # engagement says so — which can only happen once the camera
        # and vision pipeline are alive. So the parallelism here is:
        #
        #   * face pipeline + camera bring-up runs in a worker thread
        #     (~1 s for camera, ~2-4 min for SFace download first time)
        #   * agent lifecycle thread is alive immediately, idle-waiting
        #   * vision pipeline starts as soon as bridge models are ready
        #     (it does its own readiness check)
        #
        # On a warm boot (models already cached) this entire sequence
        # is < 2 s.
        bringup_task = asyncio.create_task(self._bringup_face_stack())
        agent_task = asyncio.create_task(self.agent.start())

        await asyncio.gather(bringup_task, agent_task)

        # Start the watchdog last — it monitors everything we just
        # brought up.
        self._background_tasks.append(asyncio.create_task(self._watchdog_loop()))
        self._background_tasks.append(asyncio.create_task(self._memory_flush_loop()))

        print(f"[ORCH] ━━━ startup complete ({len(self._background_tasks)} bg tasks) ━━━")

    def _print_startup_banner(self):
        """Dump every tunable that matters at boot so the user can
        eyeball whether their .env / NOVA_* envs took effect — without
        having to grep across modules."""
        from app.elevenlabs.client import (
            ENGAGE_OPEN_AFTER_S,
            PRESENCE_CLOSE_AFTER_S,
            DISENGAGE_CLOSE_AFTER_S,
        )
        from app.orchestration.engagement import (
            ENGAGED_MAX_POSE_ASYM,
            STICKY_S,
        )
        headless = os.environ.get("NOVA_HEADLESS", "0") == "1"
        preview_on = (not headless) and os.environ.get("NOVA_DEBUG", "1") == "1"
        skip_face = self._skip_face

        print("[ORCH] ──────────── config ────────────")
        print(f"[ORCH] platform: {'Pi' if IS_PI else 'laptop/desktop'}  "
              f"headless={headless}  preview={preview_on}  "
              f"skip_face={skip_face}")
        print(f"[ORCH] engagement: open_after={ENGAGE_OPEN_AFTER_S}s  "
              f"sticky={STICKY_S}s  max_asym={ENGAGED_MAX_POSE_ASYM}")
        print(f"[ORCH] close timers: no_face={PRESENCE_CLOSE_AFTER_S}s  "
              f"turned_away={DISENGAGE_CLOSE_AFTER_S}s")
        print(f"[ORCH] gate ENV: NOVA_MIC_PROFILE="
              f"{os.environ.get('NOVA_MIC_PROFILE', 'auto')}  "
              f"NOVA_GATE_BARGE_MS={os.environ.get('NOVA_GATE_BARGE_MS', '1500')}  "
              f"NOVA_GATE_DEBUG={os.environ.get('NOVA_GATE_DEBUG', '0')}")
        print(f"[ORCH] vision: detect={os.environ.get('NOVA_DETECT_W', '480')}x"
              f"{os.environ.get('NOVA_DETECT_H', '360')}  "
              f"detect_dt={os.environ.get('NOVA_DETECT_INTERVAL_S', '0.15')}s  "
              f"recognize_dt={os.environ.get('NOVA_RECOGNIZE_INTERVAL_S', '0.5')}s")
        print("[ORCH] ─────────────────────────────────")

    async def _bringup_face_stack(self):
        """Camera + vision pipeline + face monitor, all set up in a
        worker thread so we don't block the asyncio loop on cv2 init."""
        from app.face.person_memory import get_memory
        from app.face.preview import CameraPreview
        from app.face.vision_pipeline import VisionPipeline, get_latest
        from app.orchestration.face_monitor import FaceMonitor

        print("[ORCH] face stack: bringing up camera + preview")
        await asyncio.to_thread(self._construct_camera_and_preview, CameraPreview)
        print("[ORCH] face stack: camera + preview alive")

        self._memory = get_memory()
        self.latest_vision = get_latest()

        # The vision pipeline doesn't strictly need models_ready to do
        # detection — it just won't run recognition until models are
        # loaded. So we start it now and it'll start producing
        # engagement signals (which gate the agent session) within ~1 s.
        self.vision = VisionPipeline(self.latest_vision, self.engagement)
        self.vision.start()
        print("[ORCH] face stack: VisionPipeline running (recognition will "
              "activate once models finish loading)")

        self.face_monitor = FaceMonitor(
            self.event_bus, self.state, self.cooldown, preview=self.preview
        )
        self._background_tasks.append(asyncio.create_task(self.face_monitor.run()))
        print("[ORCH] face stack: FaceMonitor task scheduled")

    def _construct_camera_and_preview(self, CameraPreview):
        from app.face.camera import Camera
        self.camera = Camera()
        self.preview = CameraPreview()
        self.preview.start()

    async def _watchdog_loop(self):
        """Lightweight health checker. Runs every WATCHDOG_TICK_S.

        Today it watches:
          * the camera — staleness → force reconnect
          * the VisionPipeline thread — dead → restart
          * the FaceMonitor task — dead → log loudly (recreating an
            asyncio task with the right cancellation/event-bus wiring
            is more risk than it's worth right now)
          * Pi health — temp, throttle flags, process RSS — logged every
            HEALTH_LOG_INTERVAL_S so a long terminal session can spot
            "oh, that's why audio crackled — undervoltage at 14:32".
        """
        last_health_log = 0.0
        last_throttle_seen: str | None = None
        while self._running:
            await asyncio.sleep(WATCHDOG_TICK_S)
            try:
                if self.camera is not None:
                    age = self.camera.last_frame_age_s
                    if age > CAMERA_STALE_S and self.camera.kind is not None:
                        print(f"[WATCHDOG] camera stale ({age:.1f}s "
                              f"since last frame, kind={self.camera.kind}) "
                              f"— forcing reconnect")
                        self.camera._teardown_camera()  # triggers reconnect inside _loop

                if self.vision is not None and self.vision._running:
                    dead = []
                    for tname, t in (
                        ("detect", getattr(self.vision, "_detect_thread", None)),
                        ("recognize", getattr(self.vision, "_recognize_thread", None)),
                    ):
                        if t is not None and not t.is_alive():
                            dead.append(tname)
                    if dead:
                        print(f"[WATCHDOG] VisionPipeline thread(s) died: "
                              f"{','.join(dead)} — restarting both")
                        try:
                            self.vision._running = False
                            self.vision.start()
                        except Exception as e:
                            print(f"[WATCHDOG] restart failed: "
                                  f"{type(e).__name__}: {e}")

                # FaceMonitor lives as an asyncio.Task in
                # _background_tasks[0] (or wherever — find it by name).
                for t in self._background_tasks:
                    if t.done() and not t.cancelled():
                        exc = t.exception()
                        if exc is not None:
                            print(f"[WATCHDOG] bg task died: "
                                  f"{type(exc).__name__}: {exc}")

                # Pi health — temp / throttle / RSS — logged at the
                # slower HEALTH_LOG_INTERVAL_S cadence.
                now = time.monotonic()
                if now - last_health_log >= HEALTH_LOG_INTERVAL_S:
                    last_health_log = now
                    temp = _read_pi_temp()
                    throttle = _read_pi_throttle()
                    rss = _read_rss_mb()
                    parts = []
                    if temp is not None:
                        parts.append(f"cpu={temp:.1f}°C")
                    if throttle is not None:
                        parts.append(f"throttle={throttle}")
                    if rss is not None:
                        parts.append(f"rss={rss:.0f}MB")
                    if parts:
                        print("[HEALTH] " + "  ".join(parts))
                    # Loud one-shot alert when throttle flips into a
                    # non-OK state. Pi throttle/undervoltage is the
                    # most common cause of audio crackling on the field.
                    if (throttle is not None
                            and throttle != "OK"
                            and throttle != last_throttle_seen):
                        print(f"[HEALTH] ⚠ Pi reporting issues: {throttle}. "
                              f"Check PSU (need a real 5V/3A USB-C supply) "
                              f"and ventilation.")
                    last_throttle_seen = throttle

                    # [NET] line — composed from the NetStats snapshot
                    # the net-probe thread is keeping fresh, plus the
                    # in-session counters that the audio interface
                    # increments on every TTS chunk.
                    self._emit_net_line()
            except Exception as e:
                print(f"[WATCHDOG] tick error: {type(e).__name__}: {e}")

    def _emit_net_line(self):
        snap = self.net_stats.snapshot()
        kbps, window = self.net_stats.drain_downstream_kbps()
        parts = []
        if snap["dns_ms"] is not None:
            parts.append(f"dns={snap['dns_ms']:.0f}ms")
        else:
            parts.append("dns=FAIL")
        if snap["tcp_ms"] is not None:
            parts.append(f"tcp={snap['tcp_ms']:.0f}ms")
        else:
            parts.append("tcp=FAIL")
        if snap["consecutive_failures"] > 0:
            parts.append(f"fails={snap['consecutive_failures']}")
        if snap["first_audio_ms"] is not None:
            parts.append(f"first_audio={snap['first_audio_ms']:.0f}ms")
        # Only print the downstream kbps when we actually received
        # bytes — most ticks (no active session, or session idle) it
        # would just spam "downstream=0.0kbps".
        if kbps > 0.5:
            parts.append(f"downstream={kbps:.0f}kbps")
        print("[NET] " + "  ".join(parts))

    async def _memory_flush_loop(self):
        while self._running:
            await asyncio.sleep(30)
            if self._memory:
                await self._memory.flush()

    async def _on_face_event(self, data: dict):
        if self.face_monitor is None:
            return
        state = self.face_monitor.tracker.current_state()
        self.agent.push_face_state(state)

    async def stop(self):
        print("[ORCH] Stopping orchestrator...")
        self._running = False
        for task in self._background_tasks:
            task.cancel()
        await self.agent.stop()
        if self.vision is not None:
            self.vision.stop()
        self.net_probe.stop()
        if self.camera is not None:
            self.camera.release()
            print("[ORCH] Camera released")
        if self.preview is not None:
            self.preview.stop()
        if self._memory is not None:
            await self._memory.flush_all()
        print("[ORCH] Orchestrator stopped")
