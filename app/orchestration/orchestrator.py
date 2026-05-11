import asyncio
import os

from app.elevenlabs.client import ElevenLabsAgent
from app.orchestration.cooldown import CooldownManager
from app.orchestration.event_bus import EventBus
from app.orchestration.state import ConversationStateMachine
from config.config import ELEVENLABS_AGENT_ID, ELEVENLABS_API_KEY


class Orchestrator:
    def __init__(self):
        self.event_bus = EventBus()
        self.state = ConversationStateMachine(self.event_bus)
        self.cooldown = CooldownManager()
        self.camera = None
        self.preview = None
        self.face_monitor = None
        self.agent = ElevenLabsAgent(ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID)
        self._memory = None
        self._background_tasks = []
        self._running = False
        self._skip_face = os.getenv("NOVA_SKIP_FACE") == "1"

    async def start(self):
        self._running = True
        print("[ORCH] ━━━ Nova starting up ━━━")

        if not self._skip_face:
            self.event_bus.subscribe("face_recognized", self._on_face_event)
            self.event_bus.subscribe("face_unknown", self._on_face_event)
            self.event_bus.subscribe("face_lost", self._on_face_event)
            print("[ORCH] face event subscriptions wired")

        print("[ORCH] step 1/2 → starting ElevenLabs agent (fast, ~1–2s)")
        await self.agent.start()
        print("[ORCH] step 1/2 ✓ — agent is live. You can talk to it now.")

        if self._skip_face:
            print("[ORCH] step 2/2 skipped (NOVA_SKIP_FACE=1) — running agent-only")
        else:
            print("[ORCH] step 2/2 → starting face pipeline in background "
                  "(first run will download ~38 MB of models — visible progress below)")
            self._background_tasks.append(
                asyncio.create_task(self._start_face_pipeline_bg())
            )

        print(f"[ORCH] ━━━ startup complete ({len(self._background_tasks)} bg tasks) ━━━")

    async def _start_face_pipeline_bg(self):
        print("[ORCH] face pipeline: building in background "
              "(model downloads may take a minute on first run) ...")
        try:
            await asyncio.to_thread(self._init_face_pipeline_blocking)
            print("[ORCH] face pipeline: ready — starting poll loop")
            self._background_tasks.append(asyncio.create_task(self.face_monitor.run()))
            self._background_tasks.append(asyncio.create_task(self._memory_flush_loop()))
        except Exception as e:
            print(f"[ORCH] face pipeline failed to start: {type(e).__name__}: {e}")
            print("[ORCH] agent will keep running without face recognition")

    def _init_face_pipeline_blocking(self):
        """Synchronous bits — Camera, FaceMonitor (which loads ONNX models).
        Runs in a worker thread so the asyncio event loop stays responsive."""
        from app.face.camera import Camera
        from app.face.person_memory import get_memory
        from app.face.preview import CameraPreview
        from app.orchestration.face_monitor import FaceMonitor

        self._memory = get_memory()
        self.camera = Camera()
        self.preview = CameraPreview()
        self.preview.start()
        print("[ORCH] face pipeline: camera + preview started")

        self.face_monitor = FaceMonitor(
            self.event_bus, self.state, self.cooldown, preview=self.preview
        )
        print("[ORCH] face pipeline: FaceMonitor created")

    async def stop(self):
        print("[ORCH] Stopping orchestrator...")
        self._running = False
        await self.agent.stop()
        for task in self._background_tasks:
            task.cancel()
        if self.camera:
            self.camera.release()
            print("[ORCH] Camera released")
        if self.preview:
            self.preview.stop()
        if self._memory:
            await self._memory.flush_all()
        print("[ORCH] Orchestrator stopped")

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
