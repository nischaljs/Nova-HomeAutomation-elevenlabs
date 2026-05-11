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

        if self._skip_face:
            print("[ORCH] NOVA_SKIP_FACE=1 — skipping face pipeline")
            print("[ORCH] step 1/1 → starting ElevenLabs agent")
            await self.agent.start()
            print("[ORCH] step 1/1 ✓ — agent is live")
        else:
            self.event_bus.subscribe("face_recognized", self._on_face_event)
            self.event_bus.subscribe("face_unknown", self._on_face_event)
            self.event_bus.subscribe("face_lost", self._on_face_event)
            print("[ORCH] face event subscriptions wired")

            # Step 1: build camera + preview + face monitor, then BLOCK
            # until face recognition models are loaded. On first run this
            # is a 2–4 minute SFace download; the agent intentionally does
            # NOT start until face context is available — we don't want
            # visitors talking to an agent that's blind to who they are.
            # On subsequent runs the models are cached so this step is ~1s.
            print("[ORCH] step 1/2 → bringing up face pipeline + camera ...")
            await asyncio.to_thread(self._init_face_pipeline_blocking)
            print("[ORCH] step 1/2 → waiting for face models to be ready "
                  "(first run only: ~38 MB download — see [FACE] progress lines)")
            await asyncio.to_thread(self._wait_for_face_models)
            print("[ORCH] step 1/2 ✓ — face pipeline ready, models loaded")

            # Start face polling AFTER models are ready
            self._background_tasks.append(asyncio.create_task(self.face_monitor.run()))
            self._background_tasks.append(asyncio.create_task(self._memory_flush_loop()))

            print("[ORCH] step 2/2 → starting ElevenLabs agent")
            await self.agent.start()
            print("[ORCH] step 2/2 ✓ — agent is live. You can talk to it now.")

        print(f"[ORCH] ━━━ startup complete ({len(self._background_tasks)} bg tasks) ━━━")

    def _init_face_pipeline_blocking(self):
        """Camera, preview, FaceMonitor — synchronous setup off the event loop.
        Bridge construction itself is now instant (validation runs in a
        daemon thread); we'll block on bridge.wait_models_ready() afterward."""
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
        print("[ORCH] face pipeline: FaceMonitor created — bridge constructed")

    def _wait_for_face_models(self):
        from app.face.face_tools import get_bridge
        bridge = get_bridge()
        bridge.wait_models_ready()

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
