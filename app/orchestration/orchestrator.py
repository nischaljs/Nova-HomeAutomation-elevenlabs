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
        print("[ORCH] Starting orchestrator...")

        if self._skip_face:
            print("[ORCH] NOVA_SKIP_FACE=1 — running agent-only (no camera, no face)")
        else:
            await self._start_face_pipeline()

        await self.agent.start()
        print(f"[ORCH] Started {len(self._background_tasks)} background tasks")

    async def _start_face_pipeline(self):
        from app.face.camera import Camera
        from app.face.person_memory import get_memory
        from app.face.preview import CameraPreview
        from app.orchestration.face_monitor import FaceMonitor

        self._memory = get_memory()

        self.event_bus.subscribe("face_recognized", self._on_face_recognized)
        self.event_bus.subscribe("face_detected", self._on_face_recognized)
        self.event_bus.subscribe("face_unknown", self._on_face_unknown)
        self.event_bus.subscribe("face_lost", self._on_face_lost)

        self.camera = Camera()
        self.preview = CameraPreview()
        self.preview.start()
        print("[ORCH] Camera + preview started")

        self.face_monitor = FaceMonitor(
            self.event_bus, self.state, self.cooldown, preview=self.preview
        )
        print("[ORCH] FaceMonitor created")

        self._background_tasks.append(asyncio.create_task(self.face_monitor.run()))
        self._background_tasks.append(asyncio.create_task(self._memory_flush_loop()))

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
            await self._memory.flush()

    async def _on_face_recognized(self, data: dict):
        if data and data.get("unknown"):
            return
        self.agent.push_face_context(data)

    async def _on_face_unknown(self, data: dict):
        self.agent.push_face_context({"unknown": True})

    async def _on_face_lost(self, data: dict):
        self.agent.push_face_context(None)
