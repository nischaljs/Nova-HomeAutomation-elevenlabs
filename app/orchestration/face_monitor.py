import asyncio
import time

import cv2
import numpy as np

from app.face.face_presence_tracker import FacePresenceTracker
from app.face.face_tools import FrameBuffer, get_bridge
from app.face.person_memory import get_memory
from app.orchestration.cooldown import CooldownManager
from app.orchestration.event_bus import EventBus
from app.orchestration.state import ConversationStateMachine

POLL_INTERVAL = 0.5


class FaceMonitor:
    def __init__(
        self,
        event_bus: EventBus,
        state_machine: ConversationStateMachine,
        cooldown: CooldownManager,
        preview=None,
    ):
        self.event_bus = event_bus
        self.state = state_machine
        self.cooldown = cooldown
        self.preview = preview
        self._bridge = get_bridge()
        self._memory = get_memory()
        self._poll_count = 0
        self._paused = False
        self._tracker = FacePresenceTracker()

    @property
    def tracker(self) -> FacePresenceTracker:
        return self._tracker

    def pause(self):
        self._paused = True
        print("[FACE] Face recognition PAUSED")

    def resume(self):
        self._paused = False
        print("[FACE] Face recognition RESUMED")

    @property
    def is_paused(self) -> bool:
        return self._paused

    async def run(self):
        fb = FrameBuffer()
        print(f"[FACE] FaceMonitor started (poll every {POLL_INTERVAL}s)")
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            self._poll_count += 1
            if self._paused:
                continue
            try:
                frame, ts = fb.get()
                if frame is None:
                    continue
                results = await self._process_frame(frame)
                await self._handle_results(results)
                if self.preview is not None:
                    self.preview.update_faces(results)
            except Exception as e:
                print(f"[FACE] Error in poll cycle: {e}")

    async def _process_frame(self, frame: np.ndarray) -> list[dict]:
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        t0 = time.time()
        results = await asyncio.to_thread(self._bridge.recognize_all, small)
        elapsed = time.time() - t0
        if results and elapsed > 0.3:
            summary = []
            for r in results:
                if r.get("unknown"):
                    summary.append("unknown")
                else:
                    summary.append(f"{r.get('name')}({r.get('confidence', 0):.2f})")
            print(f"[FACE] Recognize_all: [{', '.join(summary)}] (took {elapsed:.3f}s)")
        return results

    async def _handle_results(self, results: list[dict]):
        events = self._tracker.update(results)
        for evt in events:
            event_type = evt.pop("event")
            await self.event_bus.publish(event_type, evt)
            if event_type == "face_recognized":
                print(f"[FACE] ✓ RECOGNIZED: {evt.get('name')} (conf={evt.get('confidence'):.2f}, id={evt.get('id')})")
            elif event_type == "face_unknown":
                print(f"[FACE] Unknown stable (count={evt.get('count', 1)})")
            elif event_type == "face_lost":
                print(f"[FACE] ✗ LOST: {evt.get('name')} (id={evt.get('id')})")
