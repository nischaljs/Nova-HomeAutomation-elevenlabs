import asyncio
import time

import cv2
import numpy as np

from app.face.face_presence_tracker import FacePresenceTracker
from app.face.face_tools import FrameBuffer, get_bridge
from app.face.person_memory import get_memory
from app.orchestration.cooldown import CooldownManager
from app.orchestration.event_bus import EventBus
from app.orchestration.state import ConversationState, ConversationStateMachine

POLL_INTERVAL = 1.0


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
        self._prev_face_id = None
        self._unknown_streak = 0
        self._face_present = False
        self._poll_count = 0
        self._unknown_published = False
        self._no_face_published = False
        self._paused = False

        self._tracker = FacePresenceTracker()

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
        print("[FACE] FaceMonitor started (poll every {}s)".format(POLL_INTERVAL))
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            self._poll_count += 1

            if self._paused:
                continue

            try:
                frame, ts = fb.get()
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                has_face, result = await self._process_frame(frame)
                await self._handle_result(has_face, result, w, h)
                if self.preview is not None:
                    self.preview.update_face(result if has_face else None)
            except Exception as e:
                print(f"[FACE] Error in poll cycle: {e}")

    async def _process_frame(self, frame: np.ndarray) -> tuple[bool, dict | None]:
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        t0 = time.time()
        result = await asyncio.to_thread(self._bridge.recognize, small)
        elapsed = time.time() - t0
        if result is None:
            return False, None
        if elapsed > 0.3:
            print(f"[FACE] Recognize: name={result.get('name')} "
                  f"conf={result.get('confidence')} id={result.get('id')} "
                  f"unknown={result.get('unknown')} (took {elapsed:.3f}s)")
        return True, result

    async def _handle_result(self, has_face: bool, result: dict | None, frame_w=0, frame_h=0):
        tracker_events = self._tracker.update(has_face, result)

        if tracker_events is None:
            return

        if isinstance(tracker_events, list):
            for evt in tracker_events:
                event_type = evt.pop("event")
                await self.event_bus.publish(event_type, evt)
                if event_type == "face_recognized":
                    self._on_tracker_recognized(evt)
                elif event_type == "face_unknown":
                    self._on_tracker_unknown(evt)
                elif event_type == "face_lost":
                    self._on_tracker_lost(evt)
        elif isinstance(tracker_events, dict):
            event_type = tracker_events.pop("event")
            await self.event_bus.publish(event_type, tracker_events)
            if event_type == "face_lost":
                self._on_tracker_lost(tracker_events)

    def _on_tracker_recognized(self, data: dict):
        face_id = data["id"]
        name = data.get("name", "unknown")
        conf = data.get("confidence", 0)
        if face_id != self._prev_face_id:
            print(f"[FACE] ✓ RECOGNIZED: {name} (conf={conf}, id={face_id})")
            self._prev_face_id = face_id
            self._unknown_streak = 0
            self._unknown_published = False

    def _on_tracker_unknown(self, data: dict):
        self._unknown_streak += 1
        if self._prev_face_id is not None:
            print(f"[FACE] Face changed → UNKNOWN (was id={self._prev_face_id})")
            self._prev_face_id = None
            self.cooldown.reset_session()

    def _on_tracker_lost(self, data: dict):
        face_id = data.get("id")
        print(f"[FACE] Face LOST (id={face_id})")
        self._prev_face_id = None
        self._unknown_streak = 0
        self._unknown_published = False
        self._no_face_published = True
