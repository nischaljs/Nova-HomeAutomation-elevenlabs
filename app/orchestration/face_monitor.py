import asyncio
import time

from app.face.face_presence_tracker import FacePresenceTracker
from app.face.vision_pipeline import get_latest
from app.orchestration.cooldown import CooldownManager
from app.orchestration.event_bus import EventBus
from app.orchestration.state import ConversationStateMachine

# How often to scan LatestVision for new recognize ticks. This is *not*
# the recognition rate (the VisionPipeline owns that) — just how often
# we look at the snapshot. Cheap, since it's a lock + list copy.
POLL_INTERVAL = 0.25


class FaceMonitor:
    """Consumes VisionPipeline output → drives the FacePresenceTracker.

    Before the pipeline refactor, this class did its own detection +
    recognition on every poll, duplicating work the preview was also
    doing. Now it's a pure consumer: read LatestVision, feed the
    recognized list into the tracker, publish events to the EventBus.
    """

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
        self._latest = get_latest()
        self._tracker = FacePresenceTracker()
        self._last_recognize_ts: float = 0.0
        self._poll_count = 0
        self._paused = False

    @property
    def tracker(self) -> FacePresenceTracker:
        return self._tracker

    def pause(self):
        self._paused = True
        print("[FACE] Face monitor PAUSED")

    def resume(self):
        self._paused = False
        print("[FACE] Face monitor RESUMED")

    @property
    def is_paused(self) -> bool:
        return self._paused

    async def run(self):
        print(f"[FACE] FaceMonitor started (consumes LatestVision every {POLL_INTERVAL}s)")
        # Tracker also needs to be ticked when no recognition has happened
        # for a while, so face_lost can fire after FACE_TRULY_LOST_SECONDS
        # of nobody on screen. We do that by calling update([]) on idle.
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            self._poll_count += 1
            if self._paused:
                continue
            try:
                faces_raw, recognized, scale, detect_ts, recognize_ts = (
                    self._latest.snapshot()
                )
                # Only feed the tracker when there's a *new* recognize
                # tick — otherwise the tracker would see the same faces
                # over and over and freshen their last_seen forever,
                # masking face_lost transitions.
                if recognize_ts != self._last_recognize_ts:
                    self._last_recognize_ts = recognize_ts
                    await self._handle_results(recognized)
                else:
                    # Stale recognition — but tracker still needs to age.
                    # Passing [] decays known states toward face_lost
                    # without re-publishing.
                    age = time.monotonic() - recognize_ts if recognize_ts else 1e9
                    if age > 1.5:
                        await self._handle_results([])
                if self.preview is not None:
                    self.preview.update_faces(faces_raw, recognized, scale)
            except Exception as e:
                print(f"[FACE] Error in poll cycle: {e}")

    async def _handle_results(self, results: list[dict]):
        events = self._tracker.update(results)
        for evt in events:
            event_type = evt.pop("event")
            await self.event_bus.publish(event_type, evt)
            if event_type == "face_recognized":
                print(f"[FACE] ✓ RECOGNIZED: {evt.get('name')} "
                      f"(conf={evt.get('confidence'):.2f}, id={evt.get('id')})")
            elif event_type == "face_unknown":
                print(f"[FACE] Unknown stable (count={evt.get('count', 1)})")
            elif event_type == "face_lost":
                print(f"[FACE] ✗ LOST: {evt.get('name')} (id={evt.get('id')})")
