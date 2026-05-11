import asyncio
from collections import defaultdict


STANDARD_EVENTS = {
    # Face events
    "face_detected",
    "face_recognized",
    "face_unknown",
    "face_lost",
    # Conversation events
    "conversation_started",
    "conversation_ended",
    "session_timeout",
    # Speech events
    "speech_started",
    "speech_ended",
    # TTS events
    "tts_started",
    "tts_finished",
    # System events
    "state_changed",
    "user_registered",
    "greeting",
    "speak",
    "registration_requested",
    "registration_complete",
    "name_captured",
}


class EventBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = defaultdict(list)
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    def subscribe(self, event_type: str, callback):
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback):
        try:
            self._subscribers[event_type].remove(callback)
        except ValueError:
            pass

    async def publish(self, event_type: str, data=None):
        for cb in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(data)
                else:
                    cb(data)
            except Exception as e:
                print(f"[event_bus] Error in {event_type} handler: {e}")

    def publish_sync(self, event_type: str, data=None):
        for cb in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(cb):
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(cb(data))
                    else:
                        loop.run_until_complete(cb(data))
                else:
                    cb(data)
            except Exception as e:
                print(f"[event_bus] Error in {event_type} handler: {e}")
