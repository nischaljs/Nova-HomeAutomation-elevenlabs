import time


GREET_COOLDOWN = 60
NAME_ASK_COOLDOWN = 60
UNKNOWN_SETTLE_FRAMES = 10
FACE_LOST_TIMEOUT = 30


class CooldownManager:
    def __init__(self):
        self._last_greeting: dict[str, float] = {}
        self._last_name_ask: float = 0.0
        self._session_faces: dict[str, dict] = {}
        self._unknown_first_seen: float | None = None
        self._last_face_time: float = 0.0

    def can_greet_face(self, face_id: str) -> bool:
        last = self._last_greeting.get(face_id, 0.0)
        return (time.time() - last) > GREET_COOLDOWN

    def mark_greeted(self, face_id: str):
        self._last_greeting[face_id] = time.time()

    def can_ask_name(self) -> bool:
        return (time.time() - self._last_name_ask) > NAME_ASK_COOLDOWN

    def mark_name_asked(self):
        self._last_name_ask = time.time()

    def start_unknown_session(self):
        if self._unknown_first_seen is None:
            self._unknown_first_seen = time.time()

    @property
    def unknown_settled(self) -> bool:
        if self._unknown_first_seen is None:
            return False
        return (time.time() - self._unknown_first_seen) >= UNKNOWN_SETTLE_FRAMES

    def reset_unknown(self):
        self._unknown_first_seen = None

    def touch_face(self):
        self._last_face_time = time.time()

    @property
    def face_is_present(self) -> bool:
        return (time.time() - self._last_face_time) < FACE_LOST_TIMEOUT

    def reset_session(self):
        self._session_faces.clear()
        self.reset_unknown()
