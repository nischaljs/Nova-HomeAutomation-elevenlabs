import time


GREET_COOLDOWN = 60.0
UNKNOWN_ONBOARDING_COOLDOWN = 60.0
STABLE_DETECTION_SECONDS = 2.0
FACE_LOST_CONFIRM_SECONDS = 2.0
FACE_TRULY_LOST_SECONDS = 5.0
POST_PIPELINE_COOLDOWN = 10.0
MIN_REPUBLISH_INTERVAL = 10.0
PIPELINE_IDLE_TIMEOUT = 5.0


class IdentityCooldown:
    def __init__(self, identity_id: str):
        self.identity_id = identity_id
        self.detection_start: float | None = None
        self.last_seen: float = 0.0
        self.last_greeted: float = 0.0
        self.last_onboarding: float = 0.0
        self.last_published: float = 0.0
        self.greeted_count: int = 0
        self.stable = False

    def mark_detected(self):
        now = time.time()
        if self.detection_start is None:
            self.detection_start = now
        self.last_seen = now
        if not self.stable and (now - self.detection_start) >= STABLE_DETECTION_SECONDS:
            self.stable = True

    def mark_lost(self):
        self.detection_start = None
        self.stable = False

    def mark_greeted(self):
        self.last_greeted = time.time()
        self.greeted_count += 1

    def mark_onboarding_done(self):
        self.last_onboarding = time.time()

    def mark_published(self):
        self.last_published = time.time()

    @property
    def is_stable(self) -> bool:
        return self.stable

    @property
    def can_greet(self) -> bool:
        return (time.time() - self.last_greeted) > GREET_COOLDOWN

    @property
    def can_onboard(self) -> bool:
        return (time.time() - self.last_onboarding) > UNKNOWN_ONBOARDING_COOLDOWN

    @property
    def can_republish(self) -> bool:
        return (time.time() - self.last_published) > MIN_REPUBLISH_INTERVAL

    @property
    def is_temporarily_lost(self) -> bool:
        return self.last_seen > 0 and (time.time() - self.last_seen) > FACE_LOST_CONFIRM_SECONDS

    @property
    def is_truly_lost(self) -> bool:
        return self.last_seen > 0 and (time.time() - self.last_seen) > FACE_TRULY_LOST_SECONDS


class CooldownManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries: dict[str, IdentityCooldown] = {}
            cls._instance._session_start: float = 0.0
            cls._instance._post_pipeline_until: float = 0.0
            cls._instance._pipeline_completed_at: float = 0.0
        return cls._instance

    def get(self, identity_id: str) -> IdentityCooldown:
        if identity_id not in self._entries:
            self._entries[identity_id] = IdentityCooldown(identity_id)
        return self._entries[identity_id]

    def remove(self, identity_id: str):
        self._entries.pop(identity_id, None)

    def cleanup_stale(self, max_age: float = 300.0):
        now = time.time()
        stale = [iid for iid, e in self._entries.items()
                 if e.last_seen > 0 and (now - e.last_seen) > max_age]
        for iid in stale:
            del self._entries[iid]

    def reset_all(self):
        self._entries.clear()

    def set_post_pipeline_cooldown(self):
        self._post_pipeline_until = time.time() + POST_PIPELINE_COOLDOWN
        self._pipeline_completed_at = time.time()
        print(f"[COOLDOWN] Post-pipeline face events suppressed for {POST_PIPELINE_COOLDOWN}s")
        print(f"[COOLDOWN] Pipeline idle timer started ({PIPELINE_IDLE_TIMEOUT}s)")

    def reset_pipeline_idle(self):
        self._pipeline_completed_at = 0.0

    @property
    def pipeline_idle_complete(self) -> bool:
        if self._pipeline_completed_at == 0:
            return False
        return (time.time() - self._pipeline_completed_at) >= PIPELINE_IDLE_TIMEOUT

    @property
    def post_pipeline_active(self) -> bool:
        if self._post_pipeline_until == 0:
            return False
        if time.time() < self._post_pipeline_until:
            return True
        self._post_pipeline_until = 0
        return False
