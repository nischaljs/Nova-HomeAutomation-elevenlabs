import time

from app.orchestration.cooldown_manager import CooldownManager, STABLE_DETECTION_SECONDS


KNOWN_TO_UNKNOWN_FLICKER_TOLERANCE = 3.0


class FacePresenceTracker:
    def __init__(self):
        self._cooldown = CooldownManager()
        self._current_identity_id: str | None = None
        self._current_name: str = ""
        self._current_confidence: float = 0.0
        self._face_present = False
        self._last_published_identity: str | None = None
        self._lost_published = True
        self._unknown_published = False
        self._unknown_detection_start: float | None = None
        self._flicker_unknown_start: float | None = None

    def update(self, has_face: bool, result: dict | None):
        now = time.time()

        if not has_face:
            return self._handle_no_face(now)

        if result and not result.get("unknown", True):
            return self._handle_known_face(result, now)
        else:
            return self._handle_unknown_face(now)

    def _handle_no_face(self, now: float) -> dict | None:
        if not self._face_present:
            return None

        self._face_present = False
        identity_id = self._current_identity_id or "unknown"

        entry = self._cooldown.get(identity_id)
        entry.mark_lost()

        if entry.is_truly_lost and not self._lost_published:
            self._lost_published = True
            eid = self._current_identity_id
            self._current_identity_id = None
            self._last_published_identity = None
            print(f"[PRESENCE] Face TRULY LOST: {identity_id}")
            return {"event": "face_lost", "id": eid}

        if entry.is_temporarily_lost and not self._lost_published:
            print(f"[PRESENCE] Face temporarily lost: {identity_id} (waiting for return)")
            return None

        return None

    def _handle_known_face(self, result: dict, now: float) -> dict | None:
        face_id = result["id"]
        name = result.get("name", "unknown")
        confidence = result.get("confidence", 0)

        self._face_present = True
        self._current_identity_id = face_id
        self._current_name = name
        self._current_confidence = confidence
        self._unknown_published = False
        self._unknown_detection_start = None
        self._flicker_unknown_start = None

        entry = self._cooldown.get(face_id)
        entry.mark_detected()

        if not entry.is_stable:
            return None

        if face_id != self._last_published_identity:
            self._lost_published = False
            self._last_published_identity = face_id
            print(f"[PRESENCE] Known face stable: {name} (conf={confidence})")

            from app.face.person_memory import get_memory
            mem = get_memory().get(face_id, name)

            events = [{
                "event": "face_detected",
                "id": face_id,
                "name": name,
                "confidence": confidence,
                "memory": mem,
                "stable": True,
            }]

            if entry.can_greet:
                if entry.can_republish:
                    entry.mark_published()
                    events.append({
                        "event": "face_recognized",
                        "id": face_id,
                        "name": name,
                        "confidence": confidence,
                        "memory": mem,
                    })
                else:
                    print(f"[PRESENCE] Face known but re-publish suppressed (cooldown): {name}")

            return events

        return None

    def _handle_unknown_face(self, now: float) -> dict | None:
        self._face_present = True

        if self._current_identity_id is not None:
            if self._flicker_unknown_start is None:
                self._flicker_unknown_start = now
                return None
            if (now - self._flicker_unknown_start) < KNOWN_TO_UNKNOWN_FLICKER_TOLERANCE:
                return None
            print(
                f"[PRESENCE] Face changed to UNKNOWN (was {self._current_identity_id}) "
                f"after {KNOWN_TO_UNKNOWN_FLICKER_TOLERANCE:.1f}s of continuous unknown"
            )
            self._current_identity_id = None
            self._current_name = ""
            self._lost_published = False
            self._last_published_identity = None
            self._flicker_unknown_start = None

        if self._unknown_detection_start is None:
            self._unknown_detection_start = now

        if not self._unknown_published and (now - self._unknown_detection_start) >= STABLE_DETECTION_SECONDS:
            self._unknown_published = True
            print(f"[PRESENCE] Unknown face stable for {STABLE_DETECTION_SECONDS}s")
            return [
                {"event": "face_detected", "unknown": True, "stable": True},
                {"event": "face_unknown", "stable": True},
            ]

        return None

    @property
    def current_identity_id(self) -> str | None:
        return self._current_identity_id

    @property
    def current_name(self) -> str:
        return self._current_name

    @property
    def face_present(self) -> bool:
        return self._face_present

    @property
    def last_published_identity(self) -> str | None:
        return self._last_published_identity

    def reset(self):
        self._current_identity_id = None
        self._current_name = ""
        self._face_present = False
        self._last_published_identity = None
        self._lost_published = True
        self._unknown_published = False
        self._unknown_detection_start = None
        self._flicker_unknown_start = None
