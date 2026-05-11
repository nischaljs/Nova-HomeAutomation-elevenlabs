import time
from dataclasses import dataclass, field


STABLE_DETECTION_SECONDS = 2.0
UNKNOWN_STABLE_SECONDS = 2.0
FACE_TRULY_LOST_SECONDS = 5.0


@dataclass
class _KnownState:
    face_id: str
    name: str
    first_seen: float
    last_seen: float
    confidence: float = 0.0
    published: bool = False
    lost_published: bool = False


class FacePresenceTracker:
    """Multi-face presence tracker.

    Each poll, the FaceMonitor hands us a list of recognized faces (zero
    or more known identities + zero or more `unknown` placeholders).
    The tracker emits per-person events when a face becomes stable
    (≥ STABLE_DETECTION_SECONDS continuously), and when it leaves
    (≥ FACE_TRULY_LOST_SECONDS without being seen).

    Unknown faces don't have IDs, so they're tracked in aggregate: a
    single `face_unknown` event fires once unknowns are stable, with the
    current count.
    """

    def __init__(self):
        self._known: dict[str, _KnownState] = {}
        self._unknown_count: int = 0
        self._unknown_first_seen: float | None = None
        self._unknown_published: bool = False

    def update(self, faces: list[dict]) -> list[dict]:
        now = time.time()
        events: list[dict] = []
        seen_known_ids: set[str] = set()
        unknown_count_this_poll = 0

        for face in faces:
            if face.get("unknown"):
                unknown_count_this_poll += 1
                continue
            face_id = face.get("id")
            if not face_id:
                continue
            seen_known_ids.add(face_id)
            state = self._known.get(face_id)
            if state is None:
                state = _KnownState(
                    face_id=face_id,
                    name=face.get("name", "unknown"),
                    first_seen=now,
                    last_seen=now,
                    confidence=face.get("confidence", 0.0),
                )
                self._known[face_id] = state
            else:
                state.last_seen = now
                state.confidence = face.get("confidence", state.confidence)
                state.name = face.get("name", state.name)
                if state.lost_published:
                    state.lost_published = False
                    state.first_seen = now
                    state.published = False

            if not state.published and (now - state.first_seen) >= STABLE_DETECTION_SECONDS:
                state.published = True
                events.append({
                    "event": "face_recognized",
                    "id": state.face_id,
                    "name": state.name,
                    "confidence": state.confidence,
                })

        self._unknown_count = unknown_count_this_poll
        if unknown_count_this_poll > 0:
            if self._unknown_first_seen is None:
                self._unknown_first_seen = now
            if (
                not self._unknown_published
                and (now - self._unknown_first_seen) >= UNKNOWN_STABLE_SECONDS
            ):
                self._unknown_published = True
                events.append({"event": "face_unknown", "count": unknown_count_this_poll})
        else:
            self._unknown_first_seen = None
            self._unknown_published = False

        for face_id, state in list(self._known.items()):
            if face_id in seen_known_ids:
                continue
            if (now - state.last_seen) >= FACE_TRULY_LOST_SECONDS:
                if state.published and not state.lost_published:
                    state.lost_published = True
                    events.append({
                        "event": "face_lost",
                        "id": state.face_id,
                        "name": state.name,
                    })
                if state.lost_published:
                    del self._known[face_id]

        return events

    def current_state(self) -> dict:
        """Snapshot of who is currently present and stable.

        Returns:
            {
              "known": [{id, name, confidence}, ...],   # stable known identities
              "unknown_count": int,                      # stable unknown count
            }
        """
        return {
            "known": [
                {"id": s.face_id, "name": s.name, "confidence": s.confidence}
                for s in self._known.values()
                if s.published and not s.lost_published
            ],
            "unknown_count": self._unknown_count if self._unknown_published else 0,
        }

    @property
    def current_identity_id(self) -> str | None:
        for s in self._known.values():
            if s.published and not s.lost_published:
                return s.face_id
        return None

    @property
    def face_present(self) -> bool:
        return bool(self._known) or self._unknown_count > 0

    def reset(self):
        self._known.clear()
        self._unknown_count = 0
        self._unknown_first_seen = None
        self._unknown_published = False
