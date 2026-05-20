import os
import time
from dataclasses import dataclass, field


# How long a face_id must be seen continuously before we publish
# face_recognized (or unknown faces before face_unknown). Bumped from
# 2.0s to 3.5s for crowded / outdoor deployments where strangers
# briefly pass through the camera — at 2s a passer-by could trigger a
# context update to the ElevenLabs conversation about the wrong person.
# Env override (NOVA_STABLE_DETECTION_S) so the value can be tuned per
# deployment (kiosk in a quiet room vs robot at a fair) without rebuild.
# This is the SINGLE source of truth — cooldown_manager.py imports
# from here so a future change can't silently drift between the two.
STABLE_DETECTION_SECONDS = float(os.environ.get("NOVA_STABLE_DETECTION_S", "3.5"))
UNKNOWN_STABLE_SECONDS = float(os.environ.get("NOVA_UNKNOWN_STABLE_S", "3.5"))
FACE_TRULY_LOST_SECONDS = float(os.environ.get("NOVA_FACE_LOST_S", "5.0"))


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

    **Primary speaker lock.** Once the first face_recognized event fires
    in a session, that face becomes the "primary speaker" and the
    tracker's `current_state()` filters down to ONLY that person until
    they're truly lost. The orchestrator pushes `current_state()` to
    ElevenLabs as contextual updates — without the lock, a bystander
    walking by in a crowded scene gets stable, fires their own event,
    and the agent gets told mid-conversation "now there's also Bob
    here". With the lock, the conversation stays anchored to whoever
    Nova locked onto first. Lock releases automatically when the
    primary face is dropped from `_known` (i.e. after
    FACE_TRULY_LOST_SECONDS of absence). Other faces are still tracked
    internally — they just don't pollute the ElevenLabs context.
    """

    def __init__(self):
        self._known: dict[str, _KnownState] = {}
        self._unknown_count: int = 0
        self._unknown_first_seen: float | None = None
        self._unknown_published: bool = False
        # Primary speaker lock — see class docstring. None when no
        # speaker is locked (initial state or just released).
        self._primary_id: str | None = None
        self._primary_locked_at: float = 0.0

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

        # Primary speaker lock — acquire on the first face_recognized
        # event of the session (highest-confidence if multiple fired
        # this tick), release when the primary id is no longer tracked.
        if self._primary_id is not None and self._primary_id not in self._known:
            print(f"[FACE] primary speaker lock RELEASED (id={self._primary_id} "
                  f"truly lost after {now - self._primary_locked_at:.1f}s)")
            self._primary_id = None
            self._primary_locked_at = 0.0
        if self._primary_id is None:
            new_primary = None
            best_conf = -1.0
            for evt in events:
                if evt.get("event") != "face_recognized":
                    continue
                conf = float(evt.get("confidence", 0.0))
                if conf > best_conf:
                    best_conf = conf
                    new_primary = evt["id"]
            if new_primary:
                self._primary_id = new_primary
                self._primary_locked_at = now
                pname = self._known[new_primary].name if new_primary in self._known else "?"
                print(f"[FACE] primary speaker lock ACQUIRED: {pname} "
                      f"(id={new_primary}, conf={best_conf:.2f})")

        return events

    def current_state(self) -> dict:
        """Snapshot of who is currently present and stable.

        While a primary speaker is locked (see class docstring), the
        returned `known` list contains only the primary and
        `unknown_count` is forced to 0 — this is the snapshot the
        orchestrator pushes to ElevenLabs as a contextual update, and
        we don't want a passer-by to leak into that.

        Returns:
            {
              "known": [{id, name, confidence}, ...],   # stable known identities
              "unknown_count": int,                      # stable unknown count
            }
        """
        if self._primary_id is not None and self._primary_id in self._known:
            s = self._known[self._primary_id]
            if s.published and not s.lost_published:
                return {
                    "known": [{
                        "id": s.face_id,
                        "name": s.name,
                        "confidence": s.confidence,
                    }],
                    "unknown_count": 0,
                }
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
        self._primary_id = None
        self._primary_locked_at = 0.0
