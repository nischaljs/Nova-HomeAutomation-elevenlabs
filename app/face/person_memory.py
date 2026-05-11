import json
import os
import time
from collections import OrderedDict

import re

MEMORY_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "face-recognition", "face_data", "memory"
)
MAX_CACHED = 50
SAVE_DEBOUNCE = 30


class PersonMemory:
    def __init__(self, memory_dir=MEMORY_DIR):
        self._dir = memory_dir
        os.makedirs(self._dir, exist_ok=True)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._dirty: set[str] = set()
        self._last_save: dict[str, float] = {}

    def _path(self, face_id: str) -> str:
        return os.path.join(self._dir, f"{face_id}.json")

    def get(self, face_id: str, name: str = "") -> dict:
        if face_id in self._cache:
            self._cache.move_to_end(face_id)
            return self._cache[face_id]
        mem = self._load(face_id)
        if not mem:
            mem = self._create(face_id, name)
        self._cache[face_id] = mem
        if len(self._cache) > MAX_CACHED:
            self._cache.popitem(last=False)
        return mem

    def _load(self, face_id: str) -> dict | None:
        path = self._path(face_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _create(self, face_id: str, name: str) -> dict:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        return {
            "face_id": face_id,
            "name": name,
            "first_met": now,
            "last_met": now,
            "visit_count": 1,
            "history": [],
            "notes": [],
        }

    def touch(self, face_id: str):
        mem = self._cache.get(face_id)
        if mem:
            mem["last_met"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            mem["visit_count"] += 1
            self._mark_dirty(face_id)

    def add_history(self, face_id: str, user_text: str, assistant_text: str):
        mem = self.get(face_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        mem.setdefault("history", []).append(
            {"role": "user", "content": user_text, "timestamp": now}
        )
        mem.setdefault("history", []).append(
            {"role": "assistant", "content": assistant_text, "timestamp": now}
        )
        if len(mem["history"]) > 20:
            mem["history"] = mem["history"][-20:]
        self._extract_notes(mem, user_text, assistant_text)
        self._mark_dirty(face_id)

    def get_summary(self, face_id: str) -> str:
        mem = self.get(face_id)
        parts = []
        parts.append(f"You are talking to {mem['name']}.")
        if mem.get("notes"):
            parts.append("Notes from past conversations: " + "; ".join(mem["notes"][-5:]))
        if mem.get("visit_count", 0) > 1:
            parts.append(f"This is their visit number {mem['visit_count']}.")
        history = mem.get("history", [])
        if len(history) >= 2:
            last = history[-2:]
            last_exchange = " ".join(
                f"{msg['role']}: {msg['content']}" for msg in last
            )
            parts.append(f"Last exchange: {last_exchange}")
        return " ".join(parts)

    def _extract_notes(self, mem: dict, user_text: str, assistant_text: str):
        text = user_text.lower()
        notes = mem.setdefault("notes", [])
        interest_patterns = [
            r"(?:i\s+(?:like|love|enjoy|am\s+into|am\s+interested\s+in))\s+(.+?)[\.!]",
            r"(?:i\s+(?:am\s+)?(?:fascinated|passionate)\s+(?:by|about))\s+(.+?)[\.!]",
        ]
        for pat in interest_patterns:
            for m in re.finditer(pat, text):
                note = f"interest: {m.group(1).strip()}"
                if note not in notes:
                    notes.append(note)
        fact_patterns = [
            r"(?:i\s+am\s+from)\s+(.+?)[\.!]",
            r"(?:i\s+(?:work|study|live))\s+(?:as\s+|at\s+|in\s+)?(.+?)[\.!]",
        ]
        for pat in fact_patterns:
            for m in re.finditer(pat, text):
                note = f"fact: {m.group(1).strip()}"
                if note not in notes:
                    notes.append(note)
        if len(notes) > 20:
            mem["notes"] = notes[-20:]

    def _mark_dirty(self, face_id: str):
        self._dirty.add(face_id)
        self._last_save[face_id] = time.time()

    async def flush(self):
        now = time.time()
        to_save = [
            fid for fid in self._dirty
            if now - self._last_save.get(fid, 0) >= SAVE_DEBOUNCE
        ]
        import asyncio
        for fid in to_save:
            mem = self._cache.get(fid)
            if mem:
                await asyncio.to_thread(self._write, fid, mem)
            self._dirty.discard(fid)

    def _write(self, face_id: str, mem: dict):
        path = self._path(face_id)
        try:
            with open(path, "w") as f:
                json.dump(mem, f, separators=(",", ":"))
        except OSError:
            pass

    async def flush_all(self):
        import asyncio
        for fid in list(self._dirty):
            mem = self._cache.get(fid)
            if mem:
                await asyncio.to_thread(self._write, fid, mem)
        self._dirty.clear()


PERSON_MEMORY = None


def get_memory():
    global PERSON_MEMORY
    if PERSON_MEMORY is None:
        PERSON_MEMORY = PersonMemory()
    return PERSON_MEMORY
