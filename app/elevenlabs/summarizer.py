"""Heuristic-only fallback for session memory.

The *primary* memory path is the `save_session_notes` client tool — the
ElevenLabs agent (which has the conversation in context already) calls
it inline whenever the visitor mentions something durable. That uses
the LLM minute Nova was already paying for, no second API, no extra
billing.

This module exists for the *abrupt-close* edge case: the WS dies,
network drops, or the user walks away silently before the agent ever
gets a chance to call the tool. In those cases, we fall back to a
pure-Python regex extractor over whatever transcript we managed to
capture from the user_transcript callbacks. Lower quality than the
agent's own summary, but better than losing the memory entirely.

No external LLM, no API keys. Runs offline.
"""

import re


MIN_TURNS_FOR_FALLBACK = 3
MAX_BULLETS = 5

# Regex patterns we look for in user turns. Each match emits one bullet.
# Conservative on purpose — high precision, low recall. We'd rather miss
# a fact than hallucinate one.
_HEURISTIC_PATTERNS = [
    (re.compile(
        r"(?:i\s+(?:like|love|enjoy|am\s+into|am\s+interested\s+in|"
        r"am\s+fascinated\s+by|am\s+passionate\s+about))\s+"
        r"([^.!?,;]+?)(?=[.!?,;]|$)",
        re.IGNORECASE,
    ), "interest"),
    (re.compile(
        r"(?:i\s+(?:work|study)\s+(?:as\s+|at\s+|in\s+))"
        r"([^.!?,;]+?)(?=[.!?,;]|$)",
        re.IGNORECASE,
    ), "fact"),
    (re.compile(
        r"(?:i\s+(?:live|stay)\s+(?:in\s+|at\s+))"
        r"([^.!?,;]+?)(?=[.!?,;]|$)",
        re.IGNORECASE,
    ), "fact"),
    (re.compile(
        r"(?:i\s+am\s+from\s+)([^.!?,;]+?)(?=[.!?,;]|$)",
        re.IGNORECASE,
    ), "fact"),
    (re.compile(
        r"(?:i(?:'m| am)\s+(?:working\s+on|building|making|trying\s+to))\s+"
        r"([^.!?,;]+?)(?=[.!?,;]|$)",
        re.IGNORECASE,
    ), "goal"),
    (re.compile(
        r"(?:i\s+prefer)\s+([^.!?,;]+?)(?=[.!?,;]|$)",
        re.IGNORECASE,
    ), "preference"),
]


def fallback_summarize(transcript: list[dict]) -> list[str]:
    """Extract bullet notes from a transcript when the agent didn't call
    save_session_notes itself. Returns a possibly-empty list.

    transcript items: {role: 'user'|'assistant', content: str}.
    """
    user_turns = [t for t in transcript if t.get("role") == "user"]
    if len(user_turns) < MIN_TURNS_FOR_FALLBACK:
        return []
    notes: list[str] = []
    seen: set[str] = set()
    for turn in transcript:
        if turn.get("role") != "user":
            continue
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        for pattern, kind in _HEURISTIC_PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(1).strip().rstrip(".")
                if len(value) < 3 or len(value) > 80:
                    continue
                if value.lower() in {"it", "this", "that", "him", "her", "them"}:
                    continue
                note = f"{kind}: {value}"
                key = note.lower()
                if key in seen:
                    continue
                seen.add(key)
                notes.append(note)
                if len(notes) >= MAX_BULLETS:
                    return notes
    return notes
