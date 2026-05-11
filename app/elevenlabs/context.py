from app.face.context import build_face_context, build_unknown_context
from app.face.person_memory import get_memory


def face_info_to_context_text(face_info: dict | None) -> str:
    if face_info is None:
        return "The visitor has left. You can return to idle."

    if face_info.get("unknown"):
        return (
            build_unknown_context()
            + " Ask their name naturally, like a human would. "
            "When they tell you, call the register_user tool with their name."
        )

    base = build_face_context(face_info)
    if not base:
        return ""

    return base + " Greet them by name in your next turn. Keep it warm and brief."


def face_state_to_context_text(state: dict) -> str:
    """Build a natural multi-person context string from the tracker's current state.

    state = {"known": [{id, name, confidence}, ...], "unknown_count": int}
    """
    known = state.get("known") or []
    unknown_count = int(state.get("unknown_count") or 0)
    n_known = len(known)
    total = n_known + unknown_count

    if total == 0:
        return "All visitors have left. You can return to idle."

    mem = get_memory()
    described = []
    for person in known:
        face_id = person.get("id")
        name = person.get("name", "unknown")
        conf = person.get("confidence", 0)
        m = mem.get(face_id, name) if face_id else {}
        parts = [name]
        if m.get("visit_count", 0) > 1:
            parts.append(f"visited {m['visit_count']} times")
        history = m.get("history") or []
        if len(history) >= 1:
            last_user = next(
                (h for h in reversed(history) if h.get("role") == "user"), None
            )
            if last_user and last_user.get("content"):
                snippet = last_user["content"].strip()
                if len(snippet) > 80:
                    snippet = snippet[:77] + "…"
                parts.append(f'last said: "{snippet}"')
        described.append(" — ".join(parts) + f" (match {conf:.0%})")

    if n_known == 1 and unknown_count == 0:
        return (
            f"The person in front of you is {described[0]}. "
            "Greet them by name in your next turn. Keep it warm and brief."
        )

    if n_known == 0 and unknown_count == 1:
        return (
            build_unknown_context()
            + " Ask their name naturally, like a human would. "
            "When they tell you, call the register_user tool with their name."
        )

    if n_known == 0 and unknown_count >= 2:
        return (
            f"{unknown_count} unknown visitors are in front of you at once. "
            "Welcome them warmly, then ask them to come one at a time so you "
            "can save each name properly. Do NOT call register_user while "
            "multiple unknown people are visible — wait until one is alone."
        )

    if n_known >= 2 and unknown_count == 0:
        if n_known == 2:
            who = f"{described[0]} and {described[1]}"
        else:
            who = ", ".join(described[:-1]) + f", and {described[-1]}"
        return (
            f"{n_known} known visitors are here together: {who}. "
            f"Greet them both by name like you're saying hi to a pair of friends "
            f"who walked up together. Don't repeat any context aloud — just be natural."
        )

    if n_known >= 1 and unknown_count >= 1:
        known_phrase = (
            described[0] if n_known == 1
            else (f"{described[0]} and {described[1]}" if n_known == 2
                  else ", ".join(described[:-1]) + f", and {described[-1]}")
        )
        verb = "is" if n_known == 1 else "are"
        unknown_phrase = (
            "an unknown visitor with them" if unknown_count == 1
            else f"{unknown_count} unknown visitors with them"
        )
        if unknown_count == 1:
            register_rule = (
                " When the unknown one tells you their name, call register_user "
                "with that name."
            )
        else:
            register_rule = (
                " Multiple unknown visitors are present — DO NOT call register_user "
                "until they introduce themselves one at a time."
            )
        greet_target = "person" if n_known == 1 else "people"
        return (
            f"{known_phrase} {verb} here, and {unknown_phrase}. "
            f"Greet the known {greet_target} by name and warmly welcome the new visitor(s)."
            + register_rule
        )

    return ""
