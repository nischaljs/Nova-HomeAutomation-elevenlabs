from app.face.context import build_face_context, build_unknown_context
from app.face.person_memory import get_memory


# Confidence bands. The face matcher's threshold is 0.40 so anything we
# get back has at least that. Above 0.65 we're very sure — greet by name
# the way you'd greet a friend. Between 0.55 and 0.65 we're confident but
# leave a tiny opening for a polite "is that you?". Below 0.55 we ask a
# soft yes/no first so the visitor can correct us before Nova commits to
# a name. These ratios are deliberately conservative — getting someone's
# name wrong is a much worse error than asking once.
CONF_VERY_SURE = 0.65
CONF_LIKELY = 0.55

# Per-band directives passed *into the context update*. ElevenLabs'
# system prompt already says "match the user's language", so these are
# written as instructions to the LLM (it translates) rather than as
# scripted Nepali/English greetings.
_BAND_DIRECTIVE = {
    "very_sure": (
        " You're confident this is them — greet them by name warmly, "
        "no hedging."
    ),
    "likely": (
        " You're confident but not 100% — greet them by name and "
        "naturally check, like '<Name>, right?' Don't sound robotic, "
        "just a tiny human check-in."
    ),
    "guess": (
        " You're not fully sure it's them — start with a soft yes/no "
        "check, like 'You look like <Name> — am I right?' or "
        "'तपाईं <Name> हो कि?' so they can correct you if it's not them. "
        "Don't say 'my system thinks' or anything technical — just be a "
        "person who isn't quite sure."
    ),
}


def _confidence_band(conf: float) -> str:
    if conf >= CONF_VERY_SURE:
        return "very_sure"
    if conf >= CONF_LIKELY:
        return "likely"
    return "guess"


def _band_directive(known: list[dict]) -> str:
    """Pick the *least* confident person's band for the whole-scene
    directive — if any one of them is a guess, Nova should soften her
    greeting; calling one known visitor confidently and one tentatively
    in the same sentence is weirder than just being gentle with both."""
    if not known:
        return ""
    weakest = min(_confidence_band(p.get("confidence", 0)) for p in known)
    # min() over strings: 'guess' < 'likely' < 'very_sure' alphabetically,
    # which is exactly the "weakest band wins" order we want.
    return _BAND_DIRECTIVE.get(weakest, "")


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

    band = _confidence_band(face_info.get("confidence", 0))
    return (
        base
        + " Greet them by name in your next turn. Keep it warm and brief."
        + _BAND_DIRECTIVE[band]
    )


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
        band = _confidence_band(conf)
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
        described.append(" — ".join(parts) + f" (match {conf:.0%} — {band})")

    if n_known == 1 and unknown_count == 0:
        return (
            f"The person in front of you is {described[0]}. "
            "Greet them by name in your next turn. Keep it warm and brief."
            + _band_directive(known)
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
            + _band_directive(known)
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
            + _band_directive(known)
        )

    return ""
