from app.face.person_memory import get_memory


def build_face_context(face_info: dict | None) -> str:
    if face_info is None:
        return ""

    name = face_info.get("name", "unknown")
    confidence = face_info.get("confidence", 0)
    face_id = face_info.get("id", "")

    if not face_id:
        return ""

    mem = get_memory()
    person = mem.get(face_id, name)

    parts = []
    parts.append(f"The person in front of you is {person['name']}.")
    parts.append(f"Confidence: {confidence:.0%}")

    if person.get("notes"):
        notes = "; ".join(person["notes"][-3:])
        parts.append(f"Things you know about them: {notes}")

    if person.get("visit_count", 0) > 1:
        parts.append(f"They have visited {person['visit_count']} times before.")

    history = person.get("history", [])
    if len(history) >= 2:
        last = history[-1]
        parts.append(f"Last thing they said: \"{last['content']}\"")

    return " ".join(parts)


def build_unknown_context() -> str:
    return "The person in front of you is someone you haven't met before. Be warm and welcoming."
