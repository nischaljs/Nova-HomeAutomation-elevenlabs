from app.face.context import build_face_context, build_unknown_context


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
