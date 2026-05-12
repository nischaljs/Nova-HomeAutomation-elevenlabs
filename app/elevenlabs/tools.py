import os
import time


REGISTER_DURATION_S = 2.5
REGISTER_GAP_S = 0.2


# The agent matches the visitor's language (Devanagari / Romanized
# Nepali / English) and we don't want to hard-code translations here,
# so failure responses are written as language-neutral *directives*.
# ElevenLabs' LLM reads them as tool output and produces the actual user-
# facing sentence in whatever language the conversation is in.
_FAIL_RESPONSES = {
    "TOO_FAR": (
        "I tried to save them but their face was too far from the camera. "
        "Tell them — in their own language — to step a bit closer and stay "
        "in the frame, then I'll save it. Keep it warm and one sentence."
    ),
    "NO_FACE_VISIBLE": (
        "I tried to save them but no face was visible in the frame. "
        "Tell them — in their own language — to look straight at me and "
        "stay still for a second, then I'll save it. Keep it warm and one sentence."
    ),
    "BLURRY": (
        "I tried to save them but the frames were too blurry — they were "
        "moving, or the light is too dim. Tell them — in their own "
        "language — to hold still for one second so I can see them clearly. "
        "Keep it warm and one sentence."
    ),
    "TURNED_AWAY": (
        "I tried to save them but they were turned away from the camera. "
        "Tell them — in their own language — to face me directly so I can "
        "recognize them next time. Keep it warm and one sentence."
    ),
    "MODELS_NOT_READY": (
        "My face recognition is still warming up. Tell them — in their "
        "own language — to give me one more moment, I'll be ready shortly. "
        "Keep it warm and one sentence."
    ),
    "OTHER": (
        "I couldn't get a clear look at them. Tell them — in their own "
        "language — to face me directly and stay still for a moment, then "
        "I'll save it. Keep it warm and one sentence."
    ),
}


def _classify_register_fail(rejects: list[str]) -> str:
    """Bucket the per-frame reject reasons into one dominant cause so we
    can give the user one clear, actionable instruction (instead of a
    generic 'try again').

    `register_multi` records reject strings like 'too_small(40x40)',
    'low_score(0.55)', 'blurry(var=30)', 'off_axis(asym=0.62)',
    'no_face', 'models_not_ready'.
    """
    if not rejects:
        return "NO_FACE_VISIBLE"
    counts: dict[str, int] = {}
    for r in rejects:
        if r.startswith("too_small"):
            key = "TOO_FAR"
        elif r == "no_face":
            key = "NO_FACE_VISIBLE"
        elif r.startswith("low_score"):
            # YuNet wasn't confident the box was a face — usually means
            # too far / too dim / occluded. Same advice as TOO_FAR.
            key = "TOO_FAR"
        elif r.startswith("blurry"):
            key = "BLURRY"
        elif r.startswith("off_axis"):
            key = "TURNED_AWAY"
        elif r == "models_not_ready":
            key = "MODELS_NOT_READY"
        else:
            key = "OTHER"
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _register_user_impl(parameters: dict) -> str:
    name = (parameters or {}).get("name", "").strip()
    print(f"[TOOL] register_user invoked with name='{name}' params={parameters}")
    if not name:
        return "I didn't catch a name to register."

    if os.getenv("NOVA_SKIP_FACE") == "1":
        print(f"[TOOL] NOVA_SKIP_FACE=1 — pretending to register '{name}'")
        return f"Saved {name}. Nice to meet you!"

    from app.face.face_tools import FrameBuffer, get_bridge

    bridge = get_bridge()
    if not bridge.models_ready:
        print("[TOOL] register_user deferred — face models still loading")
        return ("My face recognition is still warming up — please ask me again "
                "in a moment, I'll have it ready.")

    # Crowd safety check — only refuse if MULTIPLE unknowns are visible
    # (can't tell which one's name we got). Zero-unknown snapshots are
    # treated as a transient miss; we proceed and let register_multi
    # capture frames over the next 1.5s. If it still can't find a face,
    # it'll return failure with a sensible message.
    snapshot = FrameBuffer().get_frame()
    if snapshot is not None:
        current = bridge.recognize_all(snapshot)
        unknown_count = sum(1 for f in current if f.get("unknown"))
        if unknown_count >= 2:
            print(f"[TOOL] register_user refused: {unknown_count} unknown faces visible "
                  f"— need them one at a time to attribute the name correctly")
            return ("There are multiple new visitors in front of me — could you come up "
                    "one at a time so I save the right name with the right face?")

    fb = FrameBuffer()
    frames = []
    t_end = time.time() + REGISTER_DURATION_S
    last_grab = 0.0
    while time.time() < t_end:
        now = time.time()
        if now - last_grab >= REGISTER_GAP_S:
            frame = fb.get_frame()
            if frame is not None:
                frames.append(frame.copy())
                last_grab = now
        time.sleep(0.05)

    if not frames:
        return "Camera isn't ready yet — try again in a moment."

    print(f"[TOOL] Captured {len(frames)} frames over {REGISTER_DURATION_S}s for '{name}'")
    result = bridge.register_multi(frames, name)
    if isinstance(result, dict) and result.get("ok"):
        used = result.get("samples_used", "?")
        print(f"[TOOL] Registered '{name}' from {used} usable samples")
        return f"Saved {name} successfully. Greet them warmly by name in your own language."

    rejects = (result or {}).get("rejects", []) if isinstance(result, dict) else []
    reason = _classify_register_fail(rejects)
    response = _FAIL_RESPONSES[reason]
    print(f"[TOOL] Registration of '{name}' failed — reason={reason} rejects={rejects[:8]}")
    return response


def build_client_tools():
    """Returns a ClientTools instance with `register_user` registered AND
    diagnostic logging on every tool-call attempt — including calls to
    tools that aren't registered (which would otherwise vanish silently).
    """
    from elevenlabs.conversational_ai.conversation import ClientTools

    class LoggingClientTools(ClientTools):
        def execute_tool(self, tool_name, parameters, callback):
            registered = list(self.tools.keys())
            if tool_name in self.tools:
                print(f"[TOOL] ◀ server requested tool '{tool_name}' (registered ✓) params={parameters}")
            else:
                print(f"[TOOL] ◀ server requested tool '{tool_name}' (NOT REGISTERED ✗) "
                      f"params={parameters}  | registered tools: {registered}")
            return super().execute_tool(tool_name, parameters, callback)

    tools = LoggingClientTools()
    tools.register("register_user", _register_user_impl)
    print(f"[TOOL] Client tools wired up: {list(tools.tools.keys())}")
    return tools
