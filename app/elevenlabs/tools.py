import os
import time


REGISTER_DURATION_S = 3.0
REGISTER_GAP_S = 0.4


def _register_user_impl(parameters: dict) -> str:
    name = (parameters or {}).get("name", "").strip()
    print(f"[TOOL] register_user invoked with name='{name}' params={parameters}")
    if not name:
        return "I didn't catch a name to register."

    if os.getenv("NOVA_SKIP_FACE") == "1":
        print(f"[TOOL] NOVA_SKIP_FACE=1 — pretending to register '{name}'")
        return f"Saved {name}. Nice to meet you!"

    from app.face.face_tools import FrameBuffer, get_bridge

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
    result = get_bridge().register_multi(frames, name)
    if result:
        used = result.get("samples_used", "?")
        print(f"[TOOL] Registered '{name}' from {used} usable samples")
        return f"Got it — saved {name}. Nice to meet you!"
    print(f"[TOOL] Registration of '{name}' failed (not enough usable frames)")
    return "I couldn't get a clear look at you — try facing the camera and we'll try again."


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
