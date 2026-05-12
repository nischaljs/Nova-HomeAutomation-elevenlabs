import os

# Headless detection runs before cv2 import. On a Pi running as a robot
# (no monitor) DISPLAY and WAYLAND_DISPLAY are both unset, so we MUST NOT
# nudge Qt toward xcb — the xcb plugin would try to connect to a non-
# existent X server and SIGABRT the whole process the first time
# cv2.imshow is called. With a display present, force xcb because cv2's
# pip wheel ships only libqxcb.so (no Wayland plugin), so Wayland desktops
# would otherwise open a black window.
_HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if _HAS_DISPLAY:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
else:
    # offscreen plugin is the safe choice when nothing in the process is
    # supposed to render — it never tries to open a server connection.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # signal preview/etc. to stay disabled even if someone forgot NOVA_DEBUG=0
    os.environ.setdefault("NOVA_HEADLESS", "1")

import asyncio
import builtins
import datetime
import signal

import uvicorn
from fastapi import FastAPI

from app.platform_detect import describe as _platform_describe

from app.orchestration.orchestrator import Orchestrator
from config.config import PORT_API

_original_print = builtins.print


def _ts_print(*args, **kwargs):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]
    _original_print(f"{ts}", *args, **kwargs)


builtins.print = _ts_print

app = FastAPI()
orchestrator = Orchestrator()


@app.get("/api/check")
async def check():
    return {"message": "Nova is alive!"}


async def main():
    print(f"Nova starting (ElevenLabs branch) on {_platform_describe()} "
          f"display={'yes' if _HAS_DISPLAY else 'no — headless'}")
    await orchestrator.start()
    print(f"Liveness API on port {PORT_API}")
    print("Camera + Face Monitoring active. Agent will wake on engaged face.")

    api_server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=PORT_API, log_level="warning")
    )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _request_shutdown(signame: str):
        print(f"[MAIN] received {signame} — beginning clean shutdown")
        shutdown_event.set()
        api_server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except NotImplementedError:
            pass

    try:
        await api_server.serve()
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
