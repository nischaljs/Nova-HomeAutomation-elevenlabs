import os

# Force XCB before cv2 imports its bundled Qt. cv2's pip wheel ships only
# libqxcb.so — no Wayland plugin — so on Wayland desktops (Hyprland, GNOME-
# Wayland) the imshow window opens black. XCB routes through XWayland or
# native X11 and works on both Pi (X11/labwc) and laptop (Wayland). Harmless
# on platforms where Qt would have picked XCB anyway.
os.environ["QT_QPA_PLATFORM"] = "xcb"

import asyncio
import builtins
import datetime

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
    print(f"Nova starting (ElevenLabs branch) on {_platform_describe()}")
    await orchestrator.start()
    print(f"Liveness API on port {PORT_API}")
    print("Camera + Face Monitoring active. Agent listening on local mic.")

    api_server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=PORT_API, log_level="warning")
    )

    try:
        await api_server.serve()
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
