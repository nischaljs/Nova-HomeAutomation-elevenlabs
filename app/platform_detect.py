import os
import platform


def _read_dt_model() -> str:
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore").strip("\x00 \n")
        except (FileNotFoundError, PermissionError):
            continue
    return ""


DT_MODEL = _read_dt_model()
ARCH = platform.machine()
IS_PI = "raspberry pi" in DT_MODEL.lower()
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def describe() -> str:
    if IS_PI:
        return f"Raspberry Pi ({DT_MODEL}, arch={ARCH})"
    return f"laptop/desktop (arch={ARCH})"
