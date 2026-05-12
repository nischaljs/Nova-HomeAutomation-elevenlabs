#!/usr/bin/env bash
# One-shot Nova launcher. On first run it bootstraps everything; on every
# run after that it skips straight to launching the app.
#
# What it does:
#   1. cd to the script's own folder so it works from anywhere
#   2. Clone the sibling face-recognition repo if missing (and strip its .git)
#   3. Create .venv and install requirements if missing
#   4. Copy .env.example → .env if missing (and stop, since secrets need filling in)
#   5. Activate venv, set PYTHONUNBUFFERED, run, mirror logs to /tmp/nova-run.log

set -eu

cd "$(dirname "$0")"
ROOT="$(pwd)"
PARENT="$(dirname "$ROOT")"

# ── 1. Sibling face-recognition repo ──────────────────────────────────────
if [[ ! -d "$PARENT/face-recognition" ]]; then
  echo "[run.sh] Cloning face-recognition repo into $PARENT ..."
  git clone --depth 1 https://github.com/nischaljs/face-recognition.git "$PARENT/face-recognition"
  rm -rf "$PARENT/face-recognition/.git"
  echo "[run.sh] face-recognition cloned, .git stripped"
fi

# ── 2. Python venv + deps ─────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  echo "[run.sh] Creating .venv ..."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[run.sh] Installing dependencies (this takes ~2 minutes the first time)"
  pip install --upgrade pip --quiet
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # Re-sync if requirements.txt is newer than the venv's install marker.
  # Cheap check: stamp the venv after each install, redo install if requirements got newer.
  STAMP=.venv/.requirements.stamp
  if [[ ! -f $STAMP ]] || [[ requirements.txt -nt $STAMP ]]; then
    echo "[run.sh] requirements.txt changed — syncing dependencies ..."
    pip install -r requirements.txt
    touch $STAMP
  fi
fi

# ── 3. .env ───────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ""
  echo "[run.sh] No .env existed — created one from .env.example."
  echo "         Open .env and fill in:"
  echo "           ELEVENLABS_API_KEY=…"
  echo "           ELEVENLABS_AGENT_ID=…"
  echo "         Then re-run ./run.sh"
  exit 1
fi

# ── 4. Launch ─────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1

# Auto-detect headless. On a Pi running as a robot/kiosk there's no
# monitor, so DISPLAY and WAYLAND_DISPLAY are both empty. Without this,
# the preview thread's cv2.imshow aborts the process before the agent
# can come up.
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  export NOVA_HEADLESS=1
  echo "[run.sh] No display server detected — running headless (NOVA_HEADLESS=1)"
fi

# Log rotation: keep exactly two files on disk — the current run (which
# overwrites on launch) and the most recent previous run (renamed
# .prev). On a Pi with 32 GB SD card and a long-running deployment,
# unbounded logs would eventually fill the disk. Two files = bounded.
# We move (not truncate) so a crash log from the previous run survives
# in .prev, available for one debug cycle after the issue.
NOVA_LOG=/tmp/nova-run.log
if [[ -f "$NOVA_LOG" ]]; then
  mv -f "$NOVA_LOG" "${NOVA_LOG}.prev"
  echo "[run.sh] Rotated previous log → ${NOVA_LOG}.prev"
fi

exec python -m app.main 2>&1 | tee "$NOVA_LOG"
