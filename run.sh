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
exec python -m app.main 2>&1 | tee /tmp/nova-run.log
