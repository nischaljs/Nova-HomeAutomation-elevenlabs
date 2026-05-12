#!/usr/bin/env bash
# One-shot updater for a deployed Nova.
#
# What it does (in order):
#   1. git pull (refuses to clobber local changes — explicit by design,
#      so an exhibition Pi with a tweaked .env doesn't lose it on update)
#   2. pip install -r requirements.txt — syncs any new Python deps
#   3. If `nova.service` is registered with systemd: systemctl restart it.
#      Otherwise, do nothing and tell the user how to relaunch.
#
# Run from the project root: ./update.sh

set -eu

cd "$(dirname "$0")"

# ── 1. Pull (safe: bail if there are local changes) ───────────────────────
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[update.sh] Local changes present — refusing to git pull."
  echo "             Either commit/stash them first, or run:"
  echo "                 git stash && ./update.sh && git stash pop"
  exit 1
fi

echo "[update.sh] Pulling latest from origin ..."
git pull --ff-only

# ── 2. Sync dependencies ──────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  echo "[update.sh] No .venv yet — running ./run.sh once will create it."
  echo "             Skipping pip sync for now."
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[update.sh] Syncing Python dependencies ..."
  pip install -r requirements.txt --upgrade --quiet
  # Update the stamp run.sh uses so it doesn't re-sync needlessly on
  # the next launch.
  touch .venv/.requirements.stamp
fi

# ── 3. Restart the service if installed ───────────────────────────────────
if systemctl list-units --type=service --all 2>/dev/null | grep -q '\bnova\.service\b'; then
  echo "[update.sh] nova.service detected — restarting via systemd ..."
  sudo systemctl restart nova
  echo "[update.sh] Done. Watch logs with: journalctl -u nova -f"
else
  echo "[update.sh] nova.service not installed under systemd."
  echo "             To make this Pi auto-start Nova on boot:"
  echo "                 sudo cp deploy/nova.service /etc/systemd/system/nova.service"
  echo "                 sudo systemctl daemon-reload"
  echo "                 sudo systemctl enable --now nova"
  echo "             For now, relaunch manually: ./run.sh"
fi
