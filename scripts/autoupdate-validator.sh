#!/usr/bin/env bash
set -euo pipefail

# Autoupdate loop for the Attelier SN126 Validator (polls origin/main every 5 minutes).
# Supports Docker Compose (default) or PM2 (AUTOUPDATE_MODE=pm2).
#
# Environment overrides:
#   AUTOUPDATE_INTERVAL=300    Seconds between git checks (default 300)
#   AUTOUPDATE_BRANCH=main     Git branch to track (default main)
#   AUTOUPDATE_MODE=docker     docker | pm2 (default docker)
#   PM2_APP_NAME=attelier-sn126-validator

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${AUTOUPDATE_INTERVAL:-300}"
BRANCH="${AUTOUPDATE_BRANCH:-main}"
MODE="${AUTOUPDATE_MODE:-docker}"
PM2_APP="${PM2_APP_NAME:-attelier-sn126-validator}"

cd "$REPO_ROOT"
echo "[autoupdate] tracking origin/${BRANCH} every ${INTERVAL}s (mode=${MODE})"

while true; do
  if git fetch --quiet origin "$BRANCH"; then
    LOCAL="$(git rev-parse HEAD)"
    REMOTE="$(git rev-parse "origin/${BRANCH}")"
    if [ "$LOCAL" != "$REMOTE" ]; then
      echo "[autoupdate] updating ${LOCAL:0:7} -> ${REMOTE:0:7}"
      git pull --ff-only origin "$BRANCH"
      if [ "$MODE" = "docker" ]; then
        docker compose -f docker-compose.validator.yml up -d --build
      elif [ "$MODE" = "pm2" ]; then
        .venv/bin/python -m pip install --no-deps -e .
        pm2 restart "$PM2_APP"
      fi
      echo "[autoupdate] restarted cleanly at $(git rev-parse --short HEAD)"
    fi
  else
    echo "[autoupdate] git fetch failed; retrying in ${INTERVAL}s" >&2
  fi
  sleep "$INTERVAL"
done
