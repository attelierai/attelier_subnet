#!/usr/bin/env bash
set -euo pipefail

# One-command native installer for the Attelier SN126 Validator (no Docker required).
# Requirements: Python 3.12+ and OpenSSL 3.0+.
# Usage:
#   ./scripts/install-validator.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v openssl >/dev/null 2>&1; then
  echo "Error: openssl is required (OpenSSL 3.0+ with Ed25519 pkeyutl support)." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e '.[chain]'
else
  PYTHON_BIN="${PYTHON_BIN:-python3.12}"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  fi
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[chain]'
fi

if [ ! -f "$REPO_ROOT/.env" ] && [ -f "$REPO_ROOT/.env.validator.example" ]; then
  cp "$REPO_ROOT/.env.validator.example" "$REPO_ROOT/.env"
  echo "Created .env from .env.validator.example (edit WALLET_NAME and WALLET_HOTKEY before starting)."
fi

mkdir -p "$REPO_ROOT/state/check"
echo "Running walletless dry-run check against Finney SN126..."
.venv/bin/attelier-validator --role validator --im api --dry-run --once --state-dir "$REPO_ROOT/state/check"

echo ""
echo "Dry-run succeeded! Verified vector saved to state/check/live-dry-run.json:"
cat "$REPO_ROOT/state/check/live-dry-run.json"
echo ""
echo ""
echo "Next — edit .env (set WALLET_NAME and WALLET_HOTKEY), then start with ONE of:"
echo ""
echo "  # Option A — PM2 (no Docker):"
echo "  pm2 start $REPO_ROOT/scripts/run-validator.sh --name attelier-sn126-validator"
echo "  pm2 save"
echo ""
echo "  # Option B — systemd (no Docker, no Node/PM2):"
echo "  sudo $REPO_ROOT/scripts/install-systemd-validator.sh"
echo ""
echo "  # Option C — Foreground / tmux / screen (no Docker):"
echo "  $REPO_ROOT/scripts/run-validator.sh"
echo ""
echo "  # Option D — Docker Compose:"
echo "  docker compose -f docker-compose.validator.yml up -d --build"
