#!/usr/bin/env bash
set -euo pipefail

# One-command native installer for the Attelier SN126 Validator (Python 3.12+, OpenSSL 3.0+).
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

mkdir -p "$REPO_ROOT/state/check"
echo "Running walletless dry-run check against Finney SN126..."
.venv/bin/attelier-validator --role validator --im api --dry-run --once --state-dir "$REPO_ROOT/state/check"

echo ""
echo "Dry-run succeeded! Result saved to state/check/live-dry-run.json:"
cat "$REPO_ROOT/state/check/live-dry-run.json"
echo ""
echo ""
echo "Next — start live weight publication (replace WALLET and HOTKEY):"
echo ""
echo "  # Option A (PM2):"
echo "  pm2 start $REPO_ROOT/.venv/bin/attelier-validator \\"
echo "    --name attelier-sn126-validator --interpreter none -- \\"
echo "    --role validator --im api --wallet.name WALLET --wallet.hotkey HOTKEY \\"
echo "    --state-dir $REPO_ROOT/state/validator"
echo "  pm2 save"
echo ""
echo "  # Option B (Docker Compose):"
echo "  cp .env.validator.example .env && nano .env"
echo "  docker compose -f docker-compose.validator.yml up -d --build"
