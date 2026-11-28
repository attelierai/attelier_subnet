#!/usr/bin/env bash
set -euo pipefail

# Run the Attelier SN126 Validator natively (no Docker required).
# Loads .env from the repository root if present and executes .venv/bin/attelier-validator.
#
# Usage:
#   ./scripts/run-validator.sh                 # live run using .env
#   ./scripts/run-validator.sh --dry-run --once # walletless verification

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ATTELIER_ENV_FILE:-$REPO_ROOT/.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Map Compose-style directory overrides for seamless parity with .env.validator.example.
if [ -z "${WALLET_PATH:-}" ] && [ -n "${HOST_WALLETS_DIR:-}" ]; then
  WALLET_PATH="${HOST_WALLETS_DIR/#\~/$HOME}"
fi
if [ -z "${STATE_DIR:-}" ]; then
  if [ -n "${HOST_STATE_DIR:-}" ]; then
    STATE_DIR="${HOST_STATE_DIR/#\~/$HOME}"
    case "$STATE_DIR" in
      /*) ;;
      *) STATE_DIR="$REPO_ROOT/$STATE_DIR" ;;
    esac
  else
    STATE_DIR="$REPO_ROOT/state/validator"
  fi
fi

export VALIDATOR_ROLE="${VALIDATOR_ROLE:-validator}"
export VALIDATOR_IM="${VALIDATOR_IM:-api}"
export BITTENSOR_NETWORK="${BITTENSOR_NETWORK:-finney}"
export INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
export STATE_DIR
if [ -n "${WALLET_PATH:-}" ]; then
  export WALLET_PATH
fi

if [ ! -x "$REPO_ROOT/.venv/bin/attelier-validator" ]; then
  echo "Error: $REPO_ROOT/.venv/bin/attelier-validator not found. Run ./scripts/install-validator.sh first." >&2
  exit 1
fi

mkdir -p "$STATE_DIR"
exec "$REPO_ROOT/.venv/bin/attelier-validator" "$@"
