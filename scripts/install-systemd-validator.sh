#!/usr/bin/env bash
set -euo pipefail

# Install and start the Attelier SN126 Validator as a systemd service (no Docker, no PM2).
#
# Usage:
#   sudo ./scripts/install-systemd-validator.sh          # install + enable + start
#   ./scripts/install-systemd-validator.sh --print       # print generated unit file without installing

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${SYSTEMD_SERVICE:-attelier-sn126-validator}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
PRINT_ONLY=0

if [ "${1:-}" = "--print" ]; then
  PRINT_ONLY=1
fi

if [ ! -x "$REPO_ROOT/.venv/bin/attelier-validator" ]; then
  echo "Error: $REPO_ROOT/.venv/bin/attelier-validator not found. Run ./scripts/install-validator.sh first." >&2
  exit 1
fi

if [ "$PRINT_ONLY" -eq 0 ]; then
  if [ "$(id -u)" -ne 0 ]; then
    echo "Error: run with sudo: sudo ./scripts/install-systemd-validator.sh" >&2
    exit 1
  fi
  if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "Error: $REPO_ROOT/.env not found. Copy .env.validator.example to .env and set WALLET_NAME / WALLET_HOTKEY." >&2
    exit 1
  fi
  if grep -qE '^WALLET_(NAME|HOTKEY)=(your_|<|$)' "$REPO_ROOT/.env"; then
    echo "Error: set real WALLET_NAME and WALLET_HOTKEY values in $REPO_ROOT/.env first." >&2
    exit 1
  fi
fi

UNIT_CONTENT="[Unit]
Description=Attelier SN126 Validator (Finney M0)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${REPO_ROOT}
Environment=HOME=${SERVICE_HOME}
ExecStart=${REPO_ROOT}/scripts/run-validator.sh
Restart=always
RestartSec=15
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
"

if [ "$PRINT_ONLY" -eq 1 ]; then
  printf '%s' "$UNIT_CONTENT"
  exit 0
fi

UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
printf '%s' "$UNIT_CONTENT" > "$UNIT_PATH"
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}.service"

echo "Installed and started ${SERVICE_NAME}.service (user=${SERVICE_USER})."
echo "Check status and logs:"
echo "  systemctl status ${SERVICE_NAME}.service"
echo "  journalctl -u ${SERVICE_NAME}.service -f"
