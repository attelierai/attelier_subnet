"""Deliver bounded custody alerts to the operator's real desktop notification server."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from forge.evaluation.pilot_guard import _write
from forge.protocol.hashing import hash_canonical


def deliver(event: dict, directory: Path, executable: Path) -> dict:
    fields = {
        "event", "ownership_id", "pod_id", "backup_attempt", "repair_deadline_unix", "alert_id",
    }
    if not isinstance(event, dict) or set(event) != fields:
        raise ValueError("invalid custody notification fields")
    if event["event"] not in {"BACKUP_FAILED_AWAITING_REPAIR", "CUSTODY_NOTIFICATION_PROBE"}:
        raise ValueError("unsupported custody event")
    if not isinstance(event["ownership_id"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", event["ownership_id"]
    ):
        raise ValueError("invalid ownership identity")
    if not isinstance(event["pod_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}", event["pod_id"]
    ):
        raise ValueError("invalid pod identity")
    if type(event["backup_attempt"]) is not int or event["backup_attempt"] < 1:
        raise ValueError("invalid backup attempt")
    deadline = event["repair_deadline_unix"]
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("invalid repair deadline")
    body = {key: value for key, value in event.items() if key != "alert_id"}
    if event["alert_id"] != hash_canonical("forge/custody-alert/v1", body):
        raise ValueError("custody alert identity mismatch")
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("notification command must be an existing absolute executable")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path = directory / f"{event['alert_id']}.json"
    with (directory / ".delivery.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("event") != event or receipt.get("status") != "DELIVERED":
                raise ValueError("existing notification receipt mismatch")
        else:
            probe = event["event"] == "CUSTODY_NOTIFICATION_PROBE"
            title = "Attelier: prueba de alerta" if probe else "Attelier: respaldo pendiente"
            message = (
                "Prueba de preparación de testnet. No hay GPU alquilada por esta prueba."
                if probe else
                f"El respaldo del pod {event['pod_id']} ha fallado. "
                "Se conserva hasta el límite de reparación del presupuesto; revisar el guard."
            )
            result = subprocess.run(
                [str(executable), "--print-id", "--urgency=critical", "--expire-time=0",
                 "--app-name=Attelier", title, message],
                capture_output=True, text=True, timeout=10, check=True,
            )
            notification_id = result.stdout.strip()
            if not notification_id.isdecimal() or not 0 < int(notification_id) < 2**32:
                raise ValueError("desktop did not acknowledge a notification ID")
            receipt = {
                "schema": "forge/custody-desktop-delivery/v1",
                "status": "DELIVERED", "event": event,
                "receiver": "org.freedesktop.Notifications",
                "notification_id": int(notification_id),
                "accepted_unix": time.time(), "human_reading_verified": False,
            }
            _write(receipt_path, receipt)
    return {"status": "DELIVERED", "alert_id": event["alert_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--notify-executable", type=Path, default=Path("/usr/bin/notify-send"))
    args = parser.parse_args()
    try:
        event = json.loads(sys.stdin.read(8193))
        print(json.dumps(deliver(event, args.receipts, args.notify_executable)))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "NOT_DELIVERED", "error_type": type(error).__name__}),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
