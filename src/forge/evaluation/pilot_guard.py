"""Owned-pod deadline and closure supervisor; never creates a paid resource.

The API and ancillary envelopes are withheld in full, even if their current
ledgers are empty. This prevents independent API dispatch from spending GPU
closure funds. Provider failure still requires operator intervention.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from forge.protocol.hashing import hash_canonical


def guard_envelope(config: dict) -> dict:
    fields = {
        "cap_usd",
        "api_limit_usd",
        "ancillary_limit_usd",
        "contingency_usd",
        "compute_usd_hour",
        "backup_seconds",
        "closure_seconds",
        "poll_seconds",
        "provider_timeout_seconds",
    }
    if set(config) != fields:
        raise ValueError("guard envelope fields differ from the complete contract")
    for key, value in config.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    withheld = sum(config[k] for k in ("api_limit_usd", "ancillary_limit_usd", "contingency_usd"))
    seconds = (config["cap_usd"] - withheld) / config["compute_usd_hour"] * 3600
    # list + get + delete + list, including a poll interval on each side.
    minimum_closure = 4 * config["provider_timeout_seconds"] + 2 * config["poll_seconds"]
    if config["closure_seconds"] < minimum_closure:
        raise ValueError("closure allowance cannot cover provider operations and polling")
    working = seconds - config["backup_seconds"] - config["closure_seconds"]
    if working <= 0:
        raise ValueError("no working time remains after reserved closure")
    return {
        "schema": "forge/evaluator-pilot-guard-envelope/v1",
        **config,
        "withheld_usd": withheld,
        "compute_limit_usd": config["cap_usd"] - withheld,
        "max_billable_seconds": seconds,
        "work_seconds": working,
        "warning_usd": 0.8 * config["cap_usd"],
        "completion_guaranteed": False,
    }


def _write(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _archive(argv: list[str], timeout: float, diagnostics: Path) -> str:
    # Killing only the archive parent can orphan its SSH/upload children.
    started = time.monotonic()
    output, errors = "", ""
    detail = {"timeout_seconds": timeout, "timed_out": False, "exit_code": None}
    with subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        try:
            output, errors = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            detail["timed_out"] = True
            os.killpg(process.pid, signal.SIGKILL)
            output, errors = process.communicate()
            raise
        finally:
            # Never persist untrusted stderr, argv, or exception messages. The
            # archiver emits a separate, allowlisted phase receipt on failure.
            detail.update(
                elapsed_seconds=time.monotonic() - started,
                exit_code=process.returncode,
                stdout_bytes=len(output.encode()),
                stderr_bytes=len(errors.encode()),
                stderr_sha256=hashlib.sha256(errors.encode()).hexdigest(),
            )
            for line in errors.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("phase") in {
                    "REMOTE_ARCHIVE",
                    "DESTINATION_CHECK",
                    "UPLOAD",
                    "READBACK",
                    "RECEIPT",
                }:
                    detail["archive_phase"] = event["phase"]
            _write(diagnostics, detail)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv)
        return output


def supervise(ownership: dict, state_path: Path, *, provider: str = "runpodctl") -> int:
    """Close only the explicitly owned pod and verify absence in the full list.

    `request_started_unix` must precede the create request. A backup program
    must emit {status:VERIFIED, pod_id:..., worklist_id:...} only after checking
    remote storage readback. Its stdout/stderr are never copied into receipts.
    This function neither authorizes dispatch nor implements an archive program.
    """
    envelope = guard_envelope(ownership["envelope"])
    for key in ("pod_id", "name", "worklist_id"):
        if not isinstance(ownership.get(key), str) or not ownership[key]:
            raise ValueError(f"missing ownership {key}")
    if not ownership["name"].startswith(
        ("attelier-evaluator-pilot-", "attelier-evaluator-repeatability-")
    ):
        raise ValueError("pod must belong to this named pilot")
    if ownership.get("terminate_on_backup_failure_authorized") is not True:
        raise ValueError("bounded closure after backup failure needs explicit authority")
    started = ownership["request_started_unix"]
    if (
        type(started) not in (int, float)
        or not math.isfinite(started)
        or not 0 < started <= time.time()
    ):
        raise ValueError("invalid create-request start time")
    archive = ownership["archive_argv"]
    if (
        not isinstance(archive, list)
        or not archive
        or any(not isinstance(arg, str) or not arg for arg in archive)
        or not Path(archive[0]).is_absolute()
    ):
        raise ValueError("archive must be a fixed absolute executable argv without secrets")
    notify = ownership.get("repair_notify_argv")
    if (
        not isinstance(notify, list)
        or not notify
        or any(not isinstance(arg, str) or not arg for arg in notify)
        or not Path(notify[0]).is_absolute()
    ):
        raise ValueError("repair_notify_argv must name a reviewed absolute notification command")
    identity = hash_canonical("forge/evaluator-pilot-ownership/v1", ownership)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "ownership_id": identity,
                "pod_id": ownership["pod_id"],
                "status": "MONITORING",
            }
        )
        if state.get("ownership_id") != identity:
            raise ValueError("existing supervisor state belongs to another ownership contract")
        if state["status"] == "CLOSED":
            return 0
        hard_deadline = started + envelope["max_billable_seconds"]
        backup_deadline = hard_deadline - envelope["closure_seconds"]
        work_deadline = backup_deadline - envelope["backup_seconds"]
        # A wall-clock rollback must not extend a running supervisor's deadline.
        wall_start, monotonic_start = time.time(), time.monotonic()

        def now() -> float:
            return max(time.time(), wall_start + time.monotonic() - monotonic_start)

        def save(status: str) -> None:
            warned = state.get("warning_80_percent", False)
            state.update(status=status, checked_unix=now(), hard_deadline_unix=hard_deadline)
            state["deadline_exceeded"] = now() >= hard_deadline
            state["compute_exposure_usd"] = (
                max(0, now() - started) * envelope["compute_usd_hour"] / 3600
            )
            state["warning_80_percent"] = (
                state["compute_exposure_usd"] + envelope["withheld_usd"] >= envelope["warning_usd"]
            )
            _write(state_path, state)
            if state["warning_80_percent"] and not warned:
                print(
                    json.dumps({"event": "BUDGET_80_PERCENT", "pod_id": ownership["pod_id"]}),
                    flush=True,
                )

        def call(*args: str):
            # Continue cleanup after an outage; abandoning an overdue pod would
            # leave it billing indefinitely. An overrun is visible in the receipt.
            timeout = envelope["provider_timeout_seconds"]
            result = subprocess.run(
                [provider, "pod", *args, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            return result.stdout

        def absent() -> bool:
            rows = json.loads(call("list", "--all"))
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not row.get("id") for row in rows
            ):
                raise ValueError("provider list is not a complete pod array")
            return all(row["id"] != ownership["pod_id"] for row in rows)

        def attempt_backup() -> None:
            state["backup_attempted"] = True
            state["backup_verified"] = False
            state["backup_attempts"] = state.get("backup_attempts", 0) + 1
            save("BACKING_UP")
            try:
                remaining = min(envelope["backup_seconds"], backup_deadline - now())
                if remaining <= 0:
                    raise TimeoutError("backup window elapsed")
                diagnostic = state_path.with_name(
                    f"{state_path.stem}.archive-{state['backup_attempts']}.json"
                )
                state["backup_diagnostics"] = str(diagnostic)
                receipt = json.loads(_archive(archive, remaining, diagnostic))
                state["backup_verified"] = (
                    isinstance(receipt, dict)
                    and receipt.get("status") == "VERIFIED"
                    and receipt.get("pod_id") == ownership["pod_id"]
                    and receipt.get("worklist_id") == ownership["worklist_id"]
                    and receipt.get("readback_verified") is True
                    and receipt.get("evidence_complete") is True
                )
            except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as error:
                # Record safe diagnostics, never provider output or arguments.
                state["backup_error_type"] = type(error).__name__
                if isinstance(error, subprocess.CalledProcessError):
                    state["backup_exit_code"] = error.returncode

        try:
            save("MONITORING")
            while (
                now() < work_deadline
                and not state.get("backup_attempted")
                and not state_path.with_suffix(".close").exists()
            ):
                time.sleep(max(0, min(envelope["poll_seconds"], work_deadline - now())))
                save("MONITORING")
            try:
                if absent():
                    save("CLOSED")
                    return 0
            except (OSError, subprocess.SubprocessError, TimeoutError):
                save("PROVIDER_RETRY_REQUIRED")
            if "backup_attempted" not in state:
                attempt_backup()
            if not state["backup_verified"]:
                # Early completion can leave hours of budget. A failed backup
                # must not immediately destroy evidence while repair is possible.
                # Never retry an uncertain transfer without explicit reconciliation.
                repair_deadline = backup_deadline
                state["backup_repair_deadline_unix"] = repair_deadline
                event = {
                    "event": "BACKUP_FAILED_AWAITING_REPAIR",
                    "ownership_id": identity,
                    "pod_id": ownership["pod_id"],
                    "backup_attempt": state["backup_attempts"],
                    "repair_deadline_unix": repair_deadline,
                }
                event["alert_id"] = hash_canonical("forge/custody-alert/v1", event)
                alert_path = state_path.with_suffix(".alert.json")
                _write(alert_path, event)
                save("BACKUP_FAILED_AWAITING_REPAIR")
                if state.get("delivered_alert_id") != event["alert_id"] and now() < repair_deadline:
                    try:
                        delivered = subprocess.run(
                            notify,
                            input=json.dumps(event),
                            text=True,
                            capture_output=True,
                            timeout=min(
                                envelope["provider_timeout_seconds"], repair_deadline - now()
                            ),
                            check=True,
                        )
                        answer = json.loads(delivered.stdout)
                        if answer != {"status": "DELIVERED", "alert_id": event["alert_id"]}:
                            raise ValueError("notification delivery not acknowledged")
                        state["delivered_alert_id"] = event["alert_id"]
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        state["notification_error_type"] = type(error).__name__
                retry = state_path.with_suffix(".retry-backup")
                while not state["backup_verified"] and now() < repair_deadline:
                    save("BACKUP_FAILED_AWAITING_REPAIR")
                    if retry.exists():
                        retry.unlink()
                        attempt_backup()
                    else:
                        time.sleep(max(0, min(envelope["poll_seconds"], repair_deadline - now())))
            save("CLOSING")
            if not state["backup_verified"]:
                state["closure_reason"] = "BUDGET_DEADLINE_WITH_UNVERIFIED_BACKUP"
            # Reconcile after a restart: deletion acknowledgement is not closure.
            while True:
                try:
                    if absent():
                        save("CLOSED")
                        return 0
                    pod = json.loads(call("get", ownership["pod_id"]))
                    if (
                        not isinstance(pod, dict)
                        or pod.get("id") != ownership["pod_id"]
                        or pod.get("name") != ownership["name"]
                    ):
                        raise ValueError("provider ownership mismatch")
                    state["delete_requested"] = True
                    save("CLOSING")
                    call("delete", ownership["pod_id"])
                    if absent():
                        save("CLOSED")
                        return 0
                except (OSError, subprocess.SubprocessError, TimeoutError):
                    save("CLOSURE_RETRY_REQUIRED")
                time.sleep(envelope["poll_seconds"])
        except (OSError, ValueError, subprocess.SubprocessError, TimeoutError):
            save("OPERATOR_RECONCILIATION_REQUIRED")
            return 2
