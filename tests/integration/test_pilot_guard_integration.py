"""Exercise durable closure against an executable provider boundary, without rent."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from forge.evaluation.pilot_guard import supervise
from forge.protocol.hashing import hash_canonical


def test_desktop_delivery_acknowledges_real_process_and_reuses_durable_receipt(tmp_path):
    desktop = tmp_path / "notify-send"
    desktop.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "with Path(__file__).with_suffix('.calls').open('a') as f: f.write('accepted\\n')\n"
        "print(42)\n"
    )
    desktop.chmod(0o700)
    event = {
        "event": "BACKUP_FAILED_AWAITING_REPAIR", "ownership_id": "a" * 64,
        "pod_id": "owned-pod", "backup_attempt": 1,
        "repair_deadline_unix": time.time() + 30,
    }
    event["alert_id"] = hash_canonical("forge/custody-alert/v1", event)
    receipts = tmp_path / "receipts"
    argv = [sys.executable, "-m", "forge.evaluation.pilot_notify", "--receipts", str(receipts),
            "--notify-executable", str(desktop)]
    for _ in range(2):
        result = subprocess.run(argv, input=json.dumps(event), text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"status": "DELIVERED", "alert_id": event["alert_id"]}
    receipt_path = receipts / f"{event['alert_id']}.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["notification_id"] == 42 and receipt["human_reading_verified"] is False
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert desktop.with_suffix(".calls").read_text() == "accepted\n"
    event["pod_id"] = "foreign-pod"
    invalid = subprocess.run(argv, input=json.dumps(event), text=True, capture_output=True)
    assert invalid.returncode != 0 and not invalid.stdout
    assert desktop.with_suffix(".calls").read_text() == "accepted\n"
    # Exit zero alone is not delivery: a desktop must return a positive notification ID.
    desktop.write_text(f"#!{sys.executable}\nprint('not-an-ack')\n")
    event["alert_id"] = hash_canonical(
        "forge/custody-alert/v1", {key: value for key, value in event.items() if key != "alert_id"}
    )
    invalid = subprocess.run(argv, input=json.dumps(event), text=True, capture_output=True)
    assert invalid.returncode != 0 and not (receipts / f"{event['alert_id']}.json").exists()


@pytest.mark.parametrize(
    "prefix", ["attelier-evaluator-pilot-", "attelier-evaluator-repeatability-"]
)
def test_owned_pod_closure_restart_and_failed_archive(tmp_path, prefix):
    provider = tmp_path / "provider"
    provider.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\nfrom pathlib import Path\n"
        "root=Path(__file__).parent\n"
        "pods=json.loads((root/'pods.json').read_text())\n"
        "op=sys.argv[2]\n"
        "with (root/'requests').open('a') as f: f.write(op+'\\n')\n"
        "if op=='list': print(json.dumps(pods))\n"
        "elif op=='get': print(json.dumps(next(p for p in pods if p['id']==sys.argv[3])))\n"
        "elif op=='delete':\n"
        " (root/'pods.json').write_text(json.dumps([p for p in pods if p['id']!=sys.argv[3]]))\n"
        " print('{}')\n"
    )
    provider.chmod(0o700)
    archive = tmp_path / "archive.py"
    archive.write_text("raise SystemExit(1)\n")
    config = json.loads(
        (Path(__file__).parents[2] / "configs/evaluator-pilot-guard-envelope.json").read_text()
    )
    config.update(
        cap_usd=3.0,
        compute_usd_hour=3600,
        api_limit_usd=0.01,
        ancillary_limit_usd=0.01,
        contingency_usd=0.01,
        backup_seconds=0.2,
        closure_seconds=0.4,
        poll_seconds=0.02,
        provider_timeout_seconds=0.08,
    )
    notify = tmp_path / "notify.py"
    notify.write_text(
        "import json,sys; from pathlib import Path; e=json.load(sys.stdin); "
        "Path(__file__).with_suffix('.observed.json').write_text(json.dumps(e)); "
        "print(json.dumps({'status':'DELIVERED','alert_id':e['alert_id']}))"
    )
    owned = {"id": "pilot-owned", "name": prefix + "integration"}
    other = {"id": "unrelated", "name": "do-not-touch"}
    pods_path = tmp_path / "pods.json"
    pods_path.write_text(json.dumps([owned, other]))
    ownership = {
        "pod_id": owned["id"],
        "name": owned["name"],
        "worklist_id": "frozen-work",
        "envelope": config,
        "request_started_unix": time.time(),
        "terminate_on_backup_failure_authorized": True,
        "archive_argv": [sys.executable, str(archive)],
        "repair_notify_argv": [sys.executable, str(notify)],
    }
    state_path = tmp_path / "guard.json"
    state_path.with_suffix(".close").touch()
    assert supervise(ownership, state_path, provider=str(provider)) == 0
    state = json.loads(state_path.read_text())
    assert state["status"] == "CLOSED" and state["backup_verified"] is False
    assert state["closure_reason"] == "BUDGET_DEADLINE_WITH_UNVERIFIED_BACKUP"
    assert (
        state["delivered_alert_id"]
        == json.loads(notify.with_suffix(".observed.json").read_text())["alert_id"]
    )
    assert state["compute_exposure_usd"] > 1.0  # Preserved beyond the old 0.2s repair timer.
    diagnostic = json.loads(Path(state["backup_diagnostics"]).read_text())
    assert diagnostic["exit_code"] == 1 and diagnostic["timed_out"] is False
    assert json.loads(pods_path.read_text()) == [other]
    calls = (tmp_path / "requests").read_text()
    assert calls.endswith("delete\nlist\n")
    assert supervise(ownership, state_path, provider=str(provider)) == 0
    assert (tmp_path / "requests").read_text() == calls

    # A changed provider name fails closed even with the same pod identifier.
    ownership["request_started_unix"] = time.time() - 20_000
    state_path = tmp_path / "mismatch.json"
    state_path.with_suffix(".close").touch()
    pods_path.write_text(json.dumps([{**owned, "name": "another-campaign"}, other]))
    assert supervise(ownership, state_path, provider=str(provider)) == 2
    assert len(json.loads(pods_path.read_text())) == 2
    assert (tmp_path / "requests").read_text().count("delete\n") == 1
    assert json.loads(state_path.read_text())["status"] == "OPERATOR_RECONCILIATION_REQUIRED"

    # A resumed overdue supervisor still closes the pod instead of abandoning it.
    state_path = tmp_path / "overdue.json"
    pods_path.write_text(json.dumps([owned, other]))
    ownership["request_started_unix"] = time.time() - 20_000
    assert supervise(ownership, state_path, provider=str(provider)) == 0
    state = json.loads(state_path.read_text())
    assert state["status"] == "CLOSED" and state["deadline_exceeded"] is True
    assert state["backup_verified"] is False
    assert json.loads(pods_path.read_text()) == [other]

    # An early archive failure retains the owned pod for a bounded repair.
    # After reconciling/fixing the external archiver, an explicit marker retries it.
    ownership["request_started_unix"] = time.time()
    config.update(cap_usd=15, backup_seconds=0.2, poll_seconds=0.02)
    pods_path.write_text(json.dumps([owned, other]))
    ownership_path = tmp_path / "ownership.json"
    ownership_path.write_text(json.dumps(ownership))
    state_path = tmp_path / "repair.json"
    state_path.with_suffix(".close").touch()
    script = Path(__file__).parents[2] / "scripts/evaluator_pilot_guard.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "supervise",
            "--ownership",
            str(ownership_path),
            "--state",
            str(state_path),
            "--provider-executable",
            str(provider),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["status"] == "BACKUP_FAILED_AWAITING_REPAIR":
                    break
            assert process.poll() is None
            time.sleep(0.02)
        assert state["status"] == "BACKUP_FAILED_AWAITING_REPAIR"
        assert state["backup_exit_code"] == 1
        assert json.loads(pods_path.read_text()) == [owned, other]
        time.sleep(0.5)
        assert process.poll() is None and json.loads(pods_path.read_text()) == [owned, other]
        assert json.loads(notify.with_suffix(".observed.json").read_text())["pod_id"] == owned["id"]
        # Restart while awaiting repair: no new archive or shortened grace window.
        command = process.args
        process.terminate()
        process.communicate(timeout=5)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(0.3)
        resumed = json.loads(state_path.read_text())
        assert process.poll() is None and resumed["backup_attempts"] == 1
        assert resumed["backup_repair_deadline_unix"] == state["backup_repair_deadline_unix"]
        receipt = {
            "status": "VERIFIED",
            "pod_id": owned["id"],
            "worklist_id": "frozen-work",
            "readback_verified": True,
            "evidence_complete": True,
        }
        archive.write_text(f"print({json.dumps(receipt)!r})\n")
        state_path.with_suffix(".retry-backup").touch()
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr
        state = json.loads(state_path.read_text())
        assert state["status"] == "CLOSED" and state["backup_verified"] is True
        assert state["backup_attempts"] == 2
        assert json.loads(pods_path.read_text()) == [other]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
