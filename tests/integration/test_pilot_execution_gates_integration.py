"""Exercise real CLI admission boundaries without GPU, credentials or API calls."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from forge.evaluation.pilot_worklist import pilot_worklist
from forge.protocol.hashing import hash_canonical


@pytest.mark.parametrize("deadline", ["0", "inf"])
def test_precision_probe_rejects_unbounded_or_expired_work_before_loading_gpu(tmp_path, deadline):
    root = Path(__file__).parents[2]
    output = tmp_path / "output"
    command = [sys.executable, str(root / "scripts/evaluator_pilot_precision_probe.py")]
    for name in ("pilot", "media-index", "media-root", "vbench-config"):
        command += ["--" + name, str(tmp_path / "absent")]
    command += ["--out", str(output), "--job-id", "a" * 64, "--deadline-unix", deadline]
    result = subprocess.run(command, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert b"a future deadline and external process timeout are required" in result.stderr
    assert not output.exists()


def test_judge_requires_closed_owned_pod_and_archived_measurement_before_dispatch(tmp_path):
    root = Path(__file__).parents[2]
    pilot = root / "configs/evaluator-pilot-manifest-v1.json"
    work = pilot_worklist(json.loads(pilot.read_text()))
    owner = {
        "pod_id": "local-integration-only",
        "worklist_id": work["worklist_id"],
        "envelope": {"api_limit_usd": 1.2},
    }
    evidence = {
        "ownership": owner,
        "closed": {
            "ownership_id": hash_canonical("forge/evaluator-pilot-ownership/v1", owner),
            "status": "CLOSING",
            "backup_verified": True,
        },
        "backup": {
            "schema": "forge/evaluator-pilot-archive/v1",
            "status": "VERIFIED",
            "pod_id": owner["pod_id"],
            "worklist_id": work["worklist_id"],
            "readback_verified": True,
            "evidence_complete": False,
            "members": [],
        },
        "measurement": {},
        "media-index": {},
        "review": {},
    }
    command = [
        sys.executable,
        str(root / "scripts/evaluator_pilot_judge.py"),
        "--pilot",
        str(pilot),
    ]
    for name, value in evidence.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        command += ["--" + name, str(path)]
    out = tmp_path / "result"
    command += ["--metrics", str(tmp_path / "metrics"), "--out", str(out)]
    result = subprocess.run(command, input=b"", capture_output=True, timeout=20)
    assert result.returncode != 0
    assert b"verified owned-pod closure and backup are required" in result.stderr
    assert not out.exists()
    evidence["closed"]["status"] = "CLOSED"
    (tmp_path / "closed.json").write_text(json.dumps(evidence["closed"]))
    result = subprocess.run(command, input=b"", capture_output=True, timeout=20)
    assert result.returncode != 0
    assert b"verified owned-pod closure and backup are required" in result.stderr
    assert not out.exists()
    evidence["backup"]["evidence_complete"] = True
    (tmp_path / "backup.json").write_text(json.dumps(evidence["backup"]))
    result = subprocess.run(command, input=b"", capture_output=True, timeout=20)
    assert result.returncode != 0
    assert b"must belong to the verified archive" in result.stderr
    assert not out.exists()


def test_archived_sheet_cli_streams_large_rows_and_rejects_changed_evidence(tmp_path):
    """Exercise real gzip/tar/filesystem boundaries, without GPU or paid dispatch."""
    import gzip
    import hashlib
    import io
    import tarfile

    payload = gzip.compress(
        (json.dumps({"slot_id": "large", "sheet": "a" * (35 * 1024**2)}) + "\n").encode()
    )
    archive = tmp_path / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as packed:
        member = tarfile.TarInfo("pilot-measurements/luna-payloads.jsonl.gz")
        member.size = len(payload)
        packed.addfile(member, io.BytesIO(payload))
    payload_hash = hashlib.sha256(payload).hexdigest()
    backup = {
        "status": "VERIFIED",
        "readback_verified": True,
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "members": [{"path": member.name, "bytes": len(payload), "sha256": payload_hash}],
    }
    receipt = tmp_path / "backup.json"
    receipt.write_text(json.dumps(backup))
    command = [
        sys.executable,
        "-m",
        "forge.evaluation.pilot_payloads",
        "--backup",
        str(receipt),
        "--payload-sha256",
        payload_hash,
        "--payload-bytes",
        str(len(payload)),
        "--max-rows",
        "1",
        "--local-archive",
        str(archive),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "VERIFIED", "rows": 1, "api_calls": 0}
    from forge.evaluation.pilot_payloads import archived_payloads

    buffered = io.BytesIO(archive.read_bytes())
    rows = list(archived_payloads(
        backup, payload_sha256=payload_hash, payload_bytes=len(payload),
        max_rows=1, archive_stream=buffered,
    ))
    assert len(rows) == 1 and rows[0]["slot_id"] == "large"
    assert buffered.closed
    backup["sha256"] = "0" * 64
    receipt.write_text(json.dumps(backup))
    changed = subprocess.run(command, capture_output=True, timeout=30)
    assert changed.returncode != 0 and b"complete archive differs" in changed.stderr
    backup["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    backup["members"][0]["sha256"] = "1" * 64
    receipt.write_text(json.dumps(backup))
    command[command.index("--payload-sha256") + 1] = "1" * 64
    wrong_payload = subprocess.run(command, capture_output=True, timeout=30)
    assert wrong_payload.returncode != 0 and b"compressed payload differs" in wrong_payload.stderr
