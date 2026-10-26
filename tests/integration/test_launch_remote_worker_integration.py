"""CLI boundary checks for the custodial remote-worker adapter."""

import base64
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_remote_worker import _custody_destination, _fetch
from forge.validator.remote_job import DetachedRemoteJob


def test_remote_worker_rejects_mismatched_result_path_before_ssh(tmp_path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    result, wrong = round_dir / "worker-result.json", round_dir / "other-result.json"
    request = {
        "schema": "forge/launch-worker-request/v2",
        "round_id": "round-a",
        "plan_hash": "0" * 64,
        "round_dir": str(round_dir),
        "result_path": str(wrong),
    }
    envelope = {
        "request_hash": hash_canonical("forge/launch-worker-request/v1", request),
        "reservation_id": 1,
        "request": request,
    }
    request_path = round_dir / "worker-request.json"
    request_path.write_text(json.dumps(envelope))
    marker = tmp_path / "ssh-ran"
    config = {
        "ssh_argv": [str(tmp_path / "ssh")],
        "remote_root": "/workspace/forge/remote",
        "generator_python": "/workspace/forge/generator/bin/python",
        "reward_python": "/workspace/forge/reward/bin/python",
        "source_root": "/workspace/forge/source",
        "models": {},
        "contract_path": str(tmp_path / "contract.json"),
        "spec": "/workspace/forge/spec.json",
        "assets": "/workspace/forge/assets",
        "work_deadline_unix": time.time() + 60,
        "validators": {"v0": "hotkey"},
        "custody": {
            "pod_id": "owned",
            "worklist_id": "test",
            "archive_argv": [sys.executable],
            "archive_bucket": "example-private-archive",
            "timeout_seconds": 30,
        },
    }
    (tmp_path / "ssh").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (tmp_path / "ssh").chmod(0o700)
    config_path = tmp_path / "worker.json"
    config_path.write_text(json.dumps(config))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.validator.launch_remote_worker",
            "--config",
            str(config_path),
            "--request",
            str(request_path),
            "--result",
            str(result),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "result_path differs" in completed.stderr
    assert not marker.exists()

    config.pop("custody")
    config_path.write_text(json.dumps(config))
    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.validator.launch_remote_worker",
            "--config",
            str(config_path),
            "--request",
            str(request_path),
            "--result",
            str(result),
        ],
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0 and "config fields are invalid" in missing.stderr
    assert not marker.exists()


def test_receipt_fetch_uses_ssh_stdout_and_imports_only_indexed_json(tmp_path):
    files = {
        f"{kind}/receipt.json": b"{}"
        for kind in ("panels", "timing", "scores", "diversity", "diversity-panels", "qualification")
    }
    index = {
        "panels": ["panels/receipt.json"],
        "timing": ["timing/receipt.json"],
        "scores": ["scores/receipt.json"],
        "diversity": ["diversity/receipt.json"],
        "diversity_panels": ["diversity-panels/receipt.json"],
        "qualification": ["qualification/receipt.json"],
        "sha256": {},
    }
    source = tmp_path / "source"
    source.mkdir()
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        index["sha256"][relative] = sha256_file(path)
    index_response = {"receipt-index.json": base64.b64encode(json.dumps(index).encode()).decode()}
    payload_response = {key: base64.b64encode(value).decode() for key, value in files.items()}
    (tmp_path / "index-response.json").write_text(json.dumps(index_response))
    (tmp_path / "payload-response.json").write_text(json.dumps(payload_response))
    ssh = tmp_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        '*receipt-index.json*) cat "$FETCH_INDEX";;\n'
        '*) cat "$FETCH_PAYLOAD";;\n'
        "esac\n"
    )
    ssh.chmod(0o700)
    config = {
        "ssh_argv": [str(ssh)],
        "generator_python": "/workspace/forge/python",
        "work_deadline_unix": time.time() + 60,
    }
    import os

    old_index, old_payload = os.environ.get("FETCH_INDEX"), os.environ.get("FETCH_PAYLOAD")
    os.environ["FETCH_INDEX"], os.environ["FETCH_PAYLOAD"] = (
        str(tmp_path / "index-response.json"),
        str(tmp_path / "payload-response.json"),
    )
    try:
        destination = _custody_destination(tmp_path)
        assert not destination.exists()
        config["work_deadline_unix"] = time.time() - 1
        custody = _fetch(config, "/workspace/forge/remote", destination)
    finally:
        if old_index is None:
            os.environ.pop("FETCH_INDEX")
        else:
            os.environ["FETCH_INDEX"] = old_index
        if old_payload is None:
            os.environ.pop("FETCH_PAYLOAD")
        else:
            os.environ["FETCH_PAYLOAD"] = old_payload
    assert json.loads((custody / "receipt-index.json").read_text()) == index
    assert all((custody / relative).read_bytes() == content for relative, content in files.items())


def test_detached_remote_job_survives_ssh_loss_and_restart_without_replaying_stages(tmp_path):
    """The fake SSH process is the external transport boundary, not the job."""
    remote, local, counter = tmp_path / "remote", tmp_path / "local", tmp_path / "counter"
    ssh = tmp_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'n=$(cat "$REMOTE_SSH_COUNT" 2>/dev/null || echo 0)\n'
        'n=$((n+1)); echo "$n" > "$REMOTE_SSH_COUNT"\n'
        'last=""; for value in "$@"; do last="$value"; done\n'
        'if [ "$n" = 1 ]; then /bin/sh -c "$last"; echo 255 > "$REMOTE_SSH_EXIT"; exit 255; fi\n'
        'if [ "$n" = 2 ]; then exit 255; fi\n'
        'exec /bin/sh -c "$last"\n'
    )
    ssh.chmod(0o700)
    stage = [
        sys.executable,
        "-c",
        "import time; from pathlib import Path; time.sleep(0.3); p=Path('stages.log'); "
        "p.write_text(p.read_text() + 'render\\n' if p.exists() else 'render\\n')",
    ]
    score = [
        sys.executable,
        "-c",
        "from pathlib import Path; p=Path('stages.log'); p.write_text(p.read_text() + 'score\\n')",
    ]
    import os

    previous = os.environ.get("REMOTE_SSH_COUNT")
    previous_exit = os.environ.get("REMOTE_SSH_EXIT")
    os.environ["REMOTE_SSH_COUNT"] = str(counter)
    os.environ["REMOTE_SSH_EXIT"] = str(tmp_path / "ssh-exit")
    deadline = time.time() + 60
    try:
        job = DetachedRemoteJob(
            ssh_argv=[str(ssh)],
            remote_dir=str(remote),
            local_state=local,
            job_id="request-hash",
            deadline_unix=deadline,
            cwd=str(tmp_path),
            env={},
            stages=[stage, score],
        )
        assert job.poll()["status"] == "LAUNCHING"
        for _ in range(10):
            if (tmp_path / "ssh-exit").exists():
                break
            time.sleep(0.05)
        assert (tmp_path / "ssh-exit").read_text().strip() == "255"
        assert not (remote / "result.json").exists()
        for _ in range(30):
            if (remote / "result.json").exists():
                break
            time.sleep(0.1)
        assert json.loads((remote / "result.json").read_text())["status"] == "SUCCEEDED"
        restarted = DetachedRemoteJob(
            ssh_argv=[str(ssh)],
            remote_dir=str(remote),
            local_state=local,
            job_id="request-hash",
            deadline_unix=deadline,
            cwd=str(tmp_path),
            env={},
            stages=[stage, score],
        )
        assert restarted.poll()["status"] == "UNKNOWN_REMOTE"
        assert restarted.poll()["status"] == "SUCCEEDED"
        changed = DetachedRemoteJob(
            ssh_argv=[str(ssh)],
            remote_dir=str(remote),
            local_state=local,
            job_id="request-hash",
            deadline_unix=deadline,
            cwd=str(tmp_path),
            env={"CHANGED": "1"},
            stages=[stage, score],
        )
        with pytest.raises(ValueError, match="intent differs"):
            changed.poll()
    finally:
        if previous is None:
            os.environ.pop("REMOTE_SSH_COUNT", None)
        else:
            os.environ["REMOTE_SSH_COUNT"] = previous
        if previous_exit is None:
            os.environ.pop("REMOTE_SSH_EXIT", None)
        else:
            os.environ["REMOTE_SSH_EXIT"] = previous_exit
    assert (tmp_path / "stages.log").read_text().splitlines() == ["render", "score"]


def test_detached_remote_job_claims_one_start_when_local_clients_race(tmp_path):
    remote, local, launches = tmp_path / "remote", tmp_path / "local", tmp_path / "launches"
    ssh = tmp_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'last=""; for value in "$@"; do last="$value"; done\n'
        'if printf %s "$last" | grep -q start-claim; then\n'
        '  n=$(cat "$REMOTE_LAUNCHES" 2>/dev/null || echo 0); echo $((n+1)) > "$REMOTE_LAUNCHES"\n'
        "fi\n"
        'exec /bin/sh -c "$last"\n'
    )
    ssh.chmod(0o700)
    import os

    previous = os.environ.get("REMOTE_LAUNCHES")
    os.environ["REMOTE_LAUNCHES"] = str(launches)
    try:
        deadline = time.time() + 60

        def make_job():
            return DetachedRemoteJob(
                ssh_argv=[str(ssh)],
                remote_dir=str(remote),
                local_state=local,
                job_id="same-request",
                deadline_unix=deadline,
                cwd=str(tmp_path),
                env={},
                stages=[[sys.executable, "-c", "from pathlib import Path; Path('once').touch()"]],
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            states = list(executor.map(lambda _: make_job().poll()["status"], range(2)))
        assert set(states) <= {"LAUNCHING", "ALREADY_INTENDED", "UNKNOWN_REMOTE", "RUNNING"}
        for _ in range(30):
            if (remote / "result.json").exists():
                break
            time.sleep(0.1)
        assert json.loads((remote / "result.json").read_text())["status"] == "SUCCEEDED"
        assert launches.read_text().strip() == "1"
        assert make_job().poll()["status"] == "SUCCEEDED"
    finally:
        if previous is None:
            os.environ.pop("REMOTE_LAUNCHES", None)
        else:
            os.environ["REMOTE_LAUNCHES"] = previous


@pytest.mark.parametrize("stage", ["render", "score"])
def test_campaign_cli_requires_live_custody_deadline_before_gpu(stage, tmp_path):
    command = [
        sys.executable,
        "-m",
        "forge.evaluation.launch_campaign",
        stage,
        "--root",
        str(tmp_path / "no-models"),
    ]
    if stage == "score":
        command += ["--spec", str(tmp_path / "no-spec"), "--assets", str(tmp_path / "no-assets")]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode == 2 and "--custody-deadline" in missing.stderr
    invalid = subprocess.run(
        command + ["--custody-deadline", "nan"], capture_output=True, text=True
    )
    assert invalid.returncode != 0 and "finite and in the future" in invalid.stderr
    assert not list(tmp_path.iterdir())
