"""Process-boundary archive verification with fake SSH and AWS executables."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evaluator_pilot_archive.py"
FAKE_SSH = """import os, subprocess, sys
command=sys.argv[1].replace('/workspace/forge',os.environ['FAKE_REMOTE'])
payload=sys.stdin.read().replace('/workspace/forge',os.environ['FAKE_REMOTE'])
raise SystemExit(subprocess.run(command,shell=True,input=payload,text=True).returncode)
"""
FAKE_AWS = """import os, sys
from pathlib import Path
root=Path(os.environ['FAKE_S3']); args=sys.argv[1:]
key=args[args.index('--key')+1] if '--key' in args else None
path=root/key if key else None
if args[:2] == ['s3api','head-object']:
    if path.exists(): raise SystemExit(0)
    print('An error occurred (404) when calling the HeadObject operation',file=sys.stderr)
    raise SystemExit(254)
if args[:3] == ['s3','cp','-']:
    uri=args[3]; target=root/uri.split('/',3)[3]
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(sys.stdin.buffer.read())
    raise SystemExit(0)
if args[:3] == ['s3','cp',args[2]] and args[2].startswith('s3://') and args[3] == '-':
    sys.stdout.buffer.write((root/args[2].split('/',3)[3]).read_bytes()); raise SystemExit(0)
raise SystemExit(2)
"""


def _tool(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path


def _inputs(tmp_path: Path, ssh: Path) -> tuple[Path, Path, Path]:
    remote = tmp_path / "remote"
    (remote / "logs").mkdir(parents=True)
    (remote / "archives").mkdir()
    (remote / "logs" / "result.json").write_text('{"result":"complete"}\n')
    ownership = tmp_path / "ownership.json"
    ownership.write_text(
        json.dumps({"pod_id": "pod-1", "worklist_id": "pilot-1", "ssh_argv": [str(ssh)]})
    )
    config = tmp_path / "archive.json"
    config.write_text(
        json.dumps(
            {
                "remote_archive": "archives/pilot-evidence.tar.gz",
                "s3_bucket": "example-private-archive",
                "s3_key": "subnet/experiments/pilot-1/evidence.tar.gz",
                "allowlist": ["logs/result.json"],
                "max_member_bytes": 1024,
                "max_archive_bytes": 10_000,
                "timeout_seconds": 30,
            }
        )
    )
    return ownership, config, tmp_path / "receipt.json"


def test_archive_streams_remote_evidence_and_writes_receipt_only_after_readback(tmp_path):
    fake_bin, object_root = tmp_path / "bin", tmp_path / "s3"
    fake_bin.mkdir()
    ssh = _tool(
        fake_bin / "ssh",
        FAKE_SSH,
    )
    _tool(
        fake_bin / "aws",
        FAKE_AWS,
    )
    ownership, config, receipt = _inputs(tmp_path, ssh)
    metric = "pilot-measurements/metrics/" + "a" * 64
    metric_root = tmp_path / "remote" / metric
    metric_root.mkdir(parents=True)
    for name in ("receipt.json", "metrics_eval_results.json", "metrics_full_info.json"):
        (metric_root / name).write_text("{}")
    configured = json.loads(config.read_text())
    configured["allowlist"].append(metric + "/receipt.json")
    model_sources = (
        "subnet/src/forge/models/lora_loading.py",
        "qualified-src/forge/models/lora_loading.py",
    )
    for model_source in model_sources:
        source_path = tmp_path / "remote" / model_source
        source_path.parent.mkdir(parents=True)
        source_path.write_text("# Runtime source, not model weights.\n")
        configured["allowlist"].append(model_source)
    config.write_text(json.dumps(configured))
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_S3": str(object_root),
        "FAKE_REMOTE": str(tmp_path / "remote"),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ownership",
            str(ownership),
            "--config",
            str(config),
            "--out",
            str(receipt),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "VERIFIED"
    assert saved["pod_id"] == "pod-1" and saved["worklist_id"] == "pilot-1"
    assert saved["readback_verified"] is True
    assert saved["archive_sha256"] == saved["sha256"] and saved["bytes"] > 0
    assert saved["configured_s3_key"].endswith("evidence.tar.gz")
    assert saved["uri"].endswith(".tar.gz")
    assert saved["configured_s3_key"] not in saved["uri"]
    assert receipt.with_suffix(".intent.json").is_file()
    assert saved["evidence_complete"] is True
    assert set(model_sources).issubset({member["path"] for member in saved["members"]})
    assert {
        metric + "/" + name
        for name in ("receipt.json", "metrics_eval_results.json", "metrics_full_info.json")
    }.issubset({member["path"] for member in saved["members"]})


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "secrets/token.json",
        "models/adapter.safetensors",
        "subnet/src/forge/models/config.json",
        "qualified-src/forge/models/adapter.safetensors",
    ],
)
def test_archive_rejects_secret_allowlist_before_process_or_receipt(tmp_path, forbidden_path):
    ssh = _tool(tmp_path / "ssh", "raise SystemExit(99)\n")
    ownership, config, receipt = _inputs(tmp_path, ssh)
    bad = json.loads(config.read_text())
    bad["allowlist"] = [forbidden_path]
    config.write_text(json.dumps(bad))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ownership",
            str(ownership),
            "--config",
            str(config),
            "--out",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not receipt.exists()


@pytest.mark.parametrize("bucket", ["https://example.invalid", "example/private", "1.2.3.4"])
def test_archive_rejects_non_bucket_destination_before_process_or_receipt(tmp_path, bucket):
    ssh = _tool(tmp_path / "ssh", "raise SystemExit(99)\n")
    ownership, config, receipt = _inputs(tmp_path, ssh)
    value = json.loads(config.read_text())
    value["s3_bucket"] = bucket
    config.write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ownership",
            str(ownership),
            "--config",
            str(config),
            "--out",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not receipt.exists()


def test_failed_archive_leaves_intent_that_blocks_a_duplicate_retry(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh = _tool(
        fake_bin / "ssh",
        FAKE_SSH,
    )
    ownership, config, receipt = _inputs(tmp_path, ssh)
    value = json.loads(config.read_text())
    value["max_archive_bytes"] = 1
    config.write_text(json.dumps(value))
    environment = {**os.environ, "FAKE_REMOTE": str(tmp_path / "remote")}
    command = [
        sys.executable,
        str(SCRIPT),
        "--ownership",
        str(ownership),
        "--config",
        str(config),
        "--out",
        str(receipt),
    ]
    first = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    second = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert first.returncode != 0 and second.returncode != 0
    assert "intent already exists" in second.stderr
    assert receipt.with_suffix(".intent.json").is_file()
    assert not receipt.exists()


def test_archive_keeps_existing_evidence_and_records_legitimate_missing_members(tmp_path):
    fake_bin, object_root = tmp_path / "bin", tmp_path / "s3"
    fake_bin.mkdir()
    ssh = _tool(
        fake_bin / "ssh",
        FAKE_SSH,
    )
    _tool(fake_bin / "aws", FAKE_AWS)
    ownership, config, receipt = _inputs(tmp_path, ssh)
    value = json.loads(config.read_text())
    value["allowlist"].append("metrics/pending.json")
    config.write_text(json.dumps(value))
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_S3": str(object_root),
        "FAKE_REMOTE": str(tmp_path / "remote"),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ownership",
            str(ownership),
            "--config",
            str(config),
            "--out",
            str(receipt),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    saved = json.loads(receipt.read_text())
    assert [member["path"] for member in saved["members"]] == ["logs/result.json"]
    assert saved["missing_members"] == ["metrics/pending.json"]
    assert saved["evidence_complete"] is False


def test_native_aws_cli_streams_stdin_to_a_local_s3_endpoint():
    if shutil.which("aws") is None:
        pytest.skip("AWS CLI is unavailable")

    received: dict[str, bytes] = {}

    class Endpoint(BaseHTTPRequestHandler):
        def do_PUT(self):  # noqa: N802 - HTTP handler spelling
            received["path"] = self.path.encode()
            received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("ETag", '"synthetic"')
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = b"synthetic archive stream\n"
        result = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                "-",
                "s3://archive-test/evidence.tar.gz",
                "--no-sign-request",
                "--endpoint-url",
                f"http://127.0.0.1:{server.server_port}",
                "--region",
                "us-east-1",
                "--only-show-errors",
                "--no-cli-pager",
            ],
            input=payload,
            capture_output=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert result.returncode == 0, result.stderr.decode()
    assert received == {"path": b"/archive-test/evidence.tar.gz", "body": payload}


def test_incremental_custody_blocks_next_batch_and_keeps_prior_backup_on_failure(
    tmp_path, monkeypatch
):
    """Real producer/controller/archive processes; SSH and object storage are external fixtures."""
    import hashlib
    import time

    from forge.validator.launch_remote_worker import _archive_pending

    remote, local, fake_bin, objects = [
        tmp_path / name for name in ("remote", "round", "bin", "s3")
    ]
    for path in (remote, local, fake_bin):
        path.mkdir()
    reconnecting_ssh = FAKE_SSH.replace(
        "payload=sys.stdin.read()",
        "if 'os.open(path' in command and os.environ.get('FAIL_ACK'): raise SystemExit(255)\n"
        "payload=sys.stdin.read()",
    )
    ssh = _tool(fake_bin / "ssh", reconnecting_ssh)
    failing_aws = FAKE_AWS.replace(
        "if args[:3] == ['s3','cp','-']:",
        "if args[:3] == ['s3','cp','-'] and os.environ.get('FAIL_UPLOAD'):\n"
        "    print('token=synthetic-should-never-be-logged',file=sys.stderr)\n"
        "    raise SystemExit(9)\n"
        "if args[:3] == ['s3','cp','-']:",
    )
    _tool(fake_bin / "aws", failing_aws)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_REMOTE", str(remote))
    monkeypatch.setenv("FAKE_S3", str(objects))
    root = remote / "campaign"
    root.mkdir()
    deadline = time.time() + 12
    config = {
        "ssh_argv": [str(ssh)],
        "generator_python": sys.executable,
        "work_deadline_unix": deadline,
        "custody": {
            "pod_id": "pod-1",
            "worklist_id": "pilot-1",
            "timeout_seconds": 5,
            "archive_bucket": "example-private-archive",
            "archive_argv": [sys.executable, str(SCRIPT)],
        },
    }
    producer = tmp_path / "produce.py"
    producer.write_text(
        "import sys; from pathlib import Path\n"
        "from forge.evaluation.pilot_custody import preserve_batch\n"
        "root=Path(sys.argv[1]); deadline=float(sys.argv[2])\n"
        "for batch in ('timing','quality'):\n"
        " d=root/batch; d.mkdir(); p=d/'video.mp4'; p.write_bytes(b'fixture video')\n"
        " r=d/'report.json'; r.write_text('{}')\n"
        " preserve_batch(root,batch,[p,r],deadline)\n"
        " (root/(batch+'.released')).touch()\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(producer), str(root), str(deadline)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            if (root / "custody/timing.request.json").exists():
                break
            time.sleep(0.02)
        assert not (root / "quality").exists()
        monkeypatch.setenv("FAIL_ACK", "1")
        with pytest.raises(RuntimeError, match="SSH command failed"):
            _archive_pending(config, "/workspace/forge/campaign", local)
        assert len(list(objects.rglob("*.tar.gz"))) == 1
        assert not (root / "timing.released").exists()
        monkeypatch.delenv("FAIL_ACK")
        # Restart reconciliation reuses the verified object without another upload.
        _archive_pending(config, "/workspace/forge/campaign", local)
        assert len(list(objects.rglob("*.tar.gz"))) == 1
        for _ in range(100):
            if (root / "custody/quality.request.json").exists():
                break
            time.sleep(0.02)
        assert (root / "timing.released").exists()
        assert (root / "custody/quality.request.json").exists()
        first = list(objects.rglob("*.tar.gz"))
        assert len(first) == 1
        first_hash = hashlib.sha256(first[0].read_bytes()).hexdigest()
        monkeypatch.setenv("FAIL_UPLOAD", "1")
        with pytest.raises(subprocess.CalledProcessError):
            _archive_pending(config, "/workspace/forge/campaign", local)
        assert not (root / "custody/quality.verified.json").exists()
        assert not (root / "quality.released").exists()
        assert hashlib.sha256(first[0].read_bytes()).hexdigest() == first_hash
        states = [json.loads(p.read_text()) for p in local.rglob("receipt.status.json")]
        assert any(
            s["status"] == "FAILED" and s["phase"] == "UPLOAD" and s["exit_code"] == 9
            for s in states
        )
        assert all(
            "synthetic-should-never-be-logged" not in p.read_text() for p in local.rglob("*.json")
        )
        # Failed transfer is not retried blindly, and never releases the worker.
        with pytest.raises(ValueError, match="intent exists"):
            _archive_pending(config, "/workspace/forge/campaign", local)
        assert len(list(objects.rglob("*.tar.gz"))) == 1
        # A foreign or stale acknowledgement cannot unblock the producer.
        (root / "custody/quality.verified.json").write_text(
            json.dumps(
                {
                    "request_hash": "0" * 64,
                    "readback_verified": True,
                    "evidence_complete": True,
                }
            )
        )
        _, stderr = process.communicate(timeout=3)
        assert process.returncode != 0 and "exact batch" in stderr
        assert not (root / "quality.released").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_running_job_recovers_custody_ssh_loss_without_repeating_work_or_upload(
    tmp_path, monkeypatch
):
    """Real detached producer and archiver; fail only the external SSH transport."""
    import time

    from forge.validator.launch_remote_worker import _archive_pending, _await_job
    from forge.validator.remote_job import DetachedRemoteJob

    remote, local, fake_bin, objects = [tmp_path / n for n in ("remote", "round", "bin", "s3")]
    for directory in (remote, local, fake_bin):
        directory.mkdir()
    ssh = _tool(fake_bin / "ssh", """import os, subprocess, sys
from pathlib import Path
command=sys.argv[1].replace('/workspace/forge',os.environ['FAKE_REMOTE'])
root=Path(os.environ['FAKE_REMOTE'])
if '*.request.json' in command and not (root/'read-disconnected').exists():
    (root/'read-disconnected').touch(); raise SystemExit(255)
payload=sys.stdin.read().replace('/workspace/forge',os.environ['FAKE_REMOTE'])
result=subprocess.run(command,shell=True,input=payload,text=True)
if 'os.open(path' in command and not (root/'ack-disconnected').exists():
    (root/'ack-disconnected').touch(); raise SystemExit(255)
raise SystemExit(result.returncode)
""")
    _tool(fake_bin / "aws", FAKE_AWS)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_REMOTE", str(remote))
    monkeypatch.setenv("FAKE_S3", str(objects))
    root = remote / "campaign"
    root.mkdir()
    deadline = time.time() + 60
    producer = tmp_path / "produce.py"
    producer.write_text(
        "import sys; from pathlib import Path\n"
        "from forge.evaluation.pilot_custody import preserve_batch\n"
        "r=Path(sys.argv[1]); (r/'executed-once').open('x').close()\n"
        "p=r/'video.mp4'; p.write_bytes(b'fixture video')\n"
        "preserve_batch(r,'timing',[p],float(sys.argv[2]))\n"
        "(r/'completed').touch()\n"
    )
    config = {
        "ssh_argv": [str(ssh)], "generator_python": sys.executable,
        "work_deadline_unix": deadline,
        "custody": {"pod_id": "pod-1", "worklist_id": "pilot-1", "timeout_seconds": 10,
                    "archive_bucket": "example-private-archive",
                    "archive_argv": [sys.executable, str(SCRIPT)]},
    }
    job = DetachedRemoteJob(
        ssh_argv=[str(ssh)], remote_dir=str(remote / "job"), local_state=local,
        job_id="one-original-job", deadline_unix=deadline, cwd=str(tmp_path), env={},
        stages=[[sys.executable, str(producer), str(root), str(deadline)]],
    )
    result = _await_job(
        job, custody=lambda: _archive_pending(config, "/workspace/forge/campaign", local)
    )
    assert result["status"] == "SUCCEEDED" and (root / "completed").exists()
    assert (remote / "read-disconnected").exists() and (remote / "ack-disconnected").exists()
    assert len(list(objects.rglob("*.tar.gz"))) == 1
    receipts = list((local / "media-custody").glob("*/receipt.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["readback_verified"]
    assert job.poll()["status"] == "SUCCEEDED"


def test_archive_timeout_persists_exact_phase_without_raw_error_output(tmp_path):
    ssh = _tool(tmp_path / "ssh", "import time; time.sleep(3)\n")
    ownership, config, receipt = _inputs(tmp_path, ssh)
    value = json.loads(config.read_text())
    value["timeout_seconds"] = 1
    config.write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ownership",
            str(ownership),
            "--config",
            str(config),
            "--out",
            str(receipt),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 1 and not receipt.exists()
    status = json.loads(receipt.with_suffix(".status.json").read_text())
    assert status["status"] == "FAILED" and status["phase"] == "REMOTE_ARCHIVE"
    assert status["timed_out"] is True and status["error_type"] == "TimeoutExpired"
    assert json.loads(result.stderr)["phase"] == "REMOTE_ARCHIVE"
    assert receipt.with_suffix(".status.json").stat().st_mode & 0o777 == 0o600
