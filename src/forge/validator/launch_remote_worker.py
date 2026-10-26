"""Custodial SSH adapter for one already-reserved Launch V2 worker request.

The operator JSON names an existing SSH destination and remote workspace; this
module never creates a pod, starts a provider resource, signs, or publishes.
It uploads only frozen JSON inputs, keeps videos on the worker, and imports the
small indexed JSON receipt set into the round's trusted custody.

Config fields are ``ssh_argv``, ``remote_root`` (under ``/workspace/forge``),
``generator_python``, ``reward_python``, ``source_root``, ``models`` (a local
JSON path or object), ``contract_path``, ``spec``, ``assets``,
``work_deadline_unix``, and ``validators`` (label to hotkey).  Paths after
``remote_root`` are remote except ``models`` and ``contract_path``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shlex
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from forge.evaluation.pilot_custody import BATCHES
from forge.evaluation.pilot_custody import SCHEMA as CUSTODY_SCHEMA
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_config import LaunchConfig
from forge.validator.launch_receipts import import_campaign
from forge.validator.launch_service import LaunchServiceStore, atomic_json, write_immutable_json
from forge.validator.remote_job import DetachedRemoteJob, RemoteJobError

REQUEST_SCHEMA = "forge/launch-worker-request/v1"
MAX_RECEIPT_BYTES = 2_000_000


class RemoteTransportError(RuntimeError):
    """SSH lost contact; the existing remote job must be reconciled."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _json_source(value: object, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return _read(Path(value))
    raise ValueError(f"{name} must be a JSON object or local JSON path")


def _remote_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a remote path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(PurePosixPath("/workspace/forge"))
    ):
        raise ValueError(f"{name} must stay under /workspace/forge")
    return str(path)


def _config(path: Path) -> dict[str, Any]:
    value = _read(path)
    required = {
        "ssh_argv",
        "remote_root",
        "generator_python",
        "reward_python",
        "source_root",
        "models",
        "contract_path",
        "spec",
        "assets",
        "work_deadline_unix",
        "validators",
        "custody",
    }
    if set(value) != required:
        raise ValueError("remote worker config fields are invalid")
    if (
        not isinstance(value["ssh_argv"], list)
        or not value["ssh_argv"]
        or any(not isinstance(item, str) or not item for item in value["ssh_argv"])
        or type(value["work_deadline_unix"]) not in (int, float)
        or not math.isfinite(value["work_deadline_unix"])
        or not isinstance(value["validators"], dict)
        or not value["validators"]
        or any(
            not isinstance(key, str) or not isinstance(item, str) or not item
            for key, item in value["validators"].items()
        )
    ):
        raise ValueError("remote worker config SSH, deadline or validators are invalid")
    for name in (
        "remote_root",
        "generator_python",
        "reward_python",
        "source_root",
        "spec",
        "assets",
    ):
        value[name] = _remote_path(value[name], name)
    custody = value["custody"]
    if (
        not isinstance(custody, dict)
        or set(custody)
        != {"pod_id", "worklist_id", "archive_argv", "archive_bucket", "timeout_seconds"}
        or type(custody["timeout_seconds"]) is not int
        or not 1 <= custody["timeout_seconds"] <= 900
        or not isinstance(custody["archive_argv"], list)
        or not custody["archive_argv"]
        or any(not isinstance(arg, str) or not arg for arg in custody["archive_argv"])
        or not Path(custody["archive_argv"][0]).is_absolute()
        or not isinstance(custody["archive_bucket"], str)
        or not 3 <= len(custody["archive_bucket"]) <= 63
        or not custody["archive_bucket"][0].isalnum()
        or not custody["archive_bucket"][-1].isalnum()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789.-"
            for character in custody["archive_bucket"]
        )
        or ".." in custody["archive_bucket"]
        or custody["archive_bucket"].replace(".", "").isdigit()
        or any(
            not isinstance(custody[key], str)
            or not custody[key]
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for c in custody[key]
            )
            for key in ("pod_id", "worklist_id")
        )
    ):
        raise ValueError("custody requires owned identities, archive command and bounded timeout")
    return value


def _envelope(request_path: Path, result_path: Path) -> tuple[dict[str, Any], Path]:
    envelope = _read(request_path)
    if set(envelope) != {"request_hash", "reservation_id", "request"} or not isinstance(
        envelope["request"], dict
    ):
        raise ValueError("worker envelope is invalid")
    request = envelope["request"]
    if envelope["request_hash"] != hash_canonical(REQUEST_SCHEMA, request):
        raise ValueError("worker request hash changed")
    if request.get("schema") != "forge/launch-worker-request/v2":
        raise ValueError("worker request schema is invalid")
    round_dir = request_path.parent.resolve()
    if (
        Path(request.get("round_dir", "")).resolve() != round_dir
        or Path(request.get("result_path", "")).resolve() != result_path.resolve()
    ):
        raise ValueError("worker request round_dir or result_path differs from CLI")
    return envelope, round_dir


def _timeout(config: Mapping[str, Any]) -> float:
    remaining = float(config["work_deadline_unix"]) - time.time()
    if remaining <= 0:
        raise ValueError("remote worker deadline has elapsed")
    return remaining


def _ssh(
    config: Mapping[str, Any],
    remote: list[str],
    *,
    data: bytes | None = None,
    allow_expired: bool = False,
) -> bytes:
    command = [*config["ssh_argv"], " ".join(shlex.quote(item) for item in remote)]
    timeout = _timeout(config) if not allow_expired else 30.0
    try:
        completed = subprocess.run(
            command, input=data, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise RemoteTransportError("remote worker SSH command timed out") from error
    if completed.returncode == 255:
        raise RemoteTransportError("remote worker SSH command failed (255)")
    if completed.returncode:
        raise RuntimeError(f"remote worker SSH command failed ({completed.returncode})")
    return completed.stdout


def _remote_command(config: Mapping[str, Any], python: str, args: list[str]) -> bytes:
    command = " ".join(shlex.quote(item) for item in args)
    source_root = shlex.quote(config["source_root"])
    seconds = int(_timeout(config)) - 1
    if seconds < 1:
        raise ValueError("remote worker deadline is too close to start a GPU command")
    return _ssh(
        config,
        [
            "sh",
            "-lc",
            f"cd {source_root} && exec timeout --signal=TERM --kill-after=10s {seconds}s "
            f"env PYTHONPATH={source_root} {shlex.quote(python)} {command}",
        ],
    )


def _upload(config: Mapping[str, Any], target: str, value: Mapping[str, Any]) -> None:
    script = (
        "import os,sys; path=sys.argv[1]; data=sys.stdin.buffer.read(); "
        "os.makedirs(os.path.dirname(path),exist_ok=True); "
        "\ntry:\n fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\n"
        "except FileExistsError:\n assert open(path,'rb').read()==data\n"
        "else:\n f=os.fdopen(fd,'wb'); f.write(data); f.flush(); os.fsync(f.fileno()); f.close()\n"
    )
    _ssh(
        config,
        [config["generator_python"], "-c", script, target],
        data=(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode(),
    )


def _service(round_dir: Path) -> LaunchServiceStore:
    database = round_dir.parent / "budget.sqlite"
    with sqlite3.connect(database) as db:
        row = db.execute("SELECT config_json FROM launch_campaign WHERE singleton=1").fetchone()
    if row is None:
        raise ValueError("launch service configuration is absent")
    value = json.loads(row[0])
    if not isinstance(value, dict) or value.pop("schema", None) != "forge/launch-config/v1":
        raise ValueError("stored launch service configuration is invalid")
    return LaunchServiceStore(database, LaunchConfig(**value))


def _authorized(round_dir: Path, envelope: Mapping[str, Any]) -> LaunchServiceStore:
    service = _service(round_dir)
    request = envelope["request"]
    canonical = service.work_request(request["round_id"])
    if canonical != dict(envelope):
        raise ValueError("worker envelope differs from the reserved service request")
    with sqlite3.connect(service.path) as db:
        row = db.execute(
            "SELECT b.status FROM launch_service_rounds r JOIN budget_reservations b "
            "ON b.id=r.reservation_id WHERE r.round_id=? AND r.reservation_id=?",
            (request["round_id"], envelope["reservation_id"]),
        ).fetchone()
    if row is None or row[0] != "RESERVED":
        raise ValueError("worker reservation is no longer active")
    return service


def _fetch(config: Mapping[str, Any], remote_root: str, destination: Path) -> Path:
    script = (
        "import base64,json,sys; p=sys.argv[1]; "
        "b=open(p,'rb').read(); "
        "assert len(b)<=2000000; "
        "print(json.dumps({'receipt-index.json':base64.b64encode(b).decode()}))"
    )
    initial = json.loads(
        _ssh(
            config,
            [config["generator_python"], "-c", script, f"{remote_root}/receipt-index.json"],
            allow_expired=True,
        ).decode()
    )
    encoded = initial.get("receipt-index.json") if isinstance(initial, dict) else None
    if not isinstance(encoded, str):
        raise ValueError("remote receipt index response is invalid")
    index_bytes = base64.b64decode(encoded, validate=True)
    if len(index_bytes) > MAX_RECEIPT_BYTES:
        raise ValueError("remote receipt index is oversized")
    index = json.loads(index_bytes)
    if not isinstance(index, dict):
        raise ValueError("remote receipt index is invalid")
    names = ("panels", "timing", "scores", "diversity", "diversity_panels", "qualification")
    paths = []
    for name in names:
        entries = index.get(name)
        if not isinstance(entries, list):
            raise ValueError("remote receipt index paths are invalid")
        paths.extend(entries)
    if any(
        not isinstance(item, str)
        or PurePosixPath(item).is_absolute()
        or ".." in PurePosixPath(item).parts
        for item in paths
    ):
        raise ValueError("remote receipt index has unsafe paths")
    script = (
        "import base64,json,sys; root=sys.argv[1]; paths=json.loads(sys.argv[2]); out={}; "
        "\nfor p in paths:\n q=root+'/'+p; b=open(q,'rb').read(); assert len(b)<=2000000\n"
        " out[p]=base64.b64encode(b).decode()\n"
        "print(json.dumps(out))"
    )
    payload = json.loads(
        _ssh(
            config,
            [config["generator_python"], "-c", script, remote_root, json.dumps(paths)],
            allow_expired=True,
        ).decode()
    )
    if not isinstance(payload, dict) or set(payload) != set(paths):
        raise ValueError("remote receipt fetch response is incomplete")
    if destination.exists() or destination.is_symlink():
        raise ValueError("custody destination must be a new owned directory")
    destination.mkdir(mode=0o700)
    (destination / "receipt-index.json").write_bytes(index_bytes)
    for relative, encoded in payload.items():
        if not isinstance(encoded, str):
            raise ValueError("remote receipt payload is invalid")
        content = base64.b64decode(encoded, validate=True)
        if len(content) > MAX_RECEIPT_BYTES:
            raise ValueError("remote receipt is oversized")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = index.get("sha256", {}).get(relative)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("remote receipt index hash is invalid")
        target.write_bytes(content)
        if sha256_file(target) != expected:
            raise ValueError("remote receipt bytes differ from the index")
    return destination


def _custody_destination(round_dir: Path) -> Path:
    """Return a new child which `_fetch` alone creates as receipt custody."""
    return Path(tempfile.mkdtemp(prefix=".remote-receipts-", dir=round_dir)) / "receipts"


def _archive_pending(config: Mapping[str, Any], remote_root: str, round_dir: Path) -> None:
    """Back up each completed batch before releasing the next remote stage."""
    script = (
        "import json,sys; from pathlib import Path; r=Path(sys.argv[1])/'custody'; "
        "print(json.dumps([json.loads(p.read_text()) for p in sorted(r.glob('*.request.json')) "
        "if not p.with_name(p.name.replace('.request.json','.verified.json')).exists()]))"
    )
    pending = json.loads(
        _ssh(
            config,
            [config["generator_python"], "-c", script, remote_root],
            allow_expired=True,
        )
    )
    if not isinstance(pending, list) or len(pending) > len(BATCHES):
        raise ValueError("invalid custody request list")
    identity = config["custody"]
    for request in pending:
        body = {key: value for key, value in request.items() if key != "request_hash"}
        files, batch = request.get("files"), request.get("batch")
        if (
            request.get("schema") != CUSTODY_SCHEMA
            or batch not in BATCHES
            or request.get("request_hash") != hash_canonical(CUSTODY_SCHEMA, body)
            or not isinstance(files, dict)
            or not files
            or len(files) > 1000
            or any(
                not isinstance(p, str)
                or PurePosixPath(p).is_absolute()
                or ".." in PurePosixPath(p).parts
                or PurePosixPath(p).suffix not in {".json", ".mp4"}
                or not isinstance(h, str)
                or len(h) != 64
                for p, h in files.items()
            )
        ):
            raise ValueError("invalid frozen custody batch")
        local = round_dir / "media-custody" / request["request_hash"]
        local.mkdir(parents=True, exist_ok=True)
        write_immutable_json(local / "request.json", request)
        prefix = str(PurePosixPath(remote_root).relative_to("/workspace/forge"))
        write_immutable_json(
            local / "ownership.json",
            {
                "pod_id": identity["pod_id"],
                "worklist_id": identity["worklist_id"],
                "ssh_argv": config["ssh_argv"],
            },
        )
        write_immutable_json(
            local / "config.json",
            {
                "remote_archive": f"{prefix}/custody/{batch}.tar.gz",
                "s3_bucket": identity["archive_bucket"],
                "s3_key": f"subnet/experiments/{identity['worklist_id']}/{identity['pod_id']}/"
                f"batches/{request['request_hash']}.tar.gz",
                "allowlist": [f"{prefix}/{relative}" for relative in sorted(files)],
                "max_member_bytes": 100_000_000,
                "max_archive_bytes": 2_000_000_000,
                "timeout_seconds": identity["timeout_seconds"],
            },
        )
        receipt_path = local / "receipt.json"
        if not receipt_path.exists():
            if receipt_path.with_suffix(".intent.json").exists():
                raise ValueError("batch archive intent exists; reconcile before another attempt")
            from forge.evaluation.pilot_guard import _archive

            timeout = min(_timeout(config), identity["timeout_seconds"] + 10)
            _archive(
                [
                    *identity["archive_argv"],
                    "--ownership",
                    str(local / "ownership.json"),
                    "--config",
                    str(local / "config.json"),
                    "--out",
                    str(receipt_path),
                ],
                timeout,
                local / "process.json",
            )
        receipt = _read(receipt_path)
        expected = {f"{prefix}/{path}": digest for path, digest in files.items()}
        if (
            receipt.get("pod_id") != identity["pod_id"]
            or receipt.get("worklist_id") != identity["worklist_id"]
            or receipt.get("status") != "VERIFIED"
            or receipt.get("readback_verified") is not True
            or receipt.get("evidence_complete") is not True
            or {m["path"]: m["sha256"] for m in receipt.get("members", [])} != expected
        ):
            raise ValueError("archive receipt does not preserve the exact batch bytes")
        _upload(
            config,
            f"{remote_root}/custody/{batch}.verified.json",
            {
                "request_hash": request["request_hash"],
                "readback_verified": True,
                "evidence_complete": True,
                "uri": receipt["uri"],
                "sha256": receipt["sha256"],
            },
        )


def _await_job(job: DetachedRemoteJob, *, custody=None) -> dict[str, Any]:
    """Reconcile one remote job with short reads until result or deadline."""
    status = job.poll()
    pending = {
        "LAUNCHING",
        "ALREADY_INTENDED",
        "RUNNING",
        "UNKNOWN_TRANSPORT",
        "UNKNOWN_REMOTE",
    }
    custody_disconnects = 0
    while status["status"] in pending:
        if custody is not None:
            try:
                custody()
            except RemoteTransportError:
                # Only retry transport reconciliation. Verified archive receipts
                # are reused and an uncertain upload intent still fails closed.
                # The original job and its absolute deadline never change.
                custody_disconnects += 1
                atomic_json(job.local_state / "custody-transport-status.json", {
                    "status": "RECONCILING", "checked_unix": time.time(),
                    "transport_failures": custody_disconnects,
                    "deadline_unix": job.deadline_unix,
                })
            else:
                if custody_disconnects:
                    atomic_json(job.local_state / "custody-transport-status.json", {
                        "status": "RECONCILED", "checked_unix": time.time(),
                        "transport_failures": custody_disconnects,
                        "deadline_unix": job.deadline_unix,
                    })
                    custody_disconnects = 0
        remaining = job.deadline_unix - time.time()
        if remaining <= 0:
            raise ValueError("remote render/score job reached its deadline without a result")
        time.sleep(min(5.0, remaining))
        status = job.poll()
    if status["status"] != "SUCCEEDED":
        raise RemoteJobError("remote render/score job did not complete")
    return status


def run(config_path: Path, request_path: Path, result_path: Path) -> dict[str, Any]:
    config, envelope, round_dir = _config(config_path), *_envelope(request_path, result_path)
    if result_path.exists():
        result = _read(result_path)
        if (
            set(result) != {"request_hash", "trusted_bundle_hashes"}
            or result.get("request_hash") != envelope["request_hash"]
            or not isinstance(result.get("trusted_bundle_hashes"), dict)
            or not result["trusted_bundle_hashes"]
        ):
            raise ValueError("existing worker result belongs to another request")
        return result
    request_hash = envelope["request_hash"]
    service = _authorized(round_dir, envelope)
    remote_root = f"{config['remote_root'].rstrip('/')}/launch-{request_hash}"
    intent = round_dir / "remote-worker-intent.json"
    admission = round_dir / "effective-admission.json"
    if intent.exists() and not admission.exists():
        custody = _custody_destination(round_dir)
        try:
            return import_campaign(
                round_dir,
                _fetch(config, remote_root, custody),
                validators=config["validators"],
                request_hash=request_hash,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "preflight dispatch is unknown; no remote GPU work was retried"
            ) from error
    if not intent.exists():
        write_immutable_json(
            intent,
            {
                "schema": "forge/launch-remote-worker-intent/v1",
                "request_hash": request_hash,
                "remote_root": remote_root,
                "status": "PREFLIGHT_UNKNOWN",
            },
        )
        plan, qualification = (
            _read(round_dir / "launch-plan.json"),
            _read(round_dir / "qualification-package.json"),
        )
        contract = _read(Path(config["contract_path"]))
        models = _json_source(config["models"], "models")
        for name, value in (
            ("launch-plan.json", plan),
            ("models.json", models),
            ("contract.json", contract),
            ("qualification-package.json", qualification),
        ):
            _upload(config, f"{remote_root}/input/{name}", value)
        _remote_command(
            config,
            config["generator_python"],
            [
                "-m",
                "forge.evaluation.launch_campaign",
                "prepare",
                "--root",
                remote_root,
                "--plan",
                f"{remote_root}/input/launch-plan.json",
                "--models",
                f"{remote_root}/input/models.json",
                "--contract",
                f"{remote_root}/input/contract.json",
                "--qualification",
                f"{remote_root}/input/qualification-package.json",
            ],
        )
        output = _remote_command(
            config,
            config["generator_python"],
            ["-m", "forge.evaluation.launch_campaign", "preflight", "--root", remote_root],
        )
        receipt = json.loads(output.decode())
        if receipt.get("schema") != "forge/launch-effective-admission/v1":
            raise ValueError("remote preflight receipt schema is invalid")
        write_immutable_json(admission, receipt)
    if not admission.exists():
        raise ValueError("effective admission is absent after preflight")
    service.record_effective_admission(envelope["request"]["round_id"], _read(admission))
    job = DetachedRemoteJob(
        ssh_argv=config["ssh_argv"],
        remote_dir=f"{remote_root}/render-score-job",
        local_state=round_dir,
        job_id=request_hash,
        deadline_unix=float(config["work_deadline_unix"]),
        cwd=config["source_root"],
        env={"PYTHONPATH": config["source_root"]},
        stages=[
            [
                config["generator_python"],
                "-m",
                "forge.evaluation.launch_campaign",
                "render",
                "--root",
                remote_root,
                "--custody-deadline",
                str(config["work_deadline_unix"]),
            ],
            [
                config["reward_python"],
                "-m",
                "forge.evaluation.launch_campaign",
                "score",
                "--root",
                remote_root,
                "--spec",
                config["spec"],
                "--assets",
                config["assets"],
                "--custody-deadline",
                str(config["work_deadline_unix"]),
            ],
        ],
    )
    _await_job(job, custody=lambda: _archive_pending(config, remote_root, round_dir))
    custody = _custody_destination(round_dir)
    return import_campaign(
        round_dir,
        _fetch(config, remote_root, custody),
        validators=config["validators"],
        request_hash=request_hash,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.request, args.result), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
