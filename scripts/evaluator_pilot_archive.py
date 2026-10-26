#!/usr/bin/env python3
"""Archive an owned evaluator-pilot evidence allowlist without local media copies.

This command is intentionally dormant until a caller supplies a reviewed
ownership record and archive configuration.  It streams a remote tarball to
private S3 and streams it back for a complete SHA-256 and size comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

REMOTE_ROOT = "/workspace/forge"
SCHEMA = "forge/evaluator-pilot-archive/v1"
FORBIDDEN_SUFFIXES = (".env", ".pem", ".key", ".safetensors", ".ckpt", ".pt", ".pth", ".bin")
FORBIDDEN_PARTS = {".aws", ".ssh", "credentials", "secrets", "weights", "models"}


def _s3_bucket(value: object, name: str = "s3_bucket") -> str:
    if (
        not isinstance(value, str)
        or not 3 <= len(value) <= 63
        or not value[0].isalnum()
        or not value[-1].isalnum()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in value)
        or ".." in value
        or value.replace(".", "").isdigit()
    ):
        raise ValueError(f"{name} must be a plain S3 bucket name")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _argv(value: object, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty string list")
    return list(value)


def _relative_evidence_path(value: object) -> str:
    path = PurePosixPath(_text(value, "allowlist path"))
    # The runtime's Python package is named models; it is not the downloaded
    # model directory. Admit only direct Python sources from that exact package.
    model_source = (
        path.parent
        in {
            PurePosixPath("subnet/src/forge/models"),
            PurePosixPath("qualified-src/forge/models"),
        }
        and path.suffix == ".py"
    )
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or any(
            part in FORBIDDEN_PARTS and not (part == "models" and model_source)
            for part in path.parts
        )
        or any(part.startswith(".env") for part in path.parts)
        or path.name.endswith(FORBIDDEN_SUFFIXES)
    ):
        raise ValueError("allowlist contains a forbidden evidence path")
    return str(path)


def _config(ownership: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    allowed_ownership = {"pod_id", "worklist_id", "ssh_argv"}
    if set(ownership) != allowed_ownership:
        raise ValueError("ownership fields must be exact")
    allowed_config = {
        "remote_archive",
        "s3_bucket",
        "s3_key",
        "allowlist",
        "max_member_bytes",
        "max_archive_bytes",
        "timeout_seconds",
    }
    if set(config) != allowed_config:
        raise ValueError("archive config fields must be exact")
    allowlist = config["allowlist"]
    if not isinstance(allowlist, list) or not allowlist:
        raise ValueError("allowlist must be a nonempty list")
    paths = [_relative_evidence_path(item) for item in allowlist]
    if len(paths) != len(set(paths)):
        raise ValueError("allowlist paths must be unique")
    # A metric receipt hashes its native result. Preserve that result and its
    # input mapping even if the worker failed before producing the receipt.
    # Restrict expansion to the pilot's exact metric directory shape.
    for relative in tuple(paths):
        parts = PurePosixPath(relative).parts
        if (
            len(parts) == 4
            and parts[:2] == ("pilot-measurements", "metrics")
            and len(parts[2]) == 64
            and all(c in "0123456789abcdef" for c in parts[2])
            and parts[3] == "receipt.json"
        ):
            for filename in ("metrics_eval_results.json", "metrics_full_info.json"):
                raw = str(PurePosixPath(relative).with_name(filename))
                if raw not in paths:
                    paths.append(raw)
    archive = _relative_evidence_path(config["remote_archive"])
    if not archive.endswith(".tar.gz"):
        raise ValueError("remote_archive must end in .tar.gz")
    key = _relative_evidence_path(config["s3_key"])
    if not key.startswith("subnet/experiments/") or not key.endswith(".tar.gz"):
        raise ValueError("s3_key must be a private experiment .tar.gz destination")
    if archive in paths:
        raise ValueError("archive cannot be an evidence member")
    max_member = config["max_member_bytes"]
    max_archive = config["max_archive_bytes"]
    timeout = config["timeout_seconds"]
    if (
        type(max_member) is not int
        or max_member <= 0
        or type(max_archive) is not int
        or max_archive <= 0
        or type(timeout) is not int
        or not 1 <= timeout <= 900
    ):
        raise ValueError("max_member_bytes and timeout_seconds must be bounded positive integers")
    return {
        "pod_id": _text(ownership["pod_id"], "pod_id"),
        "worklist_id": _text(ownership["worklist_id"], "worklist_id"),
        "ssh_argv": _argv(ownership["ssh_argv"], "ssh_argv"),
        "remote_root": REMOTE_ROOT,
        "remote_archive": archive,
        "s3_bucket": _s3_bucket(config["s3_bucket"]),
        "s3_key": key,
        "allowlist": paths,
        "max_member_bytes": max_member,
        "max_archive_bytes": max_archive,
        "timeout_seconds": timeout,
    }


def _remote_archive_program(request: dict[str, Any]) -> str:
    return """import hashlib, json, tarfile, sys
from pathlib import Path
request=json.loads(sys.stdin.read())
root=Path(request['remote_root'])
archive=root/request['remote_archive']
if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
    raise RuntimeError('remote root is unsafe')
root_real=root.resolve(strict=True)
def sha256_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()
def checked_path(relative, require_exists=True):
    current=root
    for part in Path(relative).parts:
        current=current/part
        if current.is_symlink(): raise RuntimeError('remote path has a symlink ancestor')
        if not current.exists():
            if require_exists: raise RuntimeError('remote path is missing')
            return current, False
    if not current.resolve(strict=True).is_relative_to(root_real):
        raise RuntimeError('remote path escapes remote root')
    return current, True
archive_parent,_=checked_path(Path(request['remote_archive']).parent)
if archive.exists() or archive.is_symlink(): raise RuntimeError('remote archive already exists')
members=[]
missing=[]
for relative in request['allowlist']:
    path,exists=checked_path(relative,require_exists=False)
    if not exists:
        missing.append(relative)
        continue
    if not path.is_file() or path.is_symlink():
        raise RuntimeError('allowlist member is missing or unsafe')
    if path.stat().st_size > request['max_member_bytes']:
        raise RuntimeError('allowlist member exceeds size cap')
    digest=sha256_file(path)
    members.append((relative,path,path.stat().st_size,digest))
if not members: raise RuntimeError('allowlist has no existing evidence members')
try:
    with tarfile.open(archive,'x:gz') as output:
        for relative,path,_,_ in members: output.add(path,arcname=relative,recursive=False)
    checked=[]
    for relative,path,size,digest in members:
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError('allowlist member changed while archiving')
        checked.append({'path':relative,'bytes':size,'sha256':digest})
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''): digest.update(block)
    if archive.stat().st_size > request['max_archive_bytes']:
        raise RuntimeError('archive exceeds size cap')
    print(json.dumps({'bytes':archive.stat().st_size,'sha256':digest.hexdigest(),'members':checked,'missing_members':missing}))
except BaseException:
    # Preserve partial evidence. Never unlink a competing file after x-mode fails.
    raise
"""


def _run(
    argv: list[str], *, timeout: int, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, input=input_text, text=True, capture_output=True, timeout=timeout, check=False
    )


def _require_absent(bucket: str, key: str, timeout: int) -> None:
    result = _run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key], timeout=timeout
    )
    if result.returncode == 0:
        raise RuntimeError("S3 destination already exists")
    if "(404)" not in result.stderr or "HeadObject" not in result.stderr:
        raise subprocess.CalledProcessError(result.returncode, "S3 destination lookup")


def _stream_readback(uri: str, *, timeout: int, max_bytes: int) -> tuple[int, str]:
    process = subprocess.Popen(
        ["aws", "s3", "cp", uri, "-", "--only-show-errors"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError("cannot open S3 readback stream")
    digest, size = hashlib.sha256(), 0
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("S3 readback exceeded its timeout")
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if not readable:
                raise TimeoutError("S3 readback exceeded its timeout")
            chunk = os.read(process.stdout.fileno(), 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise RuntimeError("S3 readback exceeds the exact archive size")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("S3 readback deadline exhausted")
        returncode = process.wait(timeout=remaining)
        if returncode:
            raise subprocess.CalledProcessError(returncode, "S3 readback")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    return size, digest.hexdigest()


def _members(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("remote archive member receipt is invalid")
    checked: list[dict[str, Any]] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or _relative_evidence_path(item["path"]) != item["path"]
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise RuntimeError("remote archive member receipt is invalid")
        checked.append(dict(item))
    return checked


def _missing_members(value: object) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError("remote archive missing-member receipt is invalid")
    result = [_relative_evidence_path(item) for item in value]
    if len(result) != len(set(result)):
        raise RuntimeError("remote archive missing-member receipt is invalid")
    return result


def _archive_once(ownership_path: Path, config_path: Path, out: Path, progress) -> dict[str, Any]:
    """Archive once; local receipt is created only after matching full readback."""
    if out.exists():
        raise FileExistsError("archive receipt already exists; do not overwrite evidence")
    request = _config(_read_json(ownership_path), _read_json(config_path))
    configured_key = request["s3_key"]
    request["s3_key"] = f"{configured_key[:-7]}-{uuid.uuid4().hex}.tar.gz"
    intent = out.with_suffix(".intent.json")
    if intent.exists():
        raise FileExistsError("archive intent already exists; reconcile the prior transfer attempt")
    intent.parent.mkdir(parents=True, exist_ok=True)
    with intent.open("x") as stream:
        json.dump(
            {
                "schema": "forge/evaluator-pilot-archive-intent/v1",
                "pod_id": request["pod_id"],
                "worklist_id": request["worklist_id"],
                "configured_s3_key": configured_key,
                "s3_bucket": request["s3_bucket"],
                "s3_key": request["s3_key"],
                "created_unix": time.time(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    deadline = time.monotonic() + request["timeout_seconds"]

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("archive deadline exhausted")
        return value

    progress("REMOTE_ARCHIVE")
    remote = _run(
        [*request["ssh_argv"], shlex.join(["python3", "-c", _remote_archive_program(request)])],
        timeout=remaining(),
        input_text=json.dumps(request, sort_keys=True),
    )
    if remote.returncode:
        raise subprocess.CalledProcessError(remote.returncode, "remote archive")
    try:
        metadata = json.loads(remote.stdout)
        size, sha256 = metadata["bytes"], metadata["sha256"]
        members = _members(metadata["members"])
        missing_members = _missing_members(metadata["missing_members"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("remote archive receipt is invalid") from error
    if (
        type(size) is not int
        or size <= 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError("remote archive size or SHA-256 is invalid")
    if size > request["max_archive_bytes"]:
        raise RuntimeError("remote archive exceeds the configured size cap")
    declared = [member["path"] for member in members] + missing_members
    if len(members) == 0 or sorted(declared) != sorted(request["allowlist"]):
        raise RuntimeError("remote archive member receipt differs from the allowlist")
    _private_json(out.with_suffix(".manifest.json"), metadata)
    uri = f"s3://{request['s3_bucket']}/{request['s3_key']}"
    progress("DESTINATION_CHECK")
    _require_absent(request["s3_bucket"], request["s3_key"], remaining())
    progress("UPLOAD")
    source = subprocess.Popen(
        [
            *request["ssh_argv"],
            shlex.join(["cat", f"{request['remote_root']}/{request['remote_archive']}"]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if source.stdout is None:
        raise RuntimeError("cannot open remote archive stream")
    try:
        upload = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                "-",
                uri,
                "--sse",
                "AES256",
                "--only-show-errors",
            ],
            stdin=source.stdout,
            capture_output=True,
            timeout=remaining(),
            check=False,
        )
        source.stdout.close()
        if upload.returncode:
            raise subprocess.CalledProcessError(upload.returncode, "S3 upload")
        source_code = source.wait(timeout=remaining())
        if source_code:
            raise subprocess.CalledProcessError(source_code, "remote archive stream")
    finally:
        if source.poll() is None:
            source.terminate()
            source.wait(timeout=5)
    progress("READBACK")
    readback_size, readback_sha256 = _stream_readback(uri, timeout=remaining(), max_bytes=size)
    if readback_size != size or readback_sha256 != sha256:
        raise RuntimeError("S3 full readback does not match remote archive")
    receipt = {
        "schema": SCHEMA,
        "status": "VERIFIED",
        "pod_id": request["pod_id"],
        "worklist_id": request["worklist_id"],
        "uri": uri,
        "s3_bucket": request["s3_bucket"],
        "configured_s3_key": configured_key,
        "bytes": size,
        "sha256": sha256,
        "archive_sha256": sha256,
        "members": members,
        "missing_members": missing_members,
        "evidence_complete": not missing_members,
        "readback_verified": True,
        "verified_unix": time.time(),
    }
    progress("RECEIPT")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def _private_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def archive(ownership_path: Path, config_path: Path, out: Path) -> dict[str, Any]:
    """Persist phase and safe failure diagnostics independently of the final receipt."""
    started = time.monotonic()
    state = {}

    def progress(phase: str) -> None:
        state.update(
            schema="forge/evaluator-pilot-archive-status/v1",
            phase=phase,
            status="RUNNING",
            checked_unix=time.time(),
            elapsed_seconds=time.monotonic() - started,
        )
        _private_json(out.with_suffix(".status.json"), state)

    try:
        receipt = _archive_once(ownership_path, config_path, out, progress)
    except Exception as error:
        if state:
            state.update(
                status="FAILED",
                error_type=type(error).__name__,
                timed_out=isinstance(error, (TimeoutError, subprocess.TimeoutExpired)),
                exit_code=error.returncode
                if isinstance(error, subprocess.CalledProcessError)
                else None,
                checked_unix=time.time(),
                elapsed_seconds=time.monotonic() - started,
            )
            _private_json(out.with_suffix(".status.json"), state)
        raise
    state.update(
        status="VERIFIED", checked_unix=time.time(), elapsed_seconds=time.monotonic() - started
    )
    _private_json(out.with_suffix(".status.json"), state)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = archive(args.ownership, args.config, args.out)
    except Exception as error:
        # Never print exception messages/argv or provider stderr; they may contain secrets.
        status = args.out.with_suffix(".status.json")
        diagnostic = _read_json(status) if status.exists() else {"error_type": type(error).__name__}
        if isinstance(error, FileExistsError):
            diagnostic = {
                "error": "archive intent already exists or receipt exists; reconcile first"
            }
        print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
