"""Read archived pilot sheets with bounded memory and complete hash verification.

Consumers must exhaust a validation pass before dispatching any paid request.
A second pass must compare each row with its prevalidated digest before use.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import select
import subprocess
import tarfile
import time
from pathlib import Path
from typing import BinaryIO

MAX_ROW_BYTES = 128 * 1024**2
MAX_PAYLOAD_BYTES = 4 * 1024**3
MAX_ARCHIVE_BYTES = 8 * 1024**3


def _s3_bucket(value: object, name: str = "archive_bucket") -> str:
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


class _HashedReader:
    def __init__(self, stream, limit: int, io_seconds: float | None = None):
        self.stream = stream
        self.io_seconds = io_seconds
        self.limit = limit
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        if size < 0:
            size = self.limit - self.size + 1
        count = min(size, self.limit - self.size + 1)
        if count == 0:
            return b""
        if self.io_seconds is None:
            data = self.stream.read(count)
        else:
            started = time.monotonic()
            if (
                self.io_seconds <= 0
                or not select.select([self.stream], [], [], min(60, self.io_seconds))[0]
            ):
                raise TimeoutError("archive provider exceeded its bounded read allowance")
            data = os.read(self.stream.fileno(), count)
            # Judge processing between yields is not network waiting time.
            self.io_seconds -= time.monotonic() - started
        self.size += len(data)
        if self.size > self.limit:
            raise ValueError("archived stream exceeds its declared size")
        self.digest.update(data)
        return data


def archived_payloads(
    backup: dict,
    *,
    payload_sha256: str,
    payload_bytes: int,
    max_rows: int,
    local_archive: Path | None = None,
    archive_stream: BinaryIO | None = None,
    archive_bucket: str | None = None,
):
    """Verify member/archive at EOF; supplied streams are closed when finished."""
    if local_archive is not None and archive_stream is not None:
        raise ValueError("provide only one archive source")
    if (
        backup.get("status") != "VERIFIED"
        or backup.get("readback_verified") is not True
        or type(backup.get("bytes")) is not int
        or not 0 < backup["bytes"] <= MAX_ARCHIVE_BYTES
        or type(payload_bytes) is not int
        or not 0 < payload_bytes <= MAX_PAYLOAD_BYTES
        or type(max_rows) is not int
        or not 0 < max_rows <= 1024
    ):
        raise ValueError("verified archive and bounded payload identity are required")
    matches = [
        row
        for row in backup["members"]
        if row.get("sha256") == payload_sha256 and row.get("bytes") == payload_bytes
    ]
    if len(matches) != 1:
        raise ValueError("payload must identify exactly one archived member")
    member_name = matches[0]["path"]
    process = None
    if archive_stream is not None:
        source = archive_stream
    elif local_archive is not None:
        source = local_archive.open("rb")
    else:
        bucket = _s3_bucket(archive_bucket)
        uri = backup.get("uri", "")
        if (
            backup.get("s3_bucket") != bucket
            or not isinstance(uri, str)
            or not uri.startswith(f"s3://{bucket}/subnet/experiments/")
            or not uri.endswith(".tar.gz")
        ):
            raise ValueError("payload archive is outside the private experiment bucket")
        process = subprocess.Popen(
            ["aws", "s3", "cp", uri, "-", "--only-show-errors"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        source = process.stdout
    archive = _HashedReader(source, backup["bytes"], 900 if process is not None else None)
    seen, rows = False, 0
    try:
        with tarfile.open(fileobj=archive, mode="r|gz") as packed:
            for member in packed:
                if member.name != member_name:
                    continue
                if seen or not member.isfile() or member.size != payload_bytes:
                    raise ValueError("payload archive member differs from its receipt")
                seen = True
                stream = _HashedReader(packed.extractfile(member), payload_bytes)
                with gzip.GzipFile(fileobj=stream) as decoded:
                    while line := decoded.readline(MAX_ROW_BYTES + 1):
                        rows += 1
                        if len(line) > MAX_ROW_BYTES or rows > max_rows:
                            raise ValueError("payload row or comparison count exceeds its bound")
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("payload row must be an object")
                        yield value
                if stream.size != payload_bytes or stream.digest.hexdigest() != payload_sha256:
                    raise ValueError("compressed payload differs from its archived hash")
        while archive.read(1024 * 1024):
            pass
        if (
            not seen
            or archive.size != backup["bytes"]
            or archive.digest.hexdigest() != backup["sha256"]
        ):
            raise ValueError("complete archive differs from its verified receipt")
        if process is not None and process.wait(timeout=30) != 0:
            raise ValueError("archive provider stream failed")
    finally:
        source.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--local-archive", type=Path)
    parser.add_argument("--archive-bucket")
    args = parser.parse_args()
    count = sum(
        1
        for _ in archived_payloads(
            json.loads(args.backup.read_text()),
            payload_sha256=args.payload_sha256,
            payload_bytes=args.payload_bytes,
            max_rows=args.max_rows,
            local_archive=args.local_archive,
            archive_bucket=args.archive_bucket,
        )
    )
    print(json.dumps({"status": "VERIFIED", "rows": count, "api_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
