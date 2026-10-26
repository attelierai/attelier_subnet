"""A completed GPU batch cannot advance until its remote backup is acknowledged."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_service import write_immutable_json

SCHEMA = "forge/custody-batch/v1"
BATCHES = {"timing", "quality", "diversity", "anchor", "scores"}


def preserve_batch(root: Path, batch: str, paths: list[Path], deadline: float) -> None:
    """Freeze only immutable media/receipts, then wait for the owning controller."""
    if batch not in BATCHES or not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("invalid custody batch or expired deadline")
    root = root.resolve(strict=True)
    files = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("custody paths must not include symlinks")
        if not path.is_file() or path.suffix not in {".json", ".mp4"}:
            raise ValueError("custody accepts only completed JSON receipts and videos")
        files[str(relative)] = sha256_file(path)
    if not files:
        raise ValueError("empty custody batch")
    body = {"schema": SCHEMA, "batch": batch, "files": files}
    request = {**body, "request_hash": hash_canonical(SCHEMA, body)}
    directory = root / "custody"
    directory.mkdir(exist_ok=True)
    write_immutable_json(directory / f"{batch}.request.json", request)
    ack = directory / f"{batch}.verified.json"
    while time.time() < deadline:
        if ack.exists():
            receipt = json.loads(ack.read_text())
            if (
                receipt.get("request_hash") != request["request_hash"]
                or receipt.get("readback_verified") is not True
                or receipt.get("evidence_complete") is not True
            ):
                raise ValueError("custody acknowledgement does not verify this exact batch")
            return
        time.sleep(min(1.0, max(0, deadline - time.time())))
    raise TimeoutError("batch backup was not verified before the work deadline")
