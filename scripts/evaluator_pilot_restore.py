#!/usr/bin/env python3
"""Recover frozen pilot media on the remote worker, verifying archived hashes."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def restore(archive: Path, receipt: dict, destination: Path) -> dict:
    if receipt.get("status") != "VERIFIED" or not receipt.get("readback_verified"):
        raise ValueError("a verified original archive receipt is required")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != receipt["archive_sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("archive bytes differ from verified backup")
    members = {row["path"]: row for row in receipt["members"]}
    selected = {name: row for name, row in members.items() if name.startswith("pilot-output/")}
    selected.update(
        {name: members[name] for name in ("generation-freeze.txt", "pilot-runtime.json")}
    )
    if destination.exists():
        raise ValueError("recovery destination must be new")
    destination.mkdir(parents=True)
    restored = set()
    with tarfile.open(archive) as tar:
        for entry in tar:
            if entry.name not in selected:
                continue
            if entry.name in restored or not entry.isfile():
                raise ValueError("duplicate or nonregular recovery member")
            target = destination / entry.name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("recovery path escapes destination")
            expected = selected[entry.name]
            if entry.size != expected["bytes"]:
                raise ValueError("recovery member size differs")
            target.parent.mkdir(parents=True, exist_ok=True)
            hasher = hashlib.sha256()
            with tar.extractfile(entry) as source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    hasher.update(block)
                    output.write(block)
            if hasher.hexdigest() != expected["sha256"]:
                raise ValueError("recovery member hash differs")
            restored.add(entry.name)
    if restored != set(selected):
        raise ValueError("recovery members are missing")
    result = {
        "schema": "forge/evaluator-pilot-recovery/v1",
        "status": "VERIFIED",
        "source_pod_id": receipt["pod_id"],
        "archive_sha256": digest,
        "worklist_id": receipt["worklist_id"],
        "restored_members": len(restored),
    }
    (destination / "recovery-origin.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/workspace/forge")
    if Path.cwd().resolve() != root:
        raise ValueError("recovery runs only on the owned remote worker")
    print(
        json.dumps(restore(args.archive, json.loads(args.receipt.read_text()), root / "recovered"))
    )


if __name__ == "__main__":
    main()
