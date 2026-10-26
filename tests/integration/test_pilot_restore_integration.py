"""Restore a real tar archive and reject changed bytes without overwriting evidence."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


def test_recovery_preserves_generation_identity_and_rejects_corruption(tmp_path):
    path = Path(__file__).parents[2] / "scripts/evaluator_pilot_restore.py"
    spec = importlib.util.spec_from_file_location("pilot_restore", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "evidence.tar.gz"
    files = {
        "pilot-output/renders/video.mp4": b"frozen media bytes",
        "pilot-output/media-index.json": b'{"identity":"frozen"}',
        "generation-freeze.txt": b"original dependencies",
        "pilot-runtime.json": b'{"container":"original"}',
        "pilot-measurements/state.json": b'{"status":"INCOMPLETE"}',
    }
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    receipt = {
        "status": "VERIFIED",
        "readback_verified": True,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "bytes": archive.stat().st_size,
        "pod_id": "original-owned-pod",
        "worklist_id": "original-work",
        "members": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        ],
    }
    destination = tmp_path / "restored"
    result = module.restore(archive, receipt, destination)
    assert result["source_pod_id"] == "original-owned-pod"
    assert (destination / "pilot-output/renders/video.mp4").read_bytes() == files[
        "pilot-output/renders/video.mp4"
    ]
    assert not (destination / "pilot-measurements").exists()
    with pytest.raises(ValueError, match="must be new"):
        module.restore(archive, receipt, destination)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        module.restore(archive, receipt, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()
