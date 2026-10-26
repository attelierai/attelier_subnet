import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("operation", "repo_id", "filename", "limit"),
    (
        ("inspect", "oversize-license", "LICENSE", 64 * 1024),
        ("declaration", "oversize-declaration", "forge-manifest.json", 1024 * 1024),
    ),
)
def test_bridge_rejects_oversize_metadata_before_starting_a_snapshot_download(
    tmp_path, operation, repo_id, filename, limit
):
    fake_hub = tmp_path / "fake-hub"
    fake_hub.mkdir()
    (fake_hub / "hippius_hub.py").write_text(
        """\
import os
from pathlib import Path

__version__ = "0.6.0"

class Sibling:
    def __init__(self, filename, size):
        self.rfilename = filename
        self.size = size

class Info:
    private = False
    gated = False
    cardData = None
    def __init__(self, filename, size):
        self.siblings = [Sibling(filename, size)]

def model_info(repo_id, **_kwargs):
    return Info(os.environ["BRIDGE_TEST_FILE"], int(os.environ["BRIDGE_TEST_SIZE"]))

def snapshot_download(*_args, **_kwargs):
    Path(os.environ["BRIDGE_CALL_MARKER"]).write_text("called")
    raise AssertionError("snapshot download must not run")
"""
    )
    marker = tmp_path / "snapshot-called"
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(fake_hub), str(root / "src"))),
        "HOME": str(tmp_path / "anonymous-home"),
        "BRIDGE_TEST_FILE": filename,
        "BRIDGE_TEST_SIZE": str(limit + 1),
        "BRIDGE_CALL_MARKER": str(marker),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "forge.models.hippius_bridge"],
        input=json.dumps(
            {"operation": operation, "repo_id": repo_id, "revision": "sha256:" + "a" * 64}
        ),
        text=True,
        capture_output=True,
        cwd=root,
        env=environment,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {"ok": False, "error": f"{filename} exceeds size limit"}
    assert not marker.exists()


@pytest.mark.parametrize(
    ("reported_size", "reported_digest", "copies", "error"),
    (
        (9, "a" * 64, 1, "registry file size differs from pinned manifest"),
        (4, "b" * 64, 1, "registry blob digest differs from pinned manifest"),
        (4, "a" * 64, 2, "registry did not return exactly one requested file"),
    ),
)
def test_bridge_fetch_binds_oci_sibling_metadata_before_download(
    tmp_path, reported_size, reported_digest, copies, error
):
    fake_hub = tmp_path / "fake-hub"
    fake_hub.mkdir()
    (fake_hub / "hippius_hub.py").write_text(
        """\
import os
from pathlib import Path

__version__ = "0.6.0"

class Sibling:
    def __init__(self, filename, size, blob_id):
        self.rfilename = filename
        self.size = size
        self.blob_id = blob_id

class Info:
    private = False
    gated = False
    def __init__(self):
        sibling = Sibling(
            os.environ["BRIDGE_TEST_FILE"],
            int(os.environ["BRIDGE_TEST_SIZE"]),
            "sha256:" + os.environ["BRIDGE_TEST_DIGEST"],
        )
        self.siblings = [sibling] * int(os.environ["BRIDGE_TEST_COPIES"])

def model_info(_repo_id, **kwargs):
    if kwargs.get("files_metadata") is not True:
        raise AssertionError("fetch metadata was not requested")
    Path(os.environ["BRIDGE_METADATA_MARKER"]).write_text("checked")
    return Info()

def hf_hub_download(*_args, **_kwargs):
    Path(os.environ["BRIDGE_DOWNLOAD_MARKER"]).write_text("called")
    raise AssertionError("download must not run")
"""
    )
    metadata_marker = tmp_path / "metadata-called"
    download_marker = tmp_path / "download-called"
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(fake_hub), str(root / "src"))),
        "HOME": str(tmp_path / "anonymous-home"),
        "BRIDGE_TEST_FILE": "adapter/model.safetensors",
        "BRIDGE_TEST_SIZE": str(reported_size),
        "BRIDGE_TEST_DIGEST": reported_digest,
        "BRIDGE_TEST_COPIES": str(copies),
        "BRIDGE_METADATA_MARKER": str(metadata_marker),
        "BRIDGE_DOWNLOAD_MARKER": str(download_marker),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "forge.models.hippius_bridge"],
        input=json.dumps(
            {
                "operation": "fetch",
                "repo_id": "public/model",
                "revision": "sha256:" + "c" * 64,
                "path": "adapter/model.safetensors",
                "destination": str(tmp_path / "destination"),
                "bytes": 4,
                "sha256": "a" * 64,
            }
        ),
        text=True,
        capture_output=True,
        cwd=root,
        env=environment,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {"ok": False, "error": error}
    assert metadata_marker.read_text() == "checked"
    assert not download_marker.exists()


def test_bridge_keeps_sdk_cold_download_progress_off_json_stdout(tmp_path):
    fake_hub = tmp_path / "fake-hub"
    fake_hub.mkdir()
    payload = b"safe public bytes"
    digest = hashlib.sha256(payload).hexdigest()
    (fake_hub / "hippius_hub.py").write_text(
        """\
import os
from pathlib import Path

__version__ = "0.6.0"

class Sibling:
    rfilename = "adapter/model.safetensors"
    size = int(os.environ["BRIDGE_PAYLOAD_BYTES"])
    blob_id = "sha256:" + os.environ["BRIDGE_PAYLOAD_SHA256"]

class Info:
    siblings = [Sibling()]

def model_info(_repo_id, **kwargs):
    if kwargs.get("files_metadata") is not True:
        raise AssertionError("fetch metadata was not requested")
    return Info()

def hf_hub_download(_repo_id, filename, *, local_dir, **_kwargs):
    print("Downloading config.json (parallel)...")
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes.fromhex(os.environ["BRIDGE_PAYLOAD_HEX"]))
    return str(target)
"""
    )
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(fake_hub), str(root / "src"))),
        "HOME": str(tmp_path / "anonymous-home"),
        "BRIDGE_PAYLOAD_BYTES": str(len(payload)),
        "BRIDGE_PAYLOAD_SHA256": digest,
        "BRIDGE_PAYLOAD_HEX": payload.hex(),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "forge.models.hippius_bridge"],
        input=json.dumps(
            {
                "operation": "fetch",
                "repo_id": "public/model",
                "revision": "sha256:" + "c" * 64,
                "path": "adapter/model.safetensors",
                "destination": str(tmp_path / "destination"),
                "bytes": len(payload),
                "sha256": digest,
            }
        ),
        text=True,
        capture_output=True,
        cwd=root,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["ok"] is True
    assert "Downloading config.json (parallel)..." not in completed.stdout
    assert "Downloading config.json (parallel)..." in completed.stderr
