"""Provision actual HTTP bytes and an archive through the public CLI."""

import hashlib
import json
import subprocess
import sys
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


def test_metric_assets_download_verify_extract_reuse_and_refuse_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = b"small metric checkpoint fixture"
    with zipfile.ZipFile(source / "models.zip", "w") as archive:
        archive.writestr("models/selected.pth", model)
        archive.writestr("../unselected.py", b"must not extract")
    archive_bytes = (source / "models.zip").read_bytes()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(source))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cache = tmp_path / "cache"
    manifest = tmp_path / "weights.json"
    manifest.write_text(
        json.dumps(
            {
                "models.zip": {
                    "url": f"http://127.0.0.1:{server.server_port}/models.zip",
                    "bytes": len(archive_bytes),
                    "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                },
                "models/selected.pth": {
                    "archive": "models.zip",
                    "member": "models/selected.pth",
                    "bytes": len(model),
                    "sha256": hashlib.sha256(model).hexdigest(),
                },
            }
        )
    )
    command = [
        sys.executable,
        "-m",
        "forge.models.metric_assets",
        "--manifest",
        str(manifest),
        "--cache",
        str(cache),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)["verified_files"] == 2
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert (cache / "models/selected.pth").read_bytes() == model
    assert not (cache / "unselected.py").exists()
    assert not list(cache.rglob("*.partial"))
    # Offline reuse still checks the actual file bytes.
    subprocess.run(command, capture_output=True, text=True, check=True)
    (cache / "models/selected.pth").write_bytes(b"tampered")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "cached metric bytes differ" in result.stderr
    assert (cache / "models/selected.pth").read_bytes() == b"tampered"
