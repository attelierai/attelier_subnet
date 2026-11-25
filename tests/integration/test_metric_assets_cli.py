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

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/download":
                self.send_response(302)
                self.send_header(
                    "Location", f"http://localhost:{self.server.server_port}/models.zip"
                )
                self.end_headers()
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(source)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cache = tmp_path / "cache"
    manifest = tmp_path / "weights.json"
    manifest.write_text(
        json.dumps(
            {
                "models.zip": {
                    "url": f"http://127.0.0.1:{server.server_port}/download",
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
        oversized = json.loads(manifest.read_text())["models.zip"]
        limited = tmp_path / "limited.json"
        limited_cache = tmp_path / "limited-cache"
        limited.write_text(json.dumps({"small.zip": {**oversized, "bytes": 1}}))
        refused = subprocess.run(
            [*command[:3], "--manifest", str(limited), "--cache", str(limited_cache)],
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert not (limited_cache / "small.zip").exists()
        assert (limited_cache / "small.zip.partial").stat().st_size <= 1
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


def test_metric_assets_reject_non_http_source(tmp_path):
    manifest = tmp_path / "weights.json"
    manifest.write_text(
        json.dumps(
            {
                "weights.bin": {
                    "url": "file:///tmp/weights.bin",
                    "bytes": 1,
                    "sha256": "0" * 64,
                }
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.models.metric_assets",
            "--manifest",
            str(manifest),
            "--cache",
            str(tmp_path / "cache"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "metric asset URL must be an HTTP(S) URL" in result.stderr
