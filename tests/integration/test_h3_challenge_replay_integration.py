from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def test_challenge_replay_batches_and_rerun_is_post_free(tmp_path):
    posts = []
    histories = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            data = json.loads(body)
            prompt_id = str(len(posts))
            posts.append(data)
            requests = json.loads(data["prompt"]["2"]["inputs"]["requests_json"])
            for item in requests:
                result = Path(item["result_path"])
                result.parent.mkdir(parents=True, exist_ok=True)
                bundle = hashlib.sha256(Path(item["bundle_path"]).read_bytes()).hexdigest()
                result.write_text(
                    json.dumps(
                        {
                            "bundleSha256": bundle,
                            "candidateLabel": item["candidate_label"],
                            "outcome": {
                                "status": "MEASURED",
                                "maxAbs": 0.0,
                                "rms": 0.0,
                                "relativeRms": 0.0,
                                "expectedSha256": "a" * 64,
                                "actualSha256": "a" * 64,
                            },
                        }
                    )
                )
            histories[prompt_id] = {"status": {"status_str": "success"}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"prompt_id": prompt_id}).encode())

        def do_GET(self):
            prompt_id = self.path.rsplit("/", 1)[-1]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({prompt_id: histories.get(prompt_id)}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    receipts = []
    for i in range(65):
        directory = tmp_path / f"r{i}"
        directory.mkdir()
        bundle = directory / "candidate.pt"
        bundle.write_bytes(f"x{i}".encode())
        receipt = directory / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "selection": "postcommit_system_random",
                    "bundles": [
                        {
                            "step": i,
                            "path": bundle.name,
                            "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        )
        receipts.append(receipt)
    origin = f"http://127.0.0.1:{server.server_port}"
    command = [
        sys.executable,
        "scripts/verification/h3_challenge_replay.py",
        "--receipts",
        *map(str, receipts),
        "--root",
        str(tmp_path / "root"),
        "--deadline-unix",
        str(time.time() + 10_000),
        "--origin",
        origin,
    ]
    try:
        first = subprocess.run(command, capture_output=True, text=True)
        assert first.returncode == 0
        report = json.loads(first.stdout)
        assert report["status"] == "MEASURED"
        assert report["calls"] == 65
        assert [batch["calls"] for batch in report["batches"]] == [64, 1]
        assert all(batch["maxRelativeRms"] == 0.0 for batch in report["batches"])
        request_counts = [
            len(json.loads(post["prompt"]["2"]["inputs"]["requests_json"])) for post in posts
        ]
        assert request_counts == [64, 1]
        rerun = subprocess.run(command, capture_output=True, text=True)
        assert rerun.returncode == 0
        assert json.loads(rerun.stdout)["rerun"] == "verified_empty"
        assert len(posts) == 2

        malformed = tmp_path / "malformed.json"
        malformed.write_text('{"selection":"postcommit_system_random"}')
        bad_command = [
            sys.executable,
            "scripts/verification/h3_challenge_replay.py",
            "--receipts",
            str(malformed),
            "--root",
            str(tmp_path / "other-root"),
            "--deadline-unix",
            str(time.time() + 10_000),
            "--origin",
            origin,
        ]
        assert subprocess.run(bad_command, capture_output=True, text=True).returncode != 0
        assert len(posts) == 2
    finally:
        server.shutdown()
        server.server_close()
