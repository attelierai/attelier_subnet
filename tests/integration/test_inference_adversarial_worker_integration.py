"""End-to-end protocol exercise for the private adversarial H3 worker."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/inference_adversarial_worker.py"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Upstream(BaseHTTPRequestHandler):
    worker_id = "123e4567-e89b-12d3-a456-426614174000"
    jobs = {}
    submissions = []

    def log_message(self, *_):
        pass

    def reply(self, status, value):
        data = canonical(value)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("x-worker-id", self.worker_id)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        if self.path == "/challenge-quotes":
            self.reply(
                200,
                {
                    "token": str(uuid.uuid4()),
                    "expiresAt": int(time.time() * 1000) + 60_000,
                    "amountMicros": 1,
                },
            )
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            self.submissions.append(data)
            self.jobs[job_id] = data
            self.reply(202, {"id": job_id})
            return
        self.reply(404, {"error": {"code": "not_found"}})

    def do_GET(self):
        if self.path.startswith("/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            self.reply(
                200,
                {
                    "status": "succeeded",
                    "progress": 1,
                    "verificationReceipt": {
                        "schema": "forge-h3-reference-v1",
                        "jobId": job_id,
                        "workerId": self.worker_id,
                        "nonce": "0" * 64,
                        "parametersHash": "1" * 64,
                        "modelManifestHash": "2" * 64,
                        "workflowHash": "3" * 64,
                        "resultSha256": "a" * 64,
                        "signature": "x",
                    },
                },
            )
            return
        self.reply(404, {"error": {"code": "not_found"}})


def request(port, method, path, token, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    payload = canonical(body) if body is not None else None
    connection.request(
        method,
        path,
        body=payload,
        headers={
            "authorization": f"Bearer {token}",
            **({"content-type": "application/json"} if payload else {}),
        },
    )
    response = connection.getresponse()
    data = response.read()
    return response.status, dict(response.getheaders()), data


def video(path):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=832x480:r=24:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        timeout=60,
    )


def test_adversarial_modes_are_private_and_keep_the_worker_protocol():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "wrong-seed.mp4"
        video(source)
        key = root / "ed25519.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(key)], check=True
        )
        os.chmod(key, 0o600)
        upstream_port, proxy_port = free_port(), free_port()
        upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        honest, fast, replay, wrong = (str(uuid.uuid4()) for _ in range(4))
        config = root / "private.json"
        token, upstream_token = "a" * 32, "b" * 32
        config.write_text(
            json.dumps(
                {
                    "schema": "forge/inference-adversarial-worker/v1",
                    "listen": {"host": "127.0.0.1", "port": proxy_port},
                    "upstream": {
                        "url": f"http://127.0.0.1:{upstream_port}",
                        "tokenEnv": "TEST_UPSTREAM_TOKEN",
                        "workerId": Upstream.worker_id,
                    },
                    "tokenEnv": "TEST_PROXY_TOKEN",
                    "dataDirectory": str(root / "data"),
                    "signingKeyFile": str(key),
                    "modelManifestHash": "2" * 64,
                    "workflowHash": "3" * 64,
                    "schedule": {
                        fast: {"mode": "fast_profile"},
                        replay: {"mode": "replay", "source": str(source)},
                        wrong: {"mode": "wrong_video", "source": str(source)},
                    },
                }
            )
        )
        os.chmod(config, 0o600)
        env = {**os.environ, "TEST_PROXY_TOKEN": token, "TEST_UPSTREAM_TOKEN": upstream_token}
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--config", str(config)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(50):
                try:
                    if request(proxy_port, "GET", "/health", token)[0] in (200, 404):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError(process.stderr.read().decode())
            parameters = {
                "modality": "text-to-video",
                "duration": 5,
                "resolution": "480p",
                "aspectRatio": "16:9",
                "audio": True,
                "profile": "base",
                "prompt": "secret",
                "seed": 9,
            }
            verification = {
                "schema": "forge-h3-reference-v1",
                "nonce": "f" * 64,
                "parametersHash": hashlib.sha256(canonical(parameters)).hexdigest(),
            }
            for job_id in (honest, fast, replay, wrong):
                status, _, data = request(
                    proxy_port,
                    "POST",
                    "/challenge-quotes",
                    token,
                    {
                        key: parameters[key]
                        for key in (
                            "modality",
                            "duration",
                            "resolution",
                            "aspectRatio",
                            "audio",
                            "profile",
                        )
                    },
                )
                assert status == 200
                status, headers, _ = request(
                    proxy_port,
                    "POST",
                    f"/jobs/{job_id}",
                    token,
                    {
                        "parameters": parameters,
                        "quote": json.loads(data),
                        "verification": verification,
                    },
                )
                assert status == 202 and headers["x-worker-id"] == Upstream.worker_id
            assert Upstream.submissions[1]["parameters"]["profile"] == "taomate-3step"
            assert (
                Upstream.submissions[1]["verification"]["parametersHash"]
                != verification["parametersHash"]
            )
            status, _, data = request(proxy_port, "GET", f"/jobs/{honest}", token)
            honest_receipt = json.loads(data)["verificationReceipt"]
            assert (
                status == 200
                and honest_receipt["parametersHash"] == verification["parametersHash"]
                and honest_receipt["signature"] != "x"
            )
            status, _, data = request(proxy_port, "GET", f"/jobs/{fast}", token)
            fast_receipt = json.loads(data)["verificationReceipt"]
            assert (
                status == 200
                and fast_receipt["parametersHash"] == verification["parametersHash"]
                and fast_receipt["signature"] != "x"
            )
            status, _, data = request(proxy_port, "GET", f"/jobs/{replay}", token)
            receipt = json.loads(data)["verificationReceipt"]
            assert (
                status == 200
                and receipt["parametersHash"] == verification["parametersHash"]
                and receipt["signature"] != "x"
            )
            status, headers, data = request(proxy_port, "GET", f"/jobs/{replay}/result", token)
            assert status == 200 and headers["x-content-sha256"] == hashlib.sha256(data).hexdigest()
            status, _, data = request(proxy_port, "GET", f"/jobs/{wrong}", token)
            wrong_receipt = json.loads(data)["verificationReceipt"]
            assert (
                status == 200
                and wrong_receipt["resultSha256"] == "a" * 64
                and wrong_receipt["signature"] != "x"
            )
            # The schedule is never reflected in any HTTP payload or headers.
            assert (
                b"replay" not in data and b"wrong_video" not in data and b"fast_profile" not in data
            )
        finally:
            process.terminate()
            process.wait(timeout=10)
            upstream.shutdown()
            thread.join(timeout=10)
