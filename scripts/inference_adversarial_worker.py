#!/usr/bin/env python3
"""Private, loopback-only adversarial H3 worker for testnet verifier tests.

The operator-only 0600 schedule selects an attack by challenge UUID.  It never
reports that selection over HTTP.  This is not a production relay.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCHEMA = "forge-h3-reference-v1"
CONFIG = "forge/inference-adversarial-worker/v1"
JOB = re.compile(r"^/jobs/([a-f0-9-]{36})(?:/(cancel|result|original))?$")
HASH = re.compile(r"^[a-f0-9]{64}$")


def canon(v: Any) -> bytes:
    return json.dumps(
        v, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode()


def digest(v: bytes) -> str:
    return hashlib.sha256(v).hexdigest()


def require0600(p: Path) -> None:
    if not p.is_file() or p.stat().st_mode & 0o077:
        raise ValueError(f"private 0600 file required: {p}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_):
        return None


def config(path: Path) -> dict[str, Any]:
    require0600(path)
    v = json.loads(path.read_text())
    required = {
        "schema",
        "listen",
        "upstream",
        "tokenEnv",
        "dataDirectory",
        "signingKeyFile",
        "modelManifestHash",
        "workflowHash",
        "schedule",
    }
    if (
        set(v) != required
        or v["schema"] != CONFIG
        or not all(HASH.fullmatch(str(v.get(k, ""))) for k in ("modelManifestHash", "workflowHash"))
    ):
        raise ValueError("invalid adversarial config")
    if set(v["upstream"]) != {"url", "tokenEnv", "workerId"} or not isinstance(v["schedule"], dict):
        raise ValueError("invalid upstream/schedule")
    require0600(Path(v["signingKeyFile"]))
    for job, entry in v["schedule"].items():
        if (
            not re.fullmatch(r"[a-f0-9-]{36}", job)
            or not isinstance(entry, dict)
            or entry.get("mode") not in {"fast_profile", "replay", "wrong_video", "wrong_nonce"}
            or set(entry) - {"mode", "source"}
        ):
            raise ValueError("invalid schedule")
        if entry["mode"] in {"replay", "wrong_video"}:
            p = Path(str(entry.get("source", "")))
            if not p.is_absolute() or not p.is_file():
                raise ValueError("scheduled source missing")
    if not Path(v["dataDirectory"]).is_absolute():
        raise ValueError("dataDirectory must be absolute")
    return v


class Worker:
    def __init__(self, c: dict[str, Any], config_path: Path):
        self.config_path = config_path
        self.c = c
        self.root = Path(c["dataDirectory"])
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "adversarial.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,mode TEXT NOT NULL,"
            "body TEXT NOT NULL,verification TEXT NOT NULL,source TEXT,status TEXT NOT NULL,"
            "result TEXT,created REAL NOT NULL)"
        )
        self.db.commit()
        self.token = os.environ.get(c["tokenEnv"], "")
        self.upstream_token = os.environ.get(c["upstream"]["tokenEnv"], "")
        self.worker_id = c["upstream"]["workerId"]
        self.opener = build_opener(NoRedirect)
        if len(self.token) < 32 or len(self.upstream_token) < 32:
            raise ValueError("worker credentials unavailable")

    def scheduled(self, id: str):
        """Reload only the private schedule before acceptance.

        The validator allocates unpredictable holdout UUIDs after calibration.  A
        reload permits the controlled operator to add those IDs without changing an
        endpoint, credential source, key, or declared identity mid-window.
        """
        fresh = config(self.config_path)
        if {k: v for k, v in fresh.items() if k != "schedule"} != {
            k: v for k, v in self.c.items() if k != "schedule"
        }:
            raise ValueError("adversarial_immutable_configuration_changed")
        self.c["schedule"] = fresh["schedule"]
        return fresh["schedule"].get(id)

    def upstream(self, path: str, method="GET", body: bytes | None = None):
        q = Request(
            self.c["upstream"]["url"].rstrip("/") + path,
            data=body,
            method=method,
            headers={
                "authorization": "Bearer " + self.upstream_token,
                **({"content-type": "application/json"} if body is not None else {}),
            },
        )
        try:
            r = self.opener.open(q, timeout=190)
        except HTTPError as e:
            return e.code, dict(e.headers.items()), e.read(1024 * 1024 + 1)
        except URLError as e:
            raise ConnectionError("upstream_unreachable") from e
        with r:
            return r.status, dict(r.headers.items()), r.read(512 * 1024 * 1024 + 1)

    def row(self, id: str):
        return self.db.execute("SELECT * FROM jobs WHERE id=?", (id,)).fetchone()

    def store(self, id: str, mode: str, body: dict, verification: dict, source: str | None):
        with self.db:
            self.db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)",
                (
                    id,
                    mode,
                    json.dumps(body, sort_keys=True),
                    json.dumps(verification, sort_keys=True),
                    source,
                    "accepted",
                    None,
                    time.time(),
                ),
            )

    def sign(self, fields: dict[str, str]) -> dict[str, str]:
        unsigned = {"schema": SCHEMA, **fields}
        with tempfile.TemporaryDirectory() as d:
            msg, sig = Path(d) / "m", Path(d) / "s"
            msg.write_bytes(canon(unsigned))
            r = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-sign",
                    "-rawin",
                    "-inkey",
                    self.c["signingKeyFile"],
                    "-in",
                    str(msg),
                    "-out",
                    str(sig),
                ],
                capture_output=True,
                timeout=15,
            )
            if r.returncode:
                raise ValueError("receipt_signing_failed")
            return {**unsigned, "signature": base64.b64encode(sig.read_bytes()).decode()}

    def receipt(self, id: str, v: dict, result: str):
        return self.sign(
            {
                "jobId": id,
                "workerId": self.worker_id,
                "nonce": v["nonce"],
                "parametersHash": v["parametersHash"],
                "modelManifestHash": self.c["modelManifestHash"],
                "workflowHash": self.c["workflowHash"],
                "resultSha256": result,
            }
        )

    def canonical_media(self, p: Path) -> str:
        def out(args):
            r = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(p), *args, "-"],
                capture_output=True,
                timeout=180,
            )
            if r.returncode:
                raise ValueError("canonical_media_decode_failed")
            return digest(r.stdout)

        return digest(
            canon(
                {
                    "schema": "forge-h3-decoded-media/v1",
                    "width": 832,
                    "height": 480,
                    "duration": 5,
                    "fps": 24,
                    "video": out(
                        [
                            "-map",
                            "0:v:0",
                            "-vf",
                            "fps=24,scale=832:480:flags=neighbor",
                            "-pix_fmt",
                            "yuv420p",
                            "-f",
                            "rawvideo",
                        ]
                    ),
                    "audio": out(["-map", "0:a:0", "-ac", "2", "-ar", "48000", "-f", "f32le"]),
                }
            )
        )

    def media(self, id: str, source: Path):
        target = self.root / (id + ".mp4")
        if not target.exists():
            part = target.with_suffix(".partial.mp4")
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "25",
                    "-c:a",
                    "aac",
                    str(part),
                ],
                capture_output=True,
                timeout=240,
            )
            if r.returncode:
                part.unlink(missing_ok=True)
                raise ValueError("adversarial_transcode_failed")
            part.replace(target)
            os.chmod(target, 0o600)
        return target, self.canonical_media(target)


class Handler(BaseHTTPRequestHandler):
    worker: Worker

    def log_message(self, *_):
        pass

    def reply(self, status: int, data: bytes, headers: dict | None = None):
        self.send_response(status)
        self.send_header("x-worker-id", self.worker.worker_id)
        for k, v in (headers or {}).items():
            if k.lower() not in {
                "content-length",
                "connection",
                "transfer-encoding",
                "x-worker-id",
            }:
                self.send_header(k, v)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def js(self, status: int, v: Any):
        self.reply(status, canon(v), {"content-type": "application/json"})

    def err(self, status: int, code: str):
        self.js(status, {"error": {"code": code}})

    def auth(self):
        if self.headers.get("authorization") != "Bearer " + self.worker.token:
            self.err(401, "unauthorized")
            return False
        return True

    def read(self):
        try:
            n = int(self.headers.get("content-length", "-1"))
        except ValueError:
            n = -1
        if not 0 <= n <= 128 * 1024 * 1024:
            raise ValueError("body_too_large")
        return self.rfile.read(n)

    def forward(self, body: bytes | None = None):
        try:
            s, h, d = self.worker.upstream(self.path, self.command, body)
        except (ConnectionError, ValueError):
            self.err(503, "worker_reconciliation_required")
            return
        if h.get("x-worker-id") not in (None, self.worker.worker_id):
            self.err(503, "worker_identity_changed")
            return
        self.reply(s, d, {"content-type": h.get("content-type", "application/json")})

    def do_GET(self):
        if not self.auth():
            return
        m = JOB.fullmatch(self.path.split("?", 1)[0])
        row = self.worker.row(m.group(1)) if m else None
        if not row:
            self.forward()
            return
        id, action = m.groups()
        if action == "result":
            self.result(row)
            return
        if row["mode"] in {"wrong_video", "fast_profile", "wrong_nonce", "honest"}:
            s, h, d = self.worker.upstream("/jobs/" + id)
            if s != 200:
                self.reply(s, d, {"content-type": "application/json"})
                return
            v = json.loads(d)
            if v.get("status") == "succeeded":
                upstream = v.get("verificationReceipt")
                if not isinstance(upstream, dict) or not HASH.fullmatch(
                    str(upstream.get("resultSha256", ""))
                ):
                    self.err(503, "upstream_verification_unavailable")
                    return
                v["verificationReceipt"] = self.worker.receipt(
                    id, json.loads(row["verification"]), upstream["resultSha256"]
                )
            self.js(200, v)
            return
        result = json.loads(row["result"] or "{}")
        self.js(
            200,
            {
                "status": row["status"],
                "progress": 1 if row["status"] == "succeeded" else 0.1,
                **({"verificationReceipt": result["receipt"]} if "receipt" in result else {}),
            },
        )

    def result(self, row):
        id = row["id"]
        if row["mode"] in {"fast_profile", "wrong_nonce", "honest"}:
            s, h, d = self.worker.upstream("/jobs/" + id + "/result")
            if h.get("x-worker-id") not in (None, self.worker.worker_id):
                self.err(503, "worker_identity_changed")
                return
            self.reply(
                s,
                d,
                {
                    "content-type": h.get("content-type", "video/mp4"),
                    "x-content-sha256": h.get("x-content-sha256", ""),
                },
            )
            return
        target = self.worker.root / (id + ".mp4")
        if not target.is_file():
            self.err(409, "not_ready")
            return
        d = target.read_bytes()
        self.reply(200, d, {"content-type": "video/mp4", "x-content-sha256": digest(d)})

    def do_POST(self):
        if not self.auth():
            return
        m = JOB.fullmatch(self.path.split("?", 1)[0])
        if not m or m.group(2) == "cancel":
            if m and self.worker.row(m.group(1)):
                with self.worker.db:
                    self.worker.db.execute(
                        "UPDATE jobs SET status='cancelled' WHERE id=? AND status!='succeeded'",
                        (m.group(1),),
                    )
            try:
                self.forward(self.read())
            except ValueError:
                self.err(413, "body_too_large")
            return
        id = m.group(1)
        try:
            payload = json.loads(self.read())
        except (ValueError, json.JSONDecodeError):
            self.err(400, "invalid_json")
            return
        existing = self.worker.row(id)
        if existing:
            # A lost POST acknowledgement is reconciled against the same UUID.  This
            # never creates another GPU job; the upstream worker owns idempotency.
            if existing["mode"] == "replay":
                self.js(200, {"id": id})
                return
            try:
                s, _, d = self.worker.upstream("/jobs/" + id)
            except ConnectionError:
                self.err(503, "worker_reconciliation_required")
                return
            if s != 200:
                self.err(503, "worker_reconciliation_required")
                return
            try:
                upstream = json.loads(d)
            except json.JSONDecodeError:
                self.err(503, "worker_reconciliation_required")
                return
            if upstream is not None:
                self.js(200, {"id": id})
                return
            self.forward(existing["body"].encode())
            return
        try:
            attack = self.worker.scheduled(id)
        except ValueError as e:
            self.err(503, str(e))
            return
        v = payload.get("verification")
        if v is None and not attack:
            self.forward(canon(payload))
            return
        if (
            not isinstance(v, dict)
            or set(v) != {"schema", "nonce", "parametersHash"}
            or v.get("schema") != SCHEMA
            or not HASH.fullmatch(str(v.get("nonce", "")))
            or not HASH.fullmatch(str(v.get("parametersHash", "")))
        ):
            self.err(400, "invalid_verification_challenge")
            return
        try:
            if not attack:
                # Persist this proxy's acceptance before forwarding: GET after an uncertain
                # acknowledgment can reconcile the same UUID and no second worker is used.
                self.worker.store(id, "honest", payload, v, None)
                self.forward(canon(payload))
                return
            mode, source = attack["mode"], attack.get("source")
            if mode == "replay":
                self.worker.store(id, mode, payload, v, source)
                p, _ = self.worker.media(id, Path(source))
                r = self.worker.receipt(id, v, digest(p.read_bytes()))
                with self.worker.db:
                    self.worker.db.execute(
                        "UPDATE jobs SET status='succeeded',result=? WHERE id=?",
                        (json.dumps({"path": str(p), "receipt": r}), id),
                    )
                self.js(202, {"id": id})
                return
            if mode == "fast_profile":
                params = dict(payload["parameters"])
                params["profile"] = "taomate-3step"
                shape = {
                    k: params[k]
                    for k in (
                        "modality",
                        "duration",
                        "resolution",
                        "aspectRatio",
                        "audio",
                        "profile",
                    )
                }
                s, _, d = self.worker.upstream("/challenge-quotes", "POST", canon(shape))
                if s != 200:
                    self.reply(s, d, {"content-type": "application/json"})
                    return
                q = json.loads(d)
                altered = {
                    **payload,
                    "parameters": params,
                    "quote": q,
                    "verification": {**v, "parametersHash": digest(canon(params))},
                }
                self.worker.store(id, mode, altered, v, source)
                self.forward(canon(altered))
                return
            if mode == "wrong_video":
                self.worker.store(id, mode, payload, v, source)
                self.worker.media(id, Path(source))
                self.forward(canon(payload))
                return
            if mode == "wrong_nonce":
                wrong = {**v, "nonce": "0" * 64 if v["nonce"] != "0" * 64 else "1" * 64}
                altered = {**payload, "verification": wrong}
                self.worker.store(id, mode, altered, wrong, source)
                self.forward(canon(altered))
                return
            raise ValueError("invalid_attack_mode")
        except (ConnectionError, OSError, subprocess.SubprocessError, ValueError) as e:
            with self.worker.db:
                self.worker.db.execute("UPDATE jobs SET status='failed' WHERE id=?", (id,))
            self.err(503, str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    c = config(a.config)
    w = Worker(c, a.config)
    host, port = c["listen"].get("host"), c["listen"].get("port")
    if host not in {"127.0.0.1", "::1"} or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("loopback listen required")
    Handler.worker = w
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        w.db.close()


if __name__ == "__main__":
    main()
