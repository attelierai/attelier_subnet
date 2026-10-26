"""Real HTTP/SQLite/media boundaries; the paid remote GPU is a tiny HTTP fixture."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from forge.inference.scoring import score, trace_trusted_render_score
from forge.inference.service import Validator
from forge.inference.trace_protocol import (
    FIXED_PARAMETERS,
)
from forge.inference.trace_protocol import (
    canonical as trace_canonical,
)
from forge.inference.trace_protocol import (
    sha256 as trace_hash,
)


def test_durable_challenges_independent_decode_and_review_gate(tmp_path, monkeypatch):
    media = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=832x480:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(media),
        ],
        check=True,
    )
    data = media.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    jobs = {}
    submissions = []
    dropped = False
    drop_without_acceptance = False
    busy_next = False
    deny_requests = False

    class GPU(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if deny_requests:
                self.send_response(401)
                self.send_header("content-type", "application/json")
                self.send_header("x-worker-id", "fixture-worker")
                self.end_headers()
                self.wfile.write(b'{"error":{"code":"unauthorized"}}')
                return
            if self.headers.get("authorization") != "Bearer " + "x" * 40:
                self.send_error(401)
                return
            self.send_response(200)
            self.send_header("x-worker-id", "fixture-worker")
            if self.path.endswith("/result"):
                self.send_header("x-content-sha256", digest)
                self.end_headers()
                self.wfile.write(data)
                return
            self.end_headers()
            self.wfile.write(json.dumps(jobs.get(self.path.split("/")[-1])).encode())

        def do_POST(self):
            nonlocal busy_next, dropped, drop_without_acceptance
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path in ("/quotes", "/challenge-quotes"):
                result = {"token": "fixture", "amountMicros": 100, "expiresAt": 9999999999999}
            elif self.path.endswith("/cancel"):
                jobs[self.path.split("/")[-2]] = {"status": "cancelled"}
                result = {}
            else:
                job = self.path.split("/")[-1]
                submissions.append(job)
                if busy_next:
                    busy_next = False
                    self.send_response(409)
                    self.send_header("content-type", "application/json")
                    self.send_header("x-worker-id", "fixture-worker")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"code":"gpu_busy"}}')
                    return
                if drop_without_acceptance:
                    drop_without_acceptance = False
                    self.close_connection = True
                    return
                jobs[job] = {"status": "succeeded"}
                result = {}
                assert body["parameters"]["duration"] == 5
                if not dropped:
                    dropped = True
                    self.close_connection = True
                    return
            self.send_response(200)
            self.send_header("x-worker-id", "fixture-worker")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), GPU)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("H3_SUBNET_TEST_TOKEN", "x" * 40)
    registry = tmp_path / "registry.json"

    def write_registry(*, stale=False):
        registry.write_text(
            json.dumps(
                {
                    "network": "test",
                    "netuid": 578,
                    "mechanismCount": 2,
                    "generatedAt": datetime.fromtimestamp(
                        time.time() - 121 if stale else time.time(), UTC
                    ).isoformat(),
                    "uids": {"7": "fixture-hotkey"},
                }
            )
        )

    write_registry()
    config = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "source": "fixture",
        "registrationFile": str(registry),
        "miners": [
            {
                "id": "one",
                "uid": 7,
                "hotkey": "fixture-hotkey",
                "workerId": "fixture-worker",
                "url": f"http://127.0.0.1:{server.server_port}",
                "tokenEnv": "H3_SUBNET_TEST_TOKEN",
            }
        ],
    }
    p = {
        "modality": "text-to-video",
        "prompt": "Synthetic integration fixture",
        "duration": 5,
        "resolution": "480p",
        "aspectRatio": "16:9",
        "audio": True,
        "seed": 1,
        "profile": "base",
    }
    validator = Validator(tmp_path / "state", config, 1000)
    try:
        ids = validator.enqueue("batch", p, ceiling=100)
        assert validator.enqueue("batch", p, ceiling=100) == ids
        validator.step()
        validator.close()
        validator = Validator(tmp_path / "state", config, 1000)
        validator.step()
        rows = validator.rows()
        assert rows[0]["status"] == "review"
        assert submissions == ids
        assert json.loads(rows[0]["result"])["sha256"] == digest
        assert validator.scores("batch")["publishable"] is False
        with pytest.raises(ValueError):
            validator.review(ids[0], {"pass": True, "judge": "fixture"})
        validator.review(
            ids[0],
            {"pass": True, "sha256": digest, "judge": "fixture-only", "controlsPassed": True},
        )
        assert validator.scores("batch")["publishable"] is False, "one sample never qualifies"
        second = validator.enqueue("batch", {**p, "seed": 2}, ceiling=100)
        validator.step()
        validator.step()
        assert next(r for r in validator.rows() if r["id"] == second[0])["status"] == "invalid", (
            "different prompt seed cannot reuse result bytes"
        )
        score = validator.scores("batch")
        assert score["complete"] is True
        assert score["source"] == "fixture"
        assert score["miners"]["one"]["attempts"] == 2
        assert score["miners"]["one"]["succeeded"] == 1
        blocked = validator.enqueue("registry-blocked", {**p, "seed": 3}, ceiling=100)
        write_registry(stale=True)
        validator.step()
        row = next(r for r in validator.rows() if r["id"] == blocked[0])
        assert row["status"] == "queued"
        assert row["reason"] == "registration_unavailable"
        assert blocked[0] not in submissions
        write_registry()
        validator.step()
        assert next(r for r in validator.rows() if r["id"] == blocked[0])["status"] == "running"
        retry = validator.enqueue("registry-retry", {**p, "seed": 4}, ceiling=100)
        drop_without_acceptance = True
        validator.step()
        row = next(r for r in validator.rows() if r["id"] == retry[0])
        assert row["status"] == "submitting"
        assert row["reason"] == "worker_unreachable"
        assert submissions.count(retry[0]) == 1
        write_registry(stale=True)
        validator.step()
        row = next(r for r in validator.rows() if r["id"] == retry[0])
        assert row["status"] == "submitting"
        assert row["reason"] == "registration_unavailable"
        assert submissions.count(retry[0]) == 1, "stale admission cannot resend a job"
        write_registry()
        validator.step()
        assert next(r for r in validator.rows() if r["id"] == retry[0])["status"] == "running"
        assert submissions.count(retry[0]) == 2, "fresh admission can retry the same job ID"
        expired = validator.enqueue("registry-expired", {**p, "seed": 5}, ceiling=100)
        validator.db.execute(
            "UPDATE attempts SET deadline=? WHERE id=?", (time.time() - 1, expired[0])
        )
        validator.db.commit()
        write_registry(stale=True)
        validator.step()
        row = next(r for r in validator.rows() if r["id"] == expired[0])
        assert row["status"] == "inconclusive"
        assert row["reason"] == "challenge_not_submitted"
        assert row["completed"] is not None
        write_registry()
        drift_config = json.loads(json.dumps(config))
        drift = Validator(tmp_path / "drift", drift_config, 1000)
        try:
            drift_id = drift.enqueue("drift", {**p, "seed": 6}, ceiling=100)[0]
            drift_config["miners"][0]["workerId"] = "operator-replaced-worker"
            drift.step()
            drift_row = next(r for r in drift.rows() if r["id"] == drift_id)
            assert drift_row["status"] == "inconclusive"
            assert drift_row["reason"] == "worker_configuration_changed"
        finally:
            drift.close()
        busy = Validator(tmp_path / "busy", config, 1000)
        try:
            busy_id = busy.enqueue("busy", {**p, "seed": 7}, ceiling=100)[0]
            busy_next = True
            busy.step()
            busy_row = next(r for r in busy.rows() if r["id"] == busy_id)
            assert busy_row["status"] == "queued"
            assert busy_row["reason"] == "gpu_busy"
            assert busy_row["quote"] is not None
            assert submissions.count(busy_id) == 1
            busy.step()
            assert next(r for r in busy.rows() if r["id"] == busy_id)["status"] == "running"
            assert submissions.count(busy_id) == 2
        finally:
            busy.close()
        unauthorized = Validator(tmp_path / "unauthorized", config, 1000)
        try:
            unauthorized_id = unauthorized.enqueue("unauthorized", {**p, "seed": 8}, ceiling=100)[0]
            deny_requests = True
            unauthorized.step()
            unauthorized_row = next(r for r in unauthorized.rows() if r["id"] == unauthorized_id)
            assert unauthorized_row["status"] == "queued"
            assert unauthorized_row["reason"] == "worker_credential_unavailable"
            assert unauthorized_row["completed"] is None
        finally:
            deny_requests = False
            unauthorized.close()
        monkeypatch.delenv("H3_SUBNET_TEST_TOKEN")
        validator.close()
        validator = Validator(tmp_path / "state", config, 1000)
        assert validator.scores("registry-expired")["publishable"] is False
        with pytest.raises(ValueError):
            validator.enqueue("too-expensive", p, ceiling=1000)
    finally:
        validator.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_reference_mode_requires_signed_receipt_and_full_decoded_replay(tmp_path, monkeypatch):
    media = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=832x480:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(media),
        ],
        check=True,
    )
    private, public = tmp_path / "ed25519.pem", tmp_path / "pub.der"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ed25519", "-out", str(private)], check=True
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-outform", "DER", "-out", str(public)],
        check=True,
    )
    public_key = base64.b64encode(public.read_bytes()[-32:]).decode()
    blob, digest, jobs = media.read_bytes(), hashlib.sha256(media.read_bytes()).hexdigest(), {}

    def receipt(job, proof):
        signed = {
            "schema": "forge-h3-reference-v1",
            "jobId": job,
            "workerId": "worker",
            "nonce": proof["nonce"],
            "parametersHash": proof["parametersHash"],
            "modelManifestHash": "m" * 64,
            "workflowHash": "w" * 64,
            "resultSha256": digest,
        }
        message, sig = tmp_path / f"{job}.json", tmp_path / f"{job}.sig"
        message.write_text(json.dumps(signed, sort_keys=True, separators=(",", ":")))
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private),
                "-rawin",
                "-in",
                str(message),
                "-out",
                str(sig),
            ],
            check=True,
        )
        return {**signed, "signature": base64.b64encode(sig.read_bytes()).decode()}

    class GPU(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path in ("/quotes", "/challenge-quotes"):
                value = {"amountMicros": 1, "expiresAt": 9999999999999}
            else:
                jobs[self.path.rsplit("/", 1)[-1]] = body["verification"]
                value = {}
            self.send_response(200)
            self.send_header("x-worker-id", "worker")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            job = self.path.split("/")[2]
            if job not in jobs and not self.path.endswith("/result"):
                self.send_response(200)
                self.send_header("x-worker-id", "worker")
                self.end_headers()
                self.wfile.write(b"null")
                return
            self.send_response(200)
            self.send_header("x-worker-id", "worker")
            if self.path.endswith("/result"):
                self.send_header("x-content-sha256", digest)
                self.end_headers()
                self.wfile.write(blob)
                return
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"status": "succeeded", "verificationReceipt": receipt(job, jobs[job])}
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), GPU)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TOKEN", "x" * 40)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "network": "test",
                "netuid": 578,
                "mechanismCount": 2,
                "generatedAt": datetime.now(UTC).isoformat(),
                "uids": {"7": "hk"},
            }
        )
    )
    p = {
        "modality": "text-to-video",
        "prompt": "fixture",
        "duration": 5,
        "resolution": "480p",
        "aspectRatio": "16:9",
        "audio": True,
        "seed": 1,
        "profile": "base",
    }
    state = tmp_path / "state"
    config = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "source": "fixture",
        "registrationFile": str(registry),
        "referenceVerification": True,
        "calibrationBatch": "cal",
        "calibrationPerMiner": 1,
        "holdoutsPerMiner": 1,
        "reference": {
            "profile": "blackwell-nvfp4",
            "command": ["/bin/echo", json.dumps({"path": str(media)})],
            "timeoutSeconds": 30,
        },
        "miners": [
            {
                "id": "m",
                "uid": 7,
                "hotkey": "hk",
                "workerId": "worker",
                "url": f"http://127.0.0.1:{server.server_port}",
                "tokenEnv": "TOKEN",
                "modelManifestHash": "m" * 64,
                "workflowHash": "w" * 64,
                "verificationPublicKey": public_key,
            }
        ],
    }
    validator = Validator(state, config, 10)
    try:
        for batch, seed in (("cal", 1), ("hold", 1)):
            job = validator.enqueue(batch, {**p, "seed": seed}, ceiling=1)[0]
            validator.step()
            validator.step()
            validator.audit_pending()
            current = next(r for r in validator.rows() if r["id"] == job)
            assert current["status"] == "verified", (
                current["reason"],
                current["result"],
                current["reference"],
            )
        validator.freeze_calibration("hold")
        score = validator.scores("hold")
        assert score["complete"] and score["miners"]["m"]["availabilityRate"] == 1
    finally:
        validator.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_trace_trusted_render_scoring_reads_a_frozen_lane_ledger_without_rewriting_legacy(tmp_path):
    """Exercise the new scorer through the persisted audit-ledger boundary."""
    lane = {
        "hotkey": "fixture-hotkey",
        "workerId": "fixture-worker",
        "profileId": "h3-base-fl2va-nvfp4-20",
        "profileHash": "a" * 64,
        "rendererHash": "b" * 64,
    }
    policy = {"minCompleted": 4, "minVerified": 2, "windowStartUnix": 100, "windowEndUnix": 200}
    qualification = {"lane": lane, "profileQualified": True, "expiresAtUnix": 250}
    rows = [
        {
            "lane": lane,
            "status": "verified",
            "full20Verified": True,
            "resultTrusted": True,
            "completedAtUnix": 110,
            "referenceSeconds": 20,
            "workerSeconds": 10,
            "gatewayTtvSeconds": 32,
        },
        {
            "lane": lane,
            "status": "verified",
            "full20Verified": True,
            "resultTrusted": True,
            "completedAtUnix": 120,
            "referenceSeconds": 20,
            "workerSeconds": 20,
            "gatewayTtvSeconds": 40,
        },
        {
            "lane": lane,
            "status": "verified",
            "full20Verified": True,
            "resultTrusted": True,
            "completedAtUnix": 130,
            "referenceSeconds": 20,
            "workerSeconds": 30,
            "gatewayTtvSeconds": 50,
        },
        {"lane": lane, "status": "busy_capacity", "attribution": "miner", "completedAtUnix": 140},
        {"lane": lane, "status": "inconclusive", "completedAtUnix": 150},
        {"lane": lane, "status": "pending", "reauditDeadlineUnix": 190},
    ]
    ledger = tmp_path / "frozen-audit-ledger.json"
    ledger.write_text(json.dumps(rows))

    report = trace_trusted_render_score(
        json.loads(ledger.read_text()), policy=policy, qualification=qualification, now_unix=160
    )
    assert report["admission"] == "eligible"
    assert report["coverage"] == {
        "completed": 4,
        "verified": 3,
        "confirmedMinerFault": 1,
        "platformInconclusive": 1,
        "pending": 1,
        "malformedOrWrongLane": 0,
    }
    assert report["availabilityRate"] == pytest.approx(0.75)
    assert report["speedScore"] == pytest.approx((2 / 3 + 1 / 2 + 2 / 5) / 3)
    assert report["verifiedGatewayTtvMedianSeconds"] == 40

    rows[-1]["reauditDeadlineUnix"] = 159
    ledger.write_text(json.dumps(rows))
    suspended = trace_trusted_render_score(
        json.loads(ledger.read_text()), policy=policy, qualification=qualification, now_unix=160
    )
    assert suspended["admission"] == "suspended_reaudit_expired"
    assert suspended["score"] == report["score"]
    assert suspended["availabilityRate"] == report["availabilityRate"]
    assert suspended["rewardEligible"] is True

    terminal = trace_trusted_render_score(
        [
            *rows[:4],
            {
                "lane": lane,
                "status": "inconclusive",
                "completedAtUnix": 151,
                "admissionSuspended": True,
            },
        ],
        policy=policy,
        qualification=qualification,
        now_unix=160,
    )
    assert terminal["admission"] == "suspended_reaudit_expired"
    assert terminal["rewardEligible"] is True
    assert terminal["score"] == report["score"]

    all_faults = [
        {
            "lane": lane,
            "status": "timeout",
            "attribution": "miner",
            "completedAtUnix": 111 + index,
        }
        for index in range(4)
    ]
    zero = trace_trusted_render_score(
        all_faults, policy=policy, qualification=qualification, now_unix=160
    )
    assert zero["admission"] == "minimum_verified_not_met"
    assert zero["availabilityRate"] == 0
    assert zero["score"] == 0

    # Historical reference scoring keeps its original whole-window substitution rule.
    assert score(
        [{"status": "verified", "generationSeconds": 2, "reference": {"generationSeconds": 2}}]
    ) == {
        "availabilityRate": 1.0,
        "speedScore": 0.5,
        "score": 0.9,
    }


def test_trace_service_persists_request_then_uses_worker_evidence_and_local_runtime(
    tmp_path, monkeypatch
):
    """Trace mode crosses HTTP custody, SQLite intent and a local runtime subprocess."""
    profile = {
        "id": "h3-native-blackwell-20-v1",
        "modelManifestHash": "1" * 64,
        "workflowHash": "2" * 64,
        "runtimeHash": "3" * 64,
        "rendererHash": "4" * 64,
        "steps": 20,
        "parameters": FIXED_PARAMETERS,
        "numerics": {"maxRelativeRms": "0.000001"},
        "limits": {"maxChunks": 4, "maxChunkBytes": 4096, "maxBundleBytes": 8192},
    }
    profile_hash = trace_hash(profile)
    qualified_lane = {
        "hotkey": "trace-hotkey",
        "workerId": "trace-worker",
        "profileId": profile["id"],
        "profileHash": profile_hash,
        "rendererHash": profile["rendererHash"],
    }
    private, public = tmp_path / "trace-ed25519.pem", tmp_path / "trace-public.der"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)], check=True
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-outform", "DER", "-out", str(public)],
        check=True,
    )
    public_key = base64.b64encode(public.read_bytes()[-32:]).decode()
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    (profile_root / f"{profile['id']}.json").write_text(
        json.dumps(
            {
                "schema": "forge/h3-profile/v1",
                "profile": profile,
                "qualification": {
                    "status": "qualified",
                    "evidenceSha256": "5" * 64,
                    "expiresAt": 9_999_999_999,
                    "lanes": [qualified_lane],
                },
            }
        )
    )
    runtime = tmp_path / "runtime.py"
    runtime.write_text(
        "import argparse, hashlib, json\n"
        "from pathlib import Path\n"
        "from forge.inference.trace_protocol import sha256\n"
        "p=argparse.ArgumentParser()\n"
        "names=('request','receipt','profile-root','evidence-root','worker-public-key-file',\n"
        "       'output-dir','runtime-attestation','comfy-output-root')\n"
        "for n in names: p.add_argument('--'+n)\n"
        "p.add_argument('--comfy-origin'); p.add_argument('--hotkey'); "
        "p.add_argument('--timeout-seconds'); a=p.parse_args()\n"
        "r=json.loads(Path(a.request).read_text())\n"
        "q=json.loads(Path(a.receipt).read_text())\n"
        "prof=json.loads((Path(a.profile_root)/(r['profile']['id']+'.json')).read_text())['profile']\n"
        "out=Path(a.output_dir); out.mkdir()\n"
        "media=out/'rendered.mp4'; media.write_bytes(b'trusted-render')\n"
        "report={'schema':'forge-h3-verified-result/v1','jobId':r['jobId'],'requestHash':sha256(r),'receiptHash':sha256(q),'profile':r['profile'],'traceRoot':q['trace']['root'],'verdict':'verified','reason':'verified','checks':{'conditioning':'pass','predictions':{'checked':20,'matched':20,'total':20},'recurrence':'pass','terminalBinding':'pass'},'finalLatent':{'video':'a'*64,'audio':'b'*64},'rendererHash':prof['rendererHash'],'result':{'path':str(media),'sha256':hashlib.sha256(media.read_bytes()).hexdigest(),'bytes':media.stat().st_size},'timingsMs':{'preprocess':1,'replay':2,'render':3}}\n"
        "report['finalLatent']={x['component']:q['trace']['chunks'][x['chunkIndex']]['sha256']\n"
        "                       for x in q['finalLatent']}\n"
        "report['lane']={'hotkey':a.hotkey,'workerId':r['workerId'],"
        "'profileId':prof['id'],'profileHash':sha256(prof),"
        "'rendererHash':prof['rendererHash']}\n"
        "if r['requestedSeed']==8:\n"
        "  report={k:report[k] for k in ('schema','jobId','requestHash','receiptHash',\n"
        "                               'profile','traceRoot','rendererHash','lane')}\n"
        "  report.update(verdict='inconclusive',reason='fixture_runtime_unknown')\n"
        "(out/'report.json').write_text(json.dumps(report))\n"
    )
    attestation = tmp_path / "runtime-attestation.json"
    attestation.write_text("{}")
    output_root = tmp_path / "comfy-output"
    output_root.mkdir()
    payloads = [b"{}", b"external GPU fixture video latent", b"external GPU fixture audio latent"]
    names = ["capture.json", "tensors/video.pt", "tensors/audio.pt"]
    chunks = [
        {
            "index": i,
            "name": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for i, (name, payload) in enumerate(zip(names, payloads, strict=True))
    ]
    posted, submitted_requests, submitted_jobs = [], {}, set()

    def receipt(job: str, request: dict) -> dict:
        signed = {
            "schema": "forge-h3-trace/v1",
            "jobId": job,
            "workerId": "trace-worker",
            "nonce": request["nonce"],
            "requestHash": trace_hash(request),
            "parametersHash": trace_hash(request["parameters"]),
            "profile": request["profile"],
            "modelManifestHash": profile["modelManifestHash"],
            "workflowHash": profile["workflowHash"],
            "steps": profile["steps"],
            "trace": {
                "algorithm": "sha256-manifest-v1",
                "chunks": chunks,
                "root": trace_hash(chunks),
            },
            "finalLatent": [
                {"component": "video", "chunkIndex": 1},
                {"component": "audio", "chunkIndex": 2},
            ],
        }
        message, signature = tmp_path / f"{job}.json", tmp_path / f"{job}.sig"
        message.write_bytes(trace_canonical(signed))
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private),
                "-rawin",
                "-in",
                str(message),
                "-out",
                str(signature),
            ],
            check=True,
            capture_output=True,
        )
        return {**signed, "signature": base64.b64encode(signature.read_bytes()).decode()}

    class GPU(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/challenge-quotes":
                value = {"amountMicros": 1, "expiresAt": 9_999_999_999_999}
            else:
                assert "traceRequest" in body and "verification" not in body
                posted.append(body["traceRequest"])
                submitted_requests[self.path.rsplit("/", 1)[-1]] = body["traceRequest"]
                submitted_jobs.add(self.path.rsplit("/", 1)[-1])
                value = {}
            self.send_response(200)
            self.send_header("x-worker-id", "trace-worker")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            self.send_response(200)
            self.send_header("x-worker-id", "trace-worker")
            if "/evidence/" in self.path:
                chunk = payloads[int(self.path.rsplit("/", 1)[-1])]
                self.send_header("x-content-sha256", hashlib.sha256(chunk).hexdigest())
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.end_headers()
            if self.path.endswith("/evidence"):
                job = self.path.split("/")[2]
                self.wfile.write(json.dumps(receipt(job, submitted_requests[job])).encode())
            elif self.path.split("/")[2] in submitted_jobs:
                self.wfile.write(
                    json.dumps({"status": "succeeded", "metrics": {"evidenceReady": True}}).encode()
                )
            else:
                self.wfile.write(b"null")

    server = ThreadingHTTPServer(("127.0.0.1", 0), GPU)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TRACE_TOKEN", "x" * 40)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "network": "test",
                "netuid": 578,
                "mechanismCount": 2,
                "generatedAt": datetime.now(UTC).isoformat(),
                "uids": {"7": "trace-hotkey"},
            }
        )
    )
    config = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "source": "fixture",
        "registrationFile": str(registry),
        "verificationMode": "trace_trusted_render",
        "traceVerification": {
            "profileRoot": str(profile_root),
            "profile": {"id": profile["id"], "hash": profile_hash},
            "command": [sys.executable, str(runtime)],
            "timeoutSeconds": 60,
            "reauditSeconds": 60,
            "windowSeconds": 600,
            "minCompleted": 1,
            "minVerified": 1,
            "referenceSeconds": 20,
            "comfyOrigin": "http://127.0.0.1:8188",
            "comfyOutputRoot": str(output_root),
            "runtimeAttestation": str(attestation),
        },
        "miners": [
            {
                "id": "trace",
                "uid": 7,
                "hotkey": "trace-hotkey",
                "workerId": "trace-worker",
                "url": f"http://127.0.0.1:{server.server_port}",
                "tokenEnv": "TRACE_TOKEN",
                "verificationPublicKey": public_key,
            }
        ],
    }
    params = {**FIXED_PARAMETERS, "prompt": "trace fixture", "seed": 7}
    validator = Validator(tmp_path / "state", config, 10)
    try:
        profile_file = profile_root / f"{profile['id']}.json"
        envelope = json.loads(profile_file.read_text())
        for field in (None, *qualified_lane):
            lane = dict(qualified_lane)
            if field is not None:
                lane[field] = "unqualified-lane"
            envelope["qualification"]["lanes"] = [] if field is None else [lane]
            profile_file.write_text(json.dumps(envelope))
            with pytest.raises(ValueError, match="trace_lane_unqualified"):
                validator.enqueue("trace", params, ceiling=1)
            assert validator.rows() == [] and posted == []
        envelope["qualification"]["lanes"] = [qualified_lane]
        profile_file.write_text(json.dumps(envelope))
        job = validator.enqueue("trace", params, ceiling=1)[0]
        persisted = validator.db.execute(
            "SELECT trace_request FROM attempts WHERE id=?", (job,)
        ).fetchone()[0]
        assert persisted and json.loads(persisted)["schema"] == "forge-h3-request/v1"
        validator.step()
        validator.step()
        validator.audit_pending()
        row = next(item for item in validator.rows() if item["id"] == job)
        assert row["status"] == "verified"
        assert posted[0]["parameters"]["seed"] != params["seed"]
        assert not (tmp_path / "state" / f"{job}.mp4").exists()
        first_score = validator.scores("trace")["miners"]["trace"]
        assert first_score["rewardEligible"] is True
        assert first_score["speedScore"] == pytest.approx(20 / (20 + row["generation"]))
        # Reconfiguring an existing miner id cannot transfer its verified history.
        for field in ("hotkey", "workerId"):
            old = config["miners"][0][field]
            config["miners"][0][field] = "replacement"
            registration = json.loads(registry.read_text())
            registration["uids"]["7"] = config["miners"][0]["hotkey"]
            registry.write_text(json.dumps(registration))
            changed = validator.scores("trace")["miners"]["trace"]
            assert changed["verificationStatus"] == "trace_attempt_lane_changed"
            assert changed["weight"] == 0 and changed["score"] is None
            config["miners"][0][field] = old
            registration["uids"]["7"] = config["miners"][0]["hotkey"]
            registry.write_text(json.dumps(registration))
        assert (
            validator.db.execute("SELECT reference_seconds FROM trace_windows").fetchone()[0] == 20
        )

        clock_job = validator.enqueue("trace", {**params, "seed": 9}, ceiling=1)[0]
        validator.step()
        validator.db.execute("UPDATE attempts SET sent_boot='old-boot' WHERE id=?", (clock_job,))
        validator.db.commit()
        validator.step()
        validator.audit_pending()
        clock_row = next(item for item in validator.rows() if item["id"] == clock_job)
        assert clock_row["status"] == "verified"
        assert clock_row["generation"] is None

        unknown_job = validator.enqueue("trace", {**params, "seed": 8}, ceiling=1)[0]
        validator.step()
        validator.step()
        validator.audit_pending()
        validator.audit_pending()
        unknown_row = next(item for item in validator.rows() if item["id"] == unknown_job)
        assert unknown_row["status"] == "inconclusive"
        assert unknown_row["reason"] == "trace_reaudit_exhausted"
        suspended = validator.scores("trace")["miners"]["trace"]
        assert suspended["admission"] == "suspended_reaudit_expired"
        assert suspended["rewardEligible"] is True
        assert suspended["weight"] == 1
        from forge.inference.dashboard import project

        public = project(
            validator, "trace", {"block": 123, "mechanismCount": 2, "emissionSplit": [1, 1]}, None
        )
        assert public["verification"]["rewardEligible"] is True
        assert public["verification"]["renderer"]["status"] == "active"
        assert public["chain"]["observedSplit"] == {
            "optimization": 0.5,
            "inference": 0.5,
            "block": 123,
        }
        assert public["miners"][0]["admission"]["state"] == "suspended"
        assert public["miners"][0]["verifiedJobs"] == 2
        assert public["miners"][0]["inconclusiveJobs"] == 1
        served = next(item for item in public["jobs"] if item["id"] == job)
        assert served["delivery"]["by"] == "trusted_renderer"
        assert served["ttv"]["total"] == served["latencySeconds"]
        assert served["ttv"]["queue"] is None and served["ttv"]["storage"] is None
        assert str(tmp_path) not in json.dumps(public) and "trace fixture" not in json.dumps(public)
        assert all(
            item.get("delivery") is None for item in public["jobs"] if item["status"] != "verified"
        )
        with pytest.raises(ValueError, match="trace_lane_admission_suspended"):
            validator.enqueue("trace", {**params, "seed": 10}, ceiling=1)
    finally:
        validator.close()
        server.shutdown()
        server.server_close()
        thread.join()
