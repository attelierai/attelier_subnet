#!/usr/bin/env python3
"""Private two-device experiment; never writes a qualified profile or service state.

The validator pulls signed captures from an independently provisioned miner.
All tensors stay on the GPU volumes. A separate control-host client must verify
each exported diagnostic artifact before any end-to-end claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from h3_remote_runtime import validate_diagnostic_artifact
from h3_trusted_qualification_harness import diagnostic, keypair, receipt, run_graph, write_new
from h3_trusted_runtime import validate_capture_metadata

from forge.inference.trace_protocol import file_sha256, load_profile, sha256
from forge.inference.trace_runtime import (
    NativeH3Profile,
    TrustedT2VInput,
    build_trusted_baseline_graph,
    write_graph_new,
)


def case_key(row):
    parts = [row["lineage"], row["condition"]]
    if any(not re.fullmatch(r"[a-z0-9-]+", part) for part in parts):
        raise RuntimeError("case_key_invalid")
    return "/".join(parts)


def derive_capture(source: Path, destination: Path, condition: str):
    """Hard-link immutable tensors, then detach every changed file before writing."""
    shutil.copytree(source, destination, copy_function=os.link)
    if condition == "stale-request":
        return
    for name in ("receipt.json", "receipt-message.json", "receipt.sig"):
        (destination / name).unlink()
    if condition != "altered-final-latent":
        return
    import torch

    path = destination / "capture.json"
    capture = json.loads(path.read_text())
    item = next(item for item in capture["finalLatent"] if item["component"] == "video")
    tensor_path = destination / item["tensorPath"]
    value = torch.load(tensor_path, map_location="cpu", weights_only=True).clone()
    value.reshape(-1)[0] += 1
    tensor_path.unlink()
    torch.save(value, tensor_path)
    item.update(sha256=file_sha256(tensor_path), bytes=tensor_path.stat().st_size)
    path.unlink()
    write_new(path, capture)


def capture(a):
    a.root = a.root.resolve()
    payload = json.load(sys.stdin)
    row, request = payload["case"], payload["request"]
    case = a.root / "cases" / case_key(row)
    case.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_new(case / "request.json", request)
    profile = load_profile(a.profile_root, request["profile"], allow_unqualified=True)
    private, public = keypair(a.root)
    destination = case / "evidence"
    source_key = row.get("derivedFrom")
    started = time.monotonic()
    existing = row.get("existingCapture")
    if existing:
        # Recovery of an unscored development smoke only; never reuse holdout
        # generations or turn an idempotent response into a second honest sample.
        source = Path(existing).resolve()
        source.relative_to(a.root.parent)
        if (row["condition"] != "smoke-full20" or row["lineage"] != "smoke"
                or file_sha256(source / "capture.json") != row["existingCaptureSha256"]):
            raise RuntimeError("existing_smoke_capture_binding")
        validate_capture_metadata(
            json.loads((source / "capture.json").read_text()), request, profile
        )
        shutil.copytree(source, destination, copy_function=os.link)
    elif source_key:
        if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", source_key):
            raise RuntimeError("source_case_invalid")
        source = a.root / "cases" / source_key / "evidence"
        derive_capture(source, destination, row["condition"])
    else:
        trusted = TrustedT2VInput.from_validated({
            "jobId": request["jobId"], "nonce": request["nonce"],
            "parametersHash": sha256(request["parameters"]), "requestHash": sha256(request),
            "parameters": request["parameters"], "effectiveSeed": request["parameters"]["seed"],
        })
        graph = build_trusted_baseline_graph(
            trusted, NativeH3Profile.from_validated(profile),
            trace_root=case / "raw", filename_prefix="qualification/" + request["jobId"],
        )
        if row["condition"] == "steps-10":
            graph["9"]["inputs"]["steps"] = 10
        elif row["condition"] == "skip-call-7":
            graph["10"]["inputs"]["skip_model_step"] = 7
        write_graph_new(case / "capture-graph.json", graph)
        run_graph(a.comfy_origin, graph, "capture-" + row["jobId"], a.timeout_seconds)
        (case / "raw" / request["jobId"]).rename(destination)
        os.link(destination / "manifest.json", destination / "capture.json")
    if row["condition"] != "stale-request":
        receipt(destination / "capture.json", request, profile, private)
    shutil.copyfile(public, case / "worker-public-key.txt")
    if row["condition"] == "substituted-miner-media":
        # Valid but deliberately unrelated bytes, outside the signed tensor bundle.
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=magenta:s=832x480:r=24",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(case / "miner.mp4"),
        ], check=True, timeout=30)
    summary = {
        "case": case_key(row), "requestHash": sha256(request),
        "captureMs": int((time.monotonic() - started) * 1000),
        "receiptSha256": file_sha256(destination / "receipt.json"),
        "reusedUnscoredSmoke": bool(existing),
        "minerMediaSha256": file_sha256(case / "miner.mp4")
        if (case / "miner.mp4").exists() else None,
    }
    write_new(case / "capture-summary.json", summary)
    print(json.dumps(summary), flush=True)


def run(a):
    if file_sha256(a.plan) != a.plan_sha256:
        raise RuntimeError("plan_changed")
    plan = json.loads(a.plan.read_text())
    config = json.loads(a.miner_config.read_text())
    if not re.fullmatch(r"root@[A-Za-z0-9.-]+", config["target"]):
        raise RuntimeError("miner_target_invalid")
    ssh = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        "-o", "UserKnownHostsFile=" + config["knownHostsFile"],
        "-i", config["identityFile"], "-p", str(config["port"]),
    ]
    a.root.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_new(a.root / "plan.json", plan)
    results = []
    for row in plan["cases"]:
        remaining = a.deadline_epoch - time.time()
        # Enough time for one bounded capture + verifier, followed by export/closure.
        if remaining < 2 * a.timeout_seconds:
            break
        key = case_key(row)
        case = a.root / "cases" / key
        case.mkdir(parents=True, exist_ok=False, mode=0o700)
        request = row["request"]
        write_new(case / "request.json", request)
        command = [
            "env", "PYTHONPATH=/workspace/forge/code/src", "PYTHONDONTWRITEBYTECODE=1",
            "/workspace/forge/.venv/bin/python",
            "/workspace/forge/code/scripts/verification/h3_two_lane_qualification.py", "capture",
            "--root", str(a.miner_root), "--profile-root", str(a.profile_root),
            "--comfy-origin", a.comfy_origin, "--timeout-seconds", str(a.timeout_seconds),
        ]
        print(json.dumps({"event": "capture_started", "case": key}), flush=True)
        started = time.monotonic()
        try:
            done = subprocess.run(
                [*ssh, config["target"], shlex.join(command)],
                input=json.dumps({"case": row, "request": request}), text=True,
                capture_output=True, timeout=a.timeout_seconds + 90, check=False,
            )
            (case / "capture-process.log").write_text(done.stdout + done.stderr)
            if done.returncode:
                raise RuntimeError("miner_capture_failed")
            subprocess.run([
                "rsync", "--archive", "--protect-args", "-e", shlex.join(ssh),
                config["target"] + ":" + str(a.miner_root / "cases" / key) + "/",
                str(case / "miner") + "/",
            ], check=True, capture_output=True, timeout=180)
            captured = json.loads((case / "miner/capture-summary.json").read_text())
            if captured["requestHash"] != sha256(request):
                raise RuntimeError("capture_response_request_mismatch")
            output = case / "verifier"
            diagnostic(
                a, case / "request.json", case / "miner/evidence/receipt.json",
                case / "miner/evidence", case / "miner/worker-public-key.txt", output,
                walltime_seconds=max(1, a.deadline_epoch - time.time()),
            )
            report = json.loads((output / "report.json").read_text())
            verdict = report.get("diagnosticVerdict", report["verdict"])
            expected = "invalid" if row["condition"] in {
                "steps-10", "skip-call-7", "altered-final-latent", "stale-request",
            } else "verified"
            before_gpu = not (output / "baseline-graph.json").exists()
            passed = verdict == expected and report["verdict"] == "inconclusive"
            if row["condition"] == "stale-request":
                passed = passed and before_gpu
            result = {
                "case": key, "evaluationId": row["jobId"], "requestHash": sha256(request),
                "expected": expected, "diagnosticVerdict": verdict,
                "publicVerdict": report["verdict"], "reason": report["reason"],
                "passed": passed, "beforeGpu": before_gpu,
                "elapsedMs": int((time.monotonic() - started) * 1000),
                "minerMediaSha256": captured["minerMediaSha256"],
                "clientCustody": "pending" if expected == "verified" else "not_applicable",
            }
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            result = {
                "case": key, "passed": False, "diagnosticVerdict": "inconclusive",
                "reason": str(exc) if str(exc).isidentifier()
                else "experiment_infrastructure_error",
            }
        write_new(case / "result.json", result)
        results.append(result)
        print(json.dumps({"event": "case_finished", **result}), flush=True)
        if not result["passed"]:
            break
    summary = {
        "schema": "forge/h3-two-lane-experiment/v1", "planSha256": a.plan_sha256,
        "status": "PENDING_CLIENT_CUSTODY" if len(results) == len(plan["cases"])
        and all(row["passed"] for row in results) else "INCOMPLETE_OR_FAILED",
        "results": results, "profileQualified": False,
        "productionApiIntegrated": False,
    }
    write_new(a.root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def client(a):
    """Receive actual validator media bytes over SSH without saving media locally."""
    config = json.loads(a.remote_config.read_text())
    plan = json.loads(a.plan.read_text())
    ssh = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        "-o", "UserKnownHostsFile=" + config["knownHostsFile"],
        "-i", config["identityFile"], "-p", str(config["port"]), config["target"],
    ]

    def document(path):
        completed = subprocess.run(
            [*ssh, shlex.join(["cat", "--", str(path)])],
            capture_output=True, check=True, timeout=30,
        )
        if len(completed.stdout) > 2 * 1024**2:
            raise RuntimeError("remote_document_too_large")
        return json.loads(completed.stdout)

    summary = document(a.remote_root / "summary.json")
    if summary["planSha256"] != file_sha256(a.plan):
        raise RuntimeError("client_plan_mismatch")
    receipts = []
    for result in summary["results"]:
        if not result["passed"] or result["diagnosticVerdict"] != "verified":
            continue
        row = next(row for row in plan["cases"] if case_key(row) == result["case"])
        path = a.remote_root / "cases" / case_key(row) / "verifier"
        report = document(path / "report.json")
        request = row["request"]
        expected_lane = {
            "hotkey": "qualification-diagnostic-only", "workerId": request["workerId"],
            "profileId": request["profile"]["id"], "profileHash": request["profile"]["hash"],
            "rendererHash": plan["profile"]["rendererHash"],
        }
        if (report["requestHash"] != sha256(request) or report["jobId"] != request["jobId"]
                or report["profile"] != request["profile"] or report["lane"] != expected_lane):
            raise RuntimeError("client_report_request_mismatch")
        artifact = document(path / "diagnostic-artifact.json")
        media = validate_diagnostic_artifact(artifact, report)
        process = subprocess.Popen(
            [*ssh, shlex.join(["cat", "--", str(path / "result.mp4")])],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        digest, size, deadline = hashlib.sha256(), 0, time.monotonic() + 120
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError("client_media_timeout")
                if not select.select([process.stdout], [], [], 1)[0]:
                    continue
                block = os.read(process.stdout.fileno(), 1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > media["bytes"]:
                    raise RuntimeError("client_media_overflow")
                digest.update(block)
            if process.wait(timeout=10) or size != media["bytes"]:
                raise RuntimeError("client_media_incomplete")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdout.close()
        if digest.hexdigest() != media["sha256"]:
            raise RuntimeError("client_media_changed")
        if result.get("minerMediaSha256") == digest.hexdigest():
            raise RuntimeError("client_received_miner_media")
        receipts.append({
            "case": case_key(row), "requestHash": sha256(request), "reportHash": sha256(report),
            "artifactHash": sha256(artifact), "bytesReceived": size,
            "receivedSha256": digest.hexdigest(),
            "minerMediaSha256": result.get("minerMediaSha256"),
            "transport": "private_authenticated_ssh_stream", "storedMediaLocally": False,
        })
    proof = {
        "schema": "forge/h3-private-client-custody/v1", "receipts": receipts,
        "serverSummaryHash": sha256(summary), "planSha256": file_sha256(a.plan),
        "productionApiIntegrated": False, "profileQualified": False,
    }
    write_new(a.output, proof)
    print(json.dumps({"deliveries": len(receipts), "receiptSha256": file_sha256(a.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("capture", "run"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--profile-root", type=Path, required=True)
        command.add_argument("--comfy-origin", default="http://127.0.0.1:8188")
        command.add_argument("--timeout-seconds", type=int, default=600)
    command = sub.choices["run"]
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--plan-sha256", required=True)
    command.add_argument("--miner-config", type=Path, required=True)
    command.add_argument("--miner-root", type=Path, required=True)
    command.add_argument("--runtime-attestation", type=Path, required=True)
    command.add_argument("--comfy-output-root", type=Path, required=True)
    command.add_argument("--deadline-epoch", type=float, required=True)
    command = sub.add_parser("client")
    command.add_argument("--remote-config", type=Path, required=True)
    command.add_argument("--remote-root", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "client":
        client(args)
        return
    if not 60 <= args.timeout_seconds <= 600:
        parser.error("timeout must be between 60 and 600 seconds")
    capture(args) if args.action == "capture" else run(args)


if __name__ == "__main__":
    main()
