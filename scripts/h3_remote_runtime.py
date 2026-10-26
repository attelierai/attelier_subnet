#!/usr/bin/env python3
"""Transport a sealed job to an operator-controlled GPU over authenticated SSH.

This adapter implements the trusted runtime CLI on a CPU gateway. The remote
profile/code/attestation are provisioned independently by the operator. Miner
data cannot select an executable, host, profile or renderer. Large tensors stay
on the gateway and GPU disks; rsync streams only receipt-listed files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from forge.inference.trace_protocol import (
    canonical,
    file_sha256,
    load_profile_envelope,
    qualified_lane,
    read_json,
    sha256,
    verify_bundle,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "remote-config",
        "request",
        "receipt",
        "profile-root",
        "evidence-root",
        "worker-public-key-file",
        "output-dir",
        "runtime-attestation",
        "comfy-output-root",
    ):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--comfy-origin", required=True)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--allow-unqualified-diagnostic", action="store_true")
    parser.add_argument("--export-diagnostic-artifact", action="store_true")
    args = parser.parse_args()
    if args.export_diagnostic_artifact and not args.allow_unqualified_diagnostic:
        parser.error("--export-diagnostic-artifact requires --allow-unqualified-diagnostic")
    if args.output_dir.exists() or not 60 <= args.timeout_seconds <= 3600:
        raise RuntimeError("output_exists_or_timeout_invalid")
    config = read_json(args.remote_config)
    if config.get("schema") != "forge-h3-remote-runtime/v1":
        raise RuntimeError("remote_config_schema")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*@[A-Za-z0-9.-]+", config.get("target", "")):
        raise RuntimeError("remote_target_invalid")
    if type(config.get("port")) is not int or not 1 <= config["port"] <= 65535:
        raise RuntimeError("remote_port_invalid")
    for field in (
        "python",
        "codeRoot",
        "profileRoot",
        "runtimeAttestation",
        "comfyOutputRoot",
        "jobRoot",
    ):
        value = config.get(field, "")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in value.split("/"):
            raise RuntimeError("remote_path_invalid")
    if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", config.get("comfyOrigin", "")):
        raise RuntimeError("remote_comfy_not_loopback")
    for field in ("identityFile", "knownHostsFile"):
        if not Path(config.get(field, "")).is_file():
            raise RuntimeError("remote_ssh_identity_unavailable")
    request, receipt = read_json(args.request), read_json(args.receipt)
    envelope = load_profile_envelope(
        args.profile_root, request["profile"], allow_unqualified=args.allow_unqualified_diagnostic
    )
    profile = envelope["profile"]
    if not args.allow_unqualified_diagnostic:
        qualified_lane(profile, envelope["qualification"], args.hotkey, request["workerId"])
    verify_bundle(
        request,
        receipt,
        args.evidence_root,
        profile,
        args.worker_public_key_file.read_text().strip(),
    )
    observed = read_json(args.runtime_attestation)
    if observed.get("runtimeHash") != profile["runtimeHash"]:
        raise RuntimeError("local_runtime_attestation_changed")
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    ssh = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "UserKnownHostsFile=" + config["knownHostsFile"],
        "-i",
        config["identityFile"],
        "-p",
        str(config["port"]),
    ]

    def run(argv, data=None):
        left = deadline - time.monotonic()
        if left <= 0:
            raise RuntimeError("remote_runtime_deadline")
        done = subprocess.run(
            argv,
            input=data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=left,
            check=False,
        )
        if done.returncode:
            raise RuntimeError("remote_runtime_transport_failed")

    target = config["target"]
    base = config["jobRoot"].rstrip("/") + "/" + request["jobId"] + "-" + secrets.token_hex(8)
    # A unique directory is created once. Re-audits cannot overwrite previous evidence.
    setup = "umask 077; " + shlex.join(["mkdir", "-p", "--", config["jobRoot"]])
    setup += " && " + shlex.join(["mkdir", "--", base])
    setup += " && " + shlex.join(["mkdir", "--", base + "/inputs", base + "/evidence"])
    run([*ssh, target, setup])
    transport = [
        "rsync",
        "--archive",
        "--protect-args",
        "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
        "-e",
        shlex.join(ssh),
    ]
    with tempfile.TemporaryDirectory(prefix="h3-runtime-inputs-") as temporary:
        stage = Path(temporary)
        for name, value in (("request.json", request), ("receipt.json", receipt)):
            (stage / name).write_bytes(canonical(value))
        (stage / "worker-public-key.txt").write_text(args.worker_public_key_file.read_text())
        run([*transport, str(stage) + "/", target + ":" + base + "/inputs/"])
    names = b"\0".join(item["name"].encode("ascii") for item in receipt["trace"]["chunks"]) + b"\0"
    run(
        [
            *transport,
            "--from0",
            "--files-from=-",
            str(args.evidence_root.resolve()) + "/",
            target + ":" + base + "/evidence/",
        ],
        names,
    )
    left = int(deadline - time.monotonic()) - 30
    if left < 60:
        raise RuntimeError("remote_runtime_insufficient_time")
    argv = [
        "timeout",
        "--kill-after=10s",
        str(left) + "s",
        "env",
        "PYTHONPATH=" + config["codeRoot"] + "/src",
        config["python"],
        config["codeRoot"] + "/scripts/h3_trusted_runtime.py",
        "--request",
        base + "/inputs/request.json",
        "--hotkey",
        args.hotkey,
        "--receipt",
        base + "/inputs/receipt.json",
        "--profile-root",
        config["profileRoot"],
        "--evidence-root",
        base + "/evidence",
        "--worker-public-key-file",
        base + "/inputs/worker-public-key.txt",
        "--output-dir",
        base + "/output",
        "--runtime-attestation",
        config["runtimeAttestation"],
        "--comfy-origin",
        config["comfyOrigin"],
        "--comfy-output-root",
        config["comfyOutputRoot"],
        "--timeout-seconds",
        str(left),
    ]
    if args.allow_unqualified_diagnostic:
        argv.append("--allow-unqualified-diagnostic")
    if args.export_diagnostic_artifact:
        argv.append("--export-diagnostic-artifact")
    run([*ssh, target, shlex.join(argv)])
    args.output_dir.mkdir(parents=True, mode=0o700)
    run(
        [
            *transport,
            target + ":" + base + "/output/report.json",
            str(args.output_dir / "remote-report.json"),
        ]
    )
    report = read_json(args.output_dir / "remote-report.json")
    if (
        report.get("schema") != "forge-h3-verified-result/v1"
        or report.get("jobId") != request["jobId"]
        or report.get("requestHash") != sha256(request)
        or report.get("receiptHash") != sha256(receipt)
        or report.get("profile") != request["profile"]
        or report.get("traceRoot") != receipt["trace"]["root"]
        or report.get("rendererHash") != profile["rendererHash"]
        or report.get("lane")
        != {
            "hotkey": args.hotkey,
            "workerId": request["workerId"],
            "profileId": profile["id"],
            "profileHash": sha256(profile),
            "rendererHash": profile["rendererHash"],
        }
        or report.get("verdict") not in ("verified", "invalid", "inconclusive")
    ):
        raise RuntimeError("remote_report_binding")
    if report["verdict"] == "verified":
        result = report.get("result", {})
        if (
            args.allow_unqualified_diagnostic
            or result.get("path") != base + "/output/result.mp4"
            or type(result.get("bytes")) is not int
            or not 0 < result["bytes"] <= 512 * 1024**2
        ):
            raise RuntimeError("remote_result_invalid")
        run([*transport, target + ":" + result["path"], str(args.output_dir / "result.mp4")])
        media = args.output_dir / "result.mp4"
        if media.stat().st_size != result["bytes"] or file_sha256(media) != result.get("sha256"):
            raise RuntimeError("remote_result_custody")
        result["path"] = str(media.resolve())
    elif "result" in report:
        raise RuntimeError("unverified_remote_media")
    diagnostic_receipt_hash = None
    if args.export_diagnostic_artifact and report.get("diagnosticVerdict") == "verified":
        if report["verdict"] != "inconclusive" or "finalLatent" in report:
            raise RuntimeError("diagnostic_report_not_inconclusive")
        artifact_path = args.output_dir / "diagnostic-artifact.json"
        run(
            [
                *transport,
                target + ":" + base + "/output/diagnostic-artifact.json",
                str(artifact_path),
            ]
        )
        artifact = read_json(artifact_path)
        media_record = validate_diagnostic_artifact(artifact, report)
        media = args.output_dir / "diagnostic-result.mp4"
        run([*transport, target + ":" + base + "/output/result.mp4", str(media)])
        if (
            media.stat().st_size != media_record["bytes"]
            or file_sha256(media) != media_record["sha256"]
        ):
            raise RuntimeError("diagnostic_media_custody")
        diagnostic_receipt_hash = file_sha256(artifact_path)
    # This is a transport receipt, not a miner-controlled timing or score.
    (args.output_dir / "transport.json").write_text(
        json.dumps(
            {
                "schema": "forge-h3-runtime-transport/v1",
                "requestHash": sha256(request),
                "elapsedMs": int((time.monotonic() - started) * 1000),
                "remoteReportSha256": file_sha256(args.output_dir / "remote-report.json"),
                "diagnosticArtifactSha256": diagnostic_receipt_hash,
            }
        )
    )
    with (args.output_dir / "report.json").open("xb") as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(canonical(report))


def validate_diagnostic_artifact(artifact: dict, report: dict) -> dict:
    """Bind private research bytes to an inconclusive trusted-runtime receipt."""
    if (
        report.get("verdict") != "inconclusive"
        or report.get("diagnosticVerdict") != "verified"
        or "result" in report or "finalLatent" in report
        or set(artifact) != {"schema", "purpose", "reportHash", "media"}
        or artifact["schema"] != "forge/h3-qualification-artifact/v1"
        or artifact["purpose"] != "qualification_only"
        or artifact["reportHash"] != sha256(report)
    ):
        raise RuntimeError("diagnostic_artifact_binding")
    media = artifact["media"]
    if (
        not isinstance(media, dict)
        or set(media) != {"name", "bytes", "sha256"}
        or media["name"] != "result.mp4"
        or type(media["bytes"]) is not int
        or not 0 < media["bytes"] <= 512 * 1024**2
        or not isinstance(media["sha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", media["sha256"])
    ):
        raise RuntimeError("diagnostic_artifact_media")
    return media


if __name__ == "__main__":
    main()
