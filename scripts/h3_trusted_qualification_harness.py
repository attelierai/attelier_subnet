#!/usr/bin/env python3
"""Private, GPU-volume-only qualification harness for forge-h3-trace/v1.

Run ``smoke`` first.  It creates an ephemeral Ed25519 key under --root (never
prints it), makes one honest full-20 capture, signs its receipt and invokes the
trusted verifier in diagnostic mode. ``attacks`` is deliberately a separate
command and refuses to run without that accepted honest smoke summary.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

from forge.inference.trace_protocol import (
    TraceProtocolError,
    canonical,
    effective_seed,
    file_sha256,
    load_profile,
    sha256,
    verify_bundle,
)
from forge.inference.trace_runtime import (
    NativeH3Profile,
    TrustedT2VInput,
    build_trusted_baseline_graph,
    write_graph_new,
)

HERE = Path(__file__).resolve().parent


def write_new(path, value):
    if path.exists():
        raise RuntimeError("refusing_overwrite")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf8") as out:
        os.chmod(path, 0o600)
        json.dump(value, out, sort_keys=True, separators=(",", ":"))
        out.write("\n")


def request_json(origin, path, payload=None):
    import urllib.request

    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(
        origin.rstrip("/") + path,
        data=data,
        method="GET" if data is None else "POST",
        headers={} if data is None else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


def run_graph(origin, graph, label, timeout):
    reply = request_json(origin, "/prompt", {"prompt": graph, "client_id": label})
    pid = reply.get("prompt_id")
    if not isinstance(pid, str):
        raise RuntimeError("comfy_submit_invalid")
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        row = request_json(origin, "/history/" + pid).get(pid)
        if row:
            if row.get("status", {}).get("status_str") != "success":
                raise RuntimeError("comfy_capture_failed")
            return row
        time.sleep(2)
    raise RuntimeError("comfy_capture_timeout")


def keypair(root):
    key = root / "test-ed25519.pem"
    pub = root / "test-ed25519.pub"
    if not key.exists():
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)],
            check=True,
            capture_output=True,
        )
        os.chmod(key, 0o600)
        der = subprocess.run(
            ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
            check=True,
            capture_output=True,
        ).stdout
        pub.write_text(base64.b64encode(der[-32:]).decode() + "\n", encoding="ascii")
        os.chmod(pub, 0o600)
    return key, pub


def referenced(capture, root):
    rows = []

    def collect(value, depth=0):
        if depth > 24:
            raise RuntimeError("capture_too_deep")
        if isinstance(value, dict):
            if "tensorPath" in value:
                rows.append(value)
            for child in value.values():
                collect(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 4096:
                raise RuntimeError("capture_array_size")
            for child in value:
                collect(child, depth + 1)

    collect(capture)
    names = []
    for row in rows:
        name = row.get("tensorPath")
        if not isinstance(name, str) or name in names:
            continue
        path = (root / name).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise RuntimeError("capture_reference_missing")
        names.append(name)
    return names


def receipt(capture_path, request, profile, private_key, mutate=None):
    root = capture_path.parent.resolve()
    capture = json.loads(capture_path.read_text())
    names = ["capture.json", *referenced(capture, root)]
    chunks = []
    for index, name in enumerate(names):
        path = root / name
        chunks.append(
            {
                "index": index,
                "name": name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    index = {row["name"]: row["index"] for row in chunks}
    final = []
    for item in capture["finalLatent"]:
        final.append({"component": item["component"], "chunkIndex": index[item["tensorPath"]]})
    value = {
        "schema": "forge-h3-trace/v1",
        "jobId": request["jobId"],
        "workerId": request["workerId"],
        "nonce": request["nonce"],
        "requestHash": sha256(request),
        "parametersHash": sha256(request["parameters"]),
        "profile": request["profile"],
        "modelManifestHash": profile["modelManifestHash"],
        "workflowHash": profile["workflowHash"],
        "steps": 20,
        "trace": {"algorithm": "sha256-manifest-v1", "chunks": chunks, "root": sha256(chunks)},
        "finalLatent": final,
    }
    if mutate:
        mutate(value)
    message = canonical(value)
    msg = root / "receipt-message.json"
    sig = root / "receipt.sig"
    msg.write_bytes(message)
    subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            str(private_key),
            "-rawin",
            "-in",
            str(msg),
            "-out",
            str(sig),
        ],
        check=True,
        capture_output=True,
    )
    value["signature"] = base64.b64encode(sig.read_bytes()).decode()
    write_new(root / "receipt.json", value)
    return root / "receipt.json", sum(row["bytes"] for row in chunks)


def diagnostic(a, request_path, receipt_path, evidence_root, public_key, output,
               walltime_seconds=None):
    command = [
        sys.executable,
        str(HERE / "h3_trusted_runtime.py"),
        "--request", str(request_path), "--receipt", str(receipt_path),
        "--hotkey", "qualification-diagnostic-only",
        "--profile-root", str(a.profile_root), "--evidence-root", str(evidence_root),
        "--worker-public-key-file", str(public_key),
        "--runtime-attestation", str(a.runtime_attestation), "--output-dir", str(output),
        "--comfy-origin", a.comfy_origin, "--comfy-output-root", str(a.comfy_output_root),
        "--timeout-seconds", str(a.timeout_seconds), "--allow-unqualified-diagnostic",
        "--export-diagnostic-artifact",
    ]
    done = subprocess.run(
        command, text=True, capture_output=True,
        timeout=min(a.timeout_seconds * 3, walltime_seconds or a.timeout_seconds * 3), check=False
    )
    if not (output / "report.json").is_file():
        return {"verdict": "inconclusive", "reason": "verifier_no_report", "exit": done.returncode}
    report = json.loads((output / "report.json").read_text())
    report["exit"] = done.returncode
    return report


def smoke(a):
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    profile = load_profile(
        a.profile_root.resolve(),
        {"id": a.profile_id, "hash": a.profile_hash},
        allow_unqualified=True,
    )
    pgraph = NativeH3Profile.from_validated(profile)
    key, pub = keypair(root)
    job = "smoke-" + secrets.token_hex(8)
    nonce = secrets.token_hex(32)
    requested = secrets.randbelow(2**32)
    seed = effective_seed(job, nonce, requested, a.profile_hash)
    req = {
        "schema": "forge-h3-request/v1",
        "jobId": job,
        "workerId": "qualification-miner",
        "nonce": nonce,
        "requestedSeed": requested,
        "profile": {"id": a.profile_id, "hash": a.profile_hash},
        "parameters": {
            "modality": "text-to-video",
            "prompt": a.prompt,
            "duration": 5,
            "resolution": "480p",
            "aspectRatio": "16:9",
            "audio": True,
            "seed": seed,
            "profile": "base",
        },
    }
    write_new(root / "request.json", req)
    trusted = TrustedT2VInput.from_validated(
        {
            "jobId": job,
            "nonce": nonce,
            "parametersHash": sha256(req["parameters"]),
            "requestHash": sha256(req),
            "parameters": req["parameters"],
            "effectiveSeed": seed,
        }
    )
    evidence = root / "evidence"
    graph = build_trusted_baseline_graph(
        trusted, pgraph, trace_root=evidence, filename_prefix="qualification/" + job
    )
    write_graph_new(root / "capture-graph.json", graph)
    started = time.monotonic()
    run_graph(a.comfy_origin, graph, "qualification-" + job, a.timeout_seconds)
    capture = evidence / job / "manifest.json"
    sealed = evidence / job / "capture.json"
    os.link(capture, sealed)
    rec, size = receipt(sealed, req, profile, key)
    out = root / "verifier"
    report = diagnostic(a, root / "request.json", rec, evidence / job, pub, out)
    summary = {
        "schema": "forge-h3-qualification-smoke/v1",
        "jobId": job,
        "elapsedMs": int((time.monotonic() - started) * 1000),
        "captureBytes": size,
        "verifierExit": report["exit"],
        "diagnosticVerdict": report.get("diagnosticVerdict"),
        "verdict": report.get("verdict"),
        "reason": report.get("reason"),
    }
    write_new(root / "smoke-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def attacks(a):
    root = a.root.resolve()
    summary = json.loads((root / "smoke-summary.json").read_text())
    if summary.get("diagnosticVerdict") != "verified" or summary.get("verdict") != "inconclusive":
        raise RuntimeError("honest_smoke_not_accepted")
    request_path = root / "request.json"
    request = json.loads(request_path.read_text())
    if request.get("jobId") != summary.get("jobId") or request.get("profile") != {
        "id": a.profile_id,
        "hash": a.profile_hash,
    }:
        raise RuntimeError("smoke_request_profile_mismatch")
    profile = load_profile(
        a.profile_root.resolve(), request["profile"], allow_unqualified=True
    )
    pgraph = NativeH3Profile.from_validated(profile)
    key, pub = keypair(root)
    public_key = pub.read_text(encoding="ascii").strip()
    honest = root / "evidence" / request["jobId"]
    if not (honest / "capture.json").is_file():
        raise RuntimeError("honest_capture_missing")
    attacks_root = root / "attacks"
    attacks_root.mkdir(mode=0o700)
    if any(attacks_root.iterdir()):
        raise RuntimeError("attacks_directory_not_fresh")
    if Path(a.taomate_lora).name != a.taomate_lora:
        raise RuntimeError("taomate_lora_name_unsafe")

    trusted = TrustedT2VInput.from_validated(
        {
            "jobId": request["jobId"],
            "nonce": request["nonce"],
            "parametersHash": sha256(request["parameters"]),
            "requestHash": sha256(request),
            "parameters": request["parameters"],
            "effectiveSeed": request["parameters"]["seed"],
        }
    )

    def classify(report):
        verdict = report.get("diagnosticVerdict", report.get("verdict"))
        if verdict == "verified":
            return "accepted"
        if verdict == "invalid":
            return "rejected"
        return "inconclusive"

    def execute(label, graph=None, derive=None, mutate=None, source=honest):
        """Run one isolated real capture or a cheap signed derivation, then diagnose it."""
        variant = attacks_root / label
        evidence = variant / "evidence"
        variant.mkdir(mode=0o700)
        try:
            if graph is not None:
                write_graph_new(variant / "capture-graph.json", graph)
                run_graph(a.comfy_origin, graph, "qualification-attack-" + label, a.timeout_seconds)
                captured = evidence / request["jobId"] / "manifest.json"
                sealed = evidence / request["jobId"] / "capture.json"
                os.link(captured, sealed)
            else:
                destination = evidence / request["jobId"]
                shutil.copytree(source, destination, copy_function=os.link)
                # Derived attacks reuse immutable tensor files but must never
                # reuse a smoke/previous variant's receipt or OpenSSL inputs.
                for name in ("receipt.json", "receipt-message.json", "receipt.sig"):
                    stale = destination / name
                    if stale.exists():
                        if stale.is_symlink() or not stale.is_file():
                            raise RuntimeError("derived_signing_artifact_unsafe")
                        stale.unlink()
                sealed = destination / "capture.json"
                if derive:
                    document = json.loads(sealed.read_text())
                    derive(document)
                    sealed.unlink()
                    write_new(sealed, document)
            rec, size = receipt(sealed, request, profile, key, mutate)
            # Exact local custody/signature preflight happens before the costly
            # independent trusted-runtime baseline and render.
            verify_bundle(request, json.loads(rec.read_text()), sealed.parent, profile, public_key)
            report = diagnostic(a, request_path, rec, sealed.parent, pub, variant / "verifier")
            outcome = classify(report)
            row = {
                "label": label,
                "outcome": outcome,
                "diagnosticVerdict": report.get("diagnosticVerdict"),
                "verdict": report.get("verdict"),
                "reason": report.get("reason"),
                "captureBytes": size,
            }
        except TraceProtocolError as error:
            row = {
                "label": label,
                "outcome": "rejected" if error.category == "invalid" else "inconclusive",
                "reason": error.reason,
            }
        except Exception as error:
            reason = str(error)
            row = {
                "label": label,
                "outcome": "inconclusive",
                "reason": reason if reason.isidentifier() else "attack_execution_failed",
            }
        write_new(variant / "result.json", row)
        return row

    def capture_graph(label, steps, lora=None):
        graph = build_trusted_baseline_graph(
            trusted,
            pgraph,
            trace_root=attacks_root / label / "evidence",
            filename_prefix="qualification/attacks/" + label,
        )
        graph["9"]["inputs"]["steps"] = steps
        if lora:
            graph["18"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1},
            }
            graph["7"]["inputs"]["model"] = ["18", 0]
            graph["9"]["inputs"]["model"] = ["18", 0]
        return graph

    results = [
        execute("steps-10", capture_graph("steps-10", 10)),
        execute("steps-3", capture_graph("steps-3", 3)),
        execute("taomate-lora", capture_graph("taomate-lora", 20, a.taomate_lora)),
    ]

    # These are signed receipts over isolated hard-linked honest evidence.  They
    # do no GPU work, but exercise receipt/capture binding independently.
    results.append(
        execute(
            "altered-final-latent",
            mutate=lambda value: value["finalLatent"].__setitem__(
                0, {"component": "video", "chunkIndex": value["finalLatent"][1]["chunkIndex"]}
            ),
        )
    )

    def pad_metadata(document):
        trace = document.get("trace")
        forward = document.get("forward")
        if not isinstance(trace, list) or not trace or not isinstance(forward, list) or not forward:
            raise RuntimeError("shortcut_capture_missing_trace")
        for record in trace:
            record["totalSteps"] = 20
        last = trace[-1]
        while len(trace) < 20:
            copied = copy.deepcopy(last)
            copied["step"] = len(trace)
            copied["totalSteps"] = 20
            trace.append(copied)
        last_forward = forward[-1]
        while len(forward) < 20:
            copied = copy.deepcopy(last_forward)
            copied["call"] = len(forward)
            forward.append(copied)
        document["declaredSteps"] = document["callbackSteps"] = 20

    results.append(
        execute(
            "padded-steps-10",
            derive=pad_metadata,
            source=attacks_root / "steps-10" / "evidence" / request["jobId"],
        )
    )
    report = {
        "schema": "forge-h3-qualification-attacks/v1",
        "honestSmoke": summary["jobId"],
        "profile": request["profile"],
        "results": results,
        "accepted": [row["label"] for row in results if row["outcome"] == "accepted"],
    }
    write_new(attacks_root / "summary.json", report)
    print(json.dumps(report, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "attacks"):
        q = sub.add_parser(name)
        q.add_argument("--root", type=Path, required=True)
    s = sub.choices["smoke"]
    s.add_argument("--profile-root", type=Path, required=True)
    s.add_argument("--profile-id", required=True)
    s.add_argument("--profile-hash", required=True)
    s.add_argument("--runtime-attestation", type=Path, required=True)
    s.add_argument("--comfy-origin", required=True)
    s.add_argument("--comfy-output-root", type=Path, required=True)
    s.add_argument("--prompt", default="A chrome kinetic sculpture turns slowly in a quiet studio.")
    s.add_argument("--timeout-seconds", type=int, default=1800)
    x = sub.choices["attacks"]
    x.add_argument("--profile-root", type=Path, required=True)
    x.add_argument("--profile-id", required=True)
    x.add_argument("--profile-hash", required=True)
    x.add_argument("--runtime-attestation", type=Path, required=True)
    x.add_argument("--comfy-origin", required=True)
    x.add_argument("--comfy-output-root", type=Path, required=True)
    x.add_argument("--taomate-lora", required=True)
    x.add_argument("--timeout-seconds", type=int, default=1800)
    a = p.parse_args()
    smoke(a) if a.cmd == "smoke" else attacks(a)


if __name__ == "__main__":
    main()
