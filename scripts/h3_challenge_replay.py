#!/usr/bin/env python3
"""Private bounded replay of checksum-bound postcommit challenge bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

from h3_trace_campaign import request, write_new

p = argparse.ArgumentParser()
p.add_argument("--receipts", type=Path, nargs="+", required=True)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--deadline-unix", type=float, required=True)
p.add_argument("--seconds-per-call", type=float, default=20)
p.add_argument("--tag", default="canonicalchallenge")
p.add_argument("--origin", default="http://127.0.0.1:8188")
a = p.parse_args()
if not 1 <= a.seconds_per_call <= 120:
    p.error("Bounded per-call timeout must be between 1 and 120 seconds")
if not a.origin.startswith("http://127.0.0.1:") or Path(a.tag).name != a.tag:
    p.error("private origin/plain tag required")
items = []


def checked_outcome(result: dict, *, bundle_sha256: str, tag: str) -> dict:
    """Validate the node-written result before treating it as a completed replay."""
    if result.get("bundleSha256") != bundle_sha256 or result.get("candidateLabel") != tag:
        raise RuntimeError("result_custody_mismatch")
    outcome = result.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("status") != "MEASURED":
        raise RuntimeError("result_not_measured")
    for name in ("maxAbs", "rms", "relativeRms"):
        value = outcome.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise RuntimeError(f"result_{name}_not_finite")
    for name in ("expectedSha256", "actualSha256"):
        value = outcome.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"result_{name}_missing")
    return outcome


for rp in a.receipts:
    r = json.loads(rp.read_text())
    phase = r.get("selection")
    if phase not in {"postcommit_system_random", "manual_reconstruction_control"}:
        raise ValueError("selection_required")
    bundles = r.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("bundles_required")
    for b in bundles:
        if (
            not isinstance(b, dict)
            or not isinstance(b.get("path"), str)
            or not isinstance(b.get("step"), int)
        ):
            raise ValueError("bundle_descriptor_invalid")
        if not isinstance(b.get("sha256"), str) or len(b["sha256"]) != 64:
            raise ValueError("bundle_sha256_invalid")
        q = (rp.parent / b["path"]).resolve()
        q.relative_to(rp.parent.resolve())
        with q.open("rb") as f:
            h = hashlib.file_digest(f, "sha256").hexdigest()
        if h != b["sha256"]:
            raise ValueError("bundle_custody")
        out = rp.parent / "replay" / str(b["step"]) / "result.json"
        intent = out.parent / "intent.json"
        if intent.exists() and not out.exists():
            raise RuntimeError("unresolved_intent_no_retry")
        if out.exists():
            prior = json.loads(out.read_text())
            checked_outcome(prior, bundle_sha256=h, tag=a.tag)
            continue
        items.append(
            {
                "bundle_path": str(q),
                "result_path": str(out),
                "candidate_label": a.tag,
                "phase": phase,
                "sha256": h,
                "intent": intent,
            }
        )
if not items:
    print(json.dumps({"status": "MEASURED", "calls": 0, "batches": 0, "rerun": "verified_empty"}))
    raise SystemExit(0)
if time.time() + 120 + a.seconds_per_call * min(len(items), 64) >= a.deadline_unix:
    raise RuntimeError("deadline_insufficient")
summaries = []
for offset in range(0, len(items), 64):
    selected = items[offset : offset + 64]
    if time.time() + 120 + a.seconds_per_call * len(selected) >= a.deadline_unix:
        raise RuntimeError("deadline_insufficient")
    for x in selected:
        write_new(x["intent"], {"bundleSha256": x["sha256"], "selection": x["phase"], "tag": a.tag})
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_nvfp4.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "H3TraceForwardReplayBatch",
            "inputs": {
                "model": ["1", 0],
                "requests_json": json.dumps(
                    [
                        {
                            k: v
                            for k, v in x.items()
                            if k in {"bundle_path", "result_path", "candidate_label"}
                        }
                        for x in selected
                    ]
                ),
            },
        },
    }
    batch = a.root / "challenge-replays" / uuid.uuid4().hex
    write_new(batch / "intent.json", {"graph": graph, "count": len(selected)})
    s = request(a.origin, "/prompt", {"prompt": graph, "client_id": "h3-challenge-" + batch.name})
    write_new(batch / "submission.json", s)
    start = time.monotonic()
    history = None
    while time.monotonic() - start < 120 + a.seconds_per_call * len(selected):
        history = request(a.origin, "/history/" + s["prompt_id"]).get(s["prompt_id"])
        if history:
            break
        time.sleep(2)
    write_new(batch / "history.json", history)
    if not history or history.get("status", {}).get("status_str") != "success":
        raise RuntimeError("replay_inconclusive")
    metrics = []
    for x in selected:
        prior = json.loads(Path(x["result_path"]).read_text())
        metrics.append(checked_outcome(prior, bundle_sha256=x["sha256"], tag=a.tag)["relativeRms"])
    summaries.append(
        {
            "batch": batch.name,
            "calls": len(selected),
            "seconds": time.monotonic() - start,
            "maxRelativeRms": max(metrics),
        }
    )
print(json.dumps({"status": "MEASURED", "calls": len(items), "batches": summaries}))
