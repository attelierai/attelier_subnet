#!/usr/bin/env python3
"""Replay committed model calls with a selected local checkpoint on private Comfy."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from h3_trace_campaign import digest_file, request, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--unet", default="minimax_h3_fl2va_pruned_nvfp4.safetensors")
    parser.add_argument("--tag", default="canonical")
    parser.add_argument("--origin", default="http://127.0.0.1:8188")
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--seconds-per-call", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.seconds_per_call <= 120:
        parser.error("Bounded per-call timeout must be between 1 and 120 seconds")
    if not args.origin.startswith("http://127.0.0.1:") or Path(args.tag).name != args.tag:
        parser.error("Private origin and plain tag required")
    if args.tag == "canonical" and args.unet != "minimax_h3_fl2va_pruned_nvfp4.safetensors":
        parser.error("Canonical replay must use the pinned baseline checkpoint")
    pending = []
    for run_id in args.runs:
        if Path(run_id).name != run_id:
            parser.error("Plain run identifier required")
        capture_dir = args.root / "captures" / run_id
        manifest = json.loads((capture_dir / "manifest.json").read_text())
        calls = manifest.get("forward", [])
        if not calls or manifest.get("output") is None:
            raise RuntimeError("Complete capture with real model calls required")
        for call in calls:
            if time.time() >= args.deadline_unix - 180:
                raise RuntimeError("Insufficient lease for replay")
            bundle = (capture_dir / call["tensorPath"]).resolve()
            bundle.relative_to(capture_dir.resolve())
            if digest_file(bundle) != call["sha256"]:
                raise RuntimeError("Captured model-call custody mismatch")
            directory = args.root / "replays" / run_id / f"{args.tag}-{call['call']:03}"
            result = directory / "result.json"
            if result.exists():
                prior = json.loads(result.read_text())
                if prior["bundleSha256"] != call["sha256"]:
                    raise RuntimeError("Existing replay belongs to different evidence")
                print(json.dumps({"runId": run_id, "call": call["call"],
                                  "status": "already_measured"}), flush=True)
                continue
            if (directory / "intent.json").exists():
                raise RuntimeError("Unresolved replay intent; reconcile before resubmission")
            pending.append({"runId": run_id, "call": call["call"],
                            "sha256": call["sha256"], "directory": directory,
                            "bundle": bundle, "result": result})
    for offset in range(0, len(pending), 64):
        selected = pending[offset:offset + 64]
        timeout = 120 + len(selected) * args.seconds_per_call
        if time.time() + timeout >= args.deadline_unix:
            raise RuntimeError("Insufficient lease for bounded replay batch")
        batch_id = uuid.uuid4().hex
        requests = [{"bundle_path": str(x["bundle"]), "result_path": str(x["result"]),
                     "candidate_label": args.tag} for x in selected]
        graph = {
            "1": {"class_type": "UNETLoader", "inputs": {
                "unet_name": args.unet, "weight_dtype": "default"}},
            "2": {"class_type": "H3TraceForwardReplayBatch", "inputs": {
                "model": ["1", 0], "requests_json": json.dumps(requests)}},
        }
        directory = args.root / "replays" / "batches" / batch_id
        write_new(directory / "intent.json", {"graph": graph})
        for item in selected:
            write_new(item["directory"] / "intent.json", {
                "batchId": batch_id, "bundleSha256": item["sha256"], "checkpoint": args.unet})
        started = time.monotonic()
        submitted = request(args.origin, "/prompt", {"prompt": graph,
                            "client_id": "trace-replay-" + batch_id})
        write_new(directory / "submission.json", submitted)
        history = None
        while time.monotonic() - started < timeout:
            history = request(args.origin, "/history/" + submitted["prompt_id"]).get(
                submitted["prompt_id"])
            if history:
                break
            time.sleep(2)
        write_new(directory / "history.json", history)
        if not history or history.get("status", {}).get("status_str") != "success":
            raise RuntimeError("Replay infrastructure failed; inspect private history")
        for item in selected:
            receipt = json.loads(item["result"].read_text())
            if receipt["bundleSha256"] != item["sha256"]:
                raise RuntimeError("Replay custody mismatch")
            print(json.dumps({"runId": item["runId"], "call": item["call"],
                              **receipt["outcome"]}), flush=True)
        print(json.dumps({"batchId": batch_id, "calls": len(selected),
                          "seconds": round(time.monotonic() - started, 3)}), flush=True)


if __name__ == "__main__":
    main()
