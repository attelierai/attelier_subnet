#!/usr/bin/env python3
"""Remote-only pilot metrics and blind-sheet preparation, with no API calls.

Metric output is sealed from the observability reviewer by the operator; this
script does not assert that filesystem permissions prove scientific blindness.
An interrupted output is terminal and is never automatically executed again.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

from forge.evaluation.pilot_media_index import validate_pilot_media_index
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.evaluation.raster_integrity import check_frames
from forge.evaluation.vbench_runner import run_vbench_isolated
from forge.evaluation.vision_media import (
    load_video,
    make_sheets,
    select_detail_crops,
    select_detail_frames,
    select_transition_frames,
)
from forge.protocol.hashing import sha256_file


def run(args: argparse.Namespace) -> dict:
    root = args.media_root.resolve()
    output = args.out.resolve()
    for path in (root, output):
        path.relative_to(Path("/workspace/forge"))
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time():
        raise ValueError("a future absolute working deadline is required")
    pilot = json.loads(args.pilot.read_text())
    work = pilot_worklist(pilot)
    comparisons = list(work["comparisons"])
    if getattr(args, "repeat_slots", None):
        selected = json.loads(args.repeat_slots.read_text())
        by_slot = {pair["slot_id"]: pair for pair in comparisons}
        if not isinstance(selected, list) or len(set(selected)) != len(selected):
            raise ValueError("repeat slots must be a unique list")
        for slot in selected:
            original = by_slot[slot]
            comparisons.append({**original, "slot_id": slot + ":fresh-views"})
    index = json.loads(args.media_index.read_text())
    videos = validate_pilot_media_index(work, index)["videos"]
    paths = {}
    for job_id, video in videos.items():
        path = root / video["path"]
        path.resolve().relative_to(root)
        if (
            Path(video["path"]).is_absolute()
            or path.is_symlink()
            or sha256_file(path) != video["video_sha256"]
        ):
            raise ValueError("media bytes, prompt or seed differ from the frozen job")
        paths[job_id] = path
    config = json.loads(args.vbench_config.read_text())
    if set(config) != {
        "interpreter",
        "checkout",
        "cache",
        "weight_sha256",
        "dino_revision",
        "runner_image_digest",
    }:
        raise ValueError("VBench configuration fields must be exact")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    state = {
        "schema": "forge/evaluator-pilot-measurement/v1",
        "worklist_id": work["worklist_id"],
        "media_index_sha256": sha256_file(args.media_index),
        "vbench_config_sha256": sha256_file(args.vbench_config),
        "deadline_unix": args.deadline_unix,
        "status": "RUNNING",
        "metric_receipts": [],
        "payload_pairs": 0,
        "api_calls": 0,
    }

    def save():
        temporary = output / "state.new"
        with temporary.open("w") as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output / "state.json")

    def remaining():
        seconds = args.deadline_unix - time.time()
        if seconds <= 0:
            raise TimeoutError("pilot working deadline reached")
        return seconds

    save()
    try:
        for job in work["metric_jobs"]:
            identity = job["video_job_id"]
            video = videos[identity]
            state["active_job_id"] = identity
            save()
            run_vbench_isolated(
                video=paths[identity],
                prompt=video["prompt_text"],
                output=output / "metrics" / identity,
                interpreter=Path(config["interpreter"]),
                checkout=Path(config["checkout"]),
                cache=Path(config["cache"]),
                weight_sha256=config["weight_sha256"],
                dino_revision=config["dino_revision"],
                runner_image_digest=config["runner_image_digest"],
                plan_id=work["worklist_id"],
                prompt_id=identity,
                seed=video["seed"],
                manifest_hash=index["base_manifest_hash"],
                timeout_seconds=min(1800, remaining()),
            )
            receipt = output / "metrics" / identity / "receipt.json"
            state["metric_receipts"].append({"job_id": identity, "sha256": sha256_file(receipt)})
            save()
        geometry = {key: work["recipe"][key] for key in ("width", "height", "frames", "fps")}
        partial = output / "luna-payloads.jsonl.gz.partial"
        with (
            partial.open("xb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
            (output / "selection-ledger.jsonl").open("x") as selections,
        ):
            for pair in comparisons:
                remaining()
                state["active_job_id"] = pair["slot_id"]
                save()
                reference, candidate = [
                    load_video(paths[pair[key]], geometry)
                    for key in ("reference_job_id", "candidate_job_id")
                ]
                for side, decoded in (("reference", reference), ("candidate", candidate)):
                    if decoded.video_sha256 != videos[pair[f"{side}_job_id"]]["video_sha256"]:
                        raise ValueError("video changed after metric measurement")
                details = select_detail_frames(len(reference.frames))
                selection = {
                    "detail_indices": details,
                    "detail_crops": select_detail_crops(reference.frames, details),
                    "transition_indices": select_transition_frames(len(reference.frames)),
                }
                payload = {
                    "slot_id": pair["slot_id"],
                    "worklist_id": work["worklist_id"],
                    "prompt": videos[pair["reference_job_id"]]["prompt_text"],
                    "selection": selection,
                    "geometry": geometry,
                }
                for side, decoded in (("reference", reference), ("candidate", candidate)):
                    payload[f"{side}_sha256"] = decoded.video_sha256
                    payload[f"{side}_sheets"] = [
                        base64.b64encode(sheet).decode("ascii")
                        for sheet in make_sheets(decoded.frames, geometry["fps"], **selection)
                    ]
                    payload[f"{side}_raster"] = check_frames(decoded.frames, decoded.video_sha256)
                    payload[f"{side}_diagnostics"] = decoded.diagnostics
                encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
                compact = {
                    key: value for key, value in payload.items() if not key.endswith("_sheets")
                }
                for side in ("reference", "candidate"):
                    compact[f"{side}_sheet_sha256"] = [
                        hashlib.sha256(base64.b64decode(sheet)).hexdigest()
                        for sheet in payload[f"{side}_sheets"]
                    ]
                compact["payload_row_sha256"] = hashlib.sha256(encoded).hexdigest()
                selections.write(json.dumps(compact, sort_keys=True) + "\n")
                selections.flush()
                os.fsync(selections.fileno())
                compressed.write(encoded)
                state["payload_pairs"] += 1
                save()
        target = output / "luna-payloads.jsonl.gz"
        partial.rename(target)
        state.update(
            status="COMPLETE",
            payload_sha256=sha256_file(target),
            payload_bytes=target.stat().st_size,
        )
        state.pop("active_job_id", None)
        save()
        return state
    except BaseException as error:
        state.update(status="INCOMPLETE", error_type=type(error).__name__)
        save()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "media-index", "media-root", "vbench-config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--repeat-slots", type=Path)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "worklist_id", "api_calls", "payload_pairs")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
