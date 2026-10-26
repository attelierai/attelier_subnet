"""Development-only comparison of archived fusion and an open reward model.

This format deliberately has no path to calibration, certificates or weights.
Scores are descriptive; neither a sign nor a reward-model sum is a quality gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from forge.evaluation.pilot_media_index import validate_pilot_media_index
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.protocol.hashing import hash_canonical, sha256_file

SCHEMA = "forge/open-evaluator-study/v1"
AXES = ("VQ", "MQ", "TA")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A deadline/kill cannot leave a half-written receipt masquerading as a result.
    # Hard-link creation is atomic and fails if another writer already owns the name.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".study-",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal(body: dict) -> dict:
    return {**body, "plan_id": hash_canonical(SCHEMA, body)}


def validate_plan(plan: dict) -> None:
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if (
        plan.get("schema") != SCHEMA
        or plan.get("qualification") != "prohibited"
        or plan.get("split") != "development"
        or seal(body) != plan
    ):
        raise ValueError("changed or non-development study plan")
    samples = {row["sample_id"]: row for row in plan["samples"]}
    if not samples or len(samples) != len(plan["samples"]):
        raise ValueError("duplicate or empty study samples")
    for sample in samples.values():
        identity = hash_canonical(SCHEMA, sample["video_sha256"], sample["prompt_text"])
        path = Path(sample["path"])
        if sample["sample_id"] != identity or path.is_absolute() or ".." in path.parts:
            raise ValueError("sample identity or relative media path is invalid")
    requests = plan["requests"]
    if not requests or len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("duplicate or empty study requests")
    if any(r["sample_id"] not in samples for r in requests):
        raise ValueError("request references an unknown sample")
    if {r["sample_id"] for r in requests if r["repeat"] == 0} != set(samples):
        raise ValueError("every sample needs a base measurement")
    for r in requests:
        if r["repeat"] not in (0, 1) or r["request_id"] != f"{r['sample_id']}:{r['repeat']}":
            raise ValueError("invalid repeat identity")
    for pair in plan["comparisons"]:
        if any(pair[k] not in samples for k in ("reference_sample", "candidate_sample")):
            raise ValueError("comparison references an unknown sample")


def freeze_pilot(
    pilot_path: Path, index_path: Path, legacy_path: Path, spec_path: Path, *, repeats: int
) -> dict:
    """Reconstruct the original 96 pairs, including wrong-video prompt identity."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("at least one independent inference repeat is required")
    pilot, index, legacy, spec = map(read, (pilot_path, index_path, legacy_path, spec_path))
    work = pilot_worklist(pilot)
    videos = validate_pilot_media_index(work, index)["videos"]
    if legacy.get("status") != "COMPLETE" or legacy.get("qualification") != "prohibited":
        raise ValueError("legacy comparison must be a completed development receipt")
    rows = [row for row in legacy["rows"] if row["variant"] == "base"]
    by_slot = {row["slot_id"]: row for row in rows}
    if len(by_slot) != len(rows) or set(by_slot) != {r["slot_id"] for r in work["comparisons"]}:
        raise ValueError("legacy coverage differs from frozen comparisons")
    slots = {r["slot_id"]: r for r in pilot["slots"]}
    samples: dict[str, dict] = {}
    comparisons = []
    for pair in work["comparisons"]:
        slot = slots[pair["slot_id"]]
        prior = by_slot[pair["slot_id"]]
        if prior["kind"] != pair["kind"] or prior["luna"]["status"] != "OK":
            raise ValueError("legacy result is invalid or has a different control")
        orders = prior["orders"]
        if len(orders) != 2 or {o["reference_first"] for o in orders} != {True, False}:
            raise ValueError("legacy comparison requires both orientations")
        ids = []
        for key in ("reference_job_id", "candidate_job_id"):
            media = videos[pair[key]]
            # The candidate's original generation prompt is WRONG for wrong_video.
            identity = hash_canonical(SCHEMA, media["video_sha256"], slot["prompt_text"])
            samples.setdefault(identity, {
                "sample_id": identity,
                "video_sha256": media["video_sha256"],
                "path": media["path"],
                "prompt_text": slot["prompt_text"],
                "source_family": slot["source_family"],
            })
            ids.append(identity)
        comparisons.append({
            "slot_id": pair["slot_id"], "kind": pair["kind"],
            "source_family": slot["source_family"],
            "reference_sample": ids[0], "candidate_sample": ids[1],
            "legacy_fusion_orders": orders,
            "legacy_luna_axes": prior["luna"]["axes"],
            "human_quality_label": "UNREVIEWED",
        })
    if repeats > len(samples):
        raise ValueError("repeat count exceeds unique prompt/video samples")
    ordered = sorted(samples)
    legacy_repeats = []
    for row in legacy["rows"]:
        if row["variant"] == "base":
            continue
        original = by_slot[row["original_slot_id"]]
        legacy_repeats.append({
            "slot_id": row["original_slot_id"], "variant": row["variant"],
            "luna_absolute_change": {
                axis: abs(value - original["luna"]["axes"][axis])
                for axis, value in row["luna"]["axes"].items()
            },
            "status": row["luna"]["status"],
        })
    # Selection is frozen before new scores; repeats are real calls, not cache hits.
    requests = [
        {"request_id": f"{sample}:{repeat}", "sample_id": sample, "repeat": repeat}
        for repeat, selected in ((0, ordered), (1, ordered[:repeats]))
        for sample in selected
    ]
    plan = seal({
        "schema": SCHEMA, "split": "development", "qualification": "prohibited",
        "purpose": "historical evaluator comparison; no independent validation",
        "source_receipts": {
            name: sha256_file(path) for name, path in (
                ("pilot", pilot_path), ("media_index", index_path), ("legacy", legacy_path)
            )
        },
        "spec": spec,
        "geometry": {key: work["recipe"][key] for key in ("width", "height", "frames", "fps")},
        "samples": [samples[key] for key in ordered],
        "requests": requests, "comparisons": comparisons,
        "legacy_repeatability": legacy_repeats,
    })
    validate_plan(plan)
    return plan


def checked_receipts(plan: dict, receipts: list[dict]) -> dict[str, dict]:
    validate_plan(plan)
    requests = {r["request_id"]: r for r in plan["requests"]}
    samples = {s["sample_id"]: s for s in plan["samples"]}
    result = {}
    evaluator_ids = set()
    for row in receipts:
        key = row.get("request_id")
        if key not in requests or key in result or row.get("plan_id") != plan["plan_id"]:
            raise ValueError("duplicate, foreign or stale measurement")
        if row.get("status") != "OK" or row.get("qualification") != "prohibited":
            raise ValueError("measurement is incomplete or claims qualification")
        if row.get("video_sha256") != samples[requests[key]["sample_id"]]["video_sha256"]:
            raise ValueError("measurement binds different media bytes")
        if set(row["scores"]) != set(AXES):
            raise ValueError("all three open-model heads are required")
        numbers = [*row["scores"].values(), row["elapsed_seconds"], row["peak_cuda_bytes"]]
        if any(type(n) not in (int, float) or not math.isfinite(n) for n in numbers):
            raise ValueError("measurement contains non-finite or non-numeric values")
        if row["elapsed_seconds"] < 0 or row["peak_cuda_bytes"] < 0:
            raise ValueError("negative runtime measurement")
        if not isinstance(row.get("runtime"), dict) or row.get("evaluator_id") != hash_canonical(
            "forge/videoreward-study-runtime/v1", plan["spec"], row["runtime"]
        ):
            raise ValueError("measurement must bind its evaluator implementation")
        evaluator_ids.add(row["evaluator_id"])
        result[key] = row
    if len(evaluator_ids) > 1:
        raise ValueError("cannot mix evaluator implementations or environments")
    return result


def report(plan: dict, receipts: list[dict]) -> dict:
    measured = checked_receipts(plan, receipts)
    pairs, repeatability = [], []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in plan["comparisons"]:
        ref = measured.get(pair["reference_sample"] + ":0")
        cand = measured.get(pair["candidate_sample"] + ":0")
        delta = None if ref is None or cand is None else {
            axis: cand["scores"][axis] - ref["scores"][axis] for axis in AXES
        }
        entry = {**pair, "open_delta": delta, "status": "MISSING" if delta is None else "OK"}
        entry["temporal_diagnostics"] = None if ref is None or cand is None else {
            "reference": ref.get("diagnostics"), "candidate": cand.get("diagnostics"),
        }
        pairs.append(entry)
        grouped[pair["kind"]].append(entry)
    for request in plan["requests"]:
        if request["repeat"] != 1:
            continue
        first = measured.get(request["sample_id"] + ":0")
        repeated = measured.get(request["request_id"])
        repeatability.append({
            "sample_id": request["sample_id"],
            "absolute_change": None if first is None or repeated is None else {
                a: abs(first["scores"][a] - repeated["scores"][a]) for a in AXES
            },
        })
    return {
        "schema": "forge/open-evaluator-study-report/v1", "plan_id": plan["plan_id"],
        "status": "COMPLETE" if len(measured) == len(plan["requests"]) else "INCOMPLETE",
        "qualification": "prohibited", "quality_decision": "NOT_EMITTED",
        "expected_measurements": len(plan["requests"]), "measurements": len(measured),
        "unique_source_families": len({r["source_family"] for r in plan["comparisons"]}),
        "exact_media_reuse": {
            "unique_video_bytes": len({s["video_sha256"] for s in plan["samples"]}),
            "prompt_video_samples": len(plan["samples"]),
            "interpretation": "Counts include intentional shared references and wrong-video "
            "controls; not a model diversity measurement.",
        },
        "per_kind": {
            kind: {
                "pairs": len(rows), "measured": sum(r["open_delta"] is not None for r in rows),
                "mean_open_delta": {
                    axis: sum(r["open_delta"][axis] for r in rows if r["open_delta"] is not None)
                    / sum(r["open_delta"] is not None for r in rows)
                    for axis in AXES
                } if any(r["open_delta"] is not None for r in rows) else None,
            } for kind, rows in sorted(grouped.items())
        },
        "repeatability": repeatability, "comparisons": pairs,
        "legacy_repeatability": plan.get("legacy_repeatability", []),
        "runtime": {
            "measured_seconds": sum(r["elapsed_seconds"] for r in measured.values()),
            "peak_cuda_bytes": max((r["peak_cuda_bytes"] for r in measured.values()), default=None),
            "excludes": "all-frame decode/diagnostics, asset download, model load, warmup, "
            "idle, storage and transfers; not end-to-end capacity or lease cost",
        },
        "limits": [
            "Known development families; observations are not independent trials.",
            "VQ/MQ/TA and legacy fusion have different units; do not subtract their scores.",
            "Raw deltas are not calibrated detections, false positives or acceptable quality.",
            "Same-model/new-seed equality requires blinded quality review.",
            "Temporal diagnostics and exact-media duplicates are signals, not semantic diversity.",
            "Missing rows remain missing; completion never qualifies the evaluator or a miner.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-pilot")
    for field in ("pilot", "media-index", "legacy", "spec", "out"):
        freeze.add_argument("--" + field, type=Path, required=True)
    freeze.add_argument("--repeats", type=int, required=True)
    summarize = commands.add_parser("report")
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--receipts", type=Path)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-pilot":
            output = freeze_pilot(args.pilot, args.media_index, args.legacy, args.spec,
                                  repeats=args.repeats)
        else:
            receipts = [] if args.receipts is None else [
                read(path) for path in sorted(args.receipts.glob("*.json"))
            ]
            output = report(read(args.plan), receipts)
        write_new(args.out, output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"study rejected: {error}\n")
    print(json.dumps({k: output[k] for k in ("plan_id", "qualification")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
