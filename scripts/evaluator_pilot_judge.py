#!/usr/bin/env python3
"""Judge archived pilot sheets only after verified backup, closure and media review.

Compressed sheets arrive on stdin and stay in memory. All inputs are validated
before the first API call. An incomplete run is terminal; it is not auto-retried.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import sys
from pathlib import Path

from forge.evaluation.fusion import evaluator_implementation_hash, fused_axis_orders
from forge.evaluation.mvp_calibration import fit_policy_from_development
from forge.evaluation.pilot_media_index import validate_pilot_media_index
from forge.evaluation.pilot_observability import (
    freeze_pilot_observability_protocol,
    validate_pilot_observability_receipt,
)
from forge.evaluation.pilot_payloads import MAX_PAYLOAD_BYTES, MAX_ROW_BYTES, archived_payloads
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.evaluation.raster_integrity import verify_check
from forge.evaluation.vbench import validate_metric_evidence
from forge.evaluation.vision_judge import MODEL, PILOT_RESERVATION_USD, ApiBudget, judge_pair
from forge.protocol.hashing import hash_canonical, sha256_file


def run(args: argparse.Namespace) -> dict:
    def read(path):
        return json.loads(path.read_text())

    work = pilot_worklist(read(args.pilot))
    pilot = read(args.pilot)
    ownership, closed, backup = map(read, (args.ownership, args.closed, args.backup))
    measurement, index, review = map(read, (args.measurement, args.media_index, args.review))
    if (
        ownership["worklist_id"] != work["worklist_id"]
        or closed.get("ownership_id")
        != hash_canonical("forge/evaluator-pilot-ownership/v1", ownership)
        or closed.get("status") != "CLOSED"
        or closed.get("backup_verified") is not True
        or backup.get("schema") != "forge/evaluator-pilot-archive/v1"
        or backup.get("status") != "VERIFIED"
        or backup.get("readback_verified") is not True
        or backup.get("evidence_complete") is not True
        or backup.get("pod_id") != ownership["pod_id"]
        or backup.get("worklist_id") != work["worklist_id"]
    ):
        raise ValueError("verified owned-pod closure and backup are required before Luna")
    if ownership["envelope"]["api_limit_usd"] != 1.2:
        raise ValueError("this pilot uses the frozen 1.20 USD API partition")
    archived_hashes = {row["sha256"] for row in backup["members"]}
    if (
        sha256_file(args.measurement) not in archived_hashes
        or sha256_file(args.media_index) not in archived_hashes
    ):
        raise ValueError("measurement and media index must belong to the verified archive")
    if (
        measurement.get("status") != "COMPLETE"
        or measurement.get("worklist_id") != work["worklist_id"]
        or measurement.get("media_index_sha256") != sha256_file(args.media_index)
        or measurement.get("payload_pairs") != 96
        or measurement.get("api_calls") != 0
        or index.get("worklist_id") != work["worklist_id"]
        or index.get("status") != "COMPLETE"
        or index.get("recipe") != work["recipe"]
    ):
        raise ValueError("complete unjudged pilot measurements are required")
    protocol = freeze_pilot_observability_protocol(pilot, work)
    validate_pilot_observability_receipt(protocol, review, pilot=pilot, worklist=work)
    media = validate_pilot_media_index(work, index)["videos"]
    if {row["job_id"]: row["sha256"] for row in review["media"]} != {
        identity: row["video_sha256"] for identity, row in media.items()
    }:
        raise ValueError("reviewed media differs from the measured index")
    metric_hashes = {row["job_id"]: row["sha256"] for row in measurement["metric_receipts"]}
    if len(measurement["metric_receipts"]) != 88 or set(metric_hashes) != set(media):
        raise ValueError("all 88 unique metric receipts are required")
    metrics = {}
    for identity, digest in metric_hashes.items():
        path = args.metrics / identity / "receipt.json"
        if sha256_file(path) != digest or digest not in archived_hashes:
            raise ValueError("metric receipt differs from archived measurement")
        receipt = read(path)
        validate_metric_evidence(receipt)
        raw_path = path.with_name("metrics_eval_results.json")
        if (
            not raw_path.is_file()
            or sha256_file(raw_path) != receipt.get("raw_sha256")
            or receipt["raw_sha256"] not in archived_hashes
        ):
            raise ValueError("native metric result is missing or differs from archived receipt")
        expected = {
            "plan_id": work["worklist_id"],
            "prompt_id": identity,
            "video_sha256": media[identity]["video_sha256"],
            "seed": media[identity]["seed"],
            "prompt_hash": hash_canonical("forge/mvp-prompt/v1", media[identity]["prompt_text"]),
            "manifest_hash": index["base_manifest_hash"],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("metric receipt is outside the frozen video assignment")
        metrics[identity] = receipt
    if len({json.dumps(row["provenance"], sort_keys=True) for row in metrics.values()}) != 1:
        raise ValueError("metric provenance differs between pilot videos")
    size = measurement["payload_bytes"]
    if (
        type(size) is not int
        or not 0 < size <= (MAX_PAYLOAD_BYTES if args.archived_payload else 512 * 1024**2)
        or measurement["payload_sha256"] not in archived_hashes
    ):
        raise ValueError("compressed sheets exceed the selected input transport limit")
    packed = None
    if not args.archived_payload:
        packed = sys.stdin.buffer.read(size + 1)
        if (
            len(packed) != size
            or hashlib.sha256(packed).hexdigest() != measurement["payload_sha256"]
        ):
            raise ValueError("sheet stream differs from the verified measurement archive")
    pairs = {pair["slot_id"]: pair for pair in work["comparisons"]}
    geometry = {key: work["recipe"][key] for key in ("width", "height", "frames", "fps")}

    validated_payloads = {}

    def rows():
        if args.archived_payload:
            yield from archived_payloads(
                backup,
                payload_sha256=measurement["payload_sha256"],
                payload_bytes=size,
                max_rows=len(pairs),
                archive_bucket=args.archive_bucket,
            )
        else:
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
                while line := stream.readline(MAX_ROW_BYTES + 1):
                    if len(line) > MAX_ROW_BYTES:
                        raise ValueError("sheet payload exceeds its bounded row limit")
                    yield json.loads(line)

    def payloads(*, prevalidated=False):
        seen = set()
        for payload in rows():
            identity = payload["slot_id"]
            if identity not in pairs or identity in seen:
                raise ValueError("payload is duplicate or outside the frozen worklist")
            seen.add(identity)
            pair = pairs[identity]
            if (
                payload["worklist_id"] != work["worklist_id"]
                or payload["geometry"] != geometry
                or payload["prompt"] != media[pair["reference_job_id"]]["prompt_text"]
            ):
                raise ValueError("payload prompt or geometry differs from the frozen worklist")
            for side in ("reference", "candidate"):
                digest = media[pair[f"{side}_job_id"]]["video_sha256"]
                if (
                    payload[f"{side}_sha256"] != digest
                    or not verify_check(payload[f"{side}_raster"], digest, geometry)
                    or len(payload[f"{side}_sheets"]) != 6
                ):
                    raise ValueError("payload media/raster evidence is incomplete")
                for sheet in payload[f"{side}_sheets"]:
                    base64.b64decode(sheet, validate=True)
            digest = hash_canonical("forge/pilot-sheet-row/v1", payload)
            if prevalidated and validated_payloads.get(identity) != digest:
                raise ValueError("sheet row changed after the pre-dispatch validation pass")
            yield payload
        if seen != set(pairs):
            raise ValueError("not every frozen comparison has a sheet payload")

    for payload in payloads():
        validated_payloads[payload["slot_id"]] = hash_canonical("forge/pilot-sheet-row/v1", payload)
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    budget = ApiBudget(args.out / "luna-budget.json", 1.2, reservation_usd=PILOT_RESERVATION_USD)
    state = {
        "schema": "forge/evaluator-pilot-result/v1",
        "worklist_id": work["worklist_id"],
        "observability_receipt_id": review["receipt_id"],
        "status": "RUNNING",
        "qualification": "prohibited",
        "promotion_eligible": False,
        "rows": [],
    }

    def save():
        temporary = args.out / "result.new"
        with temporary.open("w") as stream:
            json.dump(state, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.out / "result.json")

    save()
    used = set()
    slots = {slot["slot_id"]: slot for slot in pilot["slots"]}
    try:
        for payload in payloads(prevalidated=True):
            if budget.path.exists() and any(
                call["state"] != "SETTLED" for call in read(budget.path)["calls"]
            ):
                raise ValueError("uncertain API charge blocks further pilot dispatch")
            identity = payload["slot_id"]
            state["active_slot_id"] = identity
            save()
            luna = judge_pair(
                payload["prompt"],
                [base64.b64decode(sheet) for sheet in payload["reference_sheets"]],
                [base64.b64decode(sheet) for sheet in payload["candidate_sheets"]],
                budget=budget,
                env_file=args.env_file,
                pair_id=identity,
            )
            response_ids = {call["response_id"] for call in luna["calls"]}
            if luna["status"] != "OK" or len(response_ids) != 2 or used & response_ids:
                raise ValueError("uncertain or reused Luna evidence stops the pilot")
            used.update(response_ids)
            pair = pairs[identity]
            orders = fused_axis_orders(
                luna["calls"], metrics[pair["reference_job_id"]], metrics[pair["candidate_job_id"]]
            )
            state["rows"].append(
                {
                    **slots[identity],
                    "split": "development",
                    "observable": True,
                    "orders": orders,
                    "luna": luna,
                    "reference_metrics_sha256": metric_hashes[pair["reference_job_id"]],
                    "candidate_metrics_sha256": metric_hashes[pair["candidate_job_id"]],
                }
            )
            save()
        provenance = {
            "luna_evaluator_id": MODEL,
            "vbench_model_hashes": hash_canonical(
                "forge/vbench-weight-lock/v1",
                next(iter(metrics.values()))["provenance"]["weight_sha256"],
            ),
            "source_implementation_hash": evaluator_implementation_hash(),
        }
        try:
            _, derivation = fit_policy_from_development(
                state["rows"], evaluator_provenance=provenance
            )
            state.update(status="DEVELOPMENT_SEPARATED", derivation=derivation)
        except ValueError:
            state["status"] = "DEVELOPMENT_NOT_SEPARATED"
        state.pop("active_slot_id", None)
        save()
        return state
    except BaseException as error:
        state.update(status="INCOMPLETE", error_type=type(error).__name__)
        save()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "pilot",
        "ownership",
        "closed",
        "backup",
        "measurement",
        "media-index",
        "review",
        "metrics",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--archive-bucket")
    parser.add_argument(
        "--archived-payload",
        action="store_true",
        help="Stream the verified S3 archive twice instead of reading compressed sheets on stdin",
    )
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "qualification": "prohibited"}))
    return 0 if result["status"] == "DEVELOPMENT_SEPARATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
