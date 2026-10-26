"""Attach replayed native VBench metrics to a public dashboard snapshot.

The composed quality axes are useful for a calibrated decision but their
normalization can obscure a large native imaging-quality movement.  This helper
keeps MUSIQ on its original 0..100 point scale and only publishes it after
replaying every signed diagnostic row used by the visible evaluation.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Verifier
from forge.evaluation.diagnostic_fusion import fuse_diagnostic_quality
from forge.evaluation.distributed import verify_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan, assignments_for
from forge.protocol.distributed_transport import report_digest
from forge.protocol.hashing import canonical_json_bytes, hash_canonical

RAW_FIELDS = {"pair", "reference_metrics", "candidate_metrics"}


def _read_raw(sources: Sequence[Path]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for source in sources:
        paths.extend(sorted(source.glob("*.json")) if source.is_dir() else [source])
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("raw evidence source is unreadable") from exc
        if not isinstance(value, dict) or set(value) != RAW_FIELDS:
            continue
        digest = hash_canonical(
            "forge/diagnostic-quality/v1",
            {
                "pair": value["pair"],
                "reference_metrics": value["reference_metrics"],
                "candidate_metrics": value["candidate_metrics"],
            },
        )
        previous = found.setdefault(digest, value)
        if canonical_json_bytes(previous) != canonical_json_bytes(value):
            raise ValueError("two raw evidence files share a content address")
    return found


def _metric_delta(raw: Mapping[str, Any]) -> float:
    try:
        reference = raw["reference_metrics"]["imaging_quality"]
        candidate = raw["candidate_metrics"]["imaging_quality"]
    except (KeyError, TypeError) as exc:
        raise ValueError("raw evidence lacks native MUSIQ") from exc
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100
        for value in (reference, candidate)
    ):
        raise ValueError("native MUSIQ must be finite on its 0..100 scale")
    return float(candidate - reference)


def _plans_by_hash(
    plans: Sequence[DistributedEvaluationPlan | Mapping[str, Any]],
) -> dict[str, DistributedEvaluationPlan]:
    result = {}
    for item in plans:
        plan = (
            item
            if isinstance(item, DistributedEvaluationPlan)
            else DistributedEvaluationPlan.from_dict(item)
        )
        digest = plan.plan_hash()
        if digest in result:
            raise ValueError("dashboard native metrics received a duplicate plan")
        result[digest] = plan
    return result


def attach_native_metrics(
    snapshot: Mapping[str, Any],
    plans: Sequence[DistributedEvaluationPlan | Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    evidence_sources: Sequence[Path],
) -> dict[str, Any]:
    """Return a snapshot with complete, replayed MUSIQ point deltas.

    The function refuses a partial mean.  Every frozen prompt must have native
    evidence from every *visible valid* contributor, then replica means are
    averaged equally across unique prompts.  It supports only the diagnostic
    Wan-50 bootstrap where King and Wan-50 are the same reference.
    """
    result = copy.deepcopy(dict(snapshot))
    plans_by_hash = _plans_by_hash(plans)
    raw_by_hash = _read_raw(evidence_sources)
    signed: dict[tuple[str, str], Mapping[str, Any]] = {}
    verifier = Sr25519Verifier()
    for report in reports:
        plan_hash = report.get("plan_hash")
        plan = plans_by_hash.get(plan_hash) if isinstance(plan_hash, str) else None
        if plan is None:
            continue
        body = verify_partial_report(plan, report, verifier)
        key = (plan_hash, str(body["validator_hotkey"]))
        if key in signed:
            raise ValueError("duplicate signed report for dashboard native metrics")
        signed[key] = report

    evaluations = result.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("dashboard snapshot has no evaluations")
    for evaluation in evaluations:
        if not isinstance(evaluation, dict) or not isinstance(evaluation.get("plan_hash"), str):
            raise ValueError("dashboard evaluation has no plan hash")
        plan = plans_by_hash.get(evaluation["plan_hash"])
        if plan is None:
            raise ValueError("dashboard evaluation plan is not registered")
        if plan.stage != "diagnostic" or plan.king_manifest_hash != plan.wan50_manifest_hash:
            raise ValueError("native MUSIQ currently requires a Wan-50 bootstrap diagnostic")
        consensus = evaluation.get("consensus")
        reports_out = consensus.get("reports") if isinstance(consensus, dict) else None
        if not isinstance(reports_out, list):
            raise ValueError("dashboard evaluation has no consensus reports")
        accepted: dict[str, Mapping[str, Any]] = {}
        visible: dict[str, Mapping[str, Any]] = {}
        for item in reports_out:
            if not isinstance(item, dict) or not isinstance(item.get("validator"), str):
                raise ValueError("dashboard consensus report is malformed")
            hotkey, digest = item.get("validator"), item.get("report_sha256")
            if hotkey in visible:
                raise ValueError("dashboard consensus repeats a validator")
            visible[hotkey] = item
            if item.get("status") != "valid":
                raise ValueError("native MUSIQ requires every committee report to be valid")
            if not isinstance(hotkey, str) or not isinstance(digest, str):
                raise ValueError("visible valid report has no immutable content hash")
            report = signed.get((plan.plan_hash(), hotkey))
            if report is None or report_digest(report) != digest:
                raise ValueError("visible report differs from supplied signed evidence")
            accepted[hotkey] = report
        if set(visible) != set(plan.committee):
            raise ValueError("dashboard consensus does not cover the frozen committee")

        by_prompt: dict[str, list[float]] = defaultdict(list)
        for hotkey, report in accepted.items():
            body = verify_partial_report(plan, report, verifier)
            expected = {
                (item.prompt_id, item.seed)
                for item in assignments_for(plan, hotkey)
                if item.kind == "quality"
            }
            king_rows = {row["prompt_id"]: row for row in body["quality_vs_king"]}
            if {(row["prompt_id"], row["seed"]) for row in body["quality_vs_king"]} != expected or {
                (row["prompt_id"], row["seed"]) for row in body["quality_vs_wan50"]
            } != expected:
                raise ValueError("native MUSIQ requires all assigned quality work")
            for row in body["quality_vs_wan50"]:
                if canonical_json_bytes(king_rows.get(row["prompt_id"])) != canonical_json_bytes(
                    row
                ):
                    raise ValueError("bootstrap quality rows differ between King and Wan-50")
                raw_hash = row.get("raw_evidence_hash")
                raw = raw_by_hash.get(raw_hash) if isinstance(raw_hash, str) else None
                if raw is None:
                    raise ValueError("signed quality row has no matching raw evidence")
                pair = raw["pair"]
                if any(
                    pair.get(receipt, {}).get("validator_hotkey") != hotkey
                    for receipt in ("reference_receipt", "candidate_receipt")
                ):
                    raise ValueError("raw render receipts belong to another validator")
                replayed = fuse_diagnostic_quality(plan, prompt_id=row["prompt_id"], **raw)
                if canonical_json_bytes(replayed) != canonical_json_bytes(row):
                    raise ValueError("signed quality row differs from replayed raw evidence")
                if hotkey != body["validator_hotkey"]:
                    raise ValueError("report hotkey changed during native replay")
                by_prompt[row["prompt_id"]].append(_metric_delta(raw))
        expected = {row["prompt_id"] for row in plan.prompts}
        if set(by_prompt) != expected or any(not by_prompt[prompt_id] for prompt_id in expected):
            raise ValueError("native MUSIQ evidence is incomplete; refusing a partial mean")
        prompt_delta = {
            prompt_id: sum(values) / len(values) for prompt_id, values in by_prompt.items()
        }
        prompts = evaluation.get("prompts")
        if (
            not isinstance(prompts, list)
            or {row.get("id") for row in prompts if isinstance(row, dict)} != expected
        ):
            raise ValueError("dashboard prompt rows differ from the frozen native panel")
        for prompt in prompts:
            prompt["native_quality_delta"] = prompt_delta[prompt["id"]]
        deltas = list(prompt_delta.values())
        evaluation["native_metrics"] = {
            "metric": "MUSIQ",
            "unit": "points",
            "reference": "wan50",
            "mean_delta": sum(deltas) / len(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "unique_prompts": len(deltas),
            "aggregation": "equal_prompt_mean_after_replica_mean",
        }
    return result
