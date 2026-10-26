"""Signed partial evidence and deterministic global aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from forge.evaluation.distributed_plan import (
    DistributedEvaluationPlan,
    allocate_work,
    assignments_for,
)
from forge.evaluation.mvp import evaluate_confirmation
from forge.evaluation.mvp_policy import QUALITY_AXES, Decision, EvaluationPolicy
from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.ports import Signer, Verifier

REPORT_SCHEMA = "forge/distributed-evaluation-report/v1"


def signed_partial_report(
    plan: DistributedEvaluationPlan,
    signer: Signer,
    *,
    quality_vs_king: Sequence[Mapping[str, Any]] = (),
    quality_vs_wan50: Sequence[Mapping[str, Any]] = (),
    timing: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Sign only the assigned receipts a validator actually completed."""

    body = {
        "schema": REPORT_SCHEMA,
        "plan_hash": plan.plan_hash(),
        "validator_hotkey": signer.validator_hotkey,
        "quality_vs_king": [dict(row) for row in quality_vs_king],
        "quality_vs_wan50": [dict(row) for row in quality_vs_wan50],
        "timing": [dict(row) for row in timing],
    }
    _validate_report_body(plan, body)
    return {**body, "signature": signer.sign(canonical_json_bytes(body)).hex()}


def verify_partial_report(
    plan: DistributedEvaluationPlan, report: Mapping[str, Any], verifier: Verifier
) -> dict[str, Any]:
    if set(report) != {
        "schema",
        "plan_hash",
        "validator_hotkey",
        "quality_vs_king",
        "quality_vs_wan50",
        "timing",
        "signature",
    }:
        raise ValueError("distributed report fields are not exact")
    body = {key: report[key] for key in report if key != "signature"}
    _validate_report_body(plan, body)
    try:
        signature = bytes.fromhex(str(report["signature"]))
    except ValueError as exc:
        raise ValueError("distributed report signature is malformed") from exc
    if not verifier.verify(str(body["validator_hotkey"]), canonical_json_bytes(body), signature):
        raise ValueError("distributed report signature is invalid")
    return dict(body)


def _validate_report_body(plan: DistributedEvaluationPlan, body: Mapping[str, Any]) -> None:
    if (
        body.get("schema") != REPORT_SCHEMA
        or body.get("plan_hash") != plan.plan_hash()
        or body.get("validator_hotkey") not in plan.committee
    ):
        raise ValueError("report is outside the frozen distributed plan")
    hotkey = str(body["validator_hotkey"])
    prompts = {row["prompt_id"]: row for row in plan.prompts}
    allowed = defaultdict(set)
    for item in assignments_for(plan, hotkey):
        allowed[item.kind].add((item.prompt_id, item.seed))
    for field, kind in (
        ("quality_vs_king", "quality"),
        ("quality_vs_wan50", "quality"),
        ("timing", "timing"),
    ):
        rows = body.get(field)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("report evidence rows are invalid")
        seen: set[tuple[str, int]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("report evidence row is invalid")
            identity = (row.get("prompt_id"), row.get("seed"))
            if (
                not isinstance(identity[0], str)
                or type(identity[1]) is not int
                or identity in seen
                or identity not in allowed[kind]
            ):
                raise ValueError("report contains unassigned or duplicate work")
            seen.add(identity)
            if kind == "quality" and any(
                row.get(key) != prompts[identity[0]].get(key)
                for key in ("subject_class", "camera_type", "style_family")
            ):
                raise ValueError("quality categories differ from the assigned prompt")
            if kind == "timing":
                timing_schema = row.get("timing_schema")
                if plan.stage == "screening" and timing_schema != "forge/screen-timing/v1":
                    raise ValueError("screening timing requires an explicit AB receipt")
                if plan.stage == "confirmation" and timing_schema not in {
                    None,
                    "forge/paired-timing/v1",
                }:
                    raise ValueError("confirmation timing requires ABBA evidence")


def aggregate_partial_reports(
    plan: DistributedEvaluationPlan, reports: Sequence[Mapping[str, Any]], verifier: Verifier
) -> dict[str, Any]:
    """Verify, audit and combine reports. A prompt remains one global sample."""

    valid: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    for report in reports:
        try:
            body = verify_partial_report(plan, report, verifier)
            hotkey = str(body["validator_hotkey"])
            previous = valid.get(hotkey)
            if previous is not None and canonical_json_bytes(previous) != canonical_json_bytes(
                body
            ):
                invalid.append(f"EQUIVOCATION:{hotkey}")
                valid.pop(hotkey, None)
            elif previous is None and f"EQUIVOCATION:{hotkey}" not in invalid:
                valid[hotkey] = body
        except ValueError as exc:
            invalid.append(f"INVALID_REPORT:{exc}")
    grouped = {
        field: _group_rows(valid, field)
        for field in ("quality_vs_king", "quality_vs_wan50", "timing")
    }
    quality_king, king_contested = _aggregate_quality(grouped["quality_vs_king"], plan)
    quality_wan50, wan50_contested = _aggregate_quality(grouped["quality_vs_wan50"], plan)
    timing, timing_contested = _aggregate_timing(grouped["timing"], plan)
    expected_quality = {row["prompt_id"] for row in plan.prompts}
    expected_timing = set(plan.timing_prompt_ids)
    missing = {
        "quality_vs_king": sorted(expected_quality - {row["prompt_id"] for row in quality_king}),
        "quality_vs_wan50": sorted(expected_quality - {row["prompt_id"] for row in quality_wan50}),
        "timing": sorted(expected_timing - {row["prompt_id"] for row in timing}),
    }
    # Audit copies are required: coverage alone cannot complete an absent assignment.
    for field, kind in (
        ("quality_vs_king", "quality"),
        ("quality_vs_wan50", "quality"),
        ("timing", "timing"),
    ):
        delivered = {(hotkey, str(row["prompt_id"])) for hotkey, row in _report_rows(valid, field)}
        assigned = {
            (item.validator_hotkey, item.prompt_id)
            for item in allocate_work(plan)
            if item.kind == kind
        }
        missing[field].extend(
            f"{hotkey}:{prompt_id}" for hotkey, prompt_id in sorted(assigned - delivered)
        )
    contested = sorted(king_contested | wan50_contested | timing_contested)
    status = "COMPLETE"
    if invalid or any(missing.values()):
        status = "INCOMPLETE"
    if contested:
        status = "CONTESTED"
    if (plan.stage == "diagnostic" or not plan.audit_ready) and status == "COMPLETE":
        status = "DIAGNOSTIC"
    return {
        "schema": "forge/distributed-evaluation-aggregate/v1",
        "plan_hash": plan.plan_hash(),
        "status": status,
        "quality_vs_king": quality_king,
        "quality_vs_wan50": quality_wan50,
        "timing": timing,
        "missing": missing,
        "contested": contested,
        "invalid": sorted(invalid),
        "contributors": sorted(valid),
        "audit_calibrated": plan.audit_ready,
        "reward_eligible": False,
    }


def _group_rows(
    reports: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, list[tuple[str, Mapping[str, Any]]]]:
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for hotkey, report in reports.items():
        for row in report[field]:
            grouped[str(row["prompt_id"])].append((hotkey, row))
    return grouped


def _report_rows(
    reports: Mapping[str, Mapping[str, Any]], field: str
) -> list[tuple[str, Mapping[str, Any]]]:
    return [(hotkey, row) for hotkey, report in reports.items() for row in report[field]]


def _aggregate_quality(
    grouped: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]], plan: DistributedEvaluationPlan
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    contested: set[str] = set()
    for prompt_id, evidence in grouped.items():
        try:
            by_order = {True: [], False: []}
            for hotkey, row in evidence:
                raw_orders = row.get("orders", [])
                if (
                    not isinstance(raw_orders, list)
                    or len(raw_orders) != 2
                    or any(
                        not isinstance(order, Mapping)
                        or type(order.get("reference_first")) is not bool
                        for order in raw_orders
                    )
                    or {order["reference_first"] for order in raw_orders} != {True, False}
                ):
                    raise ValueError
                for order in raw_orders:
                    if (
                        not isinstance(order, Mapping)
                        or order.get("reference_first") not in by_order
                    ):
                        raise ValueError
                    axes = order.get("axes")
                    if not isinstance(axes, Mapping) or set(axes) != set(QUALITY_AXES):
                        raise ValueError
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not -1 <= value <= 1
                        for value in axes.values()
                    ):
                        raise ValueError
                    by_order[bool(order["reference_first"])].append((hotkey, axes))
            if any(not values for values in by_order.values()):
                raise ValueError
            orders = []
            for reference_first in (True, False):
                values = by_order[reference_first]
                for axis in QUALITY_AXES:
                    axis_values = [float(axes[axis]) for _, axes in values]
                    if (
                        len(axis_values) > 1
                        and plan.quality_tolerance is not None
                        and max(axis_values) - min(axis_values) > plan.quality_tolerance
                    ):
                        contested.add(f"quality:{prompt_id}:{axis}")
                total = sum(plan.committee[hotkey] for hotkey, _ in values)
                axes = {
                    axis: sum(
                        plan.committee[hotkey] * float(value[axis]) for hotkey, value in values
                    )
                    / total
                    for axis in QUALITY_AXES
                }
                orders.append({"reference_first": reference_first, "axes": axes})
            base = dict(sorted(evidence, key=lambda item: item[0])[0][1])
            base["orders"] = orders
            base["evidence_kind"] = "distributed_aggregate"
            base["contributions"] = [
                {
                    "hotkey": hotkey,
                    "stake": plan.committee[hotkey],
                    "row_hash": hash_canonical("forge/distributed-row/v1", row),
                }
                for hotkey, row in sorted(evidence, key=lambda item: item[0])
            ]
            rows.append(base)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            contested.add(f"quality:{prompt_id}:malformed")
    return sorted(rows, key=lambda row: row["prompt_id"]), contested


def _aggregate_timing(
    grouped: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]], plan: DistributedEvaluationPlan
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    contested: set[str] = set()
    screening = plan.stage == "screening"
    expected_order = (
        ["reference", "candidate"]
        if screening
        else ["reference", "candidate", "candidate", "reference"]
    )
    for prompt_id, evidence in grouped.items():
        try:
            values: list[tuple[str, float]] = []
            for hotkey, row in evidence:
                runs = row.get("runs")
                if (
                    not isinstance(runs, Sequence)
                    or len(runs) != len(expected_order)
                    or [run.get("model") for run in runs if isinstance(run, Mapping)]
                    != expected_order
                ):
                    raise ValueError
                seconds = [run.get("seconds") for run in runs]
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in seconds
                ):
                    raise ValueError
                # Dimensionless paired log ratio; raw seconds are never mixed across GPUs.
                values.append(
                    (
                        hotkey,
                        (
                            math.log(seconds[0]) - math.log(seconds[1])
                            if screening
                            else math.log(seconds[0] / 2 + seconds[3] / 2)
                            - math.log(seconds[1] / 2 + seconds[2] / 2)
                        ),
                    )
                )
            ratios = [value for _, value in values]
            if (
                len(ratios) > 1
                and plan.timing_log_ratio_tolerance is not None
                and max(ratios) - min(ratios) > plan.timing_log_ratio_tolerance
            ):
                contested.add(f"timing:{prompt_id}")
            total = sum(plan.committee[hotkey] for hotkey, _ in values)
            aggregate = sum(plan.committee[hotkey] * value for hotkey, value in values) / total
            base = dict(sorted(evidence, key=lambda item: item[0])[0][1])
            candidate = math.exp(-aggregate)
            base["runs"] = (
                [
                    {"model": "reference", "seconds": 1.0},
                    {"model": "candidate", "seconds": candidate},
                ]
                if screening
                else [
                    {"model": "reference", "seconds": 1.0},
                    {"model": "candidate", "seconds": candidate},
                    {"model": "candidate", "seconds": candidate},
                    {"model": "reference", "seconds": 1.0},
                ]
            )
            if screening:
                base["timing_schema"] = "forge/screen-timing/v1"
            rows.append(base)
        except (TypeError, ValueError, KeyError, OverflowError, ZeroDivisionError):
            contested.add(f"timing:{prompt_id}:malformed")
    return sorted(rows, key=lambda row: row["prompt_id"]), contested


def replay_qualified_distributed(
    plan: DistributedEvaluationPlan,
    reports: Sequence[Mapping[str, Any]],
    verifier: Verifier,
    policy: EvaluationPolicy,
    calibration: Mapping[str, Any] | None,
    *,
    mvp_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a full distributed confirmation through the frozen MVP rule.

    This is the authority boundary used by certification.  It deliberately
    accepts no caller-provided verdict: signed rows are aggregated again, the
    calibration campaign is checked against its frozen slots, and the aggregate
    is bound to the exact 128-prompt MVP plan before scoring.
    """

    from forge.evaluation.mvp_calibration import validate_qualified_calibration
    from forge.evaluation.mvp_evidence import (
        _sha,
        _text_hash,
        _validate_quality_binding,
        _validate_timing_receipts,
    )
    from forge.evaluation.mvp_policy import _is_hash

    def incomplete(aggregate: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            **aggregate,
            "decision": {"status": Decision.INCOMPLETE, "reasons": [reason]},
        }

    aggregate = aggregate_partial_reports(plan, reports, verifier)
    if plan.stage != "confirmation":
        return incomplete(aggregate, "DISTRIBUTED_REPLAY_REQUIRES_CONFIRMATION")
    if aggregate["status"] != "COMPLETE":
        return incomplete(aggregate, f"DISTRIBUTED_AGGREGATE_{aggregate['status']}")
    if not isinstance(mvp_plan, Mapping):
        return incomplete(aggregate, "MVP_PLAN_MISSING")
    mvp_body = {key: value for key, value in mvp_plan.items() if key != "plan_id"}
    expected_prompts = [dict(prompt) for prompt in plan.prompts]
    if (
        mvp_plan.get("schema") != "forge/mvp-evaluation-plan/v1"
        or mvp_plan.get("plan_id") != _sha(mvp_body)
        or mvp_plan.get("stage") != "confirmation"
        or mvp_plan.get("policy_id") != policy.policy_id
        or policy.policy_id != plan.policy_id
        or mvp_plan.get("candidate_manifest_hash") != plan.candidate_manifest_hash
        or mvp_plan.get("incumbent_manifest_hash") != plan.king_manifest_hash
        or mvp_plan.get("wan50_manifest_hash") != plan.wan50_manifest_hash
        or canonical_json_bytes(mvp_plan.get("prompts")) != canonical_json_bytes(expected_prompts)
        or plan.chain_context.get("mvp_plan_id") != mvp_plan.get("plan_id")
    ):
        return incomplete(aggregate, "DISTRIBUTED_MVP_PLAN_UNBOUND")
    calibration_result = validate_qualified_calibration(
        calibration, policy, required_calibration_hash=plan.audit_calibration_hash
    )
    if calibration_result["status"] != "PASS":
        return {
            **aggregate,
            "decision": {
                "status": Decision.INCOMPLETE,
                "reasons": ["DISTRIBUTED_CALIBRATION_UNQUALIFIED", *calibration_result["reasons"]],
                "calibration": calibration_result,
            },
        }

    # An aggregate may contain one convenient base row, but every signed
    # contribution must bind its own media and model identity.  Otherwise a
    # validator could contribute a score for substituted media and let another
    # validator's well-formed row make the aggregate look complete.
    prompt_slots = {prompt["prompt_id"]: prompt for prompt in expected_prompts}
    try:
        artifacts_by_comparison: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for report in reports:
            body = verify_partial_report(plan, report, verifier)
            for field, reference_manifest_hash in (
                ("quality_vs_king", plan.king_manifest_hash),
                ("quality_vs_wan50", plan.wan50_manifest_hash),
            ):
                for row in body[field]:
                    prompt = prompt_slots[row["prompt_id"]]
                    if (
                        row.get("plan_id") != mvp_plan["plan_id"]
                        or row.get("prompt_hash") != _text_hash(prompt["text"])
                        or row.get("candidate_manifest_hash") != plan.candidate_manifest_hash
                        or row.get("reference_manifest_hash") != reference_manifest_hash
                        or not _is_hash(row.get("candidate_video_sha256"))
                        or not _is_hash(row.get("reference_video_sha256"))
                        or not _is_hash(row.get("raw_evidence_hash"))
                    ):
                        raise ValueError("unbound distributed quality contribution")
                    artifacts_by_comparison[(field, row["prompt_id"])].add(
                        (row["candidate_video_sha256"], row["reference_video_sha256"])
                    )
            _validate_timing_receipts(
                body["timing"],
                set(prompt_slots),
                candidate_manifest_hash=plan.candidate_manifest_hash,
                reference_manifest_hash=plan.king_manifest_hash,
            )
        if any(len(hashes) != 1 for hashes in artifacts_by_comparison.values()):
            raise ValueError("audited quality contributions use different media")
    except (KeyError, TypeError, ValueError):
        return incomplete(aggregate, "DISTRIBUTED_RAW_EVIDENCE_UNBOUND")

    bootstrap_alias = plan.king_manifest_hash == plan.wan50_manifest_hash

    def fused(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            contributions = value.get("contributions")
            if not isinstance(contributions, list) or not contributions:
                raise ValueError("distributed quality lacks signed contributions")
            value["plan_id"] = mvp_plan["plan_id"]
            value["evidence_kind"] = "validator_fused_quality"
            value["raw_evidence_hash"] = hash_canonical(
                "forge/distributed-fused-quality/v1",
                {
                    "distributed_plan_hash": plan.plan_hash(),
                    "prompt_id": value.get("prompt_id"),
                    "contributions": contributions,
                },
            )
            if bootstrap_alias:
                identity = {
                    key: value.get(key)
                    for key in (
                        "plan_id",
                        "prompt_id",
                        "seed",
                        "prompt_hash",
                        "candidate_manifest_hash",
                        "reference_manifest_hash",
                        "candidate_video_sha256",
                        "reference_video_sha256",
                        "raw_evidence_hash",
                    )
                }
                value["reference_roles"] = ["wan50", "incumbent"]
                value["comparison_id"] = hash_canonical(
                    "forge/mvp-bootstrap-comparison/v1", identity
                )
            result.append(value)
        return result

    try:
        king_rows = fused(aggregate["quality_vs_king"])
        wan50_rows = fused(aggregate["quality_vs_wan50"])
        if bootstrap_alias and king_rows != wan50_rows:
            raise ValueError("bootstrap aggregate does not reuse canonical quality evidence")
        _validate_quality_binding(
            king_rows,
            expected_prompts,
            plan_id=mvp_plan["plan_id"],
            candidate_manifest_hash=plan.candidate_manifest_hash,
            reference_manifest_hash=plan.king_manifest_hash,
            bootstrap_alias=bootstrap_alias,
        )
        _validate_quality_binding(
            wan50_rows,
            expected_prompts,
            plan_id=mvp_plan["plan_id"],
            candidate_manifest_hash=plan.candidate_manifest_hash,
            reference_manifest_hash=plan.wan50_manifest_hash,
            bootstrap_alias=bootstrap_alias,
        )
    except (KeyError, TypeError, ValueError):
        return incomplete(aggregate, "DISTRIBUTED_QUALITY_EVIDENCE_UNBOUND")
    decision = evaluate_confirmation(
        expected_prompts,
        king_rows,
        aggregate["timing"],
        calibration,
        policy,
        wan50_quality_rows=wan50_rows,
    )
    binding = {
        "schema": "forge/qualified-distributed-replay/v1",
        "distributed_plan_hash": plan.plan_hash(),
        "mvp_plan_id": mvp_plan["plan_id"],
        "policy_id": policy.policy_id,
        "calibration_hash": calibration_result["binding"]["calibration_hash"],
        "aggregate_hash": hash_canonical("forge/distributed-aggregate/v1", aggregate),
    }
    return {**aggregate, "decision": decision, "eligibility_binding": binding}
