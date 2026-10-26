"""Bounded screening and confirmation decisions for the Wan MVP."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean, stdev
from typing import Any

from scipy.stats import t

from forge.evaluation.mvp_calibration import validate_calibration
from forge.evaluation.mvp_evidence import (
    _as_records,
    _plan_slots,
    _quality_rows,
    _validate_quality_binding,
    _validate_timing_receipts,
    validate_confirmation_panel,
)
from forge.evaluation.mvp_policy import (
    CONFIRMATION_PROMPTS,
    SCREEN_PROMPTS,
    TIMING_PROMPTS,
    Decision,
    EvaluationPolicy,
    _sha,
)


def _speed_lower_bound(
    rows: Sequence[Mapping[str, Any]],
    count: int = TIMING_PROMPTS,
    allowed_prompt_ids: set[str] | None = None,
) -> tuple[float, float]:
    rows = _as_records(rows, "timing rows")
    if len(rows) != count:
        raise ValueError(f"timing evidence needs exactly {count} prompts")
    logs: list[float] = []
    prompt_ids: set[str] = set()
    for row in rows:
        prompt_id, runs = row.get("prompt_id"), row.get("runs")
        if (
            not isinstance(prompt_id, str)
            or prompt_id in prompt_ids
            or not isinstance(runs, Sequence)
        ):
            raise ValueError("timing rows need unique prompt ids and ABBA runs")
        if allowed_prompt_ids is not None and prompt_id not in allowed_prompt_ids:
            raise ValueError("timing prompt is outside the frozen evaluation panel")
        if len(runs) != 4 or [run.get("model") for run in runs if isinstance(run, Mapping)] != [
            "reference",
            "candidate",
            "candidate",
            "reference",
        ]:
            raise ValueError("timing runs must be reference/candidate/candidate/reference")
        values = [run.get("seconds") for run in runs]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError("timing values must be finite positive seconds")
        prompt_ids.add(prompt_id)
        logs.append(
            math.log(values[0] / 2 + values[3] / 2) - math.log(values[1] / 2 + values[2] / 2)
        )
    try:
        ratio = math.exp(mean(logs))
        lower = math.exp(
            mean(logs) - t.ppf(0.95, len(logs) - 1) * stdev(logs) / math.sqrt(len(logs))
        )
    except OverflowError as exc:
        raise ValueError("timing ratio exceeds the finite numeric range") from exc
    return ratio, lower


def _screen_speed_lower_bound(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    rows = _as_records(rows, "screen timing rows")
    if len(rows) != SCREEN_PROMPTS:
        raise ValueError(f"screen timing evidence needs exactly {SCREEN_PROMPTS} prompts")
    prompt_ids: set[str] = set()
    logs: list[float] = []
    for row in rows:
        prompt_id, reference, candidate = (
            row.get("prompt_id"),
            row.get("reference_seconds"),
            row.get("candidate_seconds"),
        )
        if (
            not isinstance(prompt_id, str)
            or prompt_id in prompt_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in (reference, candidate)
            )
        ):
            raise ValueError("screen timing rows need unique finite AB measurements")
        prompt_ids.add(prompt_id)
        logs.append(math.log(reference) - math.log(candidate))
    try:
        ratio = math.exp(mean(logs))
        lower = math.exp(
            mean(logs) - t.ppf(0.95, len(logs) - 1) * stdev(logs) / math.sqrt(len(logs))
        )
    except OverflowError as exc:
        raise ValueError("screen timing ratio exceeds the finite numeric range") from exc
    return ratio, lower


def evaluate_screening(
    quality_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    policy: EvaluationPolicy,
) -> dict[str, object]:
    """Screen 16 prompts for obvious defects and plausible speed potential."""

    try:
        policy.validate()
        axes, cells, _, _ = _quality_rows(quality_rows, SCREEN_PROMPTS)
        _, lower = _screen_speed_lower_bound(timing_rows)
    except ValueError as exc:
        return {"status": Decision.INCOMPLETE, "reasons": [str(exc)]}
    regressions = [axis for axis, value in axes.items() if value < -policy.quality_margin]
    regressions.extend(
        f"{cell}:{axis}"
        for cell, values in cells.items()
        for axis, value in values.items()
        if value < -policy.quality_margin
    )
    if regressions:
        return {"status": Decision.SCREEN_FAIL, "reasons": ["SCREEN_QUALITY_DEFECT", *regressions]}
    if lower < policy.min_speedup_ci:
        return {
            "status": Decision.SCREEN_FAIL,
            "reasons": ["SCREEN_NO_SPEED_POTENTIAL"],
            "speed_lower_ci": lower,
        }
    return {"status": Decision.SCREEN_PASS, "reasons": [], "speed_lower_ci": lower}


def evaluate_confirmation(
    prompts: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
    policy: EvaluationPolicy,
    *,
    wan50_quality_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Return the only decision allowed to enter a certification route."""

    calibration_result = validate_calibration(calibration, policy)
    if calibration_result["status"] != "PASS":
        return {
            "status": Decision.INCOMPLETE,
            "reasons": ["CALIBRATION_NOT_VALID", *calibration_result["reasons"]],
            "calibration": calibration_result,
        }
    if wan50_quality_rows is None:
        return {
            "status": Decision.INCOMPLETE,
            "reasons": ["WAN50_QUALITY_COMPARISON_MISSING"],
            "calibration": calibration_result,
        }
    try:
        validate_confirmation_panel(prompts)
        incumbent = _quality_rows(quality_rows, CONFIRMATION_PROMPTS)
        wan50 = _quality_rows(wan50_quality_rows, CONFIRMATION_PROMPTS)
        ratio, lower = _speed_lower_bound(
            timing_rows, allowed_prompt_ids={row["prompt_id"] for row in prompts}
        )
    except ValueError as exc:
        return {
            "status": Decision.INCOMPLETE,
            "reasons": [str(exc)],
            "calibration": calibration_result,
        }
    assessments = {"incumbent": incumbent, "wan50": wan50}
    regressions = []
    uncertain = []
    for reference, (axes, cells, lower_axes, lower_cells) in assessments.items():
        regressions.extend(
            f"{reference}:{axis}" for axis, value in axes.items() if value < -policy.quality_margin
        )
        regressions.extend(
            f"{reference}:{cell}:{axis}"
            for cell, values in cells.items()
            for axis, value in values.items()
            if value < -policy.quality_margin
        )
        uncertain.extend(
            f"{reference}:{axis}"
            for axis, value in lower_axes.items()
            if value < -policy.quality_margin
        )
        uncertain.extend(
            f"{reference}:{cell}:{axis}"
            for cell, values in lower_cells.items()
            for axis, value in values.items()
            if value < -policy.quality_margin
        )
    if regressions:
        return {
            "status": Decision.FAIL,
            "reasons": ["PROTECTED_QUALITY_REGRESSION", *regressions],
            "quality": {reference: value[0] for reference, value in assessments.items()},
            "by_category": {reference: value[1] for reference, value in assessments.items()},
            "speed_ratio": ratio,
            "speed_lower_ci": lower,
        }
    if uncertain:
        return {
            "status": Decision.INCONCLUSIVE,
            "reasons": ["QUALITY_UNCERTAINTY", *uncertain],
            "quality": {reference: value[0] for reference, value in assessments.items()},
            "quality_lower_ci": {reference: value[2] for reference, value in assessments.items()},
            "by_category": {reference: value[1] for reference, value in assessments.items()},
            "category_lower_ci": {reference: value[3] for reference, value in assessments.items()},
        }
    if lower < policy.min_speedup_ci:
        return {
            "status": Decision.INCONCLUSIVE,
            "reasons": ["SPEED_CONFIDENCE_BELOW_MINIMUM"],
            "quality": {reference: value[0] for reference, value in assessments.items()},
            "by_category": {reference: value[1] for reference, value in assessments.items()},
            "speed_ratio": ratio,
            "speed_lower_ci": lower,
        }
    return {
        "status": Decision.PROMOTE_ELIGIBLE,
        "reasons": [],
        "quality": {reference: value[0] for reference, value in assessments.items()},
        "by_category": {reference: value[1] for reference, value in assessments.items()},
        "speed_ratio": ratio,
        "speed_lower_ci": lower,
        "calibration": calibration_result,
    }


def evaluate_mvp(
    *,
    plan: Mapping[str, Any],
    quality_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    policy: EvaluationPolicy,
    calibration: Mapping[str, Any] | None = None,
    wan50_quality_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Evaluate a frozen MVP plan without invoking or relaxing legacy gates."""

    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if (
        plan.get("schema") != "forge/mvp-evaluation-plan/v1"
        or plan.get("plan_id") != _sha(body)
        or plan.get("policy_id") != policy.policy_id
    ):
        return {"status": Decision.INCOMPLETE, "reasons": ["MVP_PLAN_UNBOUND"]}
    prompts = plan.get("prompts")
    if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
        return {"status": Decision.INCOMPLETE, "reasons": ["MVP_PROMPTS_MISSING"]}
    identities = (
        plan.get("candidate_manifest_hash"),
        plan.get("incumbent_manifest_hash"),
        plan.get("wan50_manifest_hash"),
    )
    if any(
        not isinstance(identity, str)
        or len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
        for identity in identities
    ):
        return {"status": Decision.INCOMPLETE, "reasons": ["MVP_MODEL_IDENTITIES_UNBOUND"]}
    try:
        if plan.get("stage") == "screening":
            _validate_quality_binding(
                quality_rows,
                prompts,
                plan_id=plan["plan_id"],
                candidate_manifest_hash=plan["candidate_manifest_hash"],
                reference_manifest_hash=plan["incumbent_manifest_hash"],
            )
            return evaluate_screening(quality_rows, timing_rows, policy)
        if plan.get("stage") == "confirmation":
            if wan50_quality_rows is None:
                return {
                    "status": Decision.INCOMPLETE,
                    "reasons": ["WAN50_QUALITY_COMPARISON_MISSING"],
                }
            bootstrap_alias = plan["incumbent_manifest_hash"] == plan["wan50_manifest_hash"]
            if bootstrap_alias and quality_rows != wan50_quality_rows:
                raise ValueError("bootstrap alias must reuse the exact canonical evidence rows")
            _validate_quality_binding(
                quality_rows,
                prompts,
                plan_id=plan["plan_id"],
                candidate_manifest_hash=plan["candidate_manifest_hash"],
                reference_manifest_hash=plan["incumbent_manifest_hash"],
                bootstrap_alias=bootstrap_alias,
            )
            _validate_quality_binding(
                wan50_quality_rows,
                prompts,
                plan_id=plan["plan_id"],
                candidate_manifest_hash=plan["candidate_manifest_hash"],
                reference_manifest_hash=plan["wan50_manifest_hash"],
                bootstrap_alias=bootstrap_alias,
            )
            _validate_timing_receipts(
                timing_rows,
                set(_plan_slots(prompts)),
                candidate_manifest_hash=plan["candidate_manifest_hash"],
                reference_manifest_hash=plan["incumbent_manifest_hash"],
            )
            return evaluate_confirmation(
                prompts,
                quality_rows,
                timing_rows,
                calibration,
                policy,
                wan50_quality_rows=wan50_quality_rows,
            )
    except (TypeError, ValueError):
        return {"status": Decision.INCOMPLETE, "reasons": ["MVP_EVIDENCE_UNBOUND"]}
    return {"status": Decision.INCOMPLETE, "reasons": ["MVP_STAGE_INVALID"]}
