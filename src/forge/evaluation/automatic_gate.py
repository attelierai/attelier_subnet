"""Replay the historical automatic gate; not the bounded MVP evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from forge.evaluation.automatic_comparisons import _quality_comparisons, _timing_comparison
from forge.evaluation.automatic_evidence import (
    AXES,
    CALIBRATION_SCHEMA,
    SUBJECT_CLASSES,
    TEMPORAL_TRANSITION_POLICY,
    _calibration_plan_binding,
    _contract,
    _EvidenceError,
    _freshness,
    _native_detail_evidence,
    _panel,
    _policy,
    _production_binding,
    _raster_integrity_evidence,
    _response_id_uniqueness,
    _result,
    _transition_evidence,
)
from forge.evaluation.automatic_statistics import (
    _quality_bound,
    _timing_bound,
    _zero_variance_floor,
)
from forge.evaluation.vision_calibration import (
    DECISION_AGGREGATION_CODE_ID,
    DECISION_AGGREGATION_CODE_ID_V3,
    QUALITY_AGGREGATION_V2,
    QUALITY_AGGREGATION_V3,
)
from forge.protocol.hashing import hash_canonical
from forge.protocol.prompt_pack import PACK_SOURCE, verified_production_plan_prompts


def evaluate_automatic(
    plan: dict[str, Any],
    policy: dict[str, Any],
    quality: dict[str, Any],
    timing: dict[str, Any],
    calibration: dict[str, Any],
    *,
    preflight: Mapping[str, Any] | None = None,
    prompt_history_checkpoint: Mapping[str, Any] | None = None,
    development: bool = False,
) -> dict[str, Any]:
    """Return a promotion gate from local automatic evidence, never human judgment."""

    try:
        if not all(
            isinstance(value, Mapping) for value in (plan, policy, quality, timing, calibration)
        ):
            raise _EvidenceError("all evidence inputs must be objects")
        checked_policy = _policy(policy, development=development)
        contract = _contract(plan)
        if (
            tuple(contract.axes) != AXES
            or checked_policy["alpha"] != contract.alpha
            or checked_policy["max_candidates"] != contract.max_candidates
            or checked_policy["min_speedup"] != contract.min_speedup
            or checked_policy["quality_margin"] != contract.axis_noninferiority_margin
        ):
            raise _EvidenceError("automatic policy differs from the bound challenge contract")
        slots, prompts, panel_hash = _panel(plan)
        stage = plan.get("stage")
        if stage not in contract.stage_sizes():
            raise _EvidenceError("plan stage is unknown")
        expected_prompt_count, expected_seed_count = contract.stage_sizes()[stage]
        prompt_history = None
        if not development:
            if plan.get("prompt_source") != PACK_SOURCE:
                raise _EvidenceError("production plan must use a sealed prompt pack")
            try:
                incumbent = plan.get("incumbent")
                if not isinstance(incumbent, Mapping):
                    raise _EvidenceError("production plan incumbent state is required")
                prescribed, prompt_history = verified_production_plan_prompts(
                    plan,
                    contract,
                    prompt_history_checkpoint=prompt_history_checkpoint,
                    expected_certified_state_hash=incumbent.get("certified_state_hash"),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise _EvidenceError(str(error)) from None
            if plan.get("prompts") != prescribed:
                raise _EvidenceError("plan prompts differ from its sealed pack")
        if not development and (
            len(prompts) != expected_prompt_count
            or any(len(prompt["video_seeds"]) != expected_seed_count for prompt in prompts.values())
        ):
            raise _EvidenceError("production panel differs from its contract stage size")
        comparisons, candidate_hash, quality_public = _quality_comparisons(
            quality, checked_policy, plan, slots, prompts, panel_hash, development=development
        )
        timing_values, timing_public = _timing_comparison(
            timing,
            plan,
            slots,
            candidate_hash,
            comparisons["incumbent"]["reference_manifest_hash"],
            comparisons["incumbent"]["reference_videos"],
            comparisons["incumbent"]["candidate_videos"],
            development=development,
        )
        _production_binding(
            plan,
            candidate_hash,
            comparisons["incumbent"]["reference_manifest_hash"],
            development=development,
        )
        _response_id_uniqueness(quality, calibration)
        preflight_hash = None
        if not development:
            from forge.validator.runtime_preflight import validate_preflight_receipt

            try:
                validate_preflight_receipt(
                    preflight, plan, contract, candidate_manifest_hash=candidate_hash
                )
            except ValueError as error:
                raise _EvidenceError(str(error)) from error
            preflight_hash = hash_canonical("forge/runtime-preflight-receipt/v1", preflight)
        calibration_valid = (
            calibration.get("schema") == CALIBRATION_SCHEMA
            and calibration.get("evaluator_id") == checked_policy["evaluator_id"]
            and calibration.get("policy_id") == checked_policy["policy_id"]
            and calibration.get("contract_hash") == plan.get("contract_hash")
            and calibration.get("geometry") == plan.get("geometry")
            and isinstance(calibration.get("calibration_id"), str)
            and calibration["calibration_id"]
            and isinstance(calibration.get("controls"), Mapping)
        )
        if not development and checked_policy.get("quality_aggregation") in {
            QUALITY_AGGREGATION_V2,
            QUALITY_AGGREGATION_V3,
        }:
            evaluator = calibration.get("evaluator")
            expected_code = (
                DECISION_AGGREGATION_CODE_ID_V3
                if checked_policy.get("quality_aggregation") == QUALITY_AGGREGATION_V3
                else DECISION_AGGREGATION_CODE_ID
            )
            calibration_valid &= (
                isinstance(evaluator, Mapping)
                and evaluator.get("decision_aggregation_code_id") == expected_code
                and (
                    checked_policy.get("quality_aggregation") != QUALITY_AGGREGATION_V3
                    or evaluator.get("temporal_transition_policy") == TEMPORAL_TRANSITION_POLICY
                )
            )
        if not calibration_valid:
            raise _EvidenceError("calibration does not bind the evaluator policy and plan")
        if not development:
            _calibration_plan_binding(plan, calibration)
        try:
            from forge.evaluation.vision_calibration import verify_calibration
        except ImportError:
            raise _EvidenceError("calibration verifier is unavailable") from None
        if not verify_calibration(calibration, checked_policy, plan):
            raise _EvidenceError("calibration controls did not recompute")
        if not development:
            _freshness(plan, quality, calibration)
            _native_detail_evidence(quality, calibration, plan["geometry"])
            if checked_policy.get("quality_aggregation") == QUALITY_AGGREGATION_V3:
                _transition_evidence(quality, calibration, plan["geometry"])
            _raster_integrity_evidence(quality, calibration, plan["geometry"])
        alpha = checked_policy["alpha"] / (checked_policy["max_candidates"] * 2 * (2 * 49 + 1))
        precision = {
            "quality_alpha": alpha,
            "quality_margin": checked_policy["quality_margin"],
            "zero_ties_prompts_needed_per_slice": _zero_variance_floor(
                alpha, checked_policy["quality_margin"]
            ),
        }
        quality_bounds: dict[str, Any] = {}
        quality_diagnostics: dict[str, Any] = {}
        failures: list[str] = []
        coverage: dict[str, Any] = {}
        passes = True
        for role, comparison in comparisons.items():
            by_prompt = comparison["by_prompt"]
            required_prompts = 2 if development else checked_policy["min_prompts"]
            if len(by_prompt) < required_prompts:
                raise _EvidenceError(f"{role} quality has too few independent prompts")
            role_bounds: dict[str, Any] = {"global": {}}
            role_coverage: dict[str, int] = {"global": len(by_prompt)}
            for metric in ("average", *AXES):
                bound = _quality_bound(
                    [sum(values[metric]) / len(values[metric]) for values in by_prompt.values()],
                    alpha,
                )
                role_bounds["global"][metric] = bound
                passes &= bound["lower"] >= -checked_policy["quality_margin"]
                if bound["upper"] < -checked_policy["quality_margin"]:
                    failures.append(f"{role} {metric} is conclusively below quality margin")
            slice_specs = (
                [
                    ("subject_class", subject, tuple(["average", *AXES]))
                    for subject in sorted(SUBJECT_CLASSES)
                ]
                + [("camera_mode", value, ("average",)) for value in ("static", "moving")]
                + [("style_family", value, ("average",)) for value in ("realistic", "stylized")]
            )
            for property_name, value, metrics in slice_specs:
                name = f"{property_name}:{value}"
                selected = [
                    prompt_values
                    for prompt_id, prompt_values in by_prompt.items()
                    if prompts[prompt_id]["properties"].get(property_name) == value
                ]
                role_coverage[name] = len(selected)
                minimum = 2 if development else checked_policy["min_slice_prompts"]
                if len(selected) < minimum:
                    if development:
                        continue
                    raise _EvidenceError(f"{role} {name} slice is underpowered")
                role_bounds[name] = {}
                for metric in metrics:
                    bound = _quality_bound(
                        [sum(item[metric]) / len(item[metric]) for item in selected], alpha
                    )
                    role_bounds[name][metric] = bound
                    passes &= bound["lower"] >= -checked_policy["quality_margin"]
                    if bound["upper"] < -checked_policy["quality_margin"]:
                        failures.append(f"{role} {name} {metric} is conclusively below margin")
            quality_bounds[role] = role_bounds
            quality_diagnostics[role] = {
                "raw_mean_average": sum(
                    sum(values["raw_mean_average"]) / len(values["raw_mean_average"])
                    for values in by_prompt.values()
                )
                / len(by_prompt),
                "order_disagreement_by_axis": {
                    axis: sum(
                        sum(values[f"order_disagreement:{axis}"])
                        / len(values[f"order_disagreement:{axis}"])
                        for values in by_prompt.values()
                    )
                    / len(by_prompt)
                    for axis in AXES
                },
                "order_sign_flip_rate": sum(
                    sum(values["order_sign_flips"]) / len(values["order_sign_flips"])
                    for values in by_prompt.values()
                )
                / (len(by_prompt) * len(AXES)),
            }
            coverage[role] = role_coverage
        precision["available_prompts_by_role_and_slice"] = coverage
        if len(timing_values["by_prompt"]) < (2 if development else checked_policy["min_prompts"]):
            raise _EvidenceError("timing has too few independent prompts")
        speed = _timing_bound(list(timing_values["by_prompt"].values()), alpha)
        speed["lower_speedup"] = math.exp(speed["lower_log_ratio"])
        speed["upper_speedup"] = math.exp(speed["upper_log_ratio"])
        passes &= speed["lower_speedup"] >= checked_policy["min_speedup"]
        if speed["upper_speedup"] < checked_policy["min_speedup"]:
            failures.append("speed is conclusively below required speedup")
        if stage != "confirmation":
            passes = False
            failures.append("only a confirmation panel can be promotion eligible")
        calibration_status = calibration.get("status")
        if calibration_status != "PASS":
            passes = False
            failures.append(f"calibration status is {calibration_status!r}")
        if not quality_public or not timing_public:
            passes = False
            failures.append("private artifact evidence is diagnostic only")
    except _EvidenceError as exc:
        return _result("INCOMPLETE", [str(exc)])

    if calibration_status != "PASS":
        status = "DEVELOPMENT_DIAGNOSTIC" if development else "INCOMPLETE"
    else:
        status = "PROMOTE_ELIGIBLE" if passes and not development else "INCONCLUSIVE"
    if (
        calibration_status == "PASS"
        and failures
        and any("conclusively" in item for item in failures)
    ):
        status = "FAIL"
    if development:
        status = "DEVELOPMENT_DIAGNOSTIC"
    return _result(
        status,
        failures,
        candidate_manifest_hash=candidate_hash,
        quality=quality_bounds,
        quality_diagnostics=quality_diagnostics,
        coverage=coverage,
        precision=precision,
        timing=speed,
        calibration_status=calibration_status,
        preflight_hash=preflight_hash,
        prompt_history=prompt_history,
    )
