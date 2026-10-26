"""Machine-generated controls: sensitivity checks, not human preference labels."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy.stats import beta

from forge.protocol.contract import DEFAULT_AXES
from forge.protocol.hashing import hash_canonical

CONTROL_AXES = {
    "self": (),
    "freeze": ("motion_adherence", "motion_quality"),
    "blur": ("visual_quality",),
    "shuffle": ("temporal_stability", "motion_quality"),
    "wrong_video": ("prompt_adherence",),
}
QUALITY_AGGREGATION_V1 = "mean_of_blinded_orders"
QUALITY_AGGREGATION_V2 = "minimum_of_blinded_orders"
QUALITY_AGGREGATION_V3 = "symmetric_mean_prompt_clustered"
DECISION_AGGREGATION_CODE_ID = "forge/minimum-of-blinded-orders/v2"
DECISION_AGGREGATION_CODE_ID_V3 = "forge/symmetric-mean-prompt-clustered/v3"
CALIBRATION_PLAN_HASH_DOMAIN = "forge/evaluation-plan/v1"

CONTROL_RULE_V1 = {
    "version": "forge/control-rule/v1",
    "min_prompts_per_kind": 32,
    "sensitivity_lower_min": 0.8,
    "self_false_alarm_upper_max": 0.15,
    "negative_axis_delta": -0.25,
    "self_delta_tolerance": 0.125,
    "max_order_disagreement": 0.25,
    "family_alpha": 0.05,
}
CONTROL_RULE_V2 = {
    **CONTROL_RULE_V1,
    "version": "forge/control-rule/v2",
    "quality_aggregation": QUALITY_AGGREGATION_V2,
    "decision_aggregation_code_id": DECISION_AGGREGATION_CODE_ID,
}
CONTROL_RULE = {
    **CONTROL_RULE_V1,
    "version": "forge/control-rule/v3",
    "quality_aggregation": QUALITY_AGGREGATION_V3,
    "decision_aggregation_code_id": DECISION_AGGREGATION_CODE_ID_V3,
}


def pair_values(
    row: dict, aggregation: str = QUALITY_AGGREGATION_V1
) -> tuple[dict[str, float], float]:
    """Recompute a frozen decision score from both blinded calls.

    Raw receipts retain their v1 mean in ``axes``. v2 derives a lower score only
    for calibration and eligibility, so one optimistic order cannot help a model.
    """

    if aggregation not in {QUALITY_AGGREGATION_V1, QUALITY_AGGREGATION_V2, QUALITY_AGGREGATION_V3}:
        raise ValueError("unknown quality aggregation")
    calls = row.get("calls")
    if not isinstance(calls, list) or len(calls) != 2:
        raise ValueError("two blinded judgments are required")
    if {call.get("reference_is_a") for call in calls} != {True, False}:
        raise ValueError("both A/B orientations are required")
    deltas = []
    for call in calls:
        if type(call["reference_is_a"]) is not bool or call.get("model") != "gpt-6-luna":
            raise ValueError("wrong judge identity or orientation")
        answer = call["answer"]
        if answer.get("uncertain_axes") != [] or set(answer["axes"]) != set(DEFAULT_AXES):
            raise ValueError("uncertain or missing quality axes")
        values = {}
        for axis, score in answer["axes"].items():
            if any(type(score.get(k)) is not int or not 0 <= score[k] <= 4 for k in ("a", "b")):
                raise ValueError("invalid ordinal quality score")
            values[axis] = (score["b"] - score["a"]) / 4 * (1 if call["reference_is_a"] else -1)
        deltas.append(values)
    mean_axes = {axis: (deltas[0][axis] + deltas[1][axis]) / 2 for axis in DEFAULT_AXES}
    disagreement = max(abs(deltas[0][axis] - deltas[1][axis]) for axis in DEFAULT_AXES)
    if mean_axes != row.get("axes") or disagreement != row.get("order_disagreement"):
        raise ValueError("derived quality differs from blinded judgments")
    if aggregation in {QUALITY_AGGREGATION_V1, QUALITY_AGGREGATION_V3}:
        return mean_axes, disagreement
    return {axis: min(deltas[0][axis], deltas[1][axis]) for axis in DEFAULT_AXES}, disagreement


def _control_rule(evidence: dict) -> tuple[dict, str]:
    rule = evidence.get("control_rule")
    if rule == CONTROL_RULE_V1:
        return CONTROL_RULE_V1, QUALITY_AGGREGATION_V1
    if rule == CONTROL_RULE_V2:
        return CONTROL_RULE_V2, QUALITY_AGGREGATION_V2
    if rule == CONTROL_RULE:
        return CONTROL_RULE, QUALITY_AGGREGATION_V3
    raise ValueError("control rule differs from frozen implementation")


def calibrate_controls(evidence: dict) -> dict:
    rule, aggregation = _control_rule(evidence)
    grouped = evidence.get("controls")
    if not isinstance(grouped, dict) or set(grouped) != set(CONTROL_AXES):
        raise ValueError("all control families are required")
    summaries, reasons = {}, []
    status = "PASS"
    raster_status = evidence.get("raster_integrity_status", "CLEAR")
    if raster_status not in {"CLEAR", "ANOMALY"}:
        raise ValueError("raster integrity status is invalid")
    if raster_status == "ANOMALY":
        status = "INCONCLUSIVE"
        reasons.append("RASTER_INTEGRITY_ANOMALY")
    alpha = rule["family_alpha"] / len(CONTROL_AXES)
    for kind, rows in grouped.items():
        if not isinstance(rows, list):
            raise ValueError("control rows must be a list")
        seen, successes, invalid, noninformative, disagreement_diagnostics = set(), 0, 0, 0, 0
        for row in rows:
            # Multiple seeds or copies of one source do not increase control sample size.
            prompt_id = row["prompt_id"]
            if prompt_id in seen:
                raise ValueError("duplicate control prompt")
            seen.add(prompt_id)
            source_hash, control_hash = (
                row.get("reference_pixels_sha256"),
                row.get("control_pixels_sha256"),
            )
            if (
                aggregation == QUALITY_AGGREGATION_V3
                and kind != "self"
                and isinstance(source_hash, str)
                and source_hash == control_hash
            ):
                # A transformation of an already static/identical sequence has no
                # valid negative label. It is neither a detection nor a miss.
                noninformative += 1
                continue
            try:
                axes, disagreement = pair_values(row, aggregation)
                if row.get("status") != "OK" or (
                    aggregation == QUALITY_AGGREGATION_V1
                    and disagreement > rule["max_order_disagreement"]
                ):
                    raise ValueError("unstable control judgment")
                disagreement_diagnostics += int(disagreement > rule["max_order_disagreement"])
                if kind == "self":
                    detected = max(abs(x) for x in axes.values()) > 0.125
                else:
                    detected = any(axes[axis] <= -0.25 for axis in CONTROL_AXES[kind])
                successes += int(detected)
            except (ValueError, KeyError, TypeError):
                invalid += 1
        count = len(rows)
        informative = count - noninformative
        lower = float(beta.ppf(alpha, successes, informative - successes + 1)) if successes else 0.0
        upper = (
            float(beta.ppf(1 - alpha, successes + 1, informative - successes))
            if successes < informative
            else 1.0
        )
        summary = {
            "n_prompts": count,
            "detections": successes,
            "invalid": invalid,
            "rate_lower": lower,
            "rate_upper": upper,
        }
        if aggregation == QUALITY_AGGREGATION_V3:
            summary.update(informative_prompts=informative, noninformative=noninformative)
        if aggregation in {QUALITY_AGGREGATION_V2, QUALITY_AGGREGATION_V3}:
            summary["order_disagreement_over_diagnostic_threshold"] = disagreement_diagnostics
        summaries[kind] = summary
        if informative < rule["min_prompts_per_kind"] or invalid:
            status = "INCONCLUSIVE" if status != "FAIL" else status
            reasons.append(f"{kind}: insufficient or unstable controls")
        if kind == "self":
            passing, failing = upper <= 0.15, lower > 0.15
        else:
            passing, failing = lower >= 0.8, upper < 0.8
        if failing:
            status = "FAIL"
            reasons.append(f"{kind}: evaluator fails sensitivity/specificity")
        elif not passing and status == "PASS":
            status = "INCONCLUSIVE"
            reasons.append(f"{kind}: confidence interval overlaps calibration threshold")
    result = {
        key: value
        for key, value in evidence.items()
        if key not in {"status", "summary", "reasons", "calibration_id"}
    }
    result.update(status=status, summary=summaries, reasons=reasons)
    result["calibration_id"] = hash_canonical("forge/calibration/v1", result)
    return result


def verify_calibration(calibration: dict, policy: dict, plan: dict) -> bool:
    try:
        rule, aggregation = _control_rule(calibration)
        expected = calibrate_controls(calibration)
        return (
            expected == calibration
            and calibration.get("schema") == "forge/auto-calibration/v1"
            and calibration.get("evaluator_id") == policy["evaluator_id"]
            and calibration.get("policy_id") == policy["policy_id"]
            and calibration.get("contract_hash") == plan["contract_hash"]
            and calibration.get("geometry") == plan["geometry"]
            and (
                "calibration_plan_hash" not in plan
                or calibration.get("calibration_plan_hash") == plan["calibration_plan_hash"]
            )
            and (
                (aggregation == QUALITY_AGGREGATION_V1 and "quality_aggregation" not in policy)
                or (
                    aggregation == QUALITY_AGGREGATION_V2
                    and policy.get("quality_aggregation") == aggregation
                    and rule["decision_aggregation_code_id"] == DECISION_AGGREGATION_CODE_ID
                )
                or (
                    aggregation == QUALITY_AGGREGATION_V3
                    and policy.get("quality_aggregation") == aggregation
                    and rule["decision_aggregation_code_id"] == DECISION_AGGREGATION_CODE_ID_V3
                )
            )
        )
    except (ValueError, KeyError, TypeError):
        return False


def run_controls(
    plan: dict,
    report_path: Path,
    out: Path,
    *,
    budget,
    env_file: Path | None,
    max_prompts: int | None = None,
    workers: int = 1,
) -> dict:
    from forge.evaluation.automatic_quality import default_policy, validate_golden_report
    from forge.evaluation.raster_integrity import RULE_ID as RASTER_INTEGRITY_RULE_ID
    from forge.evaluation.raster_integrity import check_frames
    from forge.evaluation.vision_judge import evaluator_id, evaluator_spec, judge_pair
    from forge.evaluation.vision_media import (
        degrade_frames,
        load_video,
        make_sheets,
        media_identity,
        select_detail_crops,
        select_detail_frames,
        select_transition_frames,
    )

    if out.exists():
        raise ValueError("control output already exists; reuse requires explicit reconciliation")
    if max_prompts is not None and (type(max_prompts) is not int or max_prompts < 2):
        raise ValueError("max_prompts must be at least two")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in [1, 8]")
    validate_golden_report(plan, report_path)
    report = json.loads(report_path.read_text())
    if report["contract_hash"] != plan["contract_hash"] or report["prompts"] != plan["prompts"]:
        raise ValueError("control source must match the prescribed calibration plan")
    geometry = report["geometry"]
    prescribed_geometry = {k: plan["contract"][k] for k in ("width", "height", "frames", "fps")}
    if geometry != prescribed_geometry:
        raise ValueError("control source geometry differs from the plan")
    prompts = plan["prompts"][:max_prompts]
    if len(prompts) < 2:
        raise ValueError("at least two different prompts needed for wrong-video controls")
    if len({p["properties"]["subject_class"] for p in prompts}) < 2:
        raise ValueError("wrong-video controls require different subject classes")
    observations = {(o["prompt_id"], o["seed"]): o for o in report["observations"]}
    raster_checks = {}
    for observation in report["observations"]:
        path = report_path.parent / observation["video"]
        if path.is_symlink() or path.resolve().parent != report_path.parent.resolve():
            raise ValueError("invalid control source video")
        source = load_video(path, geometry)
        if source.video_sha256 != observation["video_sha256"]:
            raise ValueError("control source hash differs from report")
        raster_checks.setdefault(
            source.video_sha256, check_frames(source.frames, source.video_sha256)
        )
    raster_status = (
        "ANOMALY"
        if any(check["status"] == "ANOMALY" for check in raster_checks.values())
        else "CLEAR"
    )
    evidence = {
        "schema": "forge/auto-calibration/v1",
        "campaign_id": plan.get("campaign_id", plan.get("round_id")),
        "evaluator_id": evaluator_id(),
        "evaluator": evaluator_spec(),
        "policy_id": default_policy()["policy_id"],
        "contract_hash": plan["contract_hash"],
        "geometry": geometry,
        "calibration_plan_hash": hash_canonical(CALIBRATION_PLAN_HASH_DOMAIN, plan),
        "panel_hash": hash_canonical("forge/panel-prompts/v1", plan["prompts"]),
        "raster_integrity_rule_id": RASTER_INTEGRITY_RULE_ID,
        "raster_integrity_status": raster_status,
        "raster_checks": raster_checks,
        "created_unix": time.time(),
        "control_rule": CONTROL_RULE,
        "controls": {kind: [] for kind in CONTROL_AXES},
    }
    out.parent.mkdir(parents=True, exist_ok=True)

    def persist(receipt: dict) -> None:
        tmp = out.with_suffix(".tmp")
        if tmp.exists():
            raise ValueError("control temporary output already exists")
        with tmp.open("x") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(out)

    receipt = calibrate_controls(evidence)
    persist(receipt)
    if raster_status == "ANOMALY":
        return receipt

    def loaded(prompt):
        item = observations[(prompt["prompt_id"], prompt["video_seeds"][0])]
        path = report_path.parent / item["video"]
        if (
            item["status"] != "OK"
            or path.is_symlink()
            or path.resolve().parent != report_path.parent.resolve()
        ):
            raise ValueError("invalid control source video")
        video = load_video(path, geometry)
        if video.video_sha256 != item["video_sha256"]:
            raise ValueError("control source hash differs from report")
        return video

    work = [(kind, index, prompt) for kind in CONTROL_AXES for index, prompt in enumerate(prompts)]

    def judge(work_item: tuple[str, int, dict]) -> tuple[tuple[str, int], dict]:
        kind, index, prompt = work_item
        source = loaded(prompt)
        if kind == "self":
            altered = source.frames
        elif kind == "wrong_video":
            wrong = next(
                item
                for item in prompts
                if item["prompt_id"] != prompt["prompt_id"]
                and item["properties"]["subject_class"] != prompt["properties"]["subject_class"]
            )
            altered = loaded(wrong).frames
        else:
            altered = degrade_frames(source.frames, kind, seed=index)
        detail_indices = select_detail_frames(len(source.frames))
        detail_crops = select_detail_crops(source.frames, detail_indices)
        transition_indices = select_transition_frames(len(source.frames))
        row = judge_pair(
            prompt["text"],
            make_sheets(
                source.frames,
                geometry["fps"],
                detail_indices=detail_indices,
                detail_crops=detail_crops,
                transition_indices=transition_indices,
            ),
            make_sheets(
                altered,
                geometry["fps"],
                detail_indices=detail_indices,
                detail_crops=detail_crops,
                transition_indices=transition_indices,
            ),
            budget=budget,
            env_file=env_file,
            pair_id=f"{prompt['prompt_id']}:{kind}",
        )
        row.update(
            prompt_id=prompt["prompt_id"],
            kind=kind,
            reference_video_sha256=source.video_sha256,
            reference_pixels_sha256=media_identity(source.frames),
            control_pixels_sha256=media_identity(altered),
            detail_frame_indices=list(detail_indices),
            detail_crop_regions=list(detail_crops),
            transition_frame_indices=list(transition_indices),
        )
        return (kind, index), row

    completed: dict[tuple[str, int], dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(work), workers):
            futures = [executor.submit(judge, item) for item in work[start : start + workers]]
            try:
                for future in as_completed(futures):
                    key, row = future.result()
                    completed[key] = row
                    for kind in CONTROL_AXES:
                        evidence["controls"][kind] = [
                            completed[(kind, index)]
                            for index in range(len(prompts))
                            if (kind, index) in completed
                        ]
                    receipt = calibrate_controls(evidence)
                    persist(receipt)
                    print(
                        json.dumps(
                            {
                                "control": key[0],
                                "prompt": key[1],
                                "status": row["status"],
                                "deltas": row["axes"],
                            }
                        ),
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    evidence["completed_unix"] = time.time()
    receipt = calibrate_controls(evidence)
    persist(receipt)
    return receipt
