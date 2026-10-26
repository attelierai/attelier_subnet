"""Fail-closed, provisional V1 admission rule for the public model race.

This is deliberately a small policy contract, not a calibrated perceptual
metric.  It can declare a submitted evidence set *eligible* for the separate
promotion workflow, but it never writes weights and makes no scientific claim
that the three VideoReward heads fully measure video quality.

The temporal guards are intentionally narrow metadata checks.  They only apply
where a frozen prompt tags the corresponding property; they are not a general
physics, diversity, or anti-exploit proof.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from forge.protocol.hashing import hash_canonical

SCHEMA = "forge/launch-policy/v1"
AXES = ("VQ", "MQ", "TA")
REFERENCES = ("king", "anchor")


def freeze_policy(version: int = 2) -> dict[str, Any]:
    """Return the fixed, explicitly provisional V1 thresholds and identity."""
    body = {
        "schema": SCHEMA,
        "version": version,
        "status": "PROVISIONAL_NOT_PERCEPTUALLY_CALIBRATED",
        "sample_counts": {"quality_prompts": 16, "timing_prompts": 8},
        "speed": {"minimum_ratio": 1.10, "confidence": 0.95, "timing_order": "R,C,C,R"},
        "quality_margins": {"VQ": 0.35, "MQ": 0.20, "TA": 0.20},
        "temporal_guards": {
            "exact_freeze": {"candidate_frozen_fraction": 0.95, "reference_below": 0.95},
            "steady_lighting": {"add": 2.0, "multiple": 3.0, "floor": 0.1},
            "continuous_motion": {"add": 1.5, "multiple": 4.0, "floor": 0.1},
        },
        "limits": [
            "No human-label calibration is asserted.",
            "No aggregate or sum of VideoReward heads is used.",
            "Temporal guards are only applicable to explicitly tagged prompts.",
            "PROMOTE_ELIGIBLE never activates weights.",
        ],
    }
    if version == 2:
        body["temporal_guards"]["near_freeze"] = {
            "candidate_temporal_delta_max": 0.1 / 255,
            "candidate_reference_ratio_max": 0.01,
            "reference_temporal_delta_floor": 0.1 / 255,
        }
    elif version != 1:
        raise ValueError("unsupported launch policy revision")
    return {**body, "policy_id": hash_canonical("forge/launch-policy/v1", body)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".launch-", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _plan_body(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "plan_id"}


def _check_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("invalid launch policy plan schema")
    policy = plan.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("version") not in {1, 2}
        or policy != freeze_policy(policy["version"])
        or plan.get("plan_id") != hash_canonical("forge/launch-policy-plan/v1", _plan_body(plan))
    ):
        raise ValueError("changed policy or plan")
    models = plan.get("models")
    if not isinstance(models, dict) or set(models) != {"candidate", "king", "anchor"}:
        raise ValueError("plan models are invalid")
    for name, fingerprint in models.items():
        _sha(fingerprint, f"models.{name}")
    if models["candidate"] == models["king"]:
        raise ValueError("candidate must differ from the current king")
    evaluator, runtime = plan.get("evaluator"), plan.get("runtime")
    if not isinstance(evaluator, dict) or set(evaluator) != {"evaluator_id", "spec_hash"}:
        raise ValueError("plan evaluator is invalid")
    runtime_keys = {"runtime_id", "hardware_id", "dependency_lock_hash"}
    if not isinstance(runtime, dict) or set(runtime) != runtime_keys:
        raise ValueError("plan runtime is invalid")
    for group_name, group in (("evaluator", evaluator), ("runtime", runtime)):
        for key, value in group.items():
            _sha(value, f"{group_name}.{key}")
    geometry = plan.get("geometry")
    if (
        not isinstance(geometry, dict)
        or set(geometry) != {"width", "height", "frames", "fps"}
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in geometry.values())
    ):
        raise ValueError("plan geometry is invalid")
    counts = policy["sample_counts"]
    prompts = plan.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != counts["quality_prompts"]:
        raise ValueError("launch V1 requires exactly 16 prompts")
    ids = set()
    static = motion = 0
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("prompt must be an object")
        required = {"prompt_id", "text", "seed", "kind", "tags"}
        if (
            set(prompt) != required
            or not isinstance(prompt["prompt_id"], str)
            or not prompt["prompt_id"]
        ):
            raise ValueError("prompt identity is invalid")
        if (
            prompt["prompt_id"] in ids
            or not isinstance(prompt["text"], str)
            or not prompt["text"].strip()
        ):
            raise ValueError("prompt ids/text are invalid")
        if (
            isinstance(prompt["seed"], bool)
            or not isinstance(prompt["seed"], int)
            or not 0 <= prompt["seed"] < 2**63
        ):
            raise ValueError("prompt seed is invalid")
        allowed_tags = {"motion_required", "steady_lighting", "continuous_motion"}
        if (
            prompt["kind"] not in {"static", "motion"}
            or not isinstance(prompt["tags"], list)
            or any(tag not in allowed_tags for tag in prompt["tags"])
            or len(set(prompt["tags"])) != len(prompt["tags"])
        ):
            raise ValueError("prompt kind/tags are invalid")
        ids.add(prompt["prompt_id"])
        static += prompt["kind"] == "static"
        motion += prompt["kind"] == "motion"
    if static < 4 or motion < 4:
        raise ValueError("launch V1 needs at least four static and four motion prompts")
    timing = plan.get("timing_prompt_ids")
    if (
        not isinstance(timing, list)
        or len(timing) != counts["timing_prompts"]
        or len(set(timing)) != counts["timing_prompts"]
        or set(timing) - ids
    ):
        raise ValueError("launch V1 requires eight unique timing prompts")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 16:
        raise ValueError("plan assignments are invalid")
    assigned = set()
    for assignment in assignments:
        assignment_keys = {"prompt_id", "candidate", "king", "anchor"}
        if not isinstance(assignment, dict) or set(assignment) != assignment_keys:
            raise ValueError("assignment fields are invalid")
        prompt_id = assignment["prompt_id"]
        if (
            prompt_id not in ids
            or prompt_id in assigned
            or any(assignment[name] != models[name] for name in ("candidate", "king", "anchor"))
        ):
            raise ValueError("assignment does not bind the frozen models")
        assigned.add(prompt_id)


def freeze(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze a signed-by-hash plan before any generation or measurement."""
    if not isinstance(config, dict):
        raise ValueError("freeze config must be an object")
    if config.get("launch_policy_version") == 2 or config.get("policy_schema") == V2_SCHEMA:
        return freeze_v2(config)
    body = {
        "schema": SCHEMA,
        "policy": freeze_policy(),
        "models": config.get("models"),
        "evaluator": config.get("evaluator"),
        "runtime": config.get("runtime"),
        "geometry": config.get("geometry"),
        "prompts": config.get("prompts"),
        "timing_prompt_ids": config.get("timing_prompt_ids"),
        "assignments": config.get("assignments"),
    }
    result = {**body, "plan_id": hash_canonical("forge/launch-policy-plan/v1", body)}
    _check_plan(result)
    return result


def _lower(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("at least two prompt observations are required")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = float(student_t.ppf(0.95, df=len(values) - 1))
    return {
        "n_prompts": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "lower_ci": mean - critical * standard_error,
        "confidence": 0.95,
    }


def _scores(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(AXES):
        raise ValueError(f"{field} must provide exactly VQ/MQ/TA")
    return {axis: _finite(value[axis], f"{field}.{axis}") for axis in AXES}


def _diagnostics(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result = {}
    for key, number in value.items():
        if key == "flicker_guard_v1":
            result[key] = number
        else:
            result[key] = _finite(number, f"{field}.{key}")
    return result


def _quality_row(
    row: dict[str, Any], plan: dict[str, Any]
) -> tuple[tuple[str, str], dict[str, Any]]:
    required = {
        "type",
        "plan_id",
        "prompt_id",
        "reference",
        "model_ids",
        "evaluator_id",
        "geometry",
        "reference_result",
        "candidate_result",
    }
    if set(row) != required or row["type"] != "quality" or row["plan_id"] != plan["plan_id"]:
        raise ValueError("quality row has unexpected fields or plan")
    prompt_id, reference = row["prompt_id"], row["reference"]
    if reference not in REFERENCES or prompt_id not in {p["prompt_id"] for p in plan["prompts"]}:
        raise ValueError("quality row prompt/reference is foreign")
    if (
        row["evaluator_id"] != plan["evaluator"]["evaluator_id"]
        or row["geometry"] != plan["geometry"]
    ):
        raise ValueError("quality evaluator or geometry changed")
    expected = {"reference": plan["models"][reference], "candidate": plan["models"]["candidate"]}
    if row["model_ids"] != expected:
        raise ValueError("quality row binds the wrong models")
    prompt = next(item for item in plan["prompts"] if item["prompt_id"] == prompt_id)
    required_diagnostics = {
        "motion_required": "frozen_fraction",
        "steady_lighting": "luma_acceleration_median",
        "continuous_motion": "centered_transition_median",
    }
    parsed = {}
    for name in ("reference_result", "candidate_result"):
        item = row[name]
        if not isinstance(item, dict) or set(item) != {"video_sha256", "scores", "diagnostics"}:
            raise ValueError(f"{name} is invalid")
        diagnostics = _diagnostics(item["diagnostics"], f"{name}.diagnostics")
        missing = [
            required_diagnostics[tag]
            for tag in prompt["tags"]
            if required_diagnostics[tag] not in diagnostics
        ]
        if (
            plan["policy"]["version"] >= 2
            and "motion_required" in prompt["tags"]
            and "temporal_delta" not in diagnostics
        ):
            missing.append("temporal_delta")
        if missing:
            raise ValueError(f"{name} lacks applicable diagnostics: {', '.join(missing)}")
        parsed[name] = {
            "video_sha256": _sha(item["video_sha256"], f"{name}.video_sha256"),
            "scores": _scores(item["scores"], f"{name}.scores"),
            "diagnostics": diagnostics,
        }
    return (prompt_id, reference), parsed


def _timing_row(row: dict[str, Any], plan: dict[str, Any]) -> tuple[str, list[float]]:
    required = {
        "type",
        "plan_id",
        "prompt_id",
        "model_ids",
        "runtime_id",
        "hardware_id",
        "geometry",
        "times",
    }
    if set(row) != required or row["type"] != "timing" or row["plan_id"] != plan["plan_id"]:
        raise ValueError("timing row has unexpected fields or plan")
    if row["prompt_id"] not in plan["timing_prompt_ids"] or row["geometry"] != plan["geometry"]:
        raise ValueError("timing row prompt/geometry is foreign")
    if (
        row["runtime_id"] != plan["runtime"]["runtime_id"]
        or row["hardware_id"] != plan["runtime"]["hardware_id"]
    ):
        raise ValueError("timing runtime/hardware changed")
    expected_models = {
        "reference": plan["models"]["king"],
        "candidate": plan["models"]["candidate"],
    }
    if row["model_ids"] != expected_models:
        raise ValueError("timing row binds the wrong models")
    times = row["times"]
    if not isinstance(times, list) or len(times) != 4:
        raise ValueError("timing must have R,C,C,R measurements")
    return row["prompt_id"], [_finite(value, "timing", positive=True) for value in times]


def _temporal(
    prompt: dict[str, Any],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    policy: dict[str, Any],
) -> list[str]:
    tags = set(prompt["tags"])
    ref, cand = reference["diagnostics"], candidate["diagnostics"]
    reasons: list[str] = []
    if (
        "motion_required" in tags
        and "frozen_fraction" in ref
        and "frozen_fraction" in cand
        and cand["frozen_fraction"] >= 0.95
        and ref["frozen_fraction"] < 0.95
    ):
        reasons.append("exact_freeze")
    near = policy["temporal_guards"].get("near_freeze")
    if (
        near
        and "motion_required" in tags
        and (
            ref["temporal_delta"] > near["reference_temporal_delta_floor"]
            and cand["temporal_delta"] <= near["candidate_temporal_delta_max"]
            and cand["temporal_delta"]
            <= near["candidate_reference_ratio_max"] * ref["temporal_delta"]
        )
    ):
        reasons.append("near_freeze")
    for tag, metric, add, multiple in (
        ("steady_lighting", "luma_acceleration_median", 2.0, 3.0),
        ("continuous_motion", "centered_transition_median", 1.5, 4.0),
    ):
        if (
            tag in tags
            and metric in ref
            and metric in cand
            and cand[metric] > ref[metric] + add
            and cand[metric] > multiple * max(ref[metric], 0.1)
        ):
            reasons.append(tag + "_temporal_violation")
    return reasons


def _evaluate_v1(
    plan: dict[str, Any], rows: list[dict[str, Any]] | dict[str, Any]
) -> dict[str, Any]:
    """Fail closed: complete matrices are required even when a failure is obvious."""
    try:
        _check_plan(plan)
        supplied = rows["rows"] if isinstance(rows, dict) and set(rows) == {"rows"} else rows
        if not isinstance(supplied, list):
            raise ValueError("rows must be a list or {rows: [...]}")
        qualities: dict[tuple[str, str], dict[str, Any]] = {}
        timings: dict[str, list[float]] = {}
        for row in supplied:
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            if row.get("type") == "quality":
                key, parsed = _quality_row(row, plan)
                if key in qualities:
                    raise ValueError("duplicate quality row")
                qualities[key] = parsed
            elif row.get("type") == "timing":
                key, parsed = _timing_row(row, plan)
                if key in timings:
                    raise ValueError("duplicate timing row")
                timings[key] = parsed
            else:
                raise ValueError("foreign row type")
        prompt_ids = {p["prompt_id"] for p in plan["prompts"]}
        expected_quality = {(pid, ref) for pid in prompt_ids for ref in REFERENCES}
        if set(qualities) != expected_quality or set(timings) != set(plan["timing_prompt_ids"]):
            raise ValueError("quality or timing matrix is incomplete")
        for prompt_id in prompt_ids:
            king, anchor = qualities[(prompt_id, "king")], qualities[(prompt_id, "anchor")]
            if king["candidate_result"] != anchor["candidate_result"]:
                raise ValueError("candidate bytes/scores differ between king and anchor rows")
        log_speed_values = [
            math.log(
                statistics.median((values[0], values[3]))
                / statistics.median((values[1], values[2]))
            )
            for values in timings.values()
        ]
        speed_estimate = _lower(log_speed_values)
        speed = {
            "n_prompts": speed_estimate["n_prompts"],
            "ratio": math.exp(speed_estimate["mean"]),
            "lower_ci": math.exp(speed_estimate["lower_ci"]),
            "standard_error_log": speed_estimate["standard_error"],
            "confidence": speed_estimate["confidence"],
            "minimum_ratio": 1.10,
        }
        quality: dict[str, dict[str, dict[str, float | int]]] = {}
        for reference in REFERENCES:
            quality[reference] = {}
            for axis in AXES:
                deltas = [
                    qualities[(pid, reference)]["candidate_result"]["scores"][axis]
                    - qualities[(pid, reference)]["reference_result"]["scores"][axis]
                    for pid in sorted(prompt_ids)
                ]
                estimate = _lower(deltas)
                quality[reference][axis] = {
                    "n_prompts": estimate["n_prompts"],
                    "mean_delta": estimate["mean"],
                    "lower_ci": estimate["lower_ci"],
                    "standard_error": estimate["standard_error"],
                    "confidence": estimate["confidence"],
                    "margin": plan["policy"]["quality_margins"][axis],
                }
        temporal_violations = []
        prompt_map = {prompt["prompt_id"]: prompt for prompt in plan["prompts"]}
        for prompt_id in sorted(prompt_ids):
            candidate = qualities[(prompt_id, "king")]["candidate_result"]
            for reference in REFERENCES:
                reference_result = qualities[(prompt_id, reference)]["reference_result"]
                for violation in _temporal(
                    prompt_map[prompt_id], reference_result, candidate, plan["policy"]
                ):
                    temporal_violations.append(
                        {"prompt_id": prompt_id, "reference": reference, "violation": violation}
                    )
        failure_reasons = []
        inconclusive_reasons = []
        if speed["ratio"] < 1.10:
            failure_reasons.append("mean speed ratio is below 1.10")
        elif speed["lower_ci"] < 1.10:
            inconclusive_reasons.append("speed lower 95% confidence bound is below 1.10")
        for reference in REFERENCES:
            for axis in AXES:
                result = quality[reference][axis]
                if result["mean_delta"] < -result["margin"]:
                    failure_reasons.append(f"{reference} {axis} mean exceeds allowed loss")
                elif result["lower_ci"] < -result["margin"]:
                    inconclusive_reasons.append(
                        f"{reference} {axis} lower 95% confidence bound exceeds allowed loss"
                    )
        if temporal_violations:
            failure_reasons.append("applicable provisional temporal guard failed")
        status = (
            "FAIL"
            if failure_reasons
            else "INCONCLUSIVE"
            if inconclusive_reasons
            else "PROMOTE_ELIGIBLE"
        )
        return {
            "schema": "forge/launch-policy-result/v1",
            "plan_id": plan["plan_id"],
            "policy_id": plan["policy"]["policy_id"],
            "status": status,
            "promotion": "CONDITIONAL_ON_V1_POLICY_ONLY"
            if status == "PROMOTE_ELIGIBLE"
            else "NOT_ELIGIBLE",
            "weights": "NOT_ACTIVATED",
            "speed": speed,
            "quality": quality,
            "temporal_violations": temporal_violations,
            "reasons": failure_reasons + inconclusive_reasons,
            "limits": plan["policy"]["limits"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
        return {
            "schema": "forge/launch-policy-result/v1",
            "plan_id": plan_id,
            "status": "INCOMPLETE",
            "promotion": "NOT_ELIGIBLE",
            "weights": "NOT_ACTIVATED",
            "reasons": [str(exc)],
        }


# V2 is intentionally adjacent to V1 rather than a mutation of it.  Historical
# plans retain their schema, policy ID and replay code above.
V2_SCHEMA = "forge/launch-policy/v2"


def freeze_policy_v2() -> dict[str, Any]:
    body = {
        "schema": V2_SCHEMA,
        "version": 2,
        "status": "PROVISIONAL_REQUIRES_REAL_GPU_QUALIFICATION",
        "sample_counts": {
            "quality_prompts": 32,
            "timing_prompts": 8,
            "diversity_prompts": 4,
            "diversity_seeds": 3,
        },
        "speed": {"minimum_ratio": 1.10, "confidence": 0.95, "timing_order": "R,C,C,R"},
        "quality_margins": {"VQ": 0.35, "MQ": 0.20, "TA": 0.20},
        "limits": [
            "Each prompt, not each diversity seed, is one independent unit.",
            "VQ, MQ and TA pass separately against king and anchor.",
            "Synthetic controls and unknown GPU signals never qualify this policy.",
            "PASS is only locally promotion-eligible and never activates weights.",
        ],
    }
    return {**body, "policy_id": hash_canonical(V2_SCHEMA, body)}


def _v2_body(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "plan_id"}


def _check_plan_v2(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != V2_SCHEMA:
        raise ValueError("invalid launch policy V2 plan schema")
    if plan.get("policy") != freeze_policy_v2() or plan.get("plan_id") != hash_canonical(
        "forge/launch-policy-plan/v2", _v2_body(plan)
    ):
        raise ValueError("changed V2 policy or plan")
    for name in ("models", "evaluator", "runtime", "geometry"):
        if not isinstance(plan.get(name), dict):
            raise ValueError(f"V2 plan {name} is invalid")
    if (
        set(plan["models"]) != {"candidate", "king", "anchor"}
        or plan["models"]["candidate"] == plan["models"]["king"]
    ):
        raise ValueError("V2 model bindings are invalid")
    for value in plan["models"].values():
        _sha(value, "V2 model fingerprint")
    if set(plan["evaluator"]) != {"evaluator_id", "spec_hash"}:
        raise ValueError("V2 evaluator binding is invalid")
    for value in plan["evaluator"].values():
        _sha(value, "V2 evaluator binding")
    if set(plan["runtime"]) != {"runtime_id", "hardware_id", "dependency_lock_hash"}:
        raise ValueError("V2 runtime binding is invalid")
    for value in plan["runtime"].values():
        _sha(value, "V2 runtime binding")
    if set(plan["geometry"]) != {"width", "height", "frames", "fps"}:
        raise ValueError("V2 geometry is invalid")
    if plan["geometry"] != {"width": 832, "height": 480, "frames": 81, "fps": 16}:
        raise ValueError("V2 supports only the frozen Wan 832x480/81/16 geometry")
    prompts = plan.get("prompts")
    required = {"prompt_id", "text", "seed", "kind", "family", "split", "combination_id", "tags"}
    if (
        not isinstance(prompts, list)
        or len(prompts) != 32
        or any(not isinstance(p, dict) or set(p) != required for p in prompts)
    ):
        raise ValueError("V2 requires exactly 32 complete prompts")
    ids, combinations, family_counts = set(), set(), {}
    static = motion = 0
    for prompt in prompts:
        if (
            not isinstance(prompt["prompt_id"], str)
            or prompt["prompt_id"] in ids
            or not isinstance(prompt["text"], str)
            or not prompt["text"].strip()
            or type(prompt["seed"]) is not int
            or not 0 <= prompt["seed"] < 2**63
            or prompt["kind"] not in {"static", "motion"}
            or prompt["split"] != "confirmation"
            or not isinstance(prompt["family"], str)
            or not isinstance(prompt["combination_id"], str)
            or prompt["combination_id"] in combinations
            or not isinstance(prompt["tags"], list)
        ):
            raise ValueError("V2 prompt fields are invalid")
        ids.add(prompt["prompt_id"])
        combinations.add(prompt["combination_id"])
        family_counts[prompt["family"]] = family_counts.get(prompt["family"], 0) + 1
        static += prompt["kind"] == "static"
        motion += prompt["kind"] == "motion"
    if len(family_counts) != 4 or set(family_counts.values()) != {8} or (static, motion) != (8, 24):
        raise ValueError("V2 panel must be four families of eight, 8 static/24 motion")
    timing, diversity = plan.get("timing_prompt_ids"), plan.get("diversity_prompt_ids")
    if (
        not isinstance(timing, list)
        or len(timing) != 8
        or len(set(timing)) != 8
        or set(timing) - ids
    ):
        raise ValueError("V2 requires eight ABBA timing prompts")
    if (
        not isinstance(diversity, list)
        or len(diversity) != 4
        or len(set(diversity)) != 4
        or set(diversity) - ids
    ):
        raise ValueError("V2 requires four diversity prompts")
    qualification = plan.get("evaluator_qualification")
    if (
        not isinstance(qualification, dict)
        or qualification.get("schema") != "forge/evaluator-qualification/v1"
        or qualification.get("qualification")
        not in {"QUALIFIED", "NOT_QUALIFIED", "FAILED", "INCONCLUSIVE", "INCOMPLETE"}
        or not isinstance(qualification.get("manifest_id"), str)
        or not isinstance(qualification.get("receipt_id"), str)
        or qualification.get("evaluator_id") != plan["evaluator"]["evaluator_id"]
    ):
        raise ValueError("V2 evaluator qualification binding is invalid")
    if qualification.get("qualification") == "QUALIFIED" and not isinstance(
        qualification.get("guard_thresholds"), dict
    ):
        raise ValueError("V2 qualified receipt lacks decoded-video guard thresholds")


def policy_for(plan: Mapping[str, Any]) -> str:
    """Return the explicit replay path; callers must never infer policy by shape."""
    if plan.get("schema") == SCHEMA:
        return "v1"
    if plan.get("schema") == V2_SCHEMA:
        return "v2"
    raise ValueError("unknown launch policy schema")


def diversity_seeds(plan: Mapping[str, Any], prompt_id: str) -> list[int]:
    """Return the one frozen seed and two deterministic diversity draws."""
    if plan.get("schema") != V2_SCHEMA or not isinstance(prompt_id, str):
        raise ValueError("diversity seeds require a V2 policy plan and prompt id")
    prompt = next(
        (item for item in plan.get("prompts", []) if item.get("prompt_id") == prompt_id), None
    )
    if not isinstance(prompt, Mapping) or prompt_id not in plan.get("diversity_prompt_ids", []):
        raise ValueError("diversity prompt is not frozen in the V2 plan")
    base = prompt.get("seed")
    if type(base) is not int or not 0 <= base < 2**63:
        raise ValueError("frozen diversity base seed is invalid")
    seeds = [base]
    for domain in ("forge/launch-v2-diversity-seed-a/v1", "forge/launch-v2-diversity-seed-b/v1"):
        seeds.append(int(hash_canonical(domain, plan["plan_id"], prompt_id)[:16], 16) % (2**63))
    if len(set(seeds)) != 3:
        raise ValueError("derived diversity seeds collide")
    return seeds


def freeze_v2(config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema": V2_SCHEMA,
        "policy": freeze_policy_v2(),
        "models": config.get("models"),
        "evaluator": config.get("evaluator"),
        "runtime": config.get("runtime"),
        "geometry": config.get("geometry"),
        "prompts": config.get("prompts"),
        "timing_prompt_ids": config.get("timing_prompt_ids"),
        "diversity_prompt_ids": config.get("diversity_prompt_ids"),
        "evaluator_qualification": config.get("evaluator_qualification"),
    }
    plan = {**body, "plan_id": hash_canonical("forge/launch-policy-plan/v2", body)}
    _check_plan_v2(plan)
    return plan


def validate_plan(
    plan: Mapping[str, Any], *, expected_entropy_hash: str, expected_round_id: str
) -> None:
    """Recompute a V2 public panel and its two deterministic subsets.

    The launch service calls this both before persisting a plan and while
    replaying it.  It is deliberately unusable for V1: V1 keeps its existing
    `launch_panel` verifier and historical policy identity.
    """
    if policy_for(plan) != "v2":
        raise ValueError("validate_plan only accepts forge/launch-policy/v2")
    _check_plan_v2(dict(plan))
    from forge.evaluation.synthetic_prompts import launch_panel_v2

    expected = launch_panel_v2(expected_entropy_hash, round_id=expected_round_id)
    if plan.get("prompts") != expected:
        raise ValueError("V2 prompts differ from the deterministic confirmation panel")
    ordered = sorted(
        (prompt["prompt_id"] for prompt in expected),
        key=lambda value: hash_canonical(
            "forge/launch-v2-timing/v1", expected_entropy_hash, expected_round_id, value
        ),
    )
    if plan.get("timing_prompt_ids") != ordered[:8]:
        raise ValueError("V2 timing prompts differ from deterministic ABBA selection")
    # One diversity item per broad family avoids presenting four seeds from a
    # single semantic family as four independent diversity checks.
    diversity = []
    for family in sorted({prompt["family"] for prompt in expected}):
        candidates = [prompt["prompt_id"] for prompt in expected if prompt["family"] == family]
        diversity.append(
            min(
                candidates,
                key=lambda value: hash_canonical(
                    "forge/launch-v2-diversity/v1",
                    expected_entropy_hash,
                    expected_round_id,
                    family,
                    value,
                ),
            )
        )
    if plan.get("diversity_prompt_ids") != diversity:
        raise ValueError("V2 diversity prompts differ from deterministic family selection")


def _v2_result(
    plan: dict[str, Any], status: str, reasons: list[str], **evidence: Any
) -> dict[str, Any]:
    action = {
        "PASS": "PROMOTION_ELIGIBLE",
        "FAIL": "KEEP",
        "INCONCLUSIVE": "KEEP",
        "INCOMPLETE": "NO_VOTE",
    }[status]
    return {
        "schema": "forge/launch-policy-result/v2",
        "plan_id": plan.get("plan_id"),
        "policy_id": plan.get("policy", {}).get("policy_id"),
        "status": status,
        "action": action,
        "weights": "NOT_ACTIVATED",
        "reasons": reasons,
        **evidence,
    }


def _v2_decoded_guards(
    reference: Mapping[str, float], candidate: Mapping[str, float], thresholds: Mapping[str, Any]
) -> list[str]:
    """Use the exact all-frame detector exercised by qualification controls."""
    from forge.evaluation.evaluator_qualification import _decoded_guard_hits

    return sorted(_decoded_guard_hits(reference, candidate, thresholds))


def evaluate_v2(
    plan: dict[str, Any], rows: list[dict[str, Any]] | dict[str, Any]
) -> dict[str, Any]:
    try:
        _check_plan_v2(plan)
        if plan["evaluator_qualification"]["qualification"] != "QUALIFIED":
            return _v2_result(plan, "INCOMPLETE", ["evaluator lacks real-GPU qualified receipt"])
        supplied = rows["rows"] if isinstance(rows, dict) and set(rows) == {"rows"} else rows
        if not isinstance(supplied, list):
            raise ValueError("rows must be a list or {rows: [...]}")
        prompts = {p["prompt_id"]: p for p in plan["prompts"]}
        quality, timing, diversity = {}, {}, {}
        for row in supplied:
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            row_type = row.get("type")
            if row_type == "quality":
                required = {
                    "type",
                    "plan_id",
                    "prompt_id",
                    "reference",
                    "model_ids",
                    "evaluator_id",
                    "geometry",
                    "reference_result",
                    "candidate_result",
                }
                if (
                    set(row) != required
                    or row["plan_id"] != plan["plan_id"]
                    or row["prompt_id"] not in prompts
                    or row["reference"] not in REFERENCES
                ):
                    raise ValueError("V2 quality row is invalid")
                key = (row["prompt_id"], row["reference"])
                if (
                    key in quality
                    or row["model_ids"]
                    != {
                        "reference": plan["models"][row["reference"]],
                        "candidate": plan["models"]["candidate"],
                    }
                    or row["evaluator_id"] != plan["evaluator"]["evaluator_id"]
                    or row["geometry"] != plan["geometry"]
                ):
                    raise ValueError("V2 quality row binding changed")
                parsed = []
                for role in ("reference_result", "candidate_result"):
                    value = row[role]
                    if not isinstance(value, dict) or set(value) != {
                        "video_sha256",
                        "scores",
                        "diagnostics",
                    }:
                        raise ValueError("V2 quality result is invalid")
                    parsed.append(
                        {
                            "video_sha256": _sha(value["video_sha256"], "video hash"),
                            "scores": _scores(value["scores"], "scores"),
                            "diagnostics": _diagnostics(value["diagnostics"], "diagnostics"),
                        }
                    )
                quality[key] = tuple(parsed)
            elif row_type == "timing":
                if (
                    set(row) != {"type", "plan_id", "prompt_id", "times"}
                    or row["plan_id"] != plan["plan_id"]
                    or row["prompt_id"] not in plan["timing_prompt_ids"]
                    or row["prompt_id"] in timing
                    or not isinstance(row["times"], list)
                    or len(row["times"]) != 4
                ):
                    raise ValueError("V2 timing row is invalid")
                timing[row["prompt_id"]] = [
                    _finite(value, "timing", positive=True) for value in row["times"]
                ]
            elif row_type == "diversity":
                if (
                    set(row) != {"type", "plan_id", "prompt_id", "decoded_video_sha256"}
                    or row["plan_id"] != plan["plan_id"]
                    or row["prompt_id"] not in plan["diversity_prompt_ids"]
                    or row["prompt_id"] in diversity
                    or not isinstance(row["decoded_video_sha256"], list)
                    or len(row["decoded_video_sha256"]) != 3
                ):
                    raise ValueError("V2 diversity row is invalid")
                diversity[row["prompt_id"]] = [
                    _sha(value, "decoded video hash") for value in row["decoded_video_sha256"]
                ]
            else:
                raise ValueError("foreign V2 row type")
        expected_quality = {(pid, ref) for pid in prompts for ref in REFERENCES}
        if (
            set(quality) != expected_quality
            or set(timing) != set(plan["timing_prompt_ids"])
            or set(diversity) != set(plan["diversity_prompt_ids"])
        ):
            raise ValueError("V2 evidence matrix is incomplete")
        for prompt_id in prompts:
            if quality[(prompt_id, "king")][1] != quality[(prompt_id, "anchor")][1]:
                raise ValueError("candidate quality differs between references")
        speed_values = [
            math.log(
                statistics.median((value[0], value[3])) / statistics.median((value[1], value[2]))
            )
            for value in timing.values()
        ]
        speed_estimate = _lower(speed_values)
        speed = {
            "n_prompts": 8,
            "ratio": math.exp(speed_estimate["mean"]),
            "lower_ci": math.exp(speed_estimate["lower_ci"]),
            "confidence": 0.95,
            "minimum_ratio": 1.10,
        }
        quality_summary, failures, inconclusive, decoded_violations = {}, [], [], []
        for reference in REFERENCES:
            quality_summary[reference] = {}
            for axis in AXES:
                estimate = _lower(
                    [
                        quality[(pid, reference)][1]["scores"][axis]
                        - quality[(pid, reference)][0]["scores"][axis]
                        for pid in sorted(prompts)
                    ]
                )
                quality_summary[reference][axis] = {
                    "n_prompts": 32,
                    "mean_delta": estimate["mean"],
                    "lower_ci": estimate["lower_ci"],
                    "margin": plan["policy"]["quality_margins"][axis],
                    "confidence": 0.95,
                }
                if estimate["mean"] < -plan["policy"]["quality_margins"][axis]:
                    failures.append(f"{reference} {axis} mean exceeds allowed loss")
                elif estimate["lower_ci"] < -plan["policy"]["quality_margins"][axis]:
                    inconclusive.append(
                        f"{reference} {axis} lower confidence bound exceeds allowed loss"
                    )
        for prompt_id in prompts:
            candidate = quality[(prompt_id, "king")][1]["diagnostics"]
            for reference in REFERENCES:
                for guard in _v2_decoded_guards(
                    quality[(prompt_id, reference)][0]["diagnostics"],
                    candidate,
                    plan["evaluator_qualification"]["guard_thresholds"],
                ):
                    if (
                        guard == "blur"
                        and plan["evaluator_qualification"].get("qualification_version") == 2
                    ):
                        # Qualification v2 proves perceptual blur sensitivity at
                        # panel level.  A fixed high-frequency proxy is not a
                        # product veto for naturally soft valid video.
                        continue
                    if guard == "freeze" and "motion_required" not in prompts[prompt_id]["tags"]:
                        continue
                    if guard == "flicker" and "steady_lighting" not in prompts[prompt_id]["tags"]:
                        continue
                    decoded_violations.append(
                        {"prompt_id": prompt_id, "reference": reference, "guard": guard}
                    )
        if decoded_violations:
            failures.append("qualified decoded-video guard failed")
        copy_failures = [pid for pid, hashes in diversity.items() if len(set(hashes)) != 3]
        if copy_failures:
            failures.append("decoded diversity hashes show exact copied seeds")
        if speed["ratio"] < 1.10:
            failures.append("mean speed ratio is below 1.10")
        elif speed["lower_ci"] < 1.10:
            inconclusive.append("speed lower confidence bound is below 1.10")
        status = "FAIL" if failures else "INCONCLUSIVE" if inconclusive else "PASS"
        return _v2_result(
            plan,
            status,
            failures + inconclusive,
            speed=speed,
            quality=quality_summary,
            diversity={"n_prompts": 4, "copied_prompt_ids": copy_failures},
            decoded_violations=decoded_violations,
            limits=plan["policy"]["limits"],
        )
    except (KeyError, TypeError, ValueError) as error:
        return _v2_result(plan if isinstance(plan, dict) else {}, "INCOMPLETE", [str(error)])


def evaluate(plan: dict[str, Any], rows: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    return evaluate_v2(plan, rows) if policy_for(plan) == "v2" else _evaluate_v1(plan, rows)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_command = commands.add_parser("freeze")
    freeze_command.add_argument("--config", type=Path, required=True)
    freeze_command.add_argument("--out", type=Path, required=True)
    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--plan", type=Path, required=True)
    evaluate_command.add_argument("--rows", type=Path, required=True)
    evaluate_command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(_read(args.config))
    else:
        result = evaluate(_read(args.plan), _read(args.rows))
    _write_new(args.out, result)


if __name__ == "__main__":
    main()
