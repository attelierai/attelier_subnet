"""Fail-closed speed evidence gate for a fixed validator panel.

This module deliberately does not calibrate quality. It only compares two
already-calibrated validator reports that attest to exactly the same runtime,
hardware and rendered videos.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from forge.evaluation.scoring import ComparisonIncompleteError
from forge.evaluation.stats import Estimate, lower_confidence_bound
from forge.protocol.contract import ChallengeContract

SCHEMA = "forge/panel/v1"
MODEL_FAILURE_CAP = 0.05
_OK = "OK"
_MODEL_FAILURE = "MODEL_FAILURE"
_INCOMPLETE_STATUSES = {"INFRA_FAILURE", "JUDGE_FAILURE"}
_STATUSES = {_OK, _MODEL_FAILURE, *_INCOMPLETE_STATUSES}


class _Incomplete(ValueError):
    pass


def _result(status: str, reasons: list[str], **evidence: Any) -> dict[str, Any]:
    return {"status": status, "reasons": reasons, **evidence}


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _Incomplete(f"{field} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise _Incomplete(f"{field} must be positive")
    return result


def _sha256(value: Any, field: str) -> str:
    valid = isinstance(value, str) and len(value) == 64
    if not valid or any(char not in "0123456789abcdef" for char in value):
        raise _Incomplete(f"{field} must be a lowercase SHA-256")
    return value


def _estimate(value: Estimate) -> dict[str, float | int]:
    return {
        "n_prompts": value.n_prompts,
        "mean": value.mean,
        "standard_error": value.standard_error,
        "lower_bound": value.lower_bound,
        "alpha": value.alpha,
    }


def _panel(
    report: dict[str, Any],
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], set[tuple[str, int]]]:
    prompts = report.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise _Incomplete("prompts must be a non-empty list")
    panel: dict[str, tuple[str, tuple[int, ...]]] = {}
    slots: set[tuple[str, int]] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise _Incomplete("prompt must be an object")
        prompt_id = prompt.get("prompt_id")
        text = prompt.get("text")
        seeds = prompt.get("video_seeds")
        if not isinstance(prompt_id, str) or not prompt_id or not isinstance(text, str):
            raise _Incomplete("prompt_id and text are required")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise _Incomplete(f"prompt {prompt_id} has invalid video_seeds")
        seed_tuple = tuple(seeds)
        if len(set(seed_tuple)) != len(seed_tuple) or prompt_id in panel:
            raise _Incomplete("prompt ids and video seeds must be unique")
        panel[prompt_id] = (text, seed_tuple)
        slots.update((prompt_id, seed) for seed in seed_tuple)
    return panel, slots


def _observations(
    report: dict[str, Any],
    expected_slots: set[tuple[str, int]],
    contract: ChallengeContract,
    reference: bool,
    role: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], float]:
    values = report.get("observations")
    if not isinstance(values, list):
        raise _Incomplete(f"{role} observations must be a list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    model_failures = 0
    for item in values:
        if not isinstance(item, dict):
            raise _Incomplete(f"{role} observation must be an object")
        prompt_id, seed, status = item.get("prompt_id"), item.get("seed"), item.get("status")
        slot = (prompt_id, seed)
        if not isinstance(prompt_id, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise _Incomplete(f"{role} observation has an invalid slot")
        if slot in indexed:
            raise _Incomplete(f"{role} observation slots are duplicated")
        if status not in _STATUSES:
            raise _Incomplete(f"{role} observation {prompt_id}/{seed} has unknown status")
        if status in _INCOMPLETE_STATUSES:
            raise _Incomplete(f"{role} observation {prompt_id}/{seed} has {status}")
        if status == _MODEL_FAILURE:
            model_failures += 1
        else:
            _sha256(item.get("video_sha256"), f"{role} video_sha256")
            wall_time = _finite_number(
                item.get("wall_time_seconds"), f"{role} wall_time_seconds", positive=True
            )
            _finite_number(item.get("denoise_seconds"), f"{role} denoise_seconds", positive=True)
            peak_vram = item.get("peak_vram_bytes")
            forward_passes = item.get("forward_passes")
            steps = item.get("steps")
            if (
                isinstance(peak_vram, bool)
                or not isinstance(peak_vram, int)
                or peak_vram < 0
                or isinstance(forward_passes, bool)
                or not isinstance(forward_passes, int)
                or forward_passes <= 0
                or isinstance(steps, bool)
                or not isinstance(steps, int)
                or steps <= 0
            ):
                raise _Incomplete(f"{role} observation has invalid resource evidence")
            if wall_time > contract.wall_time_limit_seconds:
                raise _Incomplete(f"{role} observation exceeds contract wall-time limit")
            if peak_vram > contract.vram_limit_bytes:
                raise _Incomplete(f"{role} observation exceeds contract VRAM limit")
            step_limit = contract.reference_steps if reference else contract.max_steps
            forward_limit = (
                2 * contract.reference_steps if reference else contract.max_forward_passes
            )
            if steps > step_limit:
                raise _Incomplete(f"{role} observation exceeds contract step limit")
            if forward_passes > forward_limit:
                raise _Incomplete(f"{role} observation exceeds contract forward-pass limit")
        indexed[slot] = item
    if set(indexed) != expected_slots:
        raise _Incomplete(f"{role} observation slots do not match the prescribed panel")
    return indexed, model_failures / len(expected_slots)


def _quality(
    report: dict[str, Any],
    observations: dict[tuple[str, int], dict[str, Any]],
    expected_slots: set[tuple[str, int]],
    expected_axes: frozenset[str],
    expected_evaluator_id: str,
    role: str,
) -> tuple[str, dict[tuple[str, int], dict[str, float]]]:
    quality = report.get("quality")
    if not isinstance(quality, dict) or quality.get("calibrated") is not True:
        raise _Incomplete(f"{role} quality is not calibrated")
    evaluator_id = quality.get("evaluator_id")
    values = quality.get("observations")
    if evaluator_id != expected_evaluator_id or not isinstance(values, list):
        raise _Incomplete(f"{role} quality evidence is invalid")
    indexed: dict[tuple[str, int], dict[str, float]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise _Incomplete(f"{role} quality observation must be an object")
        slot = (item.get("prompt_id"), item.get("seed"))
        if slot in indexed or slot not in expected_slots:
            raise _Incomplete(f"{role} quality slots do not match the prescribed panel")
        if item.get("video_sha256") != observations[slot].get("video_sha256"):
            raise _Incomplete(f"{role} quality video hash does not match rendered evidence")
        axes = item.get("axes")
        if (
            not isinstance(axes, dict)
            or set(axes) != expected_axes
            or any(not isinstance(axis, str) for axis in axes)
        ):
            raise _Incomplete(f"{role} quality axes do not match the contract")
        checked = {
            axis: _finite_number(value, f"{role} quality {axis}") for axis, value in axes.items()
        }
        if any(value < 0 or value > 1 for value in checked.values()):
            raise _Incomplete(f"{role} quality axes must be in [0, 1]")
        indexed[slot] = checked
    if set(indexed) != expected_slots:
        raise _Incomplete(f"{role} quality slots do not match the prescribed panel")
    return evaluator_id, indexed


def _prompt_medians(
    reference: dict[tuple[str, int], dict[str, Any]],
    challenger: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for slot, reference_item in reference.items():
        challenger_item = challenger[slot]
        grouped[slot[0]].append(
            float(reference_item["wall_time_seconds"]) / float(challenger_item["wall_time_seconds"])
        )
    return {prompt_id: statistics.median(values) for prompt_id, values in grouped.items()}


def compare_evidence(
    reference_report: dict[str, Any],
    challenger_report: dict[str, Any],
    *,
    required_contract: ChallengeContract,
    expected_prompts: list[dict[str, Any]],
    expected_runtime_id: str,
    expected_geometry: Mapping[str, int],
    expected_evaluator_id: str,
    min_speedup: float | None = None,
    quality_margin: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare reports against trusted validator configuration.

    A speedup is the paired reference/challenger wall-time ratio, medianed by
    prompt and tested with a one-sided lower confidence bound. Quality uses
    paired prompt means for the average axis score and every reported axis.
    """

    try:
        if (
            not isinstance(required_contract, ChallengeContract)
            or not isinstance(expected_prompts, list)
            or not isinstance(expected_runtime_id, str)
            or not expected_runtime_id
            or not isinstance(expected_evaluator_id, str)
            or not expected_evaluator_id
            or not isinstance(expected_geometry, Mapping)
            or not expected_geometry
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in expected_geometry.values()
            )
            or (min_speedup is not None and min_speedup != required_contract.min_speedup)
            or quality_margin < 0
            or not 0 < alpha < 1
        ):
            raise ValueError("invalid gate thresholds")
        if not isinstance(reference_report, dict) or not isinstance(challenger_report, dict):
            raise _Incomplete("reports must be objects")
        if required_contract.protected_slices:
            raise _Incomplete("property-slice calibration is not implemented for the speed gate")
        if reference_report.get("admission") != "public_snapshot":
            raise _Incomplete("reference admission is not public_snapshot")
        speed_threshold = required_contract.min_speedup
        expected_contract_hash = required_contract.contract_hash()
        if (
            reference_report.get("contract_hash") != expected_contract_hash
            or challenger_report.get("contract_hash") != expected_contract_hash
        ):
            raise _Incomplete("reports do not match the trusted contract")
        if (
            reference_report.get("runtime_id") != expected_runtime_id
            or challenger_report.get("runtime_id") != expected_runtime_id
        ):
            raise _Incomplete("reports do not match the trusted runtime")
        expected_geometry_dict = dict(expected_geometry)
        if (
            reference_report.get("geometry") != expected_geometry_dict
            or challenger_report.get("geometry") != expected_geometry_dict
        ):
            raise _Incomplete("reports do not match the trusted geometry")
        _sha256(reference_report.get("manifest_hash"), "reference manifest_hash")
        _sha256(challenger_report.get("manifest_hash"), "challenger manifest_hash")
        if challenger_report.get("admission") != "public_snapshot":
            return _result("REJECT", ["challenger admission is not public_snapshot"])
        if challenger_report["manifest_hash"] == reference_report["manifest_hash"]:
            return _result("REJECT", ["challenger exactly duplicates the reference manifest"])
        reference_hardware = reference_report.get("hardware")
        challenger_hardware = challenger_report.get("hardware")
        if not isinstance(reference_hardware, dict) or not isinstance(challenger_hardware, dict):
            raise _Incomplete("reports require hardware evidence")
        reference_gpu = reference_hardware.get("gpu_name")
        challenger_gpu = challenger_hardware.get("gpu_name")
        if (
            not isinstance(reference_gpu, str)
            or not reference_gpu
            or reference_gpu != challenger_gpu
        ):
            raise _Incomplete("reports do not share hardware.gpu_name")
        if reference_report.get("schema") != SCHEMA or challenger_report.get("schema") != SCHEMA:
            raise _Incomplete(f"reports must declare schema {SCHEMA}")
        trusted_panel, expected_slots = _panel({"prompts": expected_prompts})
        reference_panel, reference_slots = _panel(reference_report)
        challenger_panel, challenger_slots = _panel(challenger_report)
        if (
            reference_panel != trusted_panel
            or challenger_panel != trusted_panel
            or reference_slots != expected_slots
            or challenger_slots != expected_slots
        ):
            raise _Incomplete("reports do not match the trusted prompt panel")
        reference, reference_failures = _observations(
            reference_report, expected_slots, required_contract, True, "reference"
        )
        challenger, challenger_failures = _observations(
            challenger_report, expected_slots, required_contract, False, "challenger"
        )
        if reference_failures:
            raise _Incomplete("reference has model failures")
        if challenger_failures:
            if challenger_failures > MODEL_FAILURE_CAP:
                return _result(
                    "REJECT",
                    [f"challenger model failure rate {challenger_failures:.6f} exceeds cap"],
                    challenger_failure_rate=challenger_failures,
                )
            raise _Incomplete("challenger model failures prevent paired evidence")
        reference_evaluator, reference_quality = _quality(
            reference_report,
            reference,
            expected_slots,
            frozenset(required_contract.axes),
            expected_evaluator_id,
            "reference",
        )
        challenger_evaluator, challenger_quality = _quality(
            challenger_report,
            challenger,
            expected_slots,
            frozenset(required_contract.axes),
            expected_evaluator_id,
            "challenger",
        )
        if reference_evaluator != challenger_evaluator:
            raise _Incomplete("reports use different calibrated evaluators")
        axes = set(next(iter(reference_quality.values())))
        if (
            not axes
            or any(set(item) != axes for item in reference_quality.values())
            or any(set(item) != axes for item in challenger_quality.values())
        ):
            raise _Incomplete("quality axis sets do not match")
        speed = lower_confidence_bound(list(_prompt_medians(reference, challenger).values()), alpha)
        quality_alpha = alpha / (len(axes) + 1)
        quality_values: dict[str, list[float]] = {"average": []} | {axis: [] for axis in axes}
        by_prompt: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"average": [], **{axis: [] for axis in axes}}
        )
        for slot, reference_axes in reference_quality.items():
            challenger_axes = challenger_quality[slot]
            by_prompt[slot[0]]["average"].append(
                sum(challenger_axes.values()) / len(axes) - sum(reference_axes.values()) / len(axes)
            )
            for axis in axes:
                by_prompt[slot[0]][axis].append(challenger_axes[axis] - reference_axes[axis])
        for prompt_values in by_prompt.values():
            for axis, values in prompt_values.items():
                quality_values[axis].append(sum(values) / len(values))
        quality = {
            name: lower_confidence_bound(values, quality_alpha)
            for name, values in quality_values.items()
        }
    except (_Incomplete, ComparisonIncompleteError, ValueError) as exc:
        return _result("INCOMPLETE", [str(exc)])

    reasons: list[str] = []
    if speed.lower_bound < speed_threshold:
        reasons.append(f"speed LCB {speed.lower_bound:.6f} is below {speed_threshold:.6f}")
    for slot, reference_item in reference.items():
        if challenger[slot]["forward_passes"] > reference_item["forward_passes"]:
            reasons.append(f"challenger uses more forward passes at {slot[0]}/{slot[1]}")
            break
    for name, value in quality.items():
        if value.lower_bound < -quality_margin:
            reasons.append(
                f"quality {name} LCB {value.lower_bound:.6f} is below -{quality_margin:.6f}"
            )
    evidence = {
        "speed": _estimate(speed),
        "quality": {name: _estimate(value) for name, value in sorted(quality.items())},
        "challenger_failure_rate": challenger_failures,
    }
    return _result("REJECT" if reasons else "PROMOTE_ELIGIBLE", reasons, **evidence)
