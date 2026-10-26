"""Historical diagnostic quality and timing comparisons."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from forge.evaluation.automatic_evidence import (
    AXES,
    QUALITY_SCHEMA,
    TIMING_SCHEMA,
    _contract,
    _EvidenceError,
    _number,
    _sha256,
)
from forge.evaluation.vision_calibration import (
    DECISION_AGGREGATION_CODE_ID,
    DECISION_AGGREGATION_CODE_ID_V3,
    QUALITY_AGGREGATION_V1,
    QUALITY_AGGREGATION_V2,
    QUALITY_AGGREGATION_V3,
)
from forge.protocol.hashing import hash_canonical


def _quality_comparisons(
    quality: Mapping[str, Any],
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    slots: set[tuple[str, int]],
    prompts: Mapping[str, Mapping[str, Any]],
    panel_hash: str,
    *,
    development: bool,
) -> tuple[dict[str, dict[str, Any]], str, bool]:
    if quality.get("schema") != QUALITY_SCHEMA:
        raise _EvidenceError("automatic quality schema is invalid")
    if quality.get("evaluator_id") != policy["evaluator_id"]:
        raise _EvidenceError("quality evaluator differs from policy")
    aggregation = policy.get("quality_aggregation", QUALITY_AGGREGATION_V1)
    if not development and aggregation in {QUALITY_AGGREGATION_V2, QUALITY_AGGREGATION_V3}:
        spec = quality.get("spec")
        expected_code = (
            DECISION_AGGREGATION_CODE_ID_V3
            if aggregation == QUALITY_AGGREGATION_V3
            else DECISION_AGGREGATION_CODE_ID
        )
        if (
            not isinstance(spec, Mapping)
            or spec.get("decision_aggregation_code_id") != expected_code
        ):
            raise _EvidenceError("quality evaluator does not bind decision aggregation code")
    if quality.get("panel_hash") != panel_hash or quality.get("contract_hash") != plan.get(
        "contract_hash"
    ):
        raise _EvidenceError("quality does not bind the frozen plan")
    if quality.get("geometry") != plan.get("geometry"):
        raise _EvidenceError("quality geometry differs from plan")
    entries = quality.get("comparisons")
    if not isinstance(entries, list):
        raise _EvidenceError("quality comparisons are required")
    parsed: dict[str, dict[str, Any]] = {}
    candidate_hash: str | None = None
    public_admission = True
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _EvidenceError("quality comparison must be an object")
        role = entry.get("reference_role")
        reference_hash = entry.get("reference_manifest_hash")
        challenger_hash = entry.get("candidate_manifest_hash")
        observations = entry.get("observations")
        if role not in {"golden", "incumbent"} or role in parsed:
            raise _EvidenceError("quality requires one golden and one incumbent comparison")
        _sha256(reference_hash, "quality reference manifest hash")
        _sha256(challenger_hash, "quality candidate manifest hash")
        admissions = {entry.get("reference_admission"), entry.get("candidate_admission")}
        allowed_admissions = (
            {"public_snapshot", "private_experiment"} if development else {"public_snapshot"}
        )
        if not admissions.issubset(allowed_admissions):
            raise _EvidenceError("quality comparison admission is invalid")
        is_public = admissions == {"public_snapshot"}
        if not is_public and not development:
            raise _EvidenceError("quality comparison contains a non-public artifact")
        public_admission &= is_public
        if candidate_hash is None:
            candidate_hash = challenger_hash
        if challenger_hash != candidate_hash or challenger_hash == reference_hash:
            raise _EvidenceError("quality candidate fingerprints are inconsistent or copied")
        if not isinstance(observations, list):
            raise _EvidenceError("quality observations are required")
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        candidate_videos: dict[tuple[str, int], str] = {}
        reference_videos: dict[tuple[str, int], str] = {}
        seen: set[tuple[str, int]] = set()
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise _EvidenceError("quality observation must be an object")
            slot = (observation.get("prompt_id"), observation.get("seed"))
            if slot not in slots or slot in seen or observation.get("status") != "OK":
                raise _EvidenceError("quality observations must cover each prescribed slot once")
            seen.add(slot)
            reference_video = _sha256(
                observation.get("reference_video_sha256"), "quality reference video hash"
            )
            candidate_video = _sha256(
                observation.get("candidate_video_sha256"), "quality candidate video hash"
            )
            values, diagnostic_values, disagreement, disagreement_by_axis, sign_flips = (
                _raw_quality(observation, aggregation)
            )
            if (
                aggregation == QUALITY_AGGREGATION_V1
                and disagreement > policy["max_order_disagreement"]
            ):
                raise _EvidenceError("quality order disagreement exceeds policy")
            # v2 retains its historical minimum. v3 uses the symmetric mean and
            # makes excessive orientation disagreement invalid evidence.
            average = sum(values.values()) / len(AXES)
            for name, value in {"average": average, **values}.items():
                grouped[slot[0]][name].append(value)
            grouped[slot[0]]["raw_mean_average"].append(sum(diagnostic_values.values()) / len(AXES))
            for axis, value in disagreement_by_axis.items():
                grouped[slot[0]][f"order_disagreement:{axis}"].append(value)
            grouped[slot[0]]["order_sign_flips"].append(float(sign_flips))
            reference_videos[slot] = reference_video
            candidate_videos[slot] = candidate_video
        if seen != slots:
            raise _EvidenceError("quality panel is incomplete")
        parsed[role] = {
            "reference_manifest_hash": reference_hash,
            "by_prompt": grouped,
            "reference_videos": reference_videos,
            "candidate_videos": candidate_videos,
        }
    if set(parsed) != {"golden", "incumbent"} or candidate_hash is None:
        raise _EvidenceError("quality requires golden and incumbent evidence")
    golden_candidate_videos = parsed["golden"]["candidate_videos"]
    if golden_candidate_videos != parsed["incumbent"]["candidate_videos"]:
        raise _EvidenceError("golden and incumbent comparisons used different candidate videos")
    return parsed, candidate_hash, public_admission


def _raw_quality(
    observation: Mapping[str, Any], aggregation: str
) -> tuple[dict[str, float], dict[str, float], float, dict[str, float], int]:
    calls = observation.get("calls")
    if not isinstance(calls, list) or len(calls) != 2:
        raise _EvidenceError("quality requires exactly two blind calls")
    deltas: list[dict[str, float]] = []
    orientations: set[bool] = set()
    response_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, Mapping) or type(call.get("reference_is_a")) is not bool:
            raise _EvidenceError("quality call orientation is invalid")
        if call.get("model") != "gpt-6-luna":
            raise _EvidenceError("quality call evaluator model differs")
        response_id = call.get("response_id")
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise _EvidenceError("quality call response identities are invalid")
        response_ids.add(response_id)
        answer = call.get("answer")
        if not isinstance(answer, Mapping) or set(answer) != {"axes", "uncertain_axes"}:
            raise _EvidenceError("quality call answer is incomplete")
        if answer["uncertain_axes"] != []:
            raise _EvidenceError("quality call contains uncertain axes")
        axes = answer["axes"]
        if not isinstance(axes, Mapping) or set(axes) != set(AXES):
            raise _EvidenceError("quality call axes are invalid")
        reference_is_a = call["reference_is_a"]
        orientations.add(reference_is_a)
        delta: dict[str, float] = {}
        for axis, score in axes.items():
            if not isinstance(score, Mapping) or set(score) != {"a", "b", "reason"}:
                raise _EvidenceError("quality call axis is invalid")
            if any(
                type(score[name]) is not int or not 0 <= score[name] <= 4 for name in ("a", "b")
            ):
                raise _EvidenceError("quality call score is invalid")
            if not isinstance(score["reason"], str):
                raise _EvidenceError("quality call reason is invalid")
            delta[axis] = (score["b"] - score["a"]) / 4 * (1 if reference_is_a else -1)
        deltas.append(delta)
    if orientations != {False, True}:
        raise _EvidenceError("quality calls must use opposite blind orientations")
    if aggregation == QUALITY_AGGREGATION_V3:
        sides = []
        for call in calls:
            image_hash = call.get("request_image_hash")
            side_hashes = call.get("input_image_hashes")
            if (
                not isinstance(image_hash, str)
                or len(image_hash) != 64
                or not isinstance(side_hashes, Mapping)
                or set(side_hashes) != {"A", "B"}
                or any(
                    not isinstance(values, list)
                    or len(values) != 6
                    or any(not isinstance(value, str) or len(value) != 64 for value in values)
                    for values in side_hashes.values()
                )
            ):
                raise _EvidenceError("v3 quality call input hashes are invalid")
            sides.append(side_hashes)
        if sides[0]["A"] != sides[1]["B"] or sides[0]["B"] != sides[1]["A"]:
            raise _EvidenceError("v3 blind calls do not invert identical image evidence")
    mean_values = {axis: (deltas[0][axis] + deltas[1][axis]) / 2 for axis in AXES}
    disagreement_by_axis = {axis: abs(deltas[0][axis] - deltas[1][axis]) for axis in AXES}
    disagreement = max(disagreement_by_axis.values())
    sign_flips = sum(deltas[0][axis] * deltas[1][axis] < 0 for axis in AXES)
    claimed_axes = observation.get("axes")
    if claimed_axes is not None:
        if not isinstance(claimed_axes, Mapping) or set(claimed_axes) != set(AXES):
            raise _EvidenceError("quality claimed axes are invalid")
        for axis in AXES:
            if _number(claimed_axes[axis], f"claimed {axis}", low=-1, high=1) != mean_values[axis]:
                raise _EvidenceError("quality claimed axes differ from raw calls")
    claimed_disagreement = observation.get("order_disagreement")
    if (
        claimed_disagreement is not None
        and _number(claimed_disagreement, "claimed order_disagreement", low=0) != disagreement
    ):
        raise _EvidenceError("quality claimed disagreement differs from raw calls")
    if aggregation in {QUALITY_AGGREGATION_V1, QUALITY_AGGREGATION_V3}:
        return mean_values, mean_values, disagreement, disagreement_by_axis, sign_flips
    if aggregation == QUALITY_AGGREGATION_V2:
        return (
            {axis: min(deltas[0][axis], deltas[1][axis]) for axis in AXES},
            mean_values,
            disagreement,
            disagreement_by_axis,
            sign_flips,
        )
    raise _EvidenceError("quality aggregation is invalid")


def _timing_comparison(
    timing: Mapping[str, Any],
    plan: Mapping[str, Any],
    slots: set[tuple[str, int]],
    candidate_hash: str,
    incumbent_hash: str,
    incumbent_videos: Mapping[tuple[str, int], str],
    candidate_videos: Mapping[tuple[str, int], str],
    *,
    development: bool,
) -> tuple[dict[str, Any], bool]:
    if timing.get("schema") != TIMING_SCHEMA:
        raise _EvidenceError("timing must be forge/paired-timing/v1")
    if timing.get("contract_hash") != plan.get("contract_hash"):
        raise _EvidenceError("timing contract differs from plan")
    expected_timing_panel_hash = hash_canonical(
        "forge/panel/v1", plan.get("contract_hash"), plan.get("prompts"), plan.get("geometry")
    )
    if timing.get("panel_hash") != expected_timing_panel_hash:
        raise _EvidenceError("timing panel differs from the prescribed plan")
    if timing.get("geometry") != plan.get("geometry"):
        raise _EvidenceError("timing geometry differs from plan")
    contract = _contract(plan)
    _sha256(timing.get("runtime_id"), "timing runtime_id")
    runtime = timing.get("runtime")
    recipes = timing.get("recipes")
    if runtime is not None or recipes is not None:
        if not isinstance(runtime, Mapping) or runtime.get("scheduler") != "paired-exact-recipes":
            raise _EvidenceError("timing runtime is not an exact paired recipe")
        scope = runtime.get("timing_scope")
        if (
            not isinstance(scope, Mapping)
            or scope.get("competitive_includes") != ["text_encoding", "denoising", "vae_decoding"]
            or "model_loading" not in scope.get("competitive_excludes", [])
        ):
            raise _EvidenceError("timing scope does not bind full pipeline generation")
        schedulers = runtime.get("schedulers")
        if not isinstance(schedulers, Mapping) or set(schedulers) != {"reference", "candidate"}:
            raise _EvidenceError("timing scheduler recipes are incomplete")
        if (
            not isinstance(recipes, Mapping)
            or set(recipes) != {"reference", "candidate"}
            or any(
                not isinstance(recipes[side], Mapping)
                or recipes[side].get("kind") != schedulers[side]
                for side in ("reference", "candidate")
            )
        ):
            raise _EvidenceError("timing side recipes do not match their runtime")
    hardware = timing.get("hardware")
    if (
        not isinstance(hardware, Mapping)
        or not isinstance(hardware.get("gpu_name"), str)
        or not hardware["gpu_name"]
    ):
        raise _EvidenceError("timing hardware fingerprint is invalid")
    admissions = {timing.get("reference_admission"), timing.get("candidate_admission")}
    allowed_admissions = (
        {"public_snapshot", "private_experiment"} if development else {"public_snapshot"}
    )
    if not admissions.issubset(allowed_admissions):
        raise _EvidenceError("timing admission is invalid")
    public_admission = admissions == {"public_snapshot"}
    if not public_admission and not development:
        raise _EvidenceError("private artifacts cannot enter automatic promotion")
    if timing.get("reference_manifest_hash") != incumbent_hash:
        raise _EvidenceError("timing reference must be the quality incumbent")
    if timing.get("candidate_manifest_hash") != candidate_hash:
        raise _EvidenceError("timing candidate differs from quality evidence")
    reports = timing.get("report_video_sha256")
    if not isinstance(reports, Mapping) or set(reports) != {"reference", "candidate"}:
        raise _EvidenceError("timing must bind its rendered videos")
    for side, expected in (("reference", incumbent_videos), ("candidate", candidate_videos)):
        reported = reports[side]
        if not isinstance(reported, Mapping) or set(reported) != {
            f"{prompt_id}:{seed}" for prompt_id, seed in slots
        }:
            raise _EvidenceError("timing video binding is incomplete")
        for slot, expected_hash in expected.items():
            if reported.get(f"{slot[0]}:{slot[1]}") != expected_hash:
                raise _EvidenceError("timing and quality used different video evidence")
    design = timing.get("design")
    if not isinstance(design, Mapping) or design.get("order") != [
        "reference",
        "candidate",
        "candidate",
        "reference",
    ]:
        raise _EvidenceError("timing does not use the required ABBA design")
    if type(design.get("warmups_per_side")) is not int or design["warmups_per_side"] < 1:
        raise _EvidenceError("timing must include a warmup for each side")
    measurements = timing.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != 4 * len(slots):
        raise _EvidenceError("timing measurements are required")
    by_slot: dict[tuple[str, int], dict[tuple[str, int], Mapping[str, Any]]] = defaultdict(dict)
    expected_order = {
        ("reference", 0): 0,
        ("candidate", 0): 1,
        ("candidate", 1): 2,
        ("reference", 1): 3,
    }
    exact_runtime = (
        isinstance(runtime, Mapping) and runtime.get("scheduler") == "paired-exact-recipes"
    )
    for start in range(0, len(measurements), 4):
        quartet = measurements[start : start + 4]
        pair = quartet[0].get("pair_id") if isinstance(quartet[0], Mapping) else None
        if not isinstance(pair, str) or any(
            not isinstance(item, Mapping) or item.get("pair_id") != pair for item in quartet
        ):
            raise _EvidenceError("timing pair rows must be consecutive")
        if [item.get("run_order") for item in quartet] != [0, 1, 2, 3]:
            raise _EvidenceError("timing rows are not consecutive ABBA")
    for item in measurements:
        if not isinstance(item, Mapping):
            raise _EvidenceError("timing measurement must be an object")
        slot = (item.get("prompt_id"), item.get("seed"))
        side, repeat, order = item.get("side"), item.get("repeat_index"), item.get("run_order")
        if (
            slot not in slots
            or side not in {"reference", "candidate"}
            or repeat not in {0, 1}
            or order not in range(4)
        ):
            raise _EvidenceError("timing measurement has invalid ABBA identity")
        if item.get("status") != "OK" or item.get("pair_id") != f"{slot[0]}:{slot[1]}":
            raise _EvidenceError("timing failure cannot establish a paired speed result")
        if (side, repeat) in by_slot[slot]:
            raise _EvidenceError("timing repeat is duplicated")
        if expected_order[(side, repeat)] != order:
            raise _EvidenceError("timing order is not ABBA")
        _number(
            item.get("wall_time_seconds"), "timing wall_time_seconds", low=math.nextafter(0.0, 1.0)
        )
        if (
            _number(item.get("peak_vram_bytes"), "timing peak_vram_bytes", low=0)
            > contract.vram_limit_bytes
        ):
            raise _EvidenceError("timing exceeds contract VRAM limit")
        if (
            _number(item["wall_time_seconds"], "timing wall_time_seconds")
            > contract.wall_time_limit_seconds
        ):
            raise _EvidenceError("timing exceeds contract wall-time limit")
        steps = _number(item.get("steps"), "timing steps", low=1)
        limit = contract.max_steps
        if (
            exact_runtime
            and side == "reference"
            and recipes["reference"].get("kind") == "original_unipc"
        ):
            limit = contract.reference_steps
        if not steps.is_integer() or steps > limit:
            raise _EvidenceError("timing steps exceed the speed contract")
        if exact_runtime:
            recipe = recipes[side]
            if steps != recipe.get("steps") or item.get("cfg_enabled") != (
                recipe.get("guidance_scale") > 1
            ):
                raise _EvidenceError("timing measurement differs from its frozen side recipe")
            full_pipeline = _number(
                item.get("full_pipeline_seconds"),
                "timing full_pipeline_seconds",
                low=math.nextafter(0.0, 1.0),
            )
            if full_pipeline != _number(item.get("wall_time_seconds"), "timing wall_time_seconds"):
                raise _EvidenceError(
                    "timing full pipeline clock differs from the competitive clock"
                )
        cfg_enabled = item.get("cfg_enabled")
        if type(cfg_enabled) is not bool or item.get("forward_passes") != int(steps) * (
            2 if cfg_enabled else 1
        ):
            raise _EvidenceError("timing CFG forward-pass evidence is invalid")
        if item.get("expected_forward_passes") != item["forward_passes"]:
            raise _EvidenceError("timing forward-pass receipt is inconsistent")
        if item["forward_passes"] > contract.max_forward_passes:
            raise _EvidenceError("timing forward passes exceed the speed contract")
        by_slot[slot][(side, repeat)] = item
    if set(by_slot) != slots or any(len(values) != 4 for values in by_slot.values()):
        raise _EvidenceError("timing panel is incomplete")
    grouped: dict[str, list[float]] = defaultdict(list)
    for (prompt_id, _seed), values in by_slot.items():
        first = math.log(
            values[("reference", 0)]["wall_time_seconds"]
            / values[("candidate", 0)]["wall_time_seconds"]
        )
        second = math.log(
            values[("reference", 1)]["wall_time_seconds"]
            / values[("candidate", 1)]["wall_time_seconds"]
        )
        grouped[prompt_id].append((first + second) / 2)
    return (
        {
            "by_prompt": {
                prompt_id: sum(values) / len(values) for prompt_id, values in grouped.items()
            }
        },
        public_admission,
    )
