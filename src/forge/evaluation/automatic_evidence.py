"""Historical diagnostic evidence validation and policy constants."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from forge.evaluation.raster_integrity import RULE_ID as RASTER_INTEGRITY_RULE_ID
from forge.evaluation.raster_integrity import verify_check
from forge.evaluation.vision_calibration import (
    QUALITY_AGGREGATION_V1,
    QUALITY_AGGREGATION_V2,
    QUALITY_AGGREGATION_V3,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical

QUALITY_SCHEMA = "forge/automatic-quality/v1"


TIMING_SCHEMA = "forge/paired-timing/v1"


CALIBRATION_SCHEMA = "forge/auto-calibration/v1"


AXES = (
    "prompt_adherence",
    "motion_adherence",
    "camera_adherence",
    "subject_consistency",
    "temporal_stability",
    "motion_quality",
    "visual_quality",
    "physical_plausibility",
)


SUBJECT_CLASSES = {"rigid_object", "human", "animal", "scenery"}


SPATIAL_DETAIL_POLICY = "forge/nine-stratified-plus-three-secret-crops/v2"


TEMPORAL_TRANSITION_POLICY = "forge/secret-nine-stratified-adjacent-delta/v1"


class _EvidenceError(ValueError):
    pass


def _result(status: str, reasons: list[str], **evidence: Any) -> dict[str, Any]:
    return {"status": status, "reasons": reasons, **evidence}


def _native_detail_evidence(
    quality: Mapping, calibration: Mapping, geometry: Mapping[str, Any]
) -> None:
    frames = geometry["frames"]
    width, height = geometry["width"], geometry["height"]
    for spec in (quality.get("spec"), calibration.get("evaluator")):
        if (
            not isinstance(spec, Mapping)
            or spec.get("spatial_detail_policy") != SPATIAL_DETAIL_POLICY
        ):
            raise _EvidenceError("production requires post-render native-frame selection")
    rows = [row for group in quality["comparisons"] for row in group["observations"]]
    rows.extend(row for group in calibration["controls"].values() for row in group)
    count = min(frames, 9)
    for row in rows:
        indices = row.get("detail_frame_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != count
            or any(
                type(value) is not int
                or not index * frames // count <= value < (index + 1) * frames // count
                for index, value in enumerate(indices)
            )
        ):
            raise _EvidenceError("native-frame selection does not cover every temporal stratum")
        crops = row.get("detail_crop_regions")
        if not isinstance(crops, list) or len(crops) != 3:
            raise _EvidenceError("production requires three matched post-render detail crops")
        seen_frames = set()
        for crop in crops:
            if not isinstance(crop, Mapping) or set(crop) != {
                "frame_index",
                "x",
                "y",
                "width",
                "height",
            }:
                raise _EvidenceError("detail crop coverage is not replayable")
            if any(type(value) is not int for value in crop.values()):
                raise _EvidenceError("detail crop coverage is not replayable")
            frame_index, x, y, width, height = (
                crop["frame_index"],
                crop["x"],
                crop["y"],
                crop["width"],
                crop["height"],
            )
            if frame_index not in indices or frame_index in seen_frames:
                raise _EvidenceError("detail crops must be distinct native-detail frames")
            seen_frames.add(frame_index)
            if (
                (width, height) != (geometry["width"] // 4, geometry["height"] // 4)
                or not 0 <= x <= geometry["width"] - width
                or not 0 <= y <= geometry["height"] - height
            ):
                raise _EvidenceError("detail crop coverage is outside the decoded frame")


def _transition_evidence(
    quality: Mapping, calibration: Mapping, geometry: Mapping[str, Any]
) -> None:
    """Bind v3 temporal transition representation without turning it into a score."""

    transitions = geometry["frames"] - 1
    count = min(transitions, 9)
    rows = [row for comparison in quality["comparisons"] for row in comparison["observations"]]
    rows.extend(row for group in calibration["controls"].values() for row in group)
    for row in rows:
        indices = row.get("transition_frame_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != count
            or len(set(indices)) != count
            or any(
                type(value) is not int
                or not index * transitions // count <= value < (index + 1) * transitions // count
                for index, value in enumerate(indices)
            )
        ):
            raise _EvidenceError("v3 temporal transition evidence is incomplete")


def _raster_integrity_evidence(
    quality: Mapping, calibration: Mapping, geometry: Mapping[str, Any]
) -> None:
    """Require a CLEAR, immutable raster receipt for every judged source video."""
    quality_hashes = {
        video_sha256
        for comparison in quality["comparisons"]
        for observation in comparison["observations"]
        for video_sha256 in (
            observation.get("reference_video_sha256"),
            observation.get("candidate_video_sha256"),
        )
    }
    calibration_hashes = {
        row.get("reference_video_sha256")
        for rows in calibration["controls"].values()
        for row in rows
    }
    if (
        quality.get("raster_integrity_rule_id") != RASTER_INTEGRITY_RULE_ID
        or calibration.get("raster_integrity_rule_id") != RASTER_INTEGRITY_RULE_ID
        or calibration.get("raster_integrity_status") != "CLEAR"
    ):
        raise _EvidenceError("raster integrity policy is not bound to the evidence")
    for spec in (quality.get("spec"), calibration.get("evaluator")):
        if (
            not isinstance(spec, Mapping)
            or spec.get("raster_integrity_rule_id") != RASTER_INTEGRITY_RULE_ID
        ):
            raise _EvidenceError("evaluator does not bind the raster integrity policy")
    for evidence, expected_hashes in (
        (quality, quality_hashes),
        (calibration, calibration_hashes),
    ):
        checks = evidence.get("raster_checks")
        if not isinstance(checks, Mapping):
            raise _EvidenceError("raster integrity checks are required")
        for video_sha256 in expected_hashes:
            if not isinstance(video_sha256, str) or not verify_check(
                checks.get(video_sha256), video_sha256, geometry
            ):
                raise _EvidenceError("raster integrity receipt does not bind a video")
            if checks[video_sha256].get("status") != "CLEAR":
                raise _EvidenceError("RASTER_INTEGRITY_ANOMALY")


def _number(
    value: Any, label: str, *, low: float | None = None, high: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _EvidenceError(f"{label} must be finite")
    result = float(value)
    if (low is not None and result < low) or (high is not None and result > high):
        raise _EvidenceError(f"{label} is outside its allowed range")
    return result


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise _EvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _policy(policy: Mapping[str, Any], *, development: bool) -> dict[str, Any]:
    v1_required = {
        "policy_id",
        "evaluator_id",
        "min_prompts",
        "min_slice_prompts",
        "quality_margin",
        "min_speedup",
        "alpha",
        "max_candidates",
        "max_order_disagreement",
    }
    v2_required = v1_required | {"quality_aggregation"}
    aggregation = policy.get("quality_aggregation", QUALITY_AGGREGATION_V1)
    required = (
        v2_required
        if aggregation in {QUALITY_AGGREGATION_V2, QUALITY_AGGREGATION_V3}
        else v1_required
    )
    if set(policy) != required:
        raise _EvidenceError("automatic policy fields are not exact")
    policy_body = {key: value for key, value in policy.items() if key != "policy_id"}
    domain = {
        QUALITY_AGGREGATION_V1: "forge/auto-policy/v1",
        QUALITY_AGGREGATION_V2: "forge/auto-policy/v2",
        QUALITY_AGGREGATION_V3: "forge/auto-policy/v3",
    }.get(aggregation)
    if domain is None:
        raise _EvidenceError("automatic policy quality aggregation is invalid")
    if policy["policy_id"] != hash_canonical(domain, policy_body):
        raise _EvidenceError("policy_id does not bind the automatic policy")
    if aggregation != QUALITY_AGGREGATION_V3 and not development:
        raise _EvidenceError("production requires orientation-mean policy v3")
    if aggregation not in {QUALITY_AGGREGATION_V1, QUALITY_AGGREGATION_V2, QUALITY_AGGREGATION_V3}:
        raise _EvidenceError("automatic policy quality aggregation is invalid")
    if not isinstance(policy["evaluator_id"], str) or not policy["evaluator_id"]:
        raise _EvidenceError("evaluator_id is required")
    for name in ("min_prompts", "min_slice_prompts", "max_candidates"):
        if isinstance(policy[name], bool) or not isinstance(policy[name], int) or policy[name] < 2:
            raise _EvidenceError(f"policy {name} must be an integer of at least two")
    if policy["min_prompts"] < 128 or policy["min_slice_prompts"] < 24:
        raise _EvidenceError("automatic policy cannot reduce prescribed panel coverage")
    if policy["max_candidates"] != 16:
        raise _EvidenceError("automatic policy max_candidates must be 16")
    for name, low, high in (
        ("quality_margin", 0, 1),
        ("min_speedup", 1, None),
        ("alpha", 0, 1),
        ("max_order_disagreement", 0, 1),
    ):
        value = _number(policy[name], f"policy {name}")
        if value <= low or (high is not None and value > high):
            raise _EvidenceError(f"policy {name} is invalid")
    if (
        policy["quality_margin"] > 0.02
        or policy["min_speedup"] < 1.1
        or policy["alpha"] > 0.05
        or policy["max_order_disagreement"] > 0.25
    ):
        raise _EvidenceError("automatic policy cannot weaken the promotion thresholds")
    return dict(policy)


def _contract(plan: Mapping[str, Any]) -> ChallengeContract:
    raw = plan.get("contract")
    if not isinstance(raw, Mapping):
        raise _EvidenceError("plan contract is required")
    try:
        contract = ChallengeContract.from_dict(dict(raw))
    except (TypeError, ValueError):
        raise _EvidenceError("plan contract is invalid") from None
    if plan.get("contract_hash") != contract.contract_hash():
        raise _EvidenceError("plan contract hash does not bind its contract")
    geometry = {
        "height": contract.height,
        "width": contract.width,
        "frames": contract.frames,
        "fps": contract.fps,
    }
    if plan.get("geometry") != geometry:
        raise _EvidenceError("plan geometry differs from its contract")
    return contract


def _panel(
    plan: Mapping[str, Any],
) -> tuple[set[tuple[str, int]], dict[str, Mapping[str, Any]], str]:
    prompts = plan.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise _EvidenceError("plan prompts are required")
    slots: set[tuple[str, int]] = set()
    by_id: dict[str, Mapping[str, Any]] = {}
    texts: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, Mapping):
            raise _EvidenceError("plan prompt must be an object")
        prompt_id, seeds, text = (
            prompt.get("prompt_id"),
            prompt.get("video_seeds"),
            prompt.get("text"),
        )
        properties = prompt.get("properties")
        if (
            not isinstance(prompt_id, str)
            or not isinstance(seeds, list)
            or not isinstance(properties, Mapping)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise _EvidenceError("plan prompt identity, seeds and properties are required")
        if (
            not prompt_id
            or prompt_id in by_id
            or text.strip() in texts
            or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
        ):
            raise _EvidenceError("plan prompt texts, ids and seeds must be unique and bounded")
        by_id[prompt_id] = prompt
        texts.add(text.strip())
        for seed in seeds:
            slot = (prompt_id, seed)
            if slot in slots:
                raise _EvidenceError("plan slots must be unique")
            slots.add(slot)
    return slots, by_id, hash_canonical("forge/panel-prompts/v1", prompts)


_BET_SCALES = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)


def _calibration_plan_binding(plan: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    """Require the controls receipt to name the plan declared by confirmation."""

    expected_calibration_plan = _sha256(
        plan.get("calibration_plan_hash"), "declared calibration plan hash"
    )
    actual_calibration_plan = _sha256(
        calibration.get("calibration_plan_hash"), "calibration plan hash"
    )
    if actual_calibration_plan != expected_calibration_plan:
        raise _EvidenceError("calibration does not match the declared calibration plan")


def _freshness(
    plan: Mapping[str, Any], quality: Mapping[str, Any], calibration: Mapping[str, Any]
) -> None:
    """Bind a promotion decision to a fresh, disjoint calibration campaign."""

    _calibration_plan_binding(plan, calibration)
    campaign = plan.get("campaign_id", plan.get("round_id"))
    if not isinstance(campaign, str) or not campaign:
        raise _EvidenceError("plan campaign_id or round_id is required")
    if quality.get("campaign_id") != campaign or calibration.get("campaign_id") != campaign:
        raise _EvidenceError("quality and calibration must belong to the plan campaign")
    calibration_panel = _sha256(calibration.get("panel_hash"), "calibration panel hash")
    if calibration_panel == quality.get("panel_hash"):
        raise _EvidenceError("calibration and confirmation panels must be disjoint")
    quality_created = _number(quality.get("created_unix"), "quality created_unix", low=0)
    quality_completed = _number(quality.get("completed_unix"), "quality completed_unix", low=0)
    calibration_created = _number(
        calibration.get("created_unix"), "calibration created_unix", low=0
    )
    calibration_completed = _number(
        calibration.get("completed_unix"), "calibration completed_unix", low=0
    )
    if (
        calibration_created > calibration_completed
        or calibration_completed > quality_created
        or quality_created > quality_completed
        or quality_completed - calibration_completed > 86_400
    ):
        raise _EvidenceError("calibration must precede quality evidence by at most 24 hours")


def _response_id_uniqueness(quality: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    """Reject a provider response reused anywhere in a decision's raw calls."""

    response_ids: set[str] = set()

    def collect(calls: Any, label: str) -> None:
        if not isinstance(calls, list) or len(calls) != 2:
            raise _EvidenceError(f"{label} must contain exactly two raw calls")
        for call in calls:
            if not isinstance(call, Mapping):
                raise _EvidenceError(f"{label} raw call is invalid")
            response_id = call.get("response_id")
            if not isinstance(response_id, str) or not response_id or response_id in response_ids:
                raise _EvidenceError("provider response_id is missing or replayed")
            response_ids.add(response_id)

    comparisons = quality.get("comparisons")
    if not isinstance(comparisons, list):
        raise _EvidenceError("quality comparisons are required")
    for comparison in comparisons:
        if not isinstance(comparison, Mapping) or not isinstance(
            comparison.get("observations"), list
        ):
            raise _EvidenceError("quality observations are required")
        for observation in comparison["observations"]:
            if not isinstance(observation, Mapping):
                raise _EvidenceError("quality observation is invalid")
            collect(observation.get("calls"), "quality observation")

    controls = calibration.get("controls")
    if not isinstance(controls, Mapping):
        raise _EvidenceError("calibration controls are required")
    for kind, rows in controls.items():
        if not isinstance(rows, list):
            raise _EvidenceError("calibration controls are invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise _EvidenceError("calibration control is invalid")
            collect(row.get("calls"), f"calibration control {kind}")


def _production_binding(
    plan: Mapping[str, Any], candidate_hash: str, incumbent_hash: str, *, development: bool
) -> None:
    """Bind local evidence to the locally supplied admission and incumbent snapshot.

    This validates structure inside a trusted validator plan.  It deliberately does
    not claim that either admission or incumbent certification was proven on-chain.
    """

    if development:
        return
    close_block = plan.get("close_block")
    if type(close_block) is not int or close_block < 0:
        raise _EvidenceError("production plan close_block is required")
    candidate = plan.get("candidate")
    incumbent = plan.get("incumbent")
    admitted = plan.get("admitted")
    if not isinstance(candidate, Mapping) or not isinstance(incumbent, Mapping):
        raise _EvidenceError("production plan candidate and incumbent are required")
    if not isinstance(admitted, list) or not admitted:
        raise _EvidenceError("production plan admitted candidates are required")

    candidate_manifest = _sha256(candidate.get("manifest_hash"), "plan candidate manifest hash")
    candidate_hotkey = candidate.get("hotkey")
    candidate_block = candidate.get("admission_block")
    if (
        candidate_manifest != candidate_hash
        or not isinstance(candidate_hotkey, str)
        or not candidate_hotkey
        or type(candidate_block) is not int
        or not 0 <= candidate_block <= close_block
    ):
        raise _EvidenceError("production candidate is not a complete pre-close admission")

    matching_admission = False
    seen_admissions: set[str] = set()
    for entry in admitted:
        if not isinstance(entry, Mapping):
            raise _EvidenceError("production admission is invalid")
        manifest = _sha256(entry.get("manifest_hash"), "admitted manifest hash")
        block = entry.get("block")
        hotkey = entry.get("hotkey")
        if manifest in seen_admissions or type(block) is not int or not 0 <= block <= close_block:
            raise _EvidenceError("production admissions are invalid")
        if hotkey is not None and (not isinstance(hotkey, str) or not hotkey):
            raise _EvidenceError("admitted hotkey is invalid")
        seen_admissions.add(manifest)
        if manifest == candidate_manifest and block == candidate_block:
            if hotkey is not None and hotkey != candidate_hotkey:
                raise _EvidenceError("candidate hotkey differs from its admission")
            matching_admission = True
    if not matching_admission:
        raise _EvidenceError("candidate is absent from pre-close admissions")

    incumbent_manifest = _sha256(incumbent.get("manifest_hash"), "incumbent manifest hash")
    activation_block = incumbent.get("certified_activation_block")
    if (
        incumbent_manifest != incumbent_hash
        or type(activation_block) is not int
        or not 0 <= activation_block <= close_block
    ):
        raise _EvidenceError("production incumbent does not bind quality and timing evidence")
    _sha256(incumbent.get("certified_state_hash"), "incumbent certified state hash")
