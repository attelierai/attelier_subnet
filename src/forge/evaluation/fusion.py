"""Recompute MVP quality rows from trusted raw Luna and VBench evidence.

The journal verifier authenticates the validator's persisted provider receipts.
This function never accepts a miner's asserted deltas as metric observations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSet
from pathlib import Path
from typing import Any

from forge.evaluation.vbench import quality_axis_scores
from forge.protocol.hashing import hash_canonical, sha256_file


def evaluator_implementation_hash() -> str:
    """Bind calibration to all code that turns video bytes into quality axes."""
    root = Path(__file__).parent
    names = (
        "fusion.py",
        "vbench.py",
        "vbench_runner.py",
        "vision_judge.py",
        "vision_media.py",
        "raster_integrity.py",
    )
    return hash_canonical(
        "forge/evaluator-implementation/v1", {name: sha256_file(root / name) for name in names}
    )


def fused_axis_orders(
    calls: list[Mapping[str, Any]],
    reference_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Shared numerical fusion; callers authenticate and bind raw evidence first."""
    reference_scores = quality_axis_scores(reference_metrics)
    candidate_scores = quality_axis_scores(candidate_metrics)
    deltas = {axis: candidate_scores[axis] - value for axis, value in reference_scores.items()}
    if len(calls) != 2 or {call.get("reference_is_a") for call in calls} != {True, False}:
        raise ValueError("two blind orientations are required for fusion")
    orders = []
    for call in calls:
        scores = call["answer"]["axes"]

        def delta(axis: str, scores=scores, call=call) -> float:
            raw = scores[axis]
            if any(type(raw.get(key)) is not int or not 0 <= raw[key] <= 4 for key in ("a", "b")):
                raise ValueError("invalid raw Luna score")
            return (raw["b"] - raw["a"]) / 4 * (1 if call["reference_is_a"] else -1)

        orders.append(
            {
                "reference_first": call["reference_is_a"],
                "axes": {
                    **deltas,
                    "prompt_adherence": delta("prompt_adherence"),
                    "action_adherence": delta("motion_adherence"),
                    "camera_adherence": delta("camera_adherence"),
                    "dynamic_degree": min(delta("motion_adherence"), delta("motion_quality")),
                },
            }
        )
    return orders


def fuse_quality_row(
    *,
    plan: Mapping[str, Any],
    prompt: Mapping[str, Any],
    reference_role: str,
    reference_video_sha256: str,
    candidate_video_sha256: str,
    luna: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    evaluator_provenance: Mapping[str, str],
    verify_journal: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    used_response_ids: MutableSet[str] | None = None,
) -> dict[str, Any]:
    """Validate local receipts; it does not prove the validator executed them."""
    if reference_role not in {"incumbent", "wan50"}:
        raise ValueError("unknown reference role")
    if prompt not in plan["prompts"]:
        raise ValueError("quality prompt is outside frozen panel")
    if luna.get("status") != "OK" or luna.get("uncertain_axes"):
        raise ValueError("uncertain/incomplete Luna judgments cannot pass quality")
    calls = luna.get("calls", [])
    if len(calls) != 2 or {call.get("reference_is_a") for call in calls} != {True, False}:
        raise ValueError("two blind Luna orientations are required")
    if len({call.get("response_id") for call in calls}) != 2 or any(
        not call.get("response_id") for call in calls
    ):
        raise ValueError("Luna comparisons must use two distinct provider responses")
    response_ids = {call["response_id"] for call in calls}
    if used_response_ids is not None and response_ids & used_response_ids:
        raise ValueError("Luna provider response is reused by multiple frozen slots")
    prompt_hash = hash_canonical("forge/mvp-prompt/v1", prompt["text"])
    expected = {
        "plan_id": plan["plan_id"],
        "prompt_id": prompt["prompt_id"],
        "seed": prompt["seed"],
        "prompt_hash": prompt_hash,
    }
    reference_manifest = plan[reference_role + "_manifest_hash"]
    metric_provenance = reference_metrics.get("provenance")
    if not isinstance(metric_provenance, Mapping):
        raise ValueError("VBench provenance is missing")
    metric_lock = hash_canonical(
        "forge/vbench-weight-lock/v1", metric_provenance.get("weight_sha256")
    )
    if (
        evaluator_provenance.get("vbench_model_hashes") != metric_lock
        or not isinstance(evaluator_provenance.get("luna_evaluator_id"), str)
        or evaluator_provenance.get("source_implementation_hash") != evaluator_implementation_hash()
    ):
        raise ValueError("policy evaluator provenance is incomplete or differs from metric lock")
    for receipt, video_hash, manifest in (
        (reference_metrics, reference_video_sha256, reference_manifest),
        (candidate_metrics, candidate_video_sha256, plan["candidate_manifest_hash"]),
    ):
        if receipt.get("runner_source_sha256") != sha256_file(
            Path(__file__).with_name("vbench_runner.py")
        ):
            raise ValueError("VBench runner differs from the calibrated evaluator implementation")
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("VBench receipt differs from frozen prompt slot")
        if receipt.get("video_sha256") != video_hash or receipt.get("manifest_hash") != manifest:
            raise ValueError("VBench video/model differs from quality pair")
    if reference_metrics["provenance"] != candidate_metrics["provenance"]:
        raise ValueError("comparison uses different VBench metric implementation/weights")
    for call in calls:
        answer = call.get("answer") if isinstance(call, Mapping) else None
        scores = answer.get("axes") if isinstance(answer, Mapping) else None
        if not isinstance(scores, Mapping):
            raise ValueError("Luna call is missing raw axis scores")

        if call.get("model") != evaluator_provenance["luna_evaluator_id"]:
            raise ValueError("Luna model differs from frozen evaluator provenance")
        journal_context = {
            **expected,
            "reference_role": reference_role,
            "reference_manifest_hash": reference_manifest,
            "candidate_manifest_hash": plan["candidate_manifest_hash"],
            "reference_video_sha256": reference_video_sha256,
            "candidate_video_sha256": candidate_video_sha256,
            "response_id": call["response_id"],
        }
        if verify_journal is None or not verify_journal(call, journal_context):
            raise ValueError(
                "Luna provider response lacks authenticated slot-bound journal evidence"
            )

    orders = fused_axis_orders(calls, reference_metrics, candidate_metrics)
    if used_response_ids is not None:
        used_response_ids.update(response_ids)
    return {
        **{
            key: prompt[key]
            for key in ("prompt_id", "seed", "subject_class", "camera_type", "style_family")
        },
        "prompt_hash": prompt_hash,
        "plan_id": plan["plan_id"],
        "candidate_manifest_hash": plan["candidate_manifest_hash"],
        "reference_manifest_hash": reference_manifest,
        "candidate_video_sha256": candidate_video_sha256,
        "reference_video_sha256": reference_video_sha256,
        "orders": orders,
        "evidence_kind": "validator_fused_quality",
        "raw_evidence_hash": hash_canonical(
            "forge/mvp-fused-evidence/v1",
            {"luna": luna, "reference": reference_metrics, "candidate": candidate_metrics},
        ),
    }


def fuse_bootstrap_alias_row(**kwargs: Any) -> dict[str, Any]:
    """Fuse one Wan-50 comparison once when it is also the exact incumbent.

    This is an evidence alias, never a second observation.  The normal fuser
    still owns the two blind Luna calls and response-replay protection.
    """

    plan = kwargs.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("bootstrap alias needs a frozen plan")
    if plan.get("incumbent_manifest_hash") != plan.get("wan50_manifest_hash"):
        raise ValueError("bootstrap alias requires identical incumbent and Wan-50 manifests")
    if kwargs.get("reference_role") not in {None, "wan50"}:
        raise ValueError("bootstrap alias must fuse Wan-50 once, not a second role")
    row = fuse_quality_row(**(dict(kwargs) | {"reference_role": "wan50"}))
    identity = {
        key: row[key]
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
    return row | {
        "reference_roles": ["wan50", "incumbent"],
        "comparison_id": hash_canonical("forge/mvp-bootstrap-comparison/v1", identity),
    }
