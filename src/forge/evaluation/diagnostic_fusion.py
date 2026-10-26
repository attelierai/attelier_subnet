"""Inspectable pilot comparisons, deliberately outside certified evidence."""

from collections.abc import Mapping

from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.vbench import QUALITY_AXIS_BY_VBENCH_METRIC, normalize_native_score
from forge.protocol.hashing import hash_canonical


def fuse_diagnostic_quality(
    plan: DistributedEvaluationPlan,
    *,
    prompt_id: str,
    pair: Mapping,
    reference_metrics: Mapping,
    candidate_metrics: Mapping,
) -> dict:
    """Combine actual raw observations without inventing calibrated provenance.

    The signed partial report preserves both video receipts, the provider journal
    reference, and native metric values. This format is rejected by the certified
    quality gate even when its numerical deltas look favorable.
    """
    if plan.stage != "diagnostic":
        raise ValueError("diagnostic fusion cannot produce competitive evidence")
    prompt = next(row for row in plan.prompts if row["prompt_id"] == prompt_id)
    reference, candidate = pair["reference_receipt"], pair["candidate_receipt"]
    for receipt, expected_model in (
        (reference, plan.king_manifest_hash),
        (candidate, plan.candidate_manifest_hash),
    ):
        job = receipt["job"]
        if (
            receipt.get("status") != "OK"
            or receipt.get("diagnostic_only") is not True
            or receipt.get("round_id") != plan.round_id
            or job.get("prompt_id") != prompt_id
            or job.get("seed") != prompt["seed"]
            or job.get("prompt") != prompt["text"]
            or job.get("model_manifest_hash") != expected_model
        ):
            raise ValueError("render receipt differs from the frozen comparison")
    if (
        pair.get("reference_sha256") != reference["video_sha256"]
        or pair.get("candidate_sha256") != candidate["video_sha256"]
    ):
        raise ValueError("judged videos differ from render receipts")
    luna = pair["judgment"]
    if luna.get("status") != "OK" or luna.get("uncertain_axes"):
        raise ValueError("uncertain Luna comparison remains incomplete")
    calls = luna.get("calls", [])
    if (
        len(calls) != 2
        or {call.get("reference_is_a") for call in calls} != {True, False}
        or len({call.get("response_id") for call in calls}) != 2
        or any(not call.get("response_id") for call in calls)
    ):
        raise ValueError("two distinct blind orientations are required")
    axes = {}
    for metric, axis in QUALITY_AXIS_BY_VBENCH_METRIC.items():
        # Official VBench reports per-video MUSIQ on its native 0..100 scale.
        scale = 100 if metric == "imaging_quality" else 1
        axes[axis] = normalize_native_score(candidate_metrics[metric] / scale) - (
            normalize_native_score(reference_metrics[metric] / scale)
        )
    orders = []
    for call in calls:
        scores = call["answer"]["axes"]

        def delta(axis: str, scores=scores, call=call) -> float:
            raw = scores[axis]
            if any(type(raw.get(key)) is not int or not 0 <= raw[key] <= 4 for key in ("a", "b")):
                raise ValueError("invalid native judge score")
            return (raw["b"] - raw["a"]) / 4 * (1 if call["reference_is_a"] else -1)

        orders.append(
            {
                "reference_first": call["reference_is_a"],
                "axes": {
                    **axes,
                    "prompt_adherence": delta("prompt_adherence"),
                    "action_adherence": delta("motion_adherence"),
                    "camera_adherence": delta("camera_adherence"),
                    "dynamic_degree": min(delta("motion_adherence"), delta("motion_quality")),
                },
            }
        )
    return {
        **{
            key: prompt[key]
            for key in ("prompt_id", "seed", "subject_class", "camera_type", "style_family")
        },
        "evidence_kind": "diagnostic_quality",
        "reward_eligible": False,
        "orders": orders,
        "reference_video_sha256": reference["video_sha256"],
        "candidate_video_sha256": candidate["video_sha256"],
        "raw_evidence_hash": hash_canonical(
            "forge/diagnostic-quality/v1",
            {
                "pair": pair,
                "reference_metrics": reference_metrics,
                "candidate_metrics": candidate_metrics,
            },
        ),
    }
