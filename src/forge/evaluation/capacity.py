"""Conservative, offline capacity accounting for a frozen distributed plan.

This is an admission check, not an evaluator.  It deliberately treats a
partially affordable panel as incomplete: it never selects a smaller panel or
turns an early screening result into a promotion.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from forge.evaluation.distributed_plan import DistributedEvaluationPlan, assignments_for

SCHEMA = "forge/evaluation-capacity/v1"


@dataclass(frozen=True)
class CapacityInputs:
    """Measured or conservatively estimated costs for one specified GPU class."""

    king_seconds: float
    wan50_seconds: float
    candidate_seconds: float
    gpu_usd_hour: float
    node_overhead_seconds: float | None = None
    node_overhead_usd: float | None = None
    metric_seconds_per_video: float | None = None
    judge_usd_per_quality_comparison: float | None = None
    storage_usd_per_video: float | None = None
    # A duration is derived from the frozen allocation.  A caller may supply a
    # stricter operational cap, but two hours is not an implicit protocol fact.
    max_node_seconds: float | None = None
    validator_candidate_budget_usd: float | None = 10.0
    validator_screening_spent_usd: float | None = None
    network_remaining_usd: float | None = None
    commit_delivery_seconds: float | None = None
    reveal_delivery_seconds: float | None = None
    chain_block_seconds: float | None = None

    def __post_init__(self) -> None:
        values = {
            "king_seconds": self.king_seconds,
            "wan50_seconds": self.wan50_seconds,
            "candidate_seconds": self.candidate_seconds,
            "gpu_usd_hour": self.gpu_usd_hour,
            "node_overhead_seconds": self.node_overhead_seconds,
            "node_overhead_usd": self.node_overhead_usd,
            "metric_seconds_per_video": self.metric_seconds_per_video,
            "judge_usd_per_quality_comparison": self.judge_usd_per_quality_comparison,
            "storage_usd_per_video": self.storage_usd_per_video,
        }
        required_positive = {"king_seconds", "wan50_seconds", "candidate_seconds", "gpu_usd_hour"}
        for name, value in values.items():
            if value is None and name not in required_positive:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or (name in required_positive and value <= 0):
                message = (
                    f"{name} must be positive"
                    if name in required_positive
                    else f"{name} cannot be negative"
                )
                raise ValueError(message)
        positive_when_supplied = {
            "max_node_seconds",
            "validator_candidate_budget_usd",
            "network_remaining_usd",
            "chain_block_seconds",
        }
        for name, value in (
            ("max_node_seconds", self.max_node_seconds),
            ("validator_candidate_budget_usd", self.validator_candidate_budget_usd),
            ("validator_screening_spent_usd", self.validator_screening_spent_usd),
            ("network_remaining_usd", self.network_remaining_usd),
            ("commit_delivery_seconds", self.commit_delivery_seconds),
            ("reveal_delivery_seconds", self.reveal_delivery_seconds),
            ("chain_block_seconds", self.chain_block_seconds),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name in positive_when_supplied and value == 0)
            ):
                raise ValueError(f"{name} must be a non-negative finite number when provided")


def plan_capacity(plan: DistributedEvaluationPlan, inputs: CapacityInputs) -> dict[str, Any]:
    """Return the complete fixed-panel cost and whether it fits explicit limits.

    Quality work produces a challenger video plus one reference video.  If the
    king and Wan-50 are different immutable manifests, a second reference is
    needed; the challenger video is reused only because it is the same frozen
    prompt, seed, recipe and candidate. Screening timing uses one independent
    AB pair; confirmation timing uses an ABBA quartet. Neither reuses quality
    renders.
    """

    king_is_wan50 = plan.king_manifest_hash == plan.wan50_manifest_hash
    references = 1 if king_is_wan50 else 2
    timing_runs = 2 if plan.stage == "screening" else 4
    timing_repeats = 1 if plan.stage == "screening" else 2
    timing_design = "screening_ab" if plan.stage == "screening" else "confirmation_abba"
    unknown_inputs = [
        name
        for name, value in (
            ("node_overhead_seconds", inputs.node_overhead_seconds),
            ("node_overhead_usd", inputs.node_overhead_usd),
            ("metric_seconds_per_video", inputs.metric_seconds_per_video),
            ("judge_usd_per_quality_comparison", inputs.judge_usd_per_quality_comparison),
            ("storage_usd_per_video", inputs.storage_usd_per_video),
            ("validator_candidate_budget_usd", inputs.validator_candidate_budget_usd),
            ("validator_screening_spent_usd", inputs.validator_screening_spent_usd),
            ("network_remaining_usd", inputs.network_remaining_usd),
            ("commit_delivery_seconds", inputs.commit_delivery_seconds),
            ("reveal_delivery_seconds", inputs.reveal_delivery_seconds),
            ("chain_block_seconds", inputs.chain_block_seconds),
        )
        if value is None
    ]
    nodes: dict[str, dict[str, Any]] = {}
    quality_counts: list[int] = []
    timing_counts: list[int] = []
    for hotkey in sorted(plan.committee):
        counts = Counter(item.kind for item in assignments_for(plan, hotkey))
        quality = counts["quality"]
        timing = counts["timing"]
        quality_videos = quality * (1 + references)
        timing_videos = timing * timing_runs
        videos = quality_videos + timing_videos
        quality_reference_seconds = inputs.king_seconds
        if not king_is_wan50:
            quality_reference_seconds += inputs.wan50_seconds
        generation_seconds = quality * (inputs.candidate_seconds + quality_reference_seconds)
        generation_seconds += (
            timing * timing_repeats * (inputs.candidate_seconds + inputs.king_seconds)
        )
        # The runtime invokes VBench only for quality evidence.  It caches the
        # Wan-50/king alias, so quality_videos is also the exact metric-job
        # count; timing AB/ABBA renders are retained but not scored by VBench.
        metric_videos = quality_videos
        metric_seconds = metric_videos * (inputs.metric_seconds_per_video or 0.0)
        gpu_seconds = generation_seconds + metric_seconds + (inputs.node_overhead_seconds or 0.0)
        gpu_usd = gpu_seconds / 3600 * inputs.gpu_usd_hour
        judge_comparisons = quality * references
        judge_usd = judge_comparisons * (inputs.judge_usd_per_quality_comparison or 0.0)
        storage_usd = videos * (inputs.storage_usd_per_video or 0.0)
        total_usd = gpu_usd + judge_usd + storage_usd + (inputs.node_overhead_usd or 0.0)
        quality_counts.append(quality)
        timing_counts.append(timing)
        nodes[hotkey] = {
            "assigned_work": {"quality": quality, "timing": timing},
            "videos": {"quality": quality_videos, "timing": timing_videos, "total": videos},
            "quality_comparisons": judge_comparisons,
            "generation_seconds": generation_seconds,
            "metric_videos": metric_videos,
            "metric_seconds": metric_seconds,
            "node_overhead_seconds": inputs.node_overhead_seconds,
            "gpu_seconds": gpu_seconds,
            "gpu_usd": gpu_usd,
            "judge_usd": judge_usd,
            "storage_usd": storage_usd,
            "node_overhead_usd": inputs.node_overhead_usd,
            "total_usd": total_usd,
        }
    all_gpu_seconds = sum(node["gpu_seconds"] for node in nodes.values())
    all_usd = sum(node["total_usd"] for node in nodes.values())
    makespan_seconds = max(node["gpu_seconds"] for node in nodes.values())
    failures: list[str] = []
    if inputs.max_node_seconds is not None and makespan_seconds > inputs.max_node_seconds:
        failures.append("TIME_CAP_EXCEEDED")
    validator_available_usd = None
    if (
        inputs.validator_candidate_budget_usd is not None
        and inputs.validator_screening_spent_usd is not None
    ):
        validator_available_usd = (
            inputs.validator_candidate_budget_usd - inputs.validator_screening_spent_usd
        )
    if validator_available_usd is not None and any(
        node["total_usd"] > validator_available_usd for node in nodes.values()
    ):
        failures.append("BUDGET_CAP_EXCEEDED")
    if inputs.network_remaining_usd is not None and all_usd > inputs.network_remaining_usd:
        failures.append("NETWORK_BUDGET_CAP_EXCEEDED")
    capacity_fit = not failures and not unknown_inputs
    fit_target: dict[str, Any] | None = None
    if inputs.max_node_seconds is not None:
        per_node: dict[str, dict[str, float | None]] = {}
        for hotkey, node in nodes.items():
            fixed_seconds = node["metric_seconds"] + (inputs.node_overhead_seconds or 0.0)
            available_generation_seconds = inputs.max_node_seconds - fixed_seconds
            required_speedup = (
                None
                if available_generation_seconds <= 0
                else node["generation_seconds"] / available_generation_seconds
            )
            per_node[hotkey] = {
                "available_generation_seconds": available_generation_seconds,
                "required_uniform_generation_speedup": required_speedup,
            }
        fit_target = {
            "max_node_seconds": inputs.max_node_seconds,
            "per_node": per_node,
            "interpretation": (
                "A value above 1 means the measured full-pipeline generation needs that "
                "uniform speedup on this exact allocation; it is not a hardware claim."
            ),
        }
    required_commit_window_seconds = makespan_seconds + (inputs.commit_delivery_seconds or 0.0)
    required_reveal_window_seconds = inputs.reveal_delivery_seconds
    windows: dict[str, Any] = {
        "required_commit_window_seconds": required_commit_window_seconds,
        "required_reveal_window_seconds": required_reveal_window_seconds,
        "commit_delivery_seconds": inputs.commit_delivery_seconds,
        "reveal_delivery_seconds": inputs.reveal_delivery_seconds,
        "chain_block_seconds": inputs.chain_block_seconds,
        "assumption": (
            "The commit window starts before GPU work. It must cover the slowest assigned "
            "validator, measured overhead, and durable report delivery. The reveal window "
            "starts only after commits finalize."
        ),
    }
    if inputs.chain_block_seconds is not None:
        windows["required_commit_window_blocks"] = math.ceil(
            required_commit_window_seconds / inputs.chain_block_seconds
        )
        windows["required_reveal_window_blocks"] = math.ceil(
            (required_reveal_window_seconds or 0.0) / inputs.chain_block_seconds
        )
    return {
        "schema": SCHEMA,
        "inputs": asdict(inputs),
        "plan_hash": plan.plan_hash(),
        "stage": plan.stage,
        "timing_design": timing_design,
        "references_per_quality_prompt": references,
        "reuse": {
            "candidate_quality_video": (
                "only across king/Wan50 comparisons with identical frozen candidate prompt and seed"
            ),
            "king_wan50_alias": king_is_wan50,
            "timing_reuses_quality_renders": False,
            "timing_runs_per_prompt": timing_runs,
        },
        "nodes": nodes,
        "network": {
            "gpu_seconds": all_gpu_seconds,
            "gpu_hours": all_gpu_seconds / 3600,
            "total_usd": all_usd,
            "makespan_seconds": makespan_seconds,
            "makespan_hours": makespan_seconds / 3600,
        },
        "allocation_warning": {
            "kind_balanced": len(set(quality_counts)) == 1 and len(set(timing_counts)) == 1,
            "detail": "equal total assignments do not guarantee equal quality and timing cost",
        },
        "limits": {
            "max_node_seconds": inputs.max_node_seconds,
            "validator_candidate_budget_usd": inputs.validator_candidate_budget_usd,
            "validator_screening_spent_usd": inputs.validator_screening_spent_usd,
            "validator_available_after_screening_usd": validator_available_usd,
            "network_remaining_usd": inputs.network_remaining_usd,
        },
        "fit_target": fit_target,
        "windows": windows,
        "unknown_inputs": unknown_inputs,
        "capacity_fit": capacity_fit,
        "status": (
            "INCOMPLETE_INSUFFICIENT_CAPACITY"
            if failures
            else "PROJECTION_INCOMPLETE_INPUTS"
            if unknown_inputs
            else "ADMISSIBLE_PROJECTION"
        ),
        "failures": failures,
        "dispatch_authorized": False,
        "promotion_possible": False,
        "partial_work_policy": "A partial frozen panel is INCOMPLETE and cannot promote.",
        "screening": (
            "Screening must be planned separately; it can reject early but cannot promote early."
        ),
    }
