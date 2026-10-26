"""Prompt-clustered paired comparison."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from scipy.stats import t as student_t

from forge.evaluation.scoring import (
    ComparisonIncompleteError,
    Observation,
    ScoredObservation,
    Slot,
    validate_and_score_panel,
)
from forge.protocol.contract import ChallengeContract


class VerdictStatus(StrEnum):
    PROMOTE_ELIGIBLE = "PROMOTE_ELIGIBLE"
    REJECT = "REJECT"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class Estimate:
    n_prompts: int
    mean: float
    standard_error: float
    lower_bound: float
    alpha: float


@dataclass(frozen=True)
class Verdict:
    status: VerdictStatus
    primary: Estimate | None
    axis_checks: dict[str, Estimate]
    slice_checks: dict[str, Estimate]
    challenger_failure_rate: float | None
    alpha_per_test: float
    reasons: tuple[str, ...]


def lower_confidence_bound(values: list[float], alpha: float) -> Estimate:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    n = len(values)
    if n < 2:
        raise ComparisonIncompleteError("at least two independent prompts are required")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    standard_error = math.sqrt(variance / n)
    critical = float(student_t.ppf(1 - alpha, df=n - 1))
    return Estimate(n, mean, standard_error, mean - critical * standard_error, alpha)


def _prompt_means(
    king: dict[tuple[str, int], ScoredObservation],
    challenger: dict[tuple[str, int], ScoredObservation],
    field: str | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for slot in sorted(king):
        king_value = king[slot].score if field is None else king[slot].axes[field]
        challenger_value = challenger[slot].score if field is None else challenger[slot].axes[field]
        grouped[slot[0]].append(challenger_value - king_value)
    return {prompt: sum(values) / len(values) for prompt, values in grouped.items()}


def paired_comparison(
    king_observations: list[Observation] | tuple[Observation, ...],
    challenger_observations: list[Observation] | tuple[Observation, ...],
    contract: ChallengeContract,
    *,
    expected_slots: set[Slot] | frozenset[Slot],
    prompt_properties: dict[str, dict[str, str | int]] | None = None,
) -> Verdict:
    test_count = 1 + len(contract.protected_axes) + len(contract.protected_slices) + 1
    alpha_per_test = contract.alpha / (contract.max_candidates * test_count)
    try:
        king = validate_and_score_panel(king_observations, expected_slots, contract)
        challenger = validate_and_score_panel(challenger_observations, expected_slots, contract)
        primary_values = list(_prompt_means(king, challenger).values())
        primary = lower_confidence_bound(primary_values, alpha_per_test)
        axis_checks = {
            axis: lower_confidence_bound(
                list(_prompt_means(king, challenger, axis).values()), alpha_per_test
            )
            for axis in contract.protected_axes
        }
        slice_checks: dict[str, Estimate] = {}
        primary_by_prompt = _prompt_means(king, challenger)
        if contract.protected_slices and prompt_properties is None:
            raise ComparisonIncompleteError("protected slices require prompt properties")
        for protected_slice in contract.protected_slices:
            values = [
                value
                for prompt_id, value in primary_by_prompt.items()
                if prompt_properties is not None
                and prompt_properties.get(prompt_id, {}).get(protected_slice.property_name)
                == protected_slice.value
            ]
            slice_checks[protected_slice.name] = lower_confidence_bound(values, alpha_per_test)
    except (ComparisonIncompleteError, ValueError) as exc:
        return Verdict(
            VerdictStatus.INCOMPLETE,
            None,
            {},
            {},
            None,
            alpha_per_test,
            (str(exc),),
        )

    failure_rate = sum(item.model_failed for item in challenger.values()) / len(challenger)
    reasons: list[str] = []
    if failure_rate > contract.failure_rate_cap:
        reasons.append(
            f"challenger failure rate {failure_rate:.6f} exceeds {contract.failure_rate_cap:.6f}"
        )
    if primary.lower_bound <= contract.min_effect:
        reasons.append(
            f"primary LCB {primary.lower_bound:.6f} does not exceed {contract.min_effect:.6f}"
        )
    for axis, estimate in axis_checks.items():
        if estimate.lower_bound < -contract.axis_noninferiority_margin:
            reasons.append(f"protected axis {axis} regressed: LCB={estimate.lower_bound:.6f}")
    slice_by_name = {item.name: item for item in contract.protected_slices}
    for name, estimate in slice_checks.items():
        if estimate.lower_bound < -slice_by_name[name].noninferiority_margin:
            reasons.append(f"protected slice {name} regressed: LCB={estimate.lower_bound:.6f}")
    status = VerdictStatus.REJECT if reasons else VerdictStatus.PROMOTE_ELIGIBLE
    return Verdict(
        status,
        primary,
        axis_checks,
        slice_checks,
        failure_rate,
        alpha_per_test,
        tuple(reasons),
    )
