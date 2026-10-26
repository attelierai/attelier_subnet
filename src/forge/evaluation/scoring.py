"""Per-video scoring and panel completeness checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from forge.protocol.contract import ChallengeContract

type Slot = tuple[str, int]


class ObservationStatus(StrEnum):
    OK = "OK"
    MODEL_FAILURE = "MODEL_FAILURE"
    INFRA_FAILURE = "INFRA_FAILURE"
    JUDGE_FAILURE = "JUDGE_FAILURE"


class ComparisonIncompleteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    prompt_id: str
    seed: int
    axes: dict[str, float] | None = None
    status: ObservationStatus = ObservationStatus.OK
    detail: str | None = None

    @property
    def slot(self) -> Slot:
        return (self.prompt_id, self.seed)


@dataclass(frozen=True)
class ScoredObservation:
    prompt_id: str
    seed: int
    score: float
    axes: dict[str, float]
    model_failed: bool

    @property
    def slot(self) -> Slot:
        return (self.prompt_id, self.seed)


def video_score(axis_values: dict[str, float], contract: ChallengeContract) -> float:
    if set(axis_values) != set(contract.axes):
        missing = set(contract.axes) - set(axis_values)
        extra = set(axis_values) - set(contract.axes)
        raise ValueError(f"axis mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    values: list[float] = []
    for axis in contract.axes:
        value = axis_values[axis]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"axis {axis!r} must be finite and in [0,1]")
        values.append(value)
    weighted_mean = sum(contract.axis_weights[axis] * axis_values[axis] for axis in contract.axes)
    worst_mean = sum(sorted(values)[: contract.worst_axis_count]) / contract.worst_axis_count
    return contract.score_mean_weight * weighted_mean + contract.score_worst_weight * worst_mean


def score_observation(observation: Observation, contract: ChallengeContract) -> ScoredObservation:
    if observation.status in {ObservationStatus.INFRA_FAILURE, ObservationStatus.JUDGE_FAILURE}:
        raise ComparisonIncompleteError(
            f"{observation.status.value} at {observation.prompt_id}/{observation.seed}"
        )
    if observation.status == ObservationStatus.MODEL_FAILURE:
        zeros = {axis: 0.0 for axis in contract.axes}
        return ScoredObservation(observation.prompt_id, observation.seed, 0.0, zeros, True)
    if observation.axes is None:
        raise ValueError("successful observations require all axes")
    axes = dict(observation.axes)
    return ScoredObservation(
        observation.prompt_id,
        observation.seed,
        video_score(axes, contract),
        axes,
        False,
    )


def validate_and_score_panel(
    observations: list[Observation] | tuple[Observation, ...],
    expected_slots: set[Slot] | frozenset[Slot],
    contract: ChallengeContract,
) -> dict[Slot, ScoredObservation]:
    indexed: dict[Slot, ScoredObservation] = {}
    for observation in observations:
        if observation.slot in indexed:
            raise ValueError(f"duplicated observation slot: {observation.slot}")
        indexed[observation.slot] = score_observation(observation, contract)
    actual = set(indexed)
    if actual != set(expected_slots):
        missing = set(expected_slots) - actual
        extra = actual - set(expected_slots)
        raise ComparisonIncompleteError(
            f"panel slots mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return indexed
