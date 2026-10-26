"""Deterministic signed partial evidence without a coordinator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any

from forge.evaluation.mvp_evidence import _plan_slots, validate_confirmation_panel
from forge.evaluation.mvp_policy import (
    SCREEN_PROMPTS,
    TIMING_PROMPTS,
)
from forge.protocol.hashing import hash_canonical

SCHEMA = "forge/distributed-evaluation/v1"
REPORT_SCHEMA = "forge/distributed-evaluation-report/v1"


@dataclass(frozen=True)
class AssignedWork:
    validator_hotkey: str
    kind: str
    prompt_id: str
    seed: int
    audit: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "validator_hotkey": self.validator_hotkey,
            "kind": self.kind,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "audit": self.audit,
        }


@dataclass(frozen=True)
class DistributedEvaluationPlan:
    round_id: str
    stage: str
    entropy: str
    policy_id: str
    candidate_manifest_hash: str
    king_manifest_hash: str
    wan50_manifest_hash: str
    committee: Mapping[str, int]
    prompts: tuple[Mapping[str, Any], ...]
    timing_prompt_ids: tuple[str, ...]
    quality_tolerance: float | None
    timing_log_ratio_tolerance: float | None
    audit_calibration_hash: str | None = None
    minimum_audit_replicas: int = 2
    chain_context: Mapping[str, Any] = dataclass_field(default_factory=dict)
    audit_fraction: float = 0.25
    audit_work_count: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"screening", "confirmation", "diagnostic"}:
            raise ValueError("distributed stage is invalid")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.round_id,
                self.entropy,
                self.policy_id,
                self.candidate_manifest_hash,
                self.king_manifest_hash,
                self.wan50_manifest_hash,
            )
        ):
            raise ValueError("distributed plan identifiers are required")
        if not self.committee or any(
            not isinstance(hotkey, str) or not hotkey or type(stake) is not int or stake <= 0
            for hotkey, stake in self.committee.items()
        ):
            raise ValueError("distributed committee requires positive integer stake")
        slots = _plan_slots(self.prompts)
        if len(set(self.timing_prompt_ids)) != len(self.timing_prompt_ids) or not set(
            self.timing_prompt_ids
        ).issubset(slots):
            raise ValueError("timing prompts must be unique frozen prompt ids")
        if self.stage == "screening" and (
            len(slots) != SCREEN_PROMPTS or len(self.timing_prompt_ids) != SCREEN_PROMPTS
        ):
            raise ValueError("distributed screening requires 16 quality and timing prompts")
        if self.stage == "confirmation":
            validate_confirmation_panel(self.prompts)
            if len(self.timing_prompt_ids) != TIMING_PROMPTS:
                raise ValueError("distributed confirmation requires 32 timing prompts")
        if self.stage == "diagnostic" and not slots:
            raise ValueError("distributed diagnostic requires prompts")
        if not 2 <= self.minimum_audit_replicas <= len(self.committee):
            raise ValueError("invalid minimum audit replicas")
        if (
            isinstance(self.audit_fraction, bool)
            or not isinstance(self.audit_fraction, (int, float))
            or not math.isfinite(self.audit_fraction)
            or not 0 < self.audit_fraction <= 1
        ):
            raise ValueError("audit fraction must be in (0, 1]")
        work_count = len(slots) + len(self.timing_prompt_ids)
        if self.audit_work_count is not None and (
            type(self.audit_work_count) is not int or not 1 <= self.audit_work_count <= work_count
        ):
            raise ValueError("audit work count is outside the panel")
        for tolerance in (self.quality_tolerance, self.timing_log_ratio_tolerance):
            if tolerance is not None and (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance)
                or tolerance < 0
            ):
                raise ValueError("audit tolerances must be finite non-negative numbers")
        object.__setattr__(
            self, "committee", MappingProxyType(dict(sorted(self.committee.items())))
        )
        object.__setattr__(
            self, "prompts", tuple(MappingProxyType(dict(row)) for row in self.prompts)
        )
        object.__setattr__(self, "timing_prompt_ids", tuple(self.timing_prompt_ids))
        object.__setattr__(self, "chain_context", MappingProxyType(dict(self.chain_context)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "round_id": self.round_id,
            "stage": self.stage,
            "entropy": self.entropy,
            "policy_id": self.policy_id,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "king_manifest_hash": self.king_manifest_hash,
            "wan50_manifest_hash": self.wan50_manifest_hash,
            "committee": dict(sorted(self.committee.items())),
            "prompts": [dict(row) for row in self.prompts],
            "timing_prompt_ids": list(self.timing_prompt_ids),
            "quality_tolerance": self.quality_tolerance,
            "timing_log_ratio_tolerance": self.timing_log_ratio_tolerance,
            "audit_calibration_hash": self.audit_calibration_hash,
            "minimum_audit_replicas": self.minimum_audit_replicas,
            "chain_context": dict(self.chain_context),
            "audit_fraction": self.audit_fraction,
            "audit_work_count": self.audit_work_count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DistributedEvaluationPlan:
        expected = {
            "schema",
            "round_id",
            "stage",
            "entropy",
            "policy_id",
            "candidate_manifest_hash",
            "king_manifest_hash",
            "wan50_manifest_hash",
            "committee",
            "prompts",
            "timing_prompt_ids",
            "quality_tolerance",
            "timing_log_ratio_tolerance",
            "audit_calibration_hash",
            "minimum_audit_replicas",
            "audit_fraction",
            "audit_work_count",
            "chain_context",
        }
        if set(raw) != expected or raw.get("schema") != SCHEMA:
            raise ValueError("distributed plan fields are not exact")
        values = {
            key: raw[key]
            for key in expected - {"schema", "committee", "prompts", "timing_prompt_ids"}
        }
        return cls(
            **values,
            committee=dict(raw["committee"]),
            prompts=tuple(raw["prompts"]),
            timing_prompt_ids=tuple(raw["timing_prompt_ids"]),
        )

    def plan_hash(self) -> str:
        return hash_canonical("forge/distributed-evaluation-plan/v1", self.to_dict())

    def allocation_seed(self) -> str:
        # Same panel and committee reuse king renders across candidates. Each signed
        # report still binds the candidate-specific complete plan hash.
        return hash_canonical(
            "forge/distributed-allocation-seed/v1",
            {
                "round_id": self.round_id,
                "stage": self.stage,
                "entropy": self.entropy,
                "committee": dict(self.committee),
                "prompts": [dict(p) for p in self.prompts],
                "timing_prompt_ids": list(self.timing_prompt_ids),
            },
        )

    @property
    def reward_ready(self) -> bool:
        # A hash is only a reference.  The certification route must verify the
        # referenced calibration receipt before it can enable a promotion.
        return False

    @property
    def audit_ready(self) -> bool:
        return (
            self.audit_calibration_hash is not None
            and self.quality_tolerance is not None
            and self.timing_log_ratio_tolerance is not None
        )


def allocate_work(plan: DistributedEvaluationPlan) -> tuple[AssignedWork, ...]:
    """Allocate equal validator loads, including deterministic audit replicas."""

    slots = _plan_slots(plan.prompts)
    work = [("quality", prompt_id, int(prompt["seed"])) for prompt_id, prompt in slots.items()]
    work.extend(
        ("timing", prompt_id, int(slots[prompt_id]["seed"])) for prompt_id in plan.timing_prompt_ids
    )
    work.sort(
        key=lambda item: hash_canonical(
            "forge/distributed-evaluation-order/v1", plan.allocation_seed(), item
        )
    )
    validators = sorted(plan.committee)
    audit_count = plan.audit_work_count or math.ceil(len(work) * plan.audit_fraction)
    audit_work = set(work[:audit_count])
    replica_count = plan.minimum_audit_replicas
    total_slots = len(work) + audit_count * (replica_count - 1)
    if total_slots % len(validators):
        raise ValueError("audit allocation cannot give every validator the same work count")
    assigned: list[AssignedWork] = []
    counts = {hotkey: 0 for hotkey in validators}
    for work_index, (kind, prompt_id, seed) in enumerate(work):
        identity = (kind, prompt_id, seed)
        copies = replica_count if identity in audit_work else 1
        ordered = sorted(
            validators,
            key=lambda hotkey: (
                counts[hotkey],
                hash_canonical(
                    "forge/distributed-evaluation-validator-order/v1",
                    plan.allocation_seed(),
                    work_index,
                    hotkey,
                ),
            ),
        )
        for hotkey in ordered[:copies]:
            assigned.append(
                AssignedWork(hotkey, kind, prompt_id, seed, audit=identity in audit_work)
            )
            counts[hotkey] += 1
    if len(set(counts.values())) != 1:
        raise AssertionError("deterministic allocation is not equal-load")
    return tuple(assigned)


def assignments_for(
    plan: DistributedEvaluationPlan, validator_hotkey: str
) -> tuple[AssignedWork, ...]:
    if validator_hotkey not in plan.committee:
        raise ValueError("validator is outside frozen committee")
    return tuple(item for item in allocate_work(plan) if item.validator_hotkey == validator_hotkey)
