"""Frozen bounded evaluation policy, dimensions and decision vocabulary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

QUALITY_AXES = (
    "prompt_adherence",
    "action_adherence",
    "camera_adherence",
    "subject_consistency",
    "background_consistency",
    "image_quality",
    "temporal_smoothness",
    "dynamic_degree",
)


CONTROL_KINDS = (
    "self",
    "same_model_new_run",
    "independent_seed",
    "static_scene",
    "freeze",
    "blur_mild",
    "blur_severe",
    "shuffle",
    "wrong_video",
    "detail_loss",
    "evaluator_attack_mild",
    "evaluator_attack_severe",
)


LEGITIMATE_CONTROLS = {"self", "same_model_new_run", "independent_seed", "static_scene"}


CONTROL_AXES = {
    "freeze": {"action_adherence", "temporal_smoothness", "dynamic_degree"},
    "blur_mild": {"image_quality"},
    "blur_severe": {"image_quality"},
    "shuffle": {"temporal_smoothness"},
    "wrong_video": {"prompt_adherence"},
    "detail_loss": {"image_quality", "subject_consistency"},
    "evaluator_attack_mild": {"prompt_adherence", "image_quality"},
    "evaluator_attack_severe": {"prompt_adherence", "image_quality"},
}


SCREEN_PROMPTS = 16


CONFIRMATION_PROMPTS = 128


TIMING_PROMPTS = 32


class Decision(StrEnum):
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"
    INCONCLUSIVE = "INCONCLUSIVE"
    PROMOTE_ELIGIBLE = "PROMOTE_ELIGIBLE"
    SCREEN_PASS = "SCREEN_PASS"
    SCREEN_FAIL = "SCREEN_FAIL"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationPolicy:
    """Frozen rule parameters.  A policy ID changes with every parameter."""

    version: str
    quality_margin: float
    min_speedup_ci: float
    calibration_sensitivity_lower: float
    calibration_false_alarm_upper: float
    calibration_alpha: float
    state: str
    threshold_derivation_hash: str | None
    independent_check_hash: str | None
    evaluator_provenance: Mapping[str, str] | None
    policy_id: str

    def body(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("policy_id")
        return body

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvaluationPolicy:
        if set(raw) != {
            "version",
            "quality_margin",
            "min_speedup_ci",
            "calibration_sensitivity_lower",
            "calibration_false_alarm_upper",
            "calibration_alpha",
            "state",
            "threshold_derivation_hash",
            "independent_check_hash",
            "evaluator_provenance",
            "policy_id",
        }:
            raise ValueError("MVP policy fields are not exact")
        policy = cls(**dict(raw))
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.version != "forge/mvp-evaluation-policy/v1":
            raise ValueError("unknown MVP policy version")
        if self.policy_id != _sha(self.body()):
            raise ValueError("policy id does not bind its policy")
        numeric = (
            self.quality_margin,
            self.min_speedup_ci,
            self.calibration_sensitivity_lower,
            self.calibration_false_alarm_upper,
            self.calibration_alpha,
        )
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in numeric):
            raise ValueError("policy numeric fields must be finite numbers")
        if not 0 <= self.quality_margin < 1:
            raise ValueError("quality margin must be in [0, 1)")
        if self.min_speedup_ci < 1.1:
            raise ValueError("MVP speed lower confidence bound must be at least 1.1x")
        if not 0 < self.calibration_sensitivity_lower <= 1:
            raise ValueError("invalid calibration sensitivity threshold")
        if not 0 <= self.calibration_false_alarm_upper < 1:
            raise ValueError("invalid calibration false-alarm threshold")
        if not 0 < self.calibration_alpha < 1:
            raise ValueError("invalid calibration alpha")
        if self.state not in {"development", "frozen"}:
            raise ValueError("policy state is invalid")
        if self.state == "frozen":
            if (
                not _is_hash(self.threshold_derivation_hash)
                or self.independent_check_hash is not None
            ):
                raise ValueError(
                    "frozen policy requires a derivation hash, not a future check hash"
                )
            if (
                not isinstance(self.evaluator_provenance, Mapping)
                or set(self.evaluator_provenance)
                != {"source_implementation_hash", "vbench_model_hashes", "luna_evaluator_id"}
                or not all(
                    isinstance(value, str) and value for value in self.evaluator_provenance.values()
                )
            ):
                raise ValueError("frozen policy requires evaluator provenance")


def freeze_policy(
    *,
    quality_margin: float = 0.125,
    min_speedup_ci: float = 1.1,
    calibration_sensitivity_lower: float = 0.8,
    calibration_false_alarm_upper: float = 0.15,
    calibration_alpha: float = 0.05,
    state: str = "development",
    threshold_derivation_hash: str | None = None,
    independent_check_hash: str | None = None,
    evaluator_provenance: Mapping[str, str] | None = None,
) -> EvaluationPolicy:
    body = {
        "version": "forge/mvp-evaluation-policy/v1",
        "quality_margin": quality_margin,
        "min_speedup_ci": min_speedup_ci,
        "calibration_sensitivity_lower": calibration_sensitivity_lower,
        "calibration_false_alarm_upper": calibration_false_alarm_upper,
        "calibration_alpha": calibration_alpha,
        "state": state,
        "threshold_derivation_hash": threshold_derivation_hash,
        "independent_check_hash": independent_check_hash,
        "evaluator_provenance": dict(evaluator_provenance) if evaluator_provenance else None,
    }
    policy = EvaluationPolicy(**body, policy_id=_sha(body))
    policy.validate()
    return policy


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
