"""Versioned challenge contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from forge.protocol.hashing import hash_canonical

DEFAULT_AXES = (
    "prompt_adherence",
    "motion_adherence",
    "camera_adherence",
    "subject_consistency",
    "temporal_stability",
    "motion_quality",
    "visual_quality",
    "physical_plausibility",
)


@dataclass(frozen=True)
class PropertySlice:
    name: str
    property_name: str
    value: str | int
    noninferiority_margin: float = 0.02

    def __post_init__(self) -> None:
        if not self.name or not self.property_name:
            raise ValueError("slice name and property name are required")
        if self.noninferiority_margin < 0:
            raise ValueError("slice margin must be non-negative")


def _default_weights() -> dict[str, float]:
    return {axis: 1.0 / len(DEFAULT_AXES) for axis in DEFAULT_AXES}


@dataclass(frozen=True)
class ChallengeContract:
    base_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    base_model_revision: str = "0fad780a534b6463e45facd96134c9f345acfa5b"
    format: str = "safetensors"
    max_steps: int = 16
    reference_steps: int = 50
    max_forward_passes: int = 32
    width: int = 832
    height: int = 480
    frames: int = 81
    fps: int = 16
    scheduler_shift: float = 3.0
    min_speedup: float = 1.1
    wall_time_limit_seconds: float = 900.0
    vram_limit_bytes: int = 24 * 1024**3
    axes: tuple[str, ...] = DEFAULT_AXES
    axis_weights: dict[str, float] = field(default_factory=_default_weights)
    score_mean_weight: float = 0.6
    score_worst_weight: float = 0.4
    worst_axis_count: int = 3
    screen_prompts: int = 32
    screen_seeds: int = 1
    duel_prompts: int = 256
    duel_seeds: int = 2
    confirmation_prompts: int = 512
    confirmation_seeds: int = 2
    min_effect: float = 0.02
    alpha: float = 0.05
    max_candidates: int = 16
    protected_axes: tuple[str, ...] = DEFAULT_AXES
    protected_slices: tuple[PropertySlice, ...] = ()
    axis_noninferiority_margin: float = 0.02
    failure_rate_cap: float = 0.05
    grammar_version: str = "forge/bench/grammar/v1"

    def __post_init__(self) -> None:
        if not self.base_model_id or not self.base_model_revision:
            raise ValueError("base model id and revision are required")
        if (
            self.width % 16
            or self.height % 16
            or min(self.width, self.height) < 32
            or self.frames < 5
            or (self.frames - 1) % 4
            or self.fps <= 0
        ):
            raise ValueError("invalid Wan geometry")
        if self.reference_steps <= 0 or self.max_forward_passes <= 0:
            raise ValueError("reference and forward-pass limits must be positive")
        if self.min_speedup <= 1 or self.scheduler_shift <= 0:
            raise ValueError("speedup must exceed 1 and scheduler shift must be positive")
        if self.format != "safetensors":
            raise ValueError("only safetensors submissions are supported")
        if self.max_steps <= 0 or self.wall_time_limit_seconds <= 0 or self.vram_limit_bytes <= 0:
            raise ValueError("resource limits must be positive")
        if not self.axes or len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must be non-empty and unique")
        if set(self.axis_weights) != set(self.axes):
            raise ValueError("axis_weights must contain every axis exactly once")
        if any(weight < 0 for weight in self.axis_weights.values()):
            raise ValueError("axis weights cannot be negative")
        if abs(sum(self.axis_weights.values()) - 1.0) > 1e-12:
            raise ValueError("axis weights must sum to 1")
        if self.score_mean_weight < 0 or self.score_worst_weight < 0:
            raise ValueError("score formula weights cannot be negative")
        if abs(self.score_mean_weight + self.score_worst_weight - 1.0) > 1e-12:
            raise ValueError("score formula weights must sum to 1")
        if not 1 <= self.worst_axis_count <= len(self.axes):
            raise ValueError("worst_axis_count is outside the axis range")
        for size in self.stage_sizes().values():
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError("stage sizes must be positive")
        if not 0 < self.alpha < 1 or self.max_candidates <= 0:
            raise ValueError("alpha and max_candidates must be positive and bounded")
        if self.min_effect < 0 or self.axis_noninferiority_margin < 0:
            raise ValueError("statistical margins must be non-negative")
        if not set(self.protected_axes).issubset(self.axes):
            raise ValueError("protected axes must be contract axes")
        if not 0 <= self.failure_rate_cap <= 1:
            raise ValueError("failure rate cap must be in [0,1]")
        if not self.grammar_version.startswith("forge/"):
            raise ValueError("grammar version must be forge-domain-separated")
        object.__setattr__(self, "axis_weights", MappingProxyType(dict(self.axis_weights)))

    def stage_sizes(self) -> dict[str, tuple[int, int]]:
        return {
            "screen": (self.screen_prompts, self.screen_seeds),
            "duel": (self.duel_prompts, self.duel_seeds),
            "confirmation": (self.confirmation_prompts, self.confirmation_seeds),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "format": self.format,
            "max_steps": self.max_steps,
            "reference_steps": self.reference_steps,
            "max_forward_passes": self.max_forward_passes,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            "scheduler_shift": self.scheduler_shift,
            "min_speedup": self.min_speedup,
            "wall_time_limit_seconds": self.wall_time_limit_seconds,
            "vram_limit_bytes": self.vram_limit_bytes,
            "axes": self.axes,
            "axis_weights": dict(self.axis_weights),
            "score_mean_weight": self.score_mean_weight,
            "score_worst_weight": self.score_worst_weight,
            "worst_axis_count": self.worst_axis_count,
            "stage_sizes": self.stage_sizes(),
            "min_effect": self.min_effect,
            "alpha": self.alpha,
            "max_candidates": self.max_candidates,
            "protected_axes": self.protected_axes,
            "protected_slices": [
                {
                    "name": item.name,
                    "property_name": item.property_name,
                    "value": item.value,
                    "noninferiority_margin": item.noninferiority_margin,
                }
                for item in self.protected_slices
            ],
            "axis_noninferiority_margin": self.axis_noninferiority_margin,
            "failure_rate_cap": self.failure_rate_cap,
            "grammar_version": self.grammar_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChallengeContract:
        values = dict(value)
        for stage, sizes in values.pop("stage_sizes", {}).items():
            if stage not in {"screen", "duel", "confirmation"}:
                raise ValueError("unknown stage")
            values[f"{stage}_prompts"], values[f"{stage}_seeds"] = sizes
        for name in ("axes", "protected_axes"):
            if name in values:
                values[name] = tuple(values[name])
        if "protected_slices" in values:
            values["protected_slices"] = tuple(
                PropertySlice(**item) for item in values["protected_slices"]
            )
        return cls(**values)

    def contract_hash(self) -> str:
        return hash_canonical("forge/contract/v1", self.to_dict())
