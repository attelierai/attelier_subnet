"""Shared generation recipe and reference scheduler contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from forge.models.artifact_manifest import ModelManifest
from forge.protocol.contract import ChallengeContract


@dataclass(frozen=True)
class GenerationRecipe:
    steps: int = 16
    guidance_scale: float = 5.0
    scheduler: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> GenerationRecipe:
        if set(value) - {"steps", "guidance_scale", "scheduler"}:
            raise ValueError(
                "only steps, guidance_scale and the validator scheduler may vary between candidates"
            )
        result = cls(**value)
        if result.scheduler is not None and (
            not isinstance(result.scheduler, str)
            or result.scheduler not in {"uniform_euler", "original_unipc"}
        ):
            raise ValueError("scheduler must be uniform_euler or original_unipc")
        if type(result.steps) is not int or result.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(result.guidance_scale) or not 1 <= result.guidance_scale <= 10:
            raise ValueError("guidance_scale must be finite and in [1, 10]")
        return result

    def to_dict(self) -> dict[str, Any]:
        result = {"steps": self.steps, "guidance_scale": self.guidance_scale}
        if self.scheduler is not None:
            result["scheduler"] = self.scheduler
        return result


def effective_generation_recipe(
    recipe: GenerationRecipe, *, is_original_reference: bool
) -> dict[str, int | float | str]:
    """Return the complete recipe the validator will actually execute.

    Older manifests may omit ``scheduler``.  The omission is an input
    compatibility feature, not an execution ambiguity: non-bootstrap models
    run with ``uniform_euler`` while the exact original pipeline keeps its
    ``original_unipc`` scheduler.  Consumers that compare recipes must bind
    this resolved value instead of the sparse manifest input.
    """

    return {
        "steps": recipe.steps,
        "guidance_scale": recipe.guidance_scale,
        "scheduler": recipe.scheduler
        if recipe.scheduler is not None
        else ("original_unipc" if is_original_reference else "uniform_euler"),
    }


def timing_scope() -> dict[str, Any]:
    """The competitive clock is deliberately narrower than operating cost.

    The validator starts after a verified, resident model has been selected and
    stops only after decoded frames are available.  Text encoding, denoising
    and VAE decoding are therefore all in the competitive number; public
    download, hash verification, loading, adapter switching and MP4 writing
    remain attributable operational cost rather than a model speed claim.
    """
    return {
        "schema": "forge/timing-scope/v1",
        "competitive_includes": ["text_encoding", "denoising", "vae_decoding"],
        "competitive_excludes": [
            "artifact_download",
            "artifact_hash_verification",
            "model_loading",
            "adapter_or_transformer_switching",
            "mp4_encoding",
        ],
        "clock": "cuda_synchronized_perf_counter",
    }


def paired_reference_scheduler(
    reference_model: ModelManifest,
    base: ModelManifest,
    recipe: GenerationRecipe,
    contract: ChallengeContract,
) -> str:
    """Resolve the committed validator-owned sampler, preserving legacy bootstrap.

    Explicit recipes select only a fixed library implementation and the pinned
    base configuration. Historical manifests retain their original defaults.
    """
    original_reference = (
        reference_model.role == "pipeline"
        and reference_model.reference == base.reference
        and recipe.steps == contract.reference_steps
        and recipe.guidance_scale == 5.0
    )
    return str(
        effective_generation_recipe(recipe, is_original_reference=original_reference)["scheduler"]
    )


def validate_prompts(prompts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    if not prompts:
        raise ValueError("the prescribed prompt panel is empty")
    for prompt in prompts:
        identity = prompt["prompt_id"]
        seeds = prompt["video_seeds"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("prompt ids must be unique nonempty strings")
        if not isinstance(prompt["text"], str) or not prompt["text"].strip():
            raise ValueError("prompt text must be nonempty")
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
        ):
            raise ValueError("video seeds must be unique nonnegative 63-bit integers")
        seen.add(identity)
    return prompts
