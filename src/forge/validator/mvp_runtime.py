"""Trusted bridge from a frozen MVP panel to validator-owned Wan work.

This module creates work descriptions only after authenticating the frozen
competition bindings and public preflight.  It never accepts miner code and it
does not make a development flag a substitute for certified history.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from forge.evaluation.mvp_evidence import freeze_evaluation_plan
from forge.evaluation.mvp_policy import (
    CONFIRMATION_PROMPTS,
    SCREEN_PROMPTS,
    TIMING_PROMPTS,
    EvaluationPolicy,
)
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.runtime_preflight import validate_preflight_receipt

CertifiedBindingVerifier = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class MVPWork:
    """Concrete validator jobs, frozen before any runtime paths are bound."""

    plan_id: str
    stage: str
    policy_id: str
    candidate_manifest_hash: str
    incumbent_manifest_hash: str
    wan50_manifest_hash: str
    contract_hash: str
    quality_prompts: tuple[dict[str, Any], ...]
    timing_prompts: tuple[dict[str, Any], ...]
    binding_hash: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "forge/mvp-runtime-work/v1",
            "plan_id": self.plan_id,
            "stage": self.stage,
            "policy_id": self.policy_id,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "incumbent_manifest_hash": self.incumbent_manifest_hash,
            "wan50_manifest_hash": self.wan50_manifest_hash,
            "contract_hash": self.contract_hash,
            "quality_prompt_ids": [row["prompt_id"] for row in self.quality_prompts],
            "timing_prompt_ids": [row["prompt_id"] for row in self.timing_prompts],
            "binding_hash": self.binding_hash,
        }


def _hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value


def _runtime_prompts(prompts: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows = []
    for prompt in prompts:
        prompt_id, text, seed = (
            prompt.get("prompt_id"),
            prompt.get("text"),
            prompt.get("seed"),
        )
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or not isinstance(text, str)
            or not text
            or type(seed) is not int
        ):
            raise ValueError("frozen MVP prompt cannot become a runtime slot")
        rows.append({"prompt_id": prompt_id, "text": text, "video_seeds": [seed]})
    return tuple(rows)


def _timing_sample(
    plan_id: str, prompts: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    if len(prompts) < TIMING_PROMPTS:
        raise ValueError("MVP panel cannot supply 32 timing prompts")
    ranked = sorted(
        prompts,
        key=lambda row: hash_canonical(
            "forge/mvp-timing-slot/v1", plan_id, row["prompt_id"], row["seed"]
        ),
    )
    return _runtime_prompts(ranked[:TIMING_PROMPTS])


def prepare_mvp_work(
    *,
    plan: Mapping[str, Any],
    policy: EvaluationPolicy,
    contract: ChallengeContract,
    preflight_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any] | None,
    candidate: ModelManifest,
    incumbent: ModelManifest,
    wan50: ModelManifest,
    certified_binding: Mapping[str, Any] | None,
    verify_certified_binding: CertifiedBindingVerifier | None,
) -> MVPWork:
    """Validate public admission and certified history before preparing GPU work."""
    policy.validate()
    if policy.state != "frozen":
        raise ValueError("an unfrozen MVP policy cannot prepare validator runtime work")
    if verify_certified_binding is None or certified_binding is None:
        raise ValueError("certified competition binding is required before runtime work")
    if not verify_certified_binding(certified_binding):
        raise ValueError("certified competition binding did not verify")
    stage = plan.get("stage")
    if stage not in {"screening", "confirmation"}:
        raise ValueError("MVP runtime stage is invalid")
    expected = freeze_evaluation_plan(
        stage=stage,
        prompts=plan.get("prompts", []),
        policy=policy,
        candidate_manifest_hash=candidate.fingerprint(),
        incumbent_manifest_hash=incumbent.fingerprint(),
        wan50_manifest_hash=wan50.fingerprint(),
    )
    if dict(plan) != expected:
        raise ValueError("MVP runtime plan is not the exact frozen evaluation plan")
    for key, manifest in (
        ("candidate_manifest_hash", candidate),
        ("incumbent_manifest_hash", incumbent),
        ("wan50_manifest_hash", wan50),
    ):
        _hash(plan.get(key), key)
        if plan[key] != manifest.fingerprint():
            raise ValueError(f"{key} does not bind the supplied model")
    validate_preflight_receipt(
        preflight_receipt,
        preflight_plan,
        contract,
        candidate_manifest_hash=candidate.fingerprint(),
    )
    required_binding = {
        "candidate_manifest_hash": candidate.fingerprint(),
        "incumbent_manifest_hash": incumbent.fingerprint(),
        "evaluation_plan_id": plan["plan_id"],
    }
    if any(certified_binding.get(key) != value for key, value in required_binding.items()):
        raise ValueError("certified binding does not match candidate, incumbent, or frozen plan")
    for key in ("admission_history_root", "prompt_history_root", "certified_state_hash"):
        _hash(certified_binding.get(key), key)
    quality = _runtime_prompts(plan["prompts"])
    if len(quality) != (SCREEN_PROMPTS if stage == "screening" else CONFIRMATION_PROMPTS):
        raise ValueError("MVP runtime quality panel has an unexpected size")
    timing = (
        _timing_sample(plan["plan_id"], plan["prompts"]) if stage == "confirmation" else quality
    )
    return MVPWork(
        plan_id=plan["plan_id"],
        stage=stage,
        policy_id=policy.policy_id,
        candidate_manifest_hash=candidate.fingerprint(),
        incumbent_manifest_hash=incumbent.fingerprint(),
        wan50_manifest_hash=wan50.fingerprint(),
        contract_hash=contract.contract_hash(),
        quality_prompts=quality,
        timing_prompts=timing,
        binding_hash=hash_canonical("forge/mvp-runtime-binding/v1", certified_binding),
    )


def timing_rows_from_paired_receipt(
    work: MVPWork, paired: Mapping[str, Any], *, dependency_lock_hash: str
) -> list[dict[str, Any]]:
    """Translate the runtime's ABBA receipt into ``evaluate_mvp`` timing rows."""
    _hash(dependency_lock_hash, "dependency lock hash")
    if work.stage != "confirmation" or paired.get("schema") != "forge/paired-timing/v1":
        raise ValueError("only confirmation paired timing can produce MVP timing rows")
    runtime, recipes, hardware = (
        paired.get("runtime"),
        paired.get("recipes"),
        paired.get("hardware"),
    )
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(recipes, Mapping)
        or not isinstance(hardware, Mapping)
        or runtime.get("timing_scope", {}).get("competitive_includes")
        != ["text_encoding", "denoising", "vae_decoding"]
    ):
        raise ValueError("paired runtime receipt lacks full-pipeline provenance")
    slots = {row["prompt_id"]: row for row in work.timing_prompts}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in paired.get("measurements", []):
        if not isinstance(item, Mapping) or item.get("prompt_id") not in slots:
            raise ValueError("paired timing receipt contains an unexpected slot")
        grouped.setdefault(item["prompt_id"], []).append(item)
    if set(grouped) != set(slots) or any(len(rows) != 4 for rows in grouped.values()):
        raise ValueError("paired timing receipt does not cover exactly the frozen 32 slots")
    result = []
    for prompt_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: item.get("run_order"))
        if [item.get("side") for item in ordered] != [
            "reference",
            "candidate",
            "candidate",
            "reference",
        ]:
            raise ValueError("paired timing receipt is not ABBA")
        runs = []
        for item in ordered:
            if (
                item.get("status") != "OK"
                or item.get("gpu_synchronized") is not True
                or item.get("full_pipeline_seconds") != item.get("wall_time_seconds")
            ):
                raise ValueError("paired timing receipt has failed or partial timing")
            side = item["side"]
            runs.append(
                {
                    "model": "reference" if side == "reference" else "candidate",
                    "seconds": item["full_pipeline_seconds"],
                    "gpu_synchronized": True,
                    "video_sha256": paired["report_video_sha256"][side][
                        f"{prompt_id}:{item['seed']}"
                    ],
                    "recipe_hash": hash_canonical("forge/mvp-runtime-recipe/v1", recipes[side]),
                }
            )
        result.append(
            {
                "prompt_id": prompt_id,
                "seed": slots[prompt_id]["video_seeds"][0],
                "candidate_manifest_hash": work.candidate_manifest_hash,
                "reference_manifest_hash": work.incumbent_manifest_hash,
                "gpu_synchronized": True,
                "timing_scope": "text_encode+denoise+decode",
                "hardware": {
                    "gpu": hardware["gpu_name"],
                    "precision": runtime["dtype"],
                    "dependency_lock_hash": dependency_lock_hash,
                },
                "recipes": {
                    "reference": hash_canonical(
                        "forge/mvp-runtime-recipe/v1", recipes["reference"]
                    ),
                    "candidate": hash_canonical(
                        "forge/mvp-runtime-recipe/v1", recipes["candidate"]
                    ),
                },
                "warmup": {"reference": True, "candidate": True},
                "runs": runs,
            }
        )
    return result
