"""Load admitted PEFT/Diffusers names without silently dropping the adapter.

Callers verify the manifest's file hashes first. Loading must preserve the
published alpha/rank and prove that every declared A/B matrix reached PEFT.
"""

from __future__ import annotations

import math
from pathlib import Path

from forge.models.weights import canonical_lora_name, lora_scaling, validate_lora_weights


def load_candidate_lora(pipe, directory: Path, filename: str) -> dict:
    import torch
    from safetensors.torch import load_file

    path = directory / filename
    admitted = validate_lora_weights(path)
    raw = load_file(path, device="cpu")
    tensors = {canonical_lora_name(key): value for key, value in raw.items()}
    scaling = lora_scaling(path, admitted.tensors)
    ranks = {target: rank for target, (rank, _) in scaling.items()}
    alphas = {target + ".alpha": alpha for target, (_, alpha) in scaling.items()}
    state = {key: value for key, value in tensors.items() if not key.endswith(".alpha")}
    adapter = "forge_candidate"
    pipe.transformer.load_lora_adapter(
        state,
        network_alphas=alphas,
        adapter_name=adapter,
        prefix="transformer",
        _pipeline=pipe,
    )
    if adapter not in getattr(pipe.transformer, "peft_config", {}):
        raise ValueError("LoRA loader did not activate the candidate")
    for target, rank in ranks.items():
        module = pipe.transformer.get_submodule(target.removeprefix("transformer."))
        if adapter not in getattr(module, "lora_A", {}):
            raise ValueError("LoRA target was not loaded")
        expected_scale = alphas[target + ".alpha"] / rank
        if not math.isclose(module.scaling[adapter], expected_scale, rel_tol=1e-7):
            raise ValueError("LoRA loader changed alpha/rank scaling")
        for side in ("A", "B"):
            actual = getattr(module, "lora_" + side)[adapter].weight.detach()
            expected = tensors[target + ".lora_" + side + ".weight"].to(actual)
            if not torch.equal(actual, expected):
                raise ValueError("LoRA loader did not preserve the declared matrix")
    return {
        "adapter": adapter,
        "loaded_targets": len(ranks),
        "scales": sorted({alphas[target + ".alpha"] / rank for target, rank in ranks.items()}),
    }
