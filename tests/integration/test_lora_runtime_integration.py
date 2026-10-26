"""Actual safetensors -> Diffusers/PEFT -> projection -> clean-base round trip.

Run in the pinned GPU environment. No downloaded model is needed for this
library-boundary regression; the public Wan run supplies full-model evidence.
"""

import json

import pytest

from forge.models.lora_loading import load_candidate_lora
from forge.models.weights import validate_lora_weights


@pytest.mark.parametrize("packaging", ["peft_config", "diffusers_metadata", "tensor_alpha"])
def test_adapter_files_change_real_projection_and_unload_cleanly(tmp_path, packaging):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import WanPipeline, WanTransformer3DModel
    from safetensors.torch import save_file

    torch.manual_seed(7)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
    )
    pipe = WanPipeline(None, None, None, None, transformer=model)
    target = "blocks.0.attn1.to_q"
    projection = model.get_submodule(target)
    x = torch.randn(3, 16)
    clean = projection(x).detach().clone()
    a, b = torch.randn(2, 16), torch.randn(16, 2)
    prefix = "base_model.model." if packaging == "peft_config" else "transformer."
    tensors = {prefix + target + ".lora_A.weight": a, prefix + target + ".lora_B.weight": b}
    metadata = None
    if packaging == "peft_config":
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 1}))
    elif packaging == "diffusers_metadata":
        metadata = {
            "lora_adapter_metadata": json.dumps({"transformer.r": 2, "transformer.lora_alpha": 1})
        }
    else:
        tensors[prefix + target + ".alpha"] = torch.tensor(1.0)
    path = tmp_path / "adapter_model.safetensors"
    save_file(tensors, path, metadata=metadata)
    receipt = load_candidate_lora(pipe, tmp_path, path.name)
    assert receipt["loaded_targets"] == 1 and receipt["scales"] == [0.5]
    actual = model.get_submodule(target)(x)
    expected = clean + 0.5 * (x @ a.T @ b.T)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, clean)
    pipe.unload_lora_weights()
    torch.testing.assert_close(model.get_submodule(target)(x), clean, rtol=0, atol=0)
    if packaging == "peft_config":
        first = validate_lora_weights(path).content_fingerprint
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}))
        assert validate_lora_weights(path).content_fingerprint != first
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"r": 2, "lora_alpha": 1, "use_dora": True})
        )
        with pytest.raises(ValueError, match="unsupported LoRA"):
            load_candidate_lora(pipe, tmp_path, path.name)
