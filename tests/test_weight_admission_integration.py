import json
import struct
from pathlib import Path

import pytest

from forge.models.artifact_manifest import (
    ArtifactError,
    ArtifactReference,
    ManifestFile,
    ModelManifest,
)
from forge.models.weights import validate_candidate_weights, validate_lora_weights


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]):
    offset = 0
    header = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        data.extend(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)


def manifest(role: str, files: tuple[ManifestFile, ...], base=None):
    return ModelManifest(
        ArtifactReference("hf", "public/candidate", "a" * 40),
        role,
        files,
        "apache-2.0",
        {"steps": 4, "guidance_scale": 5.0},
        base,
    )


def file(path: str, payload: bytes) -> ManifestFile:
    import hashlib

    return ManifestFile(path, hashlib.sha256(payload).hexdigest(), len(payload))


def lora_payloads(prefix: str):
    return {
        f"{prefix}block.attn.lora_A.weight": ("F32", (2, 3), struct.pack("<6f", *range(6))),
        f"{prefix}block.attn.lora_B.weight": ("F32", (4, 2), struct.pack("<8f", *range(8))),
    }


def test_repackaged_lora_aliases_share_a_tensor_content_identity(tmp_path):
    first = tmp_path / "one.safetensors"
    second = tmp_path / "two.safetensors"
    write_safetensors(first, lora_payloads("base_model.model.transformer."))
    write_safetensors(second, lora_payloads("transformer."))
    assert (
        validate_lora_weights(first).content_fingerprint
        == validate_lora_weights(second).content_fingerprint
    )



def test_peft_config_scaling_is_admitted_and_bound_to_tensor_identity(tmp_path):
    weights = tmp_path / "adapter_model.safetensors"
    write_safetensors(weights, lora_payloads("base_model.model."))
    default = validate_lora_weights(weights).content_fingerprint
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"r": 2, "lora_alpha": 2, "use_dora": False}))
    assert validate_lora_weights(weights).content_fingerprint == default
    config.write_text(json.dumps({"r": 2, "lora_alpha": 1}))
    assert validate_lora_weights(weights).content_fingerprint != default
    config.write_text(json.dumps({"r": 3, "lora_alpha": 1}))
    with pytest.raises(ArtifactError, match="rank differs"):
        validate_lora_weights(weights)
    config.write_text(json.dumps({"r": 2, "lora_alpha": 1, "use_rslora": True}))
    with pytest.raises(ArtifactError, match="unsupported LoRA"):
        validate_lora_weights(weights)

def test_lora_alias_collision_and_nonfinite_values_are_rejected(tmp_path):
    collision = tmp_path / "collision.safetensors"
    values = lora_payloads("transformer.")
    values.update(lora_payloads("base_model.model.transformer."))
    write_safetensors(collision, values)
    with pytest.raises(ArtifactError, match="aliases"):
        validate_lora_weights(collision)
    broken = tmp_path / "nan.safetensors"
    values = lora_payloads("transformer.")
    values["transformer.block.attn.lora_A.weight"] = (
        "F32",
        (2, 3),
        struct.pack("<6f", *([float("nan")] * 6)),
    )
    write_safetensors(broken, values)
    with pytest.raises(ArtifactError, match="non-finite"):
        validate_lora_weights(broken)


def test_full_transformer_cannot_change_tensor_shape_or_architecture(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    base_weights = struct.pack("<6f", *range(6))
    candidate_weights = struct.pack("<6f", *(x + 1 for x in range(6)))
    write_safetensors(
        base_root / "transformer/model.safetensors", {"block.weight": ("F32", (2, 3), base_weights)}
    )
    write_safetensors(
        candidate_root / "transformer/model.safetensors",
        {"block.weight": ("F32", (2, 3), candidate_weights)},
    )
    (base_root / "transformer/config.json").write_text(
        json.dumps({"hidden_size": 3, "_class_name": "WanTransformer3DModel"})
    )
    (candidate_root / "transformer/config.json").write_text(
        json.dumps({"hidden_size": 3, "_class_name": "Other"})
    )
    base = manifest(
        "pipeline",
        (
            file("transformer/model.safetensors", base_weights),
            file("transformer/config.json", (base_root / "transformer/config.json").read_bytes()),
        ),
    )
    candidate = manifest(
        "full_transformer",
        (
            file("transformer/model.safetensors", candidate_weights),
            file(
                "transformer/config.json", (candidate_root / "transformer/config.json").read_bytes()
            ),
        ),
        base.reference,
    )
    assert validate_candidate_weights(candidate, candidate_root, base, base_root).tensors
    (candidate_root / "transformer/config.json").write_text(json.dumps({"hidden_size": 4}))
    with pytest.raises(ArtifactError, match="architecture"):
        validate_candidate_weights(candidate, candidate_root, base, base_root)
    write_safetensors(
        candidate_root / "transformer/model.safetensors",
        {"block.weight": ("F32", (3, 2), candidate_weights)},
    )
    with pytest.raises(ArtifactError, match="names, dtypes, or shapes"):
        validate_candidate_weights(candidate, candidate_root, base, base_root)


def test_recipe_cannot_inject_runtime_options_or_nonfinite_values():
    weights = b"x"
    candidate_file = file("adapter/model.safetensors", weights)
    base = ArtifactReference("hf", "public/base", "b" * 40)
    with pytest.raises(ArtifactError, match="scheduler must be"):
        ModelManifest(
            ArtifactReference("hf", "public/candidate", "a" * 40),
            "lora",
            (candidate_file,),
            "apache-2.0",
            {"steps": 4, "guidance_scale": 5.0, "scheduler": "attacker"},
            base,
        )
    with pytest.raises(ArtifactError, match="permits only"):
        ModelManifest(
            ArtifactReference("hf", "public/candidate", "a" * 40),
            "lora",
            (candidate_file,),
            "apache-2.0",
            {"steps": 4, "guidance_scale": 5.0, "shift": 3.0},
            base,
        )
    with pytest.raises(ArtifactError, match="finite"):
        ModelManifest(
            ArtifactReference("hf", "public/candidate", "a" * 40),
            "lora",
            (candidate_file,),
            "apache-2.0",
            {"steps": 4, "guidance_scale": float("nan")},
            base,
        )


def test_lora_requires_float_positive_rank_and_real_base_target(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    base_weights = struct.pack("<6f", *range(6))
    write_safetensors(
        base_root / "transformer/model.safetensors",
        {"block.attn.weight": ("F32", (2, 3), base_weights)},
    )
    (base_root / "transformer/config.json").parent.mkdir(parents=True, exist_ok=True)
    (base_root / "transformer/config.json").write_text("{}")
    base = manifest(
        "pipeline",
        (
            file("transformer/model.safetensors", base_weights),
            file("transformer/config.json", b"{}"),
        ),
    )
    adapter = lora_payloads("transformer.foo.")
    payload = b"".join(value[2] for value in adapter.values())
    write_safetensors(candidate_root / "adapter/model.safetensors", adapter)
    candidate = manifest("lora", (file("adapter/model.safetensors", payload),), base.reference)
    with pytest.raises(ArtifactError, match="not a base transformer weight"):
        validate_candidate_weights(candidate, candidate_root, base, base_root)
    rank_zero = {
        "transformer.block.attn.lora_A.weight": ("F32", (0, 3), b""),
        "transformer.block.attn.lora_B.weight": ("F32", (2, 0), b""),
    }
    write_safetensors(candidate_root / "adapter/model.safetensors", rank_zero)
    with pytest.raises(ArtifactError, match="rank shapes"):
        validate_candidate_weights(candidate, candidate_root, base, base_root)
