"""Bounded safetensors admission checks and exact effective-tensor identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.models.artifact_manifest import ArtifactError, ModelManifest
from forge.protocol.hashing import hash_canonical

MAX_HEADER_BYTES = 16 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".alpha")
_HARMLESS_CONFIG_KEYS = {"_class_name", "_diffusers_version", "_name_or_path"}


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CandidateWeights:
    content_fingerprint: str
    tensors: tuple[TensorDescriptor, ...]


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate safetensors header key: {key}")
        result[key] = value
    return result


def _header(path: Path) -> tuple[dict[str, Any], int, int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArtifactError("safetensors file is missing its header length")
        header_bytes = int.from_bytes(prefix, "little")
        if not 2 <= header_bytes <= MAX_HEADER_BYTES:
            raise ArtifactError("safetensors header length is outside the admission limit")
        encoded = handle.read(header_bytes)
    if len(encoded) != header_bytes:
        raise ArtifactError("truncated safetensors header")
    try:
        header = json.loads(encoded, object_pairs_hook=_no_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ArtifactError("safetensors header must be an object")
    return header, 8 + header_bytes, path.stat().st_size


def _descriptor_records(path: Path) -> list[tuple[TensorDescriptor, int, int]]:
    header, data_start, total_bytes = _header(path)
    records: list[tuple[TensorDescriptor, int, int]] = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, Mapping) or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in value.items()
            ):
                raise ArtifactError("safetensors metadata must be string-to-string")
            continue
        if not isinstance(name, str) or not name or not isinstance(value, Mapping):
            raise ArtifactError("invalid safetensors tensor record")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or not all(type(item) is int and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(item) is int and item >= 0 for item in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ArtifactError(f"invalid safetensors descriptor for {name}")
        count = math.prod(shape)
        expected = count * _DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected:
            raise ArtifactError(f"safetensors shape and byte range disagree for {name}")
        if data_start + offsets[1] > total_bytes:
            raise ArtifactError(f"safetensors tensor exceeds file bounds for {name}")
        records.append((TensorDescriptor(name, dtype, tuple(shape), ""), offsets[0], offsets[1]))
    if not records:
        raise ArtifactError("safetensors file contains no tensors")
    ordered = sorted(records, key=lambda item: item[1])
    cursor = 0
    for _, start, end in ordered:
        if start != cursor:
            raise ArtifactError("safetensors byte ranges must be contiguous and non-overlapping")
        cursor = end
    if data_start + cursor != total_bytes:
        raise ArtifactError("safetensors has undeclared trailing bytes")
    return records


def _finite(dtype: str, data: bytes) -> bool:
    if dtype == "F16":
        return bool(np.isfinite(np.frombuffer(data, dtype="<f2")).all())
    if dtype == "F32":
        return bool(np.isfinite(np.frombuffer(data, dtype="<f4")).all())
    if dtype == "F64":
        return bool(np.isfinite(np.frombuffer(data, dtype="<f8")).all())
    if dtype == "BF16":
        return bool(((np.frombuffer(data, dtype="<u2") & 0x7F80) != 0x7F80).all())
    return True


def describe_safetensors(path: str | Path) -> tuple[TensorDescriptor, ...]:
    """Parse safetensors without torch and hash every tensor in bounded chunks."""

    source = Path(path)
    if source.suffix != ".safetensors" or not source.is_file() or source.is_symlink():
        raise ArtifactError("candidate weights must be a regular .safetensors file")
    descriptors: list[TensorDescriptor] = []
    _, data_start, _ = _header(source)
    with source.open("rb") as handle:
        for descriptor, start, end in _descriptor_records(source):
            handle.seek(data_start + start)
            digest = hashlib.sha256()
            remaining = end - start
            unit = _DTYPE_BYTES[descriptor.dtype]
            while remaining:
                chunk = handle.read(min(1024 * 1024 // unit * unit, remaining))
                if not chunk:
                    raise ArtifactError(f"truncated safetensors tensor {descriptor.name}")
                if not _finite(descriptor.dtype, chunk):
                    raise ArtifactError(f"non-finite tensor values in {descriptor.name}")
                digest.update(chunk)
                remaining -= len(chunk)
            descriptors.append(
                TensorDescriptor(
                    descriptor.name, descriptor.dtype, descriptor.shape, digest.hexdigest()
                )
            )
    return tuple(descriptors)


def _content_fingerprint(
    descriptors: Iterable[TensorDescriptor], normalize: Callable[[str], str]
) -> CandidateWeights:
    canonical: dict[str, TensorDescriptor] = {}
    for item in descriptors:
        name = normalize(item.name)
        if name in canonical:
            raise ArtifactError(f"multiple tensor aliases resolve to {name}")
        canonical[name] = TensorDescriptor(name, item.dtype, item.shape, item.sha256)
    records = tuple(canonical[name] for name in sorted(canonical))
    return CandidateWeights(
        hash_canonical("forge/tensor-content/v1", [item.to_dict() for item in records]), records
    )


def canonical_lora_name(name: str) -> str:
    value = name
    for prefix in ("base_model.model.", "base_model.", "model.", "transformer."):
        while value.startswith(prefix):
            value = value[len(prefix) :]
    if not value.endswith(_LORA_SUFFIXES):
        raise ArtifactError(f"unsupported LoRA tensor name: {name}")
    return f"transformer.{value}"


def _base_transformer_descriptors(paths: Iterable[Path]) -> dict[str, TensorDescriptor]:
    descriptors: dict[str, TensorDescriptor] = {}
    for path in paths:
        for item, _, _ in _descriptor_records(path):
            if item.name in descriptors:
                raise ArtifactError(f"duplicate base transformer tensor: {item.name}")
            descriptors[item.name] = item
    return descriptors


def _alpha_value(path: Path, name: str) -> float:
    _, data_start, _ = _header(path)
    for descriptor, start, end in _descriptor_records(path):
        if descriptor.name != name:
            continue
        if descriptor.dtype not in {"F16", "BF16", "F32", "F64"} or descriptor.shape not in {
            (),
            (1,),
        }:
            raise ArtifactError(f"LoRA alpha must be a floating scalar: {name}")
        with path.open("rb") as handle:
            handle.seek(data_start + start)
            data = handle.read(end - start)
        if descriptor.dtype == "BF16":
            value = float(
                np.array([int.from_bytes(data, "little") << 16], dtype="<u4").view("<f4")[0]
            )
        else:
            dtype = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}[descriptor.dtype]
            value = float(np.frombuffer(data, dtype=dtype)[0])
        if not math.isfinite(value) or not 0 < value <= 1024:
            raise ArtifactError(f"LoRA alpha must be finite and in (0, 1024]: {name}")
        return value
    raise ArtifactError(f"missing LoRA alpha tensor: {name}")


def validate_lora_weights(
    path: str | Path, base_transformer_paths: Iterable[Path] | None = None
) -> CandidateWeights:
    source = Path(path)
    raw = describe_safetensors(source)
    result = _content_fingerprint(raw, canonical_lora_name)
    original_names = {canonical_lora_name(item.name): item.name for item in raw}
    base = (
        _base_transformer_descriptors(base_transformer_paths)
        if base_transformer_paths is not None
        else None
    )
    pairs: dict[str, dict[str, TensorDescriptor]] = {}
    for item in result.tensors:
        for suffix in _LORA_SUFFIXES:
            if item.name.endswith(suffix):
                pairs.setdefault(item.name[: -len(suffix)], {})[suffix] = item
                break
    for target, tensors in pairs.items():
        if {".lora_A.weight", ".lora_B.weight"} - set(tensors):
            raise ArtifactError(f"LoRA target lacks A/B pair: {target}")
        left = tensors[".lora_A.weight"]
        right = tensors[".lora_B.weight"]
        if (
            left.dtype not in {"F16", "BF16", "F32"}
            or right.dtype not in {"F16", "BF16", "F32"}
            or len(left.shape) != 2
            or len(right.shape) != 2
            or not 1 <= left.shape[0] <= 128
            or left.shape[0] != right.shape[1]
        ):
            raise ArtifactError(f"LoRA rank shapes are invalid for {target}")
        if base is not None:
            base_name = target.removeprefix("transformer.") + ".weight"
            base_tensor = base.get(base_name)
            if base_tensor is None or len(base_tensor.shape) != 2:
                raise ArtifactError(f"LoRA target is not a base transformer weight: {target}")
            if (right.shape[0], left.shape[1]) != base_tensor.shape:
                raise ArtifactError(f"LoRA dimensions do not match base target: {target}")
        if ".alpha" in tensors:
            _alpha_value(source, original_names[tensors[".alpha"].name])
    scaling = lora_scaling(source, result.tensors)
    # An external PEFT config is executable model semantics, not just metadata.
    # Preserve the legacy identity for the default alpha=rank case.
    nondefault = {target: alpha for target, (rank, alpha) in scaling.items() if alpha != rank}
    if nondefault:
        result = CandidateWeights(
            hash_canonical("forge/lora-scaled-content/v1", result.content_fingerprint, nondefault),
            result.tensors,
        )
    return result


def lora_scaling(
    path: Path, descriptors: Iterable[TensorDescriptor]
) -> dict[str, tuple[int, float]]:
    """Resolve the admitted adapter's actual PEFT scaling without importing torch."""
    header, _, _ = _header(path)
    packed = header.get("__metadata__", {}).get("lora_adapter_metadata")
    config_path = path.parent / "adapter_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not isinstance(config, dict):
        raise ArtifactError("LoRA config must be an object")
    if packed:
        metadata = json.loads(packed)
        if not isinstance(metadata, dict):
            raise ArtifactError("LoRA adapter metadata must be an object")
        for key, value in metadata.items():
            key = key.removeprefix("transformer.")
            if key in config and config[key] != value:
                raise ArtifactError("LoRA metadata and config disagree")
            config[key] = value
    if not isinstance(config, dict):
        raise ArtifactError("LoRA config must be an object")
    if (
        any(
            config.get(key)
            for key in (
                "use_dora",
                "use_rslora",
                "fan_in_fan_out",
                "modules_to_save",
                "rank_pattern",
                "alpha_pattern",
                "loftq_config",
                "lora_bias",
                "use_qalora",
            )
        )
        or config.get("bias", "none") != "none"
        or config.get("peft_type", "LORA") != "LORA"
    ):
        raise ArtifactError("unsupported LoRA scaling or extra trainable modules")
    originals = {canonical_lora_name(key): key for key in header if key != "__metadata__"}
    result = {}
    for tensor in descriptors:
        key = canonical_lora_name(tensor.name)
        if not key.endswith(".lora_A.weight"):
            continue
        target = key.removesuffix(".lora_A.weight")
        rank = tensor.shape[0]
        if type(config.get("r", rank)) is not int or config.get("r", rank) != rank:
            raise ArtifactError("LoRA config rank differs from its tensors")
        embedded = originals.get(target + ".alpha")
        declared = config.get("lora_alpha", rank)
        alpha = _alpha_value(path, embedded) if embedded else declared
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 1024:
            raise ArtifactError("LoRA alpha must be finite and in (0, 1024]")
        if embedded and "lora_alpha" in config and alpha != declared:
            raise ArtifactError("LoRA tensor and config alpha disagree")
        result[target] = (rank, float(alpha))
    return result


def _declared_safetensors(manifest: ModelManifest, root: Path, prefix: str) -> list[Path]:
    files = [
        item.path
        for item in manifest.files
        if item.path.startswith(prefix) and item.path.endswith(".safetensors")
    ]
    if not files:
        raise ArtifactError("manifest does not declare required safetensors")
    return [root / item for item in files]


def _semantic_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid transformer config {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError("transformer config must be an object")
    return {key: item for key, item in value.items() if key not in _HARMLESS_CONFIG_KEYS}


def validate_full_transformer_weights(
    candidate: ModelManifest, candidate_root: Path, base: ModelManifest, base_root: Path
) -> CandidateWeights:
    candidate_files = _declared_safetensors(candidate, candidate_root, "transformer/")
    base_files = _declared_safetensors(base, base_root, "transformer/")
    candidate_descriptors = tuple(
        item for path in candidate_files for item in describe_safetensors(path)
    )
    base_descriptors = tuple(item for path in base_files for item in describe_safetensors(path))
    candidate_shapes = {(item.name, item.dtype, item.shape) for item in candidate_descriptors}
    base_shapes = {(item.name, item.dtype, item.shape) for item in base_descriptors}
    if candidate_shapes != base_shapes:
        raise ArtifactError(
            "full transformer tensor names, dtypes, or shapes differ from pinned base"
        )
    candidate_config = _semantic_config(candidate_root / "transformer/config.json")
    base_config = _semantic_config(base_root / "transformer/config.json")
    if candidate_config != base_config:
        raise ArtifactError("full transformer config changes the pinned Wan architecture")
    return _content_fingerprint(candidate_descriptors, lambda name: name)


def validate_candidate_weights(
    candidate: ModelManifest, candidate_root: str | Path, base: ModelManifest, base_root: str | Path
) -> CandidateWeights:
    """Validate a downloaded public candidate before any diffusers loader runs."""

    candidate_path = Path(candidate_root)
    base_path = Path(base_root)
    if candidate.role == "lora":
        file = [item.path for item in candidate.files if item.path.endswith(".safetensors")][0]
        base_files = _declared_safetensors(base, base_path, "transformer/")
        return validate_lora_weights(candidate_path / file, base_files)
    if candidate.role == "full_transformer":
        return validate_full_transformer_weights(candidate, candidate_path, base, base_path)
    if candidate.role == "pipeline" and candidate.reference == base.reference:
        return validate_full_transformer_weights(candidate, candidate_path, base, base_path)
    raise ArtifactError("only a pinned pipeline, full transformer, or LoRA is loadable")
