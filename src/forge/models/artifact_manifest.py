"""Public artifact schemas, identities and compatibility validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

Backend = Literal["hf", "hippius"]


Role = Literal["full_transformer", "lora", "pipeline"]


HF_BASE_URL = "https://huggingface.co"


MAX_FILES = 128


MAX_FILE_BYTES = 50 * 1024**3


MAX_TOTAL_BYTES = 100 * 1024**3


MAX_PIPELINE_BYTES = 32 * 1024**3


MAX_FULL_TRANSFORMER_BYTES = 6 * 1024**3


MAX_LORA_BYTES = 1024**3


HF_DOWNLOAD_DEADLINE_SECONDS = 300.0


MAX_HF_READ_TIMEOUT_SECONDS = 60.0


_ROLE_TOTAL_BYTES = {
    "pipeline": MAX_PIPELINE_BYTES,
    "full_transformer": MAX_FULL_TRANSFORMER_BYTES,
    "lora": MAX_LORA_BYTES,
}


_ROLE_FILE_BYTES = {
    "pipeline": MAX_FULL_TRANSFORMER_BYTES,
    "full_transformer": MAX_FULL_TRANSFORMER_BYTES,
    "lora": MAX_LORA_BYTES,
}


MAX_METADATA_BYTES = 2 * 1024**2


MAX_INLINE_HASH_BYTES = 2 * 1024**2


MAX_AUXILIARY_FILE_BYTES = MAX_INLINE_HASH_BYTES


MAX_PIPELINE_TOKENIZER_JSON_BYTES = 32 * 1024**2


_SHA256 = "0123456789abcdef"


_SAFE_SUFFIXES = {".json", ".model", ".safetensors", ".txt"}


_REUSABLE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
}


_IDENTIFIER_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ArtifactError(ValueError):
    """An artifact cannot safely be admitted or fetched."""


class ArtifactUnavailable(ArtifactError):
    """A public registry did not resolve an immutable artifact."""


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in _SHA256 for char in value)


def _safe_path(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or "\\" in path
        or ".." in candidate.parts
        or "." in candidate.parts
        or candidate.suffix.lower() not in _SAFE_SUFFIXES
        or any(not _PATH_COMPONENT.fullmatch(part) for part in candidate.parts)
    ):
        raise ArtifactError(f"unsafe or unsupported artifact path: {path!r}")
    return candidate


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ArtifactError("generation_config must contain JSON scalar values, mappings, or lists")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validate_generation_config(value: Mapping[str, Any]) -> None:
    allowed = {"steps", "guidance_scale", "scheduler"}
    if set(value) - allowed or not {"steps", "guidance_scale"}.issubset(value):
        raise ArtifactError("generation_config permits only steps, guidance_scale and scheduler")
    if "scheduler" in value and (
        not isinstance(value["scheduler"], str)
        or value["scheduler"] not in {"uniform_euler", "original_unipc"}
    ):
        raise ArtifactError("generation_config scheduler must be uniform_euler or original_unipc")
    steps = value["steps"]
    guidance = value["guidance_scale"]
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ArtifactError("generation_config steps must be an integer in [1, 100]")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
        or not 1 <= guidance <= 10
    ):
        raise ArtifactError("generation_config guidance_scale must be finite and in [1, 10]")


@dataclass(frozen=True)
class ArtifactReference:
    backend: Backend
    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        if self.backend not in {"hf", "hippius"}:
            raise ArtifactError("backend must be 'hf' or 'hippius'")
        if (
            not self.repo_id
            or self.repo_id.strip() != self.repo_id
            or self.repo_id.count("/") != 1
            or any(not _IDENTIFIER_COMPONENT.fullmatch(part) for part in self.repo_id.split("/"))
        ):
            raise ArtifactError("repo_id must be a public namespace/name identifier")
        if self.backend == "hf" and not _is_hex(self.revision, 40):
            raise ArtifactError("Hugging Face revision must be a lowercase 40-character commit SHA")
        if self.backend == "hippius" and not (
            self.revision.startswith("sha256:") and _is_hex(self.revision[7:], 64)
        ):
            raise ArtifactError("Hippius revision must be sha256:<64 lowercase hex characters>")

    @property
    def immutable_id(self) -> str:
        return f"{self.backend}:{self.repo_id}@{self.revision}"

    def to_dict(self) -> dict[str, str]:
        return {"backend": self.backend, "repo_id": self.repo_id, "revision": self.revision}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactReference:
        _exact_keys(value, {"backend", "repo_id", "revision"}, "artifact reference")
        if not all(isinstance(value[key], str) for key in ("backend", "repo_id", "revision")):
            raise ArtifactError("artifact reference fields must be strings")
        return cls(value["backend"], value["repo_id"], value["revision"])


@dataclass(frozen=True)
class ManifestFile:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _safe_path(self.path)
        if not _is_hex(self.sha256, 64):
            raise ArtifactError("file sha256 must be 64 lowercase hex characters")
        if not 0 < self.bytes <= MAX_FILE_BYTES:
            raise ArtifactError(f"file bytes must be between 1 and {MAX_FILE_BYTES}")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManifestFile:
        _exact_keys(value, {"path", "sha256", "bytes"}, "manifest file")
        if (
            not isinstance(value["path"], str)
            or not isinstance(value["sha256"], str)
            or not isinstance(value["bytes"], int)
            or isinstance(value["bytes"], bool)
        ):
            raise ArtifactError("manifest file has invalid field types")
        return cls(value["path"], value["sha256"], value["bytes"])


@dataclass(frozen=True)
class ArtifactInspection:
    reference: ArtifactReference
    license: str | None
    files: tuple[str, ...]


@dataclass(frozen=True)
class ModelManifest:
    reference: ArtifactReference
    role: Role
    files: tuple[ManifestFile, ...]
    license: str
    generation_config: Mapping[str, Any]
    base_reference: ArtifactReference | None = None

    def __post_init__(self) -> None:
        if self.role not in {"full_transformer", "lora", "pipeline"}:
            raise ArtifactError("role must be full_transformer, lora, or pipeline")
        if self.role in {"lora", "full_transformer"} and self.base_reference is None:
            raise ArtifactError("a LoRA or full transformer requires an immutable base_reference")
        if self.role == "pipeline" and self.base_reference is not None:
            raise ArtifactError("a pipeline manifest must not name base_reference")
        normalized_license = self.license.strip().lower()
        if normalized_license not in _REUSABLE_LICENSES:
            raise ArtifactError("license must be a supported reusable public license")
        if not self.files or len(self.files) > MAX_FILES:
            raise ArtifactError(f"manifest must contain 1 to {MAX_FILES} files")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ArtifactError("manifest file paths must be unique")
        total = sum(item.bytes for item in self.files)
        if total > MAX_TOTAL_BYTES:
            raise ArtifactError(f"manifest exceeds {MAX_TOTAL_BYTES} bytes")
        role_total_limit = _ROLE_TOTAL_BYTES[self.role]
        if total > role_total_limit:
            raise ArtifactError(
                f"{self.role} manifest exceeds {role_total_limit} bytes for the Wan 1.3B runtime"
            )
        role_file_limit = _ROLE_FILE_BYTES[self.role]
        for item in self.files:
            if item.bytes > role_file_limit:
                raise ArtifactError(
                    f"{self.role} file exceeds {role_file_limit} bytes for the Wan 1.3B runtime"
                )
        for item in self.files:
            if PurePosixPath(item.path).suffix.lower() not in {".json", ".txt"}:
                continue
            limit = MAX_AUXILIARY_FILE_BYTES
            if self.role == "pipeline" and item.path == "tokenizer/tokenizer.json":
                limit = MAX_PIPELINE_TOKENIZER_JSON_BYTES
            if item.bytes > limit:
                raise ArtifactError(f"configuration file exceeds {limit} bytes: {item.path}")
        _validate_generation_config(self.generation_config)
        weights = [item for item in self.files if item.path.endswith(".safetensors")]
        if self.role == "lora" and len(weights) != 1:
            raise ArtifactError("a LoRA manifest must declare exactly one safetensors file")
        if self.role == "full_transformer" and not any(
            item.path.startswith("transformer/") for item in weights
        ):
            raise ArtifactError("a full transformer must declare transformer safetensors")
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "license", normalized_license)
        object.__setattr__(self, "generation_config", _freeze(dict(self.generation_config)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "role": self.role,
            "files": [item.to_dict() for item in self.files],
            "license": self.license,
            "generation_config": _thaw(self.generation_config),
            "base_reference": self.base_reference.to_dict() if self.base_reference else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelManifest:
        _exact_keys(
            value,
            {"reference", "role", "files", "license", "generation_config", "base_reference"},
            "model manifest",
        )
        reference = value["reference"]
        files = value["files"]
        base_reference = value["base_reference"]
        generation_config = value["generation_config"]
        if not isinstance(reference, Mapping) or not isinstance(files, list):
            raise ArtifactError("manifest reference and files have invalid types")
        if base_reference is not None and not isinstance(base_reference, Mapping):
            raise ArtifactError("manifest base_reference has invalid type")
        if not isinstance(generation_config, Mapping):
            raise ArtifactError("manifest generation_config must be an object")
        if not all(isinstance(item, Mapping) for item in files):
            raise ArtifactError("manifest files must be objects")
        if not isinstance(value["role"], str) or not isinstance(value["license"], str):
            raise ArtifactError("manifest role and license must be strings")
        return cls(
            ArtifactReference.from_dict(reference),
            value["role"],
            tuple(ManifestFile.from_dict(item) for item in files),
            value["license"],
            generation_config,
            ArtifactReference.from_dict(base_reference) if base_reference else None,
        )

    def fingerprint(self) -> str:
        from forge.protocol.hashing import hash_canonical

        return hash_canonical("forge/model-manifest/v1", self.to_dict())


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ArtifactError(f"{label} fields must be exactly {sorted(expected)}")
