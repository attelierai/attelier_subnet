"""Public immutable model-submission commitments and manifest resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from forge.models.artifact_manifest import (
    HF_BASE_URL,
    ArtifactError,
    ArtifactReference,
    ArtifactUnavailable,
    ManifestFile,
    ModelManifest,
)
from forge.models.artifacts import (
    _hippius_bridge,
    build_manifest,
    inspect_public_artifact,
    snapshot_manifest,
)
from forge.protocol.contract import ChallengeContract

PREFIX = "forge:v2"
V3_PREFIX = "forge:v3"
V4_PREFIX = "forge:v4"
MANIFEST_NAME = "forge-manifest.json"
MAX_COMMITMENT_BYTES = 128
MAX_MANIFEST_BYTES = 1024 * 1024


@dataclass(frozen=True)
class V3SubmissionClaim:
    """A self-contained HF claim: legacy v3 or scheduler-bound v4."""

    reference: ArtifactReference
    role: str
    generation_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.reference.backend != "hf":
            raise ArtifactError("v3 automatic manifests currently support Hugging Face only")
        if self.role not in {"full_transformer", "lora"}:
            raise ArtifactError("v3 role must be full_transformer or lora")
        config = dict(self.generation_config)
        if set(config) not in (
            {"steps", "guidance_scale"},
            {"steps", "guidance_scale", "scheduler"},
        ):
            raise ArtifactError("v3 generation config must contain steps and guidance_scale")
        placeholder = (
            "transformer/placeholder.safetensors"
            if self.role == "full_transformer"
            else "placeholder.safetensors"
        )
        validated = ModelManifest(
            self.reference,
            self.role,
            (ManifestFile(placeholder, "0" * 64, 1),),
            "apache-2.0",
            config,
            ArtifactReference("hf", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0" * 40),
        )
        # Keep the parsed recipe immutable after validation.  A mutable caller
        # must not be able to turn an admitted claim into a different chain
        # commitment between preflight and publication.
        object.__setattr__(self, "generation_config", validated.generation_config)


def _guidance_token(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError("v3 guidance_scale must be numeric")
    scaled = float(value) * 10
    tenths = round(scaled)
    if not 10 <= tenths <= 100 or abs(scaled - tenths) > 1e-9:
        raise ArtifactError("v3 guidance_scale must be in [1, 10] at one-decimal precision")
    return str(tenths // 10) if tenths % 10 == 0 else f"{tenths // 10}.{tenths % 10}"


def _parse_guidance_token(value: str) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    if value in {str(number) for number in range(1, 11)}:
        return float(value)
    if len(value) == 3 and value[0] in "123456789" and value[1] == "." and value[2] in "123456789":
        return float(value)
    return None


def _parse_steps(value: str) -> int | None:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        return None
    if value != str(int(value)):
        return None
    result = int(value)
    return result if 1 <= result <= 100 else None


def encode_artifact_commitment(reference: ArtifactReference) -> str:
    """Encode a public immutable snapshot reference for the chain commitment."""

    if not isinstance(reference, ArtifactReference):
        raise ArtifactError("submission commitment requires an ArtifactReference")
    if any(character in reference.repo_id for character in ":@?#"):
        raise ArtifactError("repo_id contains reserved commitment characters")
    backend = "h" if reference.backend == "hf" else "p"
    revision = (
        reference.revision
        if reference.backend == "hf"
        else reference.revision.removeprefix("sha256:")
    )
    result = f"{PREFIX}:{backend}:{reference.repo_id}:{revision}"
    if len(result.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        raise ArtifactError("submission commitment exceeds 128 UTF-8 bytes")
    return result


def decode_artifact_commitment(text: str) -> ArtifactReference | None:
    """Return a public immutable reference, or ``None`` for a rejected commitment."""

    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        return None
    parts = text.split(":")
    if len(parts) != 5 or ":".join(parts[:2]) != PREFIX:
        return None
    _, _, backend, repo_id, revision = parts
    if backend == "h":
        candidate = ("hf", revision)
    elif backend == "p":
        candidate = ("hippius", f"sha256:{revision}")
    else:
        return None
    if any(character in repo_id for character in ":@?#"):
        return None
    try:
        return ArtifactReference(candidate[0], repo_id, candidate[1])
    except ArtifactError:
        return None


def encode_v3_submission_commitment(claim: V3SubmissionClaim) -> str:
    """Encode a public HF snapshot plus the complete Wan runtime recipe.

    V3 intentionally has no miner-provided manifest: the validator derives its
    files, hashes and public license directly from the immutable HF revision.
    """
    if not isinstance(claim, V3SubmissionClaim):
        raise ArtifactError("v3 commitment requires a V3SubmissionClaim")
    if any(character in claim.reference.repo_id for character in ":@?#"):
        raise ArtifactError("repo_id contains reserved commitment characters")
    role = "t" if claim.role == "full_transformer" else "l"
    config = dict(claim.generation_config)
    result = ":".join(
        (
            V4_PREFIX if "scheduler" in config else V3_PREFIX,
            "h",
            claim.reference.repo_id,
            claim.reference.revision,
            role,
            str(config["steps"]),
            _guidance_token(config["guidance_scale"]),
        )
    )
    if "scheduler" in config:
        result += ":" + {"uniform_euler": "e", "original_unipc": "u"}[config["scheduler"]]
    if len(result.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        raise ArtifactError("v3 submission commitment exceeds 128 UTF-8 bytes")
    return result


def decode_v3_submission_commitment(text: str) -> V3SubmissionClaim | None:
    """Decode only canonical, self-contained HF v3 claims."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        return None
    parts = text.split(":")
    if len(parts) != 8 or ":".join(parts[:2]) != V3_PREFIX:
        return None
    _, _, backend, repo_id, revision, role_token, steps_token, guidance_token = parts
    if backend != "h" or any(character in repo_id for character in ":@?#"):
        return None
    steps, guidance = _parse_steps(steps_token), _parse_guidance_token(guidance_token)
    role = {"t": "full_transformer", "l": "lora"}.get(role_token)
    if steps is None or guidance is None or role is None:
        return None
    try:
        claim = V3SubmissionClaim(
            ArtifactReference("hf", repo_id, revision),
            role,
            {"steps": steps, "guidance_scale": guidance},
        )
    except ArtifactError:
        return None
    return claim if encode_v3_submission_commitment(claim) == text else None


def decode_v4_submission_commitment(text: str) -> V3SubmissionClaim | None:
    """Decode an explicit, validator-owned scheduler without changing v3 bytes."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        return None
    parts = text.split(":")
    if len(parts) != 9 or ":".join(parts[:2]) != V4_PREFIX:
        return None
    scheduler = {"e": "uniform_euler", "u": "original_unipc"}.get(parts[-1])
    legacy = decode_v3_submission_commitment(":".join(["forge", "v3", *parts[2:-1]]))
    if scheduler is None or legacy is None:
        return None
    claim = V3SubmissionClaim(
        legacy.reference, legacy.role, {**dict(legacy.generation_config), "scheduler": scheduler}
    )
    return claim if encode_v3_submission_commitment(claim) == text else None


def claim_from_commitment_fields(kind: str, fields: tuple[str, ...]) -> V3SubmissionClaim:
    """One recipe parser for finalized admissions and certificate replay."""
    count = {"model_v3": 6, "model_v4": 7}.get(kind)
    if count is None or len(fields) != count:
        raise ArtifactError("invalid versioned model commitment")
    backend, repo, revision, role, steps, guidance = fields[:6]
    config = {"steps": int(steps), "guidance_scale": float(guidance)}
    if kind == "model_v4":
        config["scheduler"] = fields[6]
    return V3SubmissionClaim(ArtifactReference(backend, repo, revision), role, config)


def decode_submission_commitment(text: str) -> ArtifactReference | V3SubmissionClaim | None:
    """Decode v2 declarations and v3 validator-derived HF claims."""
    return (
        decode_v4_submission_commitment(text)
        or decode_v3_submission_commitment(text)
        or decode_artifact_commitment(text)
    )


def _manifest_url(reference: ArtifactReference, hf_base_url: str) -> str:
    root = hf_base_url.rstrip("/")
    repo = quote(reference.repo_id, safe="/")
    return f"{root}/{repo}/resolve/{reference.revision}/{MANIFEST_NAME}"


def _decode_manifest(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ArtifactUnavailable(f"{MANIFEST_NAME} exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ArtifactUnavailable(f"{MANIFEST_NAME} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ArtifactUnavailable(f"{MANIFEST_NAME} must be a JSON object")
    return decoded


def _read_manifest(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "forge-subnet/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - pinned HTTPS URL
            payload = response.read(MAX_MANIFEST_BYTES + 1)
    except HTTPError as exc:
        raise ArtifactUnavailable(
            f"public snapshot does not provide {MANIFEST_NAME} ({exc.code})"
        ) from exc
    except URLError as exc:
        raise ArtifactUnavailable(f"public snapshot manifest is unavailable: {exc.reason}") from exc
    return _decode_manifest(payload)


def _read_hippius_manifest(
    reference: ArtifactReference, *, endpoint: str | None, timeout_seconds: float
) -> dict[str, Any]:
    """Fetch only the public declaration from a pinned Hippius OCI snapshot."""

    result = _hippius_bridge(
        "declaration", reference, endpoint=endpoint, payload={}, timeout_seconds=timeout_seconds
    )
    declaration = result.get("declaration")
    if not isinstance(declaration, dict):
        raise ArtifactUnavailable("Hippius bridge returned an invalid declaration")
    return declaration


def _declaration_to_manifest(
    reference: ArtifactReference, declaration: Mapping[str, Any]
) -> ModelManifest:
    expected = {"schema", "role", "files", "license", "generation_config", "base_reference"}
    if set(declaration) != expected or declaration.get("schema") != "forge-manifest/v1":
        raise ArtifactError("forge-manifest declaration has an invalid schema")
    if any(
        item.get("path") == MANIFEST_NAME
        for item in declaration.get("files", [])
        if isinstance(item, Mapping)
    ):
        raise ArtifactError("forge-manifest.json must not reference itself")
    files = declaration["files"]
    base = declaration["base_reference"]
    config = declaration["generation_config"]
    if (
        not isinstance(files, list)
        or not all(isinstance(item, Mapping) for item in files)
        or not isinstance(config, Mapping)
        or not isinstance(declaration["role"], str)
        or not isinstance(declaration["license"], str)
    ):
        raise ArtifactError("forge-manifest files and generation_config are invalid")
    if base is not None and not isinstance(base, Mapping):
        raise ArtifactError("forge-manifest base_reference is invalid")
    return ModelManifest(
        reference=reference,
        role=declaration["role"],
        files=tuple(ManifestFile.from_dict(item) for item in files),
        license=declaration["license"],
        generation_config=config,
        base_reference=ArtifactReference.from_dict(base) if base else None,
    )


def prepare_declaration(manifest: ModelManifest) -> dict[str, Any]:
    """Return the uploadable declaration, deliberately excluding its own reference."""

    if not isinstance(manifest, ModelManifest):
        raise ArtifactError("submission declaration requires a ModelManifest")
    serialized = manifest.to_dict()
    return {
        "schema": "forge-manifest/v1",
        "role": serialized["role"],
        "files": serialized["files"],
        "license": serialized["license"],
        "generation_config": serialized["generation_config"],
        "base_reference": serialized["base_reference"],
    }


def resolve_submission(
    reference: ArtifactReference,
    *,
    hf_base_url: str = HF_BASE_URL,
    hippius_endpoint: str | None = None,
    timeout_seconds: float = 20.0,
) -> ModelManifest:
    """Resolve only the small declaration from a public immutable snapshot.

    The declaration intentionally excludes its own reference. The commitment is
    the reference authority; injecting it here avoids a self-referential file
    hash while retaining a canonical manifest fingerprint.
    """

    if not isinstance(reference, ArtifactReference) or timeout_seconds <= 0:
        raise ArtifactError("invalid immutable submission reference")
    inspection = inspect_public_artifact(
        reference,
        hf_base_url=hf_base_url,
        hippius_endpoint=hippius_endpoint,
        timeout_seconds=timeout_seconds,
    )
    if MANIFEST_NAME not in inspection.files:
        raise ArtifactUnavailable(f"public snapshot is missing {MANIFEST_NAME}")
    declaration = (
        _read_manifest(_manifest_url(reference, hf_base_url), timeout_seconds)
        if reference.backend == "hf"
        else _read_hippius_manifest(
            reference, endpoint=hippius_endpoint, timeout_seconds=timeout_seconds
        )
    )
    manifest = _declaration_to_manifest(reference, declaration)
    return build_manifest(
        reference,
        role=manifest.role,
        files=manifest.files,
        license=manifest.license,
        generation_config=manifest.generation_config,
        base_reference=manifest.base_reference,
        inspection=inspection,
    )


def resolve_v3_submission(
    claim: V3SubmissionClaim,
    contract: ChallengeContract,
    *,
    hf_base_url: str = HF_BASE_URL,
    timeout_seconds: float = 20.0,
) -> ModelManifest:
    """Derive a v3 manifest from immutable public HF metadata and the contract.

    The candidate cannot select a base, license, file list or file hashes.  The
    contract owns the base and the public HF revision owns all byte identities.
    """
    if not isinstance(claim, V3SubmissionClaim) or not isinstance(contract, ChallengeContract):
        raise ArtifactError("v3 resolution requires a claim and challenge contract")
    recipe = dict(claim.generation_config)
    if recipe["steps"] > contract.max_steps:
        raise ArtifactError("v3 recipe exceeds the challenge maximum steps")
    base = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    manifest = snapshot_manifest(
        claim.reference,
        role=claim.role,
        generation_config=recipe,
        base_reference=base,
        hf_base_url=hf_base_url,
        timeout_seconds=timeout_seconds,
    )
    lora_weights = sum(file.path.endswith(".safetensors") for file in manifest.files)
    if claim.role == "lora" and lora_weights != 1:
        raise ArtifactUnavailable("v3 LoRA snapshots must contain exactly one safetensors file")
    return manifest
