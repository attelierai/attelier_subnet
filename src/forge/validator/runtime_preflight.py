"""Validator-owned effective-weight preflight for a frozen public candidate batch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from forge.models.artifact_manifest import ArtifactError, ModelManifest
from forge.models.generation import GenerationRecipe, paired_reference_scheduler
from forge.models.weights import CandidateWeights, validate_candidate_weights
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file

SCHEMA = "forge/runtime-preflight/v1"
PASS = "PUBLIC_RUNTIME_PREFLIGHT_PASSED"
DUPLICATE = "DUPLICATE_EFFECTIVE_CANDIDATE"
INCOMPLETE = "INCOMPLETE_CANDIDATE_PREFLIGHT"
_INCOMPLETE_STAGES = frozenset({"fetch", "validation"})


def verify_local(manifest: ModelManifest, root: str | Path) -> None:
    """Recheck the exact files immediately before preflight or model loading."""
    root = Path(root)
    resolved = root.resolve()
    declared = {file.path for file in manifest.files}
    for item in manifest.files:
        path = root / item.path
        if path.is_symlink() or resolved not in path.resolve().parents:
            raise ValueError("model files must be regular files inside their snapshot")
        if path.stat().st_size != item.bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"model file failed verification: {item.path}")
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".py", ".json", ".model"}
            and path.relative_to(root).as_posix() not in declared
        ):
            raise ValueError(f"undeclared loadable model file: {path.name}")


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return hash_canonical("forge/runtime-preflight-plan/v1", plan)


def effective_candidate_hash(
    candidate: ModelManifest,
    weights: CandidateWeights,
    base: ModelManifest,
    contract: ChallengeContract,
) -> str:
    """Identity of exact loaded tensors and the execution recipe, not file packaging.

    LoRA A*c/B/c equivalents deliberately remain different: this uses exact
    canonical safetensor bytes.  Pipeline and full-transformer use one
    ``transformer`` lane because their loaded transformer tensors are identical.
    """
    if candidate.role not in {"pipeline", "full_transformer", "lora"}:
        raise ArtifactError("candidate role is not runtime-preflightable")
    lane = "lora" if candidate.role == "lora" else "transformer"
    return hash_canonical(
        "forge/effective-candidate/v2",
        {
            "contract_hash": contract.contract_hash(),
            "base_reference": base.reference.to_dict(),
            "lane": lane,
            "weight_content_hash": weights.content_fingerprint,
            "generation_config": {
                "steps": int(candidate.generation_config["steps"]),
                "guidance_scale": float(candidate.generation_config["guidance_scale"]),
                "scheduler": paired_reference_scheduler(
                    candidate,
                    base,
                    GenerationRecipe.from_dict(dict(candidate.generation_config)),
                    contract,
                ),
            },
        },
    )


def legacy_effective_candidate_hash(
    candidate: ModelManifest,
    weights: CandidateWeights,
    base: ModelManifest,
    contract: ChallengeContract,
) -> str | None:
    """Return a v1 alias only when it denotes the same resolved recipe.

    Explicit UniPC is intentionally not aliased to an old implicit Euler
    candidate.  Old database rows contain only v1 digests, so treating every
    same-tensor recipe as equivalent would erase a real scheduler distinction.
    """

    recipe = GenerationRecipe.from_dict(dict(candidate.generation_config))
    legacy_recipe = GenerationRecipe(steps=recipe.steps, guidance_scale=recipe.guidance_scale)
    if paired_reference_scheduler(candidate, base, recipe, contract) != paired_reference_scheduler(
        candidate, base, legacy_recipe, contract
    ):
        return None
    lane = "lora" if candidate.role == "lora" else "transformer"
    return hash_canonical(
        "forge/effective-candidate/v1",
        {
            "contract_hash": contract.contract_hash(),
            "base_reference": base.reference.to_dict(),
            "lane": lane,
            "weight_content_hash": weights.content_fingerprint,
            "generation_config": {
                "steps": int(recipe.steps),
                "guidance_scale": float(recipe.guidance_scale),
            },
        },
    )


def validate_public_candidate(
    candidate: ModelManifest,
    directory: str | Path,
    base: ModelManifest,
    base_dir: str | Path,
    contract: ChallengeContract,
) -> CandidateWeights:
    """Shared miner/validator byte and tensor checks; no admission or inference."""
    directory = Path(directory)
    if base.role != "pipeline" or (base.reference.repo_id, base.reference.revision) != (
        contract.base_model_id,
        contract.base_model_revision,
    ):
        raise ArtifactError("base snapshot differs from the pinned challenge contract")
    recipe = GenerationRecipe.from_dict(dict(candidate.generation_config))
    if recipe.steps > contract.max_steps:
        raise ArtifactError("candidate recipe exceeds the contract step limit")
    if recipe.steps * (2 if recipe.guidance_scale > 1 else 1) > contract.max_forward_passes:
        raise ArtifactError("candidate recipe exceeds the contract forward-pass limit")
    if candidate.role not in {"lora", "full_transformer"}:
        raise ArtifactError("public candidates must be LoRA or full-transformer weights")
    if candidate.base_reference != base.reference:
        raise ArtifactError("candidate base differs from preflight base")
    if directory.resolve() == Path(base_dir).resolve():
        base_files = {item.path: item for item in base.files}
        for item in candidate.files:
            base_item = base_files.get(item.path)
            if base_item != item:
                raise ArtifactError("shared candidate cache is not an exact subset of the base")
    else:
        verify_local(candidate, directory)
    return validate_candidate_weights(candidate, directory, base, base_dir)


def preflight_candidate_batch(
    plan: Mapping[str, Any],
    contract: ChallengeContract,
    base: ModelManifest,
    base_dir: str | Path,
    candidates: Iterable[tuple[ModelManifest, str | Path | None]],
) -> dict[str, Any]:
    """Validate an entire closed batch before any render or vision call.

    The caller must download immutable public snapshots first.  This function
    neither fetches nor runs a model, so its result is cheap evidence that the
    paid GPU/judge lanes can consume.
    """
    if not isinstance(contract, ChallengeContract) or not isinstance(base, ModelManifest):
        raise TypeError("contract and base manifest are required")
    if plan.get("contract_hash") != contract.contract_hash():
        raise ValueError("preflight plan contract hash mismatch")
    admitted = plan.get("admitted")
    if not isinstance(admitted, list) or len(admitted) > contract.max_candidates:
        raise ValueError("preflight requires at most the contract candidate limit")
    expected = []
    for entry in admitted:
        if not isinstance(entry, Mapping):
            raise ValueError("preflight plan admission is invalid")
        manifest_hash = _sha256(entry.get("manifest_hash"), "admitted manifest hash")
        block = entry.get("block")
        hotkey = entry.get("hotkey")
        if type(block) is not int or block < 0 or not isinstance(hotkey, str) or not hotkey:
            raise ValueError("preflight plan admission identity is invalid")
        manifest = ModelManifest.from_dict(entry.get("manifest"))
        if manifest.fingerprint() != manifest_hash:
            raise ValueError("admitted manifest hash differs from its manifest")
        expected.append((block, manifest_hash, hotkey, manifest))
    if len({item[1] for item in expected}) != len(expected):
        raise ValueError("preflight plan has duplicate manifest admissions")
    supplied_items = [
        (model, Path(directory) if directory is not None else None)
        for model, directory in candidates
    ]
    supplied = {model.fingerprint(): (model, directory) for model, directory in supplied_items}
    if len(supplied) != len(supplied_items) or set(supplied) != {item[1] for item in expected}:
        raise ValueError("preflight candidates differ from the frozen admissions")
    verify_local(base, base_dir)
    rows: list[dict[str, Any]] = []
    seen_effective: dict[str, str] = {}
    for _, manifest_hash, _, manifest in sorted(expected, key=lambda item: item[:3]):
        candidate, directory = supplied[manifest_hash]
        if candidate != manifest:
            raise ValueError("preflight candidate manifest differs from the frozen admission")
        if directory is None:
            rows.append(
                {
                    "manifest_hash": manifest_hash,
                    "status": INCOMPLETE,
                    "stage": "fetch",
                }
            )
            continue
        try:
            weights = validate_public_candidate(candidate, directory, base, base_dir, contract)
        except (ArtifactError, ValueError, OSError):
            rows.append(
                {
                    "manifest_hash": manifest_hash,
                    "status": INCOMPLETE,
                    "stage": "validation",
                }
            )
            continue
        effective = effective_candidate_hash(candidate, weights, base, contract)
        legacy_effective = legacy_effective_candidate_hash(candidate, weights, base, contract)
        duplicate_of = seen_effective.get(effective)
        if duplicate_of is None:
            seen_effective[effective] = manifest_hash
            status = PASS
        else:
            status = DUPLICATE
        rows.append(
            {
                "manifest_hash": manifest_hash,
                "weight_content_hash": weights.content_fingerprint,
                "effective_candidate_hash": effective,
                "status": status,
                **(
                    {"legacy_effective_candidate_hash": legacy_effective}
                    if legacy_effective is not None
                    else {}
                ),
                **({"duplicate_of_manifest_hash": duplicate_of} if duplicate_of else {}),
            }
        )
    return {
        "schema": SCHEMA,
        "plan_hash": _plan_hash(plan),
        "contract_hash": contract.contract_hash(),
        "base_manifest_hash": base.fingerprint(),
        "candidates": rows,
    }


def validate_preflight_receipt(
    receipt: Mapping[str, Any] | None,
    plan: Mapping[str, Any],
    contract: ChallengeContract,
    *,
    candidate_manifest_hash: str,
) -> dict[str, Any]:
    """Fail closed unless one frozen candidate passed the unique batch preflight."""
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "plan_hash",
        "contract_hash",
        "base_manifest_hash",
        "candidates",
    }:
        raise ValueError("runtime preflight receipt is required and has invalid fields")
    if receipt.get("schema") != SCHEMA or receipt.get("plan_hash") != _plan_hash(plan):
        raise ValueError("runtime preflight receipt differs from the frozen plan")
    if receipt.get("contract_hash") != contract.contract_hash():
        raise ValueError("runtime preflight receipt differs from the contract")
    _sha256(receipt.get("base_manifest_hash"), "preflight base manifest hash")
    rows = receipt.get("candidates")
    if not isinstance(rows, list) or len(rows) > contract.max_candidates:
        raise ValueError("runtime preflight candidates are invalid")
    admitted = plan.get("admitted")
    expected = (
        {
            _sha256(entry.get("manifest_hash"), "admitted manifest hash")
            for entry in admitted
            if isinstance(entry, Mapping)
        }
        if isinstance(admitted, list)
        else set()
    )
    if len(expected) != len(admitted or []):
        raise ValueError("runtime preflight plan admissions are invalid")
    indexed: dict[str, Mapping[str, Any]] = {}
    effective: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") not in {PASS, DUPLICATE, INCOMPLETE}:
            raise ValueError("runtime preflight candidate row is invalid")
        manifest_hash = _sha256(row.get("manifest_hash"), "preflight manifest hash")
        if manifest_hash in indexed or manifest_hash not in expected:
            raise ValueError("runtime preflight candidates differ from admissions")
        if row["status"] == INCOMPLETE:
            if (
                set(row) != {"manifest_hash", "status", "stage"}
                or row.get("stage") not in _INCOMPLETE_STAGES
            ):
                raise ValueError("runtime preflight incomplete row is invalid")
            indexed[manifest_hash] = row
            continue
        _sha256(row.get("weight_content_hash"), "preflight weight content hash")
        key = _sha256(row.get("effective_candidate_hash"), "effective candidate hash")
        legacy = row.get("legacy_effective_candidate_hash")
        if legacy is not None:
            _sha256(legacy, "legacy effective candidate hash")
        common = {
            "manifest_hash",
            "weight_content_hash",
            "effective_candidate_hash",
            "status",
        }
        if legacy is not None:
            common.add("legacy_effective_candidate_hash")
        if row["status"] == PASS:
            if key in effective or set(row) != common:
                raise ValueError("runtime preflight has duplicate effective candidates")
            effective.add(key)
        elif set(row) != common | {"duplicate_of_manifest_hash"}:
            raise ValueError("runtime preflight duplicate row is invalid")
        else:
            duplicate_of = row.get("duplicate_of_manifest_hash")
            source = indexed.get(duplicate_of)
            if (
                source is None
                or source.get("status") != PASS
                or source.get("effective_candidate_hash") != key
            ):
                raise ValueError("runtime preflight duplicate row is invalid")
        indexed[manifest_hash] = row
    if set(indexed) != expected:
        raise ValueError("runtime preflight must cover every frozen admission")
    candidate_manifest_hash = _sha256(candidate_manifest_hash, "candidate manifest hash")
    candidate = indexed.get(candidate_manifest_hash)
    if candidate is None or candidate["status"] != PASS:
        raise ValueError("candidate did not pass unique runtime preflight")
    return dict(candidate)
