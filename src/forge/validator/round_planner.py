"""Build replayable public evaluation plans from finalized chain snapshots.

This planner records candidate admission from public metadata only.  It never
claims downloaded weights are architecture-compatible or runtime-ready; that
requires the separate validator-side preflight before any panel can count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from forge.chain import OnChainCommitment
from forge.evaluation.benchmark import generate_prompts
from forge.models.artifact_manifest import (
    HF_BASE_URL,
    ArtifactError,
    ArtifactReference,
    ModelManifest,
)
from forge.models.generation import GenerationRecipe, effective_generation_recipe
from forge.protocol.contract import ChallengeContract
from forge.protocol.finalist_lock import build_finalist_lock, first_valid_lock_after_close
from forge.protocol.hashing import hash_canonical
from forge.protocol.prompt_pack import (
    LEGACY_SOURCE,
    PACK_SOURCE,
    STAGES,
    build_pack,
    pack_context,
    seal_pack,
    verified_plan_prompts,
)
from forge.validator.submission import V3SubmissionClaim, resolve_submission, resolve_v3_submission


class FinalizedChainPort(Protocol):
    """The small, read-only chain surface required to construct a round plan."""

    network: str
    netuid: int

    def finalized_height(self) -> int: ...

    def finalized_block_hash(self, height: int) -> str: ...

    def uid_snapshot(self, height: int) -> dict[int, str]: ...

    def committee_snapshot(self, height: int) -> dict[str, int]: ...

    def commitments(self, hotkeys: list[str], height: int) -> list[OnChainCommitment]: ...

    def commitment(self, hotkey: str, height: int) -> OnChainCommitment | None: ...


def _block(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _chain_identity(chain: FinalizedChainPort) -> tuple[str, int]:
    network, netuid = getattr(chain, "network", None), getattr(chain, "netuid", None)
    if not isinstance(network, str) or not network:
        raise ValueError("chain network is required")
    return network, _block(netuid, "chain netuid")


def _v3_claim(commitment: OnChainCommitment) -> V3SubmissionClaim:
    from forge.validator.submission import claim_from_commitment_fields

    return claim_from_commitment_fields(commitment.kind, commitment.fields)


def _resolve_model(
    commitment: OnChainCommitment, contract: ChallengeContract, *, hf_base_url: str
) -> ModelManifest:
    if commitment.kind == "model":
        if len(commitment.fields) != 3:
            raise ValueError("invalid v2 model commitment")
        manifest = resolve_submission(
            ArtifactReference(*commitment.fields), hf_base_url=hf_base_url
        )
    elif commitment.kind in {"model_v3", "model_v4"}:
        manifest = resolve_v3_submission(_v3_claim(commitment), contract, hf_base_url=hf_base_url)
    else:
        raise ValueError("not a model commitment")
    if manifest.role not in {"full_transformer", "lora"}:
        raise ValueError("candidates must be full transformer or LoRA weights")
    base = manifest.base_reference
    if base is None or (base.repo_id, base.revision) != (
        contract.base_model_id,
        contract.base_model_revision,
    ):
        raise ValueError("candidate base differs from contract")
    return manifest


def _early_full_transformer_identity(manifest: ModelManifest) -> str | None:
    """Cheap exact-blob filter before the fixed admission cap.

    This intentionally handles only full-transformer snapshots. LoRA config,
    scaling and tensor semantics require the later runtime preflight.
    """
    if manifest.role != "full_transformer":
        return None
    weights = sorted(
        (item.sha256, item.bytes) for item in manifest.files if item.path.endswith(".safetensors")
    )
    recipe = effective_generation_recipe(
        GenerationRecipe.from_dict(dict(manifest.generation_config)),
        # A full transformer is never the legacy complete pipeline whose
        # omitted scheduler keeps the original UniPC default.
        is_original_reference=False,
    )
    return hash_canonical(
        "forge/early-full-transformer-identity/v2",
        {
            "role": manifest.role,
            "base_reference": manifest.base_reference.to_dict(),
            "effective_generation_recipe": recipe,
            "safetensors": weights,
        },
    )


def _incumbent(value: object, close_block: int) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "manifest_hash",
        "certified_state_hash",
        "certified_activation_block",
    }:
        raise ValueError("incumbent binding has invalid fields")
    result = dict(value)
    _sha256(result["manifest_hash"], "incumbent manifest hash")
    _sha256(result["certified_state_hash"], "incumbent certified state hash")
    if _block(result["certified_activation_block"], "incumbent activation block") > close_block:
        raise ValueError("incumbent activation is after the close block")
    return result


def _closed_commitments(
    chain: FinalizedChainPort, close_block: int
) -> tuple[dict[str, int], list[OnChainCommitment]]:
    hotkeys_by_uid = chain.uid_snapshot(close_block)
    if not isinstance(hotkeys_by_uid, dict) or any(
        type(uid) is not int or not isinstance(hotkey, str) or not hotkey
        for uid, hotkey in hotkeys_by_uid.items()
    ):
        raise ValueError("closed UID snapshot is invalid")
    if len(set(hotkeys_by_uid.values())) != len(hotkeys_by_uid):
        raise ValueError("closed UID snapshot contains duplicate hotkeys")
    committee = chain.committee_snapshot(close_block)
    if not isinstance(committee, dict) or any(
        not isinstance(hotkey, str) or not hotkey or type(stake) is not int or stake <= 0
        for hotkey, stake in committee.items()
    ):
        raise ValueError("closed committee snapshot is invalid")
    closed_hotkeys = set(hotkeys_by_uid.values())
    commitments = chain.commitments(list(hotkeys_by_uid.values()), close_block)
    if not isinstance(commitments, list):
        raise ValueError("closed commitment snapshot is invalid")
    seen: set[str] = set()
    for item in commitments:
        if (
            not isinstance(item, OnChainCommitment)
            or item.hotkey not in closed_hotkeys
            or item.hotkey in seen
            or _block(item.block, "commitment block") > close_block
        ):
            raise ValueError("closed commitment snapshot is inconsistent")
        seen.add(item.hotkey)
    return dict(committee), commitments


@dataclass(frozen=True)
class ClosedAdmissions:
    """Canonical metadata admissions rebuilt solely from the close snapshot."""

    committee: dict[str, int]
    commitments: list[OnChainCommitment]
    admitted: list[dict[str, Any]]
    rejected: list[dict[str, str]]
    early_full_transformer_duplicates: list[dict[str, str]]
    overflow: list[dict[str, str]]
    metadata_validated: int


def resolve_closed_admissions(
    chain: FinalizedChainPort,
    contract: ChallengeContract,
    close_block: int,
    *,
    hf_base_url: str = HF_BASE_URL,
) -> ClosedAdmissions:
    """Rebuild every canonical public admission from the finalized close.

    The returned order and cap are protocol data: consumers must not accept a
    locally edited subset of ``admitted`` when binding a candidate or a lock.
    """
    committee, commitments = _closed_commitments(chain, close_block)
    admissions: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for commitment in commitments:
        if commitment.kind not in {"model", "model_v3", "model_v4"}:
            continue
        try:
            manifest = _resolve_model(commitment, contract, hf_base_url=hf_base_url)
        except (ArtifactError, ValueError) as error:
            rejected.append({"hotkey": commitment.hotkey, "reason": str(error)})
            continue
        admissions.append(
            {
                "hotkey": commitment.hotkey,
                "block": commitment.block,
                "manifest_hash": manifest.fingerprint(),
                "manifest": manifest.to_dict(),
                "commitment_kind": commitment.kind,
                "admission_status": "METADATA_VALIDATED_RUNTIME_PREFLIGHT_PENDING",
            }
        )
    # A finalized snapshot does not establish intra-block arrival ordering.
    admissions.sort(key=lambda item: (item["block"], item["manifest_hash"], item["hotkey"]))
    unique: list[dict[str, Any]] = []
    early_duplicates: list[dict[str, str]] = []
    seen_manifests: set[str] = set()
    seen_full_transformers: set[str] = set()
    for admission in admissions:
        manifest_hash = admission["manifest_hash"]
        manifest = ModelManifest.from_dict(admission["manifest"])
        reason: str | None = None
        if manifest_hash in seen_manifests:
            reason = "duplicate manifest in closed admission snapshot"
        else:
            early_key = _early_full_transformer_identity(manifest)
            if early_key is not None and early_key in seen_full_transformers:
                reason = (
                    "same public full-transformer safetensor blobs and recipe as earlier admission"
                )
            elif early_key is not None:
                seen_full_transformers.add(early_key)
        if reason is not None:
            early_duplicates.append(
                {"hotkey": admission["hotkey"], "manifest_hash": manifest_hash, "reason": reason}
            )
            rejected.append({"hotkey": admission["hotkey"], "reason": reason})
            continue
        seen_manifests.add(manifest_hash)
        unique.append(admission)
    overflow = [
        {
            "hotkey": item["hotkey"],
            "manifest_hash": item["manifest_hash"],
            "reason": "admission cap reached before runtime preflight",
        }
        for item in unique[contract.max_candidates :]
    ]
    rejected.sort(key=lambda item: (item["hotkey"], item["reason"]))
    return ClosedAdmissions(
        committee=committee,
        commitments=commitments,
        admitted=unique[: contract.max_candidates],
        rejected=rejected,
        early_full_transformer_duplicates=early_duplicates,
        overflow=overflow,
        metadata_validated=len(admissions),
    )


def build_public_round_plan(
    chain: FinalizedChainPort,
    contract: ChallengeContract,
    close_block: int,
    entropy_block: int,
    prompt_pack: Mapping[str, object] | None = None,
    pack_hotkey: str | None = None,
    stage: str = "screen",
    finalist_lock_block: int | None = None,
    incumbent: Mapping[str, object] | None = None,
    candidate_hotkey: str | None = None,
    finalist_lock: Mapping[str, object] | None = None,
    finalist_lock_hotkey: str | None = None,
    calibration_plan_hash: str | None = None,
    *,
    hf_base_url: str = HF_BASE_URL,
) -> dict[str, Any]:
    """Freeze public admissions and prompt selection from finalized chain state.

    ``admitted`` means public registry metadata was validated at an immutable
    revision.  It is deliberately weaker than runtime preflight and cannot
    certify compatibility, quality, a finalist, a crown, or a reward.
    """
    if not isinstance(contract, ChallengeContract):
        raise TypeError("contract must be a ChallengeContract")
    close_block, entropy_block = (
        _block(close_block, "close block"),
        _block(entropy_block, "entropy block"),
    )
    if stage not in STAGES:
        raise ValueError("stage is unsupported")
    if stage == "confirmation":
        calibration_plan_hash = _sha256(calibration_plan_hash, "calibration plan hash")
    elif calibration_plan_hash is not None:
        raise ValueError("only confirmation plans declare a calibration plan hash")
    if stage == "screen":
        expected_entropy_block = close_block + 1
    elif stage == "duel":
        expected_entropy_block = close_block + 2
    else:
        finalist_lock_block = _block(finalist_lock_block, "finalist lock block")
        if finalist_lock_block < close_block:
            raise ValueError("confirmation finalist lock must be at or after close")
        expected_entropy_block = finalist_lock_block + 1
    if entropy_block != expected_entropy_block:
        raise ValueError("entropy block does not match the deterministic stage schedule")
    network, netuid = _chain_identity(chain)
    finalized_head = _block(chain.finalized_height(), "finalized head")
    if finalized_head < entropy_block:
        raise ValueError("entropy block is not finalized")
    close_hash = chain.finalized_block_hash(close_block)
    entropy = chain.finalized_block_hash(entropy_block)
    if not all(isinstance(value, str) and value for value in (close_hash, entropy)):
        raise ValueError("finalized block hashes are invalid")
    closed_admissions = resolve_closed_admissions(
        chain, contract, close_block, hf_base_url=hf_base_url
    )
    committee, commitments = closed_admissions.committee, closed_admissions.commitments

    pack_hash: str | None = None
    validated_pack: dict[str, object] | None = None
    if prompt_pack is not None:
        if not isinstance(pack_hotkey, str) or not pack_hotkey:
            raise ValueError("sealed prompt packs require their committing validator hotkey")
        if pack_hotkey not in committee:
            raise ValueError("prompt-pack committer was not in the closed committee")
        pack_hash = seal_pack(prompt_pack)
        validated_pack = build_pack(
            prompt_pack.get("context", {}),
            prompt_pack.get("salt"),
            prompt_pack.get("entries"),
        )
        context = validated_pack["context"]
        if not isinstance(context, Mapping):
            raise ValueError("sealed prompt pack context is required")
        round_id = context.get("round_id")
        expected_context = pack_context(
            network, netuid, round_id, contract.contract_hash(), close_block
        )
        if not isinstance(round_id, str) or not round_id or dict(context) != expected_context:
            raise ValueError("sealed prompt pack context differs from the round")
        matching_packs = [
            item
            for item in commitments
            if item.hotkey == pack_hotkey and item.kind == "pack" and item.fields == (pack_hash,)
        ]
        if len(matching_packs) != 1:
            raise ValueError("sealed prompt pack was not committed by the close block")
        pack_commitment = matching_packs[0]
    else:
        if pack_hotkey is not None:
            raise ValueError("pack hotkey requires a sealed prompt pack")
        round_id = f"r{close_block}"
        pack_commitment = None

    admitted = closed_admissions.admitted
    rejected = closed_admissions.rejected
    early_duplicates = closed_admissions.early_full_transformer_duplicates
    overflow = closed_admissions.overflow

    frozen_incumbent = _incumbent(incumbent, close_block)
    if candidate_hotkey is not None:
        if not isinstance(candidate_hotkey, str) or not candidate_hotkey:
            raise ValueError("candidate hotkey is invalid")
        matches = [item for item in admitted if item["hotkey"] == candidate_hotkey]
        if len(matches) != 1:
            raise ValueError("candidate hotkey was not admitted in the closed snapshot")
        if finalist_lock_block is None:
            raise ValueError("selected finalist requires a lock block")
        finalist_lock_block = _block(finalist_lock_block, "finalist lock block")
        if not close_block <= finalist_lock_block < entropy_block:
            raise ValueError("finalist lock must be after close and before entropy")
        candidate = {
            "manifest_hash": matches[0]["manifest_hash"],
            "hotkey": candidate_hotkey,
            "admission_block": matches[0]["block"],
        }
        if stage == "confirmation":
            if not isinstance(finalist_lock, Mapping) or not isinstance(finalist_lock_hotkey, str):
                raise ValueError("confirmation requires a finalized finalist lock record")
            if frozen_incumbent is None:
                raise ValueError("confirmation requires an incumbent state binding")
            lock = build_finalist_lock(finalist_lock)
            first = first_valid_lock_after_close(
                chain,
                committee,
                admitted,
                network=network,
                netuid=netuid,
                round_id=round_id,
                close_block=close_block,
                close_hash=close_hash,
                contract_hash=contract.contract_hash(),
                incumbent_state_hash=frozen_incumbent["certified_state_hash"],
                upper_block=finalist_lock_block,
            )
            if first is None:
                raise ValueError("confirmation requires a finalized finalist lock record")
            first_block, first_hotkey, first_lock = first
            if (
                first_block != finalist_lock_block
                or first_hotkey != finalist_lock_hotkey
                or lock != first_lock
                or lock["candidate_manifest_hash"] != candidate["manifest_hash"]
                or lock["candidate_hotkey"] != candidate_hotkey
            ):
                raise ValueError("confirmation must use FIRST_VALID_LOCK_AFTER_C")
    else:
        if finalist_lock_block is not None:
            raise ValueError("finalist lock requires a selected candidate")
        candidate = None

    geometry = {
        "height": contract.height,
        "width": contract.width,
        "frames": contract.frames,
        "fps": contract.fps,
    }
    plan: dict[str, Any] = {
        "schema": "forge/panel-plan/v1",
        "admission_algorithm": "forge/closed-admission/v1",
        "purpose": "evaluation",
        "network": network,
        "netuid": netuid,
        "close_block": close_block,
        "close_block_hash": close_hash,
        "entropy_block": entropy_block,
        "entropy": entropy,
        "round_id": round_id,
        "campaign_id": round_id,
        "stage": stage,
        **({"calibration_plan_hash": calibration_plan_hash} if calibration_plan_hash else {}),
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash(),
        "geometry": geometry,
        "committee_rao": committee,
        "admitted": admitted,
        "early_full_transformer_duplicates": early_duplicates,
        "admission_overflow": overflow,
        "admission_counts": {
            "metadata_validated": closed_admissions.metadata_validated,
            "pending_runtime_preflight": len(admitted),
            "early_full_transformer_duplicates": len(early_duplicates),
            "overflow_before_runtime_preflight": len(overflow),
        },
        "rejected": rejected,
        "admission_status": "METADATA_VALIDATED_RUNTIME_PREFLIGHT_PENDING",
        "status": "READY_FOR_LOCAL_EVALUATION_NOT_A_CERTIFICATE",
    }
    if candidate is not None:
        plan["candidate"] = candidate
        plan["finalist_lock_block"] = finalist_lock_block
        if stage == "confirmation":
            plan["finalist_lock"] = lock
            plan["finalist_lock_hotkey"] = finalist_lock_hotkey
    if frozen_incumbent is not None:
        plan["incumbent"] = frozen_incumbent
    if prompt_pack is None:
        plan["prompt_source"] = LEGACY_SOURCE
        plan["prompts"] = [
            asdict(item) for item in generate_prompts(contract, round_id, stage, entropy)
        ]
    else:
        assert pack_hash is not None and pack_commitment is not None
        plan.update(
            {
                "prompt_source": PACK_SOURCE,
                "prompt_pack": validated_pack,
                "pack_commitment": {
                    "hash": pack_hash,
                    "block": pack_commitment.block,
                    "hotkey": pack_hotkey,
                },
                # This is a current finalized observation, not proof that the
                # pack stayed secret before entropy.
                "pack_reveal_block": finalized_head,
            }
        )
        plan["prompts"] = verified_plan_prompts(plan, contract)
    return plan
