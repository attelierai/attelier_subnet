"""Prepare and verify a testnet anchor for one signed automatic evaluation.

Anchors retain both negative and diagnostic results.  They are evidence only:
they do not create a certificate, crown a model, or submit weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from forge.chain import OnChainCommitment, encode_evaluation_record_commitment
from forge.models.artifact_manifest import HF_BASE_URL, ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.evaluation_record import (
    EvaluationRecord,
    SignedEvaluationRecord,
    create_record,
    sign_record,
    verify_record,
)
from forge.protocol.finalist_lock import (
    build_finalist_lock,
    finalist_lock_hash,
    first_valid_lock_after_close,
)
from forge.protocol.ports import Signer, Verifier
from forge.protocol.prompt_pack import PACK_SOURCE
from forge.validator.round_planner import resolve_closed_admissions
from forge.validator.submission import claim_from_commitment_fields


class EvaluationAnchorChainPort(Protocol):
    network: str
    netuid: int

    def head(self) -> int: ...

    def finalized_height(self) -> int: ...

    def finalized_block_hash(self, height: int) -> str: ...

    def uid_snapshot(self, height: int) -> dict[int, str]: ...

    def committee_snapshot(self, height: int) -> dict[str, int]: ...

    def commitment(self, hotkey: str, height: int) -> OnChainCommitment | None: ...

    def commitments(self, hotkeys: list[str], height: int) -> list[OnChainCommitment]: ...


@dataclass(frozen=True)
class EvaluationAnchorIntent:
    """A preflighted, signed record ready for an explicit chain publication."""

    signed_record: SignedEvaluationRecord
    commitment: str
    preflight_block: int
    preflight_block_hash: str

    @property
    def record(self) -> EvaluationRecord:
        return self.signed_record.record

    def receipt(self, *, status: str) -> dict[str, Any]:
        if status not in {"DRY_RUN", "ANCHOR_PENDING"}:
            raise ValueError("invalid pre-publication anchor status")
        return {
            "schema": "forge/testnet-evaluation-anchor/v1",
            "network": self.record.network,
            "netuid": self.record.netuid,
            "round_id": self.record.round_id,
            "hotkey": self.signed_record.validator_hotkey,
            "record_hash": self.record.record_hash(),
            "commitment": self.commitment,
            "preflight_block": self.preflight_block,
            "preflight_block_hash": self.preflight_block_hash,
            "status": status,
            "signed_record": self.signed_record.to_dict(),
            "statement": "SIGNED_EVALUATION_EVIDENCE_ONLY_NOT_A_CERTIFICATE_OR_REWARD",
        }


def _block(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _testnet_identity(chain: EvaluationAnchorChainPort, plan: Mapping[str, Any]) -> None:
    if getattr(chain, "network", None) != "test" or plan.get("network") != "test":
        raise ValueError("evaluation anchors are restricted to testnet")
    netuid = _block(getattr(chain, "netuid", None), "chain netuid")
    if plan.get("netuid") != netuid:
        raise ValueError("evaluation plan netuid differs from the testnet chain")


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _model_commitment_matches(manifest: ModelManifest, commitment: OnChainCommitment) -> bool:
    if commitment.kind == "model" and len(commitment.fields) == 3:
        fields = tuple(
            manifest.reference.to_dict()[key] for key in ("backend", "repo_id", "revision")
        )
        return fields == commitment.fields
    try:
        claim = claim_from_commitment_fields(commitment.kind, commitment.fields)
    except (TypeError, ValueError):
        return False
    return (
        claim.reference == manifest.reference
        and claim.role == manifest.role
        and dict(claim.generation_config) == dict(manifest.generation_config)
    )


def _chain_public_admission_binding(
    chain: EvaluationAnchorChainPort,
    plan: Mapping[str, Any],
    *,
    development: bool,
    hf_base_url: str,
) -> tuple[dict[str, int], list[dict[str, Any]]] | None:
    """Reconstruct, rather than trust, the closed public admission set."""
    required = {
        "close_block_hash",
        "entropy_block",
        "entropy",
        "committee_rao",
        "pack_commitment",
        "candidate",
        "admitted",
        "contract",
        "contract_hash",
        "admission_algorithm",
    }
    claims_chain_public = plan.get("prompt_source") == PACK_SOURCE or any(
        name in plan for name in ("close_block_hash", "entropy_block", "committee_rao")
    )
    if not claims_chain_public:
        if development:
            return None
        raise ValueError("non-development evaluation requires a chain-bound public plan")
    if not required.issubset(plan):
        raise ValueError("chain-bound plan is missing finalized admission fields")
    if plan.get("admission_algorithm") != "forge/closed-admission/v1":
        raise ValueError("chain-bound plan has an unsupported admission algorithm")
    close_block = _block(plan["close_block"], "plan close block")
    entropy_block = _block(plan["entropy_block"], "plan entropy block")
    if (
        entropy_block <= close_block
        or _block(chain.finalized_height(), "finalized head") < entropy_block
    ):
        raise ValueError("chain-bound plan has no finalized entropy block")
    if chain.finalized_block_hash(close_block) != plan["close_block_hash"]:
        raise ValueError("plan close block hash differs from finalized chain")
    if chain.finalized_block_hash(entropy_block) != plan["entropy"]:
        raise ValueError("plan entropy differs from finalized chain")
    try:
        contract = ChallengeContract.from_dict(plan["contract"])
    except (TypeError, ValueError) as error:
        raise ValueError("chain-bound plan contract is invalid") from error
    if contract.contract_hash() != plan["contract_hash"]:
        raise ValueError("plan contract hash differs from its contract")
    closed = resolve_closed_admissions(chain, contract, close_block, hf_base_url=hf_base_url)
    committee, admitted = closed.committee, closed.admitted
    if committee != plan["committee_rao"]:
        raise ValueError("plan committee differs from finalized close snapshot")
    if plan["admitted"] != admitted:
        raise ValueError("plan admissions differ from finalized canonical admissions")
    candidate = plan["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "manifest_hash",
        "hotkey",
        "admission_block",
    }:
        raise ValueError("chain-bound candidate is invalid")
    candidate_hash = _sha256(candidate["manifest_hash"], "candidate manifest hash")
    candidate_hotkey = candidate["hotkey"]
    if not isinstance(candidate_hotkey, str) or not candidate_hotkey:
        raise ValueError("chain-bound candidate hotkey is invalid")
    matches = [
        item
        for item in admitted
        if item["manifest_hash"] == candidate_hash
        and item["hotkey"] == candidate_hotkey
        and item["block"] == candidate["admission_block"]
    ]
    if len(matches) != 1:
        raise ValueError("candidate is absent from finalized canonical admissions")
    admission = matches[0]
    manifest = ModelManifest.from_dict(admission["manifest"])
    observed_candidate = chain.commitment(candidate_hotkey, close_block)
    if (
        observed_candidate is None
        or observed_candidate.hotkey != candidate_hotkey
        or observed_candidate.block != candidate["admission_block"]
        or observed_candidate.kind != admission["commitment_kind"]
        or not _model_commitment_matches(manifest, observed_candidate)
    ):
        raise ValueError("candidate commitment differs from the finalized close snapshot")
    pack = plan["pack_commitment"]
    if not isinstance(pack, Mapping) or set(pack) != {"hash", "block", "hotkey"}:
        raise ValueError("chain-bound prompt-pack commitment is invalid")
    pack_hash = _sha256(pack["hash"], "prompt-pack hash")
    pack_hotkey = pack["hotkey"]
    if not isinstance(pack_hotkey, str) or not pack_hotkey or pack_hotkey not in committee:
        raise ValueError("prompt-pack committer differs from finalized committee")
    observed_pack = chain.commitment(pack_hotkey, close_block)
    if (
        observed_pack is None
        or observed_pack.kind != "pack"
        or observed_pack.fields != (pack_hash,)
        or observed_pack.block != pack["block"]
        or observed_pack.block > close_block
    ):
        raise ValueError("prompt-pack commitment differs from finalized close snapshot")
    return committee, admitted


def _confirmation_lock_binding(
    chain: EvaluationAnchorChainPort,
    plan: Mapping[str, Any],
    closed_admissions: tuple[dict[str, int], list[dict[str, Any]]] | None,
) -> None:
    if plan.get("stage") != "confirmation":
        return
    if closed_admissions is None:
        raise ValueError("confirmation requires canonical closed admissions")
    committee, admitted = closed_admissions
    lock_block = _block(plan.get("finalist_lock_block"), "finalist lock block")
    close_block = _block(plan.get("close_block"), "close block")
    entropy_block = _block(plan.get("entropy_block"), "entropy block")
    lock_hotkey = plan.get("finalist_lock_hotkey")
    candidate, incumbent = plan.get("candidate"), plan.get("incumbent")
    if (
        not isinstance(lock_hotkey, str)
        or not lock_hotkey
        or not isinstance(candidate, Mapping)
        or not isinstance(incumbent, Mapping)
        or not close_block < lock_block
        or entropy_block != lock_block + 1
        or lock_hotkey not in committee
    ):
        raise ValueError("confirmation finalist lock is not chain-bound")
    expected = {
        "schema": "forge/finalist-lock/v1",
        "network": plan.get("network"),
        "netuid": plan.get("netuid"),
        "round_id": plan.get("round_id"),
        "close_block": close_block,
        "close_block_hash": plan.get("close_block_hash"),
        "contract_hash": plan.get("contract_hash"),
        "candidate_manifest_hash": candidate.get("manifest_hash"),
        "candidate_hotkey": candidate.get("hotkey"),
        "candidate_admission_block": candidate.get("admission_block"),
        "incumbent_state_hash": incumbent.get("certified_state_hash"),
    }
    lock = build_finalist_lock(plan.get("finalist_lock"))
    if lock != build_finalist_lock(expected):
        raise ValueError("confirmation finalist lock record differs from plan")
    observed = chain.commitment(lock_hotkey, lock_block)
    if (
        observed is None
        or observed.block != lock_block
        or observed.kind != "finalist_lock"
        or observed.fields != (finalist_lock_hash(lock),)
    ):
        raise ValueError("confirmation finalist lock differs from finalized chain")
    first = first_valid_lock_after_close(
        chain,
        committee,
        admitted,
        network=lock["network"],
        netuid=lock["netuid"],
        round_id=lock["round_id"],
        close_block=close_block,
        close_hash=lock["close_block_hash"],
        contract_hash=lock["contract_hash"],
        incumbent_state_hash=lock["incumbent_state_hash"],
        upper_block=lock_block,
    )
    if first is None or first != (lock_block, lock_hotkey, lock):
        raise ValueError("confirmation must use FIRST_VALID_LOCK_AFTER_C")


def _active_pack_guard(
    chain: EvaluationAnchorChainPort,
    plan: Mapping[str, Any],
    signer_hotkey: str,
) -> None:
    """Never overwrite this validator's sealed pack during its live close window."""
    pack_commitment = plan.get("pack_commitment")
    if not isinstance(pack_commitment, Mapping):
        return
    if pack_commitment.get("hotkey") != signer_hotkey:
        return
    close_block = _block(plan.get("close_block"), "plan close block")
    if _block(chain.head(), "chain head") <= close_block:
        raise ValueError("refusing to replace an active sealed prompt-pack commitment")


def prepare_evaluation_anchor(
    chain: EvaluationAnchorChainPort,
    signer: Signer,
    verifier: Verifier,
    plan: dict[str, Any],
    policy: dict[str, Any],
    quality: dict[str, Any],
    timing: dict[str, Any],
    calibration: dict[str, Any],
    *,
    evaluator_journal_sha256: str,
    preflight: dict[str, Any] | None = None,
    development: bool = False,
    hf_base_url: str = HF_BASE_URL,
) -> EvaluationAnchorIntent:
    """Recompute, sign, and preflight one record without publishing it.

    The chain read happens before the returned object is persisted.  This
    guarantees that even dry-run output binds a registered committee validator
    at a finalized testnet snapshot.
    """
    if not isinstance(plan, dict):
        raise ValueError("evaluation plan must be an object")
    _testnet_identity(chain, plan)
    closed_admissions = _chain_public_admission_binding(
        chain, plan, development=development, hf_base_url=hf_base_url
    )
    _confirmation_lock_binding(chain, plan, closed_admissions)
    signer_hotkey = signer.validator_hotkey
    _active_pack_guard(chain, plan, signer_hotkey)
    record = create_record(
        plan,
        policy,
        quality,
        timing,
        calibration,
        evaluator_journal_sha256=evaluator_journal_sha256,
        preflight=preflight,
        development=development,
    )
    signed = sign_record(record, signer)
    if not verify_record(signed, verifier):
        raise ValueError("sr25519 evaluation record self-verification failed")
    if signed.validator_hotkey != signer_hotkey:
        raise ValueError("signer identity changed while preparing evaluation anchor")
    preflight_block = _block(chain.finalized_height(), "finalized preflight block")
    preflight_hash = chain.finalized_block_hash(preflight_block)
    if not isinstance(preflight_hash, str) or not preflight_hash:
        raise ValueError("finalized preflight hash is invalid")
    if signed.validator_hotkey not in chain.uid_snapshot(preflight_block).values():
        raise ValueError("evaluation signer is not registered on this testnet subnet")
    if signed.validator_hotkey not in chain.committee_snapshot(preflight_block):
        raise ValueError("evaluation signer is not in the finalized validator committee")
    return EvaluationAnchorIntent(
        signed,
        encode_evaluation_record_commitment(record.record_hash()),
        preflight_block,
        preflight_hash,
    )


def verify_finalized_evaluation_anchor(
    chain: EvaluationAnchorChainPort, intent: EvaluationAnchorIntent
) -> tuple[OnChainCommitment, int, str]:
    """Read the finalized commitment back and bind it to the exact signed record."""
    if not isinstance(intent, EvaluationAnchorIntent):
        raise TypeError("intent must be an EvaluationAnchorIntent")
    finalized = _block(chain.finalized_height(), "finalized readback block")
    observed = chain.commitment(intent.signed_record.validator_hotkey, finalized)
    expected_hash = intent.record.record_hash()
    if (
        observed is None
        or observed.kind != "evaluation"
        or observed.fields != (expected_hash,)
        or observed.hotkey != intent.signed_record.validator_hotkey
        or observed.block > finalized
    ):
        raise ValueError("finalized evaluation anchor differs from the signed record; do not retry")
    final_hash = chain.finalized_block_hash(finalized)
    if not isinstance(final_hash, str) or not final_hash:
        raise ValueError("finalized readback hash is invalid")
    return observed, finalized, final_hash
