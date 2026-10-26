"""Signed, replay-bound records for a validator's automatic evaluation.

An evaluation record is evidence for future commit/reveal transport.  It is not
a certificate, crown, or instruction to submit weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.evaluation.automatic_gate import evaluate_automatic
from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.ports import Signer, Verifier

SCHEMA = "forge/evaluation-record/v1"
_STATUSES = {
    "PROMOTE_ELIGIBLE",
    "FAIL",
    "INCOMPLETE",
    "INCONCLUSIVE",
    "DEVELOPMENT_DIAGNOSTIC",
}


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required")
    return value


def _block(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _hash_evidence(value: dict[str, Any], label: str) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{label} evidence must be an object")
    return hash_canonical(f"forge/evaluation-{label}/v1", value)


def _confirmation_binding(plan: dict[str, Any]) -> tuple[str, int, str, str, str, int, str, int]:
    """Return the identities that must already be frozen before a validator signs."""
    network = _text(plan.get("network"), "network")
    netuid = _block(plan.get("netuid"), "netuid")
    round_id = _text(plan.get("round_id"), "round_id")
    close_block = _block(plan.get("close_block"), "close_block")
    candidate = plan.get("candidate")
    incumbent = plan.get("incumbent")
    admitted = plan.get("admitted")
    if not isinstance(candidate, dict) or set(candidate) != {
        "manifest_hash",
        "hotkey",
        "admission_block",
    }:
        raise ValueError("confirmation plan candidate binding is invalid")
    if not isinstance(incumbent, dict) or set(incumbent) != {
        "manifest_hash",
        "certified_state_hash",
        "certified_activation_block",
    }:
        raise ValueError("confirmation plan incumbent binding is invalid")
    candidate_hash = _sha256(candidate["manifest_hash"], "candidate manifest hash")
    candidate_hotkey = _text(candidate["hotkey"], "candidate hotkey")
    admission_block = _block(candidate["admission_block"], "candidate admission block")
    incumbent_hash = _sha256(incumbent["manifest_hash"], "incumbent manifest hash")
    _sha256(incumbent["certified_state_hash"], "incumbent state hash")
    _block(incumbent["certified_activation_block"], "incumbent activation block")
    if admission_block > close_block:
        raise ValueError("candidate admission is after the close block")
    if not isinstance(admitted, list):
        raise ValueError("confirmation plan admitted candidates are required")
    matches = [
        item
        for item in admitted
        if isinstance(item, dict)
        and set(item) >= {"manifest_hash", "hotkey", "block"}
        and item.get("manifest_hash") == candidate_hash
        and item.get("hotkey") == candidate_hotkey
        and item.get("block") == admission_block
    ]
    if len(matches) != 1:
        raise ValueError("candidate is not exactly admitted at the frozen close")
    return (
        network,
        netuid,
        round_id,
        candidate_hash,
        candidate_hotkey,
        admission_block,
        incumbent_hash,
        close_block,
    )


def _evidence_manifests(quality: dict[str, Any], timing: dict[str, Any]) -> tuple[str, str, bool]:
    comparisons = quality.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 2:
        raise ValueError("quality comparisons are incomplete")
    indexed = {item.get("reference_role"): item for item in comparisons if isinstance(item, dict)}
    incumbent = indexed.get("incumbent")
    golden = indexed.get("golden")
    if incumbent is None or golden is None or len(indexed) != 2:
        raise ValueError("quality references are incomplete")
    candidate_hash = _sha256(incumbent.get("candidate_manifest_hash"), "quality candidate hash")
    if _sha256(golden.get("candidate_manifest_hash"), "golden candidate hash") != candidate_hash:
        raise ValueError("quality references disagree about the candidate")
    incumbent_hash = _sha256(incumbent.get("reference_manifest_hash"), "quality incumbent hash")
    public = all(
        item.get("reference_admission") == "public_snapshot"
        and item.get("candidate_admission") == "public_snapshot"
        for item in (golden, incumbent)
    ) and (
        timing.get("reference_admission") == "public_snapshot"
        and timing.get("candidate_admission") == "public_snapshot"
    )
    if _sha256(timing.get("candidate_manifest_hash"), "timing candidate hash") != candidate_hash:
        raise ValueError("timing candidate differs from quality")
    if _sha256(timing.get("reference_manifest_hash"), "timing incumbent hash") != incumbent_hash:
        raise ValueError("timing incumbent differs from quality")
    return candidate_hash, incumbent_hash, public


@dataclass(frozen=True)
class EvaluationRecord:
    network: str
    netuid: int
    round_id: str
    plan_hash: str
    incumbent_state_hash: str
    incumbent_manifest_hash: str
    candidate_manifest_hash: str
    candidate_hotkey: str
    candidate_admission_block: int
    close_block: int
    policy_hash: str
    quality_hash: str
    timing_hash: str
    calibration_hash: str
    evaluator_id: str
    evaluator_journal_sha256: str
    gate_result_hash: str
    gate_status: str
    public_admission: bool
    development: bool
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported evaluation record schema")
        _text(self.network, "network")
        _block(self.netuid, "netuid")
        _text(self.round_id, "round_id")
        for field in (
            "plan_hash",
            "incumbent_state_hash",
            "incumbent_manifest_hash",
            "candidate_manifest_hash",
            "policy_hash",
            "quality_hash",
            "timing_hash",
            "calibration_hash",
            "evaluator_journal_sha256",
            "gate_result_hash",
        ):
            _sha256(getattr(self, field), field)
        _text(self.candidate_hotkey, "candidate hotkey")
        _block(self.candidate_admission_block, "candidate admission block")
        _block(self.close_block, "close block")
        if self.candidate_admission_block > self.close_block:
            raise ValueError("candidate admission is after the close block")
        _text(self.evaluator_id, "evaluator id")
        if self.gate_status not in _STATUSES:
            raise ValueError("unknown automatic gate status")
        if type(self.public_admission) is not bool or type(self.development) is not bool:
            raise ValueError("admission and development flags must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "network": self.network,
            "netuid": self.netuid,
            "round_id": self.round_id,
            "plan_hash": self.plan_hash,
            "incumbent_state_hash": self.incumbent_state_hash,
            "incumbent_manifest_hash": self.incumbent_manifest_hash,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "candidate_hotkey": self.candidate_hotkey,
            "candidate_admission_block": self.candidate_admission_block,
            "close_block": self.close_block,
            "policy_hash": self.policy_hash,
            "quality_hash": self.quality_hash,
            "timing_hash": self.timing_hash,
            "calibration_hash": self.calibration_hash,
            "evaluator_id": self.evaluator_id,
            "evaluator_journal_sha256": self.evaluator_journal_sha256,
            "gate_result_hash": self.gate_result_hash,
            "gate_status": self.gate_status,
            "public_admission": self.public_admission,
            "development": self.development,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationRecord:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("evaluation record fields are not exact")
        return cls(**value)

    def record_hash(self) -> str:
        return hash_canonical("forge/evaluation-record/v1", self.to_dict())

    def signing_payload(self) -> bytes:
        return canonical_json_bytes(
            {"domain": "forge/evaluation-record-signature/v1", "record": self.to_dict()}
        )


@dataclass(frozen=True)
class SignedEvaluationRecord:
    record: EvaluationRecord
    validator_hotkey: str
    signature: bytes

    def __post_init__(self) -> None:
        _text(self.validator_hotkey, "validator hotkey")
        if not isinstance(self.signature, bytes) or not self.signature:
            raise ValueError("evaluation record signature is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "validator_hotkey": self.validator_hotkey,
            "signature": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SignedEvaluationRecord:
        if not isinstance(value, dict) or set(value) != {"record", "validator_hotkey", "signature"}:
            raise ValueError("signed evaluation record fields are not exact")
        try:
            signature = bytes.fromhex(value["signature"])
        except (TypeError, ValueError) as error:
            raise ValueError("evaluation record signature is not hex") from error
        return cls(
            EvaluationRecord.from_dict(value["record"]), value["validator_hotkey"], signature
        )


def create_record(
    plan: dict[str, Any],
    policy: dict[str, Any],
    quality: dict[str, Any],
    timing: dict[str, Any],
    calibration: dict[str, Any],
    *,
    evaluator_journal_sha256: str,
    preflight: dict[str, Any] | None = None,
    development: bool = False,
) -> EvaluationRecord:
    """Recompute the automatic gate and bind the exact evidence it consumed."""
    if type(development) is not bool:
        raise ValueError("development must be boolean")
    (
        network,
        netuid,
        round_id,
        plan_candidate_hash,
        candidate_hotkey,
        admission_block,
        plan_incumbent_hash,
        close_block,
    ) = _confirmation_binding(plan)
    evidence_candidate_hash, evidence_incumbent_hash, public_admission = _evidence_manifests(
        quality, timing
    )
    if (
        evidence_candidate_hash != plan_candidate_hash
        or evidence_incumbent_hash != plan_incumbent_hash
    ):
        raise ValueError("evidence manifests differ from the confirmation plan")
    gate = evaluate_automatic(
        plan, policy, quality, timing, calibration, preflight=preflight, development=development
    )
    status = gate.get("status")
    if status not in _STATUSES:
        raise ValueError("automatic gate returned an unknown status")
    candidate_from_gate = gate.get("candidate_manifest_hash")
    if candidate_from_gate is not None and candidate_from_gate != plan_candidate_hash:
        raise ValueError("automatic gate candidate differs from the confirmation plan")
    evaluator_id = policy.get("evaluator_id")
    if evaluator_id != quality.get("evaluator_id"):
        raise ValueError("quality evidence differs from the frozen evaluator")
    return EvaluationRecord(
        network=network,
        netuid=netuid,
        round_id=round_id,
        plan_hash=hash_canonical("forge/evaluation-plan/v1", plan),
        incumbent_state_hash=_sha256(
            plan["incumbent"]["certified_state_hash"], "incumbent state hash"
        ),
        incumbent_manifest_hash=plan_incumbent_hash,
        candidate_manifest_hash=plan_candidate_hash,
        candidate_hotkey=candidate_hotkey,
        candidate_admission_block=admission_block,
        close_block=close_block,
        policy_hash=_hash_evidence(policy, "policy"),
        quality_hash=_hash_evidence(quality, "quality"),
        timing_hash=_hash_evidence(timing, "timing"),
        calibration_hash=_hash_evidence(calibration, "calibration"),
        evaluator_id=_text(evaluator_id, "evaluator id"),
        evaluator_journal_sha256=_sha256(evaluator_journal_sha256, "evaluator journal hash"),
        gate_result_hash=hash_canonical("forge/automatic-gate-result/v1", gate),
        gate_status=status,
        public_admission=public_admission,
        development=development,
    )


def sign_record(record: EvaluationRecord, signer: Signer) -> SignedEvaluationRecord:
    if not isinstance(record, EvaluationRecord):
        raise TypeError("record must be an EvaluationRecord")
    hotkey = _text(signer.validator_hotkey, "validator hotkey")
    return SignedEvaluationRecord(record, hotkey, bytes(signer.sign(record.signing_payload())))


def verify_record(signed: SignedEvaluationRecord, verifier: Verifier) -> bool:
    if not isinstance(signed, SignedEvaluationRecord):
        return False
    return bool(
        verifier.verify(signed.validator_hotkey, signed.record.signing_payload(), signed.signature)
    )
