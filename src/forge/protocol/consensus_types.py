"""Serialized consensus records, certificates and finalized competition state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.rewards import (
    Crown,
)


class LocalVerdict(StrEnum):
    PROMOTE = "PROMOTE"
    KEEP = "KEEP"
    INCOMPLETE = "INCOMPLETE"


class Transition(StrEnum):
    PROMOTE = "PROMOTE"
    KEEP = "KEEP"


class DecisionStatus(StrEnum):
    PROMOTE = "PROMOTE"
    KEEP = "KEEP"
    PENDING = "PENDING"


class ConsensusPolicy(StrEnum):
    """The quorum rule frozen into a round plan.

    The historical rule remains the default so old plans and receipts retain
    their original hash and decision semantics.  New distributed rounds must
    opt in explicitly to the simple stake-majority rule.
    """

    HISTORICAL_SUPERMAJORITY = "historical_supermajority"
    STAKE_MAJORITY_V2 = "stake_majority_v2"


class CertificateError(ValueError):
    pass


CERTIFICATE_SCHEMA = "forge/preconsensus/certificate/v2"


@dataclass(frozen=True)
class RoundPlan:
    network: str
    netuid: int
    round_id: str
    parent_state_hash: str
    challenge_id: str
    king_fingerprint: str
    challenger_fingerprint: str
    challenger_hotkey: str
    contract_hash: str
    prompt_seed_inputs: dict[str, Any]
    committee: dict[str, int]
    snapshot_block: int
    commit_deadline_block: int
    reveal_deadline_block: int
    activation_block: int
    consensus_policy: ConsensusPolicy = ConsensusPolicy.HISTORICAL_SUPERMAJORITY

    def __post_init__(self) -> None:
        required_strings = (
            self.network,
            self.round_id,
            self.parent_state_hash,
            self.challenge_id,
            self.king_fingerprint,
            self.challenger_fingerprint,
            self.challenger_hotkey,
            self.contract_hash,
        )
        if not all(required_strings):
            raise ValueError("round plan identifiers cannot be empty")
        if self.king_fingerprint == self.challenger_fingerprint:
            raise ValueError("challenger cannot duplicate the king fingerprint")
        if self.netuid < 0 or self.snapshot_block < 0:
            raise ValueError("netuid and snapshot block cannot be negative")
        if not (
            self.snapshot_block
            < self.commit_deadline_block
            < self.reveal_deadline_block
            < self.activation_block
        ):
            raise ValueError("round block schedule must be strictly increasing")
        if not self.committee or any(power <= 0 for power in self.committee.values()):
            raise ValueError("committee voting power must be positive")
        try:
            policy = ConsensusPolicy(self.consensus_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("round consensus policy is invalid") from error
        object.__setattr__(self, "consensus_policy", policy)
        object.__setattr__(self, "committee", MappingProxyType(dict(self.committee)))
        object.__setattr__(
            self, "prompt_seed_inputs", MappingProxyType(dict(self.prompt_seed_inputs))
        )

    @property
    def total_committee_power(self) -> int:
        return sum(self.committee.values())

    def quorum_satisfied(self, power: int) -> bool:
        """Whether power strictly exceeds this frozen plan's threshold."""
        if self.consensus_policy == ConsensusPolicy.STAKE_MAJORITY_V2:
            return 2 * power > self.total_committee_power
        return 3 * power > 2 * self.total_committee_power

    def to_dict(self) -> dict[str, Any]:
        value = {
            "network": self.network,
            "netuid": self.netuid,
            "round_id": self.round_id,
            "parent_state_hash": self.parent_state_hash,
            "challenge_id": self.challenge_id,
            "king_fingerprint": self.king_fingerprint,
            "challenger_fingerprint": self.challenger_fingerprint,
            "challenger_hotkey": self.challenger_hotkey,
            "contract_hash": self.contract_hash,
            "prompt_seed_inputs": dict(self.prompt_seed_inputs),
            "committee": dict(self.committee),
            "snapshot_block": self.snapshot_block,
            "commit_deadline_block": self.commit_deadline_block,
            "reveal_deadline_block": self.reveal_deadline_block,
            "activation_block": self.activation_block,
        }
        # Keep v1 receipts byte-for-byte reproducible.  A v2 plan carries its
        # policy in the hashed payload instead of relying on a changing default.
        if self.consensus_policy != ConsensusPolicy.HISTORICAL_SUPERMAJORITY:
            value["consensus_policy"] = self.consensus_policy.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RoundPlan:
        """Read both historical plan payloads and policy-versioned plans."""
        if not isinstance(value, dict):
            raise CertificateError("round plan serialization is invalid")
        fields = set(cls.__dataclass_fields__) - {"consensus_policy"}
        received = set(value)
        allowed = fields | {"consensus_policy"}
        if not fields <= received or not received <= allowed:
            raise CertificateError("round plan fields are invalid")
        try:
            return cls(
                **{
                    **{field: value[field] for field in fields},
                    "consensus_policy": value.get(
                        "consensus_policy", ConsensusPolicy.HISTORICAL_SUPERMAJORITY
                    ),
                }
            )
        except (TypeError, ValueError) as error:
            raise CertificateError("round plan serialization is invalid") from error

    def plan_hash(self) -> str:
        return hash_canonical("forge/preconsensus/plan/v1", self.to_dict())


@dataclass(frozen=True)
class LocalEvaluationReport:
    network: str
    netuid: int
    round_id: str
    validator_hotkey: str
    plan_hash: str
    parent_state_hash: str
    contract_hash: str
    king_fingerprint: str
    challenger_fingerprint: str
    verdict: LocalVerdict
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "network": self.network,
            "netuid": self.netuid,
            "round_id": self.round_id,
            "validator_hotkey": self.validator_hotkey,
            "plan_hash": self.plan_hash,
            "parent_state_hash": self.parent_state_hash,
            "contract_hash": self.contract_hash,
            "king_fingerprint": self.king_fingerprint,
            "challenger_fingerprint": self.challenger_fingerprint,
            "verdict": self.verdict.value,
            "metrics": self.metrics,
        }

    def report_hash(self) -> str:
        return hash_canonical("forge/preconsensus/report/v1", self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LocalEvaluationReport:
        return cls(**{**value, "verdict": LocalVerdict(value["verdict"])})

    def consistent_with(self, plan: RoundPlan) -> bool:
        return (
            self.network == plan.network
            and self.netuid == plan.netuid
            and self.round_id == plan.round_id
            and self.validator_hotkey in plan.committee
            and self.plan_hash == plan.plan_hash()
            and self.parent_state_hash == plan.parent_state_hash
            and self.contract_hash == plan.contract_hash
            and self.king_fingerprint == plan.king_fingerprint
            and self.challenger_fingerprint == plan.challenger_fingerprint
        )


@dataclass(frozen=True)
class CommitRecord:
    validator_hotkey: str
    commitment: str
    commit_block: int


@dataclass(frozen=True)
class Reveal:
    report: LocalEvaluationReport
    salt: str
    reveal_block: int


@dataclass(frozen=True)
class RoundDecision:
    status: DecisionStatus
    favorable_power: int
    conclusive_power: int
    total_committee_power: int
    valid_validators: tuple[str, ...]


@dataclass(frozen=True)
class RoundCertificate:
    network: str
    netuid: int
    round_id: str
    challenge_id: str
    plan_hash: str
    parent_state_hash: str
    previous_king_fingerprint: str
    challenger_fingerprint: str
    challenger_hotkey: str
    transition: Transition
    activation_block: int
    activation_chain_time: int
    evidence_digest: str
    commits: tuple[CommitRecord, ...]
    reveals: tuple[Reveal, ...]
    signatures: dict[str, bytes]
    schema: str = CERTIFICATE_SCHEMA

    def core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "network": self.network,
            "netuid": self.netuid,
            "round_id": self.round_id,
            "challenge_id": self.challenge_id,
            "plan_hash": self.plan_hash,
            "parent_state_hash": self.parent_state_hash,
            "previous_king_fingerprint": self.previous_king_fingerprint,
            "challenger_fingerprint": self.challenger_fingerprint,
            "challenger_hotkey": self.challenger_hotkey,
            "transition": self.transition.value,
            "activation_block": self.activation_block,
            "activation_chain_time": self.activation_chain_time,
            "evidence_digest": self.evidence_digest,
        }

    def signing_payload(self) -> bytes:
        domain = "forge/preconsensus/certificate-signature/v2"
        return canonical_json_bytes({"domain": domain, "certificate": self.core_dict()})

    def transition_hash(self) -> str:
        return hash_canonical("forge/preconsensus/transition/v2", self.core_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.core_dict()
        value["signatures"] = {
            hotkey: signature.hex() for hotkey, signature in sorted(self.signatures.items())
        }
        value["transition_hash"] = self.transition_hash()
        value["commits"] = [_commit_dict(commit) for commit in self.commits]
        value["reveals"] = [_reveal_dict(reveal) for reveal in self.reveals]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RoundCertificate:
        required = set(cls.__dataclass_fields__) | {"transition_hash"}
        if not isinstance(value, dict) or set(value) != required:
            raise CertificateError("certificate fields are invalid")
        try:
            commits = tuple(CommitRecord(**item) for item in value["commits"])
            reveals = tuple(
                Reveal(
                    LocalEvaluationReport.from_dict(item["report"]),
                    item["salt"],
                    item["reveal_block"],
                )
                for item in value["reveals"]
            )
            certificate = cls(
                network=value["network"],
                netuid=value["netuid"],
                round_id=value["round_id"],
                challenge_id=value["challenge_id"],
                plan_hash=value["plan_hash"],
                parent_state_hash=value["parent_state_hash"],
                previous_king_fingerprint=value["previous_king_fingerprint"],
                challenger_fingerprint=value["challenger_fingerprint"],
                challenger_hotkey=value["challenger_hotkey"],
                transition=Transition(value["transition"]),
                activation_block=value["activation_block"],
                activation_chain_time=value["activation_chain_time"],
                evidence_digest=value["evidence_digest"],
                commits=commits,
                reveals=reveals,
                signatures={
                    key: bytes.fromhex(signature) for key, signature in value["signatures"].items()
                },
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CertificateError("certificate serialization is invalid") from error
        if value["transition_hash"] != certificate.transition_hash():
            raise CertificateError("certificate transition hash differs from its payload")
        return certificate


def _commit_dict(commit: CommitRecord) -> dict[str, Any]:
    return {
        "validator_hotkey": commit.validator_hotkey,
        "commitment": commit.commitment,
        "commit_block": commit.commit_block,
    }


def _reveal_dict(reveal: Reveal) -> dict[str, Any]:
    return {
        "report": reveal.report.to_dict(),
        "salt": reveal.salt,
        "reveal_block": reveal.reveal_block,
    }


@dataclass(frozen=True)
class FinalizedCompetitionState:
    challenge_id: str
    state_version: int
    previous_state_hash: str
    current_king_fingerprint: str
    crown_history: tuple[Crown, ...]
    activation_block: int
    last_round_id: str | None
    last_transition_hash: str

    @classmethod
    def genesis(
        cls, challenge_id: str, baseline_fingerprint: str, activation_block: int = 0
    ) -> FinalizedCompetitionState:
        genesis_transition = hash_canonical(
            "forge/preconsensus/genesis/v1",
            challenge_id,
            baseline_fingerprint,
            activation_block,
        )
        return cls(
            challenge_id,
            0,
            "0" * 64,
            baseline_fingerprint,
            (),
            activation_block,
            None,
            genesis_transition,
        )

    def body_dict(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "state_version": self.state_version,
            "previous_state_hash": self.previous_state_hash,
            "current_king_fingerprint": self.current_king_fingerprint,
            "crown_history": [crown.__dict__ for crown in self.crown_history],
            "activation_block": self.activation_block,
            "last_round_id": self.last_round_id,
            "last_transition_hash": self.last_transition_hash,
        }

    def state_hash(self) -> str:
        return hash_canonical("forge/preconsensus/state/v1", self.body_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.body_dict()
        value["state_hash"] = self.state_hash()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FinalizedCompetitionState:
        state = cls(
            challenge_id=value["challenge_id"],
            state_version=int(value["state_version"]),
            previous_state_hash=value["previous_state_hash"],
            current_king_fingerprint=value["current_king_fingerprint"],
            crown_history=tuple(Crown(**item) for item in value["crown_history"]),
            activation_block=int(value["activation_block"]),
            last_round_id=value["last_round_id"],
            last_transition_hash=value["last_transition_hash"],
        )
        if value.get("state_hash", state.state_hash()) != state.state_hash():
            raise CertificateError("stored state hash does not match its payload")
        return state
