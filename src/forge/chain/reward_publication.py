"""Fail-closed testnet weight intent and finalized readback verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.hashing import hash_canonical


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identity(network: object, netuid: object) -> tuple[str, int]:
    if not isinstance(network, str) or not network:
        raise ValueError("network is required")
    if type(netuid) is not int or netuid < 0:
        raise ValueError("netuid must be a nonnegative integer")
    return network, netuid


@dataclass(frozen=True)
class WeightIntent:
    network: str
    netuid: int
    certified_state_hash: str
    certificate_hash: str
    runtime_hash: str
    artifact_custody_hash: str
    burn_qualification_hash: str | None
    status: str
    uid_weights: dict[int, float]
    expected_hotkeys: dict[int, str]
    reason: str

    def __post_init__(self) -> None:
        _identity(self.network, self.netuid)
        for field in (
            "certified_state_hash",
            "certificate_hash",
            "runtime_hash",
            "artifact_custody_hash",
        ):
            _hash(getattr(self, field), field)
        if self.burn_qualification_hash is not None:
            _hash(self.burn_qualification_hash, "burn qualification hash")

    def intent_hash(self) -> str:
        return hash_canonical(
            "forge/weight-intent/v2",
            {
                "network": self.network,
                "netuid": self.netuid,
                "certified_state_hash": self.certified_state_hash,
                "certificate_hash": self.certificate_hash,
                "runtime_hash": self.runtime_hash,
                "artifact_custody_hash": self.artifact_custody_hash,
                "burn_qualification_hash": self.burn_qualification_hash,
                "status": self.status,
                "uid_weights": {str(uid): weight for uid, weight in self.uid_weights.items()},
                "expected_hotkeys": {str(uid): key for uid, key in self.expected_hotkeys.items()},
                "reason": self.reason,
            },
        )


@dataclass(frozen=True)
class WeightSubmissionReceipt:
    intent_hash: str
    network: str
    netuid: int
    submitted_block: int
    submitted_extrinsic_hash: str

    def __post_init__(self) -> None:
        _hash(self.intent_hash, "intent hash")
        _identity(self.network, self.netuid)
        if type(self.submitted_block) is not int or self.submitted_block < 0:
            raise ValueError("submitted block must be nonnegative")
        _hash(self.submitted_extrinsic_hash, "submitted extrinsic hash")


def _new_intent(
    state: FinalizedCompetitionState,
    network: str,
    netuid: int,
    certificate_hash: str,
    runtime_hash: str,
    custody_hash: str,
    burn_hash: str | None,
    status: str,
    weights: dict[int, float],
    hotkeys: dict[int, str],
    reason: str,
) -> WeightIntent:
    return WeightIntent(
        network,
        netuid,
        state.state_hash(),
        certificate_hash,
        runtime_hash,
        custody_hash,
        burn_hash,
        status,
        weights,
        hotkeys,
        reason,
    )


def testnet_weight_intent(
    state: FinalizedCompetitionState,
    *,
    network: str,
    netuid: int,
    certificate_hash: str,
    runtime_hash: str,
    registered_uids: Mapping[int, str],
    public_custody: Mapping[str, str],
    verify_public_custody: Callable[[str, str], bool] | None,
    burn_hotkey: str | None,
    verify_burn_qualification: Callable[[str], str | None] | None,
) -> WeightIntent:
    """Build a testnet-only intent from externally verified custody receipts.

    Bare artifact fingerprints and a burn hotkey string never enable publication.
    Callbacks must verify the external receipts before this pure function can
    name a winner or burn target.
    """
    network, netuid = _identity(network, netuid)
    _hash(certificate_hash, "certificate hash")
    _hash(runtime_hash, "runtime hash")
    custody_hash = hash_canonical("forge/public-custody-set/v1", dict(public_custody))
    burn_hash = (
        verify_burn_qualification(burn_hotkey)
        if burn_hotkey and verify_burn_qualification
        else None
    )
    if burn_hash is not None:
        _hash(burn_hash, "burn qualification receipt")
    common = (state, network, netuid, certificate_hash, runtime_hash, custody_hash, burn_hash)
    if network != "test":
        return _new_intent(*common, "BLOCKED", {}, {}, "MVP supports testnet only")
    if not burn_hotkey or burn_hash is None:
        return _new_intent(*common, "BLOCKED", {}, {}, "target-runtime burn receipt is unverified")
    if len(set(registered_uids.values())) != len(registered_uids) or any(
        type(uid) is not int or uid < 0 or not isinstance(key, str) or not key
        for uid, key in registered_uids.items()
    ):
        return _new_intent(*common, "BLOCKED", {}, {}, "UID snapshot is invalid")
    burn_uid = next((uid for uid, key in registered_uids.items() if key == burn_hotkey), None)
    if burn_uid is None:
        return _new_intent(
            *common, "BLOCKED", {}, {}, "qualified burn hotkey is no longer registered"
        )
    latest = state.crown_history[-1] if state.crown_history else None
    winner_uid = None
    if latest and latest.fingerprint == state.current_king_fingerprint:
        receipt = public_custody.get(latest.fingerprint)
        if receipt is not None and verify_public_custody is not None:
            _hash(receipt, "public custody receipt")
            if verify_public_custody(latest.fingerprint, receipt):
                winner_uid = next(
                    (uid for uid, key in registered_uids.items() if key == latest.hotkey), None
                )
    if winner_uid is None:
        return _new_intent(
            *common,
            "BURN_REQUIRED",
            {burn_uid: 1.0},
            {burn_uid: burn_hotkey},
            "no current publicly verified winner; replace weights with burn target",
        )
    return _new_intent(
        *common,
        "WINNER",
        {winner_uid: 1.0},
        {winner_uid: latest.hotkey},
        "100% goes to the current certified publicly verified winner",
    )


def record_submission(
    intent: WeightIntent, *, submitted_block: int, submitted_extrinsic_hash: str
) -> WeightSubmissionReceipt:
    if intent.status not in {"WINNER", "BURN_REQUIRED"}:
        raise ValueError("blocked intent cannot be submitted")
    return WeightSubmissionReceipt(
        intent.intent_hash(),
        intent.network,
        intent.netuid,
        submitted_block,
        submitted_extrinsic_hash,
    )


def verify_finalized_replacement(
    intent: WeightIntent,
    receipt: WeightSubmissionReceipt,
    *,
    network: str,
    netuid: int,
    finalized_block: int,
    finalized_extrinsic_hash: str,
    finalized_runtime_hash: str,
    weights_u16: Mapping[int, int],
    registered_uids: Mapping[int, str],
) -> None:
    """Verify a final chain readback matches this exact intent and submission."""
    if intent.status not in {"WINNER", "BURN_REQUIRED"} or len(intent.uid_weights) != 1:
        raise ValueError("no publishable weight intent")
    if receipt.intent_hash != intent.intent_hash() or (receipt.network, receipt.netuid) != (
        intent.network,
        intent.netuid,
    ):
        raise ValueError("submission receipt is not bound to this intent")
    if _identity(network, netuid) != (intent.network, intent.netuid):
        raise ValueError("finalized readback belongs to another subnet")
    if type(finalized_block) is not int or finalized_block < receipt.submitted_block:
        raise ValueError("weight replacement has no finalized readback at/after submission")
    if finalized_extrinsic_hash != receipt.submitted_extrinsic_hash:
        raise ValueError("finalized extrinsic differs from recorded submission")
    if finalized_runtime_hash != intent.runtime_hash:
        raise ValueError("runtime hash differs from the qualified intent")
    if any(registered_uids.get(uid) != key for uid, key in intent.expected_hotkeys.items()):
        raise ValueError("target UID identity changed before finalized readback")
    if any(type(value) is not int or not 0 <= value <= 65535 for value in weights_u16.values()):
        raise ValueError("invalid chain weight row")
    expected_uid = next(iter(intent.uid_weights))
    if {uid: value for uid, value in weights_u16.items() if value} != {expected_uid: 65535}:
        raise ValueError("old/non-target weights remain or replacement is absent")


# This public API deliberately starts with ``testnet``; prevent pytest from
# interpreting an imported helper as a test function in integration modules.
testnet_weight_intent.__test__ = False
