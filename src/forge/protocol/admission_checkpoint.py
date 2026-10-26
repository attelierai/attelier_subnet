"""Committee-signed admission-history checkpoints.

An admission SQLite file is only a cache.  This module turns an independently
trusted finalized state hash and committee snapshot into a verifier for the
history object consumed by :mod:`forge.validator.admission`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from forge.protocol.hashing import canonical_json_bytes
from forge.protocol.ports import Signer, Verifier
from forge.validator.admission import AuthenticatedAdmissionHistory

DOMAIN = "forge/admission-checkpoint-signature/v1"
WINDOWED_DOMAIN = "forge/admission-checkpoint-signature/v2"


class AdmissionCheckpointError(ValueError):
    """A signed history lacks independently verifiable committee authority."""


def _committee(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise AdmissionCheckpointError("trusted complete committee is required")
    committee = dict(value)
    if any(
        not isinstance(key, str) or not key or type(power) is not int or power <= 0
        for key, power in committee.items()
    ):
        raise AdmissionCheckpointError("trusted committee is invalid")
    return committee


def signing_payload(history: AuthenticatedAdmissionHistory) -> bytes:
    """Bind the canonical identity snapshot and externally trusted state root."""
    return canonical_json_bytes(
        {
            "domain": WINDOWED_DOMAIN if history.is_windowed else DOMAIN,
            "network": history.network,
            "netuid": history.netuid,
            "through_block": history.through_block,
            "certified_state_hash": history.certified_state_hash,
            "history_root": history.history_root,
            **(
                {
                    "window_id": history.window_id,
                    "window_open_block": history.window_open_block,
                    "window_close_block": history.window_close_block,
                }
                if history.is_windowed
                else {}
            ),
        }
    )


@dataclass(frozen=True)
class SignedAdmissionHistory:
    history: AuthenticatedAdmissionHistory
    signatures: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if not isinstance(self.history, AuthenticatedAdmissionHistory):
            raise AdmissionCheckpointError("signed checkpoint history is invalid")
        signatures = dict(self.signatures)
        if not signatures or any(
            not isinstance(hotkey, str)
            or not hotkey
            or not isinstance(signature, bytes)
            or not signature
            for hotkey, signature in signatures.items()
        ):
            raise AdmissionCheckpointError("checkpoint signatures are invalid")
        object.__setattr__(self, "signatures", signatures)


def create_signed_history(
    history: AuthenticatedAdmissionHistory, signers: Iterable[Signer]
) -> SignedAdmissionHistory:
    """Sign an already canonical checkpoint; committee authority is not supplied here."""
    payload, signatures = signing_payload(history), {}
    for signer in signers:
        hotkey = signer.validator_hotkey
        if hotkey in signatures:
            raise AdmissionCheckpointError("duplicate checkpoint signer")
        signatures[hotkey] = signer.sign(payload)
    return SignedAdmissionHistory(history, signatures)


def verify_signed_history(
    history: AuthenticatedAdmissionHistory,
    signed: SignedAdmissionHistory,
    *,
    trusted_committee: Mapping[str, int],
    expected_certified_state_hash: str,
    verifier: Verifier,
) -> bool:
    """Verify >2/3 of the supplied full committee, not a claimed committee."""
    try:
        committee = _committee(trusted_committee)
        if history != signed.history:
            return False
        # Re-running canonical construction catches any object forged with
        # frozen dataclass internals after its constructor ran.
        if history.history_root != history.recomputed_root():
            return False
        if history.certified_state_hash != expected_certified_state_hash:
            return False
        if set(signed.signatures) - set(committee):
            return False
        payload, signed_power, total_power = signing_payload(history), 0, sum(committee.values())
        for hotkey, signature in signed.signatures.items():
            if verifier.verify(hotkey, payload, signature):
                signed_power += committee[hotkey]
            else:
                return False
        return signed_power * 3 > total_power * 2
    except (AdmissionCheckpointError, TypeError, ValueError):
        return False


def history_verifier(
    signed: SignedAdmissionHistory,
    *,
    trusted_committee: Mapping[str, int],
    expected_certified_state_hash: str,
    verifier: Verifier,
):
    """Return the fail-closed callback required by ``AdmissionLedger.admit``."""

    _committee(trusted_committee)

    def verify(history: AuthenticatedAdmissionHistory) -> bool:
        return verify_signed_history(
            history,
            signed,
            trusted_committee=trusted_committee,
            expected_certified_state_hash=expected_certified_state_hash,
            verifier=verifier,
        )

    return verify
