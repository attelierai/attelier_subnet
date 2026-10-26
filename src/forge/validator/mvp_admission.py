"""Public candidate admission bridge for the bounded MVP queue.

This module performs no download, render, or GPU work.  It consumes only a
previous validator-side preflight receipt and a separately authenticated
committee checkpoint; a JSON bundle cannot name its own trusted committee.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from forge.models.artifact_manifest import ModelManifest
from forge.protocol.admission_checkpoint import SignedAdmissionHistory, history_verifier
from forge.protocol.hashing import hash_canonical
from forge.protocol.ports import Verifier
from forge.validator.admission import (
    Admission,
    AdmissionError,
    AdmissionLedger,
    FinalizedCommitment,
)
from forge.validator.runtime_preflight import PASS


class PublicAdmissionError(AdmissionError):
    """A finalized public candidate lacks frozen validator evidence."""


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise PublicAdmissionError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class PublicCandidate:
    commitment: FinalizedCommitment
    manifest: ModelManifest
    effective_weights_hash: str
    panel_salt_commitment: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ModelManifest):
            raise PublicAdmissionError("public candidate manifest is required")
        _hash(self.effective_weights_hash, "effective weights hash")
        _hash(self.panel_salt_commitment, "panel salt commitment")
        if self.commitment.artifact_hash != self.manifest.fingerprint():
            raise PublicAdmissionError("finalized artifact hash differs from public manifest")
        recipe = hash_canonical(
            "forge/mvp-admission-recipe/v1",
            {
                "manifest_hash": self.manifest.fingerprint(),
                "generation_config": dict(self.manifest.generation_config),
                "effective_weights_hash": self.effective_weights_hash,
            },
        )
        if self.commitment.recipe_hash != recipe:
            raise PublicAdmissionError("finalized recipe differs from effective public candidate")
        if self.commitment.panel_hash != self.panel_salt_commitment:
            raise PublicAdmissionError("finalized panel commitment differs from frozen panel salt")


def admit_public_candidates(
    ledger: AdmissionLedger,
    candidates: Iterable[PublicCandidate],
    *,
    preflight_receipt: Mapping[str, Any],
    expected_preflight_plan_hash: str,
    signed_history: SignedAdmissionHistory,
    trusted_committee: Mapping[str, int],
    expected_certified_state_hash: str,
    signature_verifier: Verifier,
) -> tuple[Admission, ...]:
    """Admit a canonical finalized batch after public/preflight evidence checks."""
    _hash(expected_preflight_plan_hash, "expected preflight plan hash")
    if (
        not isinstance(preflight_receipt, Mapping)
        or preflight_receipt.get("plan_hash") != expected_preflight_plan_hash
    ):
        raise PublicAdmissionError("preflight receipt is not bound to the trusted frozen plan")
    rows = preflight_receipt.get("candidates")
    if not isinstance(rows, list):
        raise PublicAdmissionError("preflight receipt candidates are required")
    items = tuple(candidates)
    if not items:
        return ()
    if items != tuple(
        sorted(items, key=lambda item: (item.commitment.block, item.commitment.extrinsic_index))
    ):
        raise PublicAdmissionError("finalized candidates must use canonical block/extrinsic order")
    seen_effective: set[str] = set()
    rows_by_manifest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("manifest_hash"), str):
            if row["manifest_hash"] in rows_by_manifest:
                raise PublicAdmissionError("preflight receipt repeats a manifest")
            rows_by_manifest[row["manifest_hash"]] = row
    for candidate in items:
        manifest_hash = candidate.manifest.fingerprint()
        row = rows_by_manifest.get(manifest_hash)
        if row is None or row.get("status") != PASS:
            raise PublicAdmissionError("candidate lacks a passing public runtime preflight")
        if row.get("effective_candidate_hash") != candidate.effective_weights_hash:
            raise PublicAdmissionError(
                "preflight effective weights differ from finalized candidate"
            )
        if candidate.effective_weights_hash in seen_effective:
            raise PublicAdmissionError(
                "duplicate effective weights cannot consume another admission"
            )
        seen_effective.add(candidate.effective_weights_hash)
    verifier = history_verifier(
        signed_history,
        trusted_committee=trusted_committee,
        expected_certified_state_hash=expected_certified_state_hash,
        verifier=signature_verifier,
    )
    return ledger.admit_finalized_batch(
        (item.commitment for item in items),
        history=signed_history.history,
        verify_history=verifier,
        effective_hashes={
            item.commitment.commitment_hash: item.effective_weights_hash for item in items
        },
        effective_aliases={
            item.commitment.commitment_hash: tuple(
                value
                for value in (
                    rows_by_manifest[item.manifest.fingerprint()].get(
                        "legacy_effective_candidate_hash"
                    ),
                )
                if isinstance(value, str)
            )
            for item in items
        },
    )
