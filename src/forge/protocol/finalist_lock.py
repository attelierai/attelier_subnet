"""Immutable finalist lock digest for a confirmation panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from forge.protocol.hashing import hash_canonical

SCHEMA = "forge/finalist-lock/v1"
_FIELDS = {
    "schema",
    "network",
    "netuid",
    "round_id",
    "close_block",
    "close_block_hash",
    "contract_hash",
    "candidate_manifest_hash",
    "candidate_hotkey",
    "candidate_admission_block",
    "incumbent_state_hash",
}


def _chain_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 66
        or not value.startswith("0x")
        or any(c not in "0123456789abcdef" for c in value[2:])
    ):
        raise ValueError(f"{label} must be a 0x-prefixed lowercase block hash")
    return value


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _block(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def build_finalist_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS or value.get("schema") != SCHEMA:
        raise ValueError("finalist lock fields are invalid")
    result = dict(value)
    if (
        not isinstance(result["network"], str)
        or not result["network"]
        or not isinstance(result["round_id"], str)
        or not result["round_id"]
        or not isinstance(result["candidate_hotkey"], str)
        or not result["candidate_hotkey"]
    ):
        raise ValueError("finalist lock identity is invalid")
    for name in ("netuid", "close_block", "candidate_admission_block"):
        _block(result[name], name)
    if result["candidate_admission_block"] > result["close_block"]:
        raise ValueError("candidate admission is after close")
    _chain_hash(result["close_block_hash"], "close_block_hash")
    for name in (
        "contract_hash",
        "candidate_manifest_hash",
        "incumbent_state_hash",
    ):
        _sha(result[name], name)
    return result


def finalist_lock_hash(value: Mapping[str, Any]) -> str:
    return hash_canonical(SCHEMA, build_finalist_lock(value))


class LockHistoryPort(Protocol):
    def commitment(self, hotkey: str, height: int) -> Any: ...


def first_valid_lock_after_close(
    chain: LockHistoryPort,
    committee: Mapping[str, int],
    admissions: list[Mapping[str, Any]],
    *,
    network: str,
    netuid: int,
    round_id: str,
    close_block: int,
    close_hash: str,
    contract_hash: str,
    incumbent_state_hash: str,
    upper_block: int,
    max_changes: int = 64,
) -> tuple[int, str, dict[str, object]] | None:
    expected: dict[str, dict[str, object]] = {}
    for admission in admissions:
        lock = build_finalist_lock(
            {
                "schema": SCHEMA,
                "network": network,
                "netuid": netuid,
                "round_id": round_id,
                "close_block": close_block,
                "close_block_hash": close_hash,
                "contract_hash": contract_hash,
                "candidate_manifest_hash": admission["manifest_hash"],
                "candidate_hotkey": admission["hotkey"],
                "candidate_admission_block": admission["block"],
                "incumbent_state_hash": incumbent_state_hash,
            }
        )
        expected[finalist_lock_hash(lock)] = lock
    found = []
    for hotkey in sorted(committee):
        cursor, changes = upper_block, 0
        while cursor > close_block:
            observed = chain.commitment(hotkey, cursor)
            if observed is None:
                break
            if (
                getattr(observed, "hotkey", None) != hotkey
                or getattr(observed, "block", -1) > cursor
                or getattr(observed, "block", -1) < 0
            ):
                raise ValueError("finalist-lock history is malformed")
            if observed.block <= close_block:
                break
            changes += 1
            if changes > max_changes:
                raise ValueError(
                    f"finalist-lock history exceeds the {max_changes}-change audit limit"
                )
            if (
                getattr(observed, "kind", None) == "finalist_lock"
                and len(getattr(observed, "fields", ())) == 1
            ):
                lock = expected.get(observed.fields[0])
                if lock is None:
                    # A later resolver may no longer derive a candidate that
                    # was public at C.  Ignoring this marker would let a second
                    # lock reinterpret the round after that disappearance.
                    raise ValueError(
                        "finalized finalist lock is unknown after close; "
                        "overlapping locks are ambiguous"
                    )
                found.append((observed.block, hotkey, lock))
            cursor = observed.block - 1
    return min(found, key=lambda item: (item[0], item[1])) if found else None
