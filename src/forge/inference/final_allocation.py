"""Verify a gateway-decided SN126/M0 allocation without recalculating merit.

The gateway is the accounting authority.  Validators only authenticate the
immutable allocation, pin its namespace/window/recipient identities, and use
the existing M0 publisher for commit-reveal.  This deliberately has no model,
TEE, bid, price, or provider-invoice logic.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.inference.light import (
    STATUS_SCHEMA,
    _atomic_write,
    _raw_public_key,
    _strict_json,
    _verify_ed25519,
)
from forge.inference.prelaunch import GENESIS, HOUR_MS, canonical, digest

SCHEMA = "attelier/final-allocation-feed/v1"
BODY_SCHEMA = "attelier/final-allocation/v1"
DOMAIN = b"attelier/final-allocation/v1\0"
U16_MAX = 65_535


class FinalAllocationError(ValueError):
    """A final allocation is unsafe to publish or echo."""


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _identity(value: Any) -> bool:
    return isinstance(value, str) and value.isascii() and 1 <= len(value) <= 128


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_body(
    body: dict[str, Any],
    *,
    expected_network: str = "finney",
    expected_netuid: int = 126,
    expected_genesis: str = GENESIS,
) -> None:
    required = {
        "schema",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "issuedAtMs",
        "expiresAtMs",
        "window",
        "sequence",
        "policyHash",
        "allocationHash",
        "registry",
        "primaryValidatorHotkey",
        "weights",
    }
    if set(body) != required or body.get("schema") != BODY_SCHEMA:
        raise FinalAllocationError("final_allocation_body_schema")
    if (
        body.get("network") != expected_network
        or body.get("genesis") != expected_genesis
        or body.get("netuid") != expected_netuid
        or body.get("mechanism") != 0
        or not _identity(body.get("primaryValidatorHotkey"))
    ):
        raise FinalAllocationError("final_allocation_namespace")
    if (
        not all(_integer(body.get(key)) for key in ("issuedAtMs", "expiresAtMs", "sequence"))
        or body["sequence"] < 1
    ):
        raise FinalAllocationError("final_allocation_timing")
    window = body.get("window")
    if (
        not isinstance(window, dict)
        or set(window) != {"startMs", "endMs"}
        or not all(_integer(window.get(key)) for key in window)
        or window["endMs"] - window["startMs"] != HOUR_MS
        or not window["endMs"] <= body["issuedAtMs"] <= body["expiresAtMs"]
        or body["expiresAtMs"] != window["endMs"] + HOUR_MS
    ):
        raise FinalAllocationError("final_allocation_timing")
    registry = body.get("registry")
    if (
        not isinstance(registry, dict)
        or set(registry)
        != {"finalizedBlock", "finalizedBlockHash", "finalizedAtMs", "subnetOwnerColdkey"}
        or not _integer(registry.get("finalizedBlock"))
        or not _hash(registry.get("finalizedBlockHash"))
        or not _integer(registry.get("finalizedAtMs"))
        or not _identity(registry.get("subnetOwnerColdkey"))
        or not window["endMs"] <= registry["finalizedAtMs"] <= body["issuedAtMs"]
    ):
        raise FinalAllocationError("final_allocation_registry")
    if not _hash(body.get("policyHash")) or not _hash(body.get("allocationHash")):
        raise FinalAllocationError("final_allocation_hash")
    weights = body.get("weights")
    if not isinstance(weights, list) or not weights:
        raise FinalAllocationError("final_allocation_weights")
    expected_hash = digest({"registry": registry, "weights": weights})
    if body["allocationHash"] != expected_hash:
        raise FinalAllocationError("final_allocation_hash")
    seen_uids: set[int] = set()
    seen_hotkeys: set[str] = set()
    last_uid = -1
    for row in weights:
        if not isinstance(row, dict) or set(row) != {"uid", "hotkey", "coldkey", "weight"}:
            raise FinalAllocationError("final_allocation_weights")
        uid, weight = row.get("uid"), row.get("weight")
        if (
            type(uid) is not int
            or uid < 0
            or uid <= last_uid
            or uid in seen_uids
            or type(weight) is not int
            or not 0 <= weight <= U16_MAX
            or weight == 0
            and uid != 0
            or not _identity(row.get("hotkey"))
            or row["hotkey"] in seen_hotkeys
            or not _identity(row.get("coldkey"))
        ):
            raise FinalAllocationError("final_allocation_weights")
        seen_uids.add(uid)
        seen_hotkeys.add(row["hotkey"])
        last_uid = uid
    burn = weights[0]
    if (
        burn["uid"] != 0
        or burn["coldkey"] != registry["subnetOwnerColdkey"]
        or sum(row["weight"] for row in weights) != U16_MAX
    ):
        raise FinalAllocationError("final_allocation_burn")


def verify_final_allocation(
    raw: bytes,
    public_key: str,
    *,
    expected_policy_hash: str | None = None,
    now_ms: int | None = None,
    expected_network: str = "finney",
    expected_netuid: int = 126,
    expected_genesis: str = GENESIS,
) -> dict[str, Any]:
    """Verify one fresh, complete M0 allocation from the pinned gateway key."""
    envelope = _strict_json(raw)
    if (
        set(envelope) != {"schema", "keyId", "body", "digest", "signature"}
        or envelope.get("schema") != SCHEMA
    ):
        raise FinalAllocationError("final_allocation_envelope_schema")
    body = envelope.get("body")
    if not isinstance(body, dict):
        raise FinalAllocationError("final_allocation_body_schema")
    _validate_body(
        body,
        expected_network=expected_network,
        expected_netuid=expected_netuid,
        expected_genesis=expected_genesis,
    )
    key = _raw_public_key(public_key)
    if (
        envelope.get("keyId") != hashlib.sha256(key).hexdigest()
        or envelope.get("digest") != digest(body)
        or not isinstance(envelope.get("signature"), str)
        or not _verify_ed25519(key, DOMAIN + canonical(body), envelope["signature"])
    ):
        raise FinalAllocationError("final_allocation_signature")
    if expected_policy_hash is not None and body["policyHash"] != expected_policy_hash:
        raise FinalAllocationError("final_allocation_policy")
    current = int(time.time() * 1000) if now_ms is None else now_ms
    if not _integer(current) or current < body["issuedAtMs"] or current >= body["expiresAtMs"]:
        raise FinalAllocationError("final_allocation_expired_or_not_yet_valid")
    return body


def final_targets(
    body: dict[str, Any],
    *,
    expected_network: str = "finney",
    expected_netuid: int = 126,
    expected_genesis: str = GENESIS,
) -> dict[int, tuple[str, str]]:
    """Convert a validated feed to the exact identity tuples used by publication."""
    _validate_body(
        body,
        expected_network=expected_network,
        expected_netuid=expected_netuid,
        expected_genesis=expected_genesis,
    )
    return {row["uid"]: (row["hotkey"], row["coldkey"]) for row in body["weights"]}


def final_weights(
    body: dict[str, Any],
    *,
    expected_network: str = "finney",
    expected_netuid: int = 126,
    expected_genesis: str = GENESIS,
) -> dict[int, int]:
    _validate_body(
        body,
        expected_network=expected_network,
        expected_netuid=expected_netuid,
        expected_genesis=expected_genesis,
    )
    return {row["uid"]: row["weight"] for row in body["weights"]}


def onchain_weights(
    body: dict[str, Any],
    *,
    expected_network: str = "finney",
    expected_netuid: int = 126,
    expected_genesis: str = GENESIS,
) -> dict[int, int]:
    """Match the SDK's max-upscale quantization for a 65,535-share allocation.

    Gateway values deliberately sum to 65,535 because they are allocation
    shares.  Subtensor receives a relative u16 vector, whose largest item is
    65,535.  Publishing or echoing the shares as if they were raw chain values
    would produce a false mismatch whenever burn is not the largest allocation.
    """
    shares = final_weights(
        body,
        expected_network=expected_network,
        expected_netuid=expected_netuid,
        expected_genesis=expected_genesis,
    )
    top = max(shares.values())
    return {uid: round((weight / top) * U16_MAX) for uid, weight in shares.items()}


def verify_primary_echo(
    body: dict[str, Any],
    observed: dict[str, Any],
    *,
    expected_network: str = "finney",
    expected_netuid: int = 126,
    expected_genesis: str = GENESIS,
) -> None:
    """Reject an echo until chain observation matches the gateway vector exactly.

    ``observed`` is deliberately a narrow adapter boundary.  The caller obtains
    it from its own finalized chain RPC; this module never trusts a remote
    validator's HTTP claim.
    """
    _validate_body(
        body,
        expected_network=expected_network,
        expected_netuid=expected_netuid,
        expected_genesis=expected_genesis,
    )
    if (
        not isinstance(observed, dict)
        or set(observed)
        != {
            "network",
            "netuid",
            "mechanism",
            "hotkey",
            "weights",
            "observedBlock",
            "observedBlockHash",
            "lastUpdate",
        }
        or observed.get("network") != expected_network
        or observed.get("netuid") != expected_netuid
        or observed.get("mechanism") != 0
        or observed.get("hotkey") != body["primaryValidatorHotkey"]
        or not _integer(observed.get("observedBlock"))
        or not _hash(observed.get("observedBlockHash"))
        or not _integer(observed.get("lastUpdate"))
        or not body["registry"]["finalizedBlock"]
        <= observed["lastUpdate"]
        <= observed["observedBlock"]
        or not isinstance(observed.get("weights"), list)
    ):
        raise FinalAllocationError("final_allocation_primary_observation")
    expected = [
        [uid, weight]
        for uid, weight in sorted(
            onchain_weights(
                body,
                expected_network=expected_network,
                expected_netuid=expected_netuid,
                expected_genesis=expected_genesis,
            ).items()
        )
    ]
    if observed["weights"] != expected:
        raise FinalAllocationError("final_allocation_primary_mismatch")


class FinalAllocationRunner:
    """Durably accept one authenticated allocation per gateway sequence."""

    def __init__(
        self,
        state_dir: Path,
        public_key: str,
        *,
        policy_hash: str | None = None,
        registry_verifier: Callable[[dict[str, Any]], None] | None = None,
        expected_network: str = "finney",
        expected_netuid: int = 126,
        expected_genesis: str = GENESIS,
    ) -> None:
        _raw_public_key(public_key)
        if policy_hash is not None and not _hash(policy_hash):
            raise FinalAllocationError("final_allocation_policy")
        self.state_dir, self.public_key, self.policy_hash = state_dir, public_key, policy_hash
        self.registry_verifier = registry_verifier
        self.expected_network = expected_network
        self.expected_netuid = expected_netuid
        self.expected_genesis = expected_genesis
        state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = open(  # noqa: SIM115 - lock must live for this runner instance.
            state_dir / "final-allocation.lock", "a+", encoding="utf-8"
        )
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise FinalAllocationError("final_allocation_state_locked") from error
        self.database = state_dir / "light.sqlite3"
        try:
            with sqlite3.connect(self.database) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS final_allocation ("
                    "sequence INTEGER PRIMARY KEY, digest TEXT NOT NULL UNIQUE, "
                    "raw BLOB NOT NULL, body BLOB NOT NULL)"
                )
        except sqlite3.Error as error:
            raise FinalAllocationError("final_allocation_state_unavailable") from error

    def process(self, raw: bytes, *, now_ms: int | None = None) -> dict[str, Any]:
        body = verify_final_allocation(
            raw,
            self.public_key,
            expected_policy_hash=self.policy_hash,
            now_ms=now_ms,
            expected_network=self.expected_network,
            expected_netuid=self.expected_netuid,
            expected_genesis=self.expected_genesis,
        )
        if self.registry_verifier is not None:
            self.registry_verifier(
                {
                    **body["registry"],
                    "mechanism": 0,
                    "miners": [
                        {"uid": row["uid"], "hotkey": row["hotkey"], "coldkey": row["coldkey"]}
                        for row in body["weights"]
                    ],
                }
            )
        feed_digest = digest(body)
        try:
            with sqlite3.connect(self.database, timeout=30, isolation_level=None) as connection:
                connection.execute("BEGIN IMMEDIATE")
                latest = connection.execute(
                    "SELECT sequence, digest, body FROM final_allocation "
                    "ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                if latest is not None:
                    if body["sequence"] < latest[0] or (
                        body["sequence"] == latest[0] and feed_digest != latest[1]
                    ):
                        raise FinalAllocationError("final_allocation_replay_or_rollback")
                    if body["sequence"] > latest[0]:
                        prior = json.loads(latest[2])
                        if body["window"]["startMs"] < prior["window"]["endMs"]:
                            raise FinalAllocationError("final_allocation_window_overlap")
                if latest is None or body["sequence"] != latest[0]:
                    connection.execute(
                        "INSERT INTO final_allocation(sequence,digest,raw,body) VALUES (?,?,?,?)",
                        (body["sequence"], feed_digest, raw, canonical(body)),
                    )
                connection.execute("COMMIT")
            _atomic_write(self.state_dir / "latest-final-allocation.json", body)
            _atomic_write(
                self.state_dir / "status.json",
                {
                    "schema": STATUS_SCHEMA,
                    "status": "ready",
                    "sequence": body["sequence"],
                    "digest": feed_digest,
                },
            )
            return body
        except FinalAllocationError:
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            raise FinalAllocationError("final_allocation_state_unavailable") from error

    def unavailable(self, reason: str) -> None:
        """Share the light-runner status contract without creating a burn input."""
        with contextlib.suppress(FinalAllocationError, OSError, ValueError):
            _atomic_write(
                self.state_dir / "status.json",
                {"schema": STATUS_SCHEMA, "status": "unavailable", "reason": reason},
            )
