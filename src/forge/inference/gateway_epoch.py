"""Offline, signed gateway windows for the testnet inference mechanism.

Gateway telemetry is admitted only through this file's canonical, pinned-key
format.  It is deliberately not a chain publisher and never spends funds.
"""

from __future__ import annotations

import base64
import hashlib
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Verifier
from forge.inference.trace_protocol import MAX_SAFE_INTEGER, canonical

GATEWAY_SCHEMA = "attelier/gateway-window/v1"
SIGNED_SCHEMA = "attelier/gateway-signed-window/v1"
MASTER_SCHEMA = "attelier/gateway-master-window/v1"
GATEWAY_DOMAIN = b"attelier/gateway-window/v1\0"
MASTER_DOMAIN = b"attelier/gateway-master-window/v1\0"
NAMESPACE = "attelier:m1:v1"
_AGGREGATE_FIELDS = {
    "minerHotkey",
    "workerId",
    "trustMode",
    "trafficKind",
    "profileId",
    "profileHash",
    "attempted",
    "succeeded",
    "failed",
    "cancelled",
    "inconclusive",
    "completedVideoMs",
    "eligibleSucceeded",
    "eligibleVideoMs",
    "slowSucceeded",
    "TTVp50Ms",
    "TTVp95Ms",
    "computeAmountMicros",
    "knownComputeAmountMicros",
    "unknownComputeCount",
    "billedAmountMicros",
    "knownBilledAmountMicros",
    "unknownBilledCount",
    "billedUnit",
}


def digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def canonical_block_hash(value: Any) -> str:
    """Normalize an RPC block hash before it enters a signed M1 artifact.

    Substrate RPC values are conventionally ``0x``-prefixed whereas protocol
    digests are bare lowercase hex.  Signed gateway artifacts use the latter
    everywhere, avoiding a representation-dependent registry anchor.
    """
    if not isinstance(value, str):
        raise ValueError("gateway_block_hash")
    normalized = value.removeprefix("0x")
    if not _hash(normalized):
        raise ValueError("gateway_block_hash")
    return normalized


def _integer(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_SAFE_INTEGER


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and value.isascii() and 0 < len(value) <= 128


def _ed25519_verify(raw_key: str, message: bytes, signature: str) -> bool:
    try:
        key, sig = (
            base64.b64decode(raw_key, validate=True),
            base64.b64decode(signature, validate=True),
        )
        if len(key) != 32 or len(sig) != 64:
            return False
        with tempfile.TemporaryDirectory(prefix="gateway-ed25519-") as temporary:
            root = Path(temporary)
            (root / "key.der").write_bytes(bytes.fromhex("302a300506032b6570032100") + key)
            (root / "message").write_bytes(message)
            (root / "signature").write_bytes(sig)
            return (
                subprocess.run(
                    [
                        "openssl",
                        "pkeyutl",
                        "-verify",
                        "-pubin",
                        "-keyform",
                        "DER",
                        "-inkey",
                        str(root / "key.der"),
                        "-rawin",
                        "-in",
                        str(root / "message"),
                        "-sigfile",
                        str(root / "signature"),
                    ],
                    capture_output=True,
                    timeout=15,
                    check=False,
                ).returncode
                == 0
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def sign_gateway(body: dict[str, Any], private_key: Path) -> dict[str, Any]:
    _validate_body(body)
    with tempfile.TemporaryDirectory(prefix="gateway-sign-") as temporary:
        root = Path(temporary)
        message = GATEWAY_DOMAIN + canonical(body)
        (root / "message").write_bytes(message)
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private_key),
                "-rawin",
                "-in",
                str(root / "message"),
            ],
            capture_output=True,
            timeout=15,
            check=True,
        )
    return {
        "schema": SIGNED_SCHEMA,
        "body": body,
        "digest": digest(body),
        "signature": base64.b64encode(result.stdout).decode(),
    }


def _validate_body(body: dict[str, Any]) -> None:
    required = {
        "schema",
        "network",
        "netuid",
        "mechanism",
        "window",
        "sequence",
        "createdAtMs",
        "expiresAtMs",
        "policyHash",
        "poolHash",
        "ledgerRowsHash",
        "previousHash",
        "aggregates",
    }
    if (
        set(body) != required
        or body.get("schema") != GATEWAY_SCHEMA
        or body.get("network") != "test"
        or body.get("netuid") != 578
        or body.get("mechanism") != 1
    ):
        raise ValueError("gateway_body_schema")
    window = body["window"]
    if (
        not isinstance(window, dict)
        or set(window) != {"startMs", "endMs"}
        or not all(_integer(window[key]) for key in window)
        or window["startMs"] >= window["endMs"]
    ):
        raise ValueError("gateway_window")
    if (
        not all(_integer(body[key]) for key in ("sequence", "createdAtMs", "expiresAtMs"))
        or body["sequence"] < 1
        or not window["endMs"] <= body["createdAtMs"] <= body["expiresAtMs"]
    ):
        raise ValueError("gateway_freshness")
    if (
        any(not _hash(body[key]) for key in ("policyHash", "poolHash", "ledgerRowsHash"))
        or body["previousHash"] is not None
        and not _hash(body["previousHash"])
    ):
        raise ValueError("gateway_hashes")
    if not isinstance(body["aggregates"], list) or len(body["aggregates"]) > 4096:
        raise ValueError("gateway_aggregates")
    seen = set()
    for row in body["aggregates"]:
        if not isinstance(row, dict) or set(row) != _AGGREGATE_FIELDS:
            raise ValueError("gateway_aggregate_schema")
        if not _identifier(row["minerHotkey"]) or not _identifier(row["workerId"]):
            raise ValueError("gateway_worker_identity")
        if row["trustMode"] not in {"operator_whitelist", "trace_verified"} or row[
            "trafficKind"
        ] not in {"organic", "synthetic"}:
            raise ValueError("gateway_trust_or_traffic")
        valid_profile = (
            row["profileId"] is None
            and row["profileHash"] is None
            or _identifier(row["profileId"])
            and _hash(row["profileHash"])
        )
        if not valid_profile:
            raise ValueError("gateway_profile")
        counts = (
            "attempted",
            "succeeded",
            "failed",
            "cancelled",
            "inconclusive",
            "completedVideoMs",
            "eligibleSucceeded",
            "eligibleVideoMs",
            "slowSucceeded",
            "knownComputeAmountMicros",
            "unknownComputeCount",
            "knownBilledAmountMicros",
            "unknownBilledCount",
        )
        if not all(_integer(row[key]) for key in counts) or row["attempted"] != sum(
            row[key] for key in ("succeeded", "failed", "cancelled", "inconclusive")
        ):
            raise ValueError("gateway_counts")
        if (
            row["eligibleSucceeded"] + row["slowSucceeded"] != row["succeeded"]
            or row["eligibleVideoMs"] > row["completedVideoMs"]
        ):
            raise ValueError("gateway_sla_eligibility")
        if any(
            row[key] is not None and not _integer(row[key]) for key in ("TTVp50Ms", "TTVp95Ms")
        ) or (
            row["TTVp50Ms"] is not None
            and row["TTVp95Ms"] is not None
            and row["TTVp50Ms"] > row["TTVp95Ms"]
        ):
            raise ValueError("gateway_ttv")
        if (
            row["billedUnit"] != "microcredits"
            or (row["computeAmountMicros"] is not None and not _integer(row["computeAmountMicros"]))
            or (row["computeAmountMicros"] is None and row["unknownComputeCount"] == 0)
            or (row["computeAmountMicros"] is not None and row["unknownComputeCount"] != 0)
            or (row["billedAmountMicros"] is not None and not _integer(row["billedAmountMicros"]))
            or (row["billedAmountMicros"] is None and row["unknownBilledCount"] == 0)
            or (row["billedAmountMicros"] is not None and row["unknownBilledCount"] != 0)
        ):
            raise ValueError("gateway_compute_or_billing")
        identity = (
            row["minerHotkey"],
            row["workerId"],
            row["trafficKind"],
            row["profileId"],
            row["profileHash"],
        )
        if identity in seen:
            raise ValueError("gateway_duplicate_aggregate")
        seen.add(identity)
    canonical(body)


def verify_gateway(
    envelope: dict[str, Any], policy: dict[str, Any], now_ms: int | None = None
) -> dict[str, Any]:
    if (
        not isinstance(policy, dict)
        or policy.get("network") != "test"
        or policy.get("netuid") != 578
        or policy.get("mechanism") != 1
        or not isinstance(policy.get("gatewayPolicy"), dict)
        or not isinstance(policy.get("pool"), dict)
    ):
        raise ValueError("gateway_policy_namespace")
    if (
        set(envelope) != {"schema", "body", "digest", "signature"}
        or envelope.get("schema") != SIGNED_SCHEMA
        or not isinstance(envelope.get("body"), dict)
    ):
        raise ValueError("gateway_envelope_schema")
    body = envelope["body"]
    _validate_body(body)
    if envelope.get("digest") != digest(body) or not _ed25519_verify(
        policy["gatewayPublicKey"], GATEWAY_DOMAIN + canonical(body), envelope.get("signature", "")
    ):
        raise ValueError("gateway_signature")
    current = int(time.time() * 1000) if now_ms is None else now_ms
    if (
        body["expiresAtMs"] < current
        or body["policyHash"] != digest(policy["gatewayPolicy"])
        or body["poolHash"] != digest(policy["pool"])
    ):
        raise ValueError("gateway_policy_or_expiry")
    return body


def largest_remainder_weights(totals: dict[str, int]) -> list[list[Any]]:
    """Deterministic positive integer weights totalling exactly 65,535."""
    if not totals:
        return []
    if any(
        not _identifier(hotkey) or not _integer(value) or value == 0
        for hotkey, value in totals.items()
    ):
        raise ValueError("gateway_weight_totals")
    total = sum(totals.values())
    base = {hotkey: value * 65535 // total for hotkey, value in totals.items()}
    remaining = 65535 - sum(base.values())
    ranked = sorted(totals, key=lambda key: (-(totals[key] * 65535 % total), key))
    for hotkey in ranked[:remaining]:
        base[hotkey] += 1
    return [[hotkey, base[hotkey]] for hotkey in sorted(base) if base[hotkey] > 0]


def _eligible_totals(body: dict[str, Any], policy: dict[str, Any]) -> dict[str, int]:
    config = policy["gatewayPolicy"]
    minimum_success = config.get("minSuccessBps")
    maximum_ttv = config.get("maxTtvMs")
    profile = config.get("serviceProfile")
    if (
        not all(_integer(value) for value in (minimum_success, maximum_ttv))
        or minimum_success > 10000
        or not isinstance(profile, dict)
        or set(profile) != {"id", "hash", "durationMs", "resolution"}
        or not _identifier(profile.get("id"))
        or not _hash(profile.get("hash"))
        or profile.get("durationMs") != 5000
        or profile.get("resolution") != "480p"
    ):
        raise ValueError("gateway_policy_thresholds")
    workers = policy.get("workers")
    if not isinstance(workers, list):
        raise ValueError("gateway_policy_workers")
    allowed = {
        (item.get("minerHotkey"), item.get("workerId")): item
        for item in workers
        if isinstance(item, dict)
    }
    if len(allowed) != len(workers):
        raise ValueError("gateway_policy_workers")
    totals: dict[str, int] = {}
    for row in body["aggregates"]:
        binding = allowed.get((row["minerHotkey"], row["workerId"]))
        if binding is None or binding.get("trustMode") != "operator_whitelist":
            raise ValueError("gateway_worker_binding")
        # Trace verification is evidence, never an implicit authorization.
        if (
            row["trustMode"] != "operator_whitelist"
            or row["profileId"] != profile["id"]
            or row["profileHash"] != profile["hash"]
            or row["attempted"] == 0
        ):
            continue
        duration = profile["durationMs"]
        if (
            row["completedVideoMs"] != row["succeeded"] * duration
            or row["eligibleVideoMs"] != row["eligibleSucceeded"] * duration
        ):
            raise ValueError("gateway_profile_duration")
        success = row["eligibleSucceeded"] * 10000 // row["attempted"]
        if success >= minimum_success and row["eligibleVideoMs"]:
            totals[row["minerHotkey"]] = totals.get(row["minerHotkey"], 0) + row["eligibleVideoMs"]
    return totals


def master_window(
    envelope: dict[str, Any],
    policy: dict[str, Any],
    signer: Any,
    finalized_block_hash: str,
    now_ms: int | None = None,
    finalized_block: int = 0,
) -> dict[str, Any]:
    body = verify_gateway(envelope, policy, now_ms)
    finalized_block_hash = canonical_block_hash(finalized_block_hash)
    if (
        signer.validator_hotkey != policy.get("masterPublisherHotkey")
        or policy.get("masterPublisherHotkey") == policy.get("m0PublisherHotkey")
        or not _integer(finalized_block)
    ):
        raise ValueError("m0_master_separation")
    master = {
        "schema": MASTER_SCHEMA,
        "namespace": NAMESPACE,
        "sourceGatewayEnvelopeHash": digest(envelope),
        "policyHash": body["policyHash"],
        "window": body["window"],
        "sequence": body["sequence"],
        "expiresAtMs": body["expiresAtMs"],
        "weights": largest_remainder_weights(_eligible_totals(body, policy)),
        "finalizedRegistryBlockHash": finalized_block_hash,
        "finalizedRegistryBlock": finalized_block,
        "publisherHotkey": signer.validator_hotkey,
    }
    signature = signer.sign(MASTER_DOMAIN + canonical(master)).hex()
    return {
        "schema": MASTER_SCHEMA,
        "body": master,
        "digest": digest(master),
        "signature": signature,
    }


def commitment(master: dict[str, Any]) -> str:
    value = NAMESPACE + ":" + digest(master)
    if len(value.encode()) > 128:
        raise ValueError("commitment_length")
    return value


def verify_master(
    envelope: dict[str, Any], policy: dict[str, Any], now_ms: int | None = None
) -> dict[str, Any]:
    """Verify the signed master body, never an unsigned display vector."""
    if (
        set(envelope) != {"schema", "body", "digest", "signature"}
        or envelope.get("schema") != MASTER_SCHEMA
        or not isinstance(envelope.get("body"), dict)
    ):
        raise ValueError("master_envelope_schema")
    master = envelope["body"]
    fields = {
        "schema",
        "namespace",
        "sourceGatewayEnvelopeHash",
        "policyHash",
        "window",
        "sequence",
        "expiresAtMs",
        "weights",
        "finalizedRegistryBlockHash",
        "finalizedRegistryBlock",
        "publisherHotkey",
    }
    if (
        set(master) != fields
        or master.get("schema") != MASTER_SCHEMA
        or master.get("namespace") != NAMESPACE
        or not all(
            _hash(master.get(key))
            for key in ("sourceGatewayEnvelopeHash", "policyHash", "finalizedRegistryBlockHash")
        )
        or not _integer(master.get("finalizedRegistryBlock"))
        or not _identifier(master.get("publisherHotkey"))
        or not isinstance(master.get("window"), dict)
        or set(master["window"]) != {"startMs", "endMs"}
        or not all(_integer(value) for value in master["window"].values())
        or master["window"]["startMs"] >= master["window"]["endMs"]
        or not _integer(master.get("sequence"))
        or not _integer(master.get("expiresAtMs"))
    ):
        raise ValueError("master_schema")
    weights = master["weights"]
    if (
        not isinstance(weights, list)
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not _identifier(row[0])
            or type(row[1]) is not int
            or not 0 < row[1] <= 65535
            for row in weights
        )
        or weights != sorted(weights, key=lambda row: row[0])
        or len({row[0] for row in weights}) != len(weights)
        or sum(row[1] for row in weights) not in {0, 65535}
    ):
        raise ValueError("master_weights")
    if (
        not isinstance(policy, dict)
        or master["policyHash"] != digest(policy.get("gatewayPolicy", {}))
        or master["publisherHotkey"] != policy.get("masterPublisherHotkey")
        or master["publisherHotkey"] == policy.get("m0PublisherHotkey")
        or envelope.get("digest") != digest(master)
    ):
        raise ValueError("master_policy")
    try:
        signature = bytes.fromhex(envelope.get("signature", ""))
    except ValueError as exc:
        raise ValueError("master_signature") from exc
    if not Sr25519Verifier().verify(
        master["publisherHotkey"], MASTER_DOMAIN + canonical(master), signature
    ):
        raise ValueError("master_signature")
    current = int(time.time() * 1000) if now_ms is None else now_ms
    if not _integer(current) or master["expiresAtMs"] < current:
        raise ValueError("master_expired")
    return master


def light_plan(
    master_envelope: dict[str, Any],
    policy: dict[str, Any],
    anchor: dict[str, Any],
    metagraph: dict[str, Any],
    state: dict[str, Any] | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Return only an offline M1 set-weights plan and the next replay state."""
    master = verify_master(master_envelope, policy, now_ms)
    expected = {
        "network",
        "netuid",
        "mechanism",
        "finalizedRegistryBlock",
        "finalizedRegistryBlockHash",
        "observedBlock",
        "observedBlockHash",
        "commitmentBlock",
        "publisherHotkey",
        "rawCommitment",
        "m0PublisherHotkey",
    }
    if (
        set(anchor) != expected
        or anchor.get("network") != "test"
        or anchor.get("netuid") != 578
        or anchor.get("mechanism") != 1
        or anchor.get("finalizedRegistryBlock") != master["finalizedRegistryBlock"]
        or anchor.get("finalizedRegistryBlockHash") != master["finalizedRegistryBlockHash"]
        or not all(_integer(anchor.get(key)) for key in ("observedBlock", "commitmentBlock"))
        or not _hash(anchor.get("observedBlockHash"))
        or anchor.get("publisherHotkey") != master["publisherHotkey"]
        or anchor.get("rawCommitment") != commitment(master_envelope)
        or anchor.get("m0PublisherHotkey") != policy.get("m0PublisherHotkey")
        or anchor["publisherHotkey"] == anchor["m0PublisherHotkey"]
    ):
        raise ValueError("light_anchor")
    if (
        not isinstance(metagraph, dict)
        or metagraph.get("network") != "test"
        or metagraph.get("netuid") != 578
        or metagraph.get("mechanism") != 1
        or metagraph.get("observedBlock") != anchor["observedBlock"]
        or metagraph.get("observedBlockHash") != anchor["observedBlockHash"]
        or not isinstance(metagraph.get("uids"), dict)
    ):
        raise ValueError("light_metagraph")
    old = state or {}
    if not isinstance(old, dict):
        raise ValueError("light_state")
    previous = old.get("sequence", -1)
    if (
        type(previous) is not int
        or master["sequence"] < previous
        or (master["sequence"] == previous and old.get("masterDigest") != master_envelope["digest"])
    ):
        raise ValueError("light_replay_or_conflict")
    pairs = []
    for hotkey, weight in master["weights"]:
        uid = metagraph["uids"].get(hotkey)
        if type(uid) is not int or uid < 0:
            raise ValueError("light_unregistered_or_remapped")
        pairs.append([uid, weight])
    next_state = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "sequence": master["sequence"],
        "masterDigest": master_envelope["digest"],
    }
    return {
        **next_state,
        "finalizedRegistryBlock": master["finalizedRegistryBlock"],
        "finalizedRegistryBlockHash": master["finalizedRegistryBlockHash"],
        "observedBlock": anchor["observedBlock"],
        "observedBlockHash": anchor["observedBlockHash"],
        "uids": [pair[0] for pair in pairs],
        "weights": [pair[1] for pair in pairs],
        "nextState": next_state,
    }
