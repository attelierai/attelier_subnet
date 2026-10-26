"""Offline-only simulation for Attelier SN126 inference service rewards.

This module deliberately has no wallet, signer, chain-client or publication
dependency. Its output is a review artifact, never a chain instruction.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "attelier/prelaunch-simulation/v1"
LEDGER_SCHEMA = "attelier/prelaunch-ledger/v1"
POLICY_SCHEMA = "attelier/prelaunch-policy/v1"
REGISTRY_SCHEMA = "attelier/prelaunch-registry/v1"
GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
DAY_MS = 86_400_000
WEIGHT_TOTAL = 65_535
MAX_SAFE_INTEGER = 2**53 - 1
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[1-9A-HJ-NP-Za-km-z]{47,48}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CLIENT_ID = re.compile(r"[A-Za-z0-9:_-]{1,100}\Z")


def _fail(reason: str) -> None:
    raise ValueError(reason)


def _integer(value: Any, reason: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        _fail(reason)
    return value


def _hash(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        _fail(reason)
    return value


def _hotkey(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not _HOTKEY.fullmatch(value):
        _fail(reason)
    return value


def _identifier(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        _fail(reason)
    return value


def canonical(value: Any, *, allow_numeric_keys: bool = False) -> bytes:
    """Canonical JSON; numeric object keys are only allowed for UID snapshots."""

    def check(item: Any, depth: int = 0) -> None:
        if depth > 24:
            _fail("json_too_deep")
        if item is None or isinstance(item, bool):
            return
        if type(item) is int:
            if abs(item) > MAX_SAFE_INTEGER:
                _fail("json_integer_range")
            return
        if isinstance(item, str):
            if not item.isascii():
                _fail("json_non_ascii")
            return
        if isinstance(item, list):
            if len(item) > 4096:
                _fail("json_array_size")
            for child in item:
                check(child, depth + 1)
            return
        if isinstance(item, dict):
            if len(item) > 1024:
                _fail("json_object_size")
            for key, child in item.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or not key.isascii()
                    or (key.isdecimal() and not allow_numeric_keys)
                ):
                    _fail("json_key")
                check(child, depth + 1)
            return
        _fail("json_type")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path, *, allow_numeric_keys: bool = False) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail("duplicate_json_key")
            value[key] = item
        return value

    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            _fail("json_too_large")
        value = json.loads(
            raw, object_pairs_hook=unique, parse_constant=lambda _: _fail("json_nonfinite")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("json_unavailable_or_invalid") from error
    if not isinstance(value, dict):
        _fail("json_object_required")
    canonical(value, allow_numeric_keys=allow_numeric_keys)
    return value


def _namespace(value: dict[str, Any], schema: str, reason: str) -> None:
    if (
        value.get("schema") != schema
        or value.get("network") != "finney"
        or value.get("genesis") != GENESIS
        or type(value.get("netuid")) is not int
        or value["netuid"] != 126
        or type(value.get("mechanism")) is not int
        or value["mechanism"] != 0
    ):
        _fail(reason)


def _largest_remainder(total: int, portions: dict[str, int]) -> dict[str, int]:
    if total < 0 or not portions or any(value <= 0 for value in portions.values()):
        _fail("allocation_input")
    denominator = sum(portions.values())
    result = {key: total * value // denominator for key, value in portions.items()}
    remainder = total - sum(result.values())
    ranking = sorted(portions, key=lambda key: (-(total * portions[key] % denominator), key))
    for key in ranking[:remainder]:
        result[key] += 1
    return result


def _parse_decimal_rao(value: Any) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{0,30}", value):
        _fail("nominal_miner_emission_rao")
    return int(value)


def _safe_derived(value: int, reason: str) -> int:
    if not 0 <= value <= MAX_SAFE_INTEGER:
        _fail(reason)
    return value


def _validate_policy(
    policy: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    _namespace(policy, POLICY_SCHEMA, "policy_namespace")
    allowed = {
        "schema",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "gatewayPolicy",
        "workers",
        "simulation",
    }
    if (
        set(policy) != allowed
        or not isinstance(policy["gatewayPolicy"], dict)
        or not isinstance(policy["workers"], list)
        or not isinstance(policy["simulation"], dict)
    ):
        _fail("policy_schema")
    gateway = policy["gatewayPolicy"]
    if set(gateway) != {"minSuccessBps", "maxTtvMs", "serviceProfile"} or not isinstance(
        gateway["serviceProfile"], dict
    ):
        _fail("policy_gateway")
    minimum = _integer(gateway["minSuccessBps"], "policy_success")
    maximum_ttv = _integer(gateway["maxTtvMs"], "policy_ttv")
    profile = gateway["serviceProfile"]
    if (
        minimum > 10_000
        or maximum_ttv == 0
        or set(profile) != {"id", "hash", "durationMs", "resolution"}
        or _identifier(profile["id"], "policy_profile_id") != profile["id"]
        or _hash(profile["hash"], "policy_profile_hash") != profile["hash"]
        or profile["durationMs"] != 5_000
        or profile["resolution"] != "480p"
    ):
        _fail("policy_profile")
    bindings: dict[tuple[str, str], dict[str, Any]] = {}
    for worker in policy["workers"]:
        if (
            not isinstance(worker, dict)
            or set(worker) != {"minerHotkey", "workerId", "trustMode"}
            or worker["trustMode"] != "operator_whitelist"
        ):
            _fail("policy_worker")
        key = (
            _hotkey(worker["minerHotkey"], "policy_worker_hotkey"),
            _identifier(worker["workerId"], "policy_worker_id"),
        )
        if key in bindings:
            _fail("policy_worker_duplicate")
        bindings[key] = worker
    simulation = policy["simulation"]
    if set(simulation) != {
        "serviceBudgetBps",
        "targetVideosPerDay",
        "nominalMinerEmissionRaoPerDay",
    }:
        _fail("policy_simulation")
    if (
        not 0 < _integer(simulation["serviceBudgetBps"], "policy_budget") <= 10_000
        or _integer(simulation["targetVideosPerDay"], "policy_target") == 0
    ):
        _fail("policy_simulation")
    _parse_decimal_rao(simulation["nominalMinerEmissionRaoPerDay"])
    return bindings, gateway


def _validate_registry(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _namespace(registry, REGISTRY_SCHEMA, "registry_namespace")
    required = {
        "schema",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "finalizedBlock",
        "finalizedBlockHash",
        "subnetOwnerColdkey",
        "miners",
    }
    if (
        set(registry) != required
        or not isinstance(registry["miners"], list)
    ):
        _fail("registry_schema")
    if (
        _integer(registry["finalizedBlock"], "registry_block") == 0
        or _hash(
            registry["finalizedBlockHash"].removeprefix("0x")
            if isinstance(registry["finalizedBlockHash"], str)
            else registry["finalizedBlockHash"],
            "registry_block_hash",
        )
        is None
    ):
        _fail("registry_block")
    owner = _hotkey(registry["subnetOwnerColdkey"], "registry_owner")
    miners: dict[str, dict[str, Any]] = {}
    uids: set[int] = set()
    for miner in registry["miners"]:
        if not isinstance(miner, dict) or set(miner) != {"uid", "hotkey", "coldkey"}:
            _fail("registry_miner")
        uid = _integer(miner["uid"], "registry_uid")
        if uid > 65_535:
            _fail("registry_uid")
        hotkey = _hotkey(miner["hotkey"], "registry_hotkey")
        coldkey = _hotkey(miner["coldkey"], "registry_coldkey")
        if hotkey in miners or uid in uids:
            _fail("registry_duplicate")
        miners[hotkey] = {
            "uid": uid,
            "hotkey": hotkey,
            "coldkey": coldkey,
            "ownerAssociated": coldkey == owner,
        }
        uids.add(uid)
    return miners


def _validate_ledger(
    ledger: dict[str, Any],
    policy: dict[str, Any],
    bindings: dict[tuple[str, str], dict[str, Any]],
    gateway: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, int]:
    _namespace(ledger, LEDGER_SCHEMA, "ledger_namespace")
    required = {
        "schema",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "window",
        "sequence",
        "closedAtMs",
        "aggregates",
        "gatewayPolicyHash",
        "workerPoolHash",
        "policyHash",
    }
    if (
        set(ledger) != required
        or not isinstance(ledger["window"], dict)
        or not isinstance(ledger["aggregates"], list)
    ):
        _fail("ledger_schema")
    if (
        _hash(ledger["policyHash"], "ledger_policy_hash") != digest(policy)
        or _hash(ledger["gatewayPolicyHash"], "ledger_gateway_policy_hash") != digest(gateway)
        or _hash(ledger["workerPoolHash"], "ledger_worker_pool_hash")
        != digest(list(bindings.values()))
    ):
        _fail("ledger_service_policy_hash")
    window = ledger["window"]
    if set(window) != {"startMs", "endMs"}:
        _fail("ledger_window")
    start = _integer(window["startMs"], "ledger_start")
    end = _integer(window["endMs"], "ledger_end")
    if (
        end - start != DAY_MS
        or start % DAY_MS != 0
        or _integer(ledger["sequence"], "ledger_sequence") == 0
        or _integer(ledger["closedAtMs"], "ledger_closed") < end
    ):
        _fail("ledger_window")
    profile = gateway["serviceProfile"]
    expected_fields = {
        "minerHotkey",
        "workerId",
        "trustMode",
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
    }
    totals: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for row in ledger["aggregates"]:
        if not isinstance(row, dict) or set(row) != expected_fields:
            _fail("ledger_aggregate_schema")
        hotkey = _hotkey(row["minerHotkey"], "ledger_hotkey")
        worker = _identifier(row["workerId"], "ledger_worker")
        if (hotkey, worker) in seen or (hotkey, worker) not in bindings or hotkey not in registry:
            _fail("ledger_binding")
        seen.add((hotkey, worker))
        if (
            row["trustMode"] != "operator_whitelist"
            or row["profileId"] != profile["id"]
            or row["profileHash"] != profile["hash"]
        ):
            _fail("ledger_profile_or_trust")
        attempted = _integer(row["attempted"], "ledger_counts")
        succeeded = _integer(row["succeeded"], "ledger_counts")
        failed = _integer(row["failed"], "ledger_counts")
        cancelled = _integer(row["cancelled"], "ledger_counts")
        inconclusive = _integer(row["inconclusive"], "ledger_counts")
        eligible = _integer(row["eligibleSucceeded"], "ledger_counts")
        slow = _integer(row["slowSucceeded"], "ledger_counts")
        completed_ms = _integer(row["completedVideoMs"], "ledger_counts")
        eligible_ms = _integer(row["eligibleVideoMs"], "ledger_counts")
        if (
            attempted != succeeded + failed + cancelled + inconclusive
            or eligible + slow != succeeded
            or completed_ms != succeeded * profile["durationMs"]
            or eligible_ms != eligible * profile["durationMs"]
        ):
            _fail("ledger_counts")
        if attempted and eligible * 10_000 // attempted >= gateway["minSuccessBps"] and eligible_ms:
            totals[hotkey] = _safe_derived(
                totals.get(hotkey, 0) + eligible, "derived_eligible_videos"
            )
    return totals


def simulate(
    ledger: dict[str, Any], policy: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    """Return one deterministic, unsigned prelaunch review artifact."""
    bindings, gateway = _validate_policy(policy)
    miners = _validate_registry(registry)
    eligible = _validate_ledger(ledger, policy, bindings, gateway, miners)
    simulation = policy["simulation"]
    total_videos = _safe_derived(sum(eligible.values()), "derived_eligible_videos")
    budget_bps = simulation["serviceBudgetBps"]
    target = simulation["targetVideosPerDay"]
    covered = min(total_videos, target)
    effective_budget_numerator = _safe_derived(budget_bps * covered, "derived_budget")
    service_weight = WEIGHT_TOTAL * budget_bps * covered // (10_000 * target)
    miner_weights = _largest_remainder(service_weight, eligible) if service_weight else {}
    burn_weight = WEIGHT_TOTAL - service_weight
    nominal_rao = _parse_decimal_rao(simulation["nominalMinerEmissionRaoPerDay"])
    service_rao = nominal_rao * budget_bps * covered // (10_000 * target)
    counterfactual_rao = _largest_remainder(service_rao, eligible) if service_rao else {}
    rows = []
    owner_redirected = 0
    expected_total = 0
    for hotkey in sorted(eligible):
        miner = miners[hotkey]
        allocation = counterfactual_rao.get(hotkey, 0)
        quantized = nominal_rao * miner_weights.get(hotkey, 0) // WEIGHT_TOTAL
        expected = 0 if miner["ownerAssociated"] else quantized
        owner_redirected += quantized - expected
        expected_total += expected
        rows.append(
            {
                "uid": miner["uid"],
                "hotkey": hotkey,
                "eligibleVideos": eligible[hotkey],
                "proposedWeight": miner_weights.get(hotkey, 0),
                "counterfactualServiceAllocationRao": str(allocation),
                "chainExpectedMinerPayoutRao": str(expected),
                "ownerAssociated": miner["ownerAssociated"],
                "requiresExternalConsensus": True,
            }
        )
    vector = [
        {
            "kind": "miner",
            "uid": row["uid"],
            "hotkey": row["hotkey"],
            "weight": row["proposedWeight"],
        }
        for row in rows
        if row["proposedWeight"]
    ]
    vector.append({"kind": "burn", "weight": burn_weight})
    return {
        "schema": SCHEMA,
        "simulationOnly": True,
        "publishable": False,
        "chainCalls": [],
        "namespace": {"network": "finney", "genesis": GENESIS, "netuid": 126, "mechanism": 0},
        "asset": {"kind": "alpha", "netuid": 126, "decimals": 9},
        "input": {
            "genesis": GENESIS,
            "ledgerSha256": digest(ledger),
            "policySha256": ledger["policyHash"],
            "registrySha256": digest(registry),
            "gatewayPolicyHash": ledger["gatewayPolicyHash"],
            "workerPoolHash": ledger["workerPoolHash"],
            "observedBlock": registry["finalizedBlock"],
            "observedBlockHash": registry["finalizedBlockHash"],
            "finalizedRegistryBlock": registry["finalizedBlock"],
            "finalizedRegistryBlockHash": registry["finalizedBlockHash"],
        },
        "createdAtMs": ledger["closedAtMs"],
        "window": ledger["window"],
        "service": {
            "eligibleVideos": total_videos,
            "targetVideosPerDay": target,
            "configuredServiceBudgetBps": budget_bps,
            "effectiveServiceBudgetNumerator": effective_budget_numerator,
            "effectiveServiceBudgetDenominator": target,
        },
        "unsignedProposedWeights": {"total": WEIGHT_TOTAL, "entries": vector},
        "counterfactualServiceAllocation": {
            "nominalMinerEmissionRaoPerDay": str(nominal_rao),
            "allocatedRao": str(service_rao),
            "requiresExternalConsensus": True,
        },
        "ownerAdjustedExpectedPayout": {
            "totalRao": str(expected_total),
            "redirectedToSimulatedBurnRao": str(owner_redirected),
            "reason": "owner-associated miner allocations are not payable",
        },
        "miners": rows,
        "assumptions": {
            "dailyUtcWindow": True,
            "noYumaConsensusSimulation": True,
            "noTransactionOrSignature": True,
        },
    }


def _timestamp(value: Any, reason: str) -> tuple[str, int]:
    if not isinstance(value, str):
        _fail(reason)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(reason) from error
    if parsed.tzinfo is None:
        _fail(reason)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z"), int(parsed.timestamp())


def _validate_prelaunch_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        "schema",
        "mode",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "trustMode",
        "allowedClients",
        "syntheticClientIds",
        "registrationFile",
        "miners",
        "serviceProfile",
    }
    if set(config) != required or config.get("schema") != "attelier/subnet-h3/v1":
        _fail("prelaunch_config_schema")
    _namespace(config, "attelier/subnet-h3/v1", "prelaunch_config_namespace")
    if (
        config["mode"] != "prelaunch"
        or config["trustMode"] != "operator_whitelist"
        or not isinstance(config["registrationFile"], str)
        or not isinstance(config["miners"], list)
        or not isinstance(config["allowedClients"], list)
        or not isinstance(config["syntheticClientIds"], list)
        or not isinstance(config["serviceProfile"], dict)
    ):
        _fail("prelaunch_config")
    for client in [*config["allowedClients"], *config["syntheticClientIds"]]:
        if not isinstance(client, str) or client == "*" or not _CLIENT_ID.fullmatch(client):
            _fail("prelaunch_config_clients")
    profile = config["serviceProfile"]
    if set(profile) != {"id", "hash"}:
        _fail("prelaunch_config_profile")
    _identifier(profile["id"], "prelaunch_config_profile")
    _hash(profile["hash"], "prelaunch_config_profile")
    miners: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for miner in config["miners"]:
        if not isinstance(miner, dict):
            _fail("prelaunch_config_miner")
        for key in (
            "id",
            "uid",
            "hotkey",
            "workerId",
            "operatorEnabled",
            "operatorReviewedExpires",
        ):
            if key not in miner:
                _fail("prelaunch_config_miner")
        identifier = _identifier(miner["id"], "prelaunch_config_miner")
        uid = _integer(miner["uid"], "prelaunch_config_miner")
        if uid > 65_535 or identifier in identifiers or type(miner["operatorEnabled"]) is not bool:
            _fail("prelaunch_config_miner")
        _hotkey(miner["hotkey"], "prelaunch_config_miner")
        _identifier(miner["workerId"], "prelaunch_config_miner")
        reviewed = _integer(miner["operatorReviewedExpires"], "prelaunch_config_review")
        identifiers.add(identifier)
        miners.append(
            {
                "id": identifier,
                "uid": uid,
                "hotkey": miner["hotkey"],
                "workerId": miner["workerId"],
                "operatorEnabled": miner["operatorEnabled"],
                "operatorReviewedExpires": reviewed,
            }
        )
    return miners


def _validate_registry_snapshot(
    snapshot: dict[str, Any],
) -> tuple[dict[int, tuple[str, str]], str, dict[str, Any]]:
    required = {
        "schema",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "finalized",
        "mechanismCount",
        "block",
        "blockHash",
        "timestamp",
        "subnetOwnerColdkey",
        "owners",
    }
    if (
        set(snapshot) != required
        or snapshot.get("schema") != "forge/inference-registry-snapshot/v1"
    ):
        _fail("registry_snapshot_schema")
    _namespace(snapshot, "forge/inference-registry-snapshot/v1", "registry_snapshot_namespace")
    if (
        snapshot["finalized"] is not True
        or _integer(snapshot["mechanismCount"], "registry_snapshot_mechanism") != 1
    ):
        _fail("registry_snapshot_finality_or_mechanism")
    block = _integer(snapshot["block"], "registry_snapshot_block")
    if block == 0:
        _fail("registry_snapshot_block")
    block_hash = snapshot["blockHash"]
    if not isinstance(block_hash, str) or not block_hash.startswith("0x"):
        _fail("registry_snapshot_block_hash")
    _hash(block_hash[2:], "registry_snapshot_block_hash")
    generated_at, reviewed_at = _timestamp(snapshot["timestamp"], "registry_snapshot_timestamp")
    owner = _hotkey(snapshot["subnetOwnerColdkey"], "registry_snapshot_owner")
    if not isinstance(snapshot["owners"], dict):
        _fail("registry_snapshot_owners")
    owners: dict[int, tuple[str, str]] = {}
    hotkeys: set[str] = set()
    for raw_uid, record in snapshot["owners"].items():
        if not isinstance(raw_uid, str) or not raw_uid.isdecimal():
            _fail("registry_snapshot_uid")
        uid = _integer(int(raw_uid), "registry_snapshot_uid")
        if (
            uid > 65_535
            or uid in owners
            or not isinstance(record, dict)
            or set(record) != {"hotkey", "coldkey"}
        ):
            _fail("registry_snapshot_uid")
        hotkey = _hotkey(record["hotkey"], "registry_snapshot_hotkey")
        coldkey = _hotkey(record["coldkey"], "registry_snapshot_coldkey")
        if hotkey in hotkeys:
            _fail("registry_snapshot_duplicate")
        owners[uid] = (hotkey, coldkey)
        hotkeys.add(hotkey)
    return (
        owners,
        owner,
        {
            "block": block,
            "blockHash": block_hash,
            "generatedAt": generated_at,
            "reviewedAt": reviewed_at,
        },
    )


def project_prelaunch_registry(
    config: dict[str, Any], snapshot: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project an already-finalized public snapshot; no network or signing occurs here."""
    miners = _validate_prelaunch_config(config)
    owners, subnet_owner, observed = _validate_registry_snapshot(snapshot)
    workers: dict[str, dict[str, Any]] = {}
    for miner in miners:
        registered = owners.get(miner["uid"])
        if not miner["operatorEnabled"]:
            continue
        if miner["operatorReviewedExpires"] <= observed["reviewedAt"]:
            _fail("operator_review_expired")
        if registered is None or registered[0] != miner["hotkey"]:
            _fail("configured_worker_not_registered")
        workers[miner["id"]] = {
            "workerId": miner["workerId"],
            "hotkey": miner["hotkey"],
            "operatorReviewedExpires": miner["operatorReviewedExpires"],
        }
    uids = {str(uid): hotkey for uid, (hotkey, _) in sorted(owners.items())}
    registry = {
        "schema": "forge/inference-registry/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "mechanismCount": 1,
        "block": observed["block"],
        "blockHash": observed["blockHash"],
        "generatedAt": observed["generatedAt"],
        "uids": uids,
        "workers": workers,
    }
    simulation = {
        "schema": REGISTRY_SCHEMA,
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "finalizedBlock": observed["block"],
        "finalizedBlockHash": observed["blockHash"],
        "subnetOwnerColdkey": subnet_owner,
        "miners": [
            {"uid": miner["uid"], "hotkey": miner["hotkey"], "coldkey": owners[miner["uid"]][1]}
            for miner in miners
            if miner["operatorEnabled"]
        ],
    }
    return registry, simulation
