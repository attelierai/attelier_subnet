"""Offline-only simulation for Attelier SN126 inference service rewards.

This module deliberately has no wallet, signer, chain-client or publication
dependency. Its output is a review artifact, never a chain instruction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "attelier/prelaunch-simulation/v1"
LEDGER_SCHEMA = "attelier/prelaunch-ledger/v1"
POLICY_SCHEMA = "attelier/prelaunch-policy/v1"
LEDGER_SCHEMA_V2 = "attelier/prelaunch-ledger/v2"
POLICY_SCHEMA_V2 = "attelier/prelaunch-policy/v2"
REGISTRY_SCHEMA = "attelier/prelaunch-registry/v1"
GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
WEIGHT_TOTAL = 65_535
MAX_SAFE_INTEGER = 2**53 - 1
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[1-9A-HJ-NP-Za-km-z]{47,48}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CLIENT_ID = re.compile(r"[A-Za-z0-9:_-]{1,100}\Z")
_H3_CATALOG_PATH = Path(__file__).with_name("h3-checkpoint-catalog.json")


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


def _safe_fraction(numerator: int, denominator: int, reason: str) -> tuple[int, int]:
    """Reduce an exact ratio before exposing it as safe JSON integers."""
    if numerator < 0 or denominator <= 0:
        _fail(reason)
    divisor = math.gcd(numerator, denominator)
    return (
        _safe_derived(numerator // divisor, reason),
        _safe_derived(denominator // divisor, reason),
    )


def _validate_policy(
    policy: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], int]:
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
    allowed_with_window = allowed | {"windowMs"}
    if (
        (set(policy) != allowed and set(policy) != allowed_with_window)
        or not isinstance(policy["gatewayPolicy"], dict)
        or not isinstance(policy["workers"], list)
        or not isinstance(policy["simulation"], dict)
    ):
        _fail("policy_schema")
    window_ms = policy.get("windowMs", DAY_MS)
    if type(window_ms) is not int or window_ms not in {HOUR_MS, DAY_MS}:
        _fail("policy_window")
    gateway = policy["gatewayPolicy"]
    gateway_fields = {"minSuccessBps", "maxTtvMs", "serviceProfile"}
    if (
        (set(gateway) != gateway_fields and set(gateway) != gateway_fields | {"rewardEligibility"})
        or not isinstance(gateway["serviceProfile"], dict)
        or ("rewardEligibility" in gateway and gateway["rewardEligibility"] != "per_job")
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
    simulation_fields = {
        "serviceBudgetBps",
        "targetVideosPerDay",
        "nominalMinerEmissionRaoPerDay",
    }
    if set(simulation) not in (simulation_fields, simulation_fields | {"allocationMode"}):
        _fail("policy_simulation")
    if (
        not 0 < _integer(simulation["serviceBudgetBps"], "policy_budget") <= 10_000
        or _integer(simulation["targetVideosPerDay"], "policy_target") == 0
        or simulation.get("allocationMode", "target") not in {"fixed", "target"}
    ):
        _fail("policy_simulation")
    _parse_decimal_rao(simulation["nominalMinerEmissionRaoPerDay"])
    return bindings, gateway, window_ms


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
    if set(registry) != required or not isinstance(registry["miners"], list):
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
    window_ms: int,
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
        end - start != window_ms
        or start % window_ms != 0
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
        reward_eligible = (
            gateway.get("rewardEligibility") == "per_job"
            or attempted > 0
            and eligible * 10_000 // attempted >= gateway["minSuccessBps"]
        )
        if reward_eligible and eligible_ms:
            totals[hotkey] = _safe_derived(
                totals.get(hotkey, 0) + eligible, "derived_eligible_videos"
            )
    return totals


def _simulate_v1(
    ledger: dict[str, Any], policy: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    """Return one deterministic, unsigned prelaunch review artifact."""
    bindings, gateway, window_ms = _validate_policy(policy)
    miners = _validate_registry(registry)
    eligible = _validate_ledger(ledger, policy, bindings, gateway, miners, window_ms)
    simulation = policy["simulation"]
    total_videos = _safe_derived(sum(eligible.values()), "derived_eligible_videos")
    budget_bps = simulation["serviceBudgetBps"]
    target = simulation["targetVideosPerDay"]
    window_target_numerator = _safe_derived(target * window_ms, "derived_window_target")
    coverage_numerator = min(
        _safe_derived(total_videos * DAY_MS, "derived_coverage"), window_target_numerator
    )
    if simulation.get("allocationMode", "target") == "fixed":
        coverage_numerator = window_target_numerator if total_videos else 0
    coverage_denominator = window_target_numerator
    effective_budget_numerator, effective_budget_denominator = _safe_fraction(
        budget_bps * coverage_numerator, coverage_denominator, "derived_budget"
    )
    service_weight = (
        WEIGHT_TOTAL * budget_bps * coverage_numerator // (10_000 * coverage_denominator)
    )
    miner_weights = _largest_remainder(service_weight, eligible) if service_weight else {}
    burn_weight = WEIGHT_TOTAL - service_weight
    nominal_rao = _parse_decimal_rao(simulation["nominalMinerEmissionRaoPerDay"])
    service_rao = nominal_rao * budget_bps * coverage_numerator // (10_000 * target * DAY_MS)
    counterfactual_rao = _largest_remainder(service_rao, eligible) if service_rao else {}
    rows = []
    owner_redirected = 0
    expected_total = 0
    for hotkey in sorted(eligible):
        miner = miners[hotkey]
        allocation = counterfactual_rao.get(hotkey, 0)
        quantized = (
            nominal_rao * window_ms * miner_weights.get(hotkey, 0) // (DAY_MS * WEIGHT_TOTAL)
        )
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
            **(
                {"allocationMode": simulation["allocationMode"]}
                if "allocationMode" in simulation
                else {}
            ),
            "eligibleVideos": total_videos,
            "targetVideosPerDay": target,
            "windowTargetVideos": {
                "numerator": window_target_numerator,
                "denominator": DAY_MS,
            },
            "windowMs": window_ms,
            "configuredServiceBudgetBps": budget_bps,
            "effectiveServiceBudgetNumerator": effective_budget_numerator,
            "effectiveServiceBudgetDenominator": effective_budget_denominator,
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
            "dailyUtcWindow": window_ms == DAY_MS,
            "windowMs": window_ms,
            "noYumaConsensusSimulation": True,
            "noTransactionOrSignature": True,
        },
    }


def _validate_v2(
    ledger: dict[str, Any], policy: dict[str, Any], registry: dict[str, Any]
) -> tuple[dict[str, int], dict[str, int], int]:
    """Validate the separate multi-profile ledger and return video/work units.

    V2 reward merit is frozen policy work, never elapsed miner-reported time.
    """
    _namespace(policy, POLICY_SCHEMA_V2, "policy_namespace")
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
        set(policy) != allowed | {"windowMs"}
        or not isinstance(policy.get("gatewayPolicy"), dict)
        or not isinstance(policy.get("workers"), list)
        or not isinstance(policy.get("simulation"), dict)
    ):
        _fail("policy_schema")
    window_ms = policy.get("windowMs", DAY_MS)
    if type(window_ms) is not int or window_ms not in {HOUR_MS, DAY_MS}:
        _fail("policy_window")
    gateway = policy["gatewayPolicy"]
    if (
        set(gateway) != {"minSuccessBps", "rewardEligibility", "serviceProfiles"}
        or gateway["rewardEligibility"] != "per_job"
        or gateway["minSuccessBps"] != 0
        or window_ms != HOUR_MS
        or not isinstance(gateway["serviceProfiles"], list)
        or not 1 <= len(gateway["serviceProfiles"]) <= 18
    ):
        _fail("policy_gateway")
    profiles: dict[str, dict[str, Any]] = {}
    hashes: set[str] = set()
    for profile in gateway["serviceProfiles"]:
        approved_identity = (
            _approved_h3_profile_identity(profile) if isinstance(profile, dict) else None
        )
        if (
            not isinstance(profile, dict)
            or set(profile)
            != {
                "id",
                "hash",
                "durationMs",
                "resolution",
                "aspectRatio",
                "maxJobSeconds",
                "maxTtvMs",
                "rewardWorkUnits",
                "rewardBasis",
                "approvalHash",
            }
            or _identifier(profile.get("id"), "policy_profile_id") in profiles
            or _hash(profile.get("hash"), "policy_profile_hash") in hashes
            or approved_identity is None
            or profile.get("rewardBasis") not in {"reference_gpu", "pixel_frames"}
            or _hash(profile.get("approvalHash"), "policy_approval_hash") is None
            or profile.get("durationMs") not in {5_000, 10_000, 15_000}
            or profile.get("resolution") not in {"480p", "768p", "1080p"}
            or profile.get("aspectRatio") not in {"16:9", "9:16"}
            or not 1 <= _integer(profile.get("maxJobSeconds"), "policy_job") <= 7_200
            or not 1 <= _integer(profile.get("maxTtvMs"), "policy_ttv") <= 7_200_000
            or not 1 <= _integer(profile.get("rewardWorkUnits"), "policy_units") <= 1_000_000_000
            or (
                profile.get("rewardBasis") == "pixel_frames"
                and profile.get("rewardWorkUnits")
                != round(
                    1000
                    * (profile["durationMs"] // 1000 * 24)
                    * {"480p": 832 * 480, "768p": 1344 * 768, "1080p": 1920 * 1080}[
                        profile["resolution"]
                    ]
                    / (5 * 24 * 832 * 480)
                )
            )
        ):
            _fail("policy_profile")
        profiles[profile["id"]] = profile
        hashes.add(profile["hash"])
    bindings: dict[tuple[str, str], dict[str, Any]] = {}
    for worker in policy["workers"]:
        if (
            not isinstance(worker, dict)
            or set(worker) != {"minerHotkey", "workerId", "trustMode", "profileIds"}
            or worker.get("trustMode") != "operator_whitelist"
            or not isinstance(worker.get("profileIds"), list)
            or not worker["profileIds"]
            or len(set(worker["profileIds"])) != len(worker["profileIds"])
            or any(not isinstance(i, str) or i not in profiles for i in worker["profileIds"])
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
    if (
        set(simulation)
        != {
            "serviceBudgetBps",
            "targetVideosPerDay",
            "nominalMinerEmissionRaoPerDay",
            "allocationMode",
        }
        or simulation["serviceBudgetBps"] != 500
        or _integer(simulation["targetVideosPerDay"], "policy_target") == 0
        or simulation["allocationMode"] != "fixed"
    ):
        _fail("policy_simulation")
    _parse_decimal_rao(simulation["nominalMinerEmissionRaoPerDay"])
    miners = _validate_registry(registry)
    _namespace(ledger, LEDGER_SCHEMA_V2, "ledger_namespace")
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
        or not isinstance(ledger.get("window"), dict)
        or not isinstance(ledger.get("aggregates"), list)
        or ledger.get("policyHash") != digest(policy)
        or ledger.get("gatewayPolicyHash") != digest(gateway)
        or ledger.get("workerPoolHash") != digest(policy["workers"])
    ):
        _fail("ledger_schema")
    window = ledger["window"]
    if (
        set(window) != {"startMs", "endMs"}
        or _integer(window["endMs"], "ledger_end") - _integer(window["startMs"], "ledger_start")
        != window_ms
        or window["startMs"] % window_ms
        or _integer(ledger["sequence"], "ledger_sequence") == 0
        or _integer(ledger["closedAtMs"], "ledger_closed") < window["endMs"]
    ):
        _fail("ledger_window")
    videos: dict[str, int] = {}
    units: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()
    names = {
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
        "eligibleWorkUnits",
    }
    for row in ledger["aggregates"]:
        if not isinstance(row, dict) or set(row) != names:
            _fail("ledger_aggregate_schema")
        hotkey = _hotkey(row["minerHotkey"], "ledger_hotkey")
        worker = _identifier(row["workerId"], "ledger_worker")
        profile = profiles.get(row.get("profileId"))
        binding = bindings.get((hotkey, worker))
        if (
            not profile
            or not binding
            or row.get("profileHash") != profile["hash"]
            or row.get("trustMode") != "operator_whitelist"
            or row["profileId"] not in binding["profileIds"]
            or (hotkey, worker, row["profileId"]) in seen
            or hotkey not in miners
        ):
            _fail("ledger_binding")
        seen.add((hotkey, worker, row["profileId"]))
        (
            attempted,
            succeeded,
            failed,
            cancelled,
            inconclusive,
            eligible,
            slow,
            completed,
            eligible_ms,
            eligible_units,
        ) = (
            _integer(row[k], "ledger_counts")
            for k in (
                "attempted",
                "succeeded",
                "failed",
                "cancelled",
                "inconclusive",
                "eligibleSucceeded",
                "slowSucceeded",
                "completedVideoMs",
                "eligibleVideoMs",
                "eligibleWorkUnits",
            )
        )
        if (
            attempted != succeeded + failed + cancelled + inconclusive
            or eligible + slow != succeeded
            or completed != succeeded * profile["durationMs"]
            or eligible_ms != eligible * profile["durationMs"]
            or eligible_units != eligible * profile["rewardWorkUnits"]
        ):
            _fail("ledger_counts")
        if eligible:
            videos[hotkey] = _safe_derived(
                videos.get(hotkey, 0) + eligible, "derived_eligible_videos"
            )
            units[hotkey] = _safe_derived(
                units.get(hotkey, 0) + eligible_units, "derived_eligible_units"
            )
    return videos, units, window_ms


def _simulate_v2(
    ledger: dict[str, Any], policy: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    videos, eligible, window_ms = _validate_v2(ledger, policy, registry)
    # Reuse the v1 artifact math after replacing only its merit source.  The
    # public artifact preserves counts and makes the unit source explicit.
    legacy_policy = {
        **policy,
        "schema": POLICY_SCHEMA,
        "gatewayPolicy": {
            "minSuccessBps": 0,
            "maxTtvMs": 1,
            "rewardEligibility": "per_job",
            "serviceProfile": {
                "id": "legacy",
                "hash": "0" * 64,
                "durationMs": 5_000,
                "resolution": "480p",
            },
        },
        "workers": [
            {
                "minerHotkey": w["minerHotkey"],
                "workerId": w["workerId"],
                "trustMode": "operator_whitelist",
            }
            for w in policy["workers"]
        ],
    }
    # Build a minimal compatible ledger whose eligible count is the frozen work
    # unit. This changes allocation merit only; v2's original receipt remains
    # in `input` below and counts are restored after simulation.
    aggregates = []
    for hotkey, units in eligible.items():
        worker = next(w for w in legacy_policy["workers"] if w["minerHotkey"] == hotkey)
        aggregates.append(
            {
                "minerHotkey": hotkey,
                "workerId": worker["workerId"],
                "trustMode": "operator_whitelist",
                "profileId": "legacy",
                "profileHash": "0" * 64,
                "attempted": units,
                "succeeded": units,
                "failed": 0,
                "cancelled": 0,
                "inconclusive": 0,
                "completedVideoMs": units * 5_000,
                "eligibleSucceeded": units,
                "eligibleVideoMs": units * 5_000,
                "slowSucceeded": 0,
            }
        )
    legacy_ledger = {
        **ledger,
        "schema": LEDGER_SCHEMA,
        "policyHash": digest(legacy_policy),
        "gatewayPolicyHash": digest(legacy_policy["gatewayPolicy"]),
        "workerPoolHash": digest(legacy_policy["workers"]),
        "aggregates": aggregates,
    }
    artifact = _simulate_v1(legacy_ledger, legacy_policy, registry)
    artifact["input"]["ledgerSha256"] = digest(ledger)
    artifact["input"]["policySha256"] = ledger["policyHash"]
    artifact["input"]["gatewayPolicyHash"] = ledger["gatewayPolicyHash"]
    artifact["input"]["workerPoolHash"] = ledger["workerPoolHash"]
    artifact["service"]["eligibleVideos"] = sum(videos.values())
    artifact["service"]["eligibleWorkUnits"] = sum(eligible.values())
    artifact["service"]["rewardMerit"] = "eligibleWorkUnits"
    for row in artifact["miners"]:
        row["eligibleVideos"] = videos.get(row["hotkey"], 0)
        row["eligibleWorkUnits"] = eligible.get(row["hotkey"], 0)
    return artifact


def simulate(
    ledger: dict[str, Any], policy: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    if policy.get("schema") == POLICY_SCHEMA_V2 or ledger.get("schema") == LEDGER_SCHEMA_V2:
        return _simulate_v2(ledger, policy, registry)
    return _simulate_v1(ledger, policy, registry)


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


def _h3_profile_hash(profile: dict[str, Any]) -> str:
    """Mirror the gateway's H3 identity, excluding operator calibration."""
    duration = profile["durationMs"] // 1_000
    landscape = {"480p": (832, 480), "768p": (1344, 768), "1080p": (1920, 1080)}[
        profile["resolution"]
    ]
    width, height = landscape if profile["aspectRatio"] == "16:9" else landscape[::-1]
    native_frames = duration * 24 + (5 - duration * 24 % 17 + 17) % 17
    workflow_source = "c1dc580075d12dfe739e82591b2bc1fa84bd0536e53db7a5cdc16b1e93d9bd79"
    runtime = {
        "schema": "attelier/h3-runtime-identity/v1",
        "manifestSha256": "0547f0b4319dd2ca42fb23b27458d98a13b4547baf1174e591bf560a27af351e",
        "comfyRevision": "b0f4b7b294ce482a2e071d9d762c133d38c7aa07",
        "workflow": {
            "sampler": "res_multistep",
            "scheduler": "simple",
            "denoise": 1,
            "steps": 20,
            "fps": 24,
            "nativeFrameRule": "17k+5",
            "workflowSourceSha256": workflow_source,
        },
        "checkpoints": [
            {
                "role": "diffusion_model",
                "filename": "minimax_h3_fl2va_pruned_nvfp4.safetensors",
                "sha256": "8111a496dae7d47c914b8a7ef24d6ad7115c0576cd43eeb4be97c3c760df9502",
                "bytes": 20067070985,
            },
            {
                "role": "text_encoder",
                "filename": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
                "bytes": 15687142551,
            },
            {
                "role": "video_vae",
                "filename": "vae/minimax_h3_video_vae_fp16.safetensors",
                "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
                "bytes": 5207808496,
            },
            {
                "role": "audio_vae",
                "filename": "vae/minimax_h3_audio_vae_fp32.safetensors",
                "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
                "bytes": 605254808,
            },
        ],
    }
    return digest(
        {
            "schema": "attelier/h3-service-profile/v1",
            "durationMs": profile["durationMs"],
            "resolution": profile["resolution"],
            "aspectRatio": profile["aspectRatio"],
            "width": width,
            "height": height,
            "nativeFrames": native_frames,
            "fps": 24,
            "audio": True,
            "steps": 20,
            "checkpoint": {
                "id": "minimax_h3_fl2va_pruned_nvfp4.safetensors",
                "version": "h3-fl2va-nvfp4-v1",
            },
            "runtime": runtime,
            "profile": "base",
            "lora": None,
        }
    )


def _h3_catalog(*, include_retired: bool = False) -> dict[str, dict[str, Any]]:
    """Load the operator-pinned checkpoint catalogue, never miner-supplied data."""
    source = read_json(Path(os.environ.get("H3_CHECKPOINT_CATALOG_FILE", _H3_CATALOG_PATH)))
    if (
        set(source) != {"schema", "version", "definitions", "publication"}
        or source["schema"] != "attelier/h3-checkpoint-catalog/v1"
        or not isinstance(source["version"], str)
        or not isinstance(source["definitions"], list)
        or not isinstance(source["publication"], list)
    ):
        _fail("checkpoint_catalog")
    definitions: dict[str, dict[str, Any]] = {}
    for checkpoint in source["definitions"]:
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {
                "schema",
                "id",
                "version",
                "publisher",
                "repository",
                "revision",
                "precision",
                "runtime",
                "files",
            }
            or checkpoint.get("schema") != "attelier/h3-checkpoint-manifest/v2"
        ):
            _fail("checkpoint_catalog")
        checkpoint_id = _identifier(checkpoint.get("id"), "checkpoint_catalog")
        precision, files, runtime = (
            checkpoint.get("precision"),
            checkpoint.get("files"),
            checkpoint.get("runtime"),
        )
        if (
            checkpoint_id in definitions
            or not isinstance(checkpoint.get("version"), str)
            or not all(
                isinstance(checkpoint.get(key), str) and checkpoint[key]
                for key in ("publisher", "repository", "revision")
            )
            or not isinstance(precision, dict)
            or set(precision) != {"transformer", "textEncoder", "videoVae", "audioVae"}
            or precision.get("transformer") not in {"bf16", "nvfp4"}
            or precision.get("textEncoder") not in {"bf16", "nvfp4_awq"}
            or precision.get("videoVae") != "fp16"
            or precision.get("audioVae") != "fp32"
            or not isinstance(runtime, dict)
            or set(runtime) != {"kind", "comfyRepository", "comfyRevision", "loader", "memoryMode"}
            or runtime.get("kind") != "comfyui"
            or runtime.get("loader") != "native"
            or runtime.get("memoryMode") != "operator-selected"
            or not all(
                isinstance(runtime.get(key), str) and runtime[key]
                for key in ("comfyRepository", "comfyRevision")
            )
            or not isinstance(files, list)
            or len(files) != 4
        ):
            _fail("checkpoint_catalog")
        roles: set[str] = set()
        for item in files:
            if (
                not isinstance(item, dict)
                or set(item) != {"role", "filename", "destination", "sha256", "bytes"}
                or item.get("role")
                not in {"diffusion_model", "text_encoder", "video_vae", "audio_vae"}
                or item["role"] in roles
                or not all(
                    isinstance(item.get(key), str) and item[key]
                    for key in ("filename", "destination")
                )
                or _hash(item.get("sha256"), "checkpoint_catalog") is None
                or type(item.get("bytes")) is not int
                or item["bytes"] <= 0
            ):
                _fail("checkpoint_catalog")
            roles.add(item["role"])
        immutable = {
            "schema": checkpoint["schema"],
            "id": checkpoint_id,
            "version": checkpoint["version"],
            "publisher": checkpoint["publisher"],
            "repository": checkpoint["repository"],
            "revision": checkpoint["revision"],
            "precision": precision,
            "files": files,
        }
        definitions[checkpoint_id] = {
            "id": checkpoint_id,
            "version": checkpoint["version"],
            "manifestHash": digest(immutable),
        }
    checkpoints: dict[str, dict[str, Any]] = {}
    for publication in source["publication"]:
        if (
            not isinstance(publication, dict)
            or set(publication) not in ({"id", "status"}, {"id", "status", "legacy"})
            or publication.get("legacy") is not None
            and publication.get("legacy") is not True
        ):
            _fail("checkpoint_catalog")
        checkpoint_id = _identifier(publication.get("id"), "checkpoint_catalog")
        if (
            publication.get("status") not in {"published", "retired"}
            or checkpoint_id in checkpoints
        ):
            _fail("checkpoint_catalog")
        if checkpoint_id not in definitions:
            if publication.get("legacy") is True and publication.get("status") == "retired":
                continue
            _fail("checkpoint_catalog")
        if publication.get("status") == "retired" and not include_retired:
            continue
        checkpoints[checkpoint_id] = definitions[checkpoint_id]
    if not checkpoints:
        _fail("checkpoint_catalog")
    return checkpoints


def _h3_catalog_profile(profile: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[str, str]:
    """Mirror API h3-profiles.ts v2 identity exactly; calibration is excluded."""
    duration = profile["durationMs"] // 1_000
    landscape = {"480p": (832, 480), "768p": (1344, 768), "1080p": (1920, 1080)}[
        profile["resolution"]
    ]
    width, height = landscape if profile["aspectRatio"] == "16:9" else landscape[::-1]
    native_frames = duration * 24 + (5 - duration * 24 % 17 + 17) % 17
    checkpoint_ref = {
        "id": checkpoint["id"],
        "version": checkpoint["version"],
        "manifestHash": checkpoint["manifestHash"],
        "catalogId": checkpoint["id"],
    }
    identity = {
        "schema": "attelier/h3-service-profile/v2",
        "checkpoint": checkpoint_ref,
        "durationMs": profile["durationMs"],
        "resolution": profile["resolution"],
        "aspectRatio": profile["aspectRatio"],
        "width": width,
        "height": height,
        "nativeFrames": native_frames,
        "fps": 24,
        "audio": True,
        "steps": 20,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "profile": "base",
        "lora": None,
    }
    profile_id = (
        f"h3-{checkpoint['id']}-{duration}s-{profile['resolution']}-"
        f"{profile['aspectRatio'].replace(':', 'x')}-20-v1"
    )
    return profile_id, digest(identity)


def _matches_h3_catalog_profile(profile: dict[str, Any], checkpoint: dict[str, Any]) -> bool:
    try:
        expected_id, expected_hash = _h3_catalog_profile(profile, checkpoint)
    except (KeyError, TypeError):
        return False
    return profile.get("id") == expected_id and profile.get("hash") == expected_hash


def _approved_h3_profile_identity(profile: dict[str, Any]) -> tuple[str, str] | None:
    """Accept only the frozen legacy identity or a bundled catalog identity."""
    try:
        legacy = (
            f"h3-base-{profile['durationMs'] // 1_000}s-{profile['resolution']}-"
            f"{profile['aspectRatio'].replace(':', 'x')}-20-v1",
            _h3_profile_hash(profile),
        )
        if profile.get("id") == legacy[0] and profile.get("hash") == legacy[1]:
            return legacy
        for checkpoint in _h3_catalog(include_retired=True).values():
            if _matches_h3_catalog_profile(profile, checkpoint):
                return _h3_catalog_profile(profile, checkpoint)
    except (KeyError, TypeError):
        return None
    return None


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
    v2 = config.get("schema") == "attelier/subnet-h3/v2"
    v3 = config.get("schema") == "attelier/subnet-h3/v3"
    required_v2 = required | {"serviceProfiles"}
    required_v3 = required_v2 | {"checkpointId"}
    base = required_v3 if v3 else required_v2 if v2 else required
    optional = (
        {"requireFrozenLightWindow", "routingAdmission", "autoProviderAdmission"}
        if v3 else {"requireFrozenLightWindow"}
    )
    if not base <= set(config) <= base | optional or config.get("schema") not in {
        "attelier/subnet-h3/v1",
        "attelier/subnet-h3/v2",
        "attelier/subnet-h3/v3",
    }:
        _fail("prelaunch_config_schema")
    _namespace(config, str(config["schema"]), "prelaunch_config_namespace")
    if (
        config["mode"] != "prelaunch"
        or config["trustMode"] != "operator_whitelist"
        or not isinstance(config["registrationFile"], str)
        or not isinstance(config["miners"], list)
        or not isinstance(config["allowedClients"], list)
        or not isinstance(config["syntheticClientIds"], list)
        or not isinstance(config["serviceProfile"], dict)
        or (
            "requireFrozenLightWindow" in config
            and type(config["requireFrozenLightWindow"]) is not bool
        )
        or (
            "autoProviderAdmission" in config
            and type(config["autoProviderAdmission"]) is not bool
        )
    ):
        _fail("prelaunch_config")
    approved: set[str] = set()
    checkpoint = None
    if v3:
        checkpoint = _h3_catalog().get(config.get("checkpointId"))
        if checkpoint is None or config.get("routingAdmission", "bootstrap") not in {
            "bootstrap",
            "strict",
        }:
            _fail("prelaunch_config_checkpoint")
    if v2 or v3:
        if (
            config.get("requireFrozenLightWindow") is not True
            or not isinstance(config.get("serviceProfiles"), list)
            or not config["serviceProfiles"]
            or len(config["serviceProfiles"]) > 18
        ):
            _fail("prelaunch_config_profiles")
        for value in config["serviceProfiles"]:
            if (
                not isinstance(value, dict)
                or set(value)
                != {
                    "id",
                    "hash",
                    "durationMs",
                    "resolution",
                    "aspectRatio",
                    "maxJobSeconds",
                    "maxTtvMs",
                    "rewardWorkUnits",
                    "rewardBasis",
                    "approvalHash",
                }
                or _identifier(value.get("id"), "prelaunch_config_profile") in approved
                or (v3 and not _matches_h3_catalog_profile(value, checkpoint))
                or (
                    not v3
                    and _hash(value.get("hash"), "prelaunch_config_profile")
                    != _h3_profile_hash(value)
                )
                or value.get("rewardBasis") not in {"reference_gpu", "pixel_frames"}
                or _hash(value.get("approvalHash"), "prelaunch_config_profile") is None
                or value.get("durationMs") not in {5_000, 10_000, 15_000}
                or value.get("resolution") not in {"480p", "768p", "1080p"}
                or value.get("aspectRatio") not in {"16:9", "9:16"}
                or not 1
                <= _integer(value.get("maxJobSeconds"), "prelaunch_config_profile")
                <= 7_200
                or not 1 <= _integer(value.get("maxTtvMs"), "prelaunch_config_profile") <= 7_200_000
                or not 1
                <= _integer(value.get("rewardWorkUnits"), "prelaunch_config_profile")
                <= 1_000_000_000
            ):
                _fail("prelaunch_config_profiles")
            approved.add(value["id"])
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
        if "autoManaged" in miner and (not v3 or type(miner["autoManaged"]) is not bool):
            _fail("prelaunch_config_miner")
        if v2 or v3:
            ids = miner.get("profileIds")
            if (
                not isinstance(ids, list)
                or not ids
                or len(set(ids)) != len(ids)
                or any(not isinstance(value, str) or value not in approved for value in ids)
            ):
                _fail("prelaunch_config_miner")
        identifiers.add(identifier)
        miners.append(
            {
                "id": identifier,
                "uid": uid,
                "hotkey": miner["hotkey"],
                "workerId": miner["workerId"],
                "operatorEnabled": miner["operatorEnabled"],
                "operatorReviewedExpires": reviewed,
                "autoManaged": config.get("autoProviderAdmission") is True
                and miner.get("autoManaged") is True,
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
        # A stale discovered member is unavailable, not a reason to prevent
        # refreshing every other member and the full registration snapshot.
        # The reconciler needs that fresh snapshot to renew a valid approval.
        if miner["autoManaged"] and (
            miner["operatorReviewedExpires"] <= observed["reviewedAt"]
            or registered is None
            or registered[0] != miner["hotkey"]
        ):
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
            {"uid": uid, "hotkey": owners[uid][0], "coldkey": owners[uid][1]}
            for uid in dict.fromkeys(
                miner["uid"] for miner in miners if miner["id"] in workers
            )
        ],
    }
    return registry, simulation
