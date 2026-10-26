import copy
import hashlib
import json
import subprocess
import sys

GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
# Public development identities (Alice, Bob, Charlie), never production wallets.
OWNER_COLDKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OWNER_HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
OTHER = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"
PROFILE = "a" * 64
DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inputs(rows=None, *, budget=125, target=600, window_ms=None):
    gateway = {
        "minSuccessBps": 9500,
        "maxTtvMs": 180000,
        "serviceProfile": {
            "id": "h3-base-5s-480p-20-v1",
            "hash": PROFILE,
            "durationMs": 5000,
            "resolution": "480p",
        },
    }
    workers = [
        {
            "minerHotkey": OWNER_HOTKEY,
            "workerId": "fixture-owner-worker",
            "trustMode": "operator_whitelist",
        },
        {"minerHotkey": OTHER, "workerId": "other-worker", "trustMode": "operator_whitelist"},
    ]
    policy = {
        "schema": "attelier/prelaunch-policy/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "gatewayPolicy": gateway,
        "workers": workers,
        "simulation": {
            "serviceBudgetBps": budget,
            "targetVideosPerDay": target,
            "nominalMinerEmissionRaoPerDay": "2952000000000",
        },
    }
    if window_ms is not None:
        policy["windowMs"] = window_ms
    registry = {
        "schema": "attelier/prelaunch-registry/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "finalizedBlock": 9367342,
        "finalizedBlockHash": "b" * 64,
        "subnetOwnerColdkey": OWNER_COLDKEY,
        "miners": [
            {"uid": 3, "hotkey": OWNER_HOTKEY, "coldkey": OWNER_COLDKEY},
            {"uid": 4, "hotkey": OTHER, "coldkey": OTHER},
        ],
    }

    def row(hotkey, worker, eligible):
        return {
            "minerHotkey": hotkey,
            "workerId": worker,
            "trustMode": "operator_whitelist",
            "profileId": gateway["serviceProfile"]["id"],
            "profileHash": PROFILE,
            "attempted": eligible,
            "succeeded": eligible,
            "failed": 0,
            "cancelled": 0,
            "inconclusive": 0,
            "completedVideoMs": eligible * 5000,
            "eligibleSucceeded": eligible,
            "eligibleVideoMs": eligible * 5000,
            "slowSucceeded": 0,
        }

    ledger = {
        "schema": "attelier/prelaunch-ledger/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "window": {
            "startMs": DAY_MS if window_ms is None else window_ms,
            "endMs": 2 * DAY_MS if window_ms is None else 2 * window_ms,
        },
        "sequence": 1,
        "closedAtMs": 2 * DAY_MS + 1 if window_ms is None else 2 * window_ms + 1,
        "policyHash": digest(policy),
        "gatewayPolicyHash": digest(gateway),
        "workerPoolHash": digest(workers),
        "aggregates": rows
        if rows is not None
        else [row(OWNER_HOTKEY, "fixture-owner-worker", 300), row(OTHER, "other-worker", 300)],
    }
    return ledger, policy, registry


def reseal(ledger, policy):
    ledger["policyHash"] = digest(policy)
    ledger["gatewayPolicyHash"] = digest(policy["gatewayPolicy"])
    ledger["workerPoolHash"] = digest(policy["workers"])


def run_cli(tmp_path, ledger, policy, registry, *extra):
    paths = {}
    for name, value in (("ledger", ledger), ("policy", policy), ("registry", registry)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    return subprocess.run(
        [
            sys.executable,
            "scripts/inference/prelaunch.py",
            "--ledger",
            str(paths["ledger"]),
            "--policy",
            str(paths["policy"]),
            "--registry",
            str(paths["registry"]),
            *extra,
        ],
        cwd=".",
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )


def test_prelaunch_cli_full_target_uses_quantized_weights_owner_burn_and_atomic_output(tmp_path):
    ledger, policy, registry = inputs()
    out = tmp_path / "artifact.json"
    first = run_cli(tmp_path, ledger, policy, registry, "--out", str(out))
    assert first.returncode == 0 and first.stdout == ""
    encoded = out.read_text()
    second = run_cli(tmp_path, ledger, policy, registry, "--out", str(out))
    assert second.returncode == 0 and out.read_text() == encoded
    artifact = json.loads(encoded)
    assert artifact["simulationOnly"] is True and artifact["publishable"] is False
    assert artifact["asset"] == {"kind": "alpha", "netuid": 126, "decimals": 9}
    assert artifact["chainCalls"] == [] and artifact["createdAtMs"] == ledger["closedAtMs"]
    assert artifact["counterfactualServiceAllocation"]["allocatedRao"] == "36900000000"
    entries = artifact["unsignedProposedWeights"]["entries"]
    assert sum(entry["weight"] for entry in entries) == 65535
    owner = next(row for row in artifact["miners"] if row["hotkey"] == OWNER_HOTKEY)
    other = next(row for row in artifact["miners"] if row["hotkey"] == OTHER)
    assert owner["hotkey"] == OWNER_HOTKEY and owner["ownerAssociated"] is True
    assert owner["chainExpectedMinerPayoutRao"] == "0"
    assert other["chainExpectedMinerPayoutRao"] == str(
        2_952_000_000_000 * other["proposedWeight"] // 65535
    )
    assert (
        int(artifact["ownerAdjustedExpectedPayout"]["redirectedToSimulatedBurnRao"])
        == 2_952_000_000_000 * owner["proposedWeight"] // 65535
    )
    changed = copy.deepcopy(ledger)
    changed["sequence"] = 2
    collision = run_cli(tmp_path, changed, policy, registry, "--out", str(out))
    assert collision.returncode != 0 and "different_artifact" in collision.stderr


def test_prelaunch_cli_handles_empty_and_zero_quantized_service_without_fake_payout(tmp_path):
    ledger, policy, registry = inputs([], window_ms=HOUR_MS)
    policy["workers"] = []
    registry["miners"] = []
    # A closed empty window is valid only when its policy binding is resealed
    # against the empty onboarding pool; it must not reuse a prior worker hash.
    reseal(ledger, policy)
    empty = run_cli(tmp_path, ledger, policy, registry)
    assert empty.returncode == 0
    empty_artifact = json.loads(empty.stdout)
    assert empty_artifact["unsignedProposedWeights"]["entries"] == [
        {"kind": "burn", "weight": 65535}
    ]
    assert empty_artifact["assumptions"] == {"dailyUtcWindow": False, "windowMs": HOUR_MS,
                                               "noYumaConsensusSimulation": True,
                                               "noTransactionOrSignature": True}
    ledger, policy, registry = inputs(budget=1)
    ledger["aggregates"] = [ledger["aggregates"][1]]
    ledger["aggregates"][0].update(
        attempted=1, succeeded=1, completedVideoMs=5000, eligibleSucceeded=1, eligibleVideoMs=5000
    )
    tiny = run_cli(tmp_path, ledger, policy, registry)
    assert tiny.returncode == 0
    miner = json.loads(tiny.stdout)["miners"][0]
    assert miner["counterfactualServiceAllocationRao"] != "0"
    assert miner["proposedWeight"] == 0 and miner["chainExpectedMinerPayoutRao"] == "0"


def test_prelaunch_cli_rejects_service_policy_hash_and_bad_network_or_counts(tmp_path):
    ledger, policy, registry = inputs()
    wrong_hash = copy.deepcopy(ledger)
    wrong_hash["gatewayPolicyHash"] = "c" * 64
    result = run_cli(tmp_path, wrong_hash, policy, registry)
    assert result.returncode != 0 and "ledger_service_policy_hash" in result.stderr
    repriced = copy.deepcopy(policy)
    repriced["simulation"]["serviceBudgetBps"] = 126
    result = run_cli(tmp_path, ledger, repriced, registry)
    assert result.returncode != 0 and "ledger_service_policy_hash" in result.stderr
    recadenced = copy.deepcopy(policy)
    recadenced["windowMs"] = HOUR_MS
    result = run_cli(tmp_path, ledger, recadenced, registry)
    assert result.returncode != 0 and "ledger_service_policy_hash" in result.stderr
    wrong_network = copy.deepcopy(ledger)
    wrong_network["network"] = "test"
    result = run_cli(tmp_path, wrong_network, policy, registry)
    assert result.returncode != 0 and "ledger_namespace" in result.stderr
    bad_counts = copy.deepcopy(ledger)
    bad_counts["aggregates"][0]["completedVideoMs"] += 1
    result = run_cli(tmp_path, bad_counts, policy, registry)
    assert result.returncode != 0 and "ledger_counts" in result.stderr


def test_prelaunch_cli_hourly_prorates_daily_budget_and_keeps_owner_at_zero(tmp_path):
    daily_ledger, daily_policy, daily_registry = inputs()
    daily = run_cli(tmp_path, daily_ledger, daily_policy, daily_registry)
    assert daily.returncode == 0
    hourly_ledger, hourly_policy, hourly_registry = inputs(window_ms=HOUR_MS)
    for row in hourly_ledger["aggregates"]:
        row.update(
            attempted=25,
            succeeded=25,
            completedVideoMs=125_000,
            eligibleSucceeded=25,
            eligibleVideoMs=125_000,
        )
    hourly = run_cli(tmp_path, hourly_ledger, hourly_policy, hourly_registry)
    assert hourly.returncode == 0
    hourly_artifact = json.loads(hourly.stdout)
    daily_artifact = json.loads(daily.stdout)
    assert hourly_artifact["service"]["windowMs"] == HOUR_MS
    assert hourly_artifact["service"]["windowTargetVideos"] == {
        "numerator": 600 * HOUR_MS,
        "denominator": DAY_MS,
    }
    assert int(daily_artifact["counterfactualServiceAllocation"]["allocatedRao"]) == 24 * int(
        hourly_artifact["counterfactualServiceAllocation"]["allocatedRao"]
    )
    owner = next(row for row in hourly_artifact["miners"] if row["hotkey"] == OWNER_HOTKEY)
    assert owner["ownerAssociated"] is True and owner["chainExpectedMinerPayoutRao"] == "0"
    fleet_ledger, fleet_policy, fleet_registry = inputs(budget=2500, target=15_000)
    for row in fleet_ledger["aggregates"]:
        row.update(
            attempted=7500,
            succeeded=7500,
            completedVideoMs=37_500_000,
            eligibleSucceeded=7500,
            eligibleVideoMs=37_500_000,
        )
    fleet = run_cli(tmp_path, fleet_ledger, fleet_policy, fleet_registry)
    assert fleet.returncode == 0
    fleet_artifact = json.loads(fleet.stdout)
    assert fleet_artifact["service"]["effectiveServiceBudgetNumerator"] == 2500
    assert fleet_artifact["service"]["effectiveServiceBudgetDenominator"] == 1
    assert fleet_artifact["counterfactualServiceAllocation"]["allocatedRao"] == "738000000000"
    assert sum(
        entry["weight"]
        for entry in fleet_artifact["unsignedProposedWeights"]["entries"]
        if entry["kind"] == "miner"
    ) == 16383


def test_prelaunch_cli_has_no_wallet_or_publishing_arguments_and_rejects_bad_uid(tmp_path):
    help_text = subprocess.run(
        [sys.executable, "scripts/inference/prelaunch.py", "--help"],
        cwd=".",
        env={"PYTHONPATH": "src"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert (
        "--execute" not in help_text
        and "wallet" not in help_text
        and "publication" not in help_text
    )
    ledger, policy, registry = inputs()
    invalid_uid = copy.deepcopy(registry)
    invalid_uid["miners"][0]["uid"] = 65536
    result = run_cli(tmp_path, ledger, policy, invalid_uid)
    assert result.returncode != 0 and "registry_uid" in result.stderr


def registry_inputs(*, reviewed_expires=2_000_000_000):
    config = {
        "schema": "attelier/subnet-h3/v1",
        "mode": "prelaunch",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "trustMode": "operator_whitelist",
        "allowedClients": ["provider-console"],
        "syntheticClientIds": ["prelaunch-driver"],
        "registrationFile": "/var/lib/attelier-inference/h3-prelaunch-registry.json",
        "serviceProfile": {"id": "h3-base-5s-480p-20-v1", "hash": PROFILE},
        "miners": [
            {
                "id": "fixture-owner",
                "uid": 3,
                "hotkey": OWNER_HOTKEY,
                "workerId": "fixture-owner-worker",
                "operatorEnabled": True,
                "operatorReviewedExpires": reviewed_expires,
            }
        ],
    }
    snapshot = {
        "schema": "forge/inference-registry-snapshot/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "finalized": True,
        "mechanismCount": 1,
        "block": 9367342,
        "blockHash": "0x" + "b" * 64,
        "timestamp": "2026-10-26T12:00:00Z",
        "subnetOwnerColdkey": OWNER_COLDKEY,
        "owners": {
            "3": {"hotkey": OWNER_HOTKEY, "coldkey": OWNER_COLDKEY},
            "4": {"hotkey": OTHER, "coldkey": OTHER},
        },
    }
    return config, snapshot


def run_registry_cli(tmp_path, config, snapshot, *extra):
    config_path, snapshot_path = tmp_path / "config.json", tmp_path / "snapshot.json"
    config_path.write_text(json.dumps(config))
    snapshot_path.write_text(json.dumps(snapshot))
    out, simulation = tmp_path / "registry.json", tmp_path / "simulation-registry.json"
    command = [
        sys.executable,
        "scripts/inference/registry.py",
        "--config",
        str(config_path),
        "--chain-snapshot",
        str(snapshot_path),
        "--out",
        str(out),
        "--simulation-registry-out",
        str(simulation),
        *extra,
    ]
    result = subprocess.run(
        command,
        cwd=".",
        env={"PYTHONPATH": "src", "H3_SUBNET_MODE": "prelaunch"},
        capture_output=True,
        text=True,
    )
    return result, out, simulation


def test_prelaunch_registry_cli_projects_finalized_fixture_and_simulation_registry(tmp_path):
    config, snapshot = registry_inputs()
    result, out, simulation = run_registry_cli(tmp_path, config, snapshot)
    assert result.returncode == 0
    registry = json.loads(out.read_text())
    assert registry == {
        "schema": "forge/inference-registry/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "mechanismCount": 1,
        "block": 9367342,
        "blockHash": "0x" + "b" * 64,
        "generatedAt": "2026-10-26T12:00:00Z",
        "uids": {"3": OWNER_HOTKEY, "4": OTHER},
        "workers": {
            "fixture-owner": {
                "workerId": "fixture-owner-worker",
                "hotkey": OWNER_HOTKEY,
                "operatorReviewedExpires": 2_000_000_000,
            }
        },
    }
    projected = json.loads(simulation.read_text())
    assert projected["subnetOwnerColdkey"] == OWNER_COLDKEY
    assert projected["miners"] == [{"uid": 3, "hotkey": OWNER_HOTKEY, "coldkey": OWNER_COLDKEY}]


def test_prelaunch_registry_cli_fails_closed_for_expired_or_wrong_finalized_snapshot(tmp_path):
    config, snapshot = registry_inputs(reviewed_expires=1)
    result, _, _ = run_registry_cli(tmp_path, config, snapshot)
    assert result.returncode != 0 and "operator_review_expired" in result.stderr
    config, snapshot = registry_inputs()
    snapshot["genesis"] = "0x" + "a" * 64
    result, _, _ = run_registry_cli(tmp_path, config, snapshot)
    assert result.returncode != 0 and "registry_snapshot_namespace" in result.stderr
