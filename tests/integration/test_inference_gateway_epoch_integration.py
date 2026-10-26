import base64
import copy
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("bittensor")
from bittensor.keyfiles import Keypair  # noqa: E402

from forge.chain import Sr25519Signer  # noqa: E402
from forge.inference.gateway_epoch import (  # noqa: E402
    commitment,
    digest,
    light_plan,
    master_window,
    sign_gateway,
    verify_gateway,
)


def test_signed_gateway_epoch_to_current_uid_plan_rejects_tamper_replay_and_bad_anchors(tmp_path):
    key = tmp_path / "gateway.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)], check=True)
    public = subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
        capture_output=True,
        check=True,
    ).stdout[-32:]
    alice = Sr25519Signer(SimpleNamespace(hotkey=Keypair.create_from_uri("//Alice")))
    bob = Keypair.create_from_uri("//Bob").ss58_address
    profile_hash = "a" * 64
    policy = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "gatewayPublicKey": base64.b64encode(public).decode(),
        "pool": {"name": "m1-test"},
        "gatewayPolicy": {
            "minSuccessBps": 8000,
            "minAvailabilityBps": 9000,
            "maxTtvMs": 1000,
            "serviceProfile": {
                "id": "five-second-480p",
                "hash": profile_hash,
                "durationMs": 5000,
                "resolution": "480p",
            },
        },
        "workers": [
            {"minerHotkey": "miner-a", "workerId": "worker-a", "trustMode": "operator_whitelist"},
            {"minerHotkey": "miner-b", "workerId": "worker-b", "trustMode": "operator_whitelist"},
        ],
        "masterPublisherHotkey": alice.validator_hotkey,
        "m0PublisherHotkey": bob,
    }
    now = 2_000_000

    def row(hotkey, worker, completed, trust="operator_whitelist"):
        return {
            "minerHotkey": hotkey,
            "workerId": worker,
            "trustMode": trust,
            "trafficKind": "synthetic",
            "profileId": "five-second-480p",
            "profileHash": profile_hash,
            "attempted": 10,
            "succeeded": 10,
            "failed": 0,
            "cancelled": 0,
            "inconclusive": 0,
            "completedVideoMs": completed,
            "eligibleSucceeded": 10,
            "eligibleVideoMs": completed,
            "slowSucceeded": 0,
            "TTVp50Ms": 700,
            "TTVp95Ms": 900,
            "computeAmountMicros": 1,
            "knownComputeAmountMicros": 1,
            "unknownComputeCount": 0,
            "billedAmountMicros": 0,
            "knownBilledAmountMicros": 0,
            "unknownBilledCount": 0,
            "billedUnit": "microcredits",
        }

    body = {
        "schema": "attelier/gateway-window/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "window": {"startMs": 1_000_000, "endMs": 1_100_000},
        "sequence": 7,
        "createdAtMs": 1_100_001,
        "expiresAtMs": 3_000_000,
        "policyHash": digest(policy["gatewayPolicy"]),
        "poolHash": digest(policy["pool"]),
        "ledgerRowsHash": "b" * 64,
        "previousHash": None,
        "aggregates": [row("miner-a", "worker-a", 50_000), row("miner-b", "worker-b", 50_000)],
    }
    gateway = sign_gateway(body, key)
    assert verify_gateway(gateway, policy, now) == body
    master = master_window(gateway, policy, alice, "c" * 64, now)
    assert sum(weight for _, weight in master["body"]["weights"]) == 65535
    anchor = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "finalizedRegistryBlock": 0,
        "finalizedRegistryBlockHash": "c" * 64,
        "observedBlock": 1,
        "observedBlockHash": "d" * 64,
        "commitmentBlock": 1,
        "publisherHotkey": alice.validator_hotkey,
        "rawCommitment": commitment(master),
        "m0PublisherHotkey": bob,
    }
    metagraph = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "observedBlock": 1,
        "observedBlockHash": "d" * 64,
        "uids": {"miner-a": 8, "miner-b": 2},
    }
    plan = light_plan(master, policy, anchor, metagraph, now_ms=now)
    assert plan["uids"] == [8, 2] and sum(plan["weights"]) == 65535
    tampered = copy.deepcopy(master)
    tampered["body"]["weights"][0][1] += 1
    with pytest.raises(ValueError):
        light_plan(tampered, policy, anchor, metagraph, now_ms=now)
    with pytest.raises(ValueError):
        light_plan(master, policy, anchor, {**metagraph, "uids": {"miner-a": 8}}, now_ms=now)
    with pytest.raises(ValueError):
        light_plan(master, policy, {**anchor, "rawCommitment": "bad"}, metagraph, now_ms=now)
    assert light_plan(master, policy, anchor, metagraph, plan["nextState"], now_ms=now - 1)[
        "uids"
    ] == [8, 2]
    with pytest.raises(ValueError):
        light_plan(
            master,
            policy,
            anchor,
            metagraph,
            {**plan["nextState"], "masterDigest": "d" * 64},
            now_ms=now,
        )
    inflated = copy.deepcopy(body)
    inflated["aggregates"][0]["completedVideoMs"] += 1
    with pytest.raises(ValueError, match="gateway_profile_duration"):
        master_window(sign_gateway(inflated, key), policy, alice, "c" * 64, now)


def test_empty_eligible_vector_and_namespace_stay_testnet_only(tmp_path):
    key = tmp_path / "gateway.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)], check=True)
    public = subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
        capture_output=True,
        check=True,
    ).stdout[-32:]
    signer = Sr25519Signer(SimpleNamespace(hotkey=Keypair.create_from_uri("//Alice")))
    policy = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "gatewayPublicKey": base64.b64encode(public).decode(),
        "pool": {},
        "gatewayPolicy": {
            "minSuccessBps": 1,
            "minAvailabilityBps": 1,
            "maxTtvMs": 1,
            "serviceProfile": {
                "id": "p",
                "hash": "a" * 64,
                "durationMs": 5000,
                "resolution": "480p",
            },
        },
        "workers": [],
        "masterPublisherHotkey": signer.validator_hotkey,
        "m0PublisherHotkey": Keypair.create_from_uri("//Bob").ss58_address,
    }
    body = {
        "schema": "attelier/gateway-window/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "window": {"startMs": 1, "endMs": 2},
        "sequence": 1,
        "createdAtMs": 2,
        "expiresAtMs": 10,
        "policyHash": digest(policy["gatewayPolicy"]),
        "poolHash": digest(policy["pool"]),
        "ledgerRowsHash": "b" * 64,
        "previousHash": None,
        "aggregates": [],
    }
    master = master_window(sign_gateway(body, key), policy, signer, "c" * 64, 3)
    assert master["body"]["weights"] == []
    body["network"] = "main"
    with pytest.raises(ValueError):
        sign_gateway(body, key)
