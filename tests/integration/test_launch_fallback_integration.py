"""Pinned native fallback proof against a fake external archive boundary."""

import hashlib
from importlib.resources import files
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
from bittensor import storage

from forge.chain import BittensorChain
from forge.chain.launch_fallback import (
    RUNTIME_481,
    RUNTIME_483,
    RUNTIME_484,
    qualify_fallback,
    verify_fallback,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.launch_cli import preflight
from forge.validator.launch_config import LaunchConfig


def _genesis() -> str:
    return "1" * 64


class _Raw:
    def get_block_hash(self, block):
        assert block == 0
        return "genesis"

    def rpc_request(self, name, params):
        return name, params


class _At:
    def __init__(self, chain):
        self.chain = chain

    def query(self, item, arguments):
        assert arguments == [self.chain.netuid]
        if item == storage.SubtensorModule.RecycleOrBurn:
            return {self.chain.mode: None}
        if item == storage.SubtensorModule.SubnetOwnerHotkey:
            return self.chain.owner_hotkey
        raise AssertionError(item)


class _Subtensor:
    def __init__(self, chain):
        self.chain = chain
        self._client = SimpleNamespace(_substrate=SimpleNamespace(raw=_Raw()))

    def _call(self, value):
        if value == "genesis":
            return "0x" + self.chain.genesis
        name, _params = value
        if name == "state_getRuntimeVersion":
            return {"specName": "node-subtensor", "specVersion": self.chain.spec_version}
        assert name == "state_getStorage"
        return "0x" + self.chain.runtime_code.hex()

    def at(self, block):
        assert block == 99
        return _At(self.chain)


class _Chain:
    network = "test"
    netuid = 578

    def __init__(self, runtime_code):
        self.runtime_code = runtime_code
        self.genesis = _genesis()
        self.spec_version = 481
        self.mode = "Burn"
        self.owner_hotkey = "owner-not-uid-zero"
        self.subtensor = _Subtensor(self)

    def finalized_height(self):
        return 99

    def finalized_block_hash(self, block):
        assert block == 99
        return "block-99"

    def uid_snapshot(self, block):
        assert block == 99
        return {0: "some-other-hotkey", 7: self.owner_hotkey}

    def committee_snapshot(self, block):
        assert block == 99
        return {"some-other-hotkey": 1}


@pytest.fixture(scope="module")
def official_runtime_wasm():
    # External release boundary, held only in test memory: no tracked runtime blob.
    with urlopen(RUNTIME_481["wasm_url"], timeout=30) as response:
        wasm = response.read()
    assert len(wasm) == RUNTIME_481["wasm_bytes"]
    assert hashlib.sha256(wasm).hexdigest() == RUNTIME_481["runtime_hash"]
    return wasm


@pytest.fixture(scope="module")
def deployed_runtime_483_wasm():
    # Read-only archive RPC, held in memory; no keys or local WASM blob.
    chain = BittensorChain("test", 578)
    try:
        chain.finalized_height()
        raw = chain.subtensor._client._substrate.raw
        encoded = chain.subtensor._call(raw.rpc_request(
            "state_getStorage", ["0x3a636f6465", RUNTIME_483["origin_block"]]
        ))
        wasm = bytes.fromhex(encoded.removeprefix("0x"))
        assert len(wasm) == RUNTIME_483["wasm_bytes"]
        assert hashlib.sha256(wasm).hexdigest() == RUNTIME_483["runtime_hash"]
        return wasm
    finally:
        chain.close()


@pytest.fixture(scope="module")
def deployed_runtime_484_wasm():
    # Read-only archive RPC, held in memory; no keys or local WASM blob.
    chain = BittensorChain("test", 578)
    try:
        chain.finalized_height()
        raw = chain.subtensor._client._substrate.raw
        encoded = chain.subtensor._call(raw.rpc_request(
            "state_getStorage", ["0x3a636f6465", RUNTIME_484["origin_block"]]
        ))
        wasm = bytes.fromhex(encoded.removeprefix("0x"))
        assert len(wasm) == RUNTIME_484["wasm_bytes"]
        assert hashlib.sha256(wasm).hexdigest() == RUNTIME_484["runtime_hash"]
        return wasm
    finally:
        chain.close()


def test_native_fallback_proof_rechecks_owner_burn_route_and_nonzero_uid(official_runtime_wasm):
    chain = _Chain(official_runtime_wasm)
    proof = qualify_fallback(
        chain, genesis_hash=_genesis(), runtime_hash=RUNTIME_481["runtime_hash"]
    )

    assert proof["target_uid"] == 7
    assert proof["target_hotkey"] == "owner-not-uid-zero"
    assert proof["storage"]["RecycleOrBurn"] == {"Burn": None}
    assert verify_fallback(
        chain, proof, genesis_hash=_genesis(), runtime_hash=RUNTIME_481["runtime_hash"]
    )

    proof["target_uid"] = 0
    assert not verify_fallback(
        chain, proof, genesis_hash=_genesis(), runtime_hash=RUNTIME_481["runtime_hash"]
    )


def test_native_fallback_rejects_non_test_network(official_runtime_wasm):
    chain = _Chain(official_runtime_wasm)
    chain.network = "finney"
    with pytest.raises(ValueError, match="testnet only"):
        qualify_fallback(chain, genesis_hash=_genesis(), runtime_hash=RUNTIME_481["runtime_hash"])


def test_launch_preflight_exposes_unreviewed_runtime_before_paid_dispatch(official_runtime_wasm):
    contract = ChallengeContract()
    config = LaunchConfig("test", 578, _genesis(), contract.contract_hash(), "preflight")
    chain = _Chain(official_runtime_wasm)
    reviewed = preflight(config, contract, chain)
    assert reviewed["native_fallback"]["status"] == "QUALIFIED"
    assert reviewed["native_fallback"]["proof"]["target_uid"] == 7
    assert reviewed["publication_qualified"] is False
    # The advertised version alone cannot approve changed runtime bytes.
    chain.runtime_code = official_runtime_wasm + b"changed-external-runtime"
    changed = preflight(config, contract, chain)
    assert changed["status"] == "READ_ONLY_VERIFIED"
    assert changed["native_fallback"]["status"] == "NOT_QUALIFIED"
    assert changed["publication_qualified"] is False


@pytest.mark.parametrize("runtime_fixture,reviewed", [
    ("deployed_runtime_483_wasm", RUNTIME_483),
    ("deployed_runtime_484_wasm", RUNTIME_484),
])
def test_reviewed_runtime_preflight_replays_burn_proof_and_rejects_changed_runtime(
    request, runtime_fixture, reviewed,
):
    receipt = files("forge.chain").joinpath(reviewed["receipt"])
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == reviewed["receipt_sha256"]
    chain = _Chain(request.getfixturevalue(runtime_fixture))
    chain.genesis = reviewed["genesis_hash"]
    chain.spec_version = reviewed["spec_version"]
    contract = ChallengeContract()
    config = LaunchConfig("test", 578, chain.genesis, contract.contract_hash(), "runtime-reviewed")
    result = preflight(config, contract, chain)
    assert result["native_fallback"]["status"] == "QUALIFIED"
    assert result["publication_qualified"] is False
    proof = result["native_fallback"]["proof"]
    assert proof["schema"] == "forge/launch-native-fallback-proof/v2"
    assert proof["target_uid"] == 7
    assert proof["runtime_review"]["source_commit_verified"] is False
    assert verify_fallback(
        chain, proof, genesis_hash=chain.genesis, runtime_hash=reviewed["runtime_hash"]
    )
    # Even a self-consistently rehashed forged review cannot replace the pinned review.
    proof["runtime_review"]["source_commit_verified"] = True
    proof["proof_hash"] = hash_canonical(
        proof["schema"], {key: value for key, value in proof.items() if key != "proof_hash"}
    )
    assert not verify_fallback(
        chain, proof, genesis_hash=chain.genesis, runtime_hash=reviewed["runtime_hash"]
    )
    chain.mode = "Recycle"
    assert preflight(config, contract, chain)["native_fallback"]["status"] == "NOT_QUALIFIED"
    chain.mode = "Burn"
    chain.runtime_code += b"changed-runtime-same-advertised-version"
    assert preflight(config, contract, chain)["native_fallback"]["status"] == "NOT_QUALIFIED"
