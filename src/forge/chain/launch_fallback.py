"""Read-only proof that the runtime's native fallback target is actually burned."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any
from urllib.request import urlopen

from bittensor import storage

from forge.protocol.hashing import hash_canonical

RUNTIME_481 = {
    "release": "v481",
    "commit": "6a972693ae9580541225bd37696ac2fee26b8ae9",
    "wasm_url": "https://github.com/RaoFoundation/subtensor/releases/download/v481/subtensor.wasm",
    "wasm_bytes": 2_652_697,
    "runtime_hash": "48225f3199bf317966d0214588bc45fb49c28c3e7bd510293bb4da7d6aea224b",
    "coinbase_source": "https://raw.githubusercontent.com/RaoFoundation/subtensor/"
    "6a972693ae9580541225bd37696ac2fee26b8ae9/"
    "pallets/subtensor/src/coinbase/run_coinbase.rs",
    "storage_source": "https://raw.githubusercontent.com/RaoFoundation/subtensor/"
    "6a972693ae9580541225bd37696ac2fee26b8ae9/pallets/subtensor/src/lib.rs",
    "coinbase_source_sha256": "62f17ad9132495fb1d50f951d7b6cefc3ac5c08f2428f9e3150d77eef9251311",
    "storage_source_sha256": "6e5803f9e9c4b2ae8b6b44b704d1ec4c25beb24a9669c018241327d08aba26a4",
}

# This hash was exercised as deployed WASM, not attributed to an unverified
# upstream source commit. The receipt contains four complete local-fork epochs.
RUNTIME_483 = {
    "method": "deployed-wasm-controlled-fork",
    "spec_version": 483,
    "runtime_hash": "1016a0f8c23aa1cfb88a37c723fd0440bfe7b8141490b77ae793dfd7be7540eb",
    "wasm_bytes": 2_670_004,
    "genesis_hash": "8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105",
    "origin_block": "0x3bfc41da6ab5ce1bf445d9af41f6cebcdae1599eee71686af7efe5ffb8cf238b",
    "receipt": "runtime_reviews/testnet-483-burn.json",
    "receipt_sha256": "6e1e38f8d19333aff66237bac281def451ac66e1e7d952970eec6379b3d5e621",
    "source_commit_verified": False,
    "scope": "native owner-target miner incentive burn; not signed publication acceptance",
}

RUNTIME_484 = {
    "method": "deployed-wasm-controlled-fork",
    "spec_version": 484,
    "runtime_hash": "8ba7928018328ba9e08d6089b0d92c841967801733ff4cd3c4965f105881e681",
    "wasm_bytes": 2_677_869,
    "genesis_hash": "8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105",
    "origin_block": "0x75da69788584f9fe9b65cbe772e57ae4419bceda339896785c07ab7f587bfbc7",
    "receipt": "runtime_reviews/testnet-484-burn.json",
    "receipt_sha256": "85ec1532e4002fa5238239df5dacc006698082b998cedc1a1de99fa876fb48f2",
    "source_commit_verified": False,
    "scope": "native owner-target miner incentive burn; not signed publication acceptance",
}


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    value_dict = getattr(value, "value", None)
    if value_dict is not None:
        return _plain(value_dict)
    return str(value)


def _burn(value: Any) -> bool:
    value = _plain(value)
    if value == "Burn":
        return True
    return isinstance(value, Mapping) and set(value) == {"Burn"}


def _runtime_hash(chain: Any, block: int) -> str:
    raw = chain.subtensor._client._substrate.raw
    code = chain.subtensor._call(
        raw.rpc_request("state_getStorage", ["0x3a636f6465", chain.finalized_block_hash(block)])
    )
    try:
        return hashlib.sha256(bytes.fromhex(str(code).removeprefix("0x"))).hexdigest()
    except ValueError as error:
        raise ValueError("runtime :code is not hex") from error


def verify_runtime_481_release(*, fetch: Any = urlopen) -> dict[str, Any]:
    """Fetch no files to disk; prove the official release asset equals its pinned code hash."""
    with fetch(RUNTIME_481["wasm_url"], timeout=30) as response:
        wasm = response.read()
    if (
        len(wasm) != RUNTIME_481["wasm_bytes"]
        or hashlib.sha256(wasm).hexdigest() != RUNTIME_481["runtime_hash"]
    ):
        raise ValueError("official v481 release asset differs from pinned runtime")
    return dict(RUNTIME_481)


def qualify_fallback(
    chain: Any,
    *,
    genesis_hash: str,
    runtime_hash: str,
    finalized_block: int | None = None,
) -> dict[str, Any]:
    """Read one finalized snapshot and bind a burn target without assuming UID 0."""
    _hash(genesis_hash, "genesis hash")
    _hash(runtime_hash, "runtime hash")
    if getattr(chain, "network", None) != "test":
        raise ValueError("native fallback qualification is testnet only")
    block = chain.finalized_height() if finalized_block is None else finalized_block
    if type(block) is not int or block < 0 or block > chain.finalized_height():
        raise ValueError("fallback proof block must be finalized")
    raw = chain.subtensor._client._substrate.raw
    actual_genesis = str(chain.subtensor._call(raw.get_block_hash(0))).removeprefix("0x")
    if actual_genesis != genesis_hash:
        raise ValueError("fallback proof genesis differs from configured network")
    actual_runtime = _runtime_hash(chain, block)
    if actual_runtime != runtime_hash:
        raise ValueError("fallback runtime differs from the requested runtime hash")
    if actual_runtime == RUNTIME_481["runtime_hash"]:
        # Preserve the exact historical v1 proof for archive replay.
        schema = "forge/launch-native-fallback-proof/v1"
        review = {"runtime_source_proof": dict(RUNTIME_481)}
    else:
        reviewed = next(
            (item for item in (RUNTIME_483, RUNTIME_484) if item["runtime_hash"] == actual_runtime),
            None,
        )
        if reviewed is None:
            raise ValueError(
                "fallback runtime has no reviewed burn semantics for its exact code hash"
            )
        if actual_genesis != reviewed["genesis_hash"]:
            raise ValueError("reviewed runtime belongs to a different testnet genesis")
        schema = "forge/launch-native-fallback-proof/v2"
        review = {"runtime_review": dict(reviewed)}
    at = chain.subtensor.at(block)
    recycle_or_burn = at.query(storage.SubtensorModule.RecycleOrBurn, [chain.netuid])
    owner_hotkey = at.query(storage.SubtensorModule.SubnetOwnerHotkey, [chain.netuid])
    owner_hotkey = str(owner_hotkey)
    if not owner_hotkey or owner_hotkey == "None" or not _burn(recycle_or_burn):
        raise ValueError("runtime does not route the subnet owner target to burn")
    uids = chain.uid_snapshot(block)
    target_uid = next((uid for uid, hotkey in uids.items() if hotkey == owner_hotkey), None)
    if target_uid is None:
        raise ValueError("subnet owner hotkey has no UID at the proof block")
    body = {
        "schema": schema,
        "network": chain.network,
        "netuid": chain.netuid,
        "genesis_hash": actual_genesis,
        "runtime_hash": actual_runtime,
        "finalized_block": block,
        "finalized_block_hash": chain.finalized_block_hash(block),
        "target_hotkey": owner_hotkey,
        "target_uid": target_uid,
        "storage": {
            "RecycleOrBurn": _plain(recycle_or_burn),
            "SubnetOwnerHotkey": _plain(owner_hotkey),
            "uid_snapshot": {str(uid): hotkey for uid, hotkey in uids.items()},
        },
        **review,
    }
    return {**body, "proof_hash": hash_canonical(schema, body)}


def verify_fallback(
    chain: Any, proof: Mapping[str, Any], *, genesis_hash: str, runtime_hash: str
) -> bool:
    """Re-read the archived finalized snapshot; arbitrary booleans cannot grant authority."""
    if (
        not isinstance(proof, Mapping)
        or proof.get("schema")
        not in {"forge/launch-native-fallback-proof/v1", "forge/launch-native-fallback-proof/v2"}
    ):
        return False
    body = {key: value for key, value in proof.items() if key != "proof_hash"}
    if proof.get("proof_hash") != hash_canonical(str(proof["schema"]), body):
        return False
    try:
        actual = qualify_fallback(
            chain,
            genesis_hash=genesis_hash,
            runtime_hash=runtime_hash,
            finalized_block=proof["finalized_block"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return actual == dict(proof)
