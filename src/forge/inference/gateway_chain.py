"""Read-only finalized-chain adapter for gateway mechanism 1.

Nothing here constructs or submits a mutation.  It records the signed master
registry anchor separately from the later block at which the latest publisher
commitment and active UID mapping are observed.
"""

from __future__ import annotations

from typing import Any

from forge.chain import BittensorChain
from forge.inference.chain import snapshot
from forge.inference.gateway_epoch import canonical_block_hash, commitment, verify_master


def _m1_state(chain: BittensorChain) -> dict[str, Any]:
    if chain.network != "test" or chain.netuid != 578:
        raise ValueError("gateway_chain_namespace")
    state = snapshot(chain)
    if state["mechanism"] != 1 or state["mechanismCount"] != 2:
        raise ValueError("gateway_chain_m1_not_ready")
    return state


def master_preflight(
    chain: BittensorChain, policy: dict[str, Any], publisher_hotkey: str
) -> dict[str, Any]:
    """Return the exact finalized inputs required before master signing."""
    state = _m1_state(chain)
    if publisher_hotkey != policy.get("masterPublisherHotkey"):
        raise ValueError("gateway_master_publisher")
    if publisher_hotkey == policy.get("m0PublisherHotkey"):
        raise ValueError("gateway_m0_master_separation")
    try:
        uid = next(uid for uid, hotkey in state["uids"].items() if hotkey == publisher_hotkey)
    except StopIteration as exc:
        raise ValueError("gateway_master_not_registered") from exc
    return {
        "schema": "attelier/gateway-master-preflight/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "finalizedRegistryBlock": state["block"],
        "finalizedRegistryBlockHash": canonical_block_hash(state["blockHash"]),
        "mechanismCount": state["mechanismCount"],
        "publisherHotkey": publisher_hotkey,
        "publisherUid": int(uid),
        "m0PublisherHotkey": policy["m0PublisherHotkey"],
    }


def live_anchor(
    chain: BittensorChain, master_envelope: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind one verified master to latest-per-hotkey commitment and current UIDs."""
    master = verify_master(master_envelope, policy)
    if (
        canonical_block_hash(chain.finalized_block_hash(master["finalizedRegistryBlock"]))
        != master["finalizedRegistryBlockHash"]
    ):
        raise ValueError("gateway_master_registry_anchor_unknown")
    state = _m1_state(chain)
    publisher = master["publisherHotkey"]
    if publisher not in state["uids"].values():
        raise ValueError("gateway_master_publisher_not_registered")
    latest = chain.raw_commitment(publisher, state["block"])
    if latest is None:
        raise ValueError("gateway_master_commitment_missing")
    commitment_block, raw_commitment = latest
    if raw_commitment != commitment(master_envelope):
        raise ValueError("gateway_master_commitment_mismatch")
    anchor = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "finalizedRegistryBlock": master["finalizedRegistryBlock"],
        "finalizedRegistryBlockHash": master["finalizedRegistryBlockHash"],
        "observedBlock": state["block"],
        "observedBlockHash": canonical_block_hash(state["blockHash"]),
        "commitmentBlock": commitment_block,
        "publisherHotkey": publisher,
        "rawCommitment": raw_commitment,
        "m0PublisherHotkey": policy["m0PublisherHotkey"],
    }
    metagraph = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "observedBlock": state["block"],
        "observedBlockHash": canonical_block_hash(state["blockHash"]),
        "uids": {hotkey: int(uid) for uid, hotkey in state["uids"].items()},
    }
    safety = {
        "schema": "attelier/gateway-light-preflight/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "mechanismCount": state["mechanismCount"],
        "finalizedRegistryBlock": master["finalizedRegistryBlock"],
        "finalizedRegistryBlockHash": master["finalizedRegistryBlockHash"],
        "observedBlock": state["block"],
        "observedBlockHash": canonical_block_hash(state["blockHash"]),
        "commitmentBlock": commitment_block,
        "publisherHotkey": publisher,
        "m0PublisherHotkey": policy["m0PublisherHotkey"],
        "commitmentMatches": True,
    }
    return anchor, metagraph, safety
