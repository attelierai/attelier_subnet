"""Finalized testnet registration and mechanism configuration, separate from optimization."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.chain import BittensorChain


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf8") as output:
        os.chmod(temporary, 0o600)
        json.dump(value, output, sort_keys=True, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def snapshot(chain: BittensorChain) -> dict[str, Any]:
    from bittensor import storage

    if chain.network != "test":
        raise ValueError("Inference rollout is testnet only")
    block = chain.finalized_height()
    block_hash = chain.finalized_block_hash(block)
    raw = chain.subtensor._client._substrate.raw
    runtime = chain.subtensor._call(raw.rpc_request("state_getRuntimeVersion", [block_hash]))
    code = chain.subtensor._call(raw.rpc_request("state_getStorage", ["0x3a636f6465", block_hash]))
    if runtime.get("specName") != "node-subtensor":
        raise ValueError("Unexpected runtime")
    at = chain.subtensor.at(block)

    def query(name: str) -> Any:
        return at.query(getattr(storage.SubtensorModule, name), [chain.netuid])

    uids = chain.uid_snapshot(block)
    return {
        "schema": "forge/inference-registry/v1",
        "network": "test",
        "netuid": chain.netuid,
        "mechanism": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "block": block,
        "blockHash": block_hash,
        "genesisHash": str(chain.subtensor._call(raw.get_block_hash(0))),
        "runtime": int(runtime["specVersion"]),
        "runtimeHash": hashlib.sha256(bytes.fromhex(code.removeprefix("0x"))).hexdigest(),
        "mechanismCount": int(query("MechanismCountCurrent")),
        "emissionSplit": query("MechanismEmissionSplit"),
        "maxAllowedUids": int(query("MaxAllowedUids")),
        "registeredCount": int(query("SubnetworkN")),
        "owner": str(query("SubnetOwner")),
        "uids": {str(uid): hotkey for uid, hotkey in uids.items()},
        "committee": chain.committee_snapshot(block),
        "commitRevealEnabled": bool(query("CommitRevealWeightsEnabled")),
        "tempo": int(query("Tempo")),
        "weightsRateLimit": int(query("WeightsSetRateLimit")),
    }


def configuration_plan(state: dict[str, Any]) -> dict[str, Any]:
    if state["network"] != "test" or state["mechanismCount"] not in (1, 2):
        raise ValueError("Only one-to-two testnet mechanism activation is supported")
    target_max = min(128, state["maxAllowedUids"])
    if state["registeredCount"] > target_max or len(state["uids"]) > target_max:
        raise ValueError(
            "Configuration would remove participants; automatic trimming is prohibited"
        )
    return {
        "schema": "forge/inference-configuration-plan/v1",
        "network": "test",
        "netuid": state["netuid"],
        "block": state["block"],
        "blockHash": state["blockHash"],
        "runtimeHash": state["runtimeHash"],
        "owner": state["owner"],
        "previousMechanismCount": state["mechanismCount"],
        "previousMaxAllowedUids": state["maxAllowedUids"],
        "targetMechanismCount": 2,
        "targetMaxAllowedUids": target_max,
        "targetEmissionSplit": None,
        "meaning": "even 50/50 split, symbolic testnet",
        "preservedUids": state["uids"],
    }


def configure(chain: BittensorChain, wallet: Any, receipt_path: Path) -> dict[str, Any]:
    """One atomic owner transaction; no registration removal and no retry.

    Caller owns wallet access and must explicitly select execution. A durable
    SUBMITTING receipt is written before signing; an interrupted attempt requires
    reconciliation, never a blind resubmission.
    """
    from bittensor import calls

    if receipt_path.exists():
        raise ValueError("Existing configuration intent; reconcile it before another submission")
    before = snapshot(chain)
    optimization_before = chain.subtensor.at(before["block"]).read(
        "weights", netuid=chain.netuid, mechid=0
    )
    plan = configuration_plan(before)
    if wallet.coldkeypub.ss58_address != plan["owner"]:
        raise ValueError("Wallet is not the finalized subnet owner")
    actions = []
    if before["maxAllowedUids"] != plan["targetMaxAllowedUids"]:
        actions.append(
            calls.AdminUtils.sudo_set_max_allowed_uids(chain.netuid, plan["targetMaxAllowedUids"])
        )
    if before["mechanismCount"] != 2:
        actions.append(calls.AdminUtils.sudo_set_mechanism_count(chain.netuid, 2))
    if before["mechanismCount"] != 2 or before["emissionSplit"] is not None:
        actions.append(calls.AdminUtils.sudo_set_mechanism_emission_split(chain.netuid, None))
    receipt = {
        "plan": plan,
        "optimizationWeightsBefore": optimization_before,
        "status": "NO_CHANGE" if not actions else "SUBMITTING",
    }
    atomic_json(receipt_path, receipt)
    if not actions:
        return receipt
    substrate = chain.subtensor._client._substrate
    encoded = [chain.subtensor._call(substrate.compose(action)) for action in actions]
    result = chain.subtensor.submit_call(calls.Utility.batch_all(encoded), wallet, signer="coldkey")
    receipt.update(
        status="INCLUDED" if result.success else "FAILED", extrinsicId=str(result.extrinsic_id)
    )
    atomic_json(receipt_path, receipt)
    if not result.success:
        raise RuntimeError("Mechanism configuration failed; receipt retained")
    included = int(str(result.extrinsic_id).split("-", 1)[0])
    deadline = time.monotonic() + 90
    while chain.finalized_height() < included:
        if time.monotonic() >= deadline:
            raise TimeoutError("Configuration finality pending; do not resubmit")
        time.sleep(2)
    after = snapshot(chain)
    optimization_after = chain.subtensor.at(after["block"]).read(
        "weights", netuid=chain.netuid, mechid=0
    )
    receipt.update(
        optimizationWeightsAfter=optimization_after,
        optimizationWeightsUnchanged=optimization_before == optimization_after,
    )
    if (
        after["mechanismCount"] != 2
        or after["emissionSplit"] is not None
        or after["maxAllowedUids"] != plan["targetMaxAllowedUids"]
        or any(after["uids"].get(uid) != hotkey for uid, hotkey in before["uids"].items())
    ):
        receipt.update(status="READBACK_MISMATCH", observed=after)
        atomic_json(receipt_path, receipt)
        raise RuntimeError("Finalized mechanism configuration differs from intent")
    receipt.update(status="FINALIZED", observed=after)
    atomic_json(receipt_path, receipt)
    return receipt
