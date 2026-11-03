#!/usr/bin/env python3
"""Project a finalized SN126 registry for the permissioned prelaunch gateway."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.inference.prelaunch import project_prelaunch_registry, read_json


def live_prelaunch_snapshot(weights_validator_uid: int | None = None) -> dict[str, Any]:
    """Read one finalized Finney SN126 snapshot; no wallet or mutation path exists."""
    from bittensor import storage

    from forge.chain import BittensorChain

    chain = BittensorChain("finney", 126)
    try:
        block = chain.finalized_height()
        block_hash = chain.finalized_block_hash(block)
        raw = chain.subtensor._client._substrate.raw
        genesis = str(chain.subtensor._call(raw.get_block_hash(0)))
        at = chain.subtensor.at(block)

        def query(name: str) -> Any:
            value = at.query(getattr(storage.SubtensorModule, name), [126])
            return getattr(value, "value", value)

        owners = chain.owner_snapshot(block)
        snapshot = {
            "schema": "forge/inference-registry-snapshot/v1",
            "network": "finney",
            "genesis": genesis,
            "netuid": 126,
            "mechanism": 0,
            "finalized": True,
            "mechanismCount": int(query("MechanismCountCurrent")),
            "block": block,
            "blockHash": block_hash,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "subnetOwnerColdkey": str(query("SubnetOwner")),
            "owners": {
                str(uid): {"hotkey": hotkey, "coldkey": coldkey}
                for uid, (hotkey, coldkey) in owners.items()
            },
        }
        if weights_validator_uid is not None:
            # Optional observation must not stop identity/registration refresh.
            # All values share the registry's finalized block. LastUpdate can be
            # the commit block; this does not attribute a reveal to any feed.
            try:
                weights = at.query(storage.SubtensorModule.Weights, [126, weights_validator_uid])
                weights = getattr(weights, "value", weights)
                incentives = query("Incentive")
                updates = query("LastUpdate")
                runtime = chain.subtensor._call(
                    raw.rpc_request("state_getRuntimeVersion", [block_hash])
                )
                snapshot["chainObservation"] = {
                    "schema": "attelier/finalized-weights-observation/v1",
                    "status": "observed",
                    "block": block,
                    "hash": block_hash,
                    "observedAt": snapshot["timestamp"],
                    "runtimeVersion": int(runtime["specVersion"]),
                    "validatorUid": weights_validator_uid,
                    "validatorHotkey": owners[weights_validator_uid][0],
                    "lastUpdateBlock": int(updates[weights_validator_uid]),
                    "weightsU16": [[int(uid), int(weight)] for uid, weight in weights],
                    "incentivesU16": [int(value) for value in incentives],
                    "windowAttribution": "not_observed",
                }
            except Exception:
                snapshot["chainObservation"] = {"status": "unavailable"}
        return snapshot
    finally:
        chain.close()


def atomic_replace(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ValueError("registry_output_write_failed") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--simulation-registry-out", type=Path)
    parser.add_argument("--dashboard-chain-out", type=Path)
    parser.add_argument("--weights-validator-uid", type=int)
    parser.add_argument(
        "--chain-snapshot", type=Path, help="Local finalized snapshot fixture; never a live proof"
    )
    args = parser.parse_args()
    if args.weights_validator_uid is not None and not 0 <= args.weights_validator_uid <= 65_535:
        parser.error("invalid weights validator UID")
    if os.environ.get("H3_SUBNET_MODE") != "prelaunch":
        parser.error("H3_SUBNET_MODE=prelaunch is required")
    snapshot = (
        read_json(args.chain_snapshot, allow_numeric_keys=True)
        if args.chain_snapshot
        else live_prelaunch_snapshot(args.weights_validator_uid)
    )
    observation = snapshot.pop("chainObservation", None)
    registry, simulation = project_prelaunch_registry(read_json(args.config), snapshot)
    atomic_replace(args.out, registry)
    if args.simulation_registry_out:
        atomic_replace(args.simulation_registry_out, simulation)
    if args.dashboard_chain_out:
        dashboard = {
            "network": "finney",
            "netuid": 126,
            "mechanism": 0,
            "observedAt": snapshot["timestamp"],
            "block": snapshot["block"],
            "hash": snapshot["blockHash"],
            "runtime": None,
            "mechanismCount": 1,
            "weightsStatus": "not_observed",
            "weightsBlock": None,
            "observedSplit": {
                "optimization": None,
                "inference": None,
                "block": snapshot["block"],
            },
        }
        if observation is not None:
            valid = (
                isinstance(observation, dict)
                and observation.get("schema") == "attelier/finalized-weights-observation/v1"
                and observation.get("status") == "observed"
                and observation.get("block") == snapshot["block"]
                and observation.get("hash") == snapshot["blockHash"]
                and observation.get("observedAt") == snapshot["timestamp"]
                and observation.get("windowAttribution") == "not_observed"
                and isinstance(observation.get("validatorUid"), int)
                and snapshot["owners"].get(str(observation["validatorUid"]), {}).get("hotkey")
                == observation.get("validatorHotkey")
                and isinstance(observation.get("lastUpdateBlock"), int)
                and 0 <= observation["lastUpdateBlock"] <= snapshot["block"]
                and isinstance(observation.get("weightsU16"), list)
                and all(
                    isinstance(pair, list)
                    and len(pair) == 2
                    and all(type(value) is int and 0 <= value <= 65_535 for value in pair)
                    for pair in observation["weightsU16"]
                )
                and isinstance(observation.get("incentivesU16"), list)
                and all(
                    type(value) is int and 0 <= value <= 65_535
                    for value in observation["incentivesU16"]
                )
            )
            dashboard["weightsStatus"] = "observed" if valid else "observation_unavailable"
            if valid:
                dashboard["weightsBlock"] = snapshot["block"]
                dashboard["weightsObservation"] = observation
                runtime_version = observation.get("runtimeVersion")
                if type(runtime_version) is int and runtime_version > 0:
                    dashboard["runtime"] = runtime_version
        atomic_replace(args.dashboard_chain_out, dashboard)


if __name__ == "__main__":
    main()
