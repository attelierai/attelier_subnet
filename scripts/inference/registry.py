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


def live_prelaunch_snapshot() -> dict[str, Any]:
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
        return {
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
    parser.add_argument(
        "--chain-snapshot", type=Path, help="Local finalized snapshot fixture; never a live proof"
    )
    args = parser.parse_args()
    if os.environ.get("H3_SUBNET_MODE") != "prelaunch":
        parser.error("H3_SUBNET_MODE=prelaunch is required")
    snapshot = (
        read_json(args.chain_snapshot, allow_numeric_keys=True)
        if args.chain_snapshot
        else live_prelaunch_snapshot()
    )
    registry, simulation = project_prelaunch_registry(read_json(args.config), snapshot)
    atomic_replace(args.out, registry)
    if args.simulation_registry_out:
        atomic_replace(args.simulation_registry_out, simulation)
    if args.dashboard_chain_out:
        atomic_replace(
            args.dashboard_chain_out,
            {
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
            },
        )


if __name__ == "__main__":
    main()
