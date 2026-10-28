"""Read-only finalized registry verification for the SN126 light validator."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from forge.chain import BittensorChain
from forge.inference.prelaunch import GENESIS

ARCHIVE_URL = "wss://archive.chain.opentensor.ai:443"


class RegistryError(ValueError):
    """The signed registry did not match an immutable, finalized chain read."""


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def verify_finalized_registry(
    registry: dict[str, Any],
    *,
    network: str = ARCHIVE_URL,
    chain_factory: Callable[[str, int], Any] = BittensorChain,
) -> None:
    """Match hash, identities, owner and mechanism against one final block.

    The injection point is for explicit read-only chain fixtures in tests. The
    production default has no wallet, signing, or publication path.
    """
    if not isinstance(registry, dict):
        raise RegistryError("light_registry_schema")
    block = registry.get("finalizedBlock")
    expected_hash = registry.get("finalizedBlockHash")
    if type(block) is not int or block <= 0 or not isinstance(expected_hash, str):
        raise RegistryError("light_registry_schema")
    chain = None
    try:
        # The selected node must retain the signed registry block's state;
        # a head-only endpoint cannot establish historical identities.
        chain = chain_factory(network, 126)
        if block > chain.finalized_height():
            raise RegistryError("light_registry_not_finalized")
        actual_hash = chain.finalized_block_hash(block).removeprefix("0x")
        if actual_hash != expected_hash.removeprefix("0x"):
            raise RegistryError("light_registry_block_hash")
        raw = chain.subtensor._client._substrate.raw
        genesis = str(chain.subtensor._call(raw.get_block_hash(0)))
        if genesis != GENESIS:
            raise RegistryError("light_registry_genesis")
        from bittensor import storage

        at = chain.subtensor.at(block)

        def query(name: str) -> Any:
            return _value(at.query(getattr(storage.SubtensorModule, name), [126]))

        if str(query("SubnetOwner")) != registry.get("subnetOwnerColdkey"):
            raise RegistryError("light_registry_owner")
        if int(query("MechanismCountCurrent")) != 1 or registry.get("mechanism") != 0:
            raise RegistryError("light_registry_mechanism")
        owners = chain.owner_snapshot(block)
        for miner in registry.get("miners", []):
            observed = owners.get(miner.get("uid")) if isinstance(miner, dict) else None
            if not isinstance(miner, dict) or observed != (
                miner.get("hotkey"),
                miner.get("coldkey"),
            ):
                raise RegistryError("light_registry_identity")
    except RegistryError:
        raise
    except Exception as error:
        raise RegistryError("light_registry_unavailable") from error
    finally:
        if chain is not None:
            with contextlib.suppress(Exception):
                chain.close()
