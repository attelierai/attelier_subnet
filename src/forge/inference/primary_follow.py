"""Read the primary validator's published M0 row on Finney without provider credentials."""

import time
from typing import Any

from forge.inference.light_publication import (
    BURN_OWNER_COLDKEY,
    NETUID,
    NETWORK,
    U16_MAX,
    PublicationError,
)
from forge.inference.prelaunch import GENESIS, HOUR_MS

PRIMARY_HOTKEY = "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD"
API_ALLOCATION_PUBLIC_KEY = "UksNGcpVoi0yxAgZbO6zUZ1xom5muxenTnUSXuzfNH4="


def primary_snapshot(chain: Any) -> tuple[dict, dict[int, int]]:
    """Sample immutable raw weights once per network rate-limit interval."""
    from bittensor import storage

    if (chain.network, chain.netuid) != (NETWORK, NETUID):
        raise PublicationError("light_primary_namespace")
    tip = chain.finalized_height()
    raw = chain.subtensor._client._substrate.raw
    if str(chain.subtensor._call(raw.get_block_hash(0))) != GENESIS:
        raise PublicationError("light_primary_genesis")
    head = chain.subtensor.at(tip)
    rate = int(head.query(storage.SubtensorModule.WeightsSetRateLimit, [NETUID]))
    if rate != 100:
        raise PublicationError("light_primary_rate_limit")
    block = tip - tip % rate
    at = chain.subtensor.at(block)
    moment = int(at.query(storage.Timestamp.Now))
    # Allow a full rate-limit interval plus normal finality lag.
    if not -60_000 <= int(time.time() * 1000) - moment <= (rate + 30) * 12_000:
        raise PublicationError("light_primary_stale_rpc")

    def query(item, args=None):
        value = at.query(getattr(storage.SubtensorModule, item), [NETUID] if args is None else args)
        return getattr(value, "value", value)

    owners = chain.owner_snapshot(block)
    primary = next((uid for uid, pair in owners.items() if pair[0] == PRIMARY_HOTKEY), None)
    if (
        primary is None
        or int(query("MechanismCountCurrent")) != 1
        or str(query("SubnetOwner")) != BURN_OWNER_COLDKEY
        or owners.get(0, (None, None))[1] != BURN_OWNER_COLDKEY
        or not query("ValidatorPermit")[primary]
    ):
        raise PublicationError("light_primary_identity")
    updated = int(query("LastUpdate")[primary])
    if not 0 < updated <= block or tip - updated > int(query("ActivityCutoff")):
        raise PublicationError("light_primary_row_stale")
    pairs = query("Weights", [NETUID, primary])
    weights = {int(uid): int(weight) for uid, weight in pairs}
    if (
        not weights
        or len(weights) != len(pairs)
        or max(weights.values()) != U16_MAX
        or any(uid not in owners or not 0 <= weight <= U16_MAX for uid, weight in weights.items())
    ):
        raise PublicationError("light_primary_weights")
    prior = chain.owner_snapshot(updated)
    if any(prior.get(uid) != owners[uid] for uid in {*weights, primary}):
        raise PublicationError("light_primary_reassigned_uid")
    body = {
        "primaryValidatorHotkey": PRIMARY_HOTKEY,
        "finalizedBlockHash": chain.finalized_block_hash(block),
        "primaryLastUpdate": updated,
        "ledger": {"sequence": block, "window": {"startMs": moment - HOUR_MS, "endMs": moment}},
        "registry": {
            "subnetOwnerColdkey": BURN_OWNER_COLDKEY,
            "miners": [
                {"uid": uid, "hotkey": owners[uid][0], "coldkey": owners[uid][1]}
                for uid in sorted(weights)
                if uid != 0
            ],
        },
    }
    return body, weights


def follow_primary(chain, runner, publisher, *, once: bool, interval: int) -> None:
    from forge.inference.light import STATUS_SCHEMA, _atomic_write

    while True:
        try:
            body, weights = primary_snapshot(chain)
            evidence = {**body, "relativeWeights": [[uid, weights[uid]] for uid in sorted(weights)]}
            if publisher is None:
                _atomic_write(runner.state_dir / "live-dry-run.json", evidence)
            else:
                publisher.process(body, {"miners": []}, forced_weights=weights)
            _atomic_write(runner.state_dir / "primary-observation.json", evidence)
            _atomic_write(
                runner.state_dir / "status.json",
                {
                    "schema": STATUS_SCHEMA,
                    "status": "ready",
                    "sequence": body["ledger"]["sequence"],
                },
            )
        except Exception as error:
            reason = (
                str(error) if isinstance(error, PublicationError) else "light_primary_unavailable"
            )
            runner.unavailable(reason)
            print(reason, flush=True)
            if once:
                raise SystemExit(1) from error
        if once:
            return
        time.sleep(interval)
