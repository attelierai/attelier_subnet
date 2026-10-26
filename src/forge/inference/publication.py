"""Mechanism-1 publication with durable intent and separate revealed readback."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from forge.chain import BittensorChain
from forge.inference.chain import atomic_json, snapshot
from forge.inference.service import Validator


def _trace_scores_publishable(scores: dict[str, Any]) -> bool:
    """Defence in depth: service v3 evidence cannot be relabelled at publication."""
    trace = scores.get("trace")
    if not isinstance(trace, dict) or trace.get("mode") != "trace_trusted_render":
        return False
    window = trace.get("window")
    if not isinstance(window, dict) or not all(
        isinstance(window.get(key), (int, float)) for key in ("minCompleted", "minVerified")
    ):
        return False
    credited = [item for item in scores.get("miners", {}).values() if item.get("weight", 0) > 0]
    return bool(credited) and all(
        item.get("rewardEligible") is True
        and item.get("coverage", {}).get("verified", 0) >= window["minVerified"]
        and item.get("coverage", {}).get("completed", 0) >= window["minCompleted"]
        for item in credited
    )


def _bytes(value: Any) -> bytes:
    return bytes.fromhex(value.removeprefix("0x")) if isinstance(value, str) else bytes(value)


def _hash(value: Any) -> str:
    return hashlib.blake2b(_bytes(value), digest_size=32).hexdigest()


def _attributes(event: dict[str, Any]) -> list[Any]:
    attributes = event.get("attributes", [])
    return list(attributes.values()) if isinstance(attributes, dict) else list(attributes)


def _intent_path(directory: Path, network: str, netuid: int, hotkey: str) -> Path:
    safe_hotkey = "".join(char for char in hotkey if char.isalnum() or char in "-_")
    return directory / "publication-intents" / f"{network}-{netuid}-m1-{safe_hotkey}.json"


def _claim_publication_intent(
    directory: Path, network: str, netuid: int, hotkey: str, receipt_path: Path
) -> tuple[Any, Path]:
    """Acquire the sole local M1 publisher slot and durably name its receipt.

    The pointer is deliberately retained after the OS lock is released. A missing,
    unreadable or unknown target fails closed, so changing output directories cannot
    bypass an ambiguous commit/reveal submission.
    """
    path = _intent_path(directory, network, netuid, hotkey)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Ownership of this open lock transfers to publish(), whose finally closes it.
    lock = open(path.with_suffix(".lock"), "a+", encoding="utf-8")  # noqa: SIM115
    os.chmod(path.with_suffix(".lock"), 0o600)
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValueError("Another inference publication is being prepared") from exc
    try:
        if path.exists():
            try:
                pointer = json.loads(path.read_text())
                target = Path(pointer["receiptPath"])
                receipt = json.loads(target.read_text())
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Existing publication intent is ambiguous; reconcile manually"
                ) from exc
            if not isinstance(pointer, dict) or not isinstance(receipt, dict):
                raise ValueError("Existing publication intent is not an object")
            if (
                pointer.get("network") != network
                or pointer.get("netuid") != netuid
                or pointer.get("mechanism") != 1
                or pointer.get("validatorHotkey") != hotkey
                or receipt.get("schema") != "forge/inference-publication/v1"
            ):
                raise ValueError("Existing publication intent namespace is invalid")
            if receipt.get("status") not in ("REVEALED", "FAILED_CHAIN_PROVEN"):
                raise ValueError(
                    "Existing pending inference publication; reconcile before any new submission"
                )
        return lock, path
    except Exception:
        lock.close()
        raise


def _write_publication_intent(
    path: Path, network: str, netuid: int, hotkey: str, receipt_path: Path
) -> None:
    payload = {
        "schema": "forge/inference-publication-intent/v1",
        "network": network,
        "netuid": netuid,
        "mechanism": 1,
        "validatorHotkey": hotkey,
        "receiptPath": str(receipt_path.resolve()),
    }
    atomic_json(path, payload)


def publish(
    chain: BittensorChain, wallet: Any, validator: Validator, batch: str, receipt_path: Path
) -> dict[str, Any]:
    if receipt_path.exists():
        raise ValueError("Existing publication intent; reconcile before any new submission")
    state = snapshot(chain)
    scores = validator.scores(batch)
    if (
        scores["source"] != "live"
        or not scores["publishable"]
        or state["mechanismCount"] != 2
        or state["network"] != "test"
        or state["netuid"] != scores["netuid"]
    ):
        raise ValueError("No publishable real testnet service evidence")
    if (
        scores.get("schema") == "forge/inference-scores/v3"
        and not _trace_scores_publishable(scores)
    ):
        raise ValueError("Trace score evidence is not qualified for publication")
    if wallet.hotkey.ss58_address not in state["committee"]:
        raise ValueError("Validator has no finalized permit/stake")
    if any(state["uids"].get(str(m["uid"])) != m["hotkey"] for m in scores["miners"].values()):
        raise ValueError("Miner registration changed")
    intent_lock, intent_path = _claim_publication_intent(
        validator.directory,
        state["network"],
        state["netuid"],
        wallet.hotkey.ss58_address,
        receipt_path,
    )
    try:
        return _publish_claimed(chain, wallet, validator, scores, state, receipt_path, intent_path)
    finally:
        intent_lock.close()


def _publish_claimed(
    chain: BittensorChain,
    wallet: Any,
    validator: Validator,
    scores: dict[str, Any],
    state: dict[str, Any],
    receipt_path: Path,
    intent_path: Path,
) -> dict[str, Any]:
    weights = {m["uid"]: m["weight"] for m in scores["miners"].values() if m["weight"] > 0}
    validator_uid = next(
        int(uid) for uid, hotkey in state["uids"].items() if hotkey == wallet.hotkey.ss58_address
    )
    from bittensor.intents.weights import SetWeights
    from bittensor.settings import DEFAULT_ERA_PERIOD
    from bittensor.signing import resolve_signer

    uids = sorted(weights)
    plan = chain.subtensor.plan(
        SetWeights(netuid=chain.netuid, uids=uids, weights=[weights[u] for u in uids], mechid=1),
        wallet,
    )
    if not plan.ok or plan.signer != "hotkey" or plan.signer_address != wallet.hotkey.ss58_address:
        raise ValueError("SDK weight plan or signer is invalid")
    substrate = chain.subtensor._client._substrate
    call = chain.subtensor._call(substrate.decode_scale("Call", plan.call.data))
    args = {arg["name"]: arg["value"] for arg in call["call_args"]}
    if (
        call["call_module"] != "SubtensorModule"
        or call["call_function"]
        not in ("commit_timelocked_mechanism_weights", "set_mechanism_weights")
        or args.get("netuid") != chain.netuid
        or args.get("mecid") != 1
    ):
        raise ValueError("Prepared call does not explicitly address inference mechanism 1")
    receipt = {
        "schema": "forge/inference-publication/v1",
        "network": "test",
        "netuid": chain.netuid,
        "mechanism": 1,
        "status": "SUBMITTING",
        "validatorUid": validator_uid,
        "validatorHotkey": wallet.hotkey.ss58_address,
        "atBlock": state["block"],
        "runtimeHash": state["runtimeHash"],
        "scores": scores,
        "requestedWeights": weights,
        "preparedCallHash": _hash(plan.call.data),
        "callFunction": call["call_function"],
        "optimizationWeightsBefore": chain.subtensor.at(state["block"]).read(
            "weights", netuid=chain.netuid, mechid=0
        ),
    }
    if "commit" in args:
        receipt.update(
            ciphertextHash=_hash(args["commit"]),
            revealRound=args["reveal_round"],
            commitRevealVersion=args["commit_reveal_version"],
        )
    # The call is now exact and locally witnessed.  Persist the single-writer
    # pointer before its receipt or any chain side effect, never during preflight.
    _write_publication_intent(
        intent_path, state["network"], state["netuid"], wallet.hotkey.ss58_address, receipt_path
    )
    atomic_json(receipt_path, receipt)
    # Submit the exact durably witnessed SDK call, never a newly encrypted re-plan.
    result = chain.subtensor._call(
        substrate.submit(
            plan.call,
            resolve_signer(wallet, "hotkey"),
            period=DEFAULT_ERA_PERIOD,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    )
    if not result.success:
        receipt.update(status="FAILED", extrinsicId=str(result.extrinsic_id))
        atomic_json(receipt_path, receipt)
        raise RuntimeError("Inference weights rejected; reconcile the retained receipt")
    receipt.update(status="SUBMITTED", extrinsicId=str(result.extrinsic_id))
    atomic_json(receipt_path, receipt)
    return receipt


def reconcile(chain: BittensorChain, receipt: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    if chain.network != "test" or receipt["netuid"] != chain.netuid or receipt["mechanism"] != 1:
        raise ValueError("Reconciliation requires the original testnet intent")
    block = chain.finalized_height()
    if not receipt.get("extrinsicId"):
        if receipt.get("status") != "SUBMITTING" or not receipt.get("preparedCallHash"):
            raise ValueError("No durable prepared call to recover")
        start = receipt.get("submissionScannedThrough", receipt["atBlock"])
        for height in range(start + 1, min(block, start + 120) + 1):
            candidates = [
                index
                for index, extrinsic in enumerate(chain.subtensor.block_info(height).extrinsics)
                if extrinsic.get("address") == receipt["validatorHotkey"]
                and extrinsic.get("call", {}).get("call_hash", "").removeprefix("0x")
                == receipt["preparedCallHash"]
            ]
            if len(candidates) > 1:
                raise ValueError(
                    "Prepared call has multiple inclusions; operator reconciliation required"
                )
            receipt["submissionScannedThrough"] = height
            if candidates:
                receipt.update(
                    status="SUBMITTED",
                    extrinsicId=f"{height}-{candidates[0]:04d}",
                    recoveredInclusion=True,
                )
                break
        atomic_json(receipt_path, receipt)
        if not receipt.get("extrinsicId"):
            return receipt  # Absence at this head never authorizes another broadcast.
    included = int(receipt["extrinsicId"].split("-", 1)[0])
    if block < included:
        return receipt
    index = int(receipt["extrinsicId"].split("-", 1)[1])
    submission_hash = chain.finalized_block_hash(included)
    if receipt.get("submissionBlockHash", submission_hash) != submission_hash:
        raise ValueError("Finalized submission block changed")
    # Persist finalized evidence while it is available. A later automatic reveal
    # can occur after this RPC has pruned the commit block's state/events.
    extrinsic = receipt.get("submissionExtrinsic")
    if extrinsic is None:
        extrinsic = chain.subtensor.block_info(included).extrinsics[index]
    call = extrinsic["call"]
    args = {arg["name"]: arg["value"] for arg in call["call_args"]}
    if (
        extrinsic["address"] != receipt["validatorHotkey"]
        or call["call_module"] != "SubtensorModule"
        or call["call_function"]
        not in ("commit_timelocked_mechanism_weights", "set_mechanism_weights")
        or args.get("netuid") != chain.netuid
        or args.get("mecid") != 1
        or call.get("call_hash", "").removeprefix("0x") != receipt["preparedCallHash"]
    ):
        raise ValueError("Finalized extrinsic does not belong to this mechanism and validator")
    from bittensor import storage

    events = receipt.get("submissionEvents")
    if events is None:
        events = [
            event
            for event in chain.subtensor.at(included).query(storage.System.Events)
            if event.get("extrinsic_idx") == index
        ]
    if not any(
        event.get("module_id") == "System" and event.get("event_id") == "ExtrinsicSuccess"
        for event in events
    ):
        raise ValueError("No attributed finalized success event")
    receipt["extrinsicHash"] = extrinsic["extrinsic_hash"]
    receipt["submissionBlockHash"] = submission_hash
    receipt["callFunction"] = call["call_function"]
    from bittensor.settings import GLOBAL_MAX_SUBNET_COUNT

    mechanism_index = GLOBAL_MAX_SUBNET_COUNT + chain.netuid
    timelocked = call["call_function"] == "commit_timelocked_mechanism_weights"
    if timelocked:
        if (
            _hash(args["commit"]) != receipt["ciphertextHash"]
            or args["reveal_round"] != receipt["revealRound"]
            or args["commit_reveal_version"] != receipt["commitRevealVersion"]
        ):
            raise ValueError("Finalized ciphertext differs from the durable prepared call")
        expected = [
            receipt["validatorHotkey"],
            mechanism_index,
            "0x" + receipt["ciphertextHash"],
            receipt["revealRound"],
        ]
        if (
            sum(
                event.get("module_id") == "SubtensorModule"
                and event.get("event_id") == "TimelockedWeightsCommitted"
                and _attributes(event) == expected
                for event in events
            )
            != 1
        ):
            raise ValueError("No unique commit event binds the prepared ciphertext")
    receipt["submissionExtrinsic"] = extrinsic
    receipt["submissionEvents"] = events
    snapshot_uids = chain.uid_snapshot(block)
    if snapshot_uids.get(receipt["validatorUid"]) != receipt["validatorHotkey"]:
        raise ValueError("Validator UID identity changed")
    observed = chain.subtensor.at(block).read("weights", netuid=chain.netuid, mechid=1)
    actual = observed.get(receipt["validatorUid"], {})
    requested = {int(k): float(v) for k, v in receipt["requestedWeights"].items()}
    total = sum(requested.values())
    matches = set(actual) == set(requested) and all(
        abs(actual[uid] - weight / total) <= 2 / 65535 for uid, weight in requested.items()
    )
    updates = chain.subtensor.at(block).query(storage.SubtensorModule.LastUpdate, [mechanism_index])
    updated = int(updates[receipt["validatorUid"]])
    receipt.update(
        observedWeights=actual,
        observedAtBlock=block,
        observedAtHash=chain.finalized_block_hash(block),
        lastUpdate=updated,
        checkedAt=time.time(),
    )
    # CR LastUpdate is the commit block. Prove the unique queued ciphertext was
    # consumed by an automatic reveal, then check the weights at that exact block.
    revealed = receipt.get("weightsBlock")
    if not timelocked:
        revealed = included if matches and updated == included else None
    elif revealed is None:
        for height in range(
            receipt.get("scannedThrough", included) + 1,
            min(block, receipt.get("scannedThrough", included) + 120) + 1,
        ):
            block_events = chain.subtensor.at(height).query(storage.System.Events)
            reveal_events = [
                e
                for e in block_events
                if e.get("module_id") == "SubtensorModule"
                and e.get("event_id") == "TimelockedWeightsRevealed"
                and _attributes(e) == [mechanism_index, receipt["validatorHotkey"]]
            ]
            if reveal_events:
                sets = [
                    e
                    for e in block_events
                    if e.get("module_id") == "SubtensorModule"
                    and e.get("event_id") == "WeightsSet"
                    and _attributes(e) == [mechanism_index, receipt["validatorUid"]]
                ]
                queue = chain.subtensor.at(height - 1).query_map(
                    storage.SubtensorModule.TimelockedWeightCommits, [mechanism_index]
                )
                own = [
                    entry
                    for _, entries in queue
                    for entry in (entries or [])
                    if entry[0] == receipt["validatorHotkey"]
                ]
                if len(own) == len(reveal_events) == len(sets) == 1:
                    _, commit_block, ciphertext, reveal_round = own[0]
                    row = (
                        chain.subtensor.at(height)
                        .read("weights", netuid=chain.netuid, mechid=1)
                        .get(receipt["validatorUid"], {})
                    )
                    if (
                        int(commit_block) == included
                        and _hash(ciphertext) == receipt["ciphertextHash"]
                        and int(reveal_round) == receipt["revealRound"]
                        and set(row) == set(requested)
                        and all(
                            abs(row[uid] - weight / total) <= 2 / 65535
                            for uid, weight in requested.items()
                        )
                    ):
                        revealed = height
                        receipt["revealBlockHash"] = chain.finalized_block_hash(height)
                        receipt["revealedWeights"] = row
                        break
            receipt["scannedThrough"] = height
    receipt["status"] = "REVEALED" if revealed is not None else "REVEAL_PENDING"
    if revealed is not None:
        receipt["weightsBlock"] = revealed
    receipt["optimizationWeightsAfter"] = chain.subtensor.at(block).read(
        "weights", netuid=chain.netuid, mechid=0
    )
    atomic_json(receipt_path, receipt)
    return receipt
