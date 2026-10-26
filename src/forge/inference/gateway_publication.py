"""Explicit receipt-backed M1 gateway commitment publication."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from forge.chain import BittensorChain
from forge.inference.chain import atomic_json, snapshot
from forge.inference.gateway_chain import live_anchor
from forge.inference.gateway_epoch import commitment, light_plan, verify_master
from forge.inference.publication import _claim_publication_intent, _publish_claimed
from forge.inference.publication import reconcile as reconcile_weights


def _ready(chain: BittensorChain) -> dict[str, Any]:
    state = snapshot(chain)
    if chain.network != "test" or chain.netuid != 578 or state["mechanismCount"] != 2:
        raise ValueError("gateway_publication_namespace")
    return state


def publish_master_commitment(
    chain: BittensorChain,
    wallet: Any,
    master: dict[str, Any],
    policy: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Broadcast one commitment only after persisting its unique intent."""
    if receipt_path.exists():
        raise ValueError("Existing gateway master intent; reconcile before another submission")
    body = verify_master(master, policy)
    state = _ready(chain)
    hotkey = wallet.hotkey.ss58_address
    if (
        hotkey != body["publisherHotkey"]
        or hotkey == policy.get("m0PublisherHotkey")
        or hotkey not in state["uids"].values()
    ):
        raise ValueError("gateway_master_commitment_publisher")
    receipt = {
        "schema": "attelier/gateway-master-publication/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "status": "SUBMITTING",
        "publisherHotkey": hotkey,
        "commitment": commitment(master),
        "masterDigest": master["digest"],
        "atBlock": state["block"],
        "atBlockHash": state["blockHash"],
    }
    atomic_json(receipt_path, receipt)
    try:
        receipt["extrinsicId"] = chain.publish(wallet, receipt["commitment"])
        receipt["status"] = "SUBMITTED"
    except Exception as exc:
        receipt.update(status="UNKNOWN_AFTER_SUBMIT", error=type(exc).__name__)
        atomic_json(receipt_path, receipt)
        raise
    atomic_json(receipt_path, receipt)
    return reconcile_master_commitment(chain, receipt, receipt_path)


def reconcile_master_commitment(
    chain: BittensorChain, receipt: dict[str, Any], receipt_path: Path
) -> dict[str, Any]:
    if (
        receipt.get("schema") != "attelier/gateway-master-publication/v1"
        or receipt.get("network") != "test"
        or receipt.get("netuid") != 578
        or receipt.get("mechanism") != 1
    ):
        raise ValueError("gateway_master_receipt")
    state = _ready(chain)
    latest = chain.raw_commitment(str(receipt["publisherHotkey"]), state["block"])
    receipt.update(
        checkedAt=time.time(), observedBlock=state["block"], observedBlockHash=state["blockHash"]
    )
    if latest is not None and latest[1] == receipt["commitment"]:
        receipt.update(status="FINALIZED", commitmentBlock=latest[0])
    elif receipt.get("status") != "UNKNOWN_AFTER_SUBMIT":
        receipt["status"] = "PENDING"
    atomic_json(receipt_path, receipt)
    return receipt


def _authority_path(authority_directory: Path, artifact_path: Path, label: str) -> Path:
    """Require a gateway CR artifact to live directly in its canonical owner."""
    authority = authority_directory.resolve()
    if not authority.is_dir() or artifact_path.resolve().parent != authority:
        raise ValueError(f"gateway_{label}_authority")
    return authority


def _next_state(master: dict[str, Any]) -> dict[str, Any]:
    body = master["body"]
    return {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "sequence": body["sequence"],
        "masterDigest": master["digest"],
    }


def _validate_gateway_weight_receipt(receipt: dict[str, Any], master: dict[str, Any]) -> None:
    if (
        receipt.get("schema") != "forge/inference-publication/v1"
        or receipt.get("network") != "test"
        or receipt.get("netuid") != 578
        or receipt.get("mechanism") != 1
    ):
        raise ValueError("gateway_weight_receipt")
    scores, requested = receipt.get("scores"), receipt.get("requestedWeights")
    if (
        not isinstance(scores, dict)
        or scores.get("gatewayMasterDigest") != master["digest"]
        or not isinstance(requested, dict)
    ):
        raise ValueError("gateway_weight_receipt_binding")
    expected = {hotkey: weight for hotkey, weight in master["body"]["weights"]}
    miners = scores.get("miners")
    if not isinstance(miners, dict) or len(miners) != len(expected):
        raise ValueError("gateway_weight_receipt_plan")
    actual: dict[str, int] = {}
    requested_actual: dict[int, int] = {}
    for key, value in miners.items():
        if (
            not isinstance(value, dict)
            or type(value.get("uid")) is not int
            or value["uid"] < 0
            or not isinstance(value.get("hotkey"), str)
            or type(value.get("weight")) is not int
            or str(value["uid"]) != key
            or value["hotkey"] in actual
            or value["hotkey"] not in expected
        ):
            raise ValueError("gateway_weight_receipt_plan")
        actual[value["hotkey"]] = value["weight"]
        requested_actual[value["uid"]] = value["weight"]
    try:
        requested_from_receipt = {int(uid): weight for uid, weight in requested.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("gateway_weight_receipt_plan") from exc
    if actual != expected or requested_actual != requested_from_receipt:
        raise ValueError("gateway_weight_receipt_plan")


def _advance_revealed_state(state_path: Path, master: dict[str, Any]) -> None:
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("gateway_light_state") from exc
        prior = previous.get("sequence", -1) if isinstance(previous, dict) else -1
        if (
            type(prior) is not int
            or prior > master["body"]["sequence"]
            or (
                prior == master["body"]["sequence"]
                and previous.get("masterDigest") != master["digest"]
            )
        ):
            raise ValueError("gateway_light_replay_or_conflict")
    atomic_json(state_path, _next_state(master))


def _validate_light_publisher(
    wallet: Any, state: dict[str, Any], master: dict[str, Any], plan: dict[str, Any]
) -> None:
    """Keep the outer M1 permit gate and the master UID binding intact."""
    hotkey = wallet.hotkey.ss58_address
    if hotkey not in state["committee"]:
        raise ValueError("Validator has no finalized permit/stake")
    uids, weights = plan.get("uids"), plan.get("weights")
    master_weights = master["body"]["weights"]
    if (
        not isinstance(uids, list)
        or not isinstance(weights, list)
        or len(uids) != len(weights)
        or len(uids) != len(master_weights)
    ):
        raise ValueError("gateway_weight_plan")
    for (miner_hotkey, master_weight), uid, weight in zip(
        master_weights, uids, weights, strict=True
    ):
        if (
            type(uid) is not int
            or state["uids"].get(str(uid)) != miner_hotkey
            or weight != master_weight
        ):
            raise ValueError("gateway_weight_uid_remap")


def _reject_published_sequence(state_path: Path | None, plan: dict[str, Any]) -> None:
    """A revealed state may be planned again, but must never be published again."""
    if state_path is None or not state_path.exists():
        return
    try:
        previous = json.loads(state_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("gateway_light_state") from exc
    if not isinstance(previous, dict) or type(previous.get("sequence")) is not int:
        raise ValueError("gateway_light_state")
    if previous["sequence"] >= plan["sequence"]:
        raise ValueError("gateway_light_already_published")


def _authority_revealed_master(intent_path: Path, master_digest: str) -> bool:
    """Retained CR intent closes the replay hole if the state file was lost."""
    if not intent_path.exists():
        return False
    try:
        pointer = json.loads(intent_path.read_text(encoding="utf8"))
        receipt = json.loads(Path(pointer["receiptPath"]).read_text(encoding="utf8"))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("gateway_publication_authority") from exc
    return (
        receipt.get("schema") == "forge/inference-publication/v1"
        and receipt.get("status") == "REVEALED"
        and isinstance(receipt.get("scores"), dict)
        and receipt["scores"].get("gatewayMasterDigest") == master_digest
    )


def publish_gateway_weights(
    chain: BittensorChain,
    wallet: Any,
    master: dict[str, Any],
    policy: dict[str, Any],
    receipt_path: Path,
    state_path: Path | None = None,
    authority_directory: Path | None = None,
) -> dict[str, Any]:
    """Re-derive a live anchored plan, then use the normal CR publication path."""
    if receipt_path.exists():
        raise ValueError("Existing gateway weight intent; reconcile before another submission")
    if authority_directory is None:
        raise ValueError("gateway_publication_authority_required")
    authority = _authority_path(authority_directory, receipt_path, "receipt")
    if state_path is not None:
        _authority_path(authority, state_path, "state")
    anchor, metagraph, _ = live_anchor(chain, master, policy)
    prior = (
        None
        if state_path is None or not state_path.exists()
        else __import__("json").loads(state_path.read_text())
    )
    plan = light_plan(master, policy, anchor, metagraph, prior)
    _reject_published_sequence(state_path, plan)
    state = _ready(chain)
    _validate_light_publisher(wallet, state, master, plan)
    if wallet.hotkey.ss58_address not in state["uids"].values():
        raise ValueError("gateway_weight_publisher_not_registered")
    uids, weights = plan.get("uids"), plan.get("weights")
    if (
        not isinstance(uids, list)
        or not isinstance(weights, list)
        or len(uids) != len(weights)
        or sum(weights) not in (0, 65535)
    ):
        raise ValueError("gateway_weight_plan")
    if not uids:
        receipt = {
            "schema": "attelier/gateway-weight-publication/v1",
            "network": "test",
            "netuid": 578,
            "mechanism": 1,
            "status": "NOOP",
            "publisherHotkey": wallet.hotkey.ss58_address,
            "atBlock": state["block"],
            "atBlockHash": state["blockHash"],
            "uids": uids,
            "weights": weights,
            "masterDigest": plan.get("masterDigest"),
        }
        atomic_json(receipt_path, receipt)
        return receipt
    scores = {
        "gatewayMasterDigest": plan["masterDigest"],
        "miners": {
            str(uid): {"uid": uid, "weight": weight, "hotkey": state["uids"].get(str(uid))}
            for uid, weight in zip(uids, weights, strict=True)
        },
    }
    from types import SimpleNamespace

    owner = SimpleNamespace(directory=authority)
    lock, intent = _claim_publication_intent(
        owner.directory, "test", 578, wallet.hotkey.ss58_address, receipt_path
    )
    try:
        if _authority_revealed_master(intent, plan["masterDigest"]):
            raise ValueError("gateway_light_already_published")
        result = _publish_claimed(chain, wallet, owner, scores, state, receipt_path, intent)
    finally:
        lock.close()
    return result


def reconcile_gateway_weights(
    chain: BittensorChain,
    master: dict[str, Any],
    policy: dict[str, Any],
    receipt_path: Path,
    state_path: Path,
    authority_directory: Path,
) -> dict[str, Any]:
    """Reconcile one CR receipt; consume the gateway sequence only after reveal."""
    authority = _authority_path(authority_directory, receipt_path, "receipt")
    _authority_path(authority, state_path, "state")
    verify_master(master, policy)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("gateway_weight_receipt") from exc
    if not isinstance(receipt, dict):
        raise ValueError("gateway_weight_receipt")
    _validate_gateway_weight_receipt(receipt, master)
    result = reconcile_weights(chain, receipt, receipt_path)
    if result.get("status") == "REVEALED":
        _advance_revealed_state(state_path, master)
    return result
