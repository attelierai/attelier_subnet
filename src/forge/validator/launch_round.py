"""Durable deterministic preparation for one testnet Launch V2 round.

This module joins only already-verified boundaries: a finalized read-only chain
snapshot, a certified parent state, a replayed qualification package and a
measured capacity receipt.  It never signs, submits or starts work.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge.evaluation.evaluator_qualification import replay_qualification_package
from forge.evaluation.launch_policy import freeze_v2, validate_plan
from forge.evaluation.synthetic_prompts import launch_panel_v2
from forge.protocol.consensus_types import ConsensusPolicy, FinalizedCompetitionState, RoundPlan
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.admission import AuthenticatedAdmissionHistory
from forge.validator.launch_config import LaunchConfig, closed_selection, require_hash
from forge.validator.launch_service import LaunchServiceStore, write_immutable_json
from forge.validator.round_planner import FinalizedChainPort, resolve_closed_admissions


def _hash(value: object, name: str) -> str:
    return require_hash(value, name)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _immutable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(path, dict(value))


def _history_payload(history: AuthenticatedAdmissionHistory) -> dict[str, Any]:
    return {
        "network": history.network,
        "netuid": history.netuid,
        "through_block": history.through_block,
        "certified_state_hash": history.certified_state_hash,
        "history_root": history.history_root,
        "consumed_hotkeys": list(history.consumed_hotkeys),
        "registered_hotkeys": list(history.registered_hotkeys),
        "lineage_edges": [list(edge) for edge in history.lineage_edges],
        "window_id": history.window_id,
        "window_open_block": history.window_open_block,
        "window_close_block": history.window_close_block,
    }


def _history_manifest_hashes(service: LaunchServiceStore) -> set[str]:
    selected: set[str] = set()
    for row in service.snapshot()["rounds"]:
        selection = row["selection"]
        if not isinstance(selection, Mapping):
            raise ValueError("service history has an invalid selection")
        selected.add(
            _hash(selection.get("candidate_manifest_hash"), "historical candidate manifest")
        )
    return selected


def select_closed_round(
    *,
    chain: FinalizedChainPort,
    contract: ChallengeContract,
    config: LaunchConfig,
    parent: FinalizedCompetitionState,
    service: LaunchServiceStore,
    admission_history: AuthenticatedAdmissionHistory,
    close_block: int,
    finalized_block: int,
    round_dir: Path,
) -> dict[str, Any]:
    """Persist the canonical close snapshot and one FIFO candidate before E.

    The model resolver is intentionally invoked here, at the finalized close,
    rather than accepting a caller-provided admission subset.
    """
    config.verify_contract(contract)
    prior_selection_path = round_dir / "closed-selection.json"
    if prior_selection_path.exists():
        import json

        prior = json.loads(prior_selection_path.read_text())
        if not isinstance(prior, dict) or prior.get("close_block") != close_block:
            raise ValueError("round directory contains another close selection")
        service.select(prior, finalized_block=finalized_block)
        return prior
    if chain.network != config.network or chain.netuid != config.netuid:
        raise ValueError("read-only chain differs from frozen launch configuration")
    if type(close_block) is not int or close_block < 0:
        raise ValueError("close block is invalid")
    if admission_history.network != config.network or admission_history.netuid != config.netuid:
        raise ValueError("admission history belongs to another subnet")
    if admission_history.through_block != close_block:
        raise ValueError("admission history must be checkpointed exactly at the close block")
    if admission_history.certified_state_hash != parent.state_hash():
        raise ValueError("admission history is not bound to the certified parent")

    closed = resolve_closed_admissions(chain, contract, close_block)
    # The resolver may fetch public manifests; require a fresh finalized head only
    # after that work completes so a stale pre-read cannot move selection past E.
    fresh_height = chain.finalized_height()
    if not close_block <= fresh_height < close_block + config.entropy_offset_blocks:
        raise ValueError("closed admissions were not rebuilt before future entropy")
    if finalized_block > fresh_height:
        raise ValueError("finalized height moved backwards after rebuilding closed admissions")
    finalized_block = fresh_height
    registered = chain.uid_snapshot(close_block)
    if not isinstance(registered, dict) or any(
        type(uid) is not int or not isinstance(hotkey, str) or not hotkey
        for uid, hotkey in registered.items()
    ):
        raise ValueError("closed registered snapshot is invalid")
    if len(set(registered.values())) != len(registered):
        raise ValueError("closed registered snapshot has duplicate hotkeys")
    if any(commitment.hotkey not in set(registered.values()) for commitment in closed.commitments):
        raise ValueError("closed commitments are absent from registered snapshot")

    _immutable(
        round_dir / "closed-admissions.json",
        {
            "schema": "forge/launch-closed-admissions/v1",
            "close_block": close_block,
            "admitted": closed.admitted,
            "rejected": closed.rejected,
            "early_full_transformer_duplicates": closed.early_full_transformer_duplicates,
            "overflow": closed.overflow,
            "metadata_validated": closed.metadata_validated,
            "commitments": [asdict(item) for item in closed.commitments],
        },
    )
    _immutable(
        round_dir / "committee.json",
        {"schema": "forge/launch-committee/v1", "committee": closed.committee},
    )
    _immutable(
        round_dir / "registered.json",
        {
            "schema": "forge/launch-registered/v1",
            "registered": {str(uid): hotkey for uid, hotkey in sorted(registered.items())},
        },
    )
    _immutable(round_dir / "admission-history.json", _history_payload(admission_history))

    prior_manifests = _history_manifest_hashes(service)
    eligible = [
        item
        for item in closed.admitted
        if close_block - config.admission_window_blocks < item["block"] <= close_block
        and item["manifest_hash"] not in prior_manifests
    ]
    if not eligible:
        raise ValueError("no unselected admitted candidate committed in the closed window")
    candidate = eligible[0]  # resolver already provides the protocol's canonical ordering.
    selection = closed_selection(
        config,
        close_block=close_block,
        close_hash=chain.finalized_block_hash(close_block),
        finalized_block=finalized_block,
        parent_state_hash=parent.state_hash(),
        candidate_manifest_hash=candidate["manifest_hash"],
        candidate_hotkey=candidate["hotkey"],
        commitment_block=candidate["block"],
    )
    _immutable(round_dir / "closed-selection.json", selection)
    service.select(selection, finalized_block=finalized_block)
    return selection


def _ready_components(
    worker_ready: Mapping[str, Any], qualification: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(worker_ready) != {"evaluator", "runtime", "capacity"}:
        raise ValueError("worker-ready receipt must contain evaluator, runtime and capacity")
    evaluator, runtime, capacity = (
        worker_ready["evaluator"],
        worker_ready["runtime"],
        worker_ready["capacity"],
    )
    if not isinstance(evaluator, Mapping) or set(evaluator) != {"evaluator_id", "spec_hash"}:
        raise ValueError("worker-ready evaluator is invalid")
    evaluator_out = {key: _hash(evaluator[key], f"evaluator {key}") for key in evaluator}
    if evaluator_out != {key: qualification[key] for key in evaluator_out}:
        raise ValueError("worker-ready evaluator differs from replayed qualification")
    runtime_keys = {"runtime_id", "hardware_id", "dependency_lock_hash"}
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
        raise ValueError("worker-ready runtime is invalid")
    runtime_out = {key: _hash(runtime[key], f"runtime {key}") for key in runtime_keys}
    capacity_keys = {
        "measured_parallelism",
        "worst_case_execution_seconds",
        "commit_delivery_seconds",
        "reveal_delivery_seconds",
        "chain_block_seconds",
    }
    if not isinstance(capacity, Mapping) or set(capacity) != capacity_keys:
        raise ValueError("worker-ready capacity is invalid")
    if type(capacity["measured_parallelism"]) is not int or capacity["measured_parallelism"] < 1:
        raise ValueError("worker-ready measured parallelism is invalid")
    capacity_out = {"measured_parallelism": capacity["measured_parallelism"]}
    for key in capacity_keys - {"measured_parallelism"}:
        capacity_out[key] = _positive_number(capacity[key], f"capacity {key}")
    return evaluator_out, runtime_out, capacity_out


def freeze_closed_round(
    *,
    config: LaunchConfig,
    parent: FinalizedCompetitionState,
    service: LaunchServiceStore,
    selection: Mapping[str, Any],
    entropy_hash: str,
    finalized_block: int,
    basic_manifests: Mapping[str, str],
    worker_ready: Mapping[str, Any],
    qualification_package: Mapping[str, Any],
    admission_history: AuthenticatedAdmissionHistory,
    committee: Mapping[str, int],
    round_dir: Path,
    bootstrap_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], RoundPlan]:
    """Freeze deterministic V2 policy and consensus plan after E is finalized."""
    entropy_hash = _hash(entropy_hash, "entropy hash")
    if dict(selection).get("entropy_block") != dict(selection).get("close_block", -10) + 10:
        raise ValueError("selection does not preserve E = C + 10")
    if finalized_block < selection["entropy_block"]:
        raise ValueError("entropy is not finalized")
    if selection.get("config_hash") != config.config_hash():
        raise ValueError("selection belongs to another launch configuration")
    if selection.get("parent_state_hash") != parent.state_hash():
        raise ValueError("selection differs from the certified parent state")
    frozen_history_path = round_dir / "admission-history.json"
    if not frozen_history_path.exists():
        raise ValueError("closed admission history is absent")
    import json

    if json.loads(frozen_history_path.read_text()) != _history_payload(admission_history):
        raise ValueError("admission history differs from the frozen close checkpoint")
    if (
        admission_history.through_block != selection["close_block"]
        or admission_history.certified_state_hash != parent.state_hash()
    ):
        raise ValueError("admission history differs from parent state")
    if dict(basic_manifests).keys() != {"candidate", "king", "anchor"}:
        raise ValueError("basic manifests must bind candidate, king and anchor")
    models = {name: _hash(value, f"{name} manifest") for name, value in basic_manifests.items()}
    if models["candidate"] != selection["candidate_manifest_hash"]:
        raise ValueError("candidate manifest differs from closed selection")
    if models["king"] != parent.current_king_fingerprint:
        raise ValueError("king manifest differs from certified parent")
    qualification = replay_qualification_package(qualification_package)
    evaluator, runtime, capacity = _ready_components(worker_ready, qualification)
    prompts = launch_panel_v2(entropy_hash, round_id=selection["round_id"])
    timing = sorted(
        (item["prompt_id"] for item in prompts),
        key=lambda item: hash_canonical(
            "forge/launch-v2-timing/v1", entropy_hash, selection["round_id"], item
        ),
    )[:8]
    diversity = [
        min(
            (item["prompt_id"] for item in prompts if item["family"] == family),
            key=lambda item: hash_canonical(
                "forge/launch-v2-diversity/v1", entropy_hash, selection["round_id"], family, item
            ),
        )
        for family in sorted({item["family"] for item in prompts})
    ]
    policy = freeze_v2(
        {
            "models": models,
            "evaluator": evaluator,
            "runtime": runtime,
            "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "prompts": prompts,
            "timing_prompt_ids": timing,
            "diversity_prompt_ids": diversity,
            "evaluator_qualification": qualification,
        }
    )
    validate_plan(
        policy, expected_entropy_hash=entropy_hash, expected_round_id=selection["round_id"]
    )
    _immutable(round_dir / "qualification-package.json", dict(qualification_package))
    _immutable(round_dir / "launch-plan.json", policy)
    _immutable(round_dir / "state.json", parent.to_dict())

    committee_path = round_dir / "committee.json"
    if not committee_path.exists():
        raise ValueError("closed committee snapshot is absent")
    import json

    frozen_committee = json.loads(committee_path.read_text())
    if not isinstance(frozen_committee, dict) or frozen_committee.get("committee") != dict(
        committee
    ):
        raise ValueError("committee differs from the frozen close snapshot")
    committee_out = dict(committee)
    if not committee_out or any(
        not isinstance(key, str) or not key or type(value) is not int or value <= 0
        for key, value in committee_out.items()
    ):
        raise ValueError("committee snapshot is invalid")
    e = selection["entropy_block"]
    commit_deadline = e + math.ceil(
        (capacity["worst_case_execution_seconds"] + capacity["commit_delivery_seconds"])
        / capacity["chain_block_seconds"]
    )
    reveal_deadline = commit_deadline + math.ceil(
        capacity["reveal_delivery_seconds"] / capacity["chain_block_seconds"]
    )
    bootstrap_sha: str | None = None
    if bootstrap_receipt_path is not None and bootstrap_receipt_path.is_file():
        bootstrap_sha = hashlib.sha256(bootstrap_receipt_path.read_bytes()).hexdigest()
        target = round_dir / "trusted-receipts" / "bootstrap" / f"{bootstrap_sha}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != bootstrap_receipt_path.read_bytes():
            raise ValueError("bootstrap receipt custody changed")
        if not target.exists():
            target.write_bytes(bootstrap_receipt_path.read_bytes())
    elif parent.state_version == 0 and models["anchor"] != parent.current_king_fingerprint:
        raise ValueError("imported genesis requires an immutable bootstrap receipt")
    prompt_history = {
        "close_block": selection["close_block"],
        "entropy_block": e,
        "entropy_hash": entropy_hash,
        "plan_id": policy["plan_id"],
        "prompts": prompts,
        "timing_prompt_ids": timing,
        "diversity_prompt_ids": diversity,
    }
    prompt_history_root = hash_canonical("forge/launch-v2-prompt-history/v1", prompt_history)
    plan = RoundPlan(
        config.network,
        config.netuid,
        selection["round_id"],
        parent.state_hash(),
        parent.challenge_id,
        models["king"],
        models["candidate"],
        selection["candidate_hotkey"],
        config.contract_hash,
        {
            "admission_history_root": admission_history.history_root,
            "launch_plan_id": policy["plan_id"],
            "anchor_fingerprint": models["anchor"],
            "selection_hash": selection["selection_hash"],
            "entropy_block": e,
            "entropy_hash": entropy_hash,
            "qualification_receipt_id": qualification["receipt_id"],
            "prompt_history_root": prompt_history_root,
            **({"bootstrap_receipt_sha256": bootstrap_sha} if bootstrap_sha else {}),
        },
        committee_out,
        selection["close_block"],
        commit_deadline,
        reveal_deadline,
        reveal_deadline + 1,
        ConsensusPolicy.STAKE_MAJORITY_V2,
    )
    _immutable(round_dir / "round-plan.json", plan.to_dict())
    _immutable(
        round_dir / "prompt-history.json",
        {
            "schema": "forge/launch-v2-prompt-history/v1",
            **prompt_history,
            "prompt_history_root": prompt_history_root,
            **({"bootstrap_receipt_sha256": bootstrap_sha} if bootstrap_sha else {}),
        },
    )
    _immutable(
        round_dir / "capacity.json",
        {
            "schema": "forge/launch-capacity-receipt/v1",
            "round_plan_hash": plan.plan_hash(),
            "launch_plan_id": policy["plan_id"],
            **capacity,
        },
    )
    _immutable(round_dir / "worker-ready.json", dict(worker_ready))
    # Mark READY only when every immutable input needed by a restarted worker
    # is present. A crash earlier can safely finish this same frozen plan.
    service.freeze(
        selection["round_id"],
        finalized_block=finalized_block,
        entropy_hash=entropy_hash,
        plan=policy,
    )
    return policy, plan
