"""Offline file operator for a future, single-candidate Launch V1 testnet round.

This module never creates wallets, signatures, GPU jobs, RPC clients or chain
transactions.  It turns reviewed finalized snapshots and worker receipts into
the exact local files consumed by :mod:`forge.validator.launch_bridge`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Verifier
from forge.evaluation import launch_policy
from forge.evaluation.synthetic_prompts import launch_panel
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.admission_checkpoint import SignedAdmissionHistory
from forge.protocol.consensus_types import (
    ConsensusPolicy,
    FinalizedCompetitionState,
    RoundCertificate,
    RoundPlan,
)
from forge.protocol.hashing import hash_canonical
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox
from forge.validator.admission import AuthenticatedAdmissionHistory
from forge.validator.launch_bridge import (
    BUNDLE_SCHEMA,
    LaunchCertificateVerifier,
    LaunchExecutionReplayer,
    LaunchV1Bridge,
    LaunchV1Capacity,
    LocalLaunchReceiptStore,
    launch_local_report,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"refusing to overwrite {path}")
        return
    path.write_text(encoded)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _checkpoint(path: Path) -> SignedAdmissionHistory:
    value = _read(path)
    if set(value) != {"history", "signatures"}:
        raise ValueError("admission checkpoint fields are invalid")
    raw = value["history"]
    if not isinstance(raw, dict) or not isinstance(value["signatures"], dict):
        raise ValueError("admission checkpoint types are invalid")
    history = AuthenticatedAdmissionHistory(
        **{
            **raw,
            "consumed_hotkeys": tuple(raw["consumed_hotkeys"]),
            "registered_hotkeys": tuple(raw["registered_hotkeys"]),
            "lineage_edges": tuple(tuple(edge) for edge in raw["lineage_edges"]),
        }
    )
    return SignedAdmissionHistory(
        history,
        {hotkey: bytes.fromhex(signature) for hotkey, signature in value["signatures"].items()},
    )


def _store_receipt(root: Path, kind: str, source: Path) -> str:
    encoded = source.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one JSON object")
    digest = hashlib.sha256(encoded).hexdigest()
    target = root / kind / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != encoded:
        raise ValueError(f"receipt hash collision in {kind}")
    if not target.exists():
        target.write_bytes(encoded)
    return digest


def _entropy_bound(launch_plan: dict[str, Any], *, round_id: str, entropy_hash: str) -> None:
    """Require deterministic seeds and the timing subset before plan persistence."""
    if launch_policy.policy_for(launch_plan) == "v2":
        launch_policy.validate_plan(
            launch_plan, expected_entropy_hash=entropy_hash, expected_round_id=round_id
        )
        return

    expected_prompts = launch_panel(entropy_hash, round_id=round_id)
    if launch_plan["prompts"] != expected_prompts:
        raise ValueError("launch plan prompts/tags/seeds differ from the frozen synthetic panel")
    expected_timing = [
        prompt_id
        for prompt_id in sorted(
            (prompt["prompt_id"] for prompt in expected_prompts),
            key=lambda prompt_id: hash_canonical(
                "forge/launch-v1-timing/v1", entropy_hash, round_id, prompt_id
            ),
        )[:8]
    ]
    if launch_plan["timing_prompt_ids"] != expected_timing:
        raise ValueError("launch plan timing selection is not derived from future entropy")


def _manifests(paths: list[Path], launch_plan: dict[str, Any]) -> None:
    found = {ModelManifest.from_dict(_read(path)).fingerprint() for path in paths}
    if found != set(launch_plan["models"].values()):
        raise ValueError("three immutable manifests must exactly match the frozen launch plan")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    launch_plan = _read(args.launch_plan)
    (
        launch_policy._check_plan_v2(launch_plan)
        if launch_policy.policy_for(launch_plan) == "v2"
        else launch_policy._check_plan(launch_plan)
    )
    if args.network != "test":
        raise ValueError("Launch V1 operator is testnet-only")
    state = FinalizedCompetitionState.from_dict(_read(args.state))
    checkpoint = _checkpoint(args.admission_checkpoint)
    committee = _read(args.committee)
    if not all(
        isinstance(key, str) and type(value) is int and value > 0
        for key, value in committee.items()
    ):
        raise ValueError("committee snapshot is invalid")
    if checkpoint.history.network != args.network or checkpoint.history.netuid != args.netuid:
        raise ValueError("signed admission history belongs to another subnet")
    if (
        args.close_block < 0
        or args.entropy_block <= args.close_block
        or args.entropy_block >= args.commit_deadline
        or checkpoint.history.through_block != args.close_block
    ):
        raise ValueError("entropy must be a later finalized block than close")
    _sha(args.entropy_hash, "entropy hash")
    _entropy_bound(launch_plan, round_id=args.round_id, entropy_hash=args.entropy_hash)
    _manifests(args.manifest, launch_plan)
    prompt_history = {
        "close_block": args.close_block,
        "entropy_block": args.entropy_block,
        "entropy_hash": args.entropy_hash,
        "plan_id": launch_plan["plan_id"],
        "prompts": launch_plan["prompts"],
        "timing_prompt_ids": launch_plan["timing_prompt_ids"],
    }
    prompt_root = hash_canonical("forge/launch-v1-prompt-history/v1", prompt_history)
    _write_new(args.prompt_history_out, {"prompt_history_root": prompt_root, **prompt_history})
    plan = RoundPlan(
        args.network,
        args.netuid,
        args.round_id,
        state.state_hash(),
        state.challenge_id,
        launch_plan["models"]["king"],
        launch_plan["models"]["candidate"],
        args.candidate_hotkey,
        _sha(args.contract_hash, "contract hash"),
        {
            "admission_history_root": checkpoint.history.history_root,
            "prompt_history_root": prompt_root,
            "launch_plan_id": launch_plan["plan_id"],
            "anchor_fingerprint": launch_plan["models"]["anchor"],
            "entropy_block": args.entropy_block,
            "entropy_hash": args.entropy_hash,
        },
        committee,
        args.close_block,
        args.commit_deadline,
        args.reveal_deadline,
        args.activation_block,
        # Historical plans deserialize to their historical-supermajority
        # semantics.  Every newly prepared Launch V1 round freezes the current
        # stake-majority policy in its hashed payload instead.
        ConsensusPolicy.STAKE_MAJORITY_V2,
    )
    _write_new(args.out, plan.to_dict())
    return {
        "schema": "forge/launch-v1-operator-prepare/v1",
        "status": "PREPARED_NOT_OPEN",
        "plan_hash": plan.plan_hash(),
        "prompt_history_root": prompt_root,
    }


def open_round(args: argparse.Namespace) -> dict[str, Any]:
    """Open a prepared plan after capacity has been measured against its hash."""

    plan = RoundPlan.from_dict(_read(args.round_plan))
    launch_plan = _read(args.launch_plan)
    state = FinalizedCompetitionState.from_dict(_read(args.state))
    checkpoint = _checkpoint(args.admission_checkpoint)
    capacity_hash = _store_receipt(args.receipt_root, "capacity", args.capacity)
    with Store(args.db) as store:
        bridge = LaunchV1Bridge(
            store, Sr25519Verifier(), LocalLaunchReceiptStore(args.receipt_root)
        )
        opened = bridge.open_round(
            plan, state, checkpoint, launch_plan, LaunchV1Capacity(capacity_hash)
        )
    _write_new(
        args.out,
        {
            "schema": "forge/launch-v1-operator-open/v1",
            "plan_hash": plan.plan_hash(),
            "capacity_hash": capacity_hash,
        },
    )
    return {
        "schema": "forge/launch-v1-operator-open/v1",
        "status": "OPENED" if opened else "ALREADY_OPEN",
        "plan_hash": plan.plan_hash(),
    }


def bundle(args: argparse.Namespace) -> dict[str, Any]:
    plan, launch_plan = RoundPlan.from_dict(_read(args.round_plan)), _read(args.launch_plan)
    (
        launch_policy._check_plan_v2(launch_plan)
        if launch_policy.policy_for(launch_plan) == "v2"
        else launch_policy._check_plan(launch_plan)
    )
    panels = [_store_receipt(args.receipt_root, "panels", path) for path in args.panel]
    timing = [_store_receipt(args.receipt_root, "timing", path) for path in args.timing]
    scores = [_store_receipt(args.receipt_root, "scores", path) for path in args.score]
    v2 = launch_policy.policy_for(launch_plan) == "v2"
    diversity = [_store_receipt(args.receipt_root, "diversity", path) for path in args.diversity]
    diversity_panels = [
        _store_receipt(args.receipt_root, "panels", path)
        for path in getattr(args, "diversity_panel", [])
    ]
    qualification = (
        _store_receipt(args.receipt_root, "qualification", args.qualification)
        if args.qualification
        else None
    )
    if v2 and (not diversity or qualification is None):
        raise ValueError("V2 bundle requires diversity and qualification receipts")
    if not v2 and (diversity or qualification is not None):
        raise ValueError("V1 bundle does not accept V2 receipts")
    if v2:
        sources = {
            _sha(_read(path).get("source_report_sha256"), "diversity source report hash")
            for path in args.diversity
        }
        if set(diversity_panels) != sources:
            raise ValueError("V2 diversity panels must exactly cover diversity source reports")
    body = {
        "schema": "forge/launch-execution-bundle/v2" if v2 else BUNDLE_SCHEMA,
        "validator_hotkey": args.validator_hotkey,
        "round_plan_hash": plan.plan_hash(),
        "parent_state_hash": plan.parent_state_hash,
        "admission_history_root": plan.prompt_seed_inputs["admission_history_root"],
        "prompt_history_root": plan.prompt_seed_inputs["prompt_history_root"],
        "launch_plan_id": launch_plan["plan_id"],
        "models": launch_plan["models"],
        "launch_plan": launch_plan,
        "panel_receipts": panels,
        "timing_receipts": timing,
        "score_receipts": scores,
    }
    if v2:
        body.update({"diversity_receipts": diversity, "qualification_receipt": qualification})
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    target = args.receipt_root / "bundles" / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != encoded:
        raise ValueError("bundle hash collision")
    if not target.exists():
        target.write_bytes(encoded)
    _write_new(args.out, {"bundle_hash": digest, "validator_hotkey": args.validator_hotkey})
    return {"schema": "forge/launch-v1-operator-bundle/v1", "bundle_hash": digest}


def report(args: argparse.Namespace) -> dict[str, Any]:
    plan = RoundPlan.from_dict(_read(args.round_plan))
    receipt = _read(args.bundle)
    digest = _sha(receipt.get("bundle_hash"), "bundle hash")
    replayer = LaunchExecutionReplayer(LocalLaunchReceiptStore(args.receipt_root))
    local = launch_local_report(plan, args.validator_hotkey, digest, replayer)
    value = local.to_dict()
    _write_new(args.out, value)
    return {
        "schema": "forge/launch-v1-operator-report/v1",
        "report_hash": local.report_hash(),
        "verdict": value["verdict"],
    }


def apply_certificate(args: argparse.Namespace) -> dict[str, Any]:
    plan = RoundPlan.from_dict(_read(args.round_plan))
    state = FinalizedCompetitionState.from_dict(_read(args.state))
    certificate = RoundCertificate.from_dict(_read(args.certificate))
    with Store(args.db) as store:
        bridge = LaunchV1Bridge(
            store, Sr25519Verifier(), LocalLaunchReceiptStore(args.receipt_root)
        )
        result = bridge.apply(
            plan,
            state,
            certificate,
            LaunchCertificateVerifier(
                Sr25519Verifier(),
                replayer=LaunchExecutionReplayer(LocalLaunchReceiptStore(args.receipt_root)),
                store=store,
            ),
        )
    _write_new(args.out, result.to_dict())
    return {"schema": "forge/launch-v1-operator-apply/v1", "state_hash": result.state_hash()}


def receive(args: argparse.Namespace) -> dict[str, Any]:
    plan = RoundPlan.from_dict(_read(args.round_plan))
    with Store(args.db) as store:
        accepted = RoundMailbox(store, Sr25519Verifier()).receive(
            plan, _read(args.envelope), finalized_block=args.finalized_block
        )
    return {"schema": "forge/launch-v1-operator-envelope/v1", "accepted": accepted}


def intent(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare an already-certified testnet intent; never submit weights."""

    plan = RoundPlan.from_dict(_read(args.round_plan))
    state = FinalizedCompetitionState.from_dict(_read(args.state))
    certificate = RoundCertificate.from_dict(_read(args.certificate))
    raw_uids, custody = _read(args.uids), _read(args.custody)
    if not isinstance(raw_uids, dict) or not isinstance(custody, dict):
        raise ValueError("UID snapshot and custody map must be objects")
    uids = {int(uid): hotkey for uid, hotkey in raw_uids.items()}
    with Store(args.db) as store:
        bridge = LaunchV1Bridge(
            store, Sr25519Verifier(), LocalLaunchReceiptStore(args.receipt_root)
        )
        value = bridge.prepare_testnet_weight_intent(
            state,
            plan=plan,
            certificate=certificate,
            runtime_hash=args.runtime_hash,
            registered_uids=uids,
            public_custody=custody,
            burn_hotkey=args.burn_hotkey,
            burn_receipt_hash=args.burn_receipt_hash,
        )
    _write_new(args.out, asdict(value))
    return {
        "schema": "forge/launch-v1-operator-intent/v1",
        "status": value.status,
        "intent_hash": value.intent_hash(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_p = commands.add_parser("prepare")
    prepare_p.add_argument("--state", type=Path, required=True)
    prepare_p.add_argument("--launch-plan", type=Path, required=True)
    prepare_p.add_argument("--admission-checkpoint", type=Path, required=True)
    prepare_p.add_argument("--committee", type=Path, required=True)
    prepare_p.add_argument("--manifest", type=Path, action="append", required=True)
    prepare_p.add_argument("--network", required=True)
    prepare_p.add_argument("--netuid", type=int, required=True)
    prepare_p.add_argument("--round-id", required=True)
    prepare_p.add_argument("--candidate-hotkey", required=True)
    prepare_p.add_argument("--contract-hash", required=True)
    prepare_p.add_argument("--close-block", type=int, required=True)
    prepare_p.add_argument("--entropy-block", type=int, required=True)
    prepare_p.add_argument("--entropy-hash", required=True)
    prepare_p.add_argument("--commit-deadline", type=int, required=True)
    prepare_p.add_argument("--reveal-deadline", type=int, required=True)
    prepare_p.add_argument("--activation-block", type=int, required=True)
    prepare_p.add_argument("--prompt-history-out", type=Path, required=True)
    prepare_p.add_argument("--out", type=Path, required=True)
    open_p = commands.add_parser("open")
    open_p.add_argument("--db", type=Path, required=True)
    open_p.add_argument("--receipt-root", type=Path, required=True)
    open_p.add_argument("--state", type=Path, required=True)
    open_p.add_argument("--round-plan", type=Path, required=True)
    open_p.add_argument("--launch-plan", type=Path, required=True)
    open_p.add_argument("--admission-checkpoint", type=Path, required=True)
    open_p.add_argument("--capacity", type=Path, required=True)
    open_p.add_argument("--out", type=Path, required=True)
    bundle_p = commands.add_parser("bundle")
    bundle_p.add_argument("--receipt-root", type=Path, required=True)
    bundle_p.add_argument("--round-plan", type=Path, required=True)
    bundle_p.add_argument("--launch-plan", type=Path, required=True)
    bundle_p.add_argument("--validator-hotkey", required=True)
    bundle_p.add_argument("--panel", type=Path, action="append", required=True)
    bundle_p.add_argument("--timing", type=Path, action="append", required=True)
    bundle_p.add_argument("--score", type=Path, action="append", required=True)
    bundle_p.add_argument("--diversity", type=Path, action="append", default=[])
    bundle_p.add_argument("--diversity-panel", type=Path, action="append", default=[])
    bundle_p.add_argument("--qualification", type=Path)
    bundle_p.add_argument("--out", type=Path, required=True)
    report_p = commands.add_parser("report")
    report_p.add_argument("--receipt-root", type=Path, required=True)
    report_p.add_argument("--round-plan", type=Path, required=True)
    report_p.add_argument("--bundle", type=Path, required=True)
    report_p.add_argument("--validator-hotkey", required=True)
    report_p.add_argument("--out", type=Path, required=True)
    apply_p = commands.add_parser("apply-certificate")
    apply_p.add_argument("--db", type=Path, required=True)
    apply_p.add_argument("--receipt-root", type=Path, required=True)
    apply_p.add_argument("--round-plan", type=Path, required=True)
    apply_p.add_argument("--state", type=Path, required=True)
    apply_p.add_argument("--certificate", type=Path, required=True)
    apply_p.add_argument("--out", type=Path, required=True)
    receive_p = commands.add_parser("receive-envelope")
    receive_p.add_argument("--db", type=Path, required=True)
    receive_p.add_argument("--round-plan", type=Path, required=True)
    receive_p.add_argument("--envelope", type=Path, required=True)
    receive_p.add_argument("--finalized-block", type=int, required=True)
    intent_p = commands.add_parser("intent")
    intent_p.add_argument("--db", type=Path, required=True)
    intent_p.add_argument("--receipt-root", type=Path, required=True)
    intent_p.add_argument("--round-plan", type=Path, required=True)
    intent_p.add_argument("--state", type=Path, required=True)
    intent_p.add_argument("--certificate", type=Path, required=True)
    intent_p.add_argument("--uids", type=Path, required=True)
    intent_p.add_argument("--custody", type=Path, required=True)
    intent_p.add_argument("--runtime-hash", required=True)
    intent_p.add_argument("--burn-hotkey", required=True)
    intent_p.add_argument("--burn-receipt-hash", required=True)
    intent_p.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = {
        "prepare": prepare,
        "open": open_round,
        "bundle": bundle,
        "report": report,
        "apply-certificate": apply_certificate,
        "receive-envelope": receive,
        "intent": intent,
    }[args.command](args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
