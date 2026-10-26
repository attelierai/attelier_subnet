"""Explicit CLI for a validator's already-reserved competitive assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_commands(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "distributed-mvp-worker",
        help="execute this validator's frozen confirmation slots and sign a partial report",
    )
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--wallet", required=True)
    command.add_argument("--hotkey", required=True)
    command.add_argument("--wallet-path", default="~/.bittensor/wallets")
    command.add_argument("--dashboard-endpoint")
    command.add_argument("--dashboard-outbox", type=Path)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("worker config must be a JSON object")
    return value


def deliver_completed_assignment(plan, output: Path, endpoint: str, outbox: Path) -> dict:
    """Deliver persisted evidence; an unavailable dashboard never reruns GPU work."""
    from forge.validator.dashboard_ingestion_cli import _write_once, push

    plan_path = output / "dashboard-plan.json"
    _write_once(plan_path, json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False).encode())
    try:
        return push(plan_path, output / "partial-report.json", endpoint, outbox)
    except RuntimeError:
        return {"status": "PENDING", "outbox": str(outbox)}


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command != "distributed-mvp-worker":
        return None
    if bool(args.dashboard_endpoint) != (args.dashboard_outbox is not None):
        raise ValueError("dashboard endpoint and outbox must be provided together")
    if args.dashboard_endpoint:
        from forge.validator.dashboard_ingestion_cli import _endpoint

        _endpoint(args.dashboard_endpoint)
    # Imports happen only after the explicit execution command, keeping plan
    # inspection independent from CUDA, model and wallet dependencies.
    import bittensor as bt

    from forge.chain import Sr25519Signer
    from forge.evaluation.distributed_plan import DistributedEvaluationPlan
    from forge.evaluation.vision_judge import ApiBudget
    from forge.models.artifact_manifest import ModelManifest
    from forge.protocol.contract import ChallengeContract
    from forge.validator.budget import BudgetLedger
    from forge.validator.distributed_mvp_runtime import (
        AssignedRuntime,
        VBenchConfig,
        execute_assigned_confirmation,
    )
    from forge.validator.mvp_runtime import MVPWork

    raw = _read(args.config)
    required = {
        "distributed_plan",
        "mvp_plan",
        "work",
        "policy_provenance",
        "contract",
        "base_manifest",
        "base_dir",
        "candidate_manifest",
        "candidate_dir",
        "incumbent_manifest",
        "incumbent_dir",
        "wan50_manifest",
        "wan50_dir",
        "dependency_lock_hash",
        "vbench",
        "luna_budget_db",
        "luna_budget_usd",
        "luna_env_file",
        "budget_db",
        "reservation_id",
        "candidate_id",
        "operation_key",
        "minimum_reserved_usd",
    }
    if set(raw) != required:
        raise ValueError("competitive worker config fields are not exact")
    vbench = raw["vbench"]
    if not isinstance(vbench, dict) or set(vbench) != {
        "checkout",
        "cache",
        "interpreter",
        "weight_sha256",
        "dino_revision",
        "runner_image_digest",
    }:
        raise ValueError("VBench config fields are not exact")
    work = raw["work"]
    if not isinstance(work, dict):
        raise ValueError("MVP work must be an object")
    job = AssignedRuntime(
        plan=DistributedEvaluationPlan.from_dict(raw["distributed_plan"]),
        mvp_plan=raw["mvp_plan"],
        work=MVPWork(**work),
        policy_provenance=raw["policy_provenance"],
        contract=ChallengeContract.from_dict(raw["contract"]),
        base=ModelManifest.from_dict(raw["base_manifest"]),
        base_dir=Path(raw["base_dir"]),
        candidate=ModelManifest.from_dict(raw["candidate_manifest"]),
        candidate_dir=Path(raw["candidate_dir"]),
        incumbent=ModelManifest.from_dict(raw["incumbent_manifest"]),
        incumbent_dir=Path(raw["incumbent_dir"]),
        wan50=ModelManifest.from_dict(raw["wan50_manifest"]),
        wan50_dir=Path(raw["wan50_dir"]),
        dependency_lock_hash=raw["dependency_lock_hash"],
        vbench=VBenchConfig(
            Path(vbench["checkout"]),
            Path(vbench["cache"]),
            Path(vbench["interpreter"]),
            vbench["weight_sha256"],
            vbench["dino_revision"],
            vbench["runner_image_digest"],
        ),
        luna_budget=ApiBudget(Path(raw["luna_budget_db"]), raw["luna_budget_usd"]),
        luna_env_file=Path(raw["luna_env_file"]) if raw["luna_env_file"] else None,
        budget=BudgetLedger(Path(raw["budget_db"])),
        reservation_id=raw["reservation_id"],
        candidate_id=raw["candidate_id"],
        operation_key=raw["operation_key"],
        minimum_reserved_usd=raw["minimum_reserved_usd"],
    )
    wallet = bt.Wallet(
        name=args.wallet, hotkey=args.hotkey, path=Path(args.wallet_path).expanduser()
    )
    result = execute_assigned_confirmation(job, Sr25519Signer(wallet), out=args.out)
    response = {"plan_hash": result["plan_hash"], "validator_hotkey": result["validator_hotkey"]}
    if args.dashboard_endpoint:
        response["dashboard_delivery"] = deliver_completed_assignment(
            job.plan, args.out, args.dashboard_endpoint, args.dashboard_outbox
        )
    print(json.dumps(response, allow_nan=False))
    return 0
