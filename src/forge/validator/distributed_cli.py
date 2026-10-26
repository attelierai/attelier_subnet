"""Local plan reconstruction, signing and inspection for distributed validators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.evaluation.distributed import aggregate_partial_reports, signed_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan, allocate_work


def add_commands(commands: argparse._SubParsersAction) -> None:
    reconstruct = commands.add_parser(
        "distributed-round-plan", help="reconstruct a diagnostic round from finalized testnet"
    )
    reconstruct.add_argument("--netuid", type=int, required=True)
    reconstruct.add_argument("--close-block", type=int, required=True)
    reconstruct.add_argument("--entropy-block", type=int, required=True)
    reconstruct.add_argument("--candidate-manifest", type=Path, required=True)
    reconstruct.add_argument("--candidate-hotkey", required=True)
    reconstruct.add_argument("--base-manifest", type=Path, required=True)
    reconstruct.add_argument("--out", type=Path, required=True)
    for name in ("distributed-plan", "distributed-report", "distributed-aggregate"):
        command = commands.add_parser(name, help="signed distributed evaluation evidence")
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "distributed-report":
            command.add_argument("--wallet", required=True)
            command.add_argument("--hotkey", required=True)
            command.add_argument("--wallet-path", default="~/.bittensor/wallets")
            command.add_argument("--partial", type=Path, required=True)
            command.add_argument("--dashboard-endpoint")
            command.add_argument("--dashboard-outbox", type=Path)
        if name == "distributed-aggregate":
            command.add_argument("--report", type=Path, action="append", required=True)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command not in {
        "distributed-plan",
        "distributed-report",
        "distributed-aggregate",
        "distributed-round-plan",
    }:
        return None
    if args.out.exists():
        raise ValueError("output exists; retain evidence and reconcile before retry")
    if args.command == "distributed-round-plan":
        from forge.chain import BittensorChain
        from forge.models.artifact_manifest import ModelManifest
        from forge.validator.distributed_planner import diagnostic_plan

        chain = BittensorChain("test", args.netuid)
        try:
            plan = diagnostic_plan(
                chain,
                close_block=args.close_block,
                entropy_block=args.entropy_block,
                candidate=ModelManifest.from_dict(_read(args.candidate_manifest)),
                candidate_hotkey=args.candidate_hotkey,
                wan50=ModelManifest.from_dict(_read(args.base_manifest)),
            )
            _write(args.out, plan.to_dict())
            print(json.dumps({"output": str(args.out), "plan_hash": plan.plan_hash()}))
            return 0
        finally:
            chain.close()
    plan = DistributedEvaluationPlan.from_dict(_read(args.plan))
    if args.command == "distributed-plan":
        result = {
            "plan_hash": plan.plan_hash(),
            "assignments": [item.to_dict() for item in allocate_work(plan)],
            "reward_ready": plan.reward_ready,
        }
    else:
        from forge.chain import Sr25519Signer, Sr25519Verifier

        if args.command == "distributed-aggregate":
            result = aggregate_partial_reports(
                plan, [_read(path) for path in args.report], Sr25519Verifier()
            )
            result.update(reward_eligible=False, chain_anchors_verified=False)
        else:
            import bittensor as bt

            wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path)
            partial = _read(args.partial)
            if set(partial) != {"quality_vs_king", "quality_vs_wan50", "timing"}:
                raise ValueError("partial report fields are not exact")
            result = signed_partial_report(plan, Sr25519Signer(wallet), **partial)
    _write(args.out, result)
    response = {"output": str(args.out), "plan_hash": plan.plan_hash()}
    if args.command == "distributed-report" and args.dashboard_endpoint:
        if args.dashboard_outbox is None:
            raise ValueError("--dashboard-outbox is required with --dashboard-endpoint")
        from forge.validator.dashboard_ingestion_cli import push

        try:
            response["dashboard_delivery"] = push(
                args.plan, args.out, args.dashboard_endpoint, args.dashboard_outbox
            )
        except RuntimeError as exc:
            response["dashboard_delivery"] = {"status": "PENDING", "error": str(exc)}
            # Telemetry failure must not block the validator's chain/report workflow.
            # The signed report and immutable outbox already exist for later delivery.
    elif args.command == "distributed-report" and args.dashboard_outbox:
        raise ValueError("--dashboard-endpoint is required with --dashboard-outbox")
    print(json.dumps(response, allow_nan=False))
    return 0
