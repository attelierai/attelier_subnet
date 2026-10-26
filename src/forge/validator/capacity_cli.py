"""Offline capacity command for a distributed evaluation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.evaluation.capacity import CapacityInputs, plan_capacity
from forge.evaluation.distributed_plan import DistributedEvaluationPlan


def add_commands(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "distributed-capacity",
        help="price a complete frozen distributed panel; never dispatches GPU work",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--king-seconds", type=float, required=True)
    parser.add_argument("--wan50-seconds", type=float, required=True)
    parser.add_argument("--candidate-seconds", type=float, required=True)
    parser.add_argument("--gpu-usd-hour", type=float, required=True)
    parser.add_argument("--node-overhead-seconds", type=float)
    parser.add_argument("--node-overhead-usd", type=float)
    parser.add_argument("--metric-seconds-per-video", type=float)
    parser.add_argument("--judge-usd-per-quality-comparison", type=float)
    parser.add_argument("--storage-usd-per-video", type=float)
    parser.add_argument(
        "--max-node-hours",
        type=float,
        help="optional operational cap; the required duration is calculated from the plan",
    )
    parser.add_argument("--validator-candidate-budget-usd", type=float, default=10.0)
    parser.add_argument("--validator-screening-spent-usd", type=float)
    parser.add_argument("--network-remaining-usd", type=float)
    parser.add_argument("--commit-delivery-seconds", type=float)
    parser.add_argument("--reveal-delivery-seconds", type=float)
    parser.add_argument("--chain-block-seconds", type=float)
    parser.add_argument("--out", type=Path)


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command != "distributed-capacity":
        return None
    if args.out is not None and args.out.exists():
        raise ValueError("output exists; retain the prior capacity receipt")
    raw = json.loads(args.plan.read_text())
    if not isinstance(raw, dict):
        raise ValueError("plan must be a JSON object")
    result = plan_capacity(
        DistributedEvaluationPlan.from_dict(raw),
        CapacityInputs(
            king_seconds=args.king_seconds,
            wan50_seconds=args.wan50_seconds,
            candidate_seconds=args.candidate_seconds,
            gpu_usd_hour=args.gpu_usd_hour,
            node_overhead_seconds=args.node_overhead_seconds,
            node_overhead_usd=args.node_overhead_usd,
            metric_seconds_per_video=args.metric_seconds_per_video,
            judge_usd_per_quality_comparison=args.judge_usd_per_quality_comparison,
            storage_usd_per_video=args.storage_usd_per_video,
            max_node_seconds=(
                args.max_node_hours * 3600 if args.max_node_hours is not None else None
            ),
            validator_candidate_budget_usd=args.validator_candidate_budget_usd,
            validator_screening_spent_usd=args.validator_screening_spent_usd,
            network_remaining_usd=args.network_remaining_usd,
            commit_delivery_seconds=args.commit_delivery_seconds,
            reveal_delivery_seconds=args.reveal_delivery_seconds,
            chain_block_seconds=args.chain_block_seconds,
        ),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0 if result["status"] == "ADMISSIBLE_PROJECTION" else 2
