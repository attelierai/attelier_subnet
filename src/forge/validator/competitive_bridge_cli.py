"""Offline reconstruction command for the distributed competitive bridge.

It has no wallet flag and no chain client.  The command deliberately creates
an auditable replay receipt, not a certificate or a ``set_weights`` call.
Validators use the same bundle through ``DistributedExecutionReplayer`` when
forming their signed pre-consensus report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.chain import Sr25519Verifier
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.mvp_policy import EvaluationPolicy
from forge.protocol.consensus_types import RoundPlan
from forge.protocol.hashing import hash_canonical
from forge.validator.competitive_bridge import (
    BUNDLE_SCHEMA,
    DistributedExecutionReplayer,
    build_distributed_execution_bundle,
)


def add_commands(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "competitive-distributed-replay",
        help="replay signed confirmation evidence before a validator casts a competitive vote",
    )
    command.add_argument("--round-plan", type=Path, required=True)
    command.add_argument("--distributed-plan", type=Path, required=True)
    command.add_argument("--mvp-plan", type=Path, required=True)
    command.add_argument("--report", type=Path, action="append", required=True)
    command.add_argument("--policy", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--panel-validation", type=Path, required=True)
    command.add_argument("--worker-tolerances", type=Path, required=True)
    command.add_argument("--public-custody", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except FileExistsError as error:
        raise ValueError("output exists; preserve the prior replay receipt") from error


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command != "competitive-distributed-replay":
        return None
    if args.out.exists():
        raise ValueError("output exists; preserve the prior replay receipt")
    bundle = build_distributed_execution_bundle(
        round_plan=RoundPlan.from_dict(_read(args.round_plan)),
        distributed_plan=DistributedEvaluationPlan.from_dict(_read(args.distributed_plan)),
        reports=[_read(path) for path in args.report],
        mvp_plan=_read(args.mvp_plan),
        policy=EvaluationPolicy.from_dict(_read(args.policy)),
        calibration=_read(args.calibration),
        panel_validation=_read(args.panel_validation),
        worker_tolerances=_read(args.worker_tolerances),
        public_custody=_read(args.public_custody),
    )
    round_plan = RoundPlan.from_dict(bundle["round_plan"])
    gate = DistributedExecutionReplayer(Sr25519Verifier()).replay(bundle, round_plan)
    result = {
        "schema": "forge/competitive-distributed-replay-receipt/v1",
        "bundle_schema": BUNDLE_SCHEMA,
        "bundle_hash": hash_canonical("forge/mvp-execution-bundle/v1", bundle),
        "round_plan_hash": round_plan.plan_hash(),
        "distributed_plan_hash": DistributedEvaluationPlan.from_dict(
            bundle["distributed_plan"]
        ).plan_hash(),
        "gate": gate,
        # Custody is checked by the validator-owned artifact verifier before
        # signing a local report; a JSON URI cannot stand in for that check.
        "certified": False,
        "publication_enabled": False,
    }
    _write_new(args.out, result)
    print(json.dumps({"gate": gate["status"], "certified": False}, allow_nan=False))
    return 0 if gate["status"] != "INCOMPLETE" else 2
