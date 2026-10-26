"""File/SQLite commands for the durable competitive lifecycle.

The commands deliberately stop before independent execution and custody
verification.  Those are validator runtime services, not JSON flags: accepting
a purported quality result or custody receipt on a command line would create a
second, unqualified certificate lane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.chain import Sr25519Verifier
from forge.evaluation.capacity import CapacityInputs
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.protocol.admission_checkpoint import SignedAdmissionHistory
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundPlan
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox
from forge.validator.admission import AuthenticatedAdmissionHistory
from forge.validator.competitive_lifecycle import CompetitiveLifecycle


def add_commands(commands: argparse._SubParsersAction) -> None:
    open_round = commands.add_parser(
        "competitive-open", help="persist a frozen competitive round after history verification"
    )
    open_round.add_argument("--db", type=Path, required=True)
    open_round.add_argument("--plan", type=Path, required=True)
    open_round.add_argument("--state", type=Path, required=True)
    open_round.add_argument("--admission-checkpoint", type=Path, required=True)
    open_round.add_argument("--distributed-plan", type=Path, required=True)
    open_round.add_argument(
        "--capacity-inputs",
        type=Path,
        required=True,
        help="raw CapacityInputs JSON; capacity receipts are recomputed, never trusted",
    )
    open_round.add_argument("--out", type=Path, required=True)

    expire = commands.add_parser(
        "competitive-expire", help="close an expired round without making a verdict"
    )
    expire.add_argument("--db", type=Path, required=True)
    expire.add_argument("--plan", type=Path, required=True)
    expire.add_argument("--finalized-block", type=int, required=True)
    expire.add_argument("--finalized-hash", required=True)
    expire.add_argument("--out", type=Path, required=True)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError("output exists; preserve the prior lifecycle receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _checkpoint(path: Path) -> SignedAdmissionHistory:
    value = _read(path)
    if set(value) != {"history", "signatures"} or not isinstance(value["history"], dict):
        raise ValueError("admission checkpoint fields are invalid")
    try:
        raw_history = value["history"]
        history = AuthenticatedAdmissionHistory(
            **{
                **raw_history,
                "consumed_hotkeys": tuple(raw_history["consumed_hotkeys"]),
                "registered_hotkeys": tuple(raw_history["registered_hotkeys"]),
                "lineage_edges": tuple(tuple(edge) for edge in raw_history["lineage_edges"]),
            }
        )
        signatures = {
            hotkey: bytes.fromhex(signature) for hotkey, signature in value["signatures"].items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("admission checkpoint encoding is invalid") from error
    return SignedAdmissionHistory(history, signatures)


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command not in {"competitive-open", "competitive-expire"}:
        return None
    # Do this before opening a transaction.  A forgotten receipt path must not
    # mutate round state and only then report that its output already existed.
    if args.out.exists():
        raise ValueError("output exists; preserve the prior lifecycle receipt")
    plan = RoundPlan.from_dict(_read(args.plan))
    with Store(args.db) as store:
        lifecycle = CompetitiveLifecycle(store, Sr25519Verifier())
        if args.command == "competitive-open":
            state = FinalizedCompetitionState.from_dict(_read(args.state))
            try:
                capacity_inputs = CapacityInputs(**_read(args.capacity_inputs))
            except TypeError as error:
                raise ValueError("capacity inputs do not match the supported schema") from error
            created = lifecycle.open_round(
                plan,
                state,
                _checkpoint(args.admission_checkpoint),
                distributed_plan=DistributedEvaluationPlan.from_dict(_read(args.distributed_plan)),
                capacity_inputs=capacity_inputs,
            )
            result = {
                "schema": "forge/competitive-round-open/v1",
                "round_id": plan.round_id,
                "plan_hash": plan.plan_hash(),
                "parent_state_hash": plan.parent_state_hash,
                "status": "OPENED" if created else "ALREADY_OPEN",
            }
        else:
            result = lifecycle.expire_round(
                RoundMailbox(store, Sr25519Verifier()),
                plan,
                finalized_block=args.finalized_block,
                finalized_hash=args.finalized_hash,
            )
    _write(args.out, result)
    print(json.dumps(result, allow_nan=False))
    return 0
