"""CLI registration for the diagnostic distributed GPU worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_commands(commands: argparse._SubParsersAction) -> None:
    inspect = commands.add_parser(
        "distributed-worker-plan", help="validate diagnostic GPU jobs without CUDA"
    )
    inspect.add_argument("--jobs", type=Path, required=True)
    run = commands.add_parser(
        "distributed-worker", help="run assigned diagnostic Wan jobs on one GPU"
    )
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command not in {"distributed-worker-plan", "distributed-worker"}:
        return None
    from forge.models.distributed_worker import (
        inspect_diagnostic_jobs,
        load_diagnostic_jobs,
        run_diagnostic_jobs,
    )

    result = (
        inspect_diagnostic_jobs(args.jobs)
        if args.command == "distributed-worker-plan"
        else run_diagnostic_jobs(load_diagnostic_jobs(args.jobs), args.out)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
