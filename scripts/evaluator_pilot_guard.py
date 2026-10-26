#!/usr/bin/env python3
"""Calculate a pilot deadline, or supervise closure of one already-owned pod."""

import argparse
import json
from pathlib import Path

from forge.evaluation.pilot_guard import guard_envelope, supervise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--envelope", required=True, type=Path)
    plan.add_argument("--out", required=True, type=Path)
    run = sub.add_parser("supervise")
    run.add_argument("--ownership", required=True, type=Path)
    run.add_argument("--state", required=True, type=Path)
    run.add_argument("--provider-executable", default="runpodctl")
    args = parser.parse_args()
    if args.command == "plan":
        result = guard_envelope(json.loads(args.envelope.read_text()))
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(result))
        return 0
    return supervise(
        json.loads(args.ownership.read_text()), args.state, provider=args.provider_executable
    )


if __name__ == "__main__":
    raise SystemExit(main())
