#!/usr/bin/env python3
"""Freeze or report local H3 canonical-forward evidence; no GPU/network operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.inference.forward_eval import ForwardEvidenceError, calibrate, report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("calibrate")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("report")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--calibration", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (
            calibrate(args.root, args.out)
            if args.command == "calibrate"
            else report(args.root, args.calibration)
        )
    except ForwardEvidenceError as exc:
        result = {
            "schema": "forge/inference-forward-report/v1",
            "status": "INVALID",
            "reason": str(exc),
        }
    if args.command == "report":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(result.get("status", "CALIBRATED"))


if __name__ == "__main__":
    main()
