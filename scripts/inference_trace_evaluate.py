#!/usr/bin/env python3
"""Analyze a frozen local H3 trace experiment; it performs no GPU or network work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.inference.trace_eval import TraceEvidenceError, analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.experiment)
    except TraceEvidenceError as exc:
        result = {
            "schema": "forge/inference-trace-analysis/v1",
            "status": "INVALID",
            "claim": "none",
            "reason": str(exc),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(result["status"])


if __name__ == "__main__":
    main()
