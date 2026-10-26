#!/usr/bin/env python3
"""Fetch one public signed master envelope; no policy, chain, wallet, or writeback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.inference.gateway_transport import fetch_gateway_master


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = fetch_gateway_master(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    main()
