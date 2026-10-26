#!/usr/bin/env python3
"""Provide Lium list/get/delete to the existing scoped deadline supervisor."""

import argparse
import json
from pathlib import Path

from forge.evaluation.pilot_lium import LiumClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership", required=True, type=Path)
    parser.add_argument("resource", choices=["pod"])
    parser.add_argument("command", choices=["list", "get", "delete"])
    parser.add_argument("pod_id", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("-o", choices=["json"], default="json")
    args = parser.parse_args()
    ownership = json.loads(args.ownership.read_text())
    client = LiumClient(Path(__file__).resolve().parents[1] / ".env")
    print(json.dumps(client.provider_command(args.command, args.pod_id, ownership)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
