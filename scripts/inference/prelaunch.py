#!/usr/bin/env python3
"""Create an unsigned Attelier SN126 prelaunch reward simulation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.inference.prelaunch import read_json, simulate


def write_artifact(path: Path, encoded: str) -> None:
    """Atomically write only a new or byte-identical local simulation artifact."""
    if path.exists():
        try:
            if path.read_text(encoding="ascii") == encoded:
                return
        except OSError as error:
            raise ValueError("simulation_output_unreadable") from error
        raise ValueError("simulation_output_exists_with_different_artifact")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.link(temporary_path, path)
        temporary_path.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ValueError("simulation_output_write_failed") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="Write the unsigned artifact to this local path")
    args = parser.parse_args()
    artifact = simulate(read_json(args.ledger), read_json(args.policy), read_json(args.registry))
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
    if args.out:
        write_artifact(args.out, encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
