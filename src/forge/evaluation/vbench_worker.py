"""Fresh-interpreter entrypoint for one pinned VBench receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.evaluation.vbench_runner import run_vbench


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.request.read_text())
    required = {
        "video",
        "prompt",
        "output",
        "checkout",
        "cache",
        "weight_sha256",
        "dino_revision",
        "runner_image_digest",
        "plan_id",
        "prompt_id",
        "seed",
        "manifest_hash",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("VBench worker request fields are not exact")
    run_vbench(
        video=Path(raw["video"]),
        prompt=raw["prompt"],
        output=Path(raw["output"]),
        checkout=Path(raw["checkout"]),
        cache=Path(raw["cache"]),
        weight_sha256=raw["weight_sha256"],
        dino_revision=raw["dino_revision"],
        runner_image_digest=raw["runner_image_digest"],
        plan_id=raw["plan_id"],
        prompt_id=raw["prompt_id"],
        seed=raw["seed"],
        manifest_hash=raw["manifest_hash"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
