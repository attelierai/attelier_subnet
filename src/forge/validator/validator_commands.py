"""Register validator CLI arguments without executing work."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_commands(commands: argparse._SubParsersAction) -> None:
    manifest = commands.add_parser("model-manifest", help="inspect immutable public model files")
    manifest.add_argument("--backend", choices=("hf", "hippius"), default="hf")
    manifest.add_argument("--repo", required=True)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--role", choices=("pipeline", "full_transformer", "lora"), required=True)
    manifest.add_argument("--base-manifest", type=Path)
    manifest.add_argument("--steps", type=int, default=16)
    manifest.add_argument("--guidance-scale", type=float, default=5.0)
    manifest.add_argument("--scheduler", choices=("uniform_euler", "original_unipc"))
    manifest.add_argument("--out", type=Path, required=True)

    declaration = commands.add_parser(
        "prepare-submission", help="prepare immutable declaration and commitment"
    )
    declaration.add_argument("--manifest", type=Path, required=True)
    declaration.add_argument("--out", type=Path, required=True)

    public_plan = commands.add_parser(
        "public-round-plan", help="read public model commitments at finalized close"
    )
    public_plan.add_argument("--network", default="test")
    public_plan.add_argument("--netuid", type=int, required=True)
    public_plan.add_argument("--close-block", type=int, required=True)
    public_plan.add_argument("--entropy-block", type=int, required=True)
    public_plan.add_argument("--contract", type=Path, required=True)
    public_plan.add_argument("--out", type=Path, required=True)
    public_plan.add_argument("--prompt-pack", type=Path)
    public_plan.add_argument("--pack-hotkey")
    public_plan.add_argument(
        "--stage", choices=("screen", "duel", "confirmation"), default="screen"
    )
    public_plan.add_argument("--candidate-hotkey")
    public_plan.add_argument("--finalist-lock-block", type=int)
    public_plan.add_argument("--incumbent-state", type=Path)
    public_plan.add_argument("--finalist-lock", type=Path)
    public_plan.add_argument("--finalist-lock-hotkey")
    public_plan.add_argument(
        "--calibration-plan",
        type=Path,
        help="preregistered calibration plan; required for confirmation",
    )

    fetch = commands.add_parser("fetch-model", help="download and hash-check a public snapshot")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--destination", type=Path, required=True)

    preflight = commands.add_parser(
        "preflight-round", help="validate and deduplicate every public candidate before GPU work"
    )
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--base-manifest", type=Path, required=True)
    preflight.add_argument("--cache-dir", type=Path, required=True, help="remote GPU volume")
    preflight.add_argument("--out", type=Path, required=True)

    submission_preflight = commands.add_parser(
        "preflight-submission",
        help="check public candidate bytes and recipe without a round or GPU",
    )
    submission_preflight.add_argument("--manifest", type=Path, required=True)
    submission_preflight.add_argument("--base-manifest", type=Path, required=True)
    submission_preflight.add_argument("--contract", type=Path, required=True)
    submission_preflight.add_argument("--cache-dir", type=Path, required=True)
    submission_preflight.add_argument("--out", type=Path, required=True)

    resolve = commands.add_parser("resolve-submission", help="resolve public v2 chain commitment")
    resolve.add_argument("--commitment", required=True)
    resolve.add_argument("--out", type=Path, required=True)

    plan = commands.add_parser("panel-plan", help="freeze contract and prescribed prompt slots")
    plan.add_argument("--contract", type=Path, required=True)
    plan.add_argument("--round-id", required=True)
    plan.add_argument(
        "--campaign-id", help="shared identity for separate calibration/evaluation panels"
    )
    plan.add_argument("--stage", choices=("screen", "duel", "confirmation"), required=True)
    plan.add_argument("--block-hash", required=True, help="finalized post-admission entropy")
    plan.add_argument("--out", type=Path, required=True)

    render = commands.add_parser("render-panel", help="validator-owned real Wan GPU execution")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--base-manifest", type=Path, required=True)
    render.add_argument("--cache-dir", type=Path, required=True, help="remote GPU volume")
    render.add_argument("--plan", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--preflight", type=Path, help="required for public candidate rendering")
    render.add_argument("--experimental-adapter", type=Path, help="local training test; ineligible")
    render.add_argument("--reference", action="store_true", help="fixed quality reference steps")
    render.add_argument(
        "--scheduler",
        choices=("uniform_euler", "original_unipc"),
        default=None,
        help="optional assertion of the scheduler committed in the manifest",
    )

    paired = commands.add_parser(
        "render-paired", help="stage-bound AB screening or ABBA confirmation timing"
    )
    paired.add_argument("--manifest", type=Path, required=True, help="challenger manifest")
    paired.add_argument("--incumbent-manifest", type=Path, required=True)
    paired.add_argument("--base-manifest", type=Path, required=True)
    paired.add_argument("--cache-dir", type=Path, required=True, help="remote GPU volume")
    paired.add_argument("--plan", type=Path, required=True)
    paired.add_argument("--out", type=Path, required=True)
    paired.add_argument("--preflight", type=Path, help="required for public candidate timing")
    paired.add_argument("--experimental-adapter", type=Path, help="local training test; ineligible")

    policy = commands.add_parser("vision-policy", help="write the frozen automatic vision policy")
    policy.add_argument("--out", type=Path, required=True)

    judge = commands.add_parser("judge-panels", help="run explicit Luna video comparisons")
    judge.add_argument("--plan", type=Path, required=True)
    judge.add_argument("--golden", type=Path, required=True)
    judge.add_argument("--incumbent", type=Path, required=True)
    judge.add_argument("--challenger", type=Path, required=True)
    judge.add_argument("--out", type=Path, required=True)
    judge.add_argument("--api-budget", type=float, required=True)
    judge.add_argument("--ledger", type=Path, required=True)
    judge.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    judge.add_argument("--workers", type=int, default=1, choices=range(1, 9))

    controls = commands.add_parser(
        "judge-controls", help="run explicit automatic calibration controls"
    )
    controls.add_argument("--plan", type=Path, required=True)
    controls.add_argument("--golden", type=Path, required=True)
    controls.add_argument("--out", type=Path, required=True)
    controls.add_argument("--api-budget", type=float, required=True)
    controls.add_argument("--ledger", type=Path, required=True)
    controls.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    controls.add_argument("--max-prompts", type=int)
    controls.add_argument("--workers", type=int, default=1, choices=range(1, 9))

    automatic = commands.add_parser(
        "compare-automatic", help="gate automatic quality and ABBA timing evidence"
    )
    automatic.add_argument("--plan", type=Path, required=True)
    automatic.add_argument("--policy", type=Path, required=True)
    automatic.add_argument("--quality", type=Path, required=True)
    automatic.add_argument("--timing", type=Path, required=True)
    automatic.add_argument("--calibration", type=Path, required=True)
    automatic.add_argument("--preflight", type=Path, help="required outside --development")
    automatic.add_argument("--out", type=Path, required=True)
    automatic.add_argument("--development", action="store_true")

    compare = commands.add_parser("compare-panels", help="speed plus calibrated quality gate")
    compare.add_argument("--golden", type=Path, required=True, help="fixed quality anchor report")
    compare.add_argument("--incumbent", type=Path, required=True)
    compare.add_argument("--challenger", type=Path, required=True)
    compare.add_argument("--policy", type=Path, required=True, help="trusted evaluator/runtime IDs")
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
