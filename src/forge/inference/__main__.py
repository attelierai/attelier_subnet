from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.chain import BittensorChain
from forge.inference.chain import atomic_json, configuration_plan, configure, snapshot
from forge.inference.dashboard import project
from forge.inference.identity import attest, bind_registry
from forge.inference.publication import publish, reconcile
from forge.inference.service import Validator


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference mechanism 1; private testnet pilot")
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "configuration-plan",
            "export-registry",
            "configure",
            "attest",
            "enqueue",
            "step",
            "review",
            "scores",
            "dashboard",
            "publish",
            "reconcile",
        ],
    )
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--publication", type=Path, help="Existing mechanism-1 publication receipt")
    parser.add_argument(
        "--observations", type=Path, help="Measured per-miner state and observation time"
    )
    parser.add_argument("--batch")
    parser.add_argument("--job-id")
    parser.add_argument("--budget-micros", type=int)
    parser.add_argument("--ceiling-micros", type=int)
    parser.add_argument("--wallet")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--miner-id")
    parser.add_argument("--expires", type=float)
    parser.add_argument("--execute", action="store_true", help="Required for a chain mutation")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    if config and config["netuid"] != args.netuid:
        parser.error("Configuration netuid mismatch")
    chain = None
    validator = None
    value = None
    try:
        if args.action in (
            "preflight",
            "configuration-plan",
            "export-registry",
            "configure",
            "publish",
            "reconcile",
        ):
            chain = BittensorChain("test", args.netuid)
        if args.action in ("enqueue", "step", "review", "scores", "dashboard", "publish"):
            if not (config and args.directory and args.budget_micros):
                parser.error("--config, --directory and --budget-micros are required")
            validator = Validator(args.directory, config, args.budget_micros)
        if args.action in ("configure", "publish", "attest"):
            from bittensor import Wallet

            if not args.wallet or not args.out:
                parser.error("--wallet and --out required")
            wallet = Wallet(name=args.wallet, hotkey=args.hotkey)
            if args.action in ("configure", "publish") and not args.execute:
                parser.error("Chain writes require --execute")
        if args.action in ("preflight", "configuration-plan", "export-registry"):
            value = snapshot(chain)
            if args.action == "configuration-plan":
                value = configuration_plan(value)
            elif args.action == "export-registry" and config:
                value = bind_registry(value, config)
        elif args.action == "configure":
            value = configure(chain, wallet, args.out)
        elif args.action == "attest":
            if not config or not args.expires:
                parser.error("--config and --expires required")
            miner = next(m for m in config["miners"] if m["id"] == args.miner_id)
            value = attest(wallet, miner, args.netuid, args.expires)
        elif args.action == "enqueue":
            if not args.input or not args.batch or not args.ceiling_micros:
                parser.error("--input, --batch and --ceiling-micros required")
            value = {
                "jobs": validator.enqueue(
                    args.batch, json.loads(args.input.read_text()), ceiling=args.ceiling_micros
                )
            }
        elif args.action == "step":
            rows = validator.step()
            value = [
                {"id": r["id"], "miner": r["miner"], "status": r["status"], "reason": r["reason"]}
                for r in rows
            ]
        elif args.action == "review":
            if not args.input or not args.job_id:
                parser.error("--input and --job-id required")
            validator.review(args.job_id, json.loads(args.input.read_text()))
            value = {"reviewed": args.job_id}
        elif args.action in ("scores", "publish", "dashboard"):
            if not args.batch:
                parser.error("--batch required")
            if args.action == "scores":
                value = validator.scores(args.batch)
            elif args.action == "publish":
                value = publish(chain, wallet, validator, args.batch, args.out)
            else:
                if not args.input:
                    parser.error("--input must name a measured budget projection")
                value = project(
                    validator,
                    args.batch,
                    json.loads(Path(config["registrationFile"]).read_text()),
                    json.loads(args.input.read_text()),
                    json.loads(args.publication.read_text()) if args.publication else None,
                    json.loads(args.observations.read_text()) if args.observations else None,
                )
        elif args.action == "reconcile":
            if not args.out or not args.out.exists():
                parser.error("--out must name the existing publication receipt")
            value = reconcile(chain, json.loads(args.out.read_text()), args.out)
        if args.out and args.action not in ("configure", "publish", "reconcile"):
            atomic_json(args.out, value)
        print(json.dumps(value, sort_keys=True))
    finally:
        if validator:
            validator.close()
        if chain:
            chain.close()


if __name__ == "__main__":
    main()
