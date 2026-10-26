"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.protocol.consensus_types import LocalVerdict
from forge.protocol.contract import ChallengeContract
from forge.protocol.koth import CommitmentPosition
from forge.protocol.preconsensus import derive_weights_from_latest_certified_state
from forge.protocol.rewards import RewardRules, map_hotkey_weights_to_uids
from forge.sim import ModelProfile, SimNetwork, make_committee, profile

DEMO_CONTRACT = ChallengeContract(screen_prompts=16, duel_prompts=64, confirmation_prompts=128)


def demo(out: Path) -> dict:
    """One simulated round with 4 validators; D dissents and still adopts the king.

    Safe to re-run: stores are reused and nothing is crowned twice.
    """

    out.mkdir(parents=True, exist_ok=True)
    contract = DEMO_CONTRACT
    baseline = profile("public-baseline", contract, 0.70)
    entries = [
        ("miner-good", profile("good-lora", contract, 0.80), CommitmentPosition(50, 1)),
        ("miner-weak", profile("weak-lora", contract, 0.705), CommitmentPosition(50, 2)),
        (
            "miner-broken",
            profile("broken-lora", contract, 0.90, failure_rate=0.5),
            CommitmentPosition(51, 0),
        ),
    ]
    net = SimNetwork(contract, make_committee(str(out / "stores")), baseline)
    net.validators["D"].forced_verdict = LocalVerdict.KEEP
    already_done = net.validators["A"].store.count("certificates") > 0
    outcome = None
    if not already_done:
        outcome = net.run_round("round-1", entries, start_block=100, chain_time=1_000)

    store = net.validators["A"].store
    hotkey_plan = derive_weights_from_latest_certified_state(
        store,
        net.challenge_id,
        RewardRules(),
        {hotkey for hotkey, _, _ in entries},
        reference_chain_time=1_000,
    )
    uid_plan = map_hotkey_weights_to_uids(hotkey_plan, {1: "miner-good", 2: "miner-weak"})
    summary = {
        "contract_hash": contract.contract_hash(),
        "king_by_validator": {
            h: v.state.current_king_fingerprint for h, v in net.validators.items()
        },
        "state_hash_by_validator": {h: v.state.state_hash() for h, v in net.validators.items()},
        "crowns": store.crowns(net.challenge_id),
        "hotkey_weights": hotkey_plan.weights,
        "uid_weights": {str(uid): share for uid, share in uid_plan.weights.items()},
        "unallocated_share": uid_plan.unallocated_share,
        "rerun": already_done,
    }
    if outcome is not None:
        summary["stage_verdicts"] = outcome.stage_verdicts
        summary["local_verdicts"] = {h: r.verdict.value for h, r in outcome.reports.items()}
        summary["decision"] = outcome.decision.status.value if outcome.decision else None
        if outcome.plan is not None:
            (out / "round-plan.json").write_text(json.dumps(outcome.plan.to_dict(), indent=2))
        if outcome.certificate is not None:
            (out / "certificate.json").write_text(
                json.dumps(outcome.certificate.to_dict(), indent=2)
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    for validator in net.validators.values():
        validator.store.close()
    return summary


def load_models(path: Path) -> tuple[ModelProfile, dict[str, ModelProfile]]:
    """Simulated model registry: returns the baseline and {name: profile}."""

    config = json.loads(path.read_text())

    def build(item: dict) -> ModelProfile:
        return profile(
            item["name"],
            DEMO_CONTRACT,
            item["mean"],
            overrides=item.get("overrides"),
            failure_rate=item.get("failure_rate", 0.0),
        )

    return build(config["baseline"]), {item["name"]: build(item) for item in config["models"]}


def _chain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network", default="test")
    parser.add_argument("--netuid", type=int, required=True)


def _wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--hotkey", default="default")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forge")
    commands = parser.add_subparsers(dest="command", required=True)
    demo_parser = commands.add_parser("demo", help="run one simulated round (no GPU, no chain)")
    demo_parser.add_argument("--out", type=Path, default=Path("runs/demo"))

    status = commands.add_parser("chain-status", help="read-only subnet snapshot")
    _chain_args(status)

    commit = commands.add_parser("miner-commit", help="commit a simulated model on chain")
    _chain_args(commit)
    _wallet_args(commit)
    commit.add_argument("--model", required=True)
    commit.add_argument("--models", type=Path, default=Path("configs/testnet-models.json"))

    round_parser = commands.add_parser("testnet-round", help="one real round, simulated models")
    _chain_args(round_parser)
    _wallet_args(round_parser)
    round_parser.add_argument("--models", type=Path, default=Path("configs/testnet-models.json"))
    round_parser.add_argument("--state-dir", type=Path, default=Path("runs/testnet"))
    round_parser.add_argument("--unallocated-uid", type=int, default=None)

    from forge.chain import testnet_cli
    from forge.validator import (
        capacity_cli,
        competitive_bridge_cli,
        competitive_lifecycle_cli,
        creator_cli,
        dashboard_ingestion_cli,
        dashboard_v2,
        distributed_batch_cli,
        distributed_cli,
        distributed_mvp_runtime_cli,
        distributed_worker_cli,
        launch_cli,
        mvp_cli,
    )
    from forge.validator.validator_cli import handle_command
    from forge.validator.validator_commands import add_commands

    add_commands(commands)
    launch_cli.add_commands(commands)
    competitive_bridge_cli.add_commands(commands)
    competitive_lifecycle_cli.add_commands(commands)
    capacity_cli.add_commands(commands)
    creator_cli.add_commands(commands)
    dashboard_ingestion_cli.add_commands(commands)
    testnet_cli.add_commands(commands)
    mvp_cli.add_commands(commands)
    distributed_batch_cli.add_commands(commands)
    distributed_cli.add_commands(commands)
    distributed_worker_cli.add_commands(commands)
    distributed_mvp_runtime_cli.add_commands(commands)
    dashboard_v2.add_commands(commands)
    args = parser.parse_args(argv)
    for handler in (
        launch_cli.handle_command,
        competitive_bridge_cli.handle_command,
        competitive_lifecycle_cli.handle_command,
        capacity_cli.handle_command,
        creator_cli.handle_command,
        dashboard_ingestion_cli.handle_command,
        dashboard_v2.handle_command,
        distributed_batch_cli.handle_command,
        distributed_cli.handle_command,
        distributed_worker_cli.handle_command,
        distributed_mvp_runtime_cli.handle_command,
    ):
        result = handler(args)
        if result is not None:
            return result
    mvp_result = mvp_cli.handle_command(args)
    if mvp_result is not None:
        return mvp_result
    testnet_result = testnet_cli.handle_command(args)
    if testnet_result is not None:
        return testnet_result
    real_result = handle_command(args)
    if real_result is not None:
        return real_result
    if args.command == "demo":
        summary = demo(args.out)
        print(json.dumps({k: summary[k] for k in ("crowns", "hotkey_weights", "rerun")}, indent=2))
        return 0

    import bittensor as bt

    from forge.chain import BittensorChain, encode_submission

    chain = BittensorChain(args.network, args.netuid)
    try:
        if args.command == "chain-status":
            height = chain.finalized_height()
            print(
                json.dumps(
                    {
                        "finalized_block": height,
                        "uids": chain.uid_snapshot(height),
                        "committee_rao": chain.committee_snapshot(height),
                    },
                    indent=2,
                )
            )
        elif args.command == "miner-commit":
            _, models = load_models(args.models)
            wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
            text = encode_submission(models[args.model].fingerprint())
            print(json.dumps({"extrinsic": chain.publish(wallet, text), "commitment": text}))
        elif args.command == "testnet-round":
            from forge.chain.testnet import TestnetRound
            from forge.protocol.store import Store
            from forge.validator.board import FileBoard

            baseline, models = load_models(args.models)
            args.state_dir.mkdir(parents=True, exist_ok=True)
            runner = TestnetRound(
                chain=chain,
                validator_wallet=bt.Wallet(name=args.wallet, hotkey=args.hotkey),
                board=FileBoard(args.state_dir / "board"),
                store=Store(args.state_dir / "validator.sqlite"),
                contract=DEMO_CONTRACT,
                baseline=baseline,
                models={model.fingerprint(): model for model in models.values()},
                unallocated_uid=args.unallocated_uid,
            )
            summary = runner.run()
            (args.state_dir / f"{summary['round_id']}.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str)
            )
            print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    finally:
        chain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
