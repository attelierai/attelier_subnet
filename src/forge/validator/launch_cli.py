"""Wan testnet operator commands with explicit observational and live modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.launch_config import LaunchConfig
from forge.validator.launch_service import LaunchServiceStore, atomic_json


def preflight(config: LaunchConfig, contract: ChallengeContract, chain: Any) -> dict[str, Any]:
    config.verify_contract(contract)
    if (chain.network, chain.netuid) != (config.network, config.netuid):
        raise ValueError("chain client belongs to another subnet")
    height = chain.finalized_height()
    block_hash = chain.finalized_block_hash(height)
    raw = chain.subtensor._client._substrate.raw
    genesis = str(chain.subtensor._call(raw.get_block_hash(0))).removeprefix("0x")
    if genesis != config.genesis_hash:
        raise ValueError("chain genesis differs from frozen configuration")
    runtime = chain.subtensor._call(raw.rpc_request("state_getRuntimeVersion", [block_hash]))
    if not isinstance(runtime, dict) or runtime.get("specName") != "node-subtensor":
        raise ValueError("unexpected chain runtime")
    code = chain.subtensor._call(raw.rpc_request("state_getStorage", ["0x3a636f6465", block_hash]))
    if not isinstance(code, str) or not code.startswith("0x") or len(code) <= 2:
        raise ValueError("finalized runtime code is unavailable")
    runtime_hash = hashlib.sha256(bytes.fromhex(code[2:])).hexdigest()
    committee, registered = chain.committee_snapshot(height), chain.uid_snapshot(height)
    if not committee or not registered:
        raise ValueError("testnet subnet has no registered eligible committee")
    from forge.chain.launch_fallback import qualify_fallback

    try:
        proof = qualify_fallback(
            chain, genesis_hash=genesis, runtime_hash=runtime_hash, finalized_block=height
        )
        fallback = {"status": "QUALIFIED", "proof": proof}
    except ValueError as error:
        fallback = {"status": "NOT_QUALIFIED", "reason": str(error)}
    return {
        "schema": "forge/launch-preflight/v1",
        "status": "READ_ONLY_VERIFIED",
        "network": config.network,
        "netuid": config.netuid,
        "genesis_hash": genesis,
        "config_hash": config.config_hash(),
        "contract_hash": contract.contract_hash(),
        "finalized_block": height,
        "finalized_block_hash": block_hash,
        "runtime": runtime,
        "runtime_hash": runtime_hash,
        "runtime_version_hash": hash_canonical("forge/runtime-version/v1", runtime),
        "registered_uids": {str(uid): hotkey for uid, hotkey in registered.items()},
        "committee": committee,
        "native_fallback": fallback,
        "publication_qualified": False,
    }


def add_commands(commands: argparse._SubParsersAction) -> None:
    launch = commands.add_parser("launch", help="bounded Wan testnet launch operator")
    sub = launch.add_subparsers(dest="launch_action", required=True)
    for name in ("initialize", "preflight", "status", "consensus", "export", "run"):
        parser = sub.add_parser(name)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--contract", type=Path, default=Path("configs/wan-mvp.json"))
        parser.add_argument("--out", type=Path)
        if name == "initialize":
            parser.add_argument("--root", type=Path, required=True)
            parser.add_argument("--bootstrap-reference", type=Path, required=True)
            parser.add_argument("--qualification-package", type=Path, required=True)
            parser.add_argument("--activation-block", type=int, required=True)
        elif name in {"status", "export"}:
            parser.add_argument("--state", type=Path, required=True)
            if name == "export":
                parser.add_argument("--destination", type=Path, required=True)
                parser.add_argument("--bootstrap-reference", type=Path, required=True)
                parser.add_argument("--base-snapshot", type=Path)
                for field in ("consensus-status", "publication", "capacity", "worker-health"):
                    parser.add_argument(f"--{field}", type=Path)
                parser.add_argument("--observed-cost-usd", type=float)
        elif name in {"consensus", "run"}:
            if name == "consensus":
                parser.add_argument("--round-dir", type=Path, required=True)
            else:
                parser.add_argument("--root", type=Path, required=True)
                parser.add_argument(
                    "--worker-command", type=Path, help="JSON array of operator argv"
                )
                parser.add_argument("--reserve-usd", type=float, default=0.0)
                parser.add_argument("--publication-config", type=Path)
                parser.add_argument("--export", type=Path)
            parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
            parser.add_argument("--wallet", help="local testnet wallet name, never key material")
            parser.add_argument(
                "--validator",
                action="append",
                default=[],
                metavar="LABEL=HOTKEY",
                help="repeat for each local validator; shared operator remains explicit",
            )
            parser.add_argument("--watch", action="store_true")
            parser.add_argument("--poll-seconds", type=float, default=12)


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command != "launch":
        return None
    config = LaunchConfig.read(args.config)
    contract = ChallengeContract.from_dict(json.loads(args.contract.read_text()))
    config.verify_contract(contract)
    if args.launch_action == "initialize":
        from forge.validator.launch_initialize import initialize

        result = initialize(
            config=config,
            contract=contract,
            root=args.root,
            bootstrap_reference=args.bootstrap_reference,
            qualification_package=args.qualification_package,
            activation_block=args.activation_block,
        )
    elif args.launch_action in {"status", "export"}:
        if not args.state.is_file():
            raise ValueError("state does not exist; status must not initialize another campaign")
        store = LaunchServiceStore(args.state, config)
        if args.launch_action == "status":
            result = store.snapshot()
        else:
            from forge.validator.launch_service_dashboard import project_launch_operations

            def read_optional(path: Path | None) -> dict | None:
                if path is None:
                    return None
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError("export evidence must be a JSON object")
                return value

            result = project_launch_operations(
                store.snapshot(),
                config,
                bootstrap_reference=read_optional(args.bootstrap_reference),
                consensus_status=read_optional(args.consensus_status),
                publication=read_optional(args.publication),
                capacity=read_optional(args.capacity),
                worker_health=read_optional(args.worker_health),
                observed_cost_usd=args.observed_cost_usd,
            )
            if args.base_snapshot:
                base = read_optional(args.base_snapshot)
                if (
                    base.get("schema") != "forge/dashboard/v2"
                    or base.get("network", {}).get("netuid") != config.netuid
                    or base.get("network", {}).get("name") != config.network
                ):
                    raise ValueError("base dashboard snapshot belongs to another contract/subnet")
                result = {**base, "launch_operations": result}
            atomic_json(args.destination, result)
    else:
        from forge.chain import BittensorChain

        chain = BittensorChain(config.network, config.netuid)
        try:
            result = preflight(config, contract, chain)
            if args.launch_action in {"consensus", "run"}:
                import math

                if args.mode == "live" and result["native_fallback"]["status"] != "QUALIFIED":
                    raise ValueError(
                        "live launch blocked before wallet/worker access: native fallback "
                        "is not qualified for the current finalized runtime"
                    )
                if not math.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60:
                    raise ValueError("poll interval must be between 1 and 60 seconds")
            if args.launch_action == "run":
                from forge.validator.launch_runtime import tick

                labels = _wallets(args) if args.mode == "live" else None
                worker = None
                if args.worker_command:
                    worker = json.loads(args.worker_command.read_text())
                    if (
                        not isinstance(worker, list)
                        or not worker
                        or any(not isinstance(value, str) or not value for value in worker)
                    ):
                        raise ValueError("worker command must be a nonempty JSON argv array")
                publication_step = None
                if args.publication_config:
                    from forge.validator.launch_publish import publication_callback

                    publication_step = publication_callback(
                        args.root,
                        configuration=json.loads(args.publication_config.read_text()),
                        config=config,
                        contract=contract,
                        chain=chain,
                        wallets=labels or {},
                        shadow=args.mode == "shadow",
                    )
                while True:
                    result = tick(
                        args.root,
                        config=config,
                        contract=contract,
                        chain=chain,
                        wallets=labels,
                        worker_command=worker,
                        reserve_usd=args.reserve_usd,
                        shadow=args.mode == "shadow",
                        publication_step=publication_step,
                        export_destination=args.export,
                    )
                    if args.out:
                        atomic_json(args.out, result)
                    if (
                        not args.watch
                        or result["status"]
                        in {"COMPLETE", "EXPIRED", "INCOMPLETE", "CERTIFIED_AWAITING_PUBLICATION"}
                        or args.mode == "shadow"
                    ):
                        break
                    time.sleep(args.poll_seconds)
            elif args.launch_action == "consensus":
                from forge.protocol.consensus_types import RoundPlan

                plan = RoundPlan.from_dict(
                    json.loads((args.round_dir / "round-plan.json").read_text())
                )
                if (plan.network, plan.netuid) != (
                    config.network,
                    config.netuid,
                ) or plan.contract_hash != config.contract_hash:
                    raise ValueError("round belongs to another configured campaign")
                if args.mode == "shadow":
                    result = {
                        **result,
                        "status": "SHADOW_NO_SIGNATURES",
                        "round_plan_hash": plan.plan_hash(),
                        "commit_deadline_block": plan.commit_deadline_block,
                        "reveal_deadline_block": plan.reveal_deadline_block,
                    }
                else:
                    from forge.validator.launch_consensus import LaunchConsensusRunner

                    labels = _wallets(args)
                    runner = LaunchConsensusRunner(args.round_dir, chain=chain, wallets=labels)
                    while True:
                        result = runner.tick()
                        if args.out:
                            atomic_json(args.out, result)
                        if not args.watch or result["status"] in {
                            "CERTIFIED_NOT_YET_WEIGHTS",
                            "EXPIRED_NO_CERTIFICATE",
                            "INCOMPLETE_NO_VOTE",
                            "INSUFFICIENT_COMMIT_WINDOW",
                        }:
                            break
                        time.sleep(args.poll_seconds)
        finally:
            chain.close()
    if args.out:
        atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def _wallets(args: argparse.Namespace) -> dict[str, Any]:
    import re

    import bittensor as bt

    if not args.wallet or not args.validator:
        raise ValueError("live operation requires explicit local testnet signers")
    labels = {}
    for token in args.validator:
        label, separator, hotkey = token.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", label)
            or not hotkey
            or label in labels
        ):
            raise ValueError("validators must be distinct safe LABEL=HOTKEY entries")
        labels[label] = bt.Wallet(name=args.wallet, hotkey=hotkey)
    return labels
