#!/usr/bin/env python3
"""Gateway/M1 signing, planning, and explicitly-gated testnet publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.chain import BittensorChain, Sr25519Signer
from forge.inference.gateway_chain import live_anchor, master_preflight
from forge.inference.gateway_epoch import light_plan, master_window, sign_gateway
from forge.inference.gateway_publication import (
    publish_gateway_weights,
    publish_master_commitment,
    reconcile_gateway_weights,
    reconcile_master_commitment,
)
from forge.inference.gateway_transport import fetch_gateway_master


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(value):
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def add_master_source(parser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--master-envelope", type=Path)
    source.add_argument("--master-url")


def master_source(args):
    return fetch_gateway_master(args.master_url) if args.master_url else read(args.master_envelope)


def wallet(args):
    try:
        from bittensor import Wallet
    except ModuleNotFoundError as exc:
        raise SystemExit("bittensor SDK is required for master-sign") from exc
    return Sr25519Signer(
        Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=str(args.wallet_path))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gateway = commands.add_parser("gateway-sign")
    gateway.add_argument("--body", type=Path, required=True)
    gateway.add_argument("--private-key", type=Path, required=True)
    master = commands.add_parser("master-sign")
    master.add_argument("--envelope", type=Path, required=True)
    master.add_argument("--policy", type=Path, required=True)
    master.add_argument("--finalized-registry-block-hash", required=True)
    master.add_argument("--wallet-name", required=True)
    master.add_argument("--wallet-hotkey", required=True)
    master.add_argument("--wallet-path", type=Path, required=True)
    master.add_argument("--finalized-registry-block", type=int, default=0)
    master_live = commands.add_parser(
        "master-live-sign", help="read finalized testnet state, then sign; never broadcasts"
    )
    for name in ("envelope", "policy"):
        master_live.add_argument("--" + name, type=Path, required=True)
    master_live.add_argument("--wallet-name", required=True)
    master_live.add_argument("--wallet-hotkey", required=True)
    master_live.add_argument("--wallet-path", type=Path, required=True)
    master_live.add_argument("--preflight-out", type=Path)
    light = commands.add_parser("light-plan")
    for name in ("master_envelope", "policy", "anchor", "metagraph"):
        light.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    light.add_argument("--state", type=Path)
    light_live = commands.add_parser(
        "light-live-plan", help="read latest M1 commitment and current UIDs; never broadcasts"
    )
    add_master_source(light_live)
    light_live.add_argument("--policy", type=Path, required=True)
    light_live.add_argument("--state", type=Path)
    light_live.add_argument("--preflight-out", type=Path)
    commit = commands.add_parser("master-commit", help="broadcast only with --execute")
    commit.add_argument("--master-envelope", type=Path, required=True)
    commit.add_argument("--policy", type=Path, required=True)
    commit.add_argument("--receipt", type=Path, required=True)
    commit.add_argument("--wallet-name", required=True)
    commit.add_argument("--wallet-hotkey", required=True)
    commit.add_argument("--wallet-path", type=Path, required=True)
    commit.add_argument("--execute", action="store_true")
    weights = commands.add_parser("light-publish", help="broadcast only with --execute")
    add_master_source(weights)
    weights.add_argument("--policy", type=Path, required=True)
    weights.add_argument("--state", type=Path)
    weights.add_argument("--receipt", type=Path, required=True)
    weights.add_argument("--wallet-name", required=True)
    weights.add_argument("--wallet-hotkey", required=True)
    weights.add_argument("--wallet-path", type=Path, required=True)
    weights.add_argument("--execute", action="store_true")
    weights.add_argument("--publication-authority-directory", type=Path, required=True)
    reconcile = commands.add_parser("master-reconcile")
    reconcile.add_argument("--receipt", type=Path, required=True)
    light_reconcile = commands.add_parser(
        "light-reconcile", help="read/reconcile one CR light receipt; never broadcasts"
    )
    add_master_source(light_reconcile)
    light_reconcile.add_argument("--policy", type=Path, required=True)
    light_reconcile.add_argument("--receipt", type=Path, required=True)
    light_reconcile.add_argument("--state", type=Path, required=True)
    light_reconcile.add_argument("--publication-authority-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gateway-sign":
        write(sign_gateway(read(args.body), args.private_key))
    elif args.command == "master-sign":
        write(
            master_window(
                read(args.envelope),
                read(args.policy),
                wallet(args),
                args.finalized_registry_block_hash,
                finalized_block=args.finalized_registry_block,
            )
        )
    elif args.command == "master-live-sign":
        chain = BittensorChain("test", 578)
        try:
            preflight = master_preflight(chain, read(args.policy), wallet(args).validator_hotkey)
        finally:
            chain.close()
        if args.preflight_out:
            args.preflight_out.write_text(
                json.dumps(preflight, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        write(
            master_window(
                read(args.envelope),
                read(args.policy),
                wallet(args),
                preflight["finalizedRegistryBlockHash"],
                finalized_block=preflight["finalizedRegistryBlock"],
            )
        )
    elif args.command == "light-plan":
        write(
            light_plan(
                read(args.master_envelope),
                read(args.policy),
                read(args.anchor),
                read(args.metagraph),
                read(args.state) if args.state else None,
            )
        )
    elif args.command == "light-live-plan":
        chain = BittensorChain("test", 578)
        try:
            master = master_source(args)
            anchor, metagraph, preflight = live_anchor(chain, master, read(args.policy))
        finally:
            chain.close()
        if args.preflight_out:
            args.preflight_out.write_text(
                json.dumps(preflight, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        write(
            light_plan(
                master,
                read(args.policy),
                anchor,
                metagraph,
                read(args.state) if args.state else None,
            )
        )
    elif args.command == "master-commit":
        if not args.execute:
            raise SystemExit("master-commit requires --execute")
        chain = BittensorChain("test", 578)
        try:
            write(
                publish_master_commitment(
                    chain,
                    __import__("bittensor").Wallet(
                        name=args.wallet_name, hotkey=args.wallet_hotkey, path=str(args.wallet_path)
                    ),
                    read(args.master_envelope),
                    read(args.policy),
                    args.receipt,
                )
            )
        finally:
            chain.close()
    elif args.command == "light-publish":
        if not args.execute:
            raise SystemExit("light-publish requires --execute")
        chain = BittensorChain("test", 578)
        try:
            write(
                publish_gateway_weights(
                    chain,
                    __import__("bittensor").Wallet(
                        name=args.wallet_name, hotkey=args.wallet_hotkey, path=str(args.wallet_path)
                    ),
                    master_source(args),
                    read(args.policy),
                    args.receipt,
                    args.state,
                    args.publication_authority_directory,
                )
            )
        finally:
            chain.close()
    elif args.command == "light-reconcile":
        chain = BittensorChain("test", 578)
        try:
            write(
                reconcile_gateway_weights(
                    chain,
                    master_source(args),
                    read(args.policy),
                    args.receipt,
                    args.state,
                    args.publication_authority_directory,
                )
            )
        finally:
            chain.close()
    else:
        chain = BittensorChain("test", 578)
        try:
            write(reconcile_master_commitment(chain, read(args.receipt), args.receipt))
        finally:
            chain.close()


if __name__ == "__main__":
    main()
