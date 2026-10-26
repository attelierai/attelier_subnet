"""CLI for one frozen multi-challenger commitment and bounded peer reveal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.protocol.distributed_batch import DiagnosticBatch
from forge.validator.distributed_cli import _read, _write


def add_commands(commands: argparse._SubParsersAction) -> None:
    for name in ("plan", "index", "anchor", "resolve"):
        command = commands.add_parser(f"distributed-batch-{name}")
        command.add_argument("--out", type=Path, required=True)
        if name == "plan":
            command.add_argument("--plan", type=Path, action="append", required=True)
            command.add_argument("--commit-deadline-block", type=int, required=True)
            command.add_argument("--reveal-deadline-block", type=int, required=True)
        else:
            command.add_argument("--batch", type=Path, required=True)
        if name in {"index", "anchor"}:
            command.add_argument("--wallet", required=True)
            command.add_argument("--hotkey", required=True)
            command.add_argument("--wallet-path", default="~/.bittensor/wallets")
        if name == "index":
            command.add_argument("--report", type=Path, action="append", required=True)
        if name in {"anchor", "resolve"}:
            command.add_argument("--netuid", type=int, required=True)
        if name == "anchor":
            command.add_argument("--index", type=Path, required=True)
            command.add_argument("--execute", action="store_true")
        if name == "resolve":
            command.add_argument("--peer", action="append", required=True)


def handle_command(args) -> int | None:
    if not args.command.startswith("distributed-batch-"):
        return None
    if args.out.exists():
        raise ValueError("output exists; reconcile prior evidence instead of retrying")
    if args.command == "distributed-batch-plan":
        batch = DiagnosticBatch(
            tuple(DistributedEvaluationPlan.from_dict(_read(path)) for path in args.plan),
            args.commit_deadline_block,
            args.reveal_deadline_block,
        )
        _write(args.out, batch.to_dict())
    else:
        from forge.chain import BittensorChain, Sr25519Signer, Sr25519Verifier
        from forge.protocol.distributed_batch import resolve_batch, sign_batch_index

        batch = DiagnosticBatch.from_dict(_read(args.batch))
        if args.command == "distributed-batch-resolve":
            peers = {}
            for item in args.peer:
                hotkey, separator, url = item.partition("=")
                if not separator or hotkey not in batch.plans[0].committee or hotkey in peers:
                    raise ValueError("peer endpoints must uniquely name frozen validators")
                peers[hotkey] = url.rstrip("/")
            chain = BittensorChain("test", args.netuid)
            try:
                result = resolve_batch(chain, batch, peers, Sr25519Verifier())
                _write(args.out, result)
            finally:
                chain.close()
        else:
            import bittensor as bt

            wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path)
            if args.command == "distributed-batch-index":
                result = sign_batch_index(
                    batch,
                    [_read(path) for path in args.report],
                    Sr25519Signer(wallet),
                    Sr25519Verifier(),
                )
                _write(args.out, result)
            else:
                _anchor(args, batch, wallet)
    print(json.dumps({"output": str(args.out), "batch_hash": batch.batch_hash()}))
    return 0


def _anchor(args, batch, wallet):
    from forge.chain import BittensorChain, Sr25519Verifier, encode_report_commit
    from forge.protocol.distributed_batch import verify_batch_index
    from forge.protocol.distributed_transport import report_digest, verify_plan_chain_context

    index = _read(args.index)
    hotkey = wallet.hotkey.ss58_address
    verify_batch_index(batch, index, hotkey, Sr25519Verifier())
    digest = report_digest(index)
    commitment = encode_report_commit(batch.round_token(), digest)
    chain = BittensorChain("test", args.netuid)
    try:
        verify_plan_chain_context(chain, batch.plans[0], batch.commit_deadline_block)
        block = chain.finalized_height()
        if hotkey not in chain.committee_snapshot(block):
            raise ValueError("publisher no longer has a testnet validator permit")
        if chain.head() >= batch.commit_deadline_block:
            raise ValueError("batch commitment window has closed")
        result = {
            "schema": "forge/diagnostic-batch-anchor/v1",
            "network": "test",
            "netuid": args.netuid,
            "hotkey": hotkey,
            "batch_hash": batch.batch_hash(),
            "index_sha256": digest,
            "commitment": commitment,
            "preflight_block": block,
            "status": "SUBMITTING" if args.execute else "PREFLIGHT",
            "reward_eligible": False,
        }
        _write(args.out, result)
        if args.execute:
            transaction = chain.publish(wallet, commitment)
            result.update(status="SUBMITTED", extrinsic_id=transaction)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            finalized, observed = chain.finalized_publication(hotkey, commitment, transaction)
            result.update(
                status="FINALIZED"
                if observed.block <= batch.commit_deadline_block
                else "FINALIZED_LATE_INELIGIBLE",
                commitment_block=observed.block,
                finalized_block=finalized,
                finalized_block_hash=chain.finalized_block_hash(finalized),
            )
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            if result["status"] != "FINALIZED":
                raise ValueError("commitment finalized after the fixed cutoff; do not retry")
    finally:
        chain.close()
