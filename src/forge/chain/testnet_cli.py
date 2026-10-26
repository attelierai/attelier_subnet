"""Explicit testnet publication of immutable public recipes and sealed pack hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

from forge.models.artifact_manifest import ArtifactReference
from forge.protocol.contract import ChallengeContract
from forge.protocol.prompt_pack import build_pack, pack_context, seal_pack
from forge.validator.submission import (
    V3SubmissionClaim,
    encode_v3_submission_commitment,
    resolve_v3_submission,
)


def add_commands(commands: argparse._SubParsersAction) -> None:
    seal = commands.add_parser("seal-prompt-pack", help="prepare a private pack and public digest")
    seal.add_argument("--entries", required=True, type=Path)
    seal.add_argument("--contract", required=True, type=Path)
    seal.add_argument("--round-id", required=True)
    seal.add_argument("--close-block", required=True, type=int)
    seal.add_argument("--netuid", required=True, type=int)
    seal.add_argument("--out", required=True, type=Path)
    for name, help_text in (
        ("testnet-publish-model", "commit a public HF model recipe on testnet"),
        ("testnet-publish-pack", "commit a sealed prompt pack digest on testnet"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--netuid", required=True, type=int)
        command.add_argument("--wallet", required=True)
        command.add_argument("--hotkey", default="default")
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--execute", action="store_true", help="submit after preflight")
        if name.endswith("model"):
            command.add_argument("--repo", required=True)
            command.add_argument("--revision", required=True)
            command.add_argument("--role", required=True, choices=("lora", "full_transformer"))
            command.add_argument("--steps", required=True, type=int)
            command.add_argument("--guidance-scale", default=5.0, type=float)
            command.add_argument("--scheduler", choices=("uniform_euler", "original_unipc"))
            command.add_argument("--contract", required=True, type=Path)
        else:
            command.add_argument("--pack", required=True, type=Path)

    validators = commands.add_parser(
        "testnet-validators",
        help="read-only receipt of the validator set at the latest finalized block",
    )
    validators.add_argument("--netuid", required=True, type=int)
    validators.add_argument("--out", required=True, type=Path)

    lock = commands.add_parser(
        "testnet-publish-finalist-lock", help="commit a confirmation finalist lock on testnet"
    )
    lock.add_argument("--netuid", required=True, type=int)
    lock.add_argument("--wallet", required=True)
    lock.add_argument("--hotkey", default="default")
    lock.add_argument("--plan", required=True, type=Path)
    lock.add_argument("--out", required=True, type=Path)
    lock.add_argument("--execute", action="store_true", help="publish after preflight")

    evaluation = commands.add_parser(
        "testnet-publish-evaluation",
        help="sign and optionally anchor automatic evaluation evidence on testnet",
    )
    evaluation.add_argument("--netuid", required=True, type=int)
    evaluation.add_argument("--wallet", required=True)
    evaluation.add_argument("--hotkey", default="default")
    evaluation.add_argument("--plan", required=True, type=Path)
    evaluation.add_argument("--policy", required=True, type=Path)
    evaluation.add_argument("--quality", required=True, type=Path)
    evaluation.add_argument("--timing", required=True, type=Path)
    evaluation.add_argument("--calibration", required=True, type=Path)
    evaluation.add_argument("--preflight", type=Path)
    evaluation.add_argument("--journal", required=True, type=Path)
    evaluation.add_argument("--out", required=True, type=Path)
    evaluation.add_argument("--development", action="store_true")
    evaluation.add_argument(
        "--execute", action="store_true", help="publish after signing and preflight"
    )


def validator_snapshot(chain) -> dict:
    """Validators with a permit at one finalized block. Public on-chain fields only:
    coldkeys are left out because they would link hotkeys to one owner."""
    try:
        height = chain.finalized_height()
        metagraph = chain._metagraph(height)
        observed_unix = chain.chain_time(height)
    finally:
        chain.close()
    rows = zip(
        metagraph["hotkeys"],
        metagraph["validator_permit"],
        metagraph["total_stake"],
        metagraph["last_update"],
        metagraph["active"],
        strict=True,
    )
    return {
        "schema": "forge/validator-snapshot/v1",
        "network": chain.network,
        "netuid": chain.netuid,
        "block": height,
        "observed_unix": observed_unix,
        "neurons": len(metagraph["hotkeys"]),
        "validators": [
            {
                "uid": uid,
                "hotkey": hotkey,
                "stake_rao": int(stake),
                "last_update_block": int(last_update),
                "active": bool(active),
            }
            for uid, (hotkey, permit, stake, last_update, active) in enumerate(rows)
            if permit
        ],
    }


def _new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def _update(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + secrets.token_hex(8))
    _new(temporary, value)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command == "seal-prompt-pack":
        contract = ChallengeContract.from_dict(json.loads(args.contract.read_text()))
        pack = build_pack(
            pack_context(
                "test", args.netuid, args.round_id, contract.contract_hash(), args.close_block
            ),
            secrets.token_hex(32),
            json.loads(args.entries.read_text()),
        )
        _new(args.out, pack)
        print(json.dumps({"pack_hash": seal_pack(pack), "entries": len(pack["entries"])}))
        return 0
    if args.command == "testnet-validators":
        from forge.chain import BittensorChain

        _new(args.out, validator_snapshot(BittensorChain("test", args.netuid)))
        print(json.dumps({"output": str(args.out)}))
        return 0
    if args.command not in {
        "testnet-publish-model",
        "testnet-publish-pack",
        "testnet-publish-evaluation",
        "testnet-publish-finalist-lock",
    }:
        return None
    # These operations are intentionally testnet-only. They never touch weights
    # or coldkey signing; publication is explicit and preserves an intent receipt.
    import bittensor as bt

    from forge.chain import BittensorChain, encode_prompt_pack_commit

    if args.command == "testnet-publish-finalist-lock":
        from forge.chain import encode_finalist_lock_commitment
        from forge.protocol.finalist_lock import build_finalist_lock, finalist_lock_hash
        from forge.validator.round_planner import resolve_closed_admissions

        if args.out.exists():
            raise ValueError(
                "finalist-lock receipt already exists; reconcile on chain before retrying"
            )
        plan = _json_object(args.plan, "plan")
        candidate, incumbent = plan.get("candidate"), plan.get("incumbent")
        if not isinstance(candidate, dict) or not isinstance(incumbent, dict):
            raise ValueError("finalist-lock plan requires candidate and incumbent")
        lock = build_finalist_lock(
            {
                "schema": "forge/finalist-lock/v1",
                "network": plan.get("network"),
                "netuid": plan.get("netuid"),
                "round_id": plan.get("round_id"),
                "close_block": plan.get("close_block"),
                "close_block_hash": plan.get("close_block_hash"),
                "contract_hash": plan.get("contract_hash"),
                "candidate_manifest_hash": candidate.get("manifest_hash"),
                "candidate_hotkey": candidate.get("hotkey"),
                "candidate_admission_block": candidate.get("admission_block"),
                "incumbent_state_hash": incumbent.get("certified_state_hash"),
            }
        )
        chain = BittensorChain("test", args.netuid)
        try:
            wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
            height = chain.finalized_height()
            if plan.get("network") != "test" or plan.get("netuid") != args.netuid:
                raise ValueError("finalist-lock plan belongs to another testnet subnet")
            try:
                contract = ChallengeContract.from_dict(plan["contract"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("finalist-lock plan contract is invalid") from error
            if (
                plan.get("contract_hash") != contract.contract_hash()
                or plan.get("admission_algorithm") != "forge/closed-admission/v1"
            ):
                raise ValueError("finalist-lock plan admission binding is invalid")
            closed_admissions = resolve_closed_admissions(chain, contract, lock["close_block"])
            admissions = closed_admissions.admitted
            if plan.get("admitted") != admissions:
                raise ValueError("finalist-lock plan admissions are not canonical at close")
            candidate_match = [
                item
                for item in admissions
                if item["hotkey"] == lock["candidate_hotkey"]
                and item["manifest_hash"] == lock["candidate_manifest_hash"]
                and item["block"] == lock["candidate_admission_block"]
            ]
            if len(candidate_match) != 1:
                raise ValueError("finalist-lock candidate is not canonically admitted at close")
            if chain.finalized_block_hash(lock["close_block"]) != lock["close_block_hash"]:
                raise ValueError("finalist-lock plan close hash differs from finalized chain")
            hotkey = wallet.hotkey.ss58_address
            committee = chain.committee_snapshot(lock["close_block"])
            if hotkey not in chain.uid_snapshot(height).values() or hotkey not in committee:
                raise ValueError(
                    "finalist-lock signer is not a registered close committee validator"
                )
            from forge.protocol.finalist_lock import first_valid_lock_after_close

            existing = first_valid_lock_after_close(
                chain,
                committee,
                admissions,
                network=lock["network"],
                netuid=lock["netuid"],
                round_id=lock["round_id"],
                close_block=lock["close_block"],
                close_hash=lock["close_block_hash"],
                contract_hash=lock["contract_hash"],
                incumbent_state_hash=lock["incumbent_state_hash"],
                upper_block=height,
            )
            if existing is not None:
                first_block, first_hotkey, first_lock = existing
                receipt = {
                    "schema": "forge/testnet-finalist-lock/v1",
                    "network": "test",
                    "netuid": args.netuid,
                    "lock": first_lock,
                    "lock_hash": finalist_lock_hash(first_lock),
                    "commitment_block": first_block,
                    "hotkey": first_hotkey,
                    "status": "FIRST_VALID_LOCK_ALREADY_EXISTS",
                }
                _new(args.out, receipt)
                print(json.dumps(receipt))
                return 2
            pack = plan.get("pack_commitment")
            if (
                isinstance(pack, dict)
                and pack.get("hotkey") == hotkey
                and chain.head() <= lock["close_block"]
            ):
                raise ValueError("refusing to replace an active sealed prompt-pack commitment")
            receipt = {
                "schema": "forge/testnet-finalist-lock/v1",
                "network": "test",
                "netuid": args.netuid,
                "lock": lock,
                "lock_hash": finalist_lock_hash(lock),
                "commitment": encode_finalist_lock_commitment(finalist_lock_hash(lock)),
                "preflight_block": height,
                "preflight_block_hash": chain.finalized_block_hash(height),
                "status": "LOCK_PENDING" if args.execute else "DRY_RUN",
            }
            _new(args.out, receipt)
            if args.execute:
                receipt["extrinsic_id"] = chain.publish(wallet, receipt["commitment"])
                receipt["status"] = "SUBMITTED_READBACK_PENDING"
                _update(args.out, receipt)
                observed = chain.commitment(wallet.hotkey.ss58_address, chain.finalized_height())
                if (
                    observed is None
                    or observed.kind != "finalist_lock"
                    or observed.fields != (receipt["lock_hash"],)
                ):
                    raise ValueError("finalist lock readback differs; do not retry")
                if observed.block <= lock["close_block"]:
                    receipt.update(
                        commitment_block=observed.block, status="FINALIZED_LATE_INELIGIBLE"
                    )
                    _update(args.out, receipt)
                    return 2
                first = first_valid_lock_after_close(
                    chain,
                    committee,
                    admissions,
                    network=lock["network"],
                    netuid=lock["netuid"],
                    round_id=lock["round_id"],
                    close_block=lock["close_block"],
                    close_hash=lock["close_block_hash"],
                    contract_hash=lock["contract_hash"],
                    incumbent_state_hash=lock["incumbent_state_hash"],
                    upper_block=chain.finalized_height(),
                )
                if first != (observed.block, hotkey, lock):
                    receipt.update(
                        commitment_block=observed.block, status="FIRST_VALID_LOCK_SUPERSEDED"
                    )
                    _update(args.out, receipt)
                    return 2
                receipt.update(commitment_block=observed.block, status="FINALIZED_LOCK")
                _update(args.out, receipt)
            print(json.dumps({key: receipt[key] for key in ("lock_hash", "commitment", "status")}))
            return 0
        finally:
            chain.close()

    if args.command == "testnet-publish-evaluation":
        from forge.chain import Sr25519Signer, Sr25519Verifier
        from forge.protocol.evaluation_anchor import (
            prepare_evaluation_anchor,
            verify_finalized_evaluation_anchor,
        )

        if args.out.exists():
            raise ValueError(
                "evaluation receipt already exists; reconcile on chain before retrying"
            )
        plan = _json_object(args.plan, "plan")
        policy = _json_object(args.policy, "policy")
        quality = _json_object(args.quality, "quality")
        timing = _json_object(args.timing, "timing")
        calibration = _json_object(args.calibration, "calibration")
        journal_sha256 = _sha256_file(args.journal)
        chain = BittensorChain("test", args.netuid)
        try:
            wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
            signer = Sr25519Signer(wallet)
            intent = prepare_evaluation_anchor(
                chain,
                signer,
                Sr25519Verifier(),
                plan,
                policy,
                quality,
                timing,
                calibration,
                evaluator_journal_sha256=journal_sha256,
                preflight=_json_object(args.preflight, "preflight") if args.preflight else None,
                development=args.development,
            )
            receipt = intent.receipt(status="ANCHOR_PENDING" if args.execute else "DRY_RUN")
            _new(args.out, receipt)
            if args.execute:
                if wallet.hotkey.ss58_address != intent.signed_record.validator_hotkey:
                    raise ValueError("signing key differs from the preflight validator identity")
                receipt["extrinsic_id"] = chain.publish(wallet, intent.commitment)
                receipt["status"] = "SUBMITTED_READBACK_PENDING"
                _update(args.out, receipt)
                observed, finalized, finalized_hash = verify_finalized_evaluation_anchor(
                    chain, intent
                )
                receipt.update(
                    status="FINALIZED_EVALUATION_EVIDENCE_ONLY_NOT_A_CERTIFICATE_OR_REWARD",
                    commitment_block=observed.block,
                    finalized_block=finalized,
                    finalized_block_hash=finalized_hash,
                )
                _update(args.out, receipt)
            print(
                json.dumps(
                    {
                        key: receipt[key]
                        for key in ("record_hash", "commitment", "status", "preflight_block")
                    }
                )
            )
            return 0
        finally:
            chain.close()

    if args.out.exists():
        raise ValueError("publication receipt already exists; reconcile on chain before retrying")
    metadata = {}
    pack = None
    if args.command.endswith("model"):
        contract = ChallengeContract.from_dict(json.loads(args.contract.read_text()))
        claim = V3SubmissionClaim(
            ArtifactReference("hf", args.repo, args.revision),
            args.role,
            {
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                **({"scheduler": args.scheduler} if args.scheduler is not None else {}),
            },
        )
        manifest = resolve_v3_submission(claim, contract)
        text = encode_v3_submission_commitment(claim)
        metadata = {"manifest": manifest.to_dict(), "manifest_hash": manifest.fingerprint()}
    else:
        pack = json.loads(args.pack.read_text())
        if pack["context"]["network"] != "test" or pack["context"]["netuid"] != args.netuid:
            raise ValueError("pack belongs to a different testnet subnet")
        text = encode_prompt_pack_commit(seal_pack(pack))
    chain = BittensorChain("test", args.netuid)
    try:
        height = chain.finalized_height()
        wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
        # This public companion file avoids accessing a private key for dry-run.
        public_path = (
            Path(wallet.path).expanduser() / wallet.name / "hotkeys" / (args.hotkey + "pub.txt")
        )
        hotkey = json.loads(public_path.read_text())["ss58Address"]
        if hotkey not in chain.uid_snapshot(height).values():
            raise ValueError("signer is not registered on this testnet subnet")
        if pack is not None:
            if hotkey not in chain.committee_snapshot(height):
                raise ValueError("pack signer is not a permitted validator")
            if chain.head() >= pack["context"]["close_block"]:
                raise ValueError("pack submission window is already closed")
        receipt = {
            "schema": "forge/testnet-publication/v1",
            "network": "test",
            "netuid": args.netuid,
            "hotkey": hotkey,
            "commitment": text,
            "preflight_block": height,
            "preflight_block_hash": chain.finalized_block_hash(height),
            "status": "SUBMISSION_PENDING" if args.execute else "DRY_RUN",
            **metadata,
        }
        _new(args.out, receipt)
        if args.execute:
            if wallet.hotkey.ss58_address != hotkey:
                raise ValueError("signing key differs from public preflight identity")
            receipt["extrinsic_id"] = chain.publish(wallet, text)
            receipt["status"] = "SUBMITTED_READBACK_PENDING"
            _update(args.out, receipt)
            finalized, observed = chain.finalized_publication(hotkey, text, receipt["extrinsic_id"])
            if pack is not None and observed.block > pack["context"]["close_block"]:
                receipt.update(
                    status="FINALIZED_LATE_INELIGIBLE",
                    commitment_block=observed.block,
                    finalized_block=finalized,
                    finalized_block_hash=chain.finalized_block_hash(finalized),
                )
                _update(args.out, receipt)
                print(json.dumps({k: v for k, v in receipt.items() if k != "manifest"}))
                return 2
            receipt.update(
                status="FINALIZED",
                commitment_block=observed.block,
                finalized_block=finalized,
                finalized_block_hash=chain.finalized_block_hash(finalized),
            )
            _update(args.out, receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "manifest"}))
        return 0
    finally:
        chain.close()
