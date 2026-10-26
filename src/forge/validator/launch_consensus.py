"""Restartable single-operator testnet consensus over real worker receipts.

The caller owns the local wallets. No key material is transferred to the GPU.
Each call advances at most one phase; a supervisor can restart between calls.
"""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Signer, Sr25519Verifier, encode_report_commit
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundCertificate, RoundPlan
from forge.protocol.preconsensus import commit_report
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox, sign_envelope
from forge.validator.launch_bridge import (
    LaunchCertificateVerifier,
    LaunchExecutionReplayer,
    LaunchV1Bridge,
    LocalLaunchReceiptStore,
    launch_local_report,
)
from forge.validator.launch_service import atomic_json, write_immutable_json
from forge.validator.launch_voting import LaunchV1VotingWorkflow, LaunchVotingError

MINIMUM_COMMIT_WINDOW_BLOCKS = 10


class LaunchConsensusRunner:
    def __init__(self, root: Path, *, chain: Any, wallets: dict[str, Any]):
        self.root, self.chain = Path(root), chain
        self.plan = RoundPlan.from_dict(self.read("round-plan.json"))
        self.parent = FinalizedCompetitionState.from_dict(self.read("state.json"))
        if (
            self.plan.network != "test"
            or (chain.network, chain.netuid) != (self.plan.network, self.plan.netuid)
        ):
            raise ValueError("consensus runner supports only its exact testnet subnet")
        if not wallets or any(not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", k) for k in wallets):
            raise ValueError("validator labels are invalid")
        self.wallets = wallets
        self.signers = {key: Sr25519Signer(wallet) for key, wallet in wallets.items()}
        addresses = {s.validator_hotkey for s in self.signers.values()}
        if len(addresses) != len(wallets) or not addresses <= set(self.plan.committee):
            raise ValueError("validators must be distinct members of the frozen committee")
        self.receipts = LocalLaunchReceiptStore(self.root / "trusted-receipts")
        self.replayer = LaunchExecutionReplayer(self.receipts)
        self.verifier = Sr25519Verifier()

    def read(self, name: str) -> dict[str, Any]:
        result = json.loads((self.root / name).read_text())
        if not isinstance(result, dict):
            raise ValueError(f"invalid receipt: {name}")
        return result

    def save(self, name: str, value: dict[str, Any]) -> None:
        write_immutable_json(self.root / name, value)

    def store(self, label: str) -> Store:
        return Store(self.root / f"validator-{label}.sqlite")

    def status(self, status: str, **extra: Any) -> dict[str, Any]:
        result = {
            "schema": "forge/launch-consensus-status/v1",
            "status": status, "plan_hash": self.plan.plan_hash(),
            "shared_operator": True, **extra,
        }
        atomic_json(self.root / "consensus-status.json", result)
        return result

    def tick(self) -> dict[str, Any]:
        plan = self.plan
        height = self.chain.finalized_height()
        if (self.root / "certificate.json").exists():
            return self._apply_certificate()
        if height > plan.reveal_deadline_block:
            # A certificate may still be formed after the window if the durable
            # mailboxes already received quorum before it closed.
            try:
                return self._certify(height)
            except LaunchVotingError as error:
                # Invalid receipts or a failed external signer are operational
                # errors. Only missing consensus may expire automatically.
                cause = error.__cause__
                if str(cause) not in {
                    "a pending decision cannot be certified",
                    "certificate signer power lacks round quorum",
                }:
                    raise
                for label in self.signers:
                    with self.store(label) as store:
                        RoundMailbox(store, self.verifier).expire(
                            plan, finalized_block=height,
                            finalized_hash=self.chain.finalized_block_hash(height),
                        )
                return self.status("EXPIRED_NO_CERTIFICATE", finalized_block=height)
        if height <= plan.commit_deadline_block:
            return self._commit(height)
        return self._reveal(height)

    def _commit(self, height: int) -> dict[str, Any]:
        plan = self.plan
        for label, signer in self.signers.items():
            bundle_hash = self.read(f"bundle-{label}.json")["bundle_hash"]
            report = launch_local_report(plan, signer.validator_hotkey, bundle_hash, self.replayer)
            if report.verdict.value == "INCOMPLETE":
                return self.status("INCOMPLETE_NO_VOTE", validator=label)
            self.save(f"report-{label}.json", report.to_dict())
            draft_path = self.root / f"vote-draft-{label}.json"
            if not draft_path.exists():
                self.save(draft_path.name, {
                    "salt": secrets.token_hex(32), "report_hash": report.report_hash(),
                })
            draft = self.read(draft_path.name)
            if draft["report_hash"] != report.report_hash():
                raise ValueError("worker evidence changed after vote reservation")
            record, _ = commit_report(plan, report, height, salt=draft["salt"])
            text = encode_report_commit(plan.round_id, record.commitment)
            journal_path = self.root / f"chain-commit-{label}.json"
            if not journal_path.exists():
                if height >= plan.commit_deadline_block - MINIMUM_COMMIT_WINDOW_BLOCKS:
                    return self.status("INSUFFICIENT_COMMIT_WINDOW")
                self.save(journal_path.name, {
                    "status": "SUBMITTING", "hotkey": signer.validator_hotkey,
                    "commitment": text, "prepared_block": height, "plan_hash": plan.plan_hash(),
                })
                # This is the sole external write. A crash/timeout leaves the
                # intent behind; the next tick reconciles without retransmitting.
                extrinsic = self.chain.publish(self.wallets[label], text)
                journal = self.read(journal_path.name)
                atomic_json(journal_path, {
                    **journal, "status": "SUBMITTED_READBACK_PENDING",
                    "extrinsic_id": extrinsic,
                })
            journal = self.read(journal_path.name)
            if (
                journal.get("commitment") != text
                or journal.get("hotkey") != signer.validator_hotkey
                or journal.get("plan_hash") != plan.plan_hash()
            ):
                raise ValueError("on-chain publication intent changed")
            current = self.chain.finalized_height()
            observed = self.chain.commitment(signer.validator_hotkey, current)
            if (
                observed is None
                or observed.kind != "rep"
                or observed.fields != (plan.round_id, record.commitment)
                or not journal["prepared_block"] <= observed.block <= plan.commit_deadline_block
            ):
                return self.status("COMMIT_RECONCILIATION_PENDING", validator=label)
            if journal["status"] != "FINALIZED":
                atomic_json(journal_path, {
                    **journal, "status": "FINALIZED",
                    "commitment_block": observed.block, "finalized_block": current,
                    "finalized_block_hash": self.chain.finalized_block_hash(current),
                })
            with self.store(label) as store:
                envelope = LaunchV1VotingWorkflow(
                    store, self.verifier, self.replayer
                ).commit_envelope(
                    plan, signer, bundle_hash=bundle_hash,
                    finalized_block=observed.block, salt=draft["salt"],
                )
                self.save(f"commit-envelope-{label}.json", envelope)
        current = self.chain.finalized_height()
        self._receive("commit", current)
        if current > plan.commit_deadline_block:
            # Replaying already delivered commitments may cross the boundary.
            # Advance from fresh finality instead of failing a duplicate delivery.
            return self.tick()
        return self.status("COMMITTED_WAITING_REVEAL", deadline=plan.commit_deadline_block)

    def _receive(self, kind: str, height: int) -> None:
        deadline = (
            self.plan.commit_deadline_block if kind == "commit"
            else self.plan.reveal_deadline_block
        )
        for recipient in self.signers:
            with self.store(recipient) as store:
                workflow = LaunchV1VotingWorkflow(store, self.verifier, self.replayer)
                for sender in self.signers:
                    envelope = self.read(f"{kind}-envelope-{sender}.json")
                    event_block = envelope["payload"][f"{kind}_block"]
                    if height > deadline:
                        received = store.connection.execute(
                            "SELECT received_finalized_block FROM round_messages "
                            "WHERE plan_hash=? AND sender=? AND kind=?",
                            (self.plan.plan_hash(), envelope["sender"], kind),
                        ).fetchone()
                        if (
                            received is None
                            or received["received_finalized_block"] != event_block
                            or received["received_finalized_block"] > deadline
                        ):
                            raise ValueError("peer message delivery missed the frozen deadline")
                        # The workflow still verifies the signature, provenance
                        # and stored message hash; no late message is inserted.
                    workflow.receive(
                        self.plan, envelope, finalized_block=event_block,
                    )

    def _reveal(self, height: int) -> dict[str, Any]:
        plan = self.plan
        for label, signer in self.signers.items():
            commit = self.read(f"commit-envelope-{label}.json")
            observed = self.chain.commitment(signer.validator_hotkey, plan.commit_deadline_block)
            if (
                observed is None or observed.kind != "rep"
                or observed.fields != (plan.round_id, commit["payload"]["commitment"])
                or observed.block != commit["payload"]["commit_block"]
            ):
                raise ValueError("finalized commitment cut differs from signed vote")
            path = self.root / f"reveal-envelope-{label}.json"
            reveal_block = (
                self.read(path.name)["payload"]["reveal_block"] if path.exists() else height
            )
            with self.store(label) as store:
                envelope = LaunchV1VotingWorkflow(
                    store, self.verifier, self.replayer
                ).reveal_envelope(plan, signer, finalized_block=reveal_block)
                self.save(path.name, envelope)
        self._receive("reveal", height)
        return self.status("REVEALED_WAITING_ACTIVATION", activation_block=plan.activation_block)

    def _certify(self, height: int) -> dict[str, Any]:
        if height < self.plan.activation_block:
            return self.status("WAITING_ACTIVATION")
        with self.store(next(iter(self.signers))) as store:
            certificate = LaunchV1VotingWorkflow(
                store, self.verifier, self.replayer
            ).certificate(
                self.plan, activation_chain_time=self.chain.chain_time(self.plan.activation_block),
                signers=tuple(self.signers.values()),
            )
            self.save("certificate.json", certificate.to_dict())
        return self._apply_certificate()

    def _apply_certificate(self) -> dict[str, Any]:
        certificate = RoundCertificate.from_dict(self.read("certificate.json"))
        states = []
        for label in self.signers:
            with self.store(label) as store:
                verifier = LaunchCertificateVerifier(
                    self.verifier, replayer=self.replayer, store=store,
                )
                verifier.verify(certificate, self.plan, self.parent)
                # Preserve the same signed certificate envelope across restarts.
                path = self.root / "certificate-envelope.json"
                if not path.exists():
                    self.save(path.name, sign_envelope(
                        self.plan, next(iter(self.signers.values())),
                        "certificate", certificate.to_dict(),
                    ))
                RoundMailbox(store, self.verifier).receive(
                    self.plan, self.read(path.name), finalized_block=self.chain.finalized_height(),
                )
                state = LaunchV1Bridge(store, self.verifier, self.receipts).apply(
                    self.plan, self.parent, certificate, verifier,
                )
                self.save(f"certified-state-{label}.json", state.to_dict())
                states.append(state.state_hash())
        if len(set(states)) != 1:
            raise ValueError("validator states diverged")
        return self.status(
            "CERTIFIED_NOT_YET_WEIGHTS", transition=certificate.transition.value,
            certificate_hash=certificate.transition_hash(), state_hash=states[0],
        )
