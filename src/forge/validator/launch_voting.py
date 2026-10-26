"""Offline signed voting over independently replayed Launch V1 receipts.

This adapter deliberately owns no key material and makes no network or chain
calls.  The caller injects native protocol signers and distributes the returned
transport envelopes over its authenticated channel.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from forge.protocol.consensus_types import (
    CommitRecord,
    LocalEvaluationReport,
    LocalVerdict,
    Reveal,
    RoundCertificate,
    RoundPlan,
)
from forge.protocol.ports import Signer, Verifier
from forge.protocol.preconsensus import build_certificate, commit_report, decide_round
from forge.protocol.store import Store, StoreConflictError
from forge.protocol.transport import RoundMailbox, sign_envelope
from forge.validator.launch_bridge import LaunchExecutionReplayer, launch_local_report


class LaunchVotingError(ValueError):
    """A vote cannot be tied to independently replayed local execution."""


@dataclass(frozen=True)
class _LocalVote:
    bundle_hash: str
    report: LocalEvaluationReport
    salt: str
    commitment: str
    commit_block: int
    commit_envelope: dict[str, Any]
    reveal_envelope: dict[str, Any] | None


class LaunchV1VotingWorkflow:
    """Durable commit/reveal orchestration without reimplementing consensus.

    SQLite retains the local commit/reveal material, including its salt, for
    restart-safe replay. It is protected local custody, not a short-lived
    secret store.
    Every reveal, including one received from a peer, is re-executed from the
    protected receipt store before it can contribute to a certificate.
    """

    def __init__(self, store: Store, verifier: Verifier, replayer: LaunchExecutionReplayer):
        self.store, self.verifier, self.replayer = store, verifier, replayer
        self.mailbox = RoundMailbox(store, verifier)
        self.store.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS launch_v1_local_votes (
                plan_hash TEXT NOT NULL,
                validator_hotkey TEXT NOT NULL,
                bundle_hash TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                report_json TEXT NOT NULL,
                salt TEXT NOT NULL,
                commitment TEXT NOT NULL,
                commit_block INTEGER NOT NULL,
                commit_envelope_json TEXT NOT NULL,
                reveal_envelope_json TEXT,
                PRIMARY KEY(plan_hash, validator_hotkey)
            )
            """
        )
        # A signer may be an external device. Reserve its one admissible
        # commitment before asking that device to sign, so a failed request
        # cannot be retried later with different execution evidence.
        self.store.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS launch_v1_commit_reservations (
                plan_hash TEXT NOT NULL,
                validator_hotkey TEXT NOT NULL,
                bundle_hash TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                report_json TEXT NOT NULL,
                salt TEXT NOT NULL,
                commitment TEXT NOT NULL,
                commit_block INTEGER NOT NULL,
                PRIMARY KEY(plan_hash, validator_hotkey)
            )
            """
        )

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _require_open(self, plan: RoundPlan) -> None:
        row = self.store.connection.execute(
            "SELECT plan_hash, payload_json, status FROM rounds WHERE round_id=?", (plan.round_id,)
        ).fetchone()
        if row is None or row["plan_hash"] != plan.plan_hash() or row["status"] != "PENDING":
            raise LaunchVotingError("round is not durably open for this exact plan")
        if json.loads(row["payload_json"]) != plan.to_dict():
            raise LaunchVotingError("durable round payload differs from plan")

    def _replay(
        self, plan: RoundPlan, validator_hotkey: str, bundle_hash: str
    ) -> LocalEvaluationReport:
        report = launch_local_report(plan, validator_hotkey, bundle_hash, self.replayer)
        if report.verdict == LocalVerdict.INCOMPLETE:
            raise LaunchVotingError("execution replay is incomplete; it cannot be signed")
        return report

    @staticmethod
    def _commit_payload(record: CommitRecord) -> dict[str, Any]:
        return {"commitment": record.commitment, "commit_block": record.commit_block}

    @staticmethod
    def _reveal_payload(reveal: Reveal) -> dict[str, Any]:
        return {
            "report": reveal.report.to_dict(),
            "salt": reveal.salt,
            "reveal_block": reveal.reveal_block,
        }

    def _load_local_vote(self, plan: RoundPlan, hotkey: str) -> _LocalVote | None:
        row = self.store.connection.execute(
            "SELECT * FROM launch_v1_local_votes WHERE plan_hash=? AND validator_hotkey=?",
            (plan.plan_hash(), hotkey),
        ).fetchone()
        if row is None:
            return None
        try:
            return _LocalVote(
                row["bundle_hash"],
                LocalEvaluationReport.from_dict(json.loads(row["report_json"])),
                row["salt"],
                row["commitment"],
                row["commit_block"],
                json.loads(row["commit_envelope_json"]),
                None
                if row["reveal_envelope_json"] is None
                else json.loads(row["reveal_envelope_json"]),
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise LaunchVotingError("stored local vote is malformed") from error

    def _reserve_local_commit(
        self,
        plan: RoundPlan,
        *,
        hotkey: str,
        bundle_hash: str,
        report: LocalEvaluationReport,
        salt: str,
        record: CommitRecord,
    ) -> None:
        """Durably bind a signer to one commit before invoking it."""

        expected = {
            "bundle_hash": bundle_hash,
            "report_hash": report.report_hash(),
            "report_json": self._json(report.to_dict()),
            "salt": salt,
            "commitment": record.commitment,
            "commit_block": record.commit_block,
        }
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT * FROM launch_v1_commit_reservations "
                "WHERE plan_hash=? AND validator_hotkey=?",
                (plan.plan_hash(), hotkey),
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO launch_v1_commit_reservations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (plan.plan_hash(), hotkey, *expected.values()),
                )
            elif any(row[key] != value for key, value in expected.items()):
                raise LaunchVotingError("local validator already reserved different execution")

    def _reserve_certificate_attestations(
        self, plan: RoundPlan, signers: tuple[Signer, ...], transition_hash: str
    ) -> None:
        """Persist non-equivocation reservations before external certificate signatures."""

        with self.store.transaction() as db:
            for signer in signers:
                row = db.execute(
                    "SELECT transition_hash FROM attestations "
                    "WHERE round_id=? AND validator_hotkey=?",
                    (plan.round_id, signer.validator_hotkey),
                ).fetchone()
                if row is not None and row["transition_hash"] != transition_hash:
                    raise StoreConflictError("validator already signed another transition")
            for signer in signers:
                db.execute(
                    "INSERT OR IGNORE INTO attestations VALUES (?, ?, ?)",
                    (plan.round_id, signer.validator_hotkey, transition_hash),
                )

    def commit_envelope(
        self,
        plan: RoundPlan,
        signer: Signer,
        *,
        bundle_hash: str,
        finalized_block: int,
        salt: str,
    ) -> dict[str, Any]:
        """Replay, persist, sign and locally receive one opaque commit."""

        self._require_open(plan)
        if signer.validator_hotkey not in plan.committee:
            raise LaunchVotingError("signer is outside the frozen committee")
        report = self._replay(plan, signer.validator_hotkey, bundle_hash)
        record, _ = commit_report(plan, report, finalized_block, salt=salt)
        existing = self._load_local_vote(plan, signer.validator_hotkey)
        if existing is not None:
            if (
                existing.bundle_hash != bundle_hash
                or existing.report != report
                or existing.salt != salt
                or existing.commitment != record.commitment
                or existing.commit_block != finalized_block
            ):
                raise LaunchVotingError("local validator already committed different execution")
            self.store.record_report(
                plan.round_id, signer.validator_hotkey, report.report_hash(), report.to_dict()
            )
            self.mailbox.receive(plan, existing.commit_envelope, finalized_block=finalized_block)
            return existing.commit_envelope
        self._reserve_local_commit(
            plan,
            hotkey=signer.validator_hotkey,
            bundle_hash=bundle_hash,
            report=report,
            salt=salt,
            record=record,
        )
        envelope = sign_envelope(plan, signer, "commit", self._commit_payload(record))
        try:
            with self.store.transaction() as db:
                persisted = db.execute(
                    "SELECT * FROM launch_v1_local_votes WHERE plan_hash=? AND validator_hotkey=?",
                    (plan.plan_hash(), signer.validator_hotkey),
                ).fetchone()
                if persisted is not None:
                    expected = {
                        "bundle_hash": bundle_hash,
                        "report_hash": report.report_hash(),
                        "report_json": self._json(report.to_dict()),
                        "salt": salt,
                        "commitment": record.commitment,
                        "commit_block": finalized_block,
                    }
                    if any(persisted[key] != value for key, value in expected.items()):
                        raise LaunchVotingError(
                            "local validator already committed different execution"
                        )
                    saved_envelope = json.loads(persisted["commit_envelope_json"])
                else:
                    saved_envelope = envelope
                    db.execute(
                        "INSERT INTO launch_v1_local_votes "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                        (
                            plan.plan_hash(),
                            signer.validator_hotkey,
                            bundle_hash,
                            report.report_hash(),
                            self._json(report.to_dict()),
                            salt,
                            record.commitment,
                            finalized_block,
                            self._json(envelope),
                        ),
                    )
                db.execute(
                    "DELETE FROM launch_v1_commit_reservations "
                    "WHERE plan_hash=? AND validator_hotkey=?",
                    (plan.plan_hash(), signer.validator_hotkey),
                )
        except Exception as error:
            raise LaunchVotingError("could not persist local commit") from error
        try:
            self.store.record_report(
                plan.round_id, signer.validator_hotkey, report.report_hash(), report.to_dict()
            )
            self.mailbox.receive(plan, saved_envelope, finalized_block=finalized_block)
        except Exception:
            # The persisted row preserves the sole signed commitment for a safe retry.
            raise
        return saved_envelope

    def reveal_envelope(
        self, plan: RoundPlan, signer: Signer, *, finalized_block: int
    ) -> dict[str, Any]:
        """Materialize the previously committed report only in the reveal window."""

        self._require_open(plan)
        vote = self._load_local_vote(plan, signer.validator_hotkey)
        if vote is None:
            raise LaunchVotingError("no durable local commit exists for signer")
        if vote.reveal_envelope is not None:
            if self._parse_reveal(plan, vote.reveal_envelope).reveal_block != finalized_block:
                raise LaunchVotingError("reveal block differs from finalized transport receipt")
            self.mailbox.receive(plan, vote.reveal_envelope, finalized_block=finalized_block)
            return vote.reveal_envelope
        if not plan.commit_deadline_block < finalized_block <= plan.reveal_deadline_block:
            raise LaunchVotingError("reveal is outside the frozen reveal window")
        replayed = self._replay(plan, signer.validator_hotkey, vote.bundle_hash)
        if replayed != vote.report:
            raise LaunchVotingError("execution replay changed after the commit")
        reveal = Reveal(vote.report, vote.salt, finalized_block)
        envelope = sign_envelope(plan, signer, "reveal", self._reveal_payload(reveal))
        with self.store.transaction() as db:
            updated = db.execute(
                """UPDATE launch_v1_local_votes SET reveal_envelope_json=?
                WHERE plan_hash=? AND validator_hotkey=? AND reveal_envelope_json IS NULL""",
                (self._json(envelope), plan.plan_hash(), signer.validator_hotkey),
            ).rowcount
            if updated != 1:
                raise LaunchVotingError("local reveal changed concurrently")
        self.mailbox.receive(plan, envelope, finalized_block=finalized_block)
        return envelope

    def _parse_commit(self, envelope: Mapping[str, Any]) -> CommitRecord:
        payload = envelope["payload"]
        if set(payload) != {"commitment", "commit_block"}:
            raise LaunchVotingError("commit envelope payload is invalid")
        try:
            record = CommitRecord(
                envelope["sender"], payload["commitment"], payload["commit_block"]
            )
        except (KeyError, TypeError) as error:
            raise LaunchVotingError("commit envelope payload is invalid") from error
        if not isinstance(record.commitment, str) or type(record.commit_block) is not int:
            raise LaunchVotingError("commit envelope types are invalid")
        return record

    def _parse_reveal(self, plan: RoundPlan, envelope: Mapping[str, Any]) -> Reveal:
        payload = envelope["payload"]
        if set(payload) != {"report", "salt", "reveal_block"}:
            raise LaunchVotingError("reveal envelope payload is invalid")
        try:
            report = LocalEvaluationReport.from_dict(payload["report"])
            reveal = Reveal(report, payload["salt"], payload["reveal_block"])
        except (KeyError, TypeError, ValueError) as error:
            raise LaunchVotingError("reveal envelope payload is invalid") from error
        if report.validator_hotkey != envelope["sender"] or not isinstance(reveal.salt, str):
            raise LaunchVotingError("reveal sender or salt is invalid")
        bundle_hash = report.metrics.get("launch_bundle_hash")
        if not isinstance(bundle_hash, str):
            raise LaunchVotingError("reveal lacks a trusted launch bundle")
        if self._replay(plan, report.validator_hotkey, bundle_hash) != report:
            raise LaunchVotingError("received reveal does not reproduce from local receipts")
        return reveal

    def receive(self, plan: RoundPlan, envelope: dict[str, Any], *, finalized_block: int) -> bool:
        """Verify semantic receipt provenance before durable transport acceptance."""

        self._require_open(plan)
        if not isinstance(envelope, dict) or envelope.get("kind") not in {"commit", "reveal"}:
            raise LaunchVotingError("only commit or reveal envelopes are accepted here")
        if envelope["kind"] == "commit":
            if self._parse_commit(envelope).commit_block != finalized_block:
                raise LaunchVotingError("commit block differs from finalized transport receipt")
        else:
            if self._parse_reveal(plan, envelope).reveal_block != finalized_block:
                raise LaunchVotingError("reveal block differs from finalized transport receipt")
        return self.mailbox.receive(plan, envelope, finalized_block=finalized_block)

    def certificate(
        self,
        plan: RoundPlan,
        *,
        activation_chain_time: int,
        signers: Iterable[Signer],
    ) -> RoundCertificate:
        """Revalidate the mailbox evidence then reuse the canonical certificate builder."""

        self._require_open(plan)
        commits: list[CommitRecord] = []
        reveals: list[Reveal] = []
        rows = self.store.connection.execute(
            """SELECT envelope_json, received_finalized_block FROM round_messages
            WHERE plan_hash=? ORDER BY sender, kind""",
            (plan.plan_hash(),),
        ).fetchall()
        for row in rows:
            try:
                envelope = json.loads(row["envelope_json"])
            except json.JSONDecodeError as error:
                raise LaunchVotingError("durable mailbox envelope is malformed") from error
            # Re-running receive validates signature, bindings and message hash after restart.
            self.mailbox.receive(plan, envelope, finalized_block=row["received_finalized_block"])
            if envelope["kind"] == "commit":
                commit = self._parse_commit(envelope)
                if commit.commit_block != row["received_finalized_block"]:
                    raise LaunchVotingError("stored commit block differs from transport receipt")
                commits.append(commit)
            elif envelope["kind"] == "reveal":
                reveal = self._parse_reveal(plan, envelope)
                if reveal.reveal_block != row["received_finalized_block"]:
                    raise LaunchVotingError("stored reveal block differs from transport receipt")
                reveals.append(reveal)
        decision = decide_round(plan, commits, reveals)
        signer_list = tuple(signers)
        try:
            # Assemble the exact unsigned transition first.  This also rejects
            # pending evidence without asking any external signer to sign.
            unsigned = build_certificate(
                plan,
                decision,
                activation_chain_time,
                (),
                commits=tuple(commits),
                reveals=tuple(reveals),
            )
            expected_verdict = (
                LocalVerdict.PROMOTE
                if unsigned.transition.value == "PROMOTE"
                else LocalVerdict.KEEP
            )
            valid_reports = {
                reveal.report.validator_hotkey: reveal.report
                for reveal in reveals
                if reveal.report.validator_hotkey in decision.valid_validators
            }
            if len({signer.validator_hotkey for signer in signer_list}) != len(signer_list):
                raise LaunchVotingError("certificate signer is duplicated")
            if any(
                signer.validator_hotkey not in plan.committee
                or valid_reports.get(signer.validator_hotkey) is None
                or valid_reports[signer.validator_hotkey].verdict != expected_verdict
                for signer in signer_list
            ):
                raise LaunchVotingError("certificate signer lacks a matching replayed report")
            if not plan.quorum_satisfied(
                sum(plan.committee[signer.validator_hotkey] for signer in signer_list)
            ):
                raise LaunchVotingError("certificate signer power lacks round quorum")
            # Persist the intent before an external signer is invoked. A signing
            # failure leaves this exact transition retryable, but blocks a later
            # conflicting transition across restarts.
            self._reserve_certificate_attestations(plan, signer_list, unsigned.transition_hash())
            certificate = build_certificate(
                plan,
                decision,
                activation_chain_time,
                signer_list,
                commits=tuple(commits),
                reveals=tuple(reveals),
            )
            return certificate
        except Exception as error:
            raise LaunchVotingError("mailbox evidence cannot produce a certificate") from error
