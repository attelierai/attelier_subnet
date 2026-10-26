"""Verify committee decisions and apply certified state transitions."""

from __future__ import annotations

import secrets
from typing import Any

from forge.protocol.consensus_types import (
    CERTIFICATE_SCHEMA,
    CertificateError,
    CommitRecord,
    ConsensusPolicy,
    DecisionStatus,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    LocalVerdict,
    Reveal,
    RoundCertificate,
    RoundDecision,
    RoundPlan,
    Transition,
    _commit_dict,
    _reveal_dict,
)
from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.ports import Signer, Verifier
from forge.protocol.rewards import (
    Crown,
    HotkeyRewardPlan,
    RewardRules,
    derive_hotkey_weights,
)
from forge.protocol.store import Store


def report_commitment(plan: RoundPlan, report: LocalEvaluationReport, salt: str) -> str:
    if not salt:
        raise ValueError("secret salt is required")
    return hash_canonical(
        "forge/preconsensus/commit/v1",
        plan.network,
        plan.netuid,
        plan.round_id,
        report.validator_hotkey,
        plan.parent_state_hash,
        plan.contract_hash,
        report.report_hash(),
        salt,
    )


def commit_report(
    plan: RoundPlan,
    report: LocalEvaluationReport,
    commit_block: int,
    *,
    salt: str | None = None,
) -> tuple[CommitRecord, Reveal]:
    if not report.consistent_with(plan):
        raise ValueError("report is inconsistent with the round plan")
    if commit_block > plan.commit_deadline_block:
        raise ValueError("commit is late")
    secret_salt = secrets.token_hex(32) if salt is None else salt
    record = CommitRecord(
        report.validator_hotkey,
        report_commitment(plan, report, secret_salt),
        commit_block,
    )
    return record, Reveal(report, secret_salt, plan.commit_deadline_block + 1)


def reveal_is_valid(plan: RoundPlan, commit: CommitRecord, reveal: Reveal) -> bool:
    return (
        commit.validator_hotkey == reveal.report.validator_hotkey
        and commit.validator_hotkey in plan.committee
        and commit.commit_block <= plan.commit_deadline_block
        and plan.commit_deadline_block < reveal.reveal_block <= plan.reveal_deadline_block
        and reveal.report.consistent_with(plan)
        and secrets.compare_digest(
            commit.commitment, report_commitment(plan, reveal.report, reveal.salt)
        )
    )


def _valid_reports(
    plan: RoundPlan,
    commits: list[CommitRecord] | tuple[CommitRecord, ...],
    reveals: list[Reveal] | tuple[Reveal, ...],
) -> dict[str, LocalEvaluationReport]:
    """Return the exact valid reveal set used by both decision and certificate checks."""
    commit_map: dict[str, CommitRecord] = {}
    invalid_committers: set[str] = set()
    for commit in commits:
        if commit.validator_hotkey in commit_map and commit_map[commit.validator_hotkey] != commit:
            invalid_committers.add(commit.validator_hotkey)
        commit_map[commit.validator_hotkey] = commit
    reveal_map: dict[str, Reveal] = {}
    invalid_revealers: set[str] = set()
    for reveal in reveals:
        hotkey = reveal.report.validator_hotkey
        if hotkey in reveal_map and reveal_map[hotkey] != reveal:
            invalid_revealers.add(hotkey)
        reveal_map[hotkey] = reveal
    valid: dict[str, LocalEvaluationReport] = {}
    for hotkey, reveal in reveal_map.items():
        commit = commit_map.get(hotkey)
        if (
            commit is not None
            and hotkey not in invalid_committers
            and hotkey not in invalid_revealers
            and reveal_is_valid(plan, commit, reveal)
        ):
            valid[hotkey] = reveal.report
    return valid


def decide_round(
    plan: RoundPlan,
    commits: list[CommitRecord] | tuple[CommitRecord, ...],
    reveals: list[Reveal] | tuple[Reveal, ...],
    *,
    reveal_window_closed: bool = False,
) -> RoundDecision:
    """Apply the quorum rule to the exact valid reveal set this node holds."""

    valid = _valid_reports(plan, commits, reveals)
    favorable = sum(
        plan.committee[hotkey]
        for hotkey, report in valid.items()
        if report.verdict == LocalVerdict.PROMOTE
    )
    conclusive = sum(
        plan.committee[hotkey]
        for hotkey, report in valid.items()
        if report.verdict != LocalVerdict.INCOMPLETE
    )
    total = plan.total_committee_power
    against = conclusive - favorable
    best_case_favorable = favorable if reveal_window_closed else total - against
    if plan.quorum_satisfied(favorable):
        status = DecisionStatus.PROMOTE
    elif not plan.quorum_satisfied(best_case_favorable):
        status = DecisionStatus.KEEP
    else:
        status = DecisionStatus.PENDING
    return RoundDecision(status, favorable, conclusive, total, tuple(sorted(valid)))


def _evidence_digest(commits: tuple[CommitRecord, ...], reveals: tuple[Reveal, ...]) -> str:
    """Bind the exact commit/reveal set a certificate claims to have decided."""

    return hash_canonical(
        "forge/preconsensus/certificate-evidence/v2",
        sorted((_commit_dict(commit) for commit in commits), key=canonical_json_bytes),
        sorted((_reveal_dict(reveal) for reveal in reveals), key=canonical_json_bytes),
    )


def build_certificate(
    plan: RoundPlan,
    decision: RoundDecision,
    activation_chain_time: int,
    signers: list[Signer] | tuple[Signer, ...],
    *,
    commits: list[CommitRecord] | tuple[CommitRecord, ...],
    reveals: list[Reveal] | tuple[Reveal, ...],
) -> RoundCertificate:
    included_commits, included_reveals = tuple(commits), tuple(reveals)
    # A certificate never asserts that a reveal window was independently closed.
    # Thus KEEP is admissible only when it is already forced by the supplied
    # valid reveals, not merely by an unverified deadline claim.
    recomputed = decide_round(plan, included_commits, included_reveals)
    if decision != recomputed:
        raise CertificateError("supplied decision does not match commit/reveal evidence")
    if recomputed.status == DecisionStatus.PENDING:
        raise CertificateError("a pending decision cannot be certified")
    transition = Transition(recomputed.status.value)
    evidence_digest = _evidence_digest(included_commits, included_reveals)
    unsigned = RoundCertificate(
        plan.network,
        plan.netuid,
        plan.round_id,
        plan.challenge_id,
        plan.plan_hash(),
        plan.parent_state_hash,
        plan.king_fingerprint,
        plan.challenger_fingerprint,
        plan.challenger_hotkey,
        transition,
        plan.activation_block,
        activation_chain_time,
        evidence_digest,
        included_commits,
        included_reveals,
        {},
    )
    valid_reports = _valid_reports(plan, included_commits, included_reveals)
    signatures: dict[str, bytes] = {}
    for signer in signers:
        if signer.validator_hotkey not in plan.committee:
            raise CertificateError("certificate signer is outside the committee")
        if signer.validator_hotkey in signatures:
            raise CertificateError("duplicate certificate signer")
        report = valid_reports.get(signer.validator_hotkey)
        expected_verdict = (
            LocalVerdict.PROMOTE if transition == Transition.PROMOTE else LocalVerdict.KEEP
        )
        if report is None or report.verdict != expected_verdict:
            raise CertificateError("certificate signer lacks a matching valid local report")
        signatures[signer.validator_hotkey] = signer.sign(unsigned.signing_payload())
    return RoundCertificate(**{**unsigned.__dict__, "signatures": signatures})


class CertificateVerifier:
    """Checks certificates; with a store, double-signing is detected across restarts."""

    def __init__(self, signature_verifier: Verifier, store: Store | None = None):
        self.signature_verifier = signature_verifier
        self.store = store
        self._signed_transition: dict[tuple[str, str], str] = {}

    def verify(
        self,
        certificate: RoundCertificate,
        plan: RoundPlan,
        state: FinalizedCompetitionState,
    ) -> None:
        expected = (
            certificate.schema == CERTIFICATE_SCHEMA
            and certificate.network == plan.network
            and certificate.netuid == plan.netuid
            and certificate.round_id == plan.round_id
            and certificate.challenge_id == plan.challenge_id == state.challenge_id
            and certificate.plan_hash == plan.plan_hash()
            and certificate.parent_state_hash == plan.parent_state_hash == state.state_hash()
            and certificate.previous_king_fingerprint
            == plan.king_fingerprint
            == state.current_king_fingerprint
            and certificate.challenger_fingerprint == plan.challenger_fingerprint
            and certificate.challenger_hotkey == plan.challenger_hotkey
            and certificate.activation_block == plan.activation_block
        )
        if not expected:
            raise CertificateError("certificate does not match plan and current parent state")
        if certificate.evidence_digest != _evidence_digest(
            certificate.commits, certificate.reveals
        ):
            raise CertificateError(
                "certificate commit/reveal digest differs from included evidence"
            )
        decision = decide_round(plan, certificate.commits, certificate.reveals)
        if decision.status == DecisionStatus.PENDING or certificate.transition != Transition(
            decision.status.value
        ):
            raise CertificateError("certificate transition does not match commit/reveal decision")
        valid_reports = _valid_reports(plan, certificate.commits, certificate.reveals)
        expected_verdict = (
            LocalVerdict.PROMOTE
            if certificate.transition == Transition.PROMOTE
            else LocalVerdict.KEEP
        )
        signer_power = 0
        transition_hash = certificate.transition_hash()
        pending_marks: list[tuple[tuple[str, str], str]] = []
        for hotkey, signature in certificate.signatures.items():
            if hotkey not in plan.committee:
                raise CertificateError("certificate includes a non-committee signer")
            if (
                valid_reports.get(hotkey) is None
                or valid_reports[hotkey].verdict != expected_verdict
            ):
                raise CertificateError("certificate signer lacks a matching valid local report")
            if not self.signature_verifier.verify(hotkey, certificate.signing_payload(), signature):
                raise CertificateError(f"invalid certificate signature for {hotkey}")
            key = (plan.round_id, hotkey)
            previous = self._signed_transition.get(key)
            if previous is None and self.store is not None:
                previous = self.store.attested_transition(plan.round_id, hotkey)
            if previous is not None and previous != transition_hash:
                raise CertificateError(f"validator {hotkey} double-signed conflicting transitions")
            pending_marks.append((key, transition_hash))
            signer_power += plan.committee[hotkey]
        if not plan.quorum_satisfied(signer_power):
            if plan.consensus_policy == ConsensusPolicy.HISTORICAL_SUPERMAJORITY:
                raise CertificateError("certificate signature power does not exceed two thirds")
            raise CertificateError("certificate signature power does not satisfy the round quorum")
        for key, value in pending_marks:
            self._signed_transition[key] = value


def apply_certificate(
    state: FinalizedCompetitionState,
    plan: RoundPlan,
    certificate: RoundCertificate,
    verifier: CertificateVerifier,
    *,
    store: Store | None = None,
) -> FinalizedCompetitionState:
    if state.last_transition_hash == certificate.transition_hash():
        return state
    verifier.verify(certificate, plan, state)
    crowns = list(state.crown_history)
    next_king = state.current_king_fingerprint
    crown_payload: dict[str, Any] | None = None
    if certificate.transition == Transition.PROMOTE:
        next_king = certificate.challenger_fingerprint
        crown = Crown(
            crown_id=certificate.transition_hash(),
            hotkey=certificate.challenger_hotkey,
            fingerprint=certificate.challenger_fingerprint,
            chain_time=certificate.activation_chain_time,
        )
        crowns.append(crown)
        crown_payload = crown.__dict__
    next_state = FinalizedCompetitionState(
        challenge_id=state.challenge_id,
        state_version=state.state_version + 1,
        previous_state_hash=state.state_hash(),
        current_king_fingerprint=next_king,
        crown_history=tuple(crowns),
        activation_block=certificate.activation_block,
        last_round_id=certificate.round_id,
        last_transition_hash=certificate.transition_hash(),
    )
    if store is not None:
        store.apply_certificate(
            certificate_hash=certificate.transition_hash(),
            round_id=certificate.round_id,
            challenge_id=certificate.challenge_id,
            parent_state_hash=state.state_hash(),
            resulting_state_hash=next_state.state_hash(),
            state_version=next_state.state_version,
            certificate_payload=certificate.to_dict(),
            state_payload=next_state.to_dict(),
            crown_payload=crown_payload,
            attestations=dict.fromkeys(certificate.signatures, certificate.transition_hash()),
        )
    return next_state


def derive_weights_from_latest_certified_state(
    store: Store,
    challenge_id: str,
    rules: RewardRules,
    eligible_hotkeys: set[str] | frozenset[str],
    reference_chain_time: int,
) -> HotkeyRewardPlan:
    payload = store.latest_state_payload(challenge_id)
    if payload is None:
        raise ValueError("no certified state exists for this challenge")
    state = FinalizedCompetitionState.from_dict(payload)
    return derive_hotkey_weights(state.crown_history, rules, eligible_hotkeys, reference_chain_time)
