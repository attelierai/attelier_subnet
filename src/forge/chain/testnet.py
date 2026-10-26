"""One real round on a Bittensor network with simulated models.

Everything touching the chain is real: finalized blocks and hashes, on-chain
submission and report commitments, the validator committee from the metagraph,
sr25519 signatures and weights. What is simulated is the model evaluation:
fingerprints resolve to declared ``ModelProfile``s instead of downloaded weights
scored on a GPU. Swapping ``sim.compare`` for the real runtime/evaluator is the
GPU phase.

Single operator: the round coordinator also publishes the plan to a FileBoard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forge.chain import BittensorChain, Sr25519Signer, Sr25519Verifier, encode_report_commit
from forge.evaluation.benchmark import generate_prompts
from forge.evaluation.stats import VerdictStatus
from forge.protocol.consensus_types import (
    CommitRecord,
    DecisionStatus,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    LocalVerdict,
    Reveal,
    RoundPlan,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.koth import CommitmentPosition, KothRound, Submission, SubmissionState
from forge.protocol.preconsensus import (
    CertificateVerifier,
    apply_certificate,
    build_certificate,
    commit_report,
    decide_round,
)
from forge.protocol.rewards import RewardRules, derive_hotkey_weights, map_hotkey_weights_to_uids
from forge.protocol.store import Store
from forge.sim import ModelProfile, compare, screen_and_duel
from forge.validator.board import FileBoard

LOCAL_VERDICT = {
    VerdictStatus.PROMOTE_ELIGIBLE: LocalVerdict.PROMOTE,
    VerdictStatus.REJECT: LocalVerdict.KEEP,
    VerdictStatus.INCOMPLETE: LocalVerdict.INCOMPLETE,
}


@dataclass
class TestnetRound:
    chain: BittensorChain
    validator_wallet: Any
    board: FileBoard
    store: Store
    contract: ChallengeContract
    baseline: ModelProfile
    models: dict[str, ModelProfile]
    challenge_id: str = "wan13b-16step"
    commit_window: int = 6
    reveal_window: int = 6
    poll_seconds: float = 4.0
    unallocated_uid: int | None = None
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.models = {self.baseline.fingerprint(): self.baseline, **self.models}
        self.signer = Sr25519Signer(self.validator_wallet)
        self.verifier = Sr25519Verifier()

    # --- helpers -------------------------------------------------------------

    def _say(self, message: str) -> None:
        self.log.append(message)
        print(message, flush=True)

    def wait_finalized(self, height: int) -> str:
        while self.chain.finalized_height() < height:
            time.sleep(self.poll_seconds)
        return self.chain.finalized_block_hash(height)

    def state(self) -> FinalizedCompetitionState:
        genesis = FinalizedCompetitionState.genesis(self.challenge_id, self.baseline.fingerprint())
        self.store.initialize_state(
            state_hash=genesis.state_hash(),
            challenge_id=self.challenge_id,
            state_version=0,
            previous_state_hash=genesis.previous_state_hash,
            payload=genesis.to_dict(),
        )
        payload = self.store.latest_state_payload(self.challenge_id)
        assert payload is not None
        return FinalizedCompetitionState.from_dict(payload)

    # --- the round -------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        state = self.state()
        king = self.models[state.current_king_fingerprint]
        close = self.chain.finalized_height()
        round_id = f"r{close}"
        self._say(f"round {round_id}: submissions closed at finalized block {close}")

        # 1. Admission from on-chain submission commitments at the close block.
        hotkeys = list(self.chain.uid_snapshot(close).values())
        round_ = KothRound(self.challenge_id, round_id, king.fingerprint(), quota=16)
        for item in self.chain.commitments(hotkeys, close):
            if item.kind != "sub":
                continue
            fingerprint = item.fields[0]
            if fingerprint not in self.models:
                self._say(f"  skip {item.hotkey[:8]}: artifact not fetchable")
                continue
            # SDK reads expose the commitment block only; ties within a block
            # fall back to the artifact id (documented limitation).
            position = CommitmentPosition(item.block, 0)
            try:
                round_.submit(Submission(item.hotkey, fingerprint, fingerprint, position))
            except ValueError as exc:
                self._say(f"  reject {item.hotkey[:8]}: {exc}")
                continue
            round_.transition(fingerprint, SubmissionState.ADMITTED)
            self._say(f"  admitted {item.hotkey[:8]} ({self.models[fingerprint].name})")

        # 2. Screen and duel on panels seeded by blocks finalized after the close.
        stage_hashes = {
            "screen": self.wait_finalized(close + 1),
            "duel": self.wait_finalized(close + 2),
        }
        stages, duel_bounds = screen_and_duel(
            self.contract, round_, king, self.models, stage_hashes
        )
        summary: dict[str, Any] = {"round_id": round_id, "close_block": close, "stages": stages}
        finalist = round_.select_finalist(duel_bounds)
        if finalist is None:
            self._say("  no finalist: king stays")
            return self._finish(summary)
        round_.lock_finalist(finalist.artifact_id)

        # 3. Lock the finalist publicly, then draw confirmation entropy after it.
        lock_head = self.chain.head()
        confirmation_block = lock_head + 2
        self.board.put(
            "plan",
            round_id,
            "finalist",
            {
                "finalist": finalist.artifact_id,
                "hotkey": finalist.hotkey,
                "locked_at_head": lock_head,
                "confirmation_block": confirmation_block,
            },
        )
        confirmation_hash = self.wait_finalized(confirmation_block)
        round_.draw_confirmation(confirmation_hash)
        self._say(f"  finalist {self.models[finalist.artifact_id].name} locked")

        committee = self.chain.committee_snapshot(close)
        if self.signer.validator_hotkey not in committee:
            raise RuntimeError("this validator has no permit/stake in the committee snapshot")
        commit_deadline = self.chain.head() + self.commit_window
        plan = RoundPlan(
            network=self.chain.network,
            netuid=self.chain.netuid,
            round_id=round_id,
            parent_state_hash=state.state_hash(),
            challenge_id=self.challenge_id,
            king_fingerprint=king.fingerprint(),
            challenger_fingerprint=finalist.effective_fingerprint,
            challenger_hotkey=finalist.hotkey,
            contract_hash=self.contract.contract_hash(),
            prompt_seed_inputs={"stage": "confirmation", "block_hash": confirmation_hash},
            committee=committee,
            snapshot_block=close,
            commit_deadline_block=commit_deadline,
            reveal_deadline_block=commit_deadline + self.reveal_window,
            activation_block=commit_deadline + self.reveal_window + 1,
        )
        self.board.put("plan", round_id, "plan", plan.to_dict())
        self.store.create_round(round_id, self.challenge_id, plan.plan_hash(), plan.to_dict())
        summary["committee"] = committee

        # 4. Evaluate confirmation locally and commit the report on chain.
        prompts = generate_prompts(self.contract, round_id, "confirmation", confirmation_hash)
        verdict = compare(
            self.contract,
            king,
            self.models[finalist.artifact_id],
            prompts,
            evaluator_id=self.signer.validator_hotkey,
        )
        report = LocalEvaluationReport(
            network=plan.network,
            netuid=plan.netuid,
            round_id=round_id,
            validator_hotkey=self.signer.validator_hotkey,
            plan_hash=plan.plan_hash(),
            parent_state_hash=plan.parent_state_hash,
            contract_hash=plan.contract_hash,
            king_fingerprint=plan.king_fingerprint,
            challenger_fingerprint=plan.challenger_fingerprint,
            verdict=LOCAL_VERDICT[verdict.status],
            metrics={
                "lower_bound": None if verdict.primary is None else verdict.primary.lower_bound,
                "reasons": list(verdict.reasons),
            },
        )
        self.store.record_report(
            round_id, report.validator_hotkey, report.report_hash(), report.to_dict()
        )
        commit, reveal = commit_report(plan, report, self.chain.head())
        extrinsic = self.chain.publish(
            self.validator_wallet, encode_report_commit(round_id, commit.commitment)
        )
        self._say(f"  local verdict {report.verdict.value}; report committed ({extrinsic})")

        # 5. After the commit deadline: read commits as of that block, then reveal.
        self.wait_finalized(commit_deadline)
        commits = []
        for item in self.chain.commitments(list(committee), commit_deadline):
            if item.kind == "rep" and item.fields[0] == round_id:
                commits.append(CommitRecord(item.hotkey, item.fields[1], item.block))
        while self.chain.head() <= commit_deadline:
            time.sleep(self.poll_seconds)
        reveal_block = self.chain.head()
        if reveal_block > plan.reveal_deadline_block:
            raise RuntimeError("missed the reveal window")
        self.board.put(
            "reveal",
            round_id,
            self.signer.validator_hotkey,
            {"report": report.to_dict(), "salt": reveal.salt, "reveal_block": reveal_block},
        )

        # 6. After the reveal deadline: decide from everything published.
        self.wait_finalized(plan.reveal_deadline_block)
        reveals = [
            Reveal(LocalEvaluationReport.from_dict(doc["report"]), doc["salt"], doc["reveal_block"])
            for doc in self.board.list("reveal", round_id).values()
        ]
        decision = decide_round(plan, commits, reveals)
        summary["decision"] = decision.status.value
        summary["favorable_power"] = decision.favorable_power
        summary["total_power"] = decision.total_committee_power
        self._say(
            f"  decision {decision.status.value} "
            f"({decision.favorable_power}/{decision.total_committee_power} power)"
        )
        if decision.status == DecisionStatus.PENDING:
            return self._finish(summary)

        # 7. Certify, apply, and only then derive weights.
        self.wait_finalized(plan.activation_block)
        local_report = report
        expected_verdict = (
            LocalVerdict.PROMOTE if decision.status == DecisionStatus.PROMOTE else LocalVerdict.KEEP
        )
        signer_power = committee[self.signer.validator_hotkey]
        if local_report.verdict != expected_verdict or 3 * signer_power <= 2 * sum(
            committee.values()
        ):
            summary["certificate_status"] = "PENDING_EXTERNAL_SIGNATURES"
            return self._finish(summary)
        certificate = build_certificate(
            plan,
            decision,
            self.chain.chain_time(plan.activation_block),
            [self.signer],
            commits=commits,
            reveals=reveals,
        )
        self.board.put("certificate", round_id, "certificate", certificate.to_dict())
        verifier = CertificateVerifier(self.verifier, self.store)
        new_state = apply_certificate(state, plan, certificate, verifier, store=self.store)
        summary["transition"] = certificate.transition.value
        summary["king"] = self.models[new_state.current_king_fingerprint].name
        return self._finish(summary)

    def _finish(self, summary: dict[str, Any]) -> dict[str, Any]:
        state = self.state()
        height = self.chain.finalized_height()
        uids = self.chain.uid_snapshot(height)
        hotkey_plan = derive_hotkey_weights(
            state.crown_history, RewardRules(), set(uids.values()), self.chain.chain_time(height)
        )
        uid_plan = map_hotkey_weights_to_uids(hotkey_plan, uids)
        weights = dict(uid_plan.weights)
        if uid_plan.unallocated_share > 0 and self.unallocated_uid is not None:
            weights[self.unallocated_uid] = (
                weights.get(self.unallocated_uid, 0.0) + uid_plan.unallocated_share
            )
        summary["weights"] = {str(uid): share for uid, share in sorted(weights.items())}
        summary["unallocated_share"] = uid_plan.unallocated_share
        # This runner evaluates ModelProfile simulations, never public weights.
        # Keep the derived plan as diagnostic output, but make an emission call
        # impossible from this command even when a caller holds a live wallet.
        summary["weights_submitted"] = False
        summary["weights_extrinsic"] = None
        summary["weight_plan_status"] = "SIMULATED_DIAGNOSTIC_NOT_SUBMITTABLE"
        self._say(f"  simulated weight plan {summary['weights']} (not submitted)")
        return summary
