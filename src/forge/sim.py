"""CPU simulation fakes. Nothing here measures a real model or signs for real.

Fake signers use HMAC keys and the fake evaluator draws axis scores from a
declared model profile. They exist to exercise the protocol end to end.
"""

from __future__ import annotations

import hashlib
import hmac
import random
from dataclasses import dataclass, field

from forge.evaluation.benchmark import PromptCase, generate_prompts
from forge.evaluation.scoring import Observation, ObservationStatus
from forge.evaluation.stats import Verdict, VerdictStatus, paired_comparison
from forge.protocol.consensus_types import (
    CommitRecord,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    LocalVerdict,
    Reveal,
    RoundCertificate,
    RoundDecision,
    RoundPlan,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.protocol.koth import CommitmentPosition, KothRound, Submission, SubmissionState
from forge.protocol.preconsensus import (
    CertificateVerifier,
    apply_certificate,
    build_certificate,
    commit_report,
    decide_round,
)
from forge.protocol.store import Store


class HmacSigner:
    def __init__(self, validator_hotkey: str, key: bytes):
        self._hotkey = validator_hotkey
        self._key = key

    @property
    def validator_hotkey(self) -> str:
        return self._hotkey

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._key, message, hashlib.sha256).digest()


class HmacVerifier:
    def __init__(self, keys: dict[str, bytes]):
        self._keys = dict(keys)

    def verify(self, validator_hotkey: str, message: bytes, signature: bytes) -> bool:
        key = self._keys.get(validator_hotkey)
        if key is None:
            return False
        expected = hmac.new(key, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)


@dataclass(frozen=True)
class ModelProfile:
    """Declared behavior of a simulated model: mean per axis, noise, failures."""

    name: str
    axis_means: dict[str, float]
    noise: float = 0.05
    failure_rate: float = 0.0

    def fingerprint(self) -> str:
        return hash_canonical("forge/sim/model/v1", self.name, dict(self.axis_means))


def profile(
    name: str,
    contract: ChallengeContract,
    mean: float,
    *,
    overrides: dict[str, float] | None = None,
    noise: float = 0.05,
    failure_rate: float = 0.0,
) -> ModelProfile:
    means = {axis: mean for axis in contract.axes}
    means.update(overrides or {})
    return ModelProfile(name, means, noise, failure_rate)


def simulate_panel(
    model: ModelProfile,
    prompts: tuple[PromptCase, ...],
    *,
    evaluator_id: str = "shared",
    evaluator_noise: float = 0.0,
) -> list[Observation]:
    """Deterministic observations for every (prompt, seed) slot of a panel.

    The model noise depends only on the model and the slot, so every validator
    sees the same generated "video". ``evaluator_noise`` adds a per-validator
    measurement error, which is how validators come to disagree.
    """

    observations: list[Observation] = []
    for prompt in prompts:
        for seed in prompt.video_seeds:
            video_rng = random.Random(f"{model.fingerprint()}|{prompt.prompt_id}|{seed}")
            if video_rng.random() < model.failure_rate:
                observations.append(
                    Observation(prompt.prompt_id, seed, None, ObservationStatus.MODEL_FAILURE)
                )
                continue
            judge_rng = random.Random(
                f"{evaluator_id}|{model.fingerprint()}|{prompt.prompt_id}|{seed}"
            )
            axes = {}
            for axis, mean in model.axis_means.items():
                value = mean + video_rng.gauss(0, model.noise)
                value += judge_rng.gauss(0, evaluator_noise) if evaluator_noise else 0.0
                axes[axis] = min(1.0, max(0.0, value))
            observations.append(Observation(prompt.prompt_id, seed, axes))
    return observations


def compare(
    contract: ChallengeContract,
    king: ModelProfile,
    challenger: ModelProfile,
    prompts: tuple[PromptCase, ...],
    *,
    evaluator_id: str = "shared",
    evaluator_noise: float = 0.0,
) -> Verdict:
    return paired_comparison(
        simulate_panel(king, prompts, evaluator_id=evaluator_id, evaluator_noise=evaluator_noise),
        simulate_panel(
            challenger, prompts, evaluator_id=evaluator_id, evaluator_noise=evaluator_noise
        ),
        contract,
        expected_slots=frozenset(
            (prompt.prompt_id, seed) for prompt in prompts for seed in prompt.video_seeds
        ),
        prompt_properties={prompt.prompt_id: prompt.properties for prompt in prompts},
    )


def screen_and_duel(
    contract: ChallengeContract,
    round_: KothRound,
    king: ModelProfile,
    models: dict[str, ModelProfile],
    stage_hashes: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict[str, float]]:
    """Run screen then duel on shared panels; return stage verdicts and duel bounds.

    Screen only filters clear losers (mean gain <= 0); the duel applies the full
    promotion rule. Every candidate in the cohort gets the same panels.
    """

    verdicts: dict[str, dict[str, str]] = {}
    duel_bounds: dict[str, float] = {}
    for stage, expected, advanced in (
        ("screen", SubmissionState.ADMITTED, SubmissionState.SCREENED),
        ("duel", SubmissionState.SCREENED, SubmissionState.DUELED),
    ):
        prompts = generate_prompts(contract, round_.round_id, stage, stage_hashes[stage])
        verdicts[stage] = {}
        for artifact_id, submission in list(round_.submissions.items()):
            if submission.state != expected:
                continue
            verdict = compare(contract, king, models[artifact_id], prompts)
            verdicts[stage][artifact_id] = verdict.status.value
            passed = (
                verdict.primary is not None and verdict.primary.mean > 0
                if stage == "screen"
                else verdict.status == VerdictStatus.PROMOTE_ELIGIBLE
            )
            if not passed:
                round_.transition(artifact_id, SubmissionState.REJECTED)
                continue
            round_.transition(artifact_id, advanced)
            if stage == "duel":
                assert verdict.primary is not None
                duel_bounds[artifact_id] = verdict.primary.lower_bound
    return verdicts, duel_bounds


@dataclass
class SimValidator:
    hotkey: str
    power: int
    store: Store
    signer: HmacSigner
    evaluator_noise: float = 0.0
    online: bool = True
    forced_verdict: LocalVerdict | None = None
    state: FinalizedCompetitionState | None = None


@dataclass
class RoundOutcome:
    round_id: str
    finalist: str | None
    plan: RoundPlan | None
    reports: dict[str, LocalEvaluationReport] = field(default_factory=dict)
    decision: RoundDecision | None = None
    certificate: RoundCertificate | None = None
    commits: list[CommitRecord] = field(default_factory=list)
    reveals: list[Reveal] = field(default_factory=list)
    stage_verdicts: dict[str, dict[str, str]] = field(default_factory=dict)


def block_hash(label: str) -> str:
    """Stand-in for a finalized block hash."""

    return "0x" + hashlib.sha256(label.encode()).hexdigest()


class SimNetwork:
    """A committee of validators sharing one challenge, run in one process."""

    def __init__(
        self,
        contract: ChallengeContract,
        validators: list[SimValidator],
        baseline: ModelProfile,
        *,
        network: str = "sim",
        netuid: int = 0,
        challenge_id: str = "wan13b-16step",
    ):
        self.contract = contract
        self.validators = {validator.hotkey: validator for validator in validators}
        self.network = network
        self.netuid = netuid
        self.challenge_id = challenge_id
        self.models: dict[str, ModelProfile] = {baseline.fingerprint(): baseline}
        self.verifier = HmacVerifier({v.hotkey: v.signer._key for v in validators})
        self.plans: dict[str, RoundPlan] = {}
        self.certificates: list[RoundCertificate] = []
        genesis = FinalizedCompetitionState.genesis(challenge_id, baseline.fingerprint())
        for validator in validators:
            validator.store.initialize_state(
                state_hash=genesis.state_hash(),
                challenge_id=challenge_id,
                state_version=0,
                previous_state_hash=genesis.previous_state_hash,
                payload=genesis.to_dict(),
            )
            validator.state = FinalizedCompetitionState.from_dict(
                validator.store.latest_state_payload(challenge_id) or genesis.to_dict()
            )

    def lead(self) -> SimValidator:
        return next(v for v in self.validators.values() if v.online)

    def run_round(
        self,
        round_id: str,
        entries: list[tuple[str, ModelProfile, CommitmentPosition]],
        *,
        start_block: int,
        chain_time: int,
    ) -> RoundOutcome:
        """Run screen, duel and confirmation, then pre-consensus, for one round."""

        lead = self.lead()
        state = lead.state
        assert state is not None
        king = self.models[state.current_king_fingerprint]
        round_ = KothRound(self.challenge_id, round_id, king.fingerprint(), quota=8)
        for hotkey, model, position in entries:
            self.models[model.fingerprint()] = model
            submission = Submission(hotkey, model.fingerprint(), model.fingerprint(), position)
            round_.submit(submission)
            round_.transition(model.fingerprint(), SubmissionState.ADMITTED)
        outcome = RoundOutcome(round_id, None, None)
        contract_hash = self.contract.contract_hash()

        stage_hashes = {stage: block_hash(f"{round_id}/{stage}") for stage in ("screen", "duel")}
        outcome.stage_verdicts, duel_bounds = screen_and_duel(
            self.contract, round_, king, self.models, stage_hashes
        )
        finalist = round_.select_finalist(duel_bounds)
        if finalist is None:
            return outcome
        round_.lock_finalist(finalist.artifact_id)
        outcome.finalist = finalist.artifact_id
        # Confirmation entropy is drawn only after the finalist is locked.
        confirmation_hash = block_hash(f"{round_id}/confirmation")
        round_.draw_confirmation(confirmation_hash)

        plan = RoundPlan(
            network=self.network,
            netuid=self.netuid,
            round_id=round_id,
            parent_state_hash=state.state_hash(),
            challenge_id=self.challenge_id,
            king_fingerprint=king.fingerprint(),
            challenger_fingerprint=finalist.effective_fingerprint,
            challenger_hotkey=finalist.hotkey,
            contract_hash=contract_hash,
            prompt_seed_inputs={"stage": "confirmation", "block_hash": confirmation_hash},
            committee={v.hotkey: v.power for v in self.validators.values()},
            snapshot_block=start_block,
            commit_deadline_block=start_block + 10,
            reveal_deadline_block=start_block + 20,
            activation_block=start_block + 21,
        )
        outcome.plan = plan
        self.plans[plan.plan_hash()] = plan
        prompts = generate_prompts(self.contract, round_id, "confirmation", confirmation_hash)

        commits: list[CommitRecord] = []
        reveals: list[Reveal] = []
        for validator in self.validators.values():
            if not validator.online:
                continue
            validator.store.create_round(
                round_id, self.challenge_id, plan.plan_hash(), plan.to_dict()
            )
            verdict = compare(
                self.contract,
                king,
                self.models[finalist.artifact_id],
                prompts,
                evaluator_id=validator.hotkey,
                evaluator_noise=validator.evaluator_noise,
            )
            local = {
                VerdictStatus.PROMOTE_ELIGIBLE: LocalVerdict.PROMOTE,
                VerdictStatus.REJECT: LocalVerdict.KEEP,
                VerdictStatus.INCOMPLETE: LocalVerdict.INCOMPLETE,
            }[verdict.status]
            if validator.forced_verdict is not None:
                local = validator.forced_verdict
            report = LocalEvaluationReport(
                network=plan.network,
                netuid=plan.netuid,
                round_id=round_id,
                validator_hotkey=validator.hotkey,
                plan_hash=plan.plan_hash(),
                parent_state_hash=plan.parent_state_hash,
                contract_hash=contract_hash,
                king_fingerprint=plan.king_fingerprint,
                challenger_fingerprint=plan.challenger_fingerprint,
                verdict=local,
                metrics={
                    "lower_bound": None if verdict.primary is None else verdict.primary.lower_bound,
                    "reasons": list(verdict.reasons),
                },
            )
            validator.store.record_report(
                round_id, validator.hotkey, report.report_hash(), report.to_dict()
            )
            outcome.reports[validator.hotkey] = report
            commit, reveal = commit_report(plan, report, start_block + 5)
            commits.append(commit)
            reveals.append(reveal)

        decision = decide_round(plan, commits, reveals)
        outcome.commits = commits
        outcome.reveals = reveals
        outcome.decision = decision
        if decision.status.value == "PENDING":
            return outcome
        # Honest validators sign only the transition they computed themselves.
        expected_verdict = (
            LocalVerdict.PROMOTE if decision.status.value == "PROMOTE" else LocalVerdict.KEEP
        )
        signers = [
            self.validators[hotkey].signer
            for hotkey in decision.valid_validators
            if self.validators[hotkey].online
            and outcome.reports[hotkey].verdict == expected_verdict
        ]
        signer_power = sum(plan.committee[signer.validator_hotkey] for signer in signers)
        if 3 * signer_power <= 2 * plan.total_committee_power:
            return outcome
        certificate = build_certificate(
            plan, decision, chain_time, signers, commits=commits, reveals=reveals
        )
        outcome.certificate = certificate
        self.certificates.append(certificate)
        for validator in self.validators.values():
            if validator.online:
                self.apply(validator, certificate)
        if certificate.transition.value == "PROMOTE":
            round_.confirm(finalist.artifact_id)
            round_.crown(finalist.artifact_id, certificate_verified=True)
        return outcome

    def apply(self, validator: SimValidator, certificate: RoundCertificate) -> None:
        assert validator.state is not None
        verifier = CertificateVerifier(self.verifier, validator.store)
        plan = self.plans[certificate.plan_hash]
        validator.state = apply_certificate(
            validator.state, plan, certificate, verifier, store=validator.store
        )

    def sync(self, validator: SimValidator) -> None:
        """Bring a node that was offline up to the latest certified state."""

        assert validator.state is not None
        for certificate in self.certificates:
            if certificate.parent_state_hash == validator.state.state_hash():
                self.apply(validator, certificate)


def make_committee(
    store_dir: str, powers: dict[str, int] | None = None, noise: float = 0.0
) -> list[SimValidator]:
    powers = powers or {"A": 1, "B": 1, "C": 1, "D": 1}
    return [
        SimValidator(
            hotkey,
            power,
            Store(f"{store_dir}/{hotkey}.sqlite"),
            HmacSigner(hotkey, f"sim-key-{hotkey}".encode()),
            evaluator_noise=noise,
        )
        for hotkey, power in powers.items()
    ]
