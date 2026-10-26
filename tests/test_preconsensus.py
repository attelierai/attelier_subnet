import dataclasses
import json

import pytest

from forge.protocol.consensus_types import (
    CertificateError,
    ConsensusPolicy,
    DecisionStatus,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    LocalVerdict,
    Reveal,
    RoundPlan,
    Transition,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.koth import CommitmentPosition
from forge.protocol.preconsensus import (
    CertificateVerifier,
    apply_certificate,
    build_certificate,
    commit_report,
    decide_round,
    derive_weights_from_latest_certified_state,
)
from forge.protocol.rewards import RewardRules
from forge.protocol.store import Store, StoreConflictError
from forge.sim import SimNetwork, make_committee, profile

SMALL = ChallengeContract(screen_prompts=16, duel_prompts=64, confirmation_prompts=128)
BASE = profile("baseline", SMALL, 0.70)
GOOD = profile("good", SMALL, 0.80)
BETTER = profile("better", SMALL, 0.90)


@pytest.fixture
def net(tmp_path):
    return SimNetwork(SMALL, make_committee(str(tmp_path)), BASE)


def run(net, round_id="r1", model=GOOD, hotkey="m1", start=100, chain_time=1_000):
    return net.run_round(
        round_id,
        [(hotkey, model, CommitmentPosition(start - 50, 0))],
        start_block=start,
        chain_time=chain_time,
    )


def kings(net):
    return {v.hotkey: v.state.current_king_fingerprint for v in net.validators.values()}


def state_hashes(net):
    return {v.state.state_hash() for v in net.validators.values()}


def test_dissenting_validator_adopts_certified_king(net):
    net.validators["D"].forced_verdict = LocalVerdict.KEEP
    outcome = run(net)
    assert outcome.decision.status == DecisionStatus.PROMOTE
    assert outcome.decision.favorable_power == 3
    assert set(kings(net).values()) == {GOOD.fingerprint()}
    assert len(state_hashes(net)) == 1
    # D keeps its own report unchanged
    assert outcome.reports["D"].verdict == LocalVerdict.KEEP


def test_offline_validator_catches_up(net):
    d = net.validators["D"]
    d.online = False
    run(net)
    assert d.state.current_king_fingerprint == BASE.fingerprint()
    d.online = True
    net.sync(d)
    assert d.state.current_king_fingerprint == GOOD.fingerprint()
    assert len(state_hashes(net)) == 1


def test_two_against_two_does_not_promote(net):
    net.validators["C"].forced_verdict = LocalVerdict.KEEP
    net.validators["D"].forced_verdict = LocalVerdict.KEEP
    outcome = run(net)
    assert outcome.decision.status == DecisionStatus.KEEP
    assert set(kings(net).values()) == {BASE.fingerprint()}
    assert all(v.store.count("crowns") == 0 for v in net.validators.values())


def test_single_responder_has_no_quorum(net):
    for hotkey in ("B", "C", "D"):
        net.validators[hotkey].online = False
    outcome = run(net)
    assert outcome.decision.status == DecisionStatus.PENDING
    assert outcome.decision.total_committee_power == 4  # denominator not reduced
    assert outcome.certificate is None
    assert net.validators["A"].state.current_king_fingerprint == BASE.fingerprint()


def test_incomplete_reports_mean_pending_not_keep(net):
    for hotkey in ("B", "C"):
        net.validators[hotkey].forced_verdict = LocalVerdict.INCOMPLETE
    outcome = run(net)
    assert outcome.decision.status == DecisionStatus.PENDING


def test_stalled_round_closes_as_keep_once_the_reveal_window_closes(net):
    for hotkey in ("C", "D"):
        net.validators[hotkey].online = False
    outcome = run(net)  # only A and B reveal: cannot reach 2/3 yet
    assert outcome.decision.status == DecisionStatus.PENDING
    plan = outcome.plan
    pairs = [commit_report(plan, r, plan.commit_deadline_block) for r in outcome.reports.values()]
    closed = decide_round(
        plan, [c for c, _ in pairs], [r for _, r in pairs], reveal_window_closed=True
    )
    assert closed.status == DecisionStatus.KEEP


def test_unequal_power_two_validators_can_promote(tmp_path):
    committee = make_committee(str(tmp_path), {"A": 5, "B": 4, "C": 2, "D": 1})
    net = SimNetwork(SMALL, committee, BASE)
    net.validators["C"].forced_verdict = LocalVerdict.KEEP
    net.validators["D"].forced_verdict = LocalVerdict.KEEP
    outcome = run(net)
    assert outcome.decision.favorable_power == 9  # 9/12 > 2/3
    assert outcome.decision.status == DecisionStatus.PROMOTE


def test_versioned_stake_majority_makes_one_official_king(tmp_path):
    """A 51% promotion is official even when the remaining validator dissents."""
    committee = make_committee(str(tmp_path), {"A": 51, "B": 49})
    net = SimNetwork(SMALL, committee, BASE)
    state = FinalizedCompetitionState.genesis(net.challenge_id, BASE.fingerprint())
    plan = RoundPlan(
        network="test",
        netuid=9,
        round_id="majority-r1",
        parent_state_hash=state.state_hash(),
        challenge_id=net.challenge_id,
        king_fingerprint=BASE.fingerprint(),
        challenger_fingerprint=GOOD.fingerprint(),
        challenger_hotkey="m1",
        contract_hash=SMALL.contract_hash(),
        prompt_seed_inputs={"panel": "frozen"},
        committee={"A": 51, "B": 49},
        snapshot_block=10,
        commit_deadline_block=11,
        reveal_deadline_block=12,
        activation_block=13,
        consensus_policy=ConsensusPolicy.STAKE_MAJORITY_V2,
    )
    reports = {
        hotkey: LocalEvaluationReport(
            plan.network,
            plan.netuid,
            plan.round_id,
            hotkey,
            plan.plan_hash(),
            plan.parent_state_hash,
            plan.contract_hash,
            plan.king_fingerprint,
            plan.challenger_fingerprint,
            verdict,
        )
        for hotkey, verdict in {"A": LocalVerdict.PROMOTE, "B": LocalVerdict.KEEP}.items()
    }
    pairs = [
        commit_report(plan, report, 11, salt=f"{hotkey}-salt") for hotkey, report in reports.items()
    ]
    commits, reveals = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    decision = decide_round(plan, commits, reveals)
    assert decision.status == DecisionStatus.PROMOTE
    assert decision.favorable_power == 51
    certificate = build_certificate(
        plan,
        decision,
        1_000,
        [net.validators["A"].signer],
        commits=commits,
        reveals=reveals,
    )
    promoted = apply_certificate(state, plan, certificate, CertificateVerifier(net.verifier))
    assert promoted.current_king_fingerprint == GOOD.fingerprint()
    assert reports["B"].verdict == LocalVerdict.KEEP


def test_versioned_majority_keeps_absent_stake_and_rejects_equivocation(tmp_path):
    committee = make_committee(str(tmp_path), {"A": 50, "B": 50})
    net = SimNetwork(SMALL, committee, BASE)
    state = FinalizedCompetitionState.genesis(net.challenge_id, BASE.fingerprint())
    plan = RoundPlan(
        "test",
        9,
        "majority-r2",
        state.state_hash(),
        net.challenge_id,
        BASE.fingerprint(),
        GOOD.fingerprint(),
        "m1",
        SMALL.contract_hash(),
        {"panel": "frozen"},
        {"A": 50, "B": 50},
        10,
        11,
        12,
        13,
        ConsensusPolicy.STAKE_MAJORITY_V2,
    )
    promote = LocalEvaluationReport(
        plan.network,
        plan.netuid,
        plan.round_id,
        "A",
        plan.plan_hash(),
        plan.parent_state_hash,
        plan.contract_hash,
        plan.king_fingerprint,
        plan.challenger_fingerprint,
        LocalVerdict.PROMOTE,
    )
    commit, reveal = commit_report(plan, promote, 11, salt="a-promote")
    assert decide_round(plan, [commit], [reveal]).status == DecisionStatus.PENDING
    assert (
        decide_round(plan, [commit], [reveal], reveal_window_closed=True).status
        == DecisionStatus.KEEP
    )

    b_promote = dataclasses.replace(promote, validator_hotkey="B")
    b_commit, b_reveal = commit_report(plan, b_promote, 11, salt="b-promote")
    conflicting = dataclasses.replace(promote, verdict=LocalVerdict.KEEP)
    bad_commit, bad_reveal = commit_report(plan, conflicting, 11, salt="a-conflict")
    decision = decide_round(plan, [commit, bad_commit, b_commit], [reveal, bad_reveal, b_reveal])
    assert decision.valid_validators == ("B",)
    assert decision.status == DecisionStatus.PENDING


def test_historical_plan_payload_and_hash_remain_replayable(net):
    outcome = run(net)
    payload = outcome.plan.to_dict()
    assert "consensus_policy" not in payload
    restored = RoundPlan.from_dict(payload)
    assert restored == outcome.plan
    assert restored.plan_hash() == outcome.plan.plan_hash()


def test_different_signer_subsets_give_same_state(net):
    outcome = run(net)
    plan, decision = outcome.plan, outcome.decision
    results = set()
    for subset in (("A", "B", "C"), ("B", "C", "D")):
        store = Store(":memory:")
        signers = [net.validators[h].signer for h in subset]
        certificate = build_certificate(
            plan, decision, 1_000, signers, commits=outcome.commits, reveals=outcome.reveals
        )
        genesis_state = FinalizedCompetitionState.genesis(net.challenge_id, BASE.fingerprint())
        store.initialize_state(
            state_hash=genesis_state.state_hash(),
            challenge_id=net.challenge_id,
            state_version=0,
            previous_state_hash=genesis_state.previous_state_hash,
            payload=genesis_state.to_dict(),
        )
        new_state = apply_certificate(
            genesis_state, plan, certificate, CertificateVerifier(net.verifier), store=store
        )
        results.add(new_state.state_hash())
    assert len(results) == 1
    assert results == state_hashes(net)


def test_certificate_needs_more_than_two_thirds_of_power(net):
    outcome = run(net)
    two_signers = [net.validators[h].signer for h in ("A", "B")]
    weak = build_certificate(
        outcome.plan,
        outcome.decision,
        1_000,
        two_signers,
        commits=outcome.commits,
        reveals=outcome.reveals,
    )
    genesis = FinalizedCompetitionState.genesis(net.challenge_id, BASE.fingerprint())
    with pytest.raises(CertificateError, match="two thirds"):
        CertificateVerifier(net.verifier).verify(weak, outcome.plan, genesis)


def test_certificate_against_old_parent_is_rejected(net):
    first = run(net)
    second = run(net, "r2", BETTER, "m2", start=200, chain_time=2_000)
    assert second.decision.status == DecisionStatus.PROMOTE
    a = net.validators["A"]
    # replaying round 1 on top of the round 2 state must fail
    with pytest.raises(CertificateError, match="parent state"):
        CertificateVerifier(net.verifier).verify(first.certificate, first.plan, a.state)


def test_restart_does_not_duplicate_crowns_or_rewards(net, tmp_path):
    outcome = run(net)
    a = net.validators["A"]
    a.store.close()
    reopened = Store(tmp_path / "A.sqlite")
    state = FinalizedCompetitionState.from_dict(reopened.latest_state_payload(net.challenge_id))
    again = apply_certificate(
        state,
        outcome.plan,
        outcome.certificate,
        CertificateVerifier(net.verifier, reopened),
        store=reopened,
    )
    assert again == state
    assert reopened.count("crowns") == 1
    plan = derive_weights_from_latest_certified_state(
        reopened, net.challenge_id, RewardRules(), {"m1"}, 1_000
    )
    assert plan.weights == {"m1": pytest.approx(1.0)}


def test_sqlite_wta_tracks_only_latest_eligible_crown(net):
    first = run(net, "r1", GOOD, "m1", start=100, chain_time=1_000)
    second = run(net, "r2", BETTER, "m2", start=200, chain_time=2_000)
    assert first.decision.status == DecisionStatus.PROMOTE
    assert second.decision.status == DecisionStatus.PROMOTE
    store = net.validators["A"].store
    assert store.count("crowns") == 2

    current = derive_weights_from_latest_certified_state(
        store, net.challenge_id, RewardRules(), {"m1", "m2"}, 2_000
    )
    assert current.weights == {"m2": pytest.approx(1.0)}
    assert current.crown_ids

    unavailable = derive_weights_from_latest_certified_state(
        store, net.challenge_id, RewardRules(), {"m1"}, 2_000
    )
    assert unavailable.weights == {}
    assert unavailable.unallocated_share == pytest.approx(1.0)


def test_double_signing_is_caught_across_restarts(net, tmp_path):
    outcome = run(net)
    keep_decision = dataclasses.replace(outcome.decision, status=DecisionStatus.KEEP)
    signers = [net.validators[h].signer for h in ("A", "B", "C")]
    with pytest.raises(CertificateError, match="does not match"):
        build_certificate(
            outcome.plan,
            keep_decision,
            1_000,
            signers,
            commits=outcome.commits,
            reveals=outcome.reveals,
        )


def test_late_or_altered_reveal_does_not_count(net):
    outcome = run(net)
    plan = outcome.plan
    commits, reveals = [], []
    for hotkey, report in outcome.reports.items():
        commit, reveal = commit_report(plan, report, plan.commit_deadline_block)
        commits.append(commit)
        if hotkey == "A":
            reveal = Reveal(reveal.report, reveal.salt, plan.reveal_deadline_block + 1)  # late
        if hotkey == "B":
            altered = dataclasses.replace(reveal.report, verdict=LocalVerdict.KEEP)
            reveal = Reveal(altered, reveal.salt, reveal.reveal_block)  # does not match commit
        reveals.append(reveal)
    decision = decide_round(plan, commits, reveals)
    assert set(decision.valid_validators) == {"C", "D"}
    assert decision.status == DecisionStatus.PENDING


def test_late_commit_is_refused(net):
    outcome = run(net)
    with pytest.raises(ValueError, match="late"):
        commit_report(outcome.plan, outcome.reports["A"], outcome.plan.commit_deadline_block + 1)


def test_stake_change_after_snapshot_does_not_change_the_round(net):
    outcome = run(net)
    # The committee is frozen inside the plan; mutating the live set later changes nothing.
    net.validators["D"].power = 100
    assert dict(outcome.plan.committee) == {"A": 1, "B": 1, "C": 1, "D": 1}
    with pytest.raises(TypeError):
        outcome.plan.committee["D"] = 100


def test_only_one_pending_round_per_challenge(net):
    store = net.validators["A"].store
    store.create_round("x1", net.challenge_id, "p1", {"n": 1})
    with pytest.raises(StoreConflictError, match="one pending"):
        store.create_round("x2", net.challenge_id, "p2", {"n": 2})


def test_weights_come_only_from_certified_state(tmp_path):
    empty = Store(tmp_path / "empty.sqlite")
    with pytest.raises(ValueError, match="no certified state"):
        derive_weights_from_latest_certified_state(empty, "wan13b-16step", RewardRules(), set(), 0)


def test_node_missing_a_report_waits_instead_of_finalizing_another_history(net):
    net.validators["D"].forced_verdict = LocalVerdict.KEEP
    outcome = run(net)
    plan = outcome.plan
    pairs = [commit_report(plan, r, plan.commit_deadline_block) for r in outcome.reports.values()]
    commits = [commit for commit, _ in pairs]
    # this node could not download C's reveal: A, B promote and D keeps
    reveals = [reveal for _, reveal in pairs if reveal.report.validator_hotkey != "C"]
    partial = decide_round(plan, commits, reveals)
    assert partial.status == DecisionStatus.PENDING  # not KEEP
    # once the certificate reaches it, it adopts the same state as everyone else
    assert outcome.decision.status == DecisionStatus.PROMOTE


def test_v2_certificate_binds_reveals_with_sr25519_and_persistent_store(tmp_path):
    pytest.importorskip("bittensor")
    from bittensor.keyfiles import Keypair

    from forge.chain import Sr25519Signer, Sr25519Verifier

    class Wallet:
        def __init__(self, keypair):
            self.hotkey = keypair

    signers = [
        Sr25519Signer(Wallet(Keypair.create_from_uri(uri)))
        for uri in ("//Alice", "//Bob", "//Charlie")
    ]
    committee = {signer.validator_hotkey: 1 for signer in signers}
    king, challenger = "1" * 64, "2" * 64
    state = FinalizedCompetitionState.genesis("certificate-v2", king)
    plan = RoundPlan(
        network="test",
        netuid=9,
        round_id="certificate-v2-round",
        parent_state_hash=state.state_hash(),
        challenge_id="certificate-v2",
        king_fingerprint=king,
        challenger_fingerprint=challenger,
        challenger_hotkey="candidate-hotkey",
        contract_hash="3" * 64,
        prompt_seed_inputs={"stage": "confirmation"},
        committee=committee,
        snapshot_block=10,
        commit_deadline_block=11,
        reveal_deadline_block=12,
        activation_block=13,
    )

    def evidence(verdict):
        reports = [
            LocalEvaluationReport(
                plan.network,
                plan.netuid,
                plan.round_id,
                signer.validator_hotkey,
                plan.plan_hash(),
                plan.parent_state_hash,
                plan.contract_hash,
                king,
                challenger,
                verdict,
            )
            for signer in signers
        ]
        pairs = [
            commit_report(plan, report, 11, salt=f"salt-{index}")
            for index, report in enumerate(reports)
        ]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    commits, reveals = evidence(LocalVerdict.PROMOTE)
    decision = decide_round(plan, commits, reveals)
    certificate = build_certificate(
        plan, decision, 1_000, signers, commits=commits, reveals=reveals
    )
    serialized = json.loads(json.dumps(certificate.to_dict()))
    restored = type(certificate).from_dict(serialized)
    store_path = tmp_path / "certificate-v2.sqlite"
    with Store(store_path) as store:
        store.initialize_state(
            state_hash=state.state_hash(),
            challenge_id=state.challenge_id,
            state_version=state.state_version,
            previous_state_hash=state.previous_state_hash,
            payload=state.to_dict(),
        )
        verifier = CertificateVerifier(Sr25519Verifier(), store)
        promoted = apply_certificate(state, plan, restored, verifier, store=store)
        assert promoted.current_king_fingerprint == challenger
        for tampered in (
            dataclasses.replace(restored, evidence_digest="0" * 64),
            dataclasses.replace(restored, reveals=restored.reveals[:-1]),
            dataclasses.replace(restored, transition=Transition.KEEP),
        ):
            with pytest.raises(CertificateError):
                verifier.verify(tampered, plan, state)
        keep_commits, keep_reveals = evidence(LocalVerdict.KEEP)
        keep = build_certificate(
            plan,
            decide_round(plan, keep_commits, keep_reveals),
            1_000,
            signers,
            commits=keep_commits,
            reveals=keep_reveals,
        )
        with pytest.raises(CertificateError, match="double-signed"):
            CertificateVerifier(Sr25519Verifier(), store).verify(keep, plan, state)
    with Store(store_path) as reopened:
        assert (
            FinalizedCompetitionState.from_dict(
                reopened.latest_state_payload("certificate-v2")
            ).current_king_fingerprint
            == challenger
        )
