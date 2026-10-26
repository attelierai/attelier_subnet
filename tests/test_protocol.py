import dataclasses

import pytest

from forge.evaluation.benchmark import generate_prompts
from forge.evaluation.scoring import Observation, ObservationStatus
from forge.evaluation.stats import VerdictStatus, paired_comparison
from forge.protocol.contract import ChallengeContract, PropertySlice
from forge.protocol.koth import CommitmentPosition, KothRound, Submission, SubmissionState
from forge.protocol.rewards import (
    Crown,
    RewardRules,
    derive_hotkey_weights,
    map_hotkey_weights_to_uids,
)
from forge.sim import compare, profile, simulate_panel

SMALL = ChallengeContract(screen_prompts=16, duel_prompts=64, confirmation_prompts=128)
DAY = 24 * 60 * 60


def panel(stage="duel", block="0xb1", contract=SMALL):
    return generate_prompts(contract, "r1", stage, block)


def expected_slots(prompts):
    return {(prompt.prompt_id, seed) for prompt in prompts for seed in prompt.video_seeds}


# --- benchmark -------------------------------------------------------------


def test_panel_depends_only_on_contract_round_stage_and_block():
    assert panel() == panel()
    base = {case.prompt_id for case in panel()}
    assert base.isdisjoint({case.prompt_id for case in panel(block="0xb2")})
    assert base.isdisjoint({case.prompt_id for case in panel(stage="confirmation")})
    other = dataclasses.replace(SMALL, min_effect=0.03)
    assert panel()[0].prompt_id != panel(contract=other)[0].prompt_id


def test_panel_is_balanced_and_has_rubrics():
    cases = panel()
    classes = [case.properties["subject_class"] for case in cases]
    assert {classes.count(name) for name in set(classes)} == {len(cases) // 4}
    for case in cases:
        assert len(case.video_seeds) == SMALL.duel_seeds
        assert case.rubric
        if case.properties["subject_class"] == "scenery":
            assert case.properties["subject_count"] == 1
            assert case.properties["direction"] == "none"


# --- admission -------------------------------------------------------------


def _submission(hotkey, fingerprint, block=1, index=0):
    return Submission(hotkey, f"art-{hotkey}", fingerprint, CommitmentPosition(block, index))


def test_copy_of_king_cannot_enter_and_duplicates_are_rejected():
    round_ = KothRound("c", "r1", king_fingerprint="king", quota=4)
    with pytest.raises(ValueError, match="duplicate the current king"):
        round_.submit(_submission("m1", "king"))
    round_.submit(_submission("m1", "fp1"))
    with pytest.raises(ValueError, match="duplicate effective tensor"):
        round_.submit(_submission("m2", "fp1"))
    with pytest.raises(ValueError, match="one submission per hotkey"):
        round_.submit(Submission("m1", "art-other", "fp2", CommitmentPosition(2, 0)))


def test_resubmitting_the_same_submission_is_idempotent():
    round_ = KothRound("c", "r1", king_fingerprint="king", quota=4)
    first = round_.submit(_submission("m1", "fp1"))
    assert round_.submit(_submission("m1", "fp1")) == first


def test_finalist_selection_and_confirmation_ordering():
    round_ = KothRound("c", "r1", king_fingerprint="king", quota=4)
    early = _submission("m1", "fp1", block=5)
    late = _submission("m2", "fp2", block=6)
    for item in (early, late):
        round_.submit(item)
        for state in (SubmissionState.ADMITTED, SubmissionState.SCREENED, SubmissionState.DUELED):
            round_.transition(item.artifact_id, state)
    # equal bounds: the earliest commitment wins
    chosen = round_.select_finalist({early.artifact_id: 0.05, late.artifact_id: 0.05})
    assert chosen.artifact_id == early.artifact_id
    # higher bound wins regardless of commitment order
    chosen = round_.select_finalist({early.artifact_id: 0.04, late.artifact_id: 0.05})
    assert chosen.artifact_id == late.artifact_id
    with pytest.raises(ValueError, match="lock the finalist"):
        round_.draw_confirmation("0xconf")
    round_.lock_finalist(early.artifact_id)
    round_.draw_confirmation("0xconf")
    with pytest.raises(ValueError, match="cannot select or replace"):
        round_.confirm(late.artifact_id)
    round_.confirm(early.artifact_id)
    with pytest.raises(ValueError, match="verified pre-consensus certificate"):
        round_.crown(early.artifact_id, certificate_verified=False)


# --- scoring and statistics -----------------------------------------------


def test_exact_copy_is_not_promoted():
    king = profile("king", SMALL, 0.7)
    copy = dataclasses.replace(king)
    verdict = compare(SMALL, king, copy, panel())
    assert verdict.status == VerdictStatus.REJECT
    assert verdict.primary.lower_bound == 0


def test_clear_improvement_is_promoted():
    verdict = compare(SMALL, profile("king", SMALL, 0.7), profile("better", SMALL, 0.8), panel())
    assert verdict.status == VerdictStatus.PROMOTE_ELIGIBLE, verdict.reasons


def test_high_failure_rate_blocks_promotion():
    flaky = profile("flaky", SMALL, 0.95, failure_rate=0.2)
    verdict = compare(SMALL, profile("king", SMALL, 0.7), flaky, panel())
    assert verdict.status == VerdictStatus.REJECT
    assert any("failure rate" in reason for reason in verdict.reasons)


def test_protected_axis_regression_blocks_better_mean():
    regressing = profile(
        "regress", SMALL, 0.85, overrides={"camera_adherence": 0.45, "prompt_adherence": 0.95}
    )
    verdict = compare(SMALL, profile("king", SMALL, 0.7), regressing, panel())
    assert verdict.primary.mean > 0
    assert verdict.status == VerdictStatus.REJECT
    assert any("camera_adherence" in reason for reason in verdict.reasons)


def test_protected_slice_regression_blocks_promotion():
    contract = dataclasses.replace(
        SMALL, protected_slices=(PropertySlice("humans", "subject_class", "human", 0.01),)
    )
    prompts = panel(contract=contract)
    king = simulate_panel(profile("king", contract, 0.7), prompts)
    better = simulate_panel(profile("better", contract, 0.8), prompts)
    human_ids = {c.prompt_id for c in prompts if c.properties["subject_class"] == "human"}
    worse_on_humans = [
        dataclasses.replace(o, axes={a: 0.3 for a in o.axes}) if o.prompt_id in human_ids else o
        for o in better
    ]
    properties = {c.prompt_id: c.properties for c in prompts}
    verdict = paired_comparison(
        king,
        worse_on_humans,
        contract,
        expected_slots=expected_slots(prompts),
        prompt_properties=properties,
    )
    assert verdict.status == VerdictStatus.REJECT
    assert any("humans" in reason for reason in verdict.reasons)


def test_infra_failure_is_incomplete_not_a_loss():
    prompts = panel()
    king = simulate_panel(profile("king", SMALL, 0.7), prompts)
    challenger = simulate_panel(profile("better", SMALL, 0.8), prompts)
    challenger[3] = Observation(
        challenger[3].prompt_id, challenger[3].seed, None, ObservationStatus.INFRA_FAILURE
    )
    slots = expected_slots(prompts)
    assert (
        paired_comparison(king, challenger, SMALL, expected_slots=slots).status
        == VerdictStatus.INCOMPLETE
    )
    missing = challenger[:-1]
    assert (
        paired_comparison(king, missing, SMALL, expected_slots=slots).status
        == VerdictStatus.INCOMPLETE
    )


def test_extra_slots_are_incomplete_even_when_both_models_match():
    prompts = panel()
    king = simulate_panel(profile("king", SMALL, 0.7), prompts)
    challenger = simulate_panel(profile("better", SMALL, 0.8), prompts)
    slots = expected_slots(prompts)

    # A validator cannot silently expand the prescribed panel with more seeds.
    def doubled(observations):
        extra = [dataclasses.replace(o, seed=o.seed + 1) for o in observations]
        return observations + extra

    more_seeds = paired_comparison(doubled(king), doubled(challenger), SMALL, expected_slots=slots)
    assert more_seeds.status == VerdictStatus.INCOMPLETE


# --- rewards ----------------------------------------------------------------


def _crown(i, hotkey, days_ago, now=100 * DAY, baseline=False):
    return Crown(f"c{i}", hotkey, f"fp{i}", now - days_ago * DAY, baseline)


def test_only_the_current_crown_receives_all_weight():
    crowns = [_crown(0, "h0", 4), _crown(1, "h1", 3), _crown(2, "h2", 2), _crown(3, "h3", 1)]
    plan = derive_hotkey_weights(crowns, RewardRules(), {"h0", "h1", "h2", "h3"}, 100 * DAY)
    assert plan.weights == {"h3": pytest.approx(1.0)}
    assert plan.unallocated_share == 0
    assert plan.crown_ids == ("c3",)


def test_current_ineligible_or_baseline_crown_leaves_weight_unallocated():
    crowns = [_crown(1, "h1", 29), _crown(2, "base", 1, baseline=True)]
    plan = derive_hotkey_weights(crowns, RewardRules(), {"h1", "base"}, 100 * DAY)
    assert plan.weights == {}
    assert plan.unallocated_share == 1.0

    promoted = [_crown(1, "h1", 29), _crown(2, "h2", 1)]
    ineligible = derive_hotkey_weights(promoted, RewardRules(), {"h1"}, 100 * DAY)
    assert ineligible.weights == {}
    assert ineligible.unallocated_share == 1.0


def test_recycled_uid_does_not_pay_the_new_occupant():
    plan = derive_hotkey_weights([_crown(1, "old", 1)], RewardRules(), {"old"}, 100 * DAY)
    uids = map_hotkey_weights_to_uids(plan, {7: "newcomer"})
    assert uids.weights == {}
    assert uids.unallocated_share == 1.0
