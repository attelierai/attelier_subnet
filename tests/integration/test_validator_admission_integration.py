"""Integration coverage for the SQLite admission/budget boundary."""

import sqlite3

import pytest

from forge.protocol.hashing import hash_canonical
from forge.validator import (
    AdmissionError,
    AdmissionLedger,
    AuthenticatedAdmissionHistory,
    BudgetError,
    BudgetLedger,
    BudgetPolicy,
    EvaluationCoordinator,
    FinalizedCommitment,
    UncertainChargeError,
)
from forge.validator.admission import build_authenticated_history


def digest(number: int) -> str:
    return f"{number:064x}"


def history(*hotkeys: str, height: int = 100) -> AuthenticatedAdmissionHistory:
    registered = tuple(
        sorted(
            {
                *hotkeys,
                "miner-a",
                "miner-b",
                "miner-a-new-hotkey",
                "miner",
                "fresh",
                "other",
                "spent",
            }
        )
    )
    consumed = tuple(sorted(hotkeys))
    payload = {
        "network": "test",
        "netuid": 9,
        "through_block": height,
        "certified_state_hash": digest(1),
        "consumed_hotkeys": consumed,
        "registered_hotkeys": registered,
        "lineage_edges": (),
    }
    return AuthenticatedAdmissionHistory(
        "test",
        9,
        height,
        digest(1),
        hash_canonical("forge/validator-admission-history/v1", payload),
        consumed,
        registered,
    )


def commitment(hotkey: str, number: int, block: int = 99) -> FinalizedCommitment:
    return FinalizedCommitment(
        hotkey,
        block,
        number,
        digest(number),
        digest(number + 10),
        digest(number + 20),
        digest(number + 30),
    )


def verified(value: AuthenticatedAdmissionHistory) -> bool:
    return value.history_root == value.recomputed_root()


def rooted_history(
    *,
    registered: tuple[str, ...],
    consumed: tuple[str, ...] = (),
    edges: tuple[tuple[str, str], ...] = (),
) -> AuthenticatedAdmissionHistory:
    payload = {
        "network": "test",
        "netuid": 9,
        "through_block": 100,
        "certified_state_hash": digest(1),
        "consumed_hotkeys": consumed,
        "registered_hotkeys": registered,
        "lineage_edges": edges,
    }
    return AuthenticatedAdmissionHistory(
        "test",
        9,
        100,
        digest(1),
        hash_canonical("forge/validator-admission-history/v1", payload),
        consumed,
        registered,
        edges,
    )


def test_history_is_required_fifo_persists_and_hotkey_lineage_survives_restart(tmp_path):
    path = tmp_path / "validator.sqlite"
    ledger = AdmissionLedger(path, network="test", netuid=9)
    with pytest.raises(AdmissionError, match="authenticated"):
        ledger.admit(commitment("old-hotkey", 10), history=None, verify_history=None)
    first = ledger.admit(commitment("miner-a", 11), history=history(), verify_history=verified)
    second = ledger.admit(commitment("miner-b", 12), history=history(), verify_history=verified)
    assert (first.queue_sequence, second.queue_sequence) == (1, 2)
    assert ledger.next_queued().id == first.id

    reopened = AdmissionLedger(path, network="test", netuid=9)
    with pytest.raises(AdmissionError, match="already consumed"):
        reopened.admit(commitment("miner-a", 99), history=history(), verify_history=verified)
    # A different registered hotkey is intentionally not rejected by coldkey/operator identity.
    new_key = reopened.admit(
        commitment("miner-a-new-hotkey", 13), history=history(), verify_history=verified
    )
    assert new_key.queue_sequence == 3


def test_certified_prior_hotkey_is_blocked_and_conflicting_history_fails_closed(tmp_path):
    ledger = AdmissionLedger(tmp_path / "validator.sqlite", network="test", netuid=9)
    with pytest.raises(AdmissionError, match="already consumed"):
        ledger.admit(commitment("spent", 20), history=history("spent"), verify_history=verified)
    ledger.admit(commitment("fresh", 21), history=history(), verify_history=verified)
    conflicting_payload = {
        "network": "test",
        "netuid": 9,
        "through_block": 100,
        "certified_state_hash": digest(3),
        "consumed_hotkeys": (),
        "registered_hotkeys": history().registered_hotkeys,
        "lineage_edges": (),
    }
    conflicting = AuthenticatedAdmissionHistory(
        "test",
        9,
        100,
        digest(3),
        hash_canonical("forge/validator-admission-history/v1", conflicting_payload),
        (),
        history().registered_hotkeys,
    )
    with pytest.raises(AdmissionError, match="conflicts"):
        ledger.admit(
            commitment("other", 22), history=conflicting, verify_history=lambda _item: True
        )


def test_resume_is_bound_to_frozen_inputs_and_uncertain_charge_cannot_retry(tmp_path):
    path = tmp_path / "validator.sqlite"
    ledger = AdmissionLedger(path, network="test", netuid=9)
    candidate = ledger.admit(commitment("miner", 30), history=history(), verify_history=verified)
    budget = BudgetLedger(path, BudgetPolicy(10.0))
    coordinator = EvaluationCoordinator(ledger, budget)
    work = coordinator.begin_next_screening(operation_key="screen-gpu-1", reserve_usd=4.0)
    assert work and work.admission.id == candidate.id
    coordinator.infrastructure_failed(candidate.id)
    with pytest.raises(AdmissionError, match="frozen"):
        coordinator.resume(
            candidate.id,
            artifact_hash=digest(999),
            recipe_hash=candidate.recipe_hash,
            panel_hash=candidate.panel_hash,
            operation_key="retry",
            reserve_usd=2.0,
        )
    resumed = coordinator.resume(
        candidate.id,
        artifact_hash=candidate.artifact_hash,
        recipe_hash=candidate.recipe_hash,
        panel_hash=candidate.panel_hash,
        operation_key="screen-gpu-2",
        reserve_usd=2.0,
    )
    assert resumed.stage == "SCREENING"
    budget.mark_uncertain(resumed.reservation.id)
    with pytest.raises(UncertainChargeError, match="do not retry"):
        budget.reserve(candidate.id, "screen-gpu-2", 2.0)
    with pytest.raises(UncertainChargeError, match="reconcile"):
        budget.reserve(candidate.id, "full-evaluation", 5.0)


def test_rejected_full_resume_preserves_budget_across_restart(tmp_path):
    path = tmp_path / "validator.sqlite"
    ledger = AdmissionLedger(path, network="test", netuid=9)
    candidate = ledger.admit(commitment("miner", 30), history=history(), verify_history=verified)
    budget = BudgetLedger(path, BudgetPolicy(10.0))
    coordinator = EvaluationCoordinator(ledger, budget)
    coordinator.begin_next_screening(operation_key="screen", reserve_usd=1.0)
    coordinator.begin_full_evaluation(
        candidate.id, round_id="frozen-round", operation_key="full", reserve_usd=1.0
    )
    coordinator.infrastructure_failed(candidate.id)
    before = budget.summary(candidate.id)
    for number, round_id in enumerate((None, "different-round")):
        coordinator = EvaluationCoordinator(
            AdmissionLedger(path, network="test", netuid=9), BudgetLedger(path, BudgetPolicy(10.0))
        )
        with pytest.raises(AdmissionError, match="frozen round"):
            coordinator.resume(
                candidate.id,
                artifact_hash=candidate.artifact_hash,
                recipe_hash=candidate.recipe_hash,
                panel_hash=candidate.panel_hash,
                operation_key=f"rejected-{number}",
                reserve_usd=2.0,
                round_id=round_id,
            )
        assert budget.summary(candidate.id) == before
        assert ledger.get(candidate.id).status == "INFRASTRUCTURE_FAILED"
    resumed = coordinator.resume(
        candidate.id,
        artifact_hash=candidate.artifact_hash,
        recipe_hash=candidate.recipe_hash,
        panel_hash=candidate.panel_hash,
        operation_key="valid-resume",
        reserve_usd=2.0,
        round_id="frozen-round",
    )
    assert resumed.stage == "FULL_EVALUATION"
    assert ledger.get(candidate.id).status == "FULL_EVALUATION"


def test_authenticated_swap_inherits_consumption_and_fifo_uses_finalized_position(tmp_path):
    ledger = AdmissionLedger(tmp_path / "validator.sqlite", network="test", netuid=9)
    swapped = rooted_history(registered=("late", "new", "old", "other"), edges=(("old", "new"),))
    first = ledger.admit(commitment("old", 1, 80), history=swapped, verify_history=verified)
    assert first.extrinsic_index == 1
    with pytest.raises(AdmissionError, match="lineage was already consumed"):
        ledger.admit(commitment("new", 2, 81), history=swapped, verify_history=verified)
    with pytest.raises(AdmissionError, match="not registered"):
        ledger.admit(commitment("unregistered", 3, 82), history=swapped, verify_history=verified)
    # Arrival after a later finalized event cannot quietly change FIFO.
    ledger.admit(commitment("other", 9, 90), history=swapped, verify_history=verified)
    with pytest.raises(AdmissionError, match="out-of-order"):
        ledger.admit(commitment("late", 4, 89), history=swapped, verify_history=verified)


def test_windowed_history_allows_a_new_candidate_later_but_not_a_new_sample_or_pending_lineage(
    tmp_path,
):
    ledger = AdmissionLedger(tmp_path / "windowed.sqlite", network="test", netuid=9)

    def window(identifier, close, state):
        return build_authenticated_history(
            network="test",
            netuid=9,
            through_block=close,
            certified_state_hash=digest(state),
            consumed_hotkeys=(),
            registered_hotkeys=("miner",),
            window_id=identifier,
            window_open_block=close - 100,
            window_close_block=close,
        )

    first_history = window("test/9/100", 100, 1)
    first = ledger.admit(
        commitment("miner", 1, 99),
        history=first_history,
        verify_history=verified,
        effective_hash=digest(80),
    )
    later_history = window("test/9/200", 200, 2)
    with pytest.raises(AdmissionError, match="pending admission"):
        ledger.admit(
            commitment("miner", 2, 199),
            history=later_history,
            verify_history=verified,
            effective_hash=digest(81),
        )
    assert ledger.claim_next_screening().id == first.id
    ledger.transition(first.id, "INCONCLUSIVE")
    with pytest.raises(AdmissionError, match="effective candidate identity"):
        ledger.admit(
            commitment("miner", 2, 199),
            history=later_history,
            verify_history=verified,
            effective_hash=digest(80),
        )
    later = ledger.admit(
        commitment("miner", 3, 200),
        history=later_history,
        verify_history=verified,
        effective_hash=digest(81),
    )
    assert later.id != first.id and later.status == "QUEUED"


def test_multiple_full_evaluations_share_round_budget_and_preserve_claims_on_restart(tmp_path):
    path = tmp_path / "validator.sqlite"
    ledger = AdmissionLedger(path, network="test", netuid=9)
    one = ledger.admit(commitment("miner-a", 1), history=history(), verify_history=verified)
    two = ledger.admit(commitment("miner-b", 2), history=history(), verify_history=verified)
    budget = BudgetLedger(path, BudgetPolicy(10.0, session_limit_usd=10.0, session_quota_usd=9.0))
    coordinator = EvaluationCoordinator(ledger, budget)
    assert (
        coordinator.begin_next_screening(operation_key="one-screen", reserve_usd=4.0).admission.id
        == one.id
    )
    assert (
        coordinator.begin_next_screening(operation_key="two-screen", reserve_usd=3.0).admission.id
        == two.id
    )
    assert (
        coordinator.begin_full_evaluation(
            one.id, round_id="r1", operation_key="one-full", reserve_usd=1.0
        ).stage
        == "FULL_EVALUATION"
    )
    # Persist a populated legacy claim table, then exercise the real reopen migration.
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE full_evaluation_rounds RENAME TO fixture_current_claims")
        db.execute(
            "CREATE TABLE full_evaluation_rounds (network TEXT NOT NULL, netuid INTEGER NOT NULL, "
            "round_id TEXT NOT NULL, admission_id INTEGER NOT NULL, "
            "PRIMARY KEY(network, netuid, round_id))"
        )
        db.execute("INSERT INTO full_evaluation_rounds SELECT * FROM fixture_current_claims")
        db.execute("DROP TABLE fixture_current_claims")
    ledger = AdmissionLedger(path, network="test", netuid=9)
    coordinator = EvaluationCoordinator(ledger, budget)
    ledger.transition(one.id, "INFRASTRUCTURE_FAILED")
    with pytest.raises(AdmissionError, match="frozen round"):
        ledger.start_full_evaluation(one.id, round_id="reroll")
    ledger.start_full_evaluation(one.id, round_id="r1")
    assert (
        coordinator.begin_full_evaluation(
            two.id, round_id="r1", operation_key="two-full", reserve_usd=0.5
        ).stage
        == "FULL_EVALUATION"
    )
    ledger = AdmissionLedger(path, network="test", netuid=9)
    assert ledger.get(one.id).status == ledger.get(two.id).status == "FULL_EVALUATION"
    ledger.transition(two.id, "INFRASTRUCTURE_FAILED")
    with pytest.raises(AdmissionError, match="frozen round"):
        ledger.start_full_evaluation(two.id, round_id="reroll")
    reservation = budget.reserve(two.id, "provider", 0.5)
    budget.mark_uncertain(reservation.id)
    with pytest.raises(UncertainChargeError, match="reconcile"):
        budget.reserve(two.id, "alias", 0.1)
    settled = budget.reconcile_uncertain(
        reservation.id, 0.5, provider_receipt_id="provider-verified-1"
    )
    assert settled.status == "CAPTURED"
    assert budget.summary(one.id).warning_at_80_percent
    with pytest.raises(BudgetError):
        BudgetPolicy(float("nan"))
    with pytest.raises(BudgetError):
        BudgetPolicy(True)
    with pytest.raises(BudgetError, match="cannot change"):
        budget.reserve(two.id, "provider", 0.6)


def test_campaign_budget_is_explicit_before_a_validator_can_exceed_default(tmp_path):
    with pytest.raises(BudgetError, match="explicit campaign budget"):
        BudgetPolicy(10.01)
    policy = BudgetPolicy(12.0, session_limit_usd=24.0, campaign_limit_usd=24.0)
    ledger = BudgetLedger(tmp_path / "campaign.sqlite", policy)
    assert ledger.reserve(1, "validator-a", 12.0).status == "RESERVED"
    assert ledger.summary(1).campaign_limit_usd == 24.0
    assert ledger.reserve(2, "validator-b", 12.0).status == "RESERVED"
    with pytest.raises(BudgetError, match="session budget or quota"):
        ledger.reserve(3, "validator-c", 0.01)
