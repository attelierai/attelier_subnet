"""Durable no-GPU Launch service lifecycle through real SQLite restarts."""

import json
import sqlite3

import pytest

from forge.evaluation import launch_policy
from forge.evaluation.synthetic_prompts import launch_panel_v2
from forge.protocol.hashing import hash_canonical
from forge.validator.launch_config import LaunchConfig, closed_selection
from forge.validator.launch_service import LaunchServiceStore


def _hash(number: int) -> str:
    return f"{number:064x}"


def _config() -> LaunchConfig:
    return LaunchConfig("test", 578, _hash(1), _hash(2), "lifecycle-fixture")


def _selection(
    config: LaunchConfig, *, close_block: int = 300, candidate_manifest_hash: str = _hash(5)
) -> dict:
    return closed_selection(
        config,
        close_block=close_block,
        close_hash=_hash(3 + close_block),
        finalized_block=close_block,
        parent_state_hash=_hash(4),
        candidate_manifest_hash=candidate_manifest_hash,
        candidate_hotkey="miner",
        commitment_block=close_block,
    )


def _plan(selection: dict, entropy_hash: str) -> dict:
    prompts = launch_panel_v2(entropy_hash, round_id=selection["round_id"])
    timing = sorted(
        (item["prompt_id"] for item in prompts),
        key=lambda value: hash_canonical(
            "forge/launch-v2-timing/v1", entropy_hash, selection["round_id"], value
        ),
    )[:8]
    diversity = []
    for family in sorted({item["family"] for item in prompts}):
        choices = [item["prompt_id"] for item in prompts if item["family"] == family]
        diversity.append(
            min(
                choices,
                key=lambda value: hash_canonical(
                    "forge/launch-v2-diversity/v1",
                    entropy_hash,
                    selection["round_id"],
                    family,
                    value,
                ),
            )
        )
    return launch_policy.freeze_v2(
        {
            "models": {
                "candidate": selection["candidate_manifest_hash"],
                "king": _hash(6),
                "anchor": _hash(7),
            },
            "evaluator": {"evaluator_id": _hash(8), "spec_hash": _hash(9)},
            "runtime": {
                "runtime_id": _hash(10),
                "hardware_id": _hash(11),
                "dependency_lock_hash": _hash(12),
            },
            "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "prompts": prompts,
            "timing_prompt_ids": timing,
            "diversity_prompt_ids": diversity,
            "evaluator_qualification": {
                "schema": "forge/evaluator-qualification/v1",
                "qualification": "QUALIFIED",
                "manifest_id": _hash(13),
                "receipt_id": _hash(14),
                "evaluator_id": _hash(8),
                "guard_thresholds": {
                    "dark_frame_fraction": 0.0,
                    "magenta_frame_fraction": 0.0,
                    "bright_frame_fraction": 0.0,
                    "frozen_fraction": 0.95,
                    "luma_acceleration_p95": 1.0,
                    "high_frequency_proxy_ratio": 0.5,
                    "near_freeze_temporal_max": 0.1 / 255,
                    "near_freeze_ratio_max": 0.01,
                    "near_freeze_reference_floor": 0.1 / 255,
                },
            },
        }
    )


def test_service_freezes_restarts_settles_and_closes_only_bound_evidence(tmp_path):
    config, state = _config(), tmp_path / "service.sqlite"
    selection, entropy = _selection(config), _hash(20)
    service = LaunchServiceStore(state, config)
    assert service.select(selection, finalized_block=300) == selection["round_id"]
    plan = _plan(selection, entropy)
    plan_hash = service.freeze(
        selection["round_id"], finalized_block=310, entropy_hash=entropy, plan=plan
    )
    with sqlite3.connect(state) as db:
        persisted_plan = json.loads(
            db.execute(
                "SELECT plan_json FROM launch_service_rounds WHERE round_id=?",
                (selection["round_id"],),
            ).fetchone()[0]
        )
    assert persisted_plan["evaluator_qualification"]["guard_thresholds"]["frozen_fraction"] == 0.95
    launch_policy.validate_plan(
        persisted_plan, expected_entropy_hash=entropy, expected_round_id=selection["round_id"]
    )
    request = {"round_id": selection["round_id"], "plan_hash": plan_hash, "jobs": ["frozen"]}
    first = service.request_work(selection["round_id"], reserve_usd=4.0, request=request)
    restarted = LaunchServiceStore(state, config)
    assert restarted.request_work(selection["round_id"], reserve_usd=4.0, request=request) == first
    with pytest.raises(ValueError, match="external request changed"):
        restarted.request_work(
            selection["round_id"],
            reserve_usd=4.0,
            request={**request, "jobs": ["changed"]},
        )
    restarted.record_evaluation(
        selection["round_id"], request_hash=first["request_hash"], result={"status": "PASS"}
    )
    restarted.settle_work(selection["round_id"], actual_usd=3.5)
    with pytest.raises(ValueError, match="frozen service plan"):
        restarted.record_certification(
            selection["round_id"],
            certificate_hash=_hash(30),
            certified_state_hash=_hash(31),
            plan_hash=_hash(32),
        )
    restarted.record_certification(
        selection["round_id"],
        certificate_hash=_hash(30),
        certified_state_hash=_hash(31),
        plan_hash=plan_hash,
    )
    submitted = {
        "status": "SUBMITTED",
        "certificate_hash": _hash(30),
        "intent_hash": _hash(33),
        "submitted_extrinsic_hash": _hash(34),
    }
    restarted.record_publication(selection["round_id"], receipt=submitted)
    restarted.record_publication(
        selection["round_id"],
        receipt={
            **submitted,
            "status": "FINALIZED",
            "finalized_block": 350,
            "readback_hash": _hash(35),
        },
    )
    round_ = LaunchServiceStore(state, config).snapshot()["rounds"][0]
    assert round_["status"] == "COMPLETE"
    assert round_["certification"]["plan_hash"] == plan_hash
    assert round_["publication"]["status"] == "FINALIZED"
    assert LaunchServiceStore(state, config).snapshot()["budget"]["reserved_or_captured_usd"] == 3.5


def test_service_expires_a_selected_round_without_evaluation_or_certificate(tmp_path):
    config, service = _config(), LaunchServiceStore(tmp_path / "expired.sqlite", _config())
    selection = _selection(config)
    service.select(selection, finalized_block=300)
    service.expire(selection["round_id"], finalized_block=399, reason="entropy deadline elapsed")
    service.expire(selection["round_id"], finalized_block=399, reason="entropy deadline elapsed")
    round_ = service.snapshot()["rounds"][0]
    assert round_["status"] == "EXPIRED"
    assert round_["evaluation"] is round_["certification"] is round_["publication"] is None


def test_effective_admission_rejects_aliases_across_closed_windows_before_gpu(tmp_path):
    config, state = _config(), tmp_path / "effective-admissions.sqlite"
    service = LaunchServiceStore(state, config)

    def requested_round(close_block: int, manifest_hash: str) -> tuple[dict, dict, str]:
        selection, entropy = (
            _selection(config, close_block=close_block, candidate_manifest_hash=manifest_hash),
            _hash(close_block + 20),
        )
        service.select(selection, finalized_block=close_block)
        plan = _plan(selection, entropy)
        plan_hash = service.freeze(
            selection["round_id"],
            finalized_block=close_block + 10,
            entropy_hash=entropy,
            plan=plan,
        )
        service.request_work(
            selection["round_id"],
            reserve_usd=4.0,
            request={"round_id": selection["round_id"], "plan_hash": plan_hash, "jobs": ["frozen"]},
        )
        return selection, plan, plan_hash

    first_selection, first_plan, _ = requested_round(300, _hash(50))
    first_receipt = {
        "schema": "forge/launch-effective-admission/v1",
        "plan_id": first_plan["plan_id"],
        "contract_hash": config.contract_hash,
        "manifest_hash": first_selection["candidate_manifest_hash"],
        "effective_hash": _hash(51),
        "legacy_effective_hash": _hash(52),
        "weight_content_hash": _hash(53),
    }
    assert service.record_effective_admission(first_selection["round_id"], first_receipt)
    assert not service.record_effective_admission(first_selection["round_id"], first_receipt)
    assert service.effective_admission(first_selection["round_id"]) == first_receipt
    with pytest.raises(ValueError, match="cannot change"):
        service.record_effective_admission(
            first_selection["round_id"], {**first_receipt, "weight_content_hash": _hash(54)}
        )
    service.expire(first_selection["round_id"], finalized_block=310, reason="operator stopped")

    second_selection, second_plan, _ = requested_round(600, _hash(60))
    repeated_alias = {
        **first_receipt,
        "plan_id": second_plan["plan_id"],
        "manifest_hash": second_selection["candidate_manifest_hash"],
    }
    with pytest.raises(ValueError, match="already admitted"):
        service.record_effective_admission(second_selection["round_id"], repeated_alias)
    different = {
        **repeated_alias,
        "effective_hash": _hash(61),
        "legacy_effective_hash": _hash(62),
        "weight_content_hash": _hash(63),
    }
    assert service.record_effective_admission(second_selection["round_id"], different)
    assert service.effective_admission(second_selection["round_id"]) == different
    round_ = service.snapshot()["rounds"][1]
    assert round_["status"] == "REQUESTED"
    assert round_["certification"] is round_["publication"] is None


def test_work_budget_subject_never_reuses_operational_reservation_ids_after_restart(tmp_path):
    config, state = _config(), tmp_path / "mixed-budget.sqlite"
    service = LaunchServiceStore(state, config)
    with sqlite3.connect(state) as db:
        db.executemany(
            "INSERT INTO budget_reservations(candidate_id,operation_key,amount_usd,status) "
            "VALUES (?,?,?,?)",
            [
                (1, "qualification-lease", 10.0, "RESERVED"),
                (2, "qualification-v21-lease", 7.5, "RESERVED"),
            ],
        )
    selection, entropy = _selection(config), _hash(88)
    service.select(selection, finalized_block=300)
    plan_hash = service.freeze(
        selection["round_id"],
        finalized_block=310,
        entropy_hash=entropy,
        plan=_plan(selection, entropy),
    )
    request = {"round_id": selection["round_id"], "plan_hash": plan_hash, "jobs": ["frozen"]}
    envelope = service.request_work(selection["round_id"], reserve_usd=1.0, request=request)
    with sqlite3.connect(state) as db:
        candidate_id = db.execute(
            "SELECT candidate_id FROM budget_reservations WHERE id=?", (envelope["reservation_id"],)
        ).fetchone()[0]
    assert candidate_id > 2
    assert (
        LaunchServiceStore(state, config).request_work(
            selection["round_id"], reserve_usd=1.0, request=request
        )
        == envelope
    )
    # Old databases lack the mapping; a crash can also leave READY after the
    # ledger insert. Recovery must adopt that reservation rather than charge twice.
    with sqlite3.connect(state) as db:
        db.execute("DELETE FROM launch_budget_subjects")
        db.execute(
            "UPDATE launch_service_rounds SET status='READY',reservation_id=NULL,"
            "request_hash=NULL,request_json=NULL WHERE round_id=?",
            (selection["round_id"],),
        )
    recovered = LaunchServiceStore(state, config).request_work(
        selection["round_id"], reserve_usd=1.0, request=request
    )
    assert recovered["reservation_id"] == envelope["reservation_id"]
    with sqlite3.connect(state) as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM budget_reservations WHERE operation_key=?",
                (f"launch:{selection['round_id']}",),
            ).fetchone()[0]
            == 1
        )
