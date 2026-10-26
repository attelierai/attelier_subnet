"""Distributed prompt work is deterministic, signed and globally de-duplicated."""

from __future__ import annotations

from copy import deepcopy

import pytest

from forge.evaluation.distributed import (
    DistributedEvaluationPlan,
    aggregate_partial_reports,
    allocate_work,
    assignments_for,
    signed_partial_report,
)
from forge.evaluation.mvp_policy import QUALITY_AXES
from forge.sim import HmacSigner, HmacVerifier


def _prompts() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "prompt_id": f"p-{number}",
            "text": f"synthetic moving scene {number}",
            "seed": number,
            "subject_class": "object",
            "camera_type": "moving",
            "style_family": "natural",
        }
        for number in range(16)
    )


def _plan(*, quality_tolerance: float = 0.5) -> DistributedEvaluationPlan:
    return DistributedEvaluationPlan(
        round_id="distributed-test-r1",
        stage="screening",
        entropy="finalized-block-hash",
        policy_id="frozen-policy",
        candidate_manifest_hash="candidate",
        king_manifest_hash="king",
        wan50_manifest_hash="wan50",
        committee={"validator-a": 3, "validator-b": 1},
        prompts=_prompts(),
        timing_prompt_ids=tuple(f"p-{number}" for number in range(16)),
        quality_tolerance=quality_tolerance,
        timing_log_ratio_tolerance=0.5,
        audit_calibration_hash="calibrated",
    )


def _signers():
    keys = {"validator-a": b"a", "validator-b": b"b"}
    return (
        {hotkey: HmacSigner(hotkey, key) for hotkey, key in keys.items()},
        HmacVerifier(keys),
    )


def _quality(prompt_id: str, seed: int, delta: float) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "subject_class": "object",
        "camera_type": "moving",
        "style_family": "natural",
        "orders": [
            {"reference_first": True, "axes": {axis: delta for axis in QUALITY_AXES}},
            {"reference_first": False, "axes": {axis: delta for axis in QUALITY_AXES}},
        ],
    }


def _timing(prompt_id: str, seed: int, ratio: float) -> dict[str, object]:
    candidate = 1 / ratio
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "runs": [
            {"model": "reference", "seconds": 1.0},
            {"model": "candidate", "seconds": candidate},
        ],
        "timing_schema": "forge/screen-timing/v1",
    }


def _report(plan, signer, *, delta=0.0, ratio=1.5, omit=()):
    assigned = assignments_for(plan, signer.validator_hotkey)
    omitted = set(omit)
    quality = [
        _quality(item.prompt_id, item.seed, delta)
        for item in assigned
        if item.kind == "quality" and item.prompt_id not in omitted
    ]
    timing = [
        _timing(item.prompt_id, item.seed, ratio)
        for item in assigned
        if item.kind == "timing" and item.prompt_id not in omitted
    ]
    return signed_partial_report(
        plan,
        signer,
        quality_vs_king=quality,
        quality_vs_wan50=deepcopy(quality),
        timing=timing,
    )


def test_equal_load_repeated_work_is_stake_aggregated_once_per_prompt():
    plan = _plan()
    signers, verifier = _signers()
    allocation = allocate_work(plan)

    assert len(assignments_for(plan, "validator-a")) == len(assignments_for(plan, "validator-b"))
    assert len(allocation) == 40
    assert sum(item.audit for item in allocation) == 16
    assert DistributedEvaluationPlan.from_dict(plan.to_dict()).plan_hash() == plan.plan_hash()
    reports = [
        _report(plan, signers["validator-a"], delta=0.0, ratio=1.5),
        _report(plan, signers["validator-b"], delta=0.4, ratio=1.4),
    ]

    aggregate = aggregate_partial_reports(plan, reports, verifier)

    assert aggregate["status"] == "COMPLETE"
    assert len(aggregate["quality_vs_king"]) == 16
    assert len(aggregate["quality_vs_wan50"]) == 16
    assert len(aggregate["timing"]) == 16
    audited_prompt = next(
        item.prompt_id for item in allocation if item.audit and item.kind == "quality"
    )
    audited = next(
        row for row in aggregate["quality_vs_king"] if row["prompt_id"] == audited_prompt
    )
    # Three units of stake report 0.0 and one reports 0.4; the prompt remains one sample.
    assert audited["orders"][0]["axes"]["image_quality"] == pytest.approx(0.1)


def test_missing_tampered_and_equivocating_reports_cannot_complete_the_global_panel():
    plan = _plan()
    signers, verifier = _signers()
    complete = _report(plan, signers["validator-a"])
    missing_id = assignments_for(plan, "validator-b")[0].prompt_id
    missing = _report(plan, signers["validator-b"], omit={missing_id})
    assert aggregate_partial_reports(plan, [complete, missing], verifier)["status"] == "INCOMPLETE"

    tampered = deepcopy(_report(plan, signers["validator-b"]))
    tampered["quality_vs_king"][0]["orders"][0]["axes"]["image_quality"] = 0.9
    invalid = aggregate_partial_reports(plan, [complete, tampered], verifier)
    assert invalid["status"] == "INCOMPLETE"
    assert any(reason.startswith("INVALID_REPORT") for reason in invalid["invalid"])

    equivocation = _report(plan, signers["validator-a"], omit={"p-1"})
    result = aggregate_partial_reports(
        plan, [complete, equivocation, _report(plan, signers["validator-b"])], verifier
    )
    assert result["status"] == "INCOMPLETE"
    assert "EQUIVOCATION:validator-a" in result["invalid"]


def test_contested_overlap_blocks_a_promotion_input():
    plan = _plan(quality_tolerance=0.1)
    signers, verifier = _signers()
    aggregate = aggregate_partial_reports(
        plan,
        [
            _report(plan, signers["validator-a"], delta=0.0),
            _report(plan, signers["validator-b"], delta=-0.8),
        ],
        verifier,
    )

    assert aggregate["status"] == "CONTESTED"
    assert any(value.startswith("quality:") for value in aggregate["contested"])


def test_stage_bound_screening_ab_rejects_an_abba_receipt():
    plan = _plan()
    signers, verifier = _signers()
    valid = _report(plan, signers["validator-a"])
    invalid = deepcopy(_report(plan, signers["validator-b"]))
    invalid["timing"][0]["runs"].extend(
        [
            {"model": "candidate", "seconds": 0.5},
            {"model": "reference", "seconds": 1.0},
        ]
    )

    aggregate = aggregate_partial_reports(plan, [valid, invalid], verifier)

    assert aggregate["status"] == "INCOMPLETE"
    assert any(reason.startswith("INVALID_REPORT") for reason in aggregate["invalid"])
