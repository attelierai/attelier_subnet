"""The pre-score pilot receipt must bind all frozen media and treatments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from forge.evaluation.pilot_observability import (
    PERCEPTUAL_RULES,
    RECEIPT_SCHEMA,
    _sha,
    freeze_pilot_observability_protocol,
    validate_pilot_observability_receipt,
)
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.protocol.hashing import hash_canonical


def _receipt(protocol: dict[str, object]) -> dict[str, object]:
    job_ids = {
        job_id
        for requirement in protocol["requirements"]
        for job_id in (requirement["reference_job_id"], requirement["candidate_job_id"])
    }
    media = [
        {
            "job_id": job_id,
            "sha256": hash_canonical("forge/pilot-observability-test-media/v1", job_id),
        }
        for job_id in sorted(job_ids)
    ]
    digests = {row["job_id"]: row["sha256"] for row in media}
    reviews = [
        {
            **requirement,
            "reference_sha256": digests[requirement["reference_job_id"]],
            "candidate_sha256": digests[requirement["candidate_job_id"]],
            "observable": True,
            "review_status": "PRE_SCORE_COMPLETE",
        }
        for requirement in protocol["requirements"]
    ]
    body = {
        "schema": RECEIPT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "pilot_id": protocol["pilot_id"],
        "worklist_id": protocol["worklist_id"],
        "media": media,
        "reviews": reviews,
        "judging_permitted": True,
        "qualification": "prohibited",
    }
    return {**body, "receipt_id": _sha(body)}


def test_observability_receipt_covers_the_actual_96_frozen_comparisons():
    root = Path(__file__).resolve().parents[2]
    pilot = json.loads((root / "configs/evaluator-pilot-manifest-v1.json").read_text())
    protocol = freeze_pilot_observability_protocol(pilot, pilot_worklist(pilot))
    result = validate_pilot_observability_receipt(protocol, _receipt(protocol))

    assert len(protocol["requirements"]) == 96
    assert set(protocol["perceptual_rules"]) == {
        "self",
        "same_model_new_run",
        "independent_seed",
        "static_scene",
        "freeze",
        "blur_mild",
        "blur_severe",
        "shuffle",
        "wrong_video",
        "detail_loss",
        "evaluator_attack_mild",
        "evaluator_attack_severe",
    }
    assert "static content is penalized" in PERCEPTUAL_RULES["static_scene"]["reject_when"]
    assert result == {
        "status": "PASS",
        "pilot_id": pilot["pilot_id"],
        "protocol_id": protocol["protocol_id"],
        "observable_comparisons": 96,
        "judging_permitted": True,
        "qualification": "prohibited",
    }


def test_rehashed_protocol_with_changed_requirement_is_rejected_when_bound_to_pilot():
    root = Path(__file__).resolve().parents[2]
    pilot = json.loads((root / "configs/evaluator-pilot-manifest-v1.json").read_text())
    worklist = pilot_worklist(pilot)
    protocol = freeze_pilot_observability_protocol(pilot, worklist)
    altered = deepcopy(protocol)
    altered["requirements"][0]["treatment_hash"] = "0" * 64
    body = {key: value for key, value in altered.items() if key != "protocol_id"}
    altered["protocol_id"] = _sha(body)

    with pytest.raises(ValueError, match="differs from the frozen pilot worklist"):
        validate_pilot_observability_receipt(
            altered, _receipt(altered), pilot=pilot, worklist=worklist
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt["reviews"].pop(), "does not cover every frozen comparison"),
        (
            lambda receipt: receipt["reviews"][0].__setitem__("lineage_hash", "0" * 64),
            "differs from frozen lineage or treatment",
        ),
        (
            lambda receipt: receipt["reviews"][0].__setitem__("observable", False),
            "lacks verified observable media",
        ),
        (lambda receipt: receipt["media"].pop(), "lacks verified observable media"),
    ],
)
def test_observability_rejects_missing_unobservable_or_substituted_evidence(mutation, message):
    root = Path(__file__).resolve().parents[2]
    pilot = json.loads((root / "configs/evaluator-pilot-manifest-v1.json").read_text())
    protocol = freeze_pilot_observability_protocol(pilot, pilot_worklist(pilot))
    receipt = _receipt(protocol)
    mutation(receipt)
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = _sha(body)

    with pytest.raises(ValueError, match=message):
        validate_pilot_observability_receipt(protocol, receipt)
