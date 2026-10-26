"""Durable selection and confirmation lock over real SQLite and JSON evidence files."""

import json

import pytest

from forge.evaluation.launch_baseline import ParentLaunchBaselineStore


def digest(value):
    return f"{value:064x}"


def screening(candidate, anchor):
    rows = []
    for index in range(6):
        rows.append(
            {
                "prompt_id": f"screen-{index}",
                "status": "OK",
                "vetoes": [],
                "reference_seconds": 2.0,
                "candidate_seconds": 1.0,
                "reference_scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                "candidate_scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
            }
        )
    return {"candidate_hash": candidate, "anchor_hash": anchor, "rows": rows}


def test_selection_is_canonical_and_failed_v2_confirmation_locks_wan50(tmp_path):
    anchor, candidate = digest(1), digest(2)
    evidence = screening(candidate, anchor)
    path = tmp_path / "screening.json"
    path.write_text(json.dumps(evidence))
    store = ParentLaunchBaselineStore(tmp_path / "state" / "baseline.sqlite")
    selected = store.select_once([json.loads(path.read_text())], anchor_hash=anchor)
    assert selected["selected_candidate_hash"] == candidate
    # This deliberately incomplete raw V2 plan is replayed to INCOMPLETE; a caller
    # cannot supply a claimed PASS, and the first evidence then locks the decision.
    plan = {"models": {"candidate": candidate}}
    fake_qualified = {"plan": {}, "raw": {}, "result": {"qualification": "QUALIFIED"}}
    with pytest.raises(ValueError):
        store.confirm_once(plan, {"rows": []}, fake_qualified)
    # A fabricated QUALIFIED flag never reaches evaluate_v2 or imports a baseline.
    with pytest.raises(ValueError):
        store.confirm_once(
            {"models": {"candidate": candidate}, "changed": True}, {"rows": []}, fake_qualified
        )
    with pytest.raises(ValueError, match="immutable"):
        store.select_once([screening(digest(3), anchor)], anchor_hash=anchor)
