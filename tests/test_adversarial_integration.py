"""Adversarial evidence mutations against the complete automatic decision boundary."""

from forge.evaluation.automatic_gate import evaluate_automatic
from tests.test_automatic_evaluation_integration import _evidence, _history_checkpoint, _preflight


def test_gate_rejects_replayed_provider_response_across_quality_evidence():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    for comparison in quality["comparisons"]:
        for observation in comparison["observations"]:
            observation["calls"][0]["response_id"] = "replayed-response-a"
            observation["calls"][1]["response_id"] = "replayed-response-b"

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["reasons"] == ["provider response_id is missing or replayed"]


def test_gate_rejects_candidate_without_complete_preclose_admission():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    plan["admitted"] = []

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["reasons"] == ["production plan admitted candidates are required"]
