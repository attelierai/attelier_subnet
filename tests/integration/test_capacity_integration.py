"""CLI integration for capacity-only frozen plans; these are not Wan measurements."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.synthetic_prompts import synthetic_panel

ROOT = Path(__file__).resolve().parents[2]


def _frozen_plan(tmp_path: Path, validators: int, *, stage: str = "confirmation") -> Path:
    panel = synthetic_panel(
        "capacity-fixture-entropy",
        round_id=f"capacity-fixture-{validators}",
        phase=stage,
        per_cell=1 if stage == "screening" else 8,
    )
    plan = DistributedEvaluationPlan(
        round_id=f"capacity-fixture-{validators}",
        stage=stage,
        entropy="capacity-fixture-entropy",
        policy_id="capacity-fixture-only",
        candidate_manifest_hash="candidate-fixture",
        king_manifest_hash="king-fixture",
        wan50_manifest_hash="wan50-fixture",
        committee={f"validator-{number}": 1 for number in range(validators)},
        prompts=tuple(panel),
        timing_prompt_ids=tuple(
            row["prompt_id"] for row in panel[: (16 if stage == "screening" else 32)]
        ),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
    )
    path = tmp_path / f"plan-{validators}.json"
    path.write_text(json.dumps(plan.to_dict()))
    return path


def _command(plan: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.cli",
            "distributed-capacity",
            "--plan",
            str(plan),
            "--king-seconds",
            "10",
            "--wan50-seconds",
            "10",
            "--candidate-seconds",
            "10",
            "--gpu-usd-hour",
            "0.70",
            *extra,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_capacity_cli_prices_full_frozen_panels_and_refuses_partial_limits(tmp_path):
    two = _frozen_plan(tmp_path, 2)
    eight = _frozen_plan(tmp_path, 8)
    complete_costs = (
        "--node-overhead-seconds",
        "0",
        "--node-overhead-usd",
        "0",
        "--metric-seconds-per-video",
        "0",
        "--judge-usd-per-quality-comparison",
        "0",
        "--storage-usd-per-video",
        "0",
        "--validator-screening-spent-usd",
        "0",
        "--network-remaining-usd",
        "1000",
        "--commit-delivery-seconds",
        "60",
        "--reveal-delivery-seconds",
        "30",
        "--chain-block-seconds",
        "12",
    )
    accepted = _command(two, *complete_costs)
    assert accepted.returncode == 0, accepted.stderr
    two_result = json.loads(accepted.stdout)
    assert two_result["status"] == "ADMISSIBLE_PROJECTION"
    assert two_result["limits"]["max_node_seconds"] is None
    assert two_result["limits"]["validator_candidate_budget_usd"] == 10
    assert two_result["references_per_quality_prompt"] == 2
    assert two_result["reuse"]["timing_reuses_quality_renders"] is False
    assert two_result["dispatch_authorized"] is False
    assert two_result["promotion_possible"] is False
    assert two_result["windows"]["required_commit_window_seconds"] == (
        two_result["network"]["makespan_seconds"] + 60
    )
    assert two_result["windows"]["required_commit_window_blocks"] >= 1
    assert sum(node["metric_seconds"] for node in two_result["nodes"].values()) == 0

    eight_result = _command(eight, *complete_costs)
    assert eight_result.returncode == 0, eight_result.stderr
    eight_payload = json.loads(eight_result.stdout)
    assert eight_payload["network"]["makespan_seconds"] < two_result["network"]["makespan_seconds"]
    assert eight_payload["network"]["total_usd"] > 0
    assert eight_payload["allocation_warning"]["kind_balanced"] is False

    refused = _command(
        two,
        "--max-node-hours",
        "0.01",
        *complete_costs,
        "--validator-candidate-budget-usd",
        "0.01",
        "--validator-screening-spent-usd",
        "0",
        "--network-remaining-usd",
        "0.01",
    )
    assert refused.returncode == 2
    result = json.loads(refused.stdout)
    assert result["status"] == "INCOMPLETE_INSUFFICIENT_CAPACITY"
    assert {"TIME_CAP_EXCEEDED", "BUDGET_CAP_EXCEEDED"} <= set(result["failures"])

    missing_screening = _command(
        two,
        "--node-overhead-seconds",
        "0",
        "--node-overhead-usd",
        "0",
        "--metric-seconds-per-video",
        "0",
        "--judge-usd-per-quality-comparison",
        "0",
        "--storage-usd-per-video",
        "0",
    )
    assert missing_screening.returncode == 2
    result = json.loads(missing_screening.stdout)
    assert result["status"] == "PROJECTION_INCOMPLETE_INPUTS"
    assert "validator_screening_spent_usd" in result["unknown_inputs"]

    invalid = _command(two, "--king-seconds", "-1")
    assert invalid.returncode != 0
    assert "king_seconds must be positive" in invalid.stderr


def test_capacity_cli_prices_screening_ab_without_changing_confirmation_abba(tmp_path):
    screening = _frozen_plan(tmp_path, 2, stage="screening")
    costs = (
        "--node-overhead-seconds",
        "0",
        "--node-overhead-usd",
        "0",
        "--metric-seconds-per-video",
        "0",
        "--judge-usd-per-quality-comparison",
        "0",
        "--storage-usd-per-video",
        "0",
        "--validator-screening-spent-usd",
        "0",
        "--network-remaining-usd",
        "1000",
    )
    result = json.loads(_command(screening, *costs).stdout)

    assert result["timing_design"] == "screening_ab"
    assert result["reuse"]["timing_runs_per_prompt"] == 2
    # Sixteen AB slots plus four deterministic audit copies, two videos each.
    assert sum(node["videos"]["timing"] for node in result["nodes"].values()) == 40
    # Twenty quality assignments, including audits; king and Wan-50 differ in
    # this fixture, so each uses candidate plus both references.
    assert sum(node["videos"]["quality"] for node in result["nodes"].values()) == 60


def test_capacity_counts_vbench_for_quality_videos_not_timing_renders(tmp_path):
    screening = _frozen_plan(tmp_path, 2, stage="screening")
    result = json.loads(
        _command(
            screening,
            "--node-overhead-seconds",
            "0",
            "--node-overhead-usd",
            "0",
            "--metric-seconds-per-video",
            "1",
            "--judge-usd-per-quality-comparison",
            "0",
            "--storage-usd-per-video",
            "0",
            "--validator-screening-spent-usd",
            "0",
            "--network-remaining-usd",
            "1000",
            "--commit-delivery-seconds",
            "0",
            "--reveal-delivery-seconds",
            "0",
            "--chain-block-seconds",
            "12",
        ).stdout
    )
    assert sum(node["metric_videos"] for node in result["nodes"].values()) == sum(
        node["videos"]["quality"] for node in result["nodes"].values()
    )
    assert sum(node["metric_seconds"] for node in result["nodes"].values()) == 60
