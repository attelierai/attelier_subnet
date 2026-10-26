"""Exercise the actual frozen development plan and offline budget CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_frozen_pilot_preflight_preserves_unknown_costs_and_never_authorizes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    costs = json.loads((root / "configs/evaluator-pilot-costs.json").read_text())
    pilot = root / "configs/evaluator-pilot-manifest-v1.json"
    for label, updates, status in (
        ("proposed", {}, "INCOMPLETE"),
        ("unknown", {"storage_usd": None}, "INCOMPLETE"),
        ("over", {"gpu_usd_hour": 10.0}, "OVER_CAP"),
    ):
        cost_path, output = tmp_path / f"{label}-costs.json", tmp_path / f"{label}-out.json"
        cost_path.write_text(json.dumps({**costs, **updates}))
        command = [
            sys.executable,
            str(root / "scripts/evaluator_pilot_preflight.py"),
            "--pilot",
            str(pilot),
            "--costs",
            str(cost_path),
            "--out",
            str(output),
            "--guard-envelope",
            str(root / "configs/evaluator-pilot-guard-envelope.json"),
        ]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        assert result.returncode == 2, result.stderr
        receipt = json.loads(output.read_text())
        assert receipt["status"] == status
        assert receipt["dispatch_authorized"] is False
        assert receipt["promotion_eligible"] is False
        if label == "proposed":
            assert receipt["guard_work_margin_seconds"] > 0
            assert receipt["guard_problems"] == []
        assert receipt["ledger"]["wan50_gpu_renders"] == 32
        assert receipt["ledger"]["blinded_luna_calls"] == 192
        work = receipt["worklist"]
        assert len(work["renders"]) == 32
        assert len(work["warmups"]) == 2
        assert len(work["transforms"]) == 56
        assert len(work["metric_jobs"]) == 88
        jobs = {row["job_id"]: row for row in work["renders"]}
        for pair in work["comparisons"]:
            if pair["kind"] in {"same_model_new_run", "independent_seed"}:
                a, b = jobs[pair["reference_job_id"]], jobs[pair["candidate_job_id"]]
                assert a["job_id"] != b["job_id"]
                assert (a["seed"] == b["seed"]) == (pair["kind"] == "same_model_new_run")
        assert len({row["call_id"] for row in work["blind_calls"]}) == 192
        if label == "unknown":
            assert "storage_usd" in receipt["missing_costs"]
            assert "all_in_estimate_usd" not in receipt
        before = output.read_bytes()
        retry = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        assert retry.returncode != 0
        assert output.read_bytes() == before


def test_retry_preflight_carries_closed_attempts_and_rejects_invalid_receipts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    prior = {
        "schema": "forge/evaluator-pilot-campaign-result/v1",
        "all_owned_pods_absent": True,
        "compute_and_running_disk_estimate_usd": 0.45,
        "attempts": [
            {
                "pod_id": "closed-pod",
                "closed": True,
                "absence_verified": True,
                "compute_estimate_usd": 0.45,
            }
        ],
    }
    costs = json.loads((root / "configs/evaluator-pilot-costs.json").read_text())
    guard = json.loads((root / "configs/evaluator-pilot-guard-envelope.json").read_text())
    for label, cap, invalid in (
        ("authorized", 5, None),
        ("proposal", 5.5, None),
        ("duplicate", 5.5, "duplicate"),
        ("unclosed", 5.5, "unclosed"),
        ("tampered-total", 5.5, "total"),
    ):
        receipt = json.loads(json.dumps(prior))
        if invalid == "duplicate":
            receipt["attempts"] *= 2
        elif invalid == "unclosed":
            receipt["attempts"][0]["absence_verified"] = False
        elif invalid == "total":
            receipt["compute_and_running_disk_estimate_usd"] = 0
        prior_path, cost_path, guard_path, out = [
            tmp_path / f"{label}-{s}.json" for s in ("prior", "cost", "guard", "out")
        ]
        prior_path.write_text(json.dumps(receipt))
        cost_path.write_text(json.dumps({**costs, "cap_usd": cap}))
        guard_path.write_text(json.dumps({**guard, "cap_usd": cap}))
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/evaluator_pilot_preflight.py"),
                "--pilot",
                str(root / "configs/evaluator-pilot-manifest-v1.json"),
                "--costs",
                str(cost_path),
                "--guard-envelope",
                str(guard_path),
                "--prior-campaign-result",
                str(prior_path),
                "--out",
                str(out),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if invalid:
            assert result.returncode != 0
            assert not out.exists()
            continue
        assert result.returncode == 2, result.stderr
        report = json.loads(out.read_text())
        assert report["dispatch_authorized"] is False
        assert report["prior_compute_usd"] == 0.45
        assert report["campaign_estimate_usd"] == report["all_in_estimate_usd"] + 0.45
        assert report["guard_config"]["contingency_usd"] == 0.95
        assert report["guard_envelope"]["warning_usd"] == 0.8 * cap
        assert report["status"] == ("OVER_CAP" if cap == 5 else "INCOMPLETE")
        assert (report["guard_work_margin_seconds"] > 0) == (cap == 5.5)
