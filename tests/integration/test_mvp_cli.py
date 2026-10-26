"""CLI/filesystem integration with synthetic text; no GPU/model claims."""

import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "forge.cli", *map(str, args)], capture_output=True, text=True
    )


def test_cli_separates_stage_packages_and_preserves_frozen_files(tmp_path):
    policy = tmp_path / "policy.json"
    assert run_cli("mvp-policy", "--out", policy).returncode == 0
    assert run_cli("mvp-policy", "--out", policy).returncode != 0

    def prompts(prefix, per_cell):
        return [
            {
                "prompt_id": f"{prefix}-{s}-{c}-{f}-{i}",
                "source_id": f"{prefix}-source-{s}-{c}-{f}-{i}",
                "concept_id": f"{prefix}-concept-{s}-{c}-{f}-{i}",
                "text": f"{prefix} synthetic case {s} {c} {f} {i}",
                "seed": 123,
                "subject_class": s,
                "camera_type": c,
                "style_family": f,
            }
            for s in ("human", "animal", "object", "landscape")
            for c in ("static", "moving")
            for f in ("realistic", "stylized")
            for i in range(per_cell)
        ]

    screen, confirmation = tmp_path / "screen.json", tmp_path / "confirmation.json"
    screen.write_text(json.dumps(prompts("screen", 1)))
    confirmation.write_text(json.dumps(prompts("sealed", 8)))
    args = [
        "mvp-freeze",
        "--policy",
        policy,
        "--screening",
        screen,
        "--confirmation",
        confirmation,
        "--candidate-manifest-hash",
        "a" * 64,
        "--incumbent-manifest-hash",
        "b" * 64,
        "--wan50-manifest-hash",
        "b" * 64,
        "--out-dir",
        tmp_path / "packages",
    ]
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    package_dir = tmp_path / "packages"
    public = json.loads((package_dir / "commitment.json").read_text())
    assert public["confirmation_prompts"] == 128
    assert "sealed synthetic" not in json.dumps(public)
    assert "sealed synthetic" not in (package_dir / "screening.private.json").read_text()
    assert (package_dir / "confirmation.private.json").stat().st_mode & 0o777 == 0o600
    assert run_cli(*args).returncode != 0
    before = set(package_dir.iterdir())
    leaked = prompts("screen", 8)
    confirmation.write_text(json.dumps(leaked))
    args[-1] = tmp_path / "rejected"
    assert run_cli(*args).returncode != 0
    assert not Path(args[-1]).exists()
    assert set(package_dir.iterdir()) == before


def test_cli_exports_real_local_ledger_without_future_prompts(tmp_path):
    from forge.validator import AdmissionLedger, BudgetLedger
    from tests.integration.test_validator_admission_integration import commitment, history, verified

    database = tmp_path / "queue.sqlite"
    ledger = AdmissionLedger(database, network="test", netuid=9)
    admitted = ledger.admit(commitment("miner", 1), history=history(), verify_history=verified)
    budget_db = tmp_path / "budget.sqlite"
    budget = BudgetLedger(budget_db)
    budget.reserve(admitted.id, "planned-local-operation", 0.5)
    output = tmp_path / "snapshot.json"
    result = run_cli(
        "mvp-dashboard",
        "--db",
        database,
        "--network",
        "test",
        "--netuid",
        9,
        "--candidate-id",
        admitted.id,
        "--budget-db",
        budget_db,
        "--out",
        output,
    )
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(output.read_text())
    assert snapshot["evidence_kind"] == "local_integration"
    assert len(snapshot["queue"]) == 1 and snapshot["queue"][0]["status"] == "queued"
    assert snapshot["budget"] == {"limit_usd": 10, "spent_usd": 0.0, "reserved_usd": 0.5}
    assert snapshot["winner"] is None and not snapshot["activation"]["enabled"]
    assert "panel_hash" not in output.read_text() and "seed" not in output.read_text()
    assert (
        run_cli(
            "mvp-dashboard",
            "--db",
            tmp_path / "missing",
            "--netuid",
            9,
            "--out",
            tmp_path / "unknown.json",
        ).returncode
        != 0
    )
