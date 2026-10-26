"""Durable Launch-service state projection for the public status endpoint."""

import json
import subprocess
import sys
from dataclasses import replace

from test_launch_reference_integration import _manifests, _screening

from forge.evaluation.launch_baseline import ParentLaunchBaselineStore
from forge.evaluation.launch_reference import qualified_bootstrap_reference
from forge.protocol.contract import ChallengeContract
from forge.validator.launch_config import LaunchConfig, closed_selection
from forge.validator.launch_service import LaunchServiceStore
from forge.validator.launch_service_dashboard import export_launch_operations


def test_export_cli_accepts_replayable_bootstrap_receipt_without_a_certified_snapshot(tmp_path):
    contract = ChallengeContract()
    wan50, imported = _manifests(contract)
    baseline = ParentLaunchBaselineStore(tmp_path / "baseline.sqlite")
    baseline.select_once(
        [_screening(imported.fingerprint(), wan50.fingerprint(), eligible=False)],
        anchor_hash=wan50.fingerprint(),
    )
    receipt = qualified_bootstrap_reference(
        baseline, wan50_manifest=wan50, imported_manifest=imported, contract=contract
    )
    config = replace(_config(), contract_hash=contract.contract_hash())
    LaunchServiceStore(tmp_path / "service.sqlite", config)
    for name, value in (("config", config.to_dict()), ("contract", contract.to_dict()),
                        ("bootstrap", receipt)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    destination = tmp_path / "status.json"
    argv = [sys.executable, "-m", "forge.cli", "launch", "export", "--config",
            str(tmp_path / "config.json"), "--contract", str(tmp_path / "contract.json"),
            "--state", str(tmp_path / "service.sqlite"), "--bootstrap-reference",
            str(tmp_path / "bootstrap.json"), "--destination", str(destination)]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(destination.read_text())
    assert report["bootstrap_reference"]["fingerprint"] == wan50.fingerprint()
    assert report["king"] is None and report["evaluation"]["status"] == "NO_EVALUATION"
    assert report["cost"]["observed_usd"] is None
    assert "wan50_manifest" not in destination.read_text()
    # A substituted receipt cannot replace the previously exported verified projection.
    receipt["bootstrap_reference_hash"] = imported.fingerprint()
    (tmp_path / "bootstrap.json").write_text(json.dumps(receipt))
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode != 0 and json.loads(destination.read_text()) == report


def _hash(number: int) -> str:
    return f"{number:064x}"


def _config() -> LaunchConfig:
    return LaunchConfig(
        network="test",
        netuid=578,
        genesis_hash=_hash(1),
        contract_hash=_hash(2),
        campaign_id="dashboard-fixture",
    )


def _selection(config: LaunchConfig) -> dict:
    return closed_selection(
        config,
        close_block=300,
        close_hash=_hash(3),
        finalized_block=300,
        parent_state_hash=_hash(4),
        candidate_manifest_hash=_hash(5),
        candidate_hotkey="miner",
        commitment_block=300,
    )


def test_launch_operations_projects_bootstrap_pending_and_expired_durable_state_atomically(
    tmp_path,
):
    config = _config()
    store = LaunchServiceStore(tmp_path / "service.sqlite", config)
    destination = tmp_path / "runtime-data" / "status.json"
    bootstrap = {"fingerprint": _hash(6), "label": "Wan bootstrap", "manifest_hash": _hash(7)}

    initial = export_launch_operations(destination, store, bootstrap_reference=bootstrap)
    assert initial["king"] is None
    assert initial["evaluation"]["status"] == "NO_EVALUATION"
    assert initial["certificate"] == {"status": "NO_CERTIFICATE", "certificate_hash": None}
    assert initial["publication"]["submission_status"] == "NOT_SUBMITTED"
    assert json.loads(destination.read_text()) == initial
    assert not list(destination.parent.glob(".status.json.*"))

    selection = _selection(config)
    store.select(selection, finalized_block=300)
    pending = export_launch_operations(
        destination,
        store,
        bootstrap_reference=bootstrap,
        capacity={"status": "MEASURED", "seconds": 120.0},
        worker_health={"status": "IDLE"},
    )
    assert pending["evaluation"] == {
        "status": "PENDING",
        "round_id": selection["round_id"],
        "reason": None,
    }
    assert pending["queue"] == {
        "active": [{"round_id": selection["round_id"], "status": "WAITING_ENTROPY"}],
        "active_count": 1,
    }

    # The durable round status is the only source for this terminal state; no
    # certificate or publication is invented while a round expires.
    with store._db() as db:  # noqa: SLF001 - real SQLite state transition fixture
        db.execute(
            "UPDATE launch_service_rounds SET status='EXPIRED' WHERE round_id=?",
            (selection["round_id"],),
        )
    expired = export_launch_operations(
        destination,
        store,
        bootstrap_reference=bootstrap,
        consensus_status={"status": "EXPIRED_NO_CERTIFICATE", "finalized_block": 399},
    )
    assert expired["queue"] == {"active": [], "active_count": 0}
    assert expired["evaluation"] == {
        "status": "INCOMPLETE",
        "round_id": selection["round_id"],
        "reason": "round expired before complete evaluation",
    }
    assert expired["certificate"] == {
        "status": "EXPIRED_NO_CERTIFICATE",
        "certificate_hash": None,
        "finalized_block": 399,
    }
    assert expired["king"] is None
