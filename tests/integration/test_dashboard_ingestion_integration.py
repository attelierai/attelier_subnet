"""Validator report ingestion crosses HTTP, SQLite and real Sr25519 signatures."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

bt = pytest.importorskip("bittensor")

from forge.chain import Sr25519Signer  # noqa: E402
from forge.evaluation.distributed import signed_partial_report  # noqa: E402
from forge.evaluation.distributed_plan import DistributedEvaluationPlan  # noqa: E402
from forge.evaluation.mvp_policy import QUALITY_AXES  # noqa: E402
from forge.validator.dashboard_ingestion import (  # noqa: E402
    DashboardIngestionHandler,
    DashboardReportStore,
    _Server,
    load_registered_plans,
)
from forge.validator.dashboard_ingestion_cli import push  # noqa: E402


def _plan(wallet, peer) -> DistributedEvaluationPlan:
    return DistributedEvaluationPlan(
        round_id="ingestion-fixture-round",
        stage="diagnostic",
        entropy="fixture-entropy",
        policy_id="diagnostic-only",
        candidate_manifest_hash="candidate",
        king_manifest_hash="wan50",
        wan50_manifest_hash="wan50",
        committee={wallet.hotkey.ss58_address: 1, peer.hotkey.ss58_address: 1},
        prompts=(
            {
                "prompt_id": "p0",
                "text": "fixture prompt must never be public",
                "seed": 7,
                "subject_class": "object",
                "camera_type": "moving",
                "style_family": "natural",
            },
        ),
        timing_prompt_ids=(),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
        audit_work_count=1,
    )


def _quality() -> dict:
    return {
        "prompt_id": "p0",
        "seed": 7,
        "subject_class": "object",
        "camera_type": "moving",
        "style_family": "natural",
        "orders": [
            {"reference_first": order, "axes": dict.fromkeys(QUALITY_AXES, 0.0)}
            for order in (True, False)
        ],
    }


def test_signed_report_http_ingestion_is_private_immutable_and_restart_safe(tmp_path):
    wallet = bt.Wallet(name="ingest", hotkey="validator", path=str(tmp_path / "wallets"))
    wallet.create_new_hotkey(suppress=True, use_password=False, overwrite=False)
    peer = bt.Wallet(name="ingest-peer", hotkey="validator", path=str(tmp_path / "wallets"))
    peer.create_new_hotkey(suppress=True, use_password=False, overwrite=False)
    plan = _plan(wallet, peer)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()))
    other_plan = replace(plan, candidate_manifest_hash="other-candidate")
    other_path = tmp_path / "other-plan.json"
    other_path.write_text(json.dumps(other_plan.to_dict()))
    assert len(load_registered_plans([plan_path, other_path])) == 2
    report = signed_partial_report(
        plan, Sr25519Signer(wallet), quality_vs_king=[_quality()], quality_vs_wan50=[_quality()]
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    database = tmp_path / "dashboard.sqlite"
    registry = tmp_path / "registered"
    registry.mkdir()
    server = _Server(
        ("127.0.0.1", 0),
        DashboardIngestionHandler,
        DashboardReportStore(database, load_registered_plans([plan_path]), registry),
        None,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        from forge.validator.distributed_mvp_runtime_cli import deliver_completed_assignment

        assignment = tmp_path / "completed-assignment"
        assignment.mkdir()
        (assignment / "partial-report.json").write_bytes(report_path.read_bytes())
        result = deliver_completed_assignment(plan, assignment, endpoint, tmp_path / "outbox.json")
        assert json.loads((assignment / "dashboard-plan.json").read_text()) == plan.to_dict()

        assert result["status"] == "stored"
        duplicate = push(plan_path, report_path, endpoint, tmp_path / "outbox.json")
        assert duplicate["status"] == "duplicate"
        with urlopen(endpoint + "/api/v1/validator-reports", timeout=5) as response:
            public = json.loads(response.read())
        assert public["reports"][0]["signature_verified"]
        assert not public["reports"][0]["chain_verified"]
        assert "fixture prompt" not in json.dumps(public)
        assert 'signature":"' not in json.dumps(public)

        conflict = signed_partial_report(plan, Sr25519Signer(wallet))
        request = Request(
            endpoint + "/api/v1/validator-reports",
            data=json.dumps(conflict).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 409
        tampered = dict(report)
        tampered["plan_hash"] = "0" * 64
        request.data = json.dumps(tampered).encode()
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 400
        tampered = {**report, "signature": "00" * 64}
        request.data = json.dumps(tampered).encode()
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 400
        request.data = b'{"plan_hash": NaN}'
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 400
        request.add_header("Content-Length", str(4 * 1024 * 1024 + 1))
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 400
        # A new frozen round is installed by the operator while HTTP stays up.
        # Clients cannot register it themselves, and later retirement preserves history.
        later = replace(plan, round_id="later-round", candidate_manifest_hash="later")
        later_path = registry / "later.json"
        later_report = signed_partial_report(
            later,
            Sr25519Signer(wallet),
            quality_vs_king=[_quality()],
            quality_vs_wan50=[_quality()],
        )
        request = Request(
            endpoint + "/api/v1/validator-reports",
            data=json.dumps(later_report).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 400
        staging = registry / "later.tmp"
        staging.write_text(json.dumps(later.to_dict()))
        staging.replace(later_path)
        with urlopen(request, timeout=5) as response:
            assert json.load(response)["status"] == "stored"
        later_path.unlink()
        fixture = tmp_path / "private-snapshot.json"
        fixture.write_text(json.dumps({"schema": "forge/dashboard/v2", "_fixture": {}}))
        server.snapshot = fixture
        with pytest.raises(HTTPError) as exc:
            urlopen(endpoint + "/data/status.json", timeout=5)
        assert exc.value.code == 503
        assert str(tmp_path).encode() not in exc.value.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    pending = deliver_completed_assignment(plan, assignment, endpoint, tmp_path / "outbox.json")
    assert pending["status"] == "PENDING"
    assert json.loads((assignment / "partial-report.json").read_text()) == report

    restarted = DashboardReportStore(database, load_registered_plans([plan_path]))
    assert len(restarted.public_snapshot()["reports"]) == 2

    # A real CLI validator still produces its signed report when telemetry is down.
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {"quality_vs_king": [_quality()], "quality_vs_wan50": [_quality()], "timing": []}
        )
    )
    pending_report, outbox = tmp_path / "peer-report.json", tmp_path / "peer-outbox.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.cli",
            "distributed-report",
            "--plan",
            str(plan_path),
            "--partial",
            str(partial),
            "--out",
            str(pending_report),
            "--wallet",
            "ingest-peer",
            "--hotkey",
            "validator",
            "--wallet-path",
            str(tmp_path / "wallets"),
            "--dashboard-endpoint",
            endpoint,
            "--dashboard-outbox",
            str(outbox),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["dashboard_delivery"]["status"] == "PENDING"
    assert json.loads(outbox.read_text()) == json.loads(pending_report.read_text())
    server = _Server(("127.0.0.1", 0), DashboardIngestionHandler, restarted, None)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        # Timer entrypoint retries the existing signed payload without a wallet.
        retry_plans, retry_outbox = tmp_path / "retry-plans", tmp_path / "retry-outbox"
        retry_plans.mkdir()
        retry_outbox.mkdir()
        (retry_plans / "plan.json").write_bytes(plan_path.read_bytes())
        (retry_outbox / "report.json").write_bytes(outbox.read_bytes())
        retried = subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.cli",
                "dashboard-flush",
                "--plan-directory",
                str(retry_plans),
                "--outbox-directory",
                str(retry_outbox),
                "--endpoint",
                endpoint,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert retried.returncode == 0, retried.stderr
        assert json.loads(retried.stdout)["delivered"][0]["status"] == "stored"
        assert (retry_outbox / "report.json").read_bytes() == outbox.read_bytes()
        assert push(plan_path, pending_report, endpoint, outbox)["status"] == "duplicate"
        assert push(plan_path, pending_report, endpoint, outbox)["status"] == "duplicate"
        with urlopen(endpoint + "/api/v1/validator-reports", timeout=5) as response:
            assert len(json.load(response)["reports"]) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
