"""Real CLI, wallet signatures, peer HTTP files and persisted aggregation.

Numerical fixture observations test transport only; they are never Wan evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

bt = pytest.importorskip("bittensor")

from forge.chain import Sr25519Verifier  # noqa: E402
from forge.evaluation.distributed_plan import (  # noqa: E402
    DistributedEvaluationPlan,
    assignments_for,
)
from forge.evaluation.mvp_policy import QUALITY_AXES  # noqa: E402
from forge.evaluation.synthetic_prompts import synthetic_panel  # noqa: E402
from forge.protocol.distributed_batch import DiagnosticBatch, verify_batch_index  # noqa: E402
from forge.protocol.distributed_transport import fetch_peer_report, report_digest  # noqa: E402


def test_two_cli_validators_exchange_signed_reports_without_a_shared_server(tmp_path):
    wallets = []
    for name in ("peer-a", "peer-b"):
        wallet = bt.Wallet(name=name, hotkey="worker", path=str(tmp_path / "wallets"))
        wallet.create_new_hotkey(suppress=True, use_password=False, overwrite=False)
        wallets.append(wallet)
    plan = DistributedEvaluationPlan(
        round_id="fixture-no-wan-claims",
        stage="diagnostic",
        entropy="fixture-finalized-entropy",
        policy_id="diagnostic-only",
        candidate_manifest_hash="fixture-candidate",
        king_manifest_hash="fixture-king",
        wan50_manifest_hash="fixture-king",
        committee={
            wallet.hotkey.ss58_address: power
            for wallet, power in zip(wallets, (51, 49), strict=True)
        },
        prompts=tuple(
            synthetic_panel("fixture", round_id="fixture", phase="diagnostic", per_cell=1)[:6]
        ),
        timing_prompt_ids=(),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
        audit_work_count=2,
        chain_context={"network": "test", "netuid": 9, "close_block": 10, "entropy_block": 12},
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()))
    other_plan_path = tmp_path / "other-plan.json"
    other_plan_path.write_text(json.dumps(replace(plan, candidate_manifest_hash="other").to_dict()))

    def cli(*args, succeeds=True):
        completed = subprocess.run(
            [sys.executable, "-m", "forge.cli", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (completed.returncode == 0) == succeeds, completed.stderr

    cli("distributed-plan", "--plan", plan_path, "--out", tmp_path / "assignment.json")
    batch_path = tmp_path / "batch.json"
    cli(
        "distributed-batch-plan",
        "--plan",
        plan_path,
        "--plan",
        other_plan_path,
        "--commit-deadline-block",
        30,
        "--reveal-deadline-block",
        40,
        "--out",
        batch_path,
    )
    batch = DiagnosticBatch.from_dict(json.loads(batch_path.read_text()))
    servers, peers, received = [], [], []
    try:
        for wallet in wallets:
            peer = tmp_path / wallet.name
            peer.mkdir()
            rows = []
            prompts = {row["prompt_id"]: row for row in plan.prompts}
            assigned = assignments_for(plan, wallet.hotkey.ss58_address)
            assert len(assigned) == 4
            for item in assigned:
                prompt = prompts[item.prompt_id]
                rows.append(
                    {
                        **{
                            key: prompt[key]
                            for key in (
                                "prompt_id",
                                "seed",
                                "subject_class",
                                "camera_type",
                                "style_family",
                            )
                        },
                        "evidence_kind": "synthetic_transport_fixture",
                        "orders": [
                            {"reference_first": order, "axes": dict.fromkeys(QUALITY_AXES, 0.0)}
                            for order in (True, False)
                        ],
                    }
                )
            evidence = {"quality_vs_king": rows, "quality_vs_wan50": rows, "timing": []}
            source = peer / "partial.json"
            source.write_text(json.dumps(evidence))
            target = peer / "report.json"
            cli(
                "distributed-report",
                "--plan",
                plan_path,
                "--partial",
                source,
                "--wallet",
                wallet.name,
                "--hotkey",
                "worker",
                "--wallet-path",
                tmp_path / "wallets",
                "--out",
                target,
            )
            report = json.loads(target.read_text())
            other_report_path = peer / "other-report.json"
            cli(
                "distributed-report",
                "--plan",
                other_plan_path,
                "--partial",
                source,
                "--wallet",
                wallet.name,
                "--hotkey",
                "worker",
                "--wallet-path",
                tmp_path / "wallets",
                "--out",
                other_report_path,
            )
            index_path = peer / "batch-index.json"
            cli(
                "distributed-batch-index",
                "--batch",
                batch_path,
                "--report",
                target,
                "--wallet",
                wallet.name,
                "--hotkey",
                "worker",
                "--wallet-path",
                tmp_path / "wallets",
                "--out",
                index_path,
                succeeds=False,
            )
            cli(
                "distributed-batch-index",
                "--batch",
                batch_path,
                "--report",
                target,
                "--report",
                other_report_path,
                "--wallet",
                wallet.name,
                "--hotkey",
                "worker",
                "--wallet-path",
                tmp_path / "wallets",
                "--out",
                index_path,
            )
            index = json.loads(index_path.read_text())
            verify_batch_index(batch, index, wallet.hotkey.ss58_address, Sr25519Verifier())
            assert set(index["reports"]) == {p.plan_hash() for p in batch.plans}
            digest = report_digest(report)
            public = peer / "public"
            public.mkdir()
            (public / f"{digest}.json").write_text(target.read_text())
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(public))
            )
            Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
            url = f"http://127.0.0.1:{server.server_port}/{digest}.json"
            other_view = fetch_peer_report(url, digest)
            assert other_view == report
            with pytest.raises(ValueError, match="content address"):
                fetch_peer_report(url, "0" * 64)
            downloaded = tmp_path / f"received-{wallet.name}.json"
            downloaded.write_text(json.dumps(other_view))
            received.append(downloaded)
            peers.append(peer)
        results = []
        for peer in peers:
            cli(
                "distributed-aggregate",
                "--plan",
                plan_path,
                "--report",
                received[0],
                "--report",
                received[1],
                "--out",
                peer / "global.json",
            )
            results.append(json.loads((peer / "global.json").read_text()))
        assert results[0] == results[1]
        assert results[0]["status"] == "DIAGNOSTIC"
        assert not results[0]["reward_eligible"]
        assert len(results[0]["quality_vs_king"]) == 6
        assert sum(len(row["contributions"]) for row in results[0]["quality_vs_king"]) == 8
        tampered = json.loads(received[1].read_text())
        tampered["quality_vs_king"][0]["orders"][0]["axes"]["image_quality"] = 1.0
        received[1].write_text(json.dumps(tampered))
        cli(
            "distributed-aggregate",
            "--plan",
            plan_path,
            "--report",
            received[0],
            "--report",
            received[1],
            "--out",
            tmp_path / "tampered.json",
        )
        result = json.loads((tmp_path / "tampered.json").read_text())
        assert result["status"] == "INCOMPLETE" and result["invalid"]
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_real_wallet_cli_aggregates_stage_bound_screening_ab(tmp_path):
    wallets = []
    for name in ("screen-a", "screen-b"):
        wallet = bt.Wallet(name=name, hotkey="worker", path=str(tmp_path / "wallets"))
        wallet.create_new_hotkey(suppress=True, use_password=False, overwrite=False)
        wallets.append(wallet)
    prompts = tuple(synthetic_panel("screening-ab", round_id="r", phase="screening", per_cell=1))
    plan = DistributedEvaluationPlan(
        round_id="screening-ab-r1",
        stage="screening",
        entropy="screening-ab-entropy",
        policy_id="screening-policy",
        candidate_manifest_hash="candidate",
        king_manifest_hash="king",
        wan50_manifest_hash="king",
        committee={wallet.hotkey.ss58_address: 1 for wallet in wallets},
        prompts=prompts,
        timing_prompt_ids=tuple(row["prompt_id"] for row in prompts),
        quality_tolerance=0.5,
        timing_log_ratio_tolerance=0.5,
        audit_calibration_hash="audit-calibration",
    )
    plan_path = tmp_path / "screening-plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()))
    reports = []
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    for wallet in wallets:
        quality, timing = [], []
        for item in assignments_for(plan, wallet.hotkey.ss58_address):
            prompt = prompt_by_id[item.prompt_id]
            if item.kind == "quality":
                quality.append(
                    {
                        **{
                            key: prompt[key]
                            for key in (
                                "prompt_id",
                                "seed",
                                "subject_class",
                                "camera_type",
                                "style_family",
                            )
                        },
                        "orders": [
                            {"reference_first": value, "axes": dict.fromkeys(QUALITY_AXES, 0.0)}
                            for value in (True, False)
                        ],
                    }
                )
            else:
                timing.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "seed": prompt["seed"],
                        "timing_schema": "forge/screen-timing/v1",
                        "runs": [
                            {"model": "reference", "seconds": 2.0},
                            {"model": "candidate", "seconds": 1.0},
                        ],
                    }
                )
        partial = tmp_path / f"{wallet.name}-partial.json"
        partial.write_text(
            json.dumps({"quality_vs_king": quality, "quality_vs_wan50": quality, "timing": timing})
        )
        report = tmp_path / f"{wallet.name}-report.json"
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
                "--wallet",
                wallet.name,
                "--hotkey",
                "worker",
                "--wallet-path",
                str(tmp_path / "wallets"),
                "--out",
                str(report),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        reports.append(report)
    output = tmp_path / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.cli",
            "distributed-aggregate",
            "--plan",
            str(plan_path),
            "--report",
            str(reports[0]),
            "--report",
            str(reports[1]),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    aggregate = json.loads(output.read_text())
    assert aggregate["status"] == "COMPLETE"
    assert all(len(row["runs"]) == 2 for row in aggregate["timing"])
    assert all(row["timing_schema"] == "forge/screen-timing/v1" for row in aggregate["timing"])
