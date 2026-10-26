"""Runtime durability at real filesystem/SQLite and subprocess boundaries."""

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.evaluation.launch_baseline import ParentLaunchBaselineStore
from forge.evaluation.launch_reference import qualified_bootstrap_reference
from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.protocol.store import Store
from forge.validator.admission import build_authenticated_history
from forge.validator.launch_bridge import LaunchExecutionReplayer, LocalLaunchReceiptStore
from forge.validator.launch_config import LaunchConfig, closed_selection
from forge.validator.launch_consensus import MINIMUM_COMMIT_WINDOW_BLOCKS, LaunchConsensusRunner
from forge.validator.launch_round import freeze_closed_round
from forge.validator.launch_runtime import tick
from forge.validator.launch_service import LaunchServiceStore

_fixture_spec = importlib.util.spec_from_file_location(
    "launch_bridge_fixture", Path(__file__).with_name("test_launch_bridge_integration.py")
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
_external_gpu_contract_package = _fixture._external_gpu_contract_package
sha = _fixture.sha
signers = _fixture.signers
policy = _fixture.policy
_raw_evidence_v2 = _fixture._raw_evidence_v2


def _baseline_screening(candidate: str, anchor: str) -> dict:
    return {
        "candidate_hash": candidate,
        "anchor_hash": anchor,
        "rows": [
            {
                "prompt_id": f"screen-{index}",
                "status": "OK",
                "vetoes": [],
                "reference_seconds": 2.0,
                "candidate_seconds": 1.0,
                "reference_scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                "candidate_scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
            }
            for index in range(6)
        ],
    }


def _baseline_manifests(contract: ChallengeContract):
    from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest

    reference = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    wan50 = ModelManifest(
        reference,
        "pipeline",
        (ManifestFile("model_index.json", sha(10), 1),),
        "apache-2.0",
        {"steps": 50, "guidance_scale": 5.0},
    )
    imported = ModelManifest(
        reference,
        "pipeline",
        (ManifestFile("model_index.json", sha(10), 1),),
        "apache-2.0",
        {"steps": 40, "guidance_scale": 5.0, "scheduler": "uniform_euler"},
    )
    return wan50, imported


def _run_launch_initialize(
    *, root: Path, config: Path, contract: Path, bootstrap: Path, package: Path, activation: int
):
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.cli",
            "launch",
            "initialize",
            "--config",
            str(config),
            "--contract",
            str(contract),
            "--root",
            str(root),
            "--bootstrap-reference",
            str(bootstrap),
            "--qualification-package",
            str(package),
            "--activation-block",
            str(activation),
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


class _Chain:
    network, netuid = "test", 578

    def __init__(self, height: int):
        self.height, self.rows = height, {}
        self.advance_after_commitment_read_to = None
        self.publish_calls = 0

    def finalized_height(self):
        return self.height

    def finalized_block_hash(self, height):
        return sha(height)

    def uid_snapshot(self, height):
        return {0: "miner"}

    def chain_time(self, height):
        return height

    def publish(self, wallet, text):
        from forge.chain import decode_commitment

        self.publish_calls += 1
        hotkey = wallet.hotkey.ss58_address
        self.rows[hotkey] = decode_commitment(hotkey, self.height, text)
        return f"{self.height}-fixture"

    def commitment(self, hotkey, height):
        if self.advance_after_commitment_read_to is not None:
            self.height = self.advance_after_commitment_read_to
            self.advance_after_commitment_read_to = None
        return self.rows.get(hotkey)


def test_cli_offline_initializer_replays_imported_bootstrap_and_preserves_canonical_ledger(
    tmp_path,
):
    """The operator process can resume exact genesis but cannot replace its evidence."""
    contract = ChallengeContract()
    wan50, imported = _baseline_manifests(contract)
    base, _, hardware, runtime_doc = policy()
    members = signers()
    committee = {member.validator_hotkey: 1 for member in members}
    parent = FinalizedCompetitionState.genesis("initializer", wan50.fingerprint())
    frozen, package = _external_gpu_contract_package(
        tmp_path / "qualified",
        candidate=imported.fingerprint(),
        king=wan50.fingerprint(),
        anchor=wan50.fingerprint(),
        runtime={
            key: base["runtime"][key]
            for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
        },
        entropy=sha(800),
        round_id="initializer-confirmation",
    )
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=100,
        certified_state_hash=parent.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    confirmation_plan = replace(
        _fixture.round_plan(
            parent, committee, frozen, history.history_root, "initializer-confirmation", sha(800)
        ),
        contract_hash=contract.contract_hash(),
    )
    receipts = LocalLaunchReceiptStore(tmp_path / "trusted-receipts")
    bundle = _raw_evidence_v2(
        tmp_path / "trusted-receipts",
        frozen,
        {"candidate": imported, "king": wan50, "anchor": wan50},
        hardware,
        runtime_doc,
        confirmation_plan,
        members[0].validator_hotkey,
        package,
    )
    replay = LaunchExecutionReplayer(receipts).replay(
        bundle, confirmation_plan, validator_hotkey=members[0].validator_hotkey
    )
    assert replay["status"] == "PASS"
    baseline = ParentLaunchBaselineStore(tmp_path / "baseline.sqlite")
    baseline.select_once(
        [_baseline_screening(imported.fingerprint(), wan50.fingerprint())],
        anchor_hash=wan50.fingerprint(),
    )
    assert baseline.confirm_once(frozen, replay["rows"], package)["status"] == "IMPORTED_BASELINE"
    bootstrap = qualified_bootstrap_reference(
        baseline, wan50_manifest=wan50, imported_manifest=imported, contract=contract
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bootstrap_path, package_path = inputs / "bootstrap.json", inputs / "qualification.json"
    bootstrap_path.write_text(json.dumps(bootstrap))
    package_path.write_text(json.dumps(package))
    config = LaunchConfig(
        network="test",
        netuid=578,
        genesis_hash=sha(1),
        contract_hash=contract.contract_hash(),
        campaign_id="initializer-fixture",
    )
    config_path, contract_path = inputs / "config.json", inputs / "contract.json"
    config_path.write_text(json.dumps(config.to_dict()))
    contract_path.write_text(json.dumps(contract.to_dict()))
    root = tmp_path / "operator"

    first = _run_launch_initialize(
        root=root, config=config_path, contract=contract_path, bootstrap=bootstrap_path,
        package=package_path, activation=123,
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["authority"] == "NO_RENDER_NO_SIGNATURE_NO_PUBLICATION"
    state = FinalizedCompetitionState.from_dict(json.loads((root / "state.json").read_text()))
    assert state.current_king_fingerprint == imported.fingerprint()
    assert json.loads((root / "manifests" / "anchor.json").read_text()) == {
        "fingerprint": wan50.fingerprint()
    }
    ledger = (root / "budget.sqlite").read_bytes()
    exact_bootstrap = (root / "bootstrap-reference.json").read_bytes()

    resumed = _run_launch_initialize(
        root=root, config=config_path, contract=contract_path, bootstrap=bootstrap_path,
        package=package_path, activation=123,
    )
    assert resumed.returncode == 0, resumed.stderr
    conflict = _run_launch_initialize(
        root=root, config=config_path, contract=contract_path, bootstrap=bootstrap_path,
        package=package_path, activation=124,
    )
    assert conflict.returncode != 0
    assert (root / "budget.sqlite").read_bytes() == ledger
    assert (root / "bootstrap-reference.json").read_bytes() == exact_bootstrap
    (root / "round-0").mkdir()
    progressed = _run_launch_initialize(
        root=root, config=config_path, contract=contract_path, bootstrap=bootstrap_path,
        package=package_path, activation=123,
    )
    assert progressed.returncode != 0


def test_runtime_shadow_then_live_dispatches_once_and_expires_without_worker_result(tmp_path):
    base, manifests, hardware, runtime_doc = policy()
    candidate, king = manifests["candidate"].fingerprint(), manifests["king"].fingerprint()
    policy_fixture, package = _external_gpu_contract_package(
        tmp_path / "qualification",
        candidate=candidate,
        king=king,
        anchor=king,
        runtime={
            key: base["runtime"][key]
            for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
        },
        entropy=sha(310),
        round_id="runtime-round",
    )
    contract = ChallengeContract()
    config = LaunchConfig(
        network="test",
        netuid=578,
        genesis_hash=sha(1),
        contract_hash=contract.contract_hash(),
        campaign_id="runtime-fixture",
    )
    root = tmp_path / "operator"
    root.mkdir()
    dashboard = root / "runtime-data" / "operations.json"
    validators = signers()[:2]
    committee = {item.validator_hotkey: 1 for item in validators}
    wallets = {
        f"v{index}": SimpleNamespace(hotkey=item._keypair) for index, item in enumerate(validators)
    }
    parent = FinalizedCompetitionState.genesis("runtime", king)
    for label in wallets:
        with Store(root / f"canonical-validator-{label}.sqlite") as store:
            store.initialize_state(
                state_hash=parent.state_hash(),
                challenge_id=parent.challenge_id,
                state_version=parent.state_version,
                previous_state_hash=parent.previous_state_hash,
                payload=parent.to_dict(),
            )
    (root / "state.json").write_text(json.dumps(parent.to_dict()))
    (root / "manifests").mkdir()
    for name, fingerprint in (("king", king), ("anchor", king)):
        (root / "manifests" / f"{name}.json").write_text(json.dumps({"fingerprint": fingerprint}))
    worker_ready = {
        "evaluator": policy_fixture["evaluator"],
        "runtime": policy_fixture["runtime"],
        "capacity": {
            "measured_parallelism": 1,
            "worst_case_execution_seconds": 30,
            "commit_delivery_seconds": 12,
            "reveal_delivery_seconds": 1,
            "chain_block_seconds": 1,
        },
    }
    (root / "worker-ready.json").write_text(json.dumps(worker_ready))
    (root / "qualification-package.json").write_text(json.dumps(package))
    service = LaunchServiceStore(root / "budget.sqlite", config)
    selection = closed_selection(
        config,
        close_block=300,
        close_hash=sha(3),
        finalized_block=300,
        parent_state_hash=parent.state_hash(),
        candidate_manifest_hash=candidate,
        candidate_hotkey="miner",
        commitment_block=300,
    )
    service.select(selection, finalized_block=300)
    round_dir = root / "round-300"
    round_dir.mkdir()
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=300,
        certified_state_hash=parent.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
        window_id="launch-300",
        window_open_block=0,
        window_close_block=300,
    )
    (round_dir / "admission-history.json").write_text(
        json.dumps(
            {
                "network": history.network,
                "netuid": history.netuid,
                "through_block": history.through_block,
                "certified_state_hash": history.certified_state_hash,
                "history_root": history.history_root,
                "consumed_hotkeys": [],
                "registered_hotkeys": ["miner"],
                "lineage_edges": [],
                "window_id": "launch-300",
                "window_open_block": 0,
                "window_close_block": 300,
            }
        )
    )
    (round_dir / "committee.json").write_text(
        json.dumps({"schema": "forge/launch-committee/v1", "committee": committee})
    )
    freeze_closed_round(
        config=config,
        parent=parent,
        service=service,
        selection=selection,
        entropy_hash=sha(310),
        finalized_block=310,
        basic_manifests={"candidate": candidate, "king": king, "anchor": king},
        worker_ready=worker_ready,
        qualification_package=package,
        admission_history=history,
        committee=committee,
        round_dir=round_dir,
    )
    chain = _Chain(310)
    shadow = tick(root, config=config, contract=contract, chain=chain, shadow=True)
    assert shadow["status"] == "SHADOW_READY_NO_SIGN_NO_GPU"
    assert service.snapshot()["rounds"][0]["status"] == "READY"

    script = root / "worker.py"
    script.write_text(
        "from pathlib import Path; import sys; Path(sys.argv[-1] + '.marker').write_text('ran')"
    )
    prepared = tick(
        root, config=config, contract=contract, chain=chain, wallets=wallets,
        shadow=False, reserve_usd=1.0, export_destination=dashboard,
    )
    assert prepared["status"] == "REQUESTED_AWAITING_OPERATOR_WORKER"
    reserved = service.snapshot()["budget"]["reserved_or_captured_usd"]
    # Emulate a crash after the SQLite reservation commit but before its file
    # mirror and process intent reached disk. The reserved request is authority.
    (round_dir / "worker-request.json").unlink()
    live = tick(
        root,
        config=config,
        contract=contract,
        chain=chain,
        wallets=wallets,
        worker_command=[sys.executable, str(script)],
        shadow=False,
        reserve_usd=1.0,
    )
    assert live["status"] == "WORKER_DISPATCHED_UNKNOWN"
    assert service.snapshot()["budget"]["reserved_or_captured_usd"] == reserved
    for _ in range(40):
        if (round_dir / "worker-result.json.marker").exists():
            break
        time.sleep(0.05)
    assert (round_dir / "worker-result.json.marker").read_text() == "ran"
    restarted = tick(
        root,
        config=config,
        contract=contract,
        chain=chain,
        wallets=wallets,
        worker_command=[sys.executable, str(script)],
        shadow=False,
        reserve_usd=1.0,
    )
    assert restarted["status"] == "WORKER_DISPATCH_UNKNOWN_NO_RELAUNCH"
    assert service.snapshot()["rounds"][0]["status"] == "REQUESTED"

    # The operator-controlled custody step provides complete V2 bundles; the
    # worker marker itself is never treated as evidence.
    plan = __import__("forge.protocol.consensus_types", fromlist=["RoundPlan"]).RoundPlan.from_dict(
        json.loads((round_dir / "round-plan.json").read_text())
    )
    frozen_plan = json.loads((round_dir / "launch-plan.json").read_text())
    # The fake external GPU supplies its downloaded-tensor preflight through
    # the same durable admission boundary used by the authenticated worker.
    service.record_effective_admission(plan.round_id, {
        "schema": "forge/launch-effective-admission/v1",
        "plan_id": frozen_plan["plan_id"], "contract_hash": contract.contract_hash(),
        "manifest_hash": candidate, "effective_hash": sha(911),
        "legacy_effective_hash": None, "weight_content_hash": sha(912),
    })
    bundles = {
        signer.validator_hotkey: _raw_evidence_v2(
            round_dir / "trusted-receipts",
            frozen_plan,
            manifests,
            hardware,
            runtime_doc,
            plan,
            signer.validator_hotkey,
            package,
        )
        for signer in validators
    }
    (round_dir / "worker-result.json").write_text(
        json.dumps(
            {
                "request_hash": service.snapshot()["rounds"][0]["request_hash"],
                "trusted_bundle_hashes": bundles,
            }
        )
    )
    replayed = tick(
        root, config=config, contract=contract, chain=chain, wallets=wallets,
        shadow=False, export_destination=dashboard,
    )
    assert replayed["status"] == "COMMITTED_WAITING_REVEAL"
    assert json.loads(dashboard.read_text())["certificate"]["status"] == "PENDING"
    # A missing peer delivery still cannot be backdated after the frozen window.
    late_root = tmp_path / "late-peer-round"
    shutil.copytree(round_dir, late_root)
    with Store(late_root / "validator-v0.sqlite") as store:
        store.connection.execute(
            "DELETE FROM round_messages WHERE kind='commit' AND sender=?",
            (validators[1].validator_hotkey,),
        )
        store.connection.commit()
    chain.height = plan.commit_deadline_block
    chain.advance_after_commitment_read_to = plan.commit_deadline_block + 1
    with pytest.raises(ValueError, match="peer message delivery missed the frozen deadline"):
        LaunchConsensusRunner(late_root, chain=chain, wallets=wallets).tick()
    with Store(late_root / "validator-v0.sqlite") as store:
        assert store.connection.execute(
            "SELECT COUNT(*) FROM round_messages WHERE kind='commit'"
        ).fetchone()[0] == 1
    assert not list(late_root.glob("reveal-envelope-*.json"))

    # An already delivered commit may be replayed while finality crosses the
    # boundary. Reuse its durable receipt and advance without another publish.
    chain.height = plan.commit_deadline_block
    chain.advance_after_commitment_read_to = plan.commit_deadline_block + 1
    revealed = tick(
        root, config=config, contract=contract, chain=chain, wallets=wallets, shadow=False
    )
    assert revealed["status"] == "REVEALED_WAITING_ACTIVATION"
    assert chain.publish_calls == 2
    chain.height = plan.activation_block
    calls = []

    def publication(context):
        calls.append(context)
        return {
            "status": "SUBMITTED",
            "certificate_hash": context["certificate"].get("transition_hash", "0" * 64),
            "intent_hash": sha(801),
            "submitted_extrinsic_hash": sha(802),
        }

    certified = tick(
        root,
        config=config,
        contract=contract,
        chain=chain,
        wallets=wallets,
        shadow=False,
        publication_step=publication,
        export_destination=dashboard,
    )
    assert certified["status"] == "SUBMITTED"
    assert len(calls) == 1
    snapshot = service.snapshot()["rounds"][0]
    assert snapshot["evaluation"]["status"] == "PASS", snapshot["evaluation"]
    assert snapshot["status"] == "CERTIFIED"

    def finalized(context):
        return {
            "status": "FINALIZED",
            "certificate_hash": context["certificate"]["transition_hash"],
            "intent_hash": sha(801),
            "submitted_extrinsic_hash": sha(802),
            "finalized_block": chain.height,
            "readback_hash": sha(803),
        }

    complete = tick(
        root,
        config=config,
        contract=contract,
        chain=chain,
        wallets=wallets,
        shadow=False,
        publication_step=finalized,
        export_destination=dashboard,
    )
    assert complete["status"] == "COMPLETE"
    assert service.snapshot()["rounds"][0]["status"] == "COMPLETE"
    projected = json.loads(dashboard.read_text())
    assert projected["bootstrap_reference"]["fingerprint"] == king
    assert projected["publication"]["effective_status"] == "FINALIZED"

    advanced = FinalizedCompetitionState.from_dict(json.loads((root / "state.json").read_text()))
    assert json.loads((root / "manifests" / "king.json").read_text())["fingerprint"] == candidate
    for label in wallets:
        with Store(root / f"canonical-validator-{label}.sqlite") as store:
            assert (
                FinalizedCompetitionState.from_dict(
                    store.latest_state_payload(advanced.challenge_id)
                ).state_hash()
                == advanced.state_hash()
            )
    # A promoted king does not change the original quality anchor or require
    # an imported-baseline receipt when opening the following round.
    second = closed_selection(
        config, close_block=600, close_hash=sha(600), finalized_block=600,
        parent_state_hash=advanced.state_hash(), candidate_manifest_hash=sha(999),
        candidate_hotkey="miner", commitment_block=600,
    )
    service.select(second, finalized_block=600)
    second_dir = root / "round-600"
    second_dir.mkdir()
    second_history = build_authenticated_history(
        network="test", netuid=578, through_block=600,
        certified_state_hash=advanced.state_hash(), consumed_hotkeys=("miner",),
        registered_hotkeys=("miner",), window_id="launch-600",
        window_open_block=300, window_close_block=600,
    )
    from forge.validator.launch_round import _history_payload
    (second_dir / "admission-history.json").write_text(json.dumps(_history_payload(second_history)))
    (second_dir / "committee.json").write_text(
        json.dumps({"schema": "forge/launch-committee/v1", "committee": committee})
    )
    _, next_plan = freeze_closed_round(
        config=config, parent=advanced, service=service, selection=second,
        entropy_hash=sha(610), finalized_block=610,
        basic_manifests={"candidate": sha(999), "king": candidate, "anchor": king},
        worker_ready=worker_ready, qualification_package=package,
        admission_history=second_history, committee=committee, round_dir=second_dir,
    )
    assert next_plan.king_fingerprint == candidate
    assert next_plan.prompt_seed_inputs["anchor_fingerprint"] == king
    assert "bootstrap_receipt_sha256" not in next_plan.prompt_seed_inputs
    execution_blocks = math.ceil(
        worker_ready["capacity"]["worst_case_execution_seconds"]
        / worker_ready["capacity"]["chain_block_seconds"]
    )
    # Finishing exactly at consensus' no-new-commit cutoff was formerly
    # accepted and immediately became uncommittable.  READY must expire before
    # it creates a budget reservation or starts a worker.
    chain.height = (
        next_plan.commit_deadline_block
        - execution_blocks
        - MINIMUM_COMMIT_WINDOW_BLOCKS
    )
    late = tick(
        root, config=config, contract=contract, chain=chain, wallets=wallets,
        shadow=False, reserve_usd=1.0, worker_command=[sys.executable, str(script)],
        export_destination=dashboard,
    )
    assert late["status"] == "EXPIRED"
    assert not (second_dir / "dispatch-intent.json").exists()
    assert service.snapshot()["budget"]["reserved_or_captured_usd"] == reserved

    # A pre-existing reservation can survive a process restart.  It must be
    # rechecked before dispatch too, so it cannot create a worker or signature
    # at the same consensus cutoff.
    third = closed_selection(
        config, close_block=900, close_hash=sha(900), finalized_block=900,
        parent_state_hash=advanced.state_hash(), candidate_manifest_hash=sha(1000),
        candidate_hotkey="miner-2", commitment_block=900,
    )
    service.select(third, finalized_block=900)
    third_dir = root / "round-900"
    third_dir.mkdir()
    third_history = build_authenticated_history(
        network="test", netuid=578, through_block=900,
        certified_state_hash=advanced.state_hash(), consumed_hotkeys=(),
        registered_hotkeys=("miner", "miner-2"), window_id="launch-900",
        window_open_block=600, window_close_block=900,
    )
    (third_dir / "admission-history.json").write_text(
        json.dumps(_history_payload(third_history))
    )
    (third_dir / "committee.json").write_text(
        json.dumps({"schema": "forge/launch-committee/v1", "committee": committee})
    )
    _, third_plan = freeze_closed_round(
        config=config, parent=advanced, service=service, selection=third,
        entropy_hash=sha(910), finalized_block=910,
        basic_manifests={"candidate": sha(1000), "king": candidate, "anchor": king},
        worker_ready=worker_ready, qualification_package=package,
        admission_history=third_history, committee=committee, round_dir=third_dir,
    )
    third_service_plan_hash = service.snapshot()["rounds"][-1]["plan_hash"]
    service.request_work(
        third_plan.round_id,
        reserve_usd=1.0,
        request={
            "schema": "forge/launch-worker-request/v2",
            "round_id": third_plan.round_id,
            "plan_hash": third_service_plan_hash,
            "round_dir": str(third_dir),
            "result_path": str(third_dir / "worker-result.json"),
        },
    )
    published_before = dict(chain.rows)
    chain.height = (
        third_plan.commit_deadline_block
        - execution_blocks
        - MINIMUM_COMMIT_WINDOW_BLOCKS
    )
    requested_late = tick(
        root, config=config, contract=contract, chain=chain, wallets=wallets,
        shadow=False, worker_command=[sys.executable, str(script)],
        export_destination=dashboard,
    )
    assert requested_late["status"] == "EXPIRED"
    assert not (third_dir / "dispatch-intent.json").exists()
    assert not (third_dir / "worker-result.json.marker").exists()
    assert chain.rows == published_before
