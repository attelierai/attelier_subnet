"""Qualified certificate to two native journals, reveal and restart.

Only GPU and raw chain/SDK boundaries are fake. The runtime binary, raw receipt
replay, sr25519 certificates, SQLite authority and publication code are real.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from launch_native_chain_fixture import CompositeChain
from test_launch_bridge_integration import (
    _external_gpu_contract_package,
    _raw_evidence_v2,
    policy,
    put,
    round_plan,
    sha,
    signers,
)
from test_launch_fallback_integration import (  # noqa: F401
    deployed_runtime_483_wasm,
    deployed_runtime_484_wasm,
    official_runtime_wasm,
)

from forge.chain import Sr25519Verifier
from forge.chain.launch_fallback import RUNTIME_483
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.protocol.preconsensus import build_certificate, commit_report, decide_round
from forge.protocol.store import Store
from forge.validator.admission import build_authenticated_history
from forge.validator.launch_bridge import (
    LaunchCertificateVerifier,
    LaunchExecutionReplayer,
    LaunchV1Bridge,
    LaunchV1Capacity,
    LocalLaunchReceiptStore,
    launch_local_report,
)
from forge.validator.launch_certified_dashboard import (
    CertifiedDashboardError,
    export_native_certified_snapshot,
)
from forge.validator.launch_config import LaunchConfig
from forge.validator.launch_publish import publication_callback


@pytest.mark.parametrize("runtime_fixture", [
    "official_runtime_wasm", "deployed_runtime_483_wasm", "deployed_runtime_484_wasm",
])
def test_keep_certificate_publishes_every_validator_and_reconciles_after_restart(
    tmp_path, request, runtime_fixture
):
    runtime_wasm = request.getfixturevalue(runtime_fixture)
    base, manifests, hardware, runtime = policy()
    contract = ChallengeContract()
    config = LaunchConfig(
        "test", 578, RUNTIME_483["genesis_hash"], contract.contract_hash(), "publish-fixture"
    )
    parent = FinalizedCompetitionState.genesis("wan", manifests["king"].fingerprint())
    members = signers()[:2]
    wallets = {
        f"v{i}": SimpleNamespace(
            hotkey=m._keypair, coldkey=m._keypair, coldkeypub=m._keypair
        ) for i, m in enumerate(members)
    }
    root, directory = tmp_path / "operator", tmp_path / "operator" / "round-100"
    receipts_root = directory / "trusted-receipts"
    directory.mkdir(parents=True)
    frozen, package = _external_gpu_contract_package(
        tmp_path / "fake-gpu", candidate=manifests["candidate"].fingerprint(),
        king=parent.current_king_fingerprint, anchor=parent.current_king_fingerprint,
        runtime={key: base["runtime"][key] for key in (
            "runtime_id", "hardware_id", "dependency_lock_hash"
        )}, entropy=sha(700), round_id="publish-fixture",
    )
    history = build_authenticated_history(
        network="test", netuid=578, through_block=100,
        certified_state_hash=parent.state_hash(), consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    plan = replace(
        round_plan(
            parent, {m.validator_hotkey: 1 for m in members}, frozen,
            history.history_root, "publish-fixture", sha(700),
        ),
        contract_hash=contract.contract_hash(),
    )
    capacity = put(receipts_root, "capacity", {
        "schema": "forge/launch-capacity-receipt/v1", "round_plan_hash": plan.plan_hash(),
        "launch_plan_id": frozen["plan_id"], "measured_parallelism": 1,
        "worst_case_execution_seconds": 1.0, "commit_delivery_seconds": 1.0,
        "reveal_delivery_seconds": 1.0, "chain_block_seconds": 1.0,
    })
    replay = LaunchExecutionReplayer(LocalLaunchReceiptStore(receipts_root))
    reports = [
        launch_local_report(
            plan, signer.validator_hotkey,
            _raw_evidence_v2(
                receipts_root, frozen, manifests, hardware, runtime, plan,
                signer.validator_hotkey, package, candidate_score=-1.0,
            ), replay,
        ) for signer in members
    ]
    assert {report.verdict.value for report in reports} == {"KEEP"}
    commits, reveals = zip(*(
        commit_report(plan, report, 150, salt=f"publish-{i}")
        for i, report in enumerate(reports)
    ), strict=True)
    certificate = build_certificate(
        plan, decide_round(plan, commits, reveals), 401, members,
        commits=commits, reveals=reveals,
    )
    for label in wallets:
        with Store(directory / f"validator-{label}.sqlite") as store:
            bridge = LaunchV1Bridge(store, Sr25519Verifier(), replay.receipts)
            bridge.open_round(
                plan, parent, create_signed_history(history, members),
                frozen, LaunchV1Capacity(capacity),
            )
            certified = bridge.apply(
                plan, parent, certificate,
                LaunchCertificateVerifier(Sr25519Verifier(), replayer=replay, store=store),
            )
    for name, value in {
        "state.json": parent.to_dict(), "round-plan.json": plan.to_dict(),
        "launch-plan.json": frozen, "qualification-package.json": package,
    }.items():
        (directory / name).write_text(json.dumps(value))
    chain = CompositeChain(wallets, runtime_wasm, config.genesis_hash)
    chain.spec_version = {
        "official_runtime_wasm": 481, "deployed_runtime_483_wasm": 483,
        "deployed_runtime_484_wasm": 484,
    }[runtime_fixture]
    chain.finalized = plan.activation_block
    kwargs = {
        "configuration": {
            "evidence_origin": "external_gpu_qualified_v2", "validator_labels": list(wallets),
        },
        "config": config, "contract": contract, "chain": chain, "wallets": wallets,
        "shadow": False,
    }
    context = {
        "round_dir": str(directory), "certificate": certificate.to_dict(),
        "state": certified.to_dict(),
    }
    (directory / "certificate.json").write_text(json.dumps(certificate.to_dict()))
    hotkeys = {label: wallet.hotkey.ss58_address for label, wallet in wallets.items()}
    for label in wallets:
        (directory / f"certified-state-{label}.json").write_text(json.dumps(certified.to_dict()))
    challenger = f"launch-v1-{directory.name}-{plan.challenger_fingerprint[:12]}"
    diagnostic = f"operator-experiment-{directory.name}"
    snapshot = {
        "schema": "forge/dashboard/v2", "king": None, "crowns": [],
        "challengers": [{"id": challenger, "round_id": diagnostic,
                         "manifest_hash": plan.challenger_fingerprint}],
        "evaluations": [{"challenger_id": challenger, "plan_hash": frozen["plan_id"]}],
        "rounds": [{"round_id": diagnostic, "diagnostic": True}],
    }
    first = publication_callback(root, **kwargs)(context)
    assert first is not None and first["status"] == "SUBMITTED"
    with pytest.raises(CertifiedDashboardError, match="not finalized"):
        export_native_certified_snapshot(
            snapshot, directory, chain=chain, validator_hotkeys=hotkeys,
        )
    assert len(chain.subtensor._client._substrate.submissions) == 2
    chain.reveal("v0")
    partial = publication_callback(root, **kwargs)(context)
    assert partial is not None and partial["status"] == "SUBMITTED"
    chain.reveal("v1")
    complete = publication_callback(root, **kwargs)(context)
    assert complete is not None and complete["status"] == "FINALIZED"
    assert publication_callback(root, **kwargs)(context) == complete
    assert len(chain.subtensor._client._substrate.submissions) == 2
    for label in wallets:
        assert (directory / f"weights-readback-{label}.json").is_file()
    exported = export_native_certified_snapshot(
        snapshot, directory, chain=chain, validator_hotkeys=hotkeys,
    )
    assert exported["king"] is None and exported["crowns"] == []
    assert exported["evaluations"][0]["verdict"]["status"] == "KEEP"
    assert (
        exported["launch_v1_certification"]["activation"]["status"]
        == "FINALIZED_WEIGHTS_VERIFIED"
    )
    assert exported["launch_v1_certification"]["activation"]["validator_count"] == 2
    with pytest.raises(CertifiedDashboardError, match="distinct"):
        export_native_certified_snapshot(
            snapshot, directory, chain=chain,
            validator_hotkeys={label: hotkeys["v0"] for label in wallets},
        )
    # A copied JSON readback cannot replace the native attribution proof.
    original_events = chain.events
    chain.events = {
        block: [event for event in events if event["name"] == "TimelockedWeightsCommitted"]
        for block, events in original_events.items()
    }
    with pytest.raises(CertifiedDashboardError, match="not attributable"):
        export_native_certified_snapshot(
            snapshot, directory, chain=chain, validator_hotkeys=hotkeys,
        )
    chain.events = original_events
    assert len(chain.subtensor._client._substrate.submissions) == 2
