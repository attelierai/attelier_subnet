"""Durable V2 bootstrap reference replay over SQLite baseline evidence."""

import copy
from dataclasses import replace

import pytest

# Shared external-boundary fixtures keep this test on the real SQLite, receipt,
# sr25519 and replay paths.  The fake GPU is confined to that provider boundary.
from test_launch_bridge_integration import (
    _external_gpu_contract_package,
    _raw_evidence_v2,
    policy,
    put,
    round_plan,
    signers,
)

from forge.chain import Sr25519Verifier
from forge.evaluation.launch_baseline import ParentLaunchBaselineStore
from forge.evaluation.launch_reference import (
    qualified_bootstrap_reference,
    replay_bootstrap_reference,
)
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
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


def _hash(number: int) -> str:
    return f"{number:064x}"


def _screening(candidate: str, anchor: str, *, eligible: bool) -> dict:
    score = 0.0 if eligible else -1.0
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
                "candidate_scores": {"VQ": score, "MQ": score, "TA": score},
            }
            for index in range(6)
        ],
    }


def _manifests(contract: ChallengeContract) -> tuple[ModelManifest, ModelManifest]:
    base = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    wan50 = ModelManifest(
        base,
        "pipeline",
        (ManifestFile("model_index.json", _hash(10), 1),),
        "apache-2.0",
        {"steps": 50, "guidance_scale": 5.0},
    )
    # An imported Wan40 may reuse the exact public pipeline but must make its
    # changed scheduler explicit; it is not restricted to LoRA/transformer.
    imported = ModelManifest(
        base,
        "pipeline",
        (ManifestFile("model_index.json", _hash(10), 1),),
        "apache-2.0",
        {"steps": 40, "guidance_scale": 5.0, "scheduler": "uniform_euler"},
    )
    return wan50, imported


def test_pending_selected_baseline_cannot_create_bootstrap_receipt(tmp_path):
    contract = ChallengeContract()
    wan50, imported = _manifests(contract)
    store = ParentLaunchBaselineStore(tmp_path / "baseline.sqlite")
    store.select_once(
        [_screening(imported.fingerprint(), wan50.fingerprint(), eligible=True)],
        anchor_hash=wan50.fingerprint(),
    )

    with pytest.raises(ValueError, match="pending"):
        qualified_bootstrap_reference(
            store,
            wan50_manifest=wan50,
            imported_manifest=imported,
            contract=contract,
        )


def test_ineligible_baseline_replays_to_immutable_wan50_anchor(tmp_path):
    contract = ChallengeContract()
    wan50, imported = _manifests(contract)
    store = ParentLaunchBaselineStore(tmp_path / "baseline.sqlite")
    store.select_once(
        [_screening(imported.fingerprint(), wan50.fingerprint(), eligible=False)],
        anchor_hash=wan50.fingerprint(),
    )

    receipt = qualified_bootstrap_reference(
        store,
        wan50_manifest=wan50,
        imported_manifest=imported,
        contract=contract,
    )

    assert receipt["quality_anchor_hash"] == wan50.fingerprint()
    assert receipt["bootstrap_reference_hash"] == wan50.fingerprint()
    assert receipt["reward_hotkey"] is None
    assert receipt["wan50_manifest"] == wan50.to_dict()
    assert (
        replay_bootstrap_reference(
            receipt,
            contract_hash=contract.contract_hash(),
            genesis_reference_hash=wan50.fingerprint(),
        )
        == wan50.fingerprint()
    )
    receipt["bootstrap_reference_hash"] = imported.fingerprint()
    with pytest.raises(ValueError):
        replay_bootstrap_reference(
            receipt,
            contract_hash=contract.contract_hash(),
            genesis_reference_hash=wan50.fingerprint(),
        )


def test_imported_baseline_keeps_wan50_anchor_through_two_certified_v2_rounds(tmp_path):
    """An imported genesis king remains bound to its original Wan50 quality anchor."""
    contract = ChallengeContract()
    wan50, imported = _manifests(contract)
    base, _, hardware, runtime_doc = policy()
    members = signers()
    committee = {member.validator_hotkey: 1 for member in members}
    receipts_root = tmp_path / "trusted-receipts"
    receipts = LocalLaunchReceiptStore(receipts_root)
    replayer = LaunchExecutionReplayer(receipts)

    baseline_state = FinalizedCompetitionState.genesis("wan", wan50.fingerprint())
    confirmation_frozen, package = _external_gpu_contract_package(
        tmp_path / "external-baseline-confirmation",
        candidate=imported.fingerprint(),
        king=wan50.fingerprint(),
        anchor=wan50.fingerprint(),
        runtime={
            key: base["runtime"][key]
            for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
        },
        entropy=_hash(700),
        round_id="baseline-confirmation",
    )
    confirmation_history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=100,
        certified_state_hash=baseline_state.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    confirmation_plan = replace(
        round_plan(
            baseline_state,
            committee,
            confirmation_frozen,
            confirmation_history.history_root,
            "baseline-confirmation",
            _hash(700),
        ),
        contract_hash=contract.contract_hash(),
    )
    confirmation_bundle = _raw_evidence_v2(
        receipts_root,
        confirmation_frozen,
        {"candidate": imported, "king": wan50, "anchor": wan50},
        hardware,
        runtime_doc,
        confirmation_plan,
        members[0].validator_hotkey,
        package,
    )
    confirmation_replay = replayer.replay(
        confirmation_bundle, confirmation_plan, validator_hotkey=members[0].validator_hotkey
    )
    assert confirmation_replay["status"] == "PASS", confirmation_replay

    baseline_store = ParentLaunchBaselineStore(tmp_path / "baseline.sqlite")
    baseline_store.select_once(
        [_screening(imported.fingerprint(), wan50.fingerprint(), eligible=True)],
        anchor_hash=wan50.fingerprint(),
    )
    confirmed = baseline_store.confirm_once(
        confirmation_frozen, confirmation_replay["rows"], package
    )
    assert confirmed["status"] == "IMPORTED_BASELINE"
    bootstrap = qualified_bootstrap_reference(
        baseline_store,
        wan50_manifest=wan50,
        imported_manifest=imported,
        contract=contract,
    )
    assert (
        replay_bootstrap_reference(
            bootstrap,
            contract_hash=contract.contract_hash(),
            genesis_reference_hash=imported.fingerprint(),
        )
        == wan50.fingerprint()
    )
    bootstrap_file_hash = put(receipts_root, "bootstrap", bootstrap)

    tampered = copy.deepcopy(bootstrap)
    tampered["confirmation"]["rows"][0]["candidate_result"]["scores"]["VQ"] = 9.0
    tampered_body = {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = hash_canonical(
        "forge/launch-qualified-bootstrap-reference/v1", tampered_body
    )
    with pytest.raises(ValueError):
        replay_bootstrap_reference(
            tampered,
            contract_hash=contract.contract_hash(),
            genesis_reference_hash=imported.fingerprint(),
        )

    base_reference = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    first_candidate = ModelManifest(
        ArtifactReference("hf", "fixture/imported-round-one", "1" * 40),
        "lora",
        (ManifestFile("adapter_model.safetensors", _hash(301), 128),),
        "mit",
        {"steps": 16, "guidance_scale": 5.0, "scheduler": "original_unipc"},
        base_reference,
    )
    second_candidate = ModelManifest(
        ArtifactReference("hf", "fixture/imported-round-two", "2" * 40),
        "lora",
        (ManifestFile("adapter_model.safetensors", _hash(302), 128),),
        "mit",
        {"steps": 16, "guidance_scale": 5.0, "scheduler": "original_unipc"},
        base_reference,
    )

    def certify(parent, candidate, round_id, entropy):
        frozen, round_package = _external_gpu_contract_package(
            tmp_path / f"external-{round_id}",
            candidate=candidate.fingerprint(),
            king=parent.current_king_fingerprint,
            anchor=wan50.fingerprint(),
            runtime={
                key: base["runtime"][key]
                for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
            },
            entropy=entropy,
            round_id=round_id,
        )
        history = build_authenticated_history(
            network="test",
            netuid=578,
            through_block=100,
            certified_state_hash=parent.state_hash(),
            consumed_hotkeys=(),
            registered_hotkeys=("miner",),
        )
        raw_plan = round_plan(parent, committee, frozen, history.history_root, round_id, entropy)
        plan = replace(
            raw_plan,
            contract_hash=contract.contract_hash(),
            prompt_seed_inputs={
                **raw_plan.prompt_seed_inputs,
                "bootstrap_receipt_sha256": bootstrap_file_hash,
            },
        )
        capacity = put(
            receipts_root,
            "capacity",
            {
                "schema": "forge/launch-capacity-receipt/v1",
                "round_plan_hash": plan.plan_hash(),
                "launch_plan_id": frozen["plan_id"],
                "measured_parallelism": 1,
                "worst_case_execution_seconds": 1.0,
                "commit_delivery_seconds": 1.0,
                "reveal_delivery_seconds": 1.0,
                "chain_block_seconds": 1.0,
            },
        )
        manifests = {
            "candidate": candidate,
            "king": parent_manifest[parent.current_king_fingerprint],
            "anchor": wan50,
        }
        assert bridge.open_round(
            plan,
            parent,
            create_signed_history(history, members),
            frozen,
            LaunchV1Capacity(capacity),
        )
        reports = [
            launch_local_report(
                plan,
                member.validator_hotkey,
                _raw_evidence_v2(
                    receipts_root,
                    frozen,
                    manifests,
                    hardware,
                    runtime_doc,
                    plan,
                    member.validator_hotkey,
                    round_package,
                ),
                replayer,
            )
            for member in members
        ]
        assert {report.verdict.value for report in reports} == {"PROMOTE"}
        commits, reveals = zip(
            *(
                commit_report(plan, report, 150, salt=f"{round_id}-{index}")
                for index, report in enumerate(reports)
            ),
            strict=True,
        )
        certificate = build_certificate(
            plan,
            decide_round(plan, commits, reveals),
            401,
            members,
            commits=commits,
            reveals=reveals,
        )
        return bridge.apply(
            plan,
            parent,
            certificate,
            LaunchCertificateVerifier(
                signature_verifier=Sr25519Verifier(), replayer=replayer, store=store
            ),
        )

    parent_manifest = {imported.fingerprint(): imported}
    with Store(tmp_path / "rounds.sqlite") as store:
        bridge = LaunchV1Bridge(
            store,
            Sr25519Verifier(),
            receipts,
        )
        first = certify(
            FinalizedCompetitionState.genesis("wan", imported.fingerprint()),
            first_candidate,
            "imported-round-one",
            _hash(710),
        )
        assert first.current_king_fingerprint == first_candidate.fingerprint()
        parent_manifest[first.current_king_fingerprint] = first_candidate
        second = certify(first, second_candidate, "imported-round-two", _hash(711))
        assert second.current_king_fingerprint == second_candidate.fingerprint()
