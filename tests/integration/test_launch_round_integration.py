"""Durable V2 freeze through real filesystem receipts, without chain/GPU writes."""

import importlib.util
import json
from pathlib import Path

from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.validator.admission import build_authenticated_history
from forge.validator.launch_config import LaunchConfig, closed_selection
from forge.validator.launch_round import freeze_closed_round
from forge.validator.launch_service import LaunchServiceStore

_fixture_spec = importlib.util.spec_from_file_location(
    "launch_bridge_fixture", Path(__file__).with_name("test_launch_bridge_integration.py")
)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
_external_gpu_contract_package = _fixture._external_gpu_contract_package
sha = _fixture.sha


def test_v2_round_freeze_replays_qualification_and_persists_bound_receipts(tmp_path):
    candidate, king = sha(901), sha(902)
    policy_fixture, qualification = _external_gpu_contract_package(
        tmp_path / "qualification", candidate=candidate, king=king
    )
    contract = ChallengeContract()
    config = LaunchConfig(
        network="test",
        netuid=578,
        genesis_hash=sha(1),
        contract_hash=contract.contract_hash(),
        campaign_id="round-freeze-fixture",
    )
    parent = FinalizedCompetitionState.genesis("round-freeze", king)
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=300,
        certified_state_hash=parent.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    service = LaunchServiceStore(tmp_path / "service.sqlite", config)
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
    round_dir = tmp_path / "round"
    round_dir.mkdir()
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
                "window_id": None,
                "window_open_block": None,
                "window_close_block": None,
            }
        )
    )
    (round_dir / "committee.json").write_text(
        json.dumps({"schema": "forge/launch-committee/v1", "committee": {"validator": 1}})
    )
    worker_ready = {
        "evaluator": policy_fixture["evaluator"],
        "runtime": policy_fixture["runtime"],
        "capacity": {
            "measured_parallelism": 1,
            "worst_case_execution_seconds": 20,
            "commit_delivery_seconds": 5,
            "reveal_delivery_seconds": 5,
            "chain_block_seconds": 2,
        },
    }

    bootstrap = tmp_path / "bootstrap-reference.json"
    bootstrap.write_text(json.dumps({"source": "fixture"}))

    policy, plan = freeze_closed_round(
        config=config,
        parent=parent,
        service=service,
        selection=selection,
        entropy_hash="e" * 64,
        finalized_block=310,
        basic_manifests={"candidate": candidate, "king": king, "anchor": sha(998)},
        worker_ready=worker_ready,
        qualification_package=qualification,
        admission_history=history,
        committee={"validator": 1},
        round_dir=round_dir,
        bootstrap_receipt_path=bootstrap,
    )

    assert plan.consensus_policy.value == "stake_majority_v2"
    assert plan.commit_deadline_block == 323
    assert plan.prompt_seed_inputs["launch_plan_id"] == policy["plan_id"]
    capacity = json.loads((round_dir / "capacity.json").read_text())
    assert capacity["round_plan_hash"] == plan.plan_hash()
    assert capacity["launch_plan_id"] == policy["plan_id"]
    assert (
        json.loads((round_dir / "prompt-history.json").read_text())["prompt_history_root"]
        == plan.prompt_seed_inputs["prompt_history_root"]
    )
