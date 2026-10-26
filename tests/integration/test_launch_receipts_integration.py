"""Operator transfer into trusted custody, followed by full V2 raw replay."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.admission import build_authenticated_history
from forge.validator.launch_receipts import import_campaign, replay_baseline_campaign

_spec = importlib.util.spec_from_file_location(
    "launch_bridge_receipt_fixture", Path(__file__).with_name("test_launch_bridge_integration.py")
)
assert _spec is not None and _spec.loader is not None
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def test_operator_import_replays_external_gpu_bundle_and_rejects_changed_transfer(tmp_path):
    base, manifests, hardware, runtime_doc = _fixture.policy()
    members = _fixture.signers()[:2]
    committee = {member.validator_hotkey: 1 for member in members}
    parent = FinalizedCompetitionState.genesis("receipts", manifests["king"].fingerprint())
    frozen, qualification = _fixture._external_gpu_contract_package(
        tmp_path / "external-provider",
        candidate=manifests["candidate"].fingerprint(),
        king=manifests["king"].fingerprint(),
        anchor=manifests["anchor"].fingerprint(),
        runtime={
            key: base["runtime"][key]
            for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
        },
        entropy=_fixture.sha(900),
        round_id="receipt-transfer",
    )
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=100,
        certified_state_hash=parent.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    plan = _fixture.round_plan(
        parent, committee, frozen, history.history_root, "receipt-transfer", _fixture.sha(900)
    )
    source, round_dir = tmp_path / "operator-fetch", tmp_path / "round"
    round_dir.mkdir()
    digest = _fixture._raw_evidence_v2(
        source,
        frozen,
        manifests,
        hardware,
        runtime_doc,
        plan,
        members[0].validator_hotkey,
        qualification,
    )
    bundle = json.loads((source / "bundles" / f"{digest}.json").read_text())
    index = {
        "schema": "forge/launch-campaign-receipts/v2",
        "plan_id": frozen["plan_id"],
        "campaign_hash": _fixture.sha(902),
        "panels": [f"panels/{value}.json" for value in bundle["panel_receipts"]],
        "timing": [f"timing/{value}.json" for value in bundle["timing_receipts"]],
        "scores": [f"scores/{value}.json" for value in bundle["score_receipts"]],
        "diversity": [f"diversity/{value}.json" for value in bundle["diversity_receipts"]],
        "qualification": [f"qualification/{bundle['qualification_receipt']}.json"],
    }
    index["diversity_panels"] = [
        f"panels/{json.loads((source / path).read_text())['source_report_sha256']}.json"
        for path in index["diversity"]
    ]
    index["sha256"] = {
        path: sha256_file(source / path)
        for paths in index.values()
        if isinstance(paths, list)
        for path in paths
    }
    (source / "receipt-index.json").write_text(json.dumps(index))
    request_hash = _fixture.sha(901)
    for name, value in (
        ("round-plan.json", plan.to_dict()),
        ("launch-plan.json", frozen),
        ("worker-request.json", {"request_hash": request_hash}),
    ):
        (round_dir / name).write_text(json.dumps(value))
    validators = {f"v{index}": member.validator_hotkey for index, member in enumerate(members)}
    result = import_campaign(round_dir, source, validators=validators, request_hash=request_hash)
    assert set(result["trusted_bundle_hashes"]) == set(committee)
    assert (round_dir / "bundle-v0.json").is_file()
    assert (round_dir / "bundle-v1.json").is_file()
    assert (
        import_campaign(round_dir, source, validators=validators, request_hash=request_hash)
        == result
    )
    score = source / index["scores"][0]
    score.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        import_campaign(round_dir, source, validators=validators, request_hash=request_hash)
    assert json.loads((round_dir / "worker-result.json").read_text()) == result


def test_baseline_campaign_replays_raw_v2_receipts_without_consensus_bundle(tmp_path):
    """A selected baseline uses the same custody graph, not a fabricated round."""
    selection_id = "baseline-selection-20261020-immutable"
    entropy = hash_canonical("forge/baseline-confirmation-entropy/v1", selection_id)
    base, manifests, hardware, runtime_doc = _fixture.policy()
    members = _fixture.signers()[:2]
    parent = FinalizedCompetitionState.genesis("baseline", manifests["king"].fingerprint())
    frozen, qualification = _fixture._external_gpu_contract_package(
        tmp_path / "simulated-external-provider",
        candidate=manifests["candidate"].fingerprint(),
        king=manifests["king"].fingerprint(),
        anchor=manifests["king"].fingerprint(),
        runtime={
            key: base["runtime"][key]
            for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
        },
        entropy=entropy,
        round_id=selection_id[:32],
    )
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=100,
        certified_state_hash=parent.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    round_plan = _fixture.round_plan(
        parent,
        {member.validator_hotkey: 1 for member in members},
        frozen,
        history.history_root,
        "fixture-custody-only",
        entropy,
    )
    source = tmp_path / "baseline-campaign"
    bundle_hash = _fixture._raw_evidence_v2(
        source,
        frozen,
        {
            "candidate": manifests["candidate"],
            "king": manifests["king"],
            "anchor": manifests["king"],
        },
        hardware,
        runtime_doc,
        round_plan,
        members[0].validator_hotkey,
        qualification,
    )
    bundle = json.loads((source / "bundles" / f"{bundle_hash}.json").read_text())
    campaign_body = {
        "schema": "forge/launch-campaign/v2",
        "plan_id": frozen["plan_id"],
        "contract_hash": round_plan.contract_hash,
        "jobs": [],
    }
    campaign = {
        **campaign_body,
        "campaign_hash": hash_canonical("forge/launch-campaign/v2", campaign_body),
    }
    (source / "launch-plan.json").write_text(json.dumps(frozen))
    (source / "campaign.json").write_text(json.dumps(campaign))
    index = {
        "schema": "forge/launch-campaign-receipts/v2",
        "plan_id": frozen["plan_id"],
        "campaign_hash": campaign["campaign_hash"],
        "panels": [f"panels/{value}.json" for value in bundle["panel_receipts"]],
        "timing": [f"timing/{value}.json" for value in bundle["timing_receipts"]],
        "scores": [f"scores/{value}.json" for value in bundle["score_receipts"]],
        "diversity": [f"diversity/{value}.json" for value in bundle["diversity_receipts"]],
        "diversity_panels": [],
        "qualification": [f"qualification/{bundle['qualification_receipt']}.json"],
    }
    diversity_panels = []
    for value in bundle["diversity_receipts"]:
        diversity = json.loads((source / "diversity" / f"{value}.json").read_text())
        diversity_panels.append(f"panels/{diversity['source_report_sha256']}.json")
    index["diversity_panels"] = diversity_panels
    index["sha256"] = {
        path: sha256_file(source / path)
        for paths in index.values()
        if isinstance(paths, list)
        for path in paths
    }
    (source / "receipt-index.json").write_text(json.dumps(index))

    replay = replay_baseline_campaign(
        source,
        tmp_path / "baseline-custody",
        plan=frozen,
        contract_hash=round_plan.contract_hash,
        selection_id=selection_id,
    )
    assert replay["result"]["status"] == "PASS"
    assert len(replay["rows"]) == 76

    # The renderer's pinned environment intentionally has no chain SDK. Run
    # the real raw replay in a fresh interpreter whose external chain boundary
    # rejects every bittensor import; project and evaluator dependencies remain
    # real packages, rather than mocked internals.
    isolated = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import json
import sys
from pathlib import Path

class NoBittensor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bittensor" or fullname.startswith("bittensor."):
            raise ModuleNotFoundError("bittensor is unavailable in the renderer environment")
        return None

sys.meta_path.insert(0, NoBittensor())

from forge.validator.launch_receipts import replay_baseline_campaign

result = replay_baseline_campaign(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    plan=json.loads(sys.argv[3]),
    contract_hash=sys.argv[4],
    selection_id=sys.argv[5],
)
if "bittensor" in sys.modules:
    raise RuntimeError("offline replay imported bittensor")
if result["result"]["status"] != "PASS":
    raise RuntimeError("offline replay did not pass")
print(json.dumps({"rows": len(result["rows"]), "status": result["result"]["status"]}))
""",
            str(source),
            str(tmp_path / "isolated-baseline-custody"),
            json.dumps(frozen),
            round_plan.contract_hash,
            selection_id,
        ],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    assert json.loads(isolated.stdout) == {"rows": 76, "status": "PASS"}

    changed = source / index["scores"][0]
    changed.write_text("{}")
    index["sha256"][index["scores"][0]] = sha256_file(changed)
    (source / "receipt-index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="score receipt"):
        replay_baseline_campaign(
            source,
            tmp_path / "baseline-custody-changed",
            plan=frozen,
            contract_hash=round_plan.contract_hash,
            selection_id=selection_id,
        )
