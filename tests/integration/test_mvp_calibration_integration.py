"""Calibration holdout coverage through the public MVP evaluator boundary."""

from __future__ import annotations

import json
import subprocess
import sys

from forge.evaluation.mvp_calibration import fit_policy_from_development, validate_calibration
from forge.evaluation.mvp_policy import CONTROL_AXES, CONTROL_KINDS, QUALITY_AXES, _sha
from forge.evaluation.rtx5090_probe import build_probe_manifest
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.models.distributed_worker import load_diagnostic_jobs
from forge.protocol.contract import ChallengeContract

PROVENANCE = {
    "source_implementation_hash": "forge-fusion@pinned",
    "vbench_model_hashes": "vbench-weights@pinned",
    "luna_evaluator_id": "luna@pinned",
}


def _orders(delta_by_axis: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"reference_first": True, "axes": dict(delta_by_axis)},
        {"reference_first": False, "axes": dict(delta_by_axis)},
    ]


def _row(split: str, kind: str, index: int, *, detected: bool = True) -> dict[str, object]:
    axes = {axis: 0.0 for axis in QUALITY_AXES}
    if kind in CONTROL_AXES and detected:
        for axis in CONTROL_AXES[kind]:
            axes[axis] = -0.5
    unique = f"{split}-{kind}-{index}"
    return {
        "split": split,
        "kind": kind,
        "source_id": f"source-{unique}",
        "concept_id": f"concept-{unique}",
        "source_family": f"source-family-{unique}",
        "concept_family": f"concept-family-{unique}",
        "observable": True,
        "orders": _orders(axes),
    }


def _calibration(*, calibration_blur_detected: bool = True) -> tuple[object, dict[str, object]]:
    development = [_row("development", kind, 0) for kind in CONTROL_KINDS]
    policy, derivation = fit_policy_from_development(development, evaluator_provenance=PROVENANCE)
    rows = list(development)
    for split in ("calibration", "independent"):
        for kind in CONTROL_KINDS:
            rows.extend(
                _row(
                    split,
                    kind,
                    index,
                    detected=not (split == "calibration" and kind == "blur_mild")
                    or calibration_blur_detected,
                )
                for index in range(32)
            )
    receipt = {
        "policy_id": policy.policy_id,
        "evaluator_provenance": PROVENANCE,
        "rows": rows,
        "threshold_derivation": derivation,
    }
    receipt["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": rows,
        }
    )
    return policy, receipt


def test_calibration_requires_transfer_to_calibration_and_independent_holdout():
    policy, receipt = _calibration()

    result = validate_calibration(receipt, policy)

    assert result["status"] == "PASS"
    assert set(result["summary"]) == {"calibration", "independent"}
    assert result["summary"]["independent"]["blur_mild"]["sensitivity_lower"] >= 0.8


def test_failed_calibration_split_cannot_be_rescued_by_independent_holdout():
    policy, receipt = _calibration(calibration_blur_detected=False)

    result = validate_calibration(receipt, policy)

    assert result["status"] == "INCONCLUSIVE"
    assert "CALIBRATION_SPLIT_BLUR_MILD_SENSITIVITY" in result["reasons"]
    assert set(result["summary"]) == {"calibration"}


def test_rtx5090_probe_separates_fresh_splits_but_cannot_claim_calibration():
    probe = build_probe_manifest("rtx5090-20261004")

    assert probe["promotion_eligible"] is False
    assert probe["expected_gate"] == "INCONCLUSIVE"
    assert probe["media_budget"]["total_media_files"] == 31
    assert probe["media_budget"]["quality_gpu_renders"] == 15
    assert probe["timing"]["runs_per_prompt"] == [
        "reference",
        "candidate",
        "candidate",
        "reference",
    ]
    assert {row["split"] for row in probe["sources"]} == {
        "development",
        "calibration",
        "independent",
    }
    assert {row["kind"] for row in probe["control_observations"]} == set(CONTROL_KINDS)
    static = next(row for row in probe["control_observations"] if row["kind"] == "static_scene")
    assert static["split"] == "calibration"
    assert static["observable"] is None
    assert probe["transform_definitions"]["freeze"]["eligible"] == "requested_motion"


def test_rtx5090_execution_helper_prepares_existing_worker_and_abba_inputs(tmp_path):
    contract = ChallengeContract(max_steps=50, reference_steps=50)
    base = ModelManifest(
        ArtifactReference("hf", contract.base_model_id, contract.base_model_revision),
        "pipeline",
        (ManifestFile("model_index.json", "a" * 64, 1),),
        "apache-2.0",
        {"steps": 50, "guidance_scale": 5.0},
    )
    probe_path = tmp_path / "probe.json"
    contract_path = tmp_path / "contract.json"
    manifest_path = tmp_path / "base.json"
    probe_path.write_text(json.dumps(build_probe_manifest("rtx5090-test")))
    contract_path.write_text(json.dumps(contract.to_dict()))
    manifest_path.write_text(json.dumps(base.to_dict()))
    prepared = tmp_path / "prepared"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rtx5090_probe.py",
            "prepare",
            "--probe",
            str(probe_path),
            "--contract",
            str(contract_path),
            "--base-manifest",
            str(manifest_path),
            "--base-dir",
            str(tmp_path / "model-cache"),
            "--validator-hotkey",
            "test-hotkey",
            "--max-generation-seconds",
            "900",
            "--out",
            str(prepared),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["quality_jobs"] == 15
    assert result["quality_warmups"] == 3
    assert result["timing_measured_renders"] == 8
    assert result["timing_warmups"] == 2
    assert len(load_diagnostic_jobs(prepared / "quality-jobs.json").jobs) == 15
    controls = json.loads((prepared / "controls.json").read_text())["items"]
    static = next(row for row in controls if row["kind"] == "static_scene")
    assert static["split"] == "calibration"
    assert static["candidate_id"] == "calibration-same-model-rerun"
