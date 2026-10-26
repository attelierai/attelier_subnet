"""The GPU worker accepts only frozen, diagnostic assignments before CUDA exists."""

import json

import pytest

from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.models.distributed_worker import (
    JOB_SCHEMA,
    inspect_diagnostic_jobs,
    load_diagnostic_jobs,
)
from forge.protocol.contract import ChallengeContract


def _sha(number: int) -> str:
    return f"{number:064x}"


def _bundle(tmp_path, *, diagnostic_only=True, scheduler="original_unipc"):
    contract = ChallengeContract(max_steps=50, reference_steps=50)
    base_ref = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    base = ModelManifest(
        base_ref,
        "pipeline",
        (ManifestFile("model_index.json", _sha(1), 1),),
        "apache-2.0",
        {"steps": 50, "guidance_scale": 5.0},
    )
    payload = {
        "schema": JOB_SCHEMA,
        "diagnostic_only": diagnostic_only,
        "validator_hotkey": "validator-a",
        "round_id": "diagnostic-r1",
        "contract": contract.to_dict(),
        "base_manifest": base.to_dict(),
        "base_dir": "/remote/wan-base",
        "max_generation_seconds": 120.0,
        "jobs": [
            {
                "job_id": "quality-p0001-base",
                "kind": "quality_reference",
                "prompt_id": "p-0001",
                "prompt": "A red kite moving through a blue sky.",
                "seed": 7,
                "model_manifest": base.to_dict(),
                "model_dir": "/remote/wan-base",
                "scheduler": scheduler,
            }
        ],
    }
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(payload))
    return path


def test_diagnostic_worker_plan_is_cpu_only_and_binds_exact_job_identity(tmp_path):
    path = _bundle(tmp_path)

    result = inspect_diagnostic_jobs(path)
    bundle = load_diagnostic_jobs(path)

    assert result["diagnostic_only"] is True
    assert result["jobs"] == [
        {"job_id": "quality-p0001-base", "job_hash": bundle.jobs[0].job_hash()}
    ]
    assert result["bundle_hash"] == bundle.bundle_hash()
    assert bundle.jobs[0].identity()["scheduler"] == "original_unipc"


def test_diagnostic_worker_rejects_competitive_or_recipe_inconsistent_jobs(tmp_path):
    path = _bundle(tmp_path, diagnostic_only=False)
    with pytest.raises(ValueError, match="diagnostic_only"):
        load_diagnostic_jobs(path)

    path = _bundle(tmp_path, scheduler="uniform_euler")
    with pytest.raises(ValueError, match="scheduler differs"):
        load_diagnostic_jobs(path)


def test_worker_uses_explicit_candidate_scheduler_and_rejects_uncommitted_override(tmp_path):
    path = _bundle(tmp_path)
    payload = json.loads(path.read_text())
    job = payload["jobs"][0]
    job["kind"] = "quality_candidate"
    job["model_manifest"]["generation_config"] = {
        "steps": 40,
        "guidance_scale": 5.0,
        "scheduler": "original_unipc",
    }
    path.write_text(json.dumps(payload))
    inspected = inspect_diagnostic_jobs(path)
    loaded = load_diagnostic_jobs(path)
    assert loaded.jobs[0].scheduler == "original_unipc"
    assert inspected["jobs"][0]["job_hash"] == loaded.jobs[0].job_hash()
    job["scheduler"] = "uniform_euler"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="scheduler differs"):
        load_diagnostic_jobs(path)
