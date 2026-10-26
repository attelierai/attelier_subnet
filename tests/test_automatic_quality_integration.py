import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from forge.evaluation.automatic_quality import ApiBudget, score_reports
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import sha256_file


def _video(path: Path, *, checker_attack: bool = False) -> None:
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, :, index, 0] = 255
    if checker_attack:
        checker = (np.indices((32, 32)).sum(axis=0) % 2 * 2 - 1).astype(np.int16) * 64
        frames[1:] = np.clip(128 + checker[..., None], 0, 255).astype(np.uint8)
    writer = imageio_ffmpeg.write_frames(str(path), (32, 32), fps=5)
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()


def _manifest(contract: ChallengeContract, steps: int) -> ModelManifest:
    return ModelManifest(
        ArtifactReference("hf", contract.base_model_id, contract.base_model_revision),
        "pipeline",
        (ManifestFile("model.safetensors", "0" * 64, 1),),
        "apache-2.0",
        {"steps": steps, "guidance_scale": 5.0},
    )


def _report(
    path: Path,
    contract: ChallengeContract,
    prompts: list[dict],
    role: str,
    manifest: ModelManifest,
    *,
    valid_video_hash: bool = False,
    checker_attack: bool = False,
) -> Path:
    path.parent.mkdir()
    video_path = path.parent / "video.mp4"
    _video(video_path, checker_attack=checker_attack)
    steps = contract.reference_steps if role == "golden" else 2
    observation = {
        "prompt_id": prompts[0]["prompt_id"],
        "seed": prompts[0]["video_seeds"][0],
        "status": "OK",
        "video": "video.mp4",
        "video_sha256": sha256_file(video_path) if valid_video_hash else "0" * 64,
        "steps": steps,
        "forward_passes": 2 * steps,
    }
    runtime = (
        {"scheduler": "original_unipc"} if role == "golden" else {"scheduler": "UniformFlowEuler"}
    )
    if role != "golden":
        observation["timing_repeats"] = [
            {"run_order": order} for order in ((0, 3) if role == "incumbent" else (1, 2))
        ]
    geometry = {"width": 32, "height": 32, "frames": 5, "fps": 5}
    path.write_text(
        json.dumps(
            {
                "schema": "forge/panel/v1",
                "contract_hash": contract.contract_hash(),
                "contract": contract.to_dict(),
                "runtime": runtime,
                "manifest_hash": manifest.fingerprint(),
                "manifest": manifest.to_dict(),
                "prompts": prompts,
                "observations": [observation],
                "geometry": geometry,
                "admission": "public_snapshot",
            }
        )
    )
    return path


def test_mismatched_video_hash_stops_before_any_luna_request(tmp_path):
    contract = ChallengeContract(
        width=32,
        height=32,
        frames=5,
        fps=5,
        reference_steps=5,
        screen_prompts=1,
        duel_prompts=1,
        confirmation_prompts=1,
    )
    prompts = [{"prompt_id": "p1", "text": "a blue cube slides", "video_seeds": [7]}]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "forge/panel-plan/v1",
                "contract": contract.to_dict(),
                "contract_hash": contract.contract_hash(),
                "prompts": prompts,
            }
        )
    )
    golden = _report(
        tmp_path / "golden/report.json", contract, prompts, "golden", _manifest(contract, 5)
    )
    incumbent = _report(
        tmp_path / "incumbent/report.json",
        contract,
        prompts,
        "incumbent",
        _manifest(contract, 2),
    )
    candidate = _report(
        tmp_path / "candidate/report.json",
        contract,
        prompts,
        "candidate",
        _manifest(contract, 1),
    )
    out = tmp_path / "quality.json"

    with pytest.raises(ValueError, match="golden video hash does not match"):
        score_reports(
            plan,
            golden,
            incumbent,
            candidate,
            out,
            ApiBudget(tmp_path / "api-budget.json", 1.0),
            None,
        )
    assert not out.exists()


def test_raster_anomaly_persists_incomplete_quality_without_luna_request(tmp_path):
    contract = ChallengeContract(
        width=32,
        height=32,
        frames=5,
        fps=5,
        reference_steps=5,
        screen_prompts=1,
        duel_prompts=1,
        confirmation_prompts=1,
    )
    prompts = [{"prompt_id": "p1", "text": "a blue cube slides", "video_seeds": [7]}]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "forge/panel-plan/v1",
                "contract": contract.to_dict(),
                "contract_hash": contract.contract_hash(),
                "prompts": prompts,
            }
        )
    )
    golden = _report(
        tmp_path / "golden/report.json",
        contract,
        prompts,
        "golden",
        _manifest(contract, 5),
        valid_video_hash=True,
    )
    incumbent = _report(
        tmp_path / "incumbent/report.json",
        contract,
        prompts,
        "incumbent",
        _manifest(contract, 2),
        valid_video_hash=True,
    )
    candidate = _report(
        tmp_path / "candidate/report.json",
        contract,
        prompts,
        "candidate",
        _manifest(contract, 2),
        valid_video_hash=True,
        checker_attack=True,
    )
    out, budget_path = tmp_path / "quality.json", tmp_path / "api-budget.json"

    evidence = score_reports(
        plan,
        golden,
        incumbent,
        candidate,
        out,
        ApiBudget(budget_path, 1.0),
        None,
    )

    assert evidence["status"] == "INCOMPLETE"
    assert evidence["reasons"] == ["RASTER_INTEGRITY_ANOMALY"]
    assert out.exists() and not budget_path.exists()


def test_quality_rejects_a_video_rendered_above_its_manifest_recipe_before_luna(tmp_path):
    contract = ChallengeContract(
        width=32,
        height=32,
        frames=5,
        fps=5,
        reference_steps=5,
        screen_prompts=1,
        duel_prompts=1,
        confirmation_prompts=1,
    )
    prompts = [{"prompt_id": "p1", "text": "a blue cube slides", "video_seeds": [7]}]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "forge/panel-plan/v1",
                "contract": contract.to_dict(),
                "contract_hash": contract.contract_hash(),
                "prompts": prompts,
            }
        )
    )
    golden = _report(
        tmp_path / "golden/report.json",
        contract,
        prompts,
        "golden",
        _manifest(contract, 5),
        valid_video_hash=True,
    )
    incumbent = _report(
        tmp_path / "incumbent/report.json",
        contract,
        prompts,
        "incumbent",
        _manifest(contract, 2),
        valid_video_hash=True,
    )
    candidate = _report(
        tmp_path / "candidate/report.json",
        contract,
        prompts,
        "candidate",
        _manifest(contract, 1),
        valid_video_hash=True,
    )
    out = tmp_path / "quality.json"

    with pytest.raises(ValueError, match="candidate quality recipe differs"):
        score_reports(
            plan,
            golden,
            incumbent,
            candidate,
            out,
            ApiBudget(tmp_path / "api-budget.json", 1.0),
            None,
        )
    assert not out.exists()
