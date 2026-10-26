import json

from forge.cli import main


def test_demo_runs_twice_without_duplicating_crowns(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out)]) == 0
    first = json.loads((out / "summary.json").read_text())
    assert main(["demo", "--out", str(out)]) == 0
    second = json.loads((out / "summary.json").read_text())

    assert first["local_verdicts"]["D"] == "KEEP"
    assert first["decision"] == "PROMOTE"
    assert len(set(first["state_hash_by_validator"].values())) == 1
    assert second["rerun"] is True
    assert len(second["crowns"]) == 1
    assert second["state_hash_by_validator"] == first["state_hash_by_validator"]
    assert second["hotkey_weights"] == {"miner-good": 1.0}


def test_real_panel_plan_roundtrip_and_incomplete_quality_is_not_a_winner(tmp_path):
    from forge.protocol.contract import ChallengeContract

    contract = ChallengeContract(screen_prompts=2, screen_seeds=1)
    contract_path = tmp_path / "contract.json"
    plan_path = tmp_path / "plan.json"
    contract_path.write_text(json.dumps(contract.to_dict()))
    assert (
        main(
            [
                "panel-plan",
                "--contract",
                str(contract_path),
                "--round-id",
                "r123",
                "--stage",
                "screen",
                "--block-hash",
                "finalized-test-entropy",
                "--out",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text())
    assert len(plan["prompts"]) == 2
    assert plan["contract_hash"] == contract.contract_hash()
    assert ChallengeContract.from_dict(plan["contract"]) == contract

    report = {
        "schema": "forge/panel/v1",
        "contract_hash": contract.contract_hash(),
        "runtime_id": "1" * 64,
        "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
        "hardware": {"gpu_name": "test device"},
        "admission": "public_snapshot",
        "manifest_hash": "2" * 64,
        "prompts": plan["prompts"],
        "quality": None,
        "observations": [
            {
                "prompt_id": p["prompt_id"],
                "seed": p["video_seeds"][0],
                "status": "OK",
                "video_sha256": "3" * 64,
                "wall_time_seconds": 2.0,
                "denoise_seconds": 1.0,
                "peak_vram_bytes": 1000,
                "steps": 16,
                "forward_passes": 32,
            }
            for p in plan["prompts"]
        ],
    }
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(json.dumps(report))
    report["manifest_hash"] = "4" * 64
    candidate.write_text(json.dumps(report))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"runtime_id": "1" * 64, "evaluator_id": "trusted-judge-v1"}))
    decision = tmp_path / "decision.json"
    assert (
        main(
            [
                "compare-panels",
                "--plan",
                str(plan_path),
                "--policy",
                str(policy),
                "--golden",
                str(reference),
                "--incumbent",
                str(reference),
                "--challenger",
                str(candidate),
                "--out",
                str(decision),
            ]
        )
        == 2
    )
    result = json.loads(decision.read_text())
    assert result["status"] == "NOT_ELIGIBLE"
    assert all(c["status"] == "INCOMPLETE" for c in result["comparisons"].values())


def test_simulated_testnet_round_cannot_request_weight_submission(tmp_path):
    import pytest

    with pytest.raises(SystemExit) as error:
        main(
            [
                "testnet-round",
                "--netuid",
                "578",
                "--wallet",
                "unused",
                "--state-dir",
                str(tmp_path / "state"),
                "--set-weights",
            ]
        )
    assert error.value.code == 2


def test_preflight_round_checks_real_cache_capacity_before_any_download(tmp_path):
    import os
    from pathlib import Path

    import pytest

    from forge.models.artifact_manifest import (
        ArtifactError,
        ArtifactReference,
        ManifestFile,
        ModelManifest,
    )
    from forge.protocol.contract import ChallengeContract
    from forge.validator.validator_cli import PREFLIGHT_DISK_FLOOR_BYTES

    # `/dev/shm` is a real small filesystem in the Linux test environment.  This
    # test does not fake `statvfs`: its declared closed batch intentionally
    # exceeds that filesystem, so the CLI must reject it before opening a
    # network download or writing any cache file.
    cache_parent = Path("/dev/shm")
    if not cache_parent.is_dir():
        pytest.skip("no real small tmpfs is available")
    available = os.statvfs(cache_parent).f_bavail * os.statvfs(cache_parent).f_frsize
    worst_case_required = 136 * 1024**3
    if available >= worst_case_required:
        pytest.skip("tmpfs has enough space for the bounded worst-case batch")

    contract = ChallengeContract(screen_prompts=2, screen_seeds=1, max_candidates=16)
    contract_path = tmp_path / "contract.json"
    plan_path = tmp_path / "plan.json"
    contract_path.write_text(json.dumps(contract.to_dict()))
    assert (
        main(
            [
                "panel-plan",
                "--contract",
                str(contract_path),
                "--round-id",
                "capacity",
                "--stage",
                "screen",
                "--block-hash",
                "finalized-capacity-test",
                "--out",
                str(plan_path),
            ]
        )
        == 0
    )
    base_reference = ArtifactReference("hf", "public/base", "b" * 40)
    six_gib = 6 * 1024**3
    base = ModelManifest(
        base_reference,
        "pipeline",
        tuple(
            ManifestFile(
                f"transformer/model-{index}.safetensors",
                f"{index:064x}",
                six_gib if index < 5 else 2 * 1024**3,
            )
            for index in range(6)
        ),
        "apache-2.0",
        {"steps": 16, "guidance_scale": 5.0},
    )
    candidates = tuple(
        ModelManifest(
            ArtifactReference("hf", f"public/candidate{index}", f"{index + 1:040x}"),
            "full_transformer",
            (ManifestFile("transformer/model.safetensors", f"{index + 16:064x}", six_gib),),
            "apache-2.0",
            {"steps": 16, "guidance_scale": 5.0},
            base_reference,
        )
        for index in range(16)
    )
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base.to_dict()))
    plan = json.loads(plan_path.read_text())
    plan["admitted"] = [
        {
            "manifest_hash": candidate.fingerprint(),
            "manifest": candidate.to_dict(),
            "block": 10 + index,
            "hotkey": f"candidate-{index}",
        }
        for index, candidate in enumerate(candidates)
    ]
    plan_path.write_text(json.dumps(plan))
    cache_dir = cache_parent / f"forge-capacity-{tmp_path.name}"
    try:
        with pytest.raises(ArtifactError, match="insufficient free space"):
            main(
                [
                    "preflight-round",
                    "--plan",
                    str(plan_path),
                    "--base-manifest",
                    str(base_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--out",
                    str(tmp_path / "preflight.json"),
                ]
            )
        assert cache_dir.is_dir()
        assert not list(cache_dir.iterdir())
        assert not (tmp_path / "preflight.json").exists()
        assert available < worst_case_required
        assert PREFLIGHT_DISK_FLOOR_BYTES == 2 * 1024**3
    finally:
        cache_dir.rmdir()
