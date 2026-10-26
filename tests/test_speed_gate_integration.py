import json

from forge.evaluation.speed_gate import compare_evidence
from forge.protocol.contract import ChallengeContract


def _sha(index):
    return f"{index:064x}"


CONTRACT = ChallengeContract(
    base_model_revision="test-revision",
    screen_prompts=2,
    duel_prompts=2,
    confirmation_prompts=2,
)
GEOMETRY = {"width": 832, "height": 480, "frames": 81, "fps": 16}


def _report(*, wall_time, steps, manifest, runtime="wan-runtime@pinned", calibrated=True):
    prompts = [
        {"prompt_id": "p1", "text": "a red sphere rolls", "video_seeds": [11, 12]},
        {"prompt_id": "p2", "text": "a horse runs", "video_seeds": [21, 22]},
    ]
    observations, quality = [], []
    index = 0
    for prompt in prompts:
        for seed in prompt["video_seeds"]:
            index += 1
            video = _sha(index)
            observations.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "seed": seed,
                    "status": "OK",
                    "video_sha256": video,
                    "wall_time_seconds": wall_time,
                    "denoise_seconds": wall_time - 0.1,
                    "peak_vram_bytes": 1_000,
                    "forward_passes": 2 * steps,
                    "steps": steps,
                }
            )
            quality.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "seed": seed,
                    "video_sha256": video,
                    "axes": {axis: 0.8 for axis in CONTRACT.axes},
                }
            )
    return {
        "schema": "forge/panel/v1",
        "contract_hash": CONTRACT.contract_hash(),
        "runtime_id": runtime,
        "hardware": {"gpu_name": "RTX 4090"},
        "geometry": GEOMETRY,
        "manifest_hash": _sha(manifest),
        "admission": "public_snapshot",
        "prompts": prompts,
        "observations": observations,
        "quality": {
            "evaluator_id": "calibrated-v1",
            "calibrated": calibrated,
            "observations": quality,
        },
    }


def _compare(reference, challenger):
    return compare_evidence(
        reference,
        challenger,
        required_contract=CONTRACT,
        expected_prompts=reference["prompts"],
        expected_runtime_id="wan-runtime@pinned",
        expected_geometry=GEOMETRY,
        expected_evaluator_id="calibrated-v1",
    )


def test_speed_gate_accepts_complete_calibrated_json_evidence(tmp_path):
    reference_path = tmp_path / "reference.json"
    challenger_path = tmp_path / "challenger.json"
    reference_path.write_text(json.dumps(_report(wall_time=4.0, steps=50, manifest=90)))
    challenger_path.write_text(json.dumps(_report(wall_time=2.0, steps=16, manifest=91)))

    result = _compare(
        json.loads(reference_path.read_text()), json.loads(challenger_path.read_text())
    )

    assert result["status"] == "PROMOTE_ELIGIBLE"
    assert result["speed"]["lower_bound"] >= 1.1


def test_speed_gate_fails_closed_for_runtime_or_uncalibrated_quality(tmp_path):
    reference_path = tmp_path / "reference.json"
    challenger_path = tmp_path / "challenger.json"
    reference_path.write_text(json.dumps(_report(wall_time=4.0, steps=50, manifest=90)))
    challenger_path.write_text(
        json.dumps(_report(wall_time=2.0, steps=16, manifest=91, runtime="other-runtime"))
    )
    runtime_mismatch = _compare(
        json.loads(reference_path.read_text()), json.loads(challenger_path.read_text())
    )
    assert runtime_mismatch["status"] == "INCOMPLETE"

    challenger_path.write_text(
        json.dumps(_report(wall_time=2.0, steps=16, manifest=91, calibrated=False))
    )
    uncalibrated = _compare(
        json.loads(reference_path.read_text()), json.loads(challenger_path.read_text())
    )
    assert uncalibrated["status"] == "INCOMPLETE"


def test_speed_gate_rejects_truncated_trusted_panel_or_missing_contract_axis(tmp_path):
    reference = _report(wall_time=4.0, steps=50, manifest=90)
    challenger = _report(wall_time=2.0, steps=16, manifest=91)
    challenger["observations"] = challenger["observations"][:-1]
    truncated_path = tmp_path / "truncated.json"
    truncated_path.write_text(json.dumps(challenger))
    truncated = _compare(reference, json.loads(truncated_path.read_text()))
    assert truncated["status"] == "INCOMPLETE"

    challenger = _report(wall_time=2.0, steps=16, manifest=91)
    del challenger["quality"]["observations"][0]["axes"]["motion_adherence"]
    axes_missing = _compare(reference, challenger)
    assert axes_missing["status"] == "INCOMPLETE"


def test_speed_gate_rejects_private_or_exact_copy_candidates():
    reference = _report(wall_time=4.0, steps=50, manifest=90)
    private = _report(wall_time=2.0, steps=16, manifest=91)
    private["admission"] = "private_experiment"
    assert _compare(reference, private)["status"] == "REJECT"

    copied = _report(wall_time=2.0, steps=16, manifest=90)
    assert _compare(reference, copied)["status"] == "REJECT"
