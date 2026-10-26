"""Filesystem/CLI contract for the provisional launch policy, without GPU inference."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SHA = "a" * 64


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def cli(*args, ok=True):
    run = subprocess.run(
        [sys.executable, "-m", "forge.evaluation.launch_policy", *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if ok:
        assert run.returncode == 0, run.stderr
    else:
        assert run.returncode != 0
    return run


def config():
    models = {"candidate": "c" * 64, "king": "9" * 64, "anchor": "b" * 64}
    prompts = []
    for number in range(16):
        tags = []
        if number == 0:
            tags = ["motion_required"]
        if number == 1:
            tags = ["steady_lighting"]
        if number == 2:
            tags = ["continuous_motion"]
        prompts.append(
            {
                "prompt_id": f"p{number}",
                "text": f"scene {number}",
                "seed": number,
                "kind": "motion" if number < 8 else "static",
                "tags": tags,
            }
        )
    return {
        "models": models,
        "evaluator": {"evaluator_id": "d" * 64, "spec_hash": "e" * 64},
        "runtime": {
            "runtime_id": "f" * 64,
            "hardware_id": "1" * 64,
            "dependency_lock_hash": "2" * 64,
        },
        "geometry": {"width": 832, "height": 480, "frames": 49, "fps": 16},
        "prompts": prompts,
        "timing_prompt_ids": [f"p{i}" for i in range(8)],
        "assignments": [{"prompt_id": f"p{i}", **models} for i in range(16)],
    }


def rows(plan):
    output = []
    for prompt in plan["prompts"]:
        candidate = {
            "video_sha256": ("3" + format(prompt["seed"], "063x"))[-64:],
            "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
            "diagnostics": {
                "frozen_fraction": 0.0,
                "luma_acceleration_median": 1.0,
                "centered_transition_median": 1.0,
                "temporal_delta": 0.02,
            },
        }
        for reference in ("king", "anchor"):
            output.append(
                {
                    "type": "quality",
                    "plan_id": plan["plan_id"],
                    "prompt_id": prompt["prompt_id"],
                    "reference": reference,
                    "model_ids": {
                        "reference": plan["models"][reference],
                        "candidate": plan["models"]["candidate"],
                    },
                    "evaluator_id": plan["evaluator"]["evaluator_id"],
                    "geometry": plan["geometry"],
                    "reference_result": {
                        "video_sha256": SHA,
                        "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                        "diagnostics": {
                            "frozen_fraction": 0.0,
                            "luma_acceleration_median": 1.0,
                            "centered_transition_median": 1.0,
                            "temporal_delta": 0.02,
                        },
                    },
                    "candidate_result": candidate,
                }
            )
    for number, prompt_id in enumerate(plan["timing_prompt_ids"]):
        output.append(
            {
                "type": "timing",
                "plan_id": plan["plan_id"],
                "prompt_id": prompt_id,
                "model_ids": {
                    "reference": plan["models"]["king"],
                    "candidate": plan["models"]["candidate"],
                },
                "runtime_id": plan["runtime"]["runtime_id"],
                "hardware_id": plan["runtime"]["hardware_id"],
                "geometry": plan["geometry"],
                "times": [12 + number / 10, 10, 10.1, 12.1 + number / 10],
            }
        )
    return output


def test_launch_policy_cli_freezes_and_evaluates_fail_closed(tmp_path):
    source, frozen = tmp_path / "config.json", tmp_path / "plan.json"
    put(source, config())
    cli("freeze", "--config", source, "--out", frozen)
    plan = json.loads(frozen.read_text())
    assert plan["policy"]["status"] == "PROVISIONAL_NOT_PERCEPTUALLY_CALIBRATED"
    assert len(plan["prompts"]) == 16 and len(plan["timing_prompt_ids"]) == 8
    input_rows, result_path = tmp_path / "rows.json", tmp_path / "result.json"
    good_rows = rows(plan)
    put(input_rows, {"rows": good_rows})
    cli("evaluate", "--plan", frozen, "--rows", input_rows, "--out", result_path)
    result = json.loads(result_path.read_text())
    assert result["status"] == "PROMOTE_ELIGIBLE"
    assert result["weights"] == "NOT_ACTIVATED"
    assert set(result["quality"]) == {"king", "anchor"}
    assert result["speed"]["lower_ci"] >= 1.10
    # Receipt names are write-once, so a rerun cannot replace accepted evidence.
    original = result_path.read_bytes()
    cli("evaluate", "--plan", frozen, "--rows", input_rows, "--out", result_path, ok=False)
    assert result_path.read_bytes() == original

    # Mean speed can pass while its one-sided bound remains too uncertain to promote.
    variable_speed = json.loads(json.dumps(good_rows))
    for index, row in enumerate(item for item in variable_speed if item["type"] == "timing"):
        row["times"] = [14, 10, 10, 14] if index % 2 == 0 else [10, 10, 10, 10]
    variable_input = tmp_path / "variable-speed.json"
    variable_result = tmp_path / "variable-speed-result.json"
    put(variable_input, variable_speed)
    cli("evaluate", "--plan", frozen, "--rows", variable_input, "--out", variable_result)
    uncertain = json.loads(variable_result.read_text())
    assert uncertain["status"] == "INCONCLUSIVE"
    assert uncertain["speed"]["ratio"] > 1.10 > uncertain["speed"]["lower_ci"]

    # A loss against the anchor cannot be hidden by passing the king comparison.
    anchor_loss = json.loads(json.dumps(good_rows))
    for row in anchor_loss:
        if row["type"] != "quality":
            continue
        row["candidate_result"]["scores"]["MQ"] = -1.0
        if row["reference"] == "king":
            row["reference_result"]["scores"]["MQ"] = -1.0
    anchor_input, anchor_result = tmp_path / "anchor.json", tmp_path / "anchor-result.json"
    put(anchor_input, anchor_loss)
    cli("evaluate", "--plan", frozen, "--rows", anchor_input, "--out", anchor_result)
    assert json.loads(anchor_result.read_text())["status"] == "FAIL"

    # Freeze is a known temporal violation only for its explicitly tagged prompt.
    frozen_video = json.loads(json.dumps(good_rows))
    for row in frozen_video:
        if row["type"] == "quality" and row["prompt_id"] == "p0":
            row["candidate_result"]["diagnostics"]["frozen_fraction"] = 0.96
    temporal_input, temporal_result = tmp_path / "temporal.json", tmp_path / "temporal-result.json"
    put(temporal_input, frozen_video)
    cli("evaluate", "--plan", frozen, "--rows", temporal_input, "--out", temporal_result)
    temporal = json.loads(temporal_result.read_text())
    assert temporal["status"] == "FAIL"
    assert {item["violation"] for item in temporal["temporal_violations"]} == {"exact_freeze"}

    # Captured codec-noise near-freeze: frozen_fraction can remain zero.
    near_freeze = json.loads(json.dumps(good_rows))
    for row in near_freeze:
        if row["type"] == "quality" and row["prompt_id"] == "p0":
            row["reference_result"]["diagnostics"]["temporal_delta"] = 0.02903127856552601
            row["candidate_result"]["diagnostics"]["temporal_delta"] = 1.2503315701906104e-05
    near_input, near_result = tmp_path / "near.json", tmp_path / "near-result.json"
    put(near_input, near_freeze)
    cli("evaluate", "--plan", frozen, "--rows", near_input, "--out", near_result)
    assert {
        item["violation"] for item in json.loads(near_result.read_text())["temporal_violations"]
    } == {"near_freeze"}

    # Missing or tampered evidence is INCOMPLETE, never a qualified promotion.
    incomplete_input = tmp_path / "incomplete.json"
    incomplete_result = tmp_path / "incomplete-result.json"
    put(incomplete_input, good_rows[:-1])
    cli("evaluate", "--plan", frozen, "--rows", incomplete_input, "--out", incomplete_result)
    assert json.loads(incomplete_result.read_text())["status"] == "INCOMPLETE"
    missing_diagnostic = json.loads(json.dumps(good_rows))
    del next(
        row for row in missing_diagnostic if row["type"] == "quality" and row["prompt_id"] == "p0"
    )["candidate_result"]["diagnostics"]["frozen_fraction"]
    diagnostic_input = tmp_path / "missing-diagnostic.json"
    diagnostic_result = tmp_path / "missing-diagnostic-result.json"
    put(diagnostic_input, missing_diagnostic)
    cli("evaluate", "--plan", frozen, "--rows", diagnostic_input, "--out", diagnostic_result)
    assert json.loads(diagnostic_result.read_text())["status"] == "INCOMPLETE"
    tampered = json.loads(json.dumps(good_rows))
    tampered[0]["plan_id"] = "0" * 64
    tampered_input, tampered_result = tmp_path / "tampered.json", tmp_path / "tampered-result.json"
    put(tampered_input, tampered)
    cli("evaluate", "--plan", frozen, "--rows", tampered_input, "--out", tampered_result)
    assert json.loads(tampered_result.read_text())["status"] == "INCOMPLETE"
