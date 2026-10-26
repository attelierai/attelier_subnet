import json
from dataclasses import asdict

import numpy as np

from forge.evaluation.automatic_evidence import AXES
from forge.evaluation.automatic_gate import evaluate_automatic
from forge.evaluation.benchmark import generate_prompts
from forge.evaluation.raster_integrity import RULE_ID as RASTER_INTEGRITY_RULE_ID
from forge.evaluation.raster_integrity import check_frames
from forge.evaluation.vision_calibration import (
    CONTROL_AXES,
    CONTROL_RULE,
    CONTROL_RULE_V1,
    calibrate_controls,
)
from forge.evaluation.vision_judge import evaluator_id, evaluator_spec
from forge.evaluation.vision_media import select_transition_frames
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.protocol.prompt_history import genesis_checkpoint
from forge.protocol.prompt_pack import (
    PACK_SOURCE,
    build_pack,
    pack_context,
    seal_pack,
    verified_plan_prompts,
)
from forge.validator.runtime_preflight import PASS, SCHEMA


def _sha(number):
    return f"{number:064x}"


def _detail_crops(contract):
    width, height = contract.width // 4, contract.height // 4
    indices = [
        i * contract.frames // min(contract.frames, 9) for i in range(min(contract.frames, 9))
    ]
    return [
        {"frame_index": index, "x": 0, "y": 0, "width": width, "height": height}
        for index in indices[:3]
    ]


def _clear_raster_check(video_sha256, geometry):
    frames = np.zeros(
        (geometry["frames"], geometry["height"], geometry["width"], 3), dtype=np.uint8
    )
    return check_frames(frames, video_sha256)


def _checker_raster_check(video_sha256, geometry):
    frames = np.full(
        (geometry["frames"], geometry["height"], geometry["width"], 3), 128, dtype=np.uint8
    )
    checker = (np.indices((geometry["height"], geometry["width"])).sum(axis=0) % 2 * 2 - 1).astype(
        np.int16
    ) * 64
    frames[1:] = np.clip(128 + checker[..., None], 0, 255).astype(np.uint8)
    return check_frames(frames, video_sha256)


def _preflight(plan):
    candidate = plan["candidate"]["manifest_hash"]
    return {
        "schema": SCHEMA,
        "plan_hash": hash_canonical("forge/runtime-preflight-plan/v1", plan),
        "contract_hash": plan["contract_hash"],
        "base_manifest_hash": _sha(700),
        "candidates": [
            {
                "manifest_hash": candidate,
                "weight_content_hash": _sha(701),
                "effective_candidate_hash": _sha(702),
                "status": PASS,
            }
        ],
    }


def _history_checkpoint(plan):
    return genesis_checkpoint(
        plan["network"],
        plan["netuid"],
        certified_state_hash=plan["incumbent"]["certified_state_hash"],
    )


def _evidence(prompt_count=1024, quality_delta=0.0):
    contract = ChallengeContract(
        width=32,
        height=32,
        frames=5,
        fps=5,
        confirmation_prompts=prompt_count * 4,
        confirmation_seeds=1,
        grammar_version="forge/bench/grammar/v3",
    )
    geometry = {"height": 32, "width": 32, "frames": 5, "fps": 5}
    contract_hash = contract.contract_hash()
    round_id = "confirmation-round-1"
    entropy = _sha(99)
    generated = [
        asdict(item) for item in generate_prompts(contract, round_id, "confirmation", entropy)
    ]
    pack = build_pack(
        pack_context("test", 9, round_id, contract_hash, 100),
        "a" * 64,
        [
            {
                "stage": "confirmation",
                "text": prompt["text"],
                "properties": prompt["properties"],
                "rubric": list(prompt["rubric"]),
                "source_group": prompt["prompt_id"],
            }
            for prompt in generated
        ],
    )
    candidate_hash, incumbent_hash, golden_hash = _sha(101), _sha(102), _sha(103)
    plan = {
        "contract_hash": contract_hash,
        "geometry": geometry,
        "stage": "confirmation",
        "round_id": round_id,
        "entropy": entropy,
        "contract": contract.to_dict(),
        "network": "test",
        "netuid": 9,
        "prompt_source": PACK_SOURCE,
        "prompt_pack": pack,
        "pack_commitment": {"hash": seal_pack(pack), "block": 100, "hotkey": "validator-hotkey"},
        "close_block": 100,
        "entropy_block": 101,
        "pack_reveal_block": 101,
        "finalist_lock_block": 100,
        "admitted": [
            {
                "manifest_hash": candidate_hash,
                "block": 99,
                "hotkey": "candidate-hotkey",
            }
        ],
        "candidate": {
            "manifest_hash": candidate_hash,
            "hotkey": "candidate-hotkey",
            "admission_block": 99,
        },
        "incumbent": {
            "manifest_hash": incumbent_hash,
            "certified_state_hash": _sha(104),
            "certified_activation_block": 98,
        },
    }
    plan["prompts"] = verified_plan_prompts(plan, contract)
    checkpoint = genesis_checkpoint(
        "test", 9, certified_state_hash=plan["incumbent"]["certified_state_hash"]
    )
    plan["prompt_history"] = {
        "schema": "forge/prompt-history-bundle/v1",
        "checkpoint_root": checkpoint["root"],
        "links": [],
    }
    plan["prompt_history_root"] = checkpoint["root"]
    calibration_plan = {
        "schema": "forge/panel-plan/v1",
        "purpose": "calibration",
        "contract": contract.to_dict(),
        "contract_hash": contract_hash,
        "round_id": "calibration-round-1",
        "campaign_id": round_id,
        "stage": "duel",
        "entropy": _sha(98),
        "prompts": plan["prompts"],
    }
    plan["calibration_plan_hash"] = hash_canonical("forge/evaluation-plan/v1", calibration_plan)
    prompts = plan["prompts"]
    policy = {
        "policy_id": "",
        "evaluator_id": evaluator_id(),
        "min_prompts": prompt_count,
        "min_slice_prompts": prompt_count,
        "quality_margin": 0.02,
        "min_speedup": 1.1,
        "alpha": 0.05,
        "max_candidates": 16,
        "max_order_disagreement": 0.25,
        "quality_aggregation": "symmetric_mean_prompt_clustered",
    }
    policy["policy_id"] = hash_canonical(
        "forge/auto-policy/v3", {key: value for key, value in policy.items() if key != "policy_id"}
    )
    panel_hash = hash_canonical("forge/panel-prompts/v1", prompts)

    def observations(reference_offset, candidate_offset):
        score_a, score_b = (
            (2, 2)
            if quality_delta == 0
            else (max(0, int(round(-4 * quality_delta))), max(0, int(round(4 * quality_delta))))
        )
        return [
            {
                "prompt_id": prompt["prompt_id"],
                "seed": prompt["video_seeds"][0],
                "reference_video_sha256": _sha(reference_offset),
                "candidate_video_sha256": _sha(candidate_offset),
                "detail_frame_indices": [
                    i * contract.frames // min(contract.frames, 9)
                    for i in range(min(contract.frames, 9))
                ],
                "detail_crop_regions": _detail_crops(contract),
                "transition_frame_indices": list(select_transition_frames(contract.frames)),
                "axes": {axis: quality_delta for axis in AXES},
                "order_disagreement": 0.0,
                "status": "OK",
                "calls": [
                    {
                        "answer": {
                            "axes": {
                                axis: {"a": score_a, "b": score_b, "reason": "visible"}
                                for axis in AXES
                            },
                            "uncertain_axes": [],
                        },
                        "reference_is_a": True,
                        "model": "gpt-6-luna",
                        "response_id": f"resp-a-{reference_offset}-{index}",
                        "request_image_hash": _sha(reference_offset + candidate_offset + index),
                        "input_image_hashes": {
                            "A": [_sha(reference_offset)] * 6,
                            "B": [_sha(candidate_offset)] * 6,
                        },
                    },
                    {
                        "answer": {
                            "axes": {
                                axis: {"a": score_b, "b": score_a, "reason": "visible"}
                                for axis in AXES
                            },
                            "uncertain_axes": [],
                        },
                        "reference_is_a": False,
                        "model": "gpt-6-luna",
                        "response_id": f"resp-b-{reference_offset}-{index}",
                        "request_image_hash": _sha(
                            reference_offset + candidate_offset + index + 1_000_000
                        ),
                        "input_image_hashes": {
                            "A": [_sha(candidate_offset)] * 6,
                            "B": [_sha(reference_offset)] * 6,
                        },
                    },
                ],
            }
            for index, prompt in enumerate(prompts)
        ]

    incumbent_observations = observations(10_000, 20_000)
    quality = {
        "schema": "forge/automatic-quality/v1",
        "evaluator_id": policy["evaluator_id"],
        "spec": evaluator_spec(),
        "panel_hash": panel_hash,
        "campaign_id": plan["round_id"],
        "created_unix": 2.0,
        "completed_unix": 3.0,
        "contract_hash": contract_hash,
        "geometry": geometry,
        "comparisons": [
            {
                "reference_role": "golden",
                "reference_manifest_hash": golden_hash,
                "candidate_manifest_hash": candidate_hash,
                "reference_admission": "public_snapshot",
                "candidate_admission": "public_snapshot",
                "observations": observations(30_000, 20_000),
            },
            {
                "reference_role": "incumbent",
                "reference_manifest_hash": incumbent_hash,
                "candidate_manifest_hash": candidate_hash,
                "reference_admission": "public_snapshot",
                "candidate_admission": "public_snapshot",
                "observations": incumbent_observations,
            },
        ],
    }
    quality["raster_integrity_rule_id"] = RASTER_INTEGRITY_RULE_ID
    quality["raster_checks"] = {
        video_sha256: _clear_raster_check(video_sha256, geometry)
        for video_sha256 in {_sha(10_000), _sha(20_000), _sha(30_000)}
    }
    measurements = []
    reference_videos, candidate_videos = {}, {}
    for index, prompt in enumerate(prompts):
        prompt_id, seed = prompt["prompt_id"], prompt["video_seeds"][0]
        pair_id = f"{prompt_id}:{seed}"
        reference_videos[pair_id] = incumbent_observations[index]["reference_video_sha256"]
        candidate_videos[pair_id] = incumbent_observations[index]["candidate_video_sha256"]
        for side, repeat, order, wall in (
            ("reference", 0, 0, 4.0),
            ("candidate", 0, 1, 2.0),
            ("candidate", 1, 2, 2.0),
            ("reference", 1, 3, 4.0),
        ):
            measurements.append(
                {
                    "status": "OK",
                    "pair_id": pair_id,
                    "pair_index": index,
                    "side": side,
                    "run_order": order,
                    "repeat_index": repeat,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "wall_time_seconds": wall,
                    "peak_vram_bytes": 1_000,
                    "steps": 16,
                    "cfg_enabled": True,
                    "forward_passes": 32,
                    "expected_forward_passes": 32,
                }
            )
    timing = {
        "schema": "forge/paired-timing/v1",
        "contract_hash": contract_hash,
        "panel_hash": hash_canonical("forge/panel/v1", contract_hash, prompts, geometry),
        "runtime_id": _sha(55),
        "hardware": {"gpu_name": "H100"},
        "geometry": geometry,
        "design": {
            "order": ["reference", "candidate", "candidate", "reference"],
            "warmups_per_side": 1,
        },
        "reference_manifest_hash": incumbent_hash,
        "candidate_manifest_hash": candidate_hash,
        "reference_admission": "public_snapshot",
        "candidate_admission": "public_snapshot",
        "report_video_sha256": {"reference": reference_videos, "candidate": candidate_videos},
        "measurements": measurements,
    }
    calibration_input = {
        "schema": "forge/auto-calibration/v1",
        "evaluator_id": policy["evaluator_id"],
        "evaluator": evaluator_spec(),
        "contract_hash": contract_hash,
        "geometry": geometry,
        "policy_id": policy["policy_id"],
        "campaign_id": plan["round_id"],
        "calibration_plan_hash": plan["calibration_plan_hash"],
        "panel_hash": _sha(44),
        "created_unix": 1.0,
        "completed_unix": 1.5,
        "control_rule": CONTROL_RULE,
        "controls": {},
    }
    for kind, affected_axes in CONTROL_AXES.items():
        rows = []
        for index in range(32):
            delta = -0.25 if kind != "self" else 0.0
            scores_a = {
                axis: (2 if delta == 0 or axis not in affected_axes else 2) for axis in AXES
            }
            scores_b = {
                axis: (2 if delta == 0 or axis not in affected_axes else 1) for axis in AXES
            }
            rows.append(
                {
                    "prompt_id": f"control-{kind}-{index}",
                    "reference_video_sha256": _sha(30_000),
                    "detail_frame_indices": [
                        i * contract.frames // min(contract.frames, 9)
                        for i in range(min(contract.frames, 9))
                    ],
                    "detail_crop_regions": _detail_crops(contract),
                    "transition_frame_indices": list(select_transition_frames(contract.frames)),
                    "status": "OK",
                    "axes": {axis: delta if axis in affected_axes else 0.0 for axis in AXES},
                    "order_disagreement": 0.0,
                    "calls": [
                        {
                            "reference_is_a": True,
                            "model": "gpt-6-luna",
                            "response_id": f"control-{kind}-{index}-a",
                            "answer": {
                                "axes": {
                                    axis: {
                                        "a": scores_a[axis],
                                        "b": scores_b[axis],
                                        "reason": "control",
                                    }
                                    for axis in AXES
                                },
                                "uncertain_axes": [],
                            },
                        },
                        {
                            "reference_is_a": False,
                            "model": "gpt-6-luna",
                            "response_id": f"control-{kind}-{index}-b",
                            "answer": {
                                "axes": {
                                    axis: {
                                        "a": scores_b[axis],
                                        "b": scores_a[axis],
                                        "reason": "control",
                                    }
                                    for axis in AXES
                                },
                                "uncertain_axes": [],
                            },
                        },
                    ],
                }
            )
        calibration_input["controls"][kind] = rows
    calibration = calibrate_controls(calibration_input)
    calibration["raster_integrity_rule_id"] = RASTER_INTEGRITY_RULE_ID
    calibration["raster_integrity_status"] = "CLEAR"
    calibration["raster_checks"] = {_sha(30_000): _clear_raster_check(_sha(30_000), geometry)}
    calibration = calibrate_controls(calibration)
    return plan, policy, quality, timing, calibration


def test_automatic_gate_promotes_complete_public_confirmation_evidence(tmp_path):
    evidence_path = tmp_path / "automatic-evidence.json"
    evidence_path.write_text(json.dumps(_evidence()))
    plan, policy, quality, timing, calibration = json.loads(evidence_path.read_text())

    preflight = _preflight(plan)
    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=preflight,
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "PROMOTE_ELIGIBLE"
    assert result["timing"]["lower_speedup"] >= policy["min_speedup"]
    assert result["precision"]["zero_ties_prompts_needed_per_slice"] > 512

    for measurement in timing["measurements"]:
        measurement.update(cfg_enabled=False, forward_passes=16, expected_forward_passes=16)
    assert (
        evaluate_automatic(
            plan,
            policy,
            quality,
            timing,
            calibration,
            preflight=preflight,
            prompt_history_checkpoint=_history_checkpoint(plan),
        )["status"]
        == "PROMOTE_ELIGIBLE"
    )

    evidence = _evidence(prompt_count=128, quality_delta=-0.5)
    assert (
        evaluate_automatic(
            *evidence,
            preflight=_preflight(evidence[0]),
            prompt_history_checkpoint=_history_checkpoint(evidence[0]),
        )["status"]
        == "FAIL"
    )


def test_v3_keeps_one_order_only_optimism_as_explicit_prompt_clustered_diagnostic():
    plan, policy, quality, timing, calibration = _evidence()
    for comparison in quality["comparisons"]:
        for observation in comparison["observations"]:
            for call in observation["calls"]:
                for axis in AXES:
                    call["answer"]["axes"][axis].update(a=0, b=4)
            observation["axes"] = {axis: 0.0 for axis in AXES}
            observation["order_disagreement"] = 2.0

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "PROMOTE_ELIGIBLE"
    assert result["quality_diagnostics"]["golden"]["order_sign_flip_rate"] == 1.0
    assert all(
        value == 2.0
        for value in result["quality_diagnostics"]["golden"]["order_disagreement_by_axis"].values()
    )


def test_exact_per_side_paired_runtime_receipt_flows_through_the_automatic_gate():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    timing["runtime"] = {
        "scheduler": "paired-exact-recipes",
        "schedulers": {"reference": "uniform_euler", "candidate": "uniform_euler"},
        "timing_scope": {
            "competitive_includes": ["text_encoding", "denoising", "vae_decoding"],
            "competitive_excludes": ["model_loading"],
        },
    }
    timing["recipes"] = {
        "reference": {"steps": 16, "guidance_scale": 5.0, "kind": "uniform_euler"},
        "candidate": {"steps": 16, "guidance_scale": 5.0, "kind": "uniform_euler"},
    }
    for measurement in timing["measurements"]:
        measurement["full_pipeline_seconds"] = measurement["wall_time_seconds"]

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    # The 128-prompt fixture intentionally has insufficient statistical power;
    # reaching INCONCLUSIVE proves the exact runtime receipt passed parsing.
    assert result["status"] == "INCONCLUSIVE"


def test_production_rejects_calibration_selected_from_another_plan():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    other_calibration_plan = {
        "schema": "forge/panel-plan/v1",
        "purpose": "calibration",
        "campaign_id": plan["round_id"],
        "entropy": _sha(97),
    }
    calibration_input = {
        key: value
        for key, value in calibration.items()
        if key not in {"status", "summary", "reasons", "calibration_id"}
    }
    calibration_input["calibration_plan_hash"] = hash_canonical(
        "forge/evaluation-plan/v1", other_calibration_plan
    )
    calibration = calibrate_controls(calibration_input)

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["reasons"] == ["calibration does not match the declared calibration plan"]


def test_production_rejects_fixed_first_middle_last_spatial_evidence():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    quality["comparisons"][0]["observations"][0]["detail_frame_indices"] = [0, 40, 80]

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["reasons"] == ["native-frame selection does not cover every temporal stratum"]


def test_production_rejects_a_bound_raster_anomaly():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    candidate_hash = _sha(20_000)
    quality["raster_checks"][candidate_hash] = _checker_raster_check(
        candidate_hash, plan["geometry"]
    )

    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=_history_checkpoint(plan),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["reasons"] == ["RASTER_INTEGRITY_ANOMALY"]


def test_development_v1_keeps_its_historical_disagreement_hard_gate():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    policy.pop("quality_aggregation")
    policy["policy_id"] = hash_canonical(
        "forge/auto-policy/v1", {key: value for key, value in policy.items() if key != "policy_id"}
    )
    calibration = calibrate_controls(
        {
            key: value
            for key, value in calibration.items()
            if key not in {"status", "summary", "reasons", "calibration_id"}
        }
        | {"policy_id": policy["policy_id"], "control_rule": CONTROL_RULE_V1}
    )
    observation = quality["comparisons"][0]["observations"][0]
    for axis in AXES:
        observation["calls"][0]["answer"]["axes"][axis].update(a=1, b=3)
    observation["axes"] = {axis: 0.25 for axis in AXES}
    observation["order_disagreement"] = 0.5

    assert (
        evaluate_automatic(plan, policy, quality, timing, calibration, development=True)["status"]
        == "INCOMPLETE"
    )


def test_automatic_gate_fails_closed_on_truncated_panel_or_private_artifact(tmp_path):
    evidence_path = tmp_path / "truncated-evidence.json"
    evidence_path.write_text(json.dumps(_evidence()))
    plan, policy, quality, timing, calibration = json.loads(evidence_path.read_text())
    quality["comparisons"][0]["observations"].pop()
    assert evaluate_automatic(plan, policy, quality, timing, calibration)["status"] == "INCOMPLETE"

    plan, policy, quality, timing, calibration = _evidence()
    timing["candidate_admission"] = "private_experiment"
    assert evaluate_automatic(plan, policy, quality, timing, calibration)["status"] == "INCOMPLETE"

    plan, policy, quality, timing, calibration = _evidence()
    plan["prompts"][0]["text"] = "tampered panel prompt"
    assert evaluate_automatic(plan, policy, quality, timing, calibration)["status"] == "INCOMPLETE"

    plan, policy, quality, timing, calibration = _evidence()
    del plan["prompt_source"]
    assert evaluate_automatic(plan, policy, quality, timing, calibration)["status"] == "INCOMPLETE"

    plan, policy, quality, timing, calibration = _evidence()
    calibration["panel_hash"] = quality["panel_hash"]
    calibration = calibrate_controls(
        {
            key: value
            for key, value in calibration.items()
            if key not in {"status", "summary", "reasons", "calibration_id"}
        }
    )
    assert evaluate_automatic(plan, policy, quality, timing, calibration)["status"] == "INCOMPLETE"


def test_v3_calibration_refuses_noninformative_identical_temporal_transformations():
    _, _, _, _, calibration = _evidence(prompt_count=128)
    source = calibration["controls"]["freeze"][0]
    source["reference_pixels_sha256"] = _sha(99_999)
    source["control_pixels_sha256"] = _sha(99_999)
    rebuilt = calibrate_controls(
        {
            key: value
            for key, value in calibration.items()
            if key not in {"status", "summary", "reasons", "calibration_id"}
        }
    )
    assert rebuilt["summary"]["freeze"]["noninformative"] == 1
    assert rebuilt["summary"]["freeze"]["informative_prompts"] == 31
    assert rebuilt["status"] == "INCONCLUSIVE"


def test_production_fails_closed_without_an_external_history_checkpoint():
    plan, policy, quality, timing, calibration = _evidence(prompt_count=128)
    result = evaluate_automatic(
        plan, policy, quality, timing, calibration, preflight=_preflight(plan)
    )
    assert result == {
        "status": "INCOMPLETE",
        "reasons": ["independently certified prompt history checkpoint is required"],
    }

    checkpoint = _history_checkpoint(plan)
    checkpoint["certified_state_hash"] = _sha(123_456)
    result = evaluate_automatic(
        plan,
        policy,
        quality,
        timing,
        calibration,
        preflight=_preflight(plan),
        prompt_history_checkpoint=checkpoint,
    )
    assert result == {
        "status": "INCOMPLETE",
        "reasons": ["prompt history checkpoint is not bound to the incumbent state"],
    }
