"""V2 panel/policy filesystem contract; no GPU or score-provider call."""

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from forge.evaluation.evaluator_qualification import (
    SOURCE_TEMPORAL_DELTA_FLOOR,
    V2_DECISION_RULE,
    finalize_qualification,
    prepare_controls,
    replay_qualification_package,
    source_manifest,
)
from forge.evaluation.flicker_guard import algorithm_id
from forge.evaluation.flicker_guard import envelope as flicker_guard_envelope
from forge.evaluation.launch_policy import V2_SCHEMA, diversity_seeds, freeze_v2, validate_plan
from forge.evaluation.launch_worker import diversity_receipt
from forge.evaluation.synthetic_prompts import launch_panel_v2
from forge.evaluation.vision_media import load_video
from forge.protocol.hashing import hash_canonical, sha256_file

ROOT = Path(__file__).parents[2]


def _config(entropy="e" * 64, round_id="testnet-v2"):
    prompts = launch_panel_v2(entropy, round_id=round_id)
    timing = sorted(
        (item["prompt_id"] for item in prompts),
        key=lambda value: hash_canonical("forge/launch-v2-timing/v1", entropy, round_id, value),
    )[:8]
    diversity = []
    for family in sorted({item["family"] for item in prompts}):
        choices = [item["prompt_id"] for item in prompts if item["family"] == family]
        diversity.append(
            min(
                choices,
                key=lambda value: hash_canonical(
                    "forge/launch-v2-diversity/v1", entropy, round_id, family, value
                ),
            )
        )
    qualification = {
        "schema": "forge/evaluator-qualification/v1",
        "qualification": "QUALIFIED",
        "manifest_id": "1" * 64,
        "receipt_id": "2" * 64,
        "evaluator_id": "d" * 64,
        "guard_thresholds": {
            "dark_frame_fraction": 0.0,
            "magenta_frame_fraction": 0.0,
            "bright_frame_fraction": 0.0,
            "frozen_fraction": 0.95,
            "luma_acceleration_p95": 1.0,
            "high_frequency_proxy_ratio": 0.5,
            "near_freeze_temporal_max": 0.1 / 255,
            "near_freeze_ratio_max": 0.01,
            "near_freeze_reference_floor": 0.1 / 255,
        },
    }
    return {
        "policy_schema": V2_SCHEMA,
        "models": {"candidate": "c" * 64, "king": "9" * 64, "anchor": "b" * 64},
        "evaluator": {"evaluator_id": "d" * 64, "spec_hash": "e" * 64},
        "runtime": {
            "runtime_id": "f" * 64,
            "hardware_id": "1" * 64,
            "dependency_lock_hash": "2" * 64,
        },
        "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
        "prompts": prompts,
        "timing_prompt_ids": timing,
        "diversity_prompt_ids": diversity,
        "evaluator_qualification": qualification,
    }


def _rows(plan):
    rows = []
    for prompt in plan["prompts"]:
        diagnostics = {
            "dark_frame_fraction": 0.0,
            "magenta_frame_fraction": 0.0,
            "bright_frame_fraction": 0.0,
            "frozen_fraction": 0.0,
            "luma_acceleration_p95": 0.1,
            "high_frequency_proxy": 1.0,
            "temporal_delta": 0.1,
        }
        candidate = {
            "video_sha256": f"{prompt['seed']:064x}",
            "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
            "diagnostics": diagnostics,
        }
        for reference in ("king", "anchor"):
            rows.append(
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
                        "video_sha256": "a" * 64,
                        "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                        "diagnostics": diagnostics,
                    },
                    "candidate_result": candidate,
                }
            )
    for prompt_id in plan["timing_prompt_ids"]:
        rows.append(
            {
                "type": "timing",
                "plan_id": plan["plan_id"],
                "prompt_id": prompt_id,
                "times": [12.0, 8.0, 8.0, 12.0],
            }
        )
    for index, prompt_id in enumerate(plan["diversity_prompt_ids"]):
        rows.append(
            {
                "type": "diversity",
                "plan_id": plan["plan_id"],
                "prompt_id": prompt_id,
                "decoded_video_sha256": [f"{index * 10 + seed:064x}" for seed in range(3)],
            }
        )
    return rows


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "forge.evaluation.launch_policy", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _video(path, value, *, static=False):
    frames = np.full((4, 32, 32, 3), value, dtype=np.uint8)
    for frame_index, frame in enumerate(frames):
        offset = value % 24 if static else (value + frame_index * 5) % 24
        frame[:, offset : offset + 8] = 255 - value
    writer = imageio_ffmpeg.write_frames(
        str(path), (32, 32), fps=8, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p"
    )
    writer.send(None)
    for frame in frames:
        writer.send(frame.tobytes())
    writer.close()


def test_v2_panel_is_deterministic_disjoint_and_policy_has_exact_outcomes(tmp_path):
    entropy, round_id = "e" * 64, "testnet-v2"
    development = launch_panel_v2(entropy, round_id=round_id, split="development")
    confirmation = launch_panel_v2(entropy, round_id=round_id)
    assert len(confirmation) == 32
    assert sum(item["kind"] == "static" for item in confirmation) == 8
    assert {item["family"] for item in confirmation} == {"human", "animal", "object", "scenery"}
    assert {item["combination_id"] for item in development}.isdisjoint(
        {item["combination_id"] for item in confirmation}
    )
    static_markers = ("motionless", "still air", "no movement")
    for panel in (development, confirmation):
        for prompt in panel:
            if prompt["kind"] == "static":
                assert "motionless" in prompt["text"]
                assert "camera remains fixed" in prompt["text"].lower()
            else:
                assert not any(marker in prompt["text"].lower() for marker in static_markers)
                assert "motion_required" in prompt["tags"]

    source, frozen, evidence, result_path = (
        tmp_path / name for name in ("config.json", "plan.json", "rows.json", "result.json")
    )
    source.write_text(json.dumps(_config(entropy, round_id)))
    _run("freeze", "--config", source, "--out", frozen)
    plan = json.loads(frozen.read_text())
    validate_plan(plan, expected_entropy_hash=entropy, expected_round_id=round_id)
    evidence.write_text(json.dumps({"rows": _rows(plan)}))
    _run("evaluate", "--plan", frozen, "--rows", evidence, "--out", result_path)
    result = json.loads(result_path.read_text())
    assert result["status"] == "PASS" and result["action"] == "PROMOTION_ELIGIBLE"

    copied = _rows(plan)
    next(row for row in copied if row["type"] == "diversity")["decoded_video_sha256"] = [
        "0" * 64
    ] * 3
    evidence.write_text(json.dumps({"rows": copied}))
    failed = __import__("forge.evaluation.launch_policy", fromlist=["evaluate"]).evaluate(
        plan, {"rows": copied}
    )
    assert failed["status"] == "FAIL" and failed["action"] == "KEEP"

    unqualified = json.loads(json.dumps(plan))
    unqualified["evaluator_qualification"]["qualification"] = "NOT_QUALIFIED"
    # It is no longer hash-valid, therefore this is still fail-closed rather than a fake pass.
    assert (
        __import__("forge.evaluation.launch_policy", fromlist=["evaluate"]).evaluate(
            unqualified, {"rows": _rows(plan)}
        )["status"]
        == "INCOMPLETE"
    )


def test_v2_qualification_accepts_soft_detail_but_vetoes_black_frames(tmp_path):
    config = _config()
    qualification = config["evaluator_qualification"]
    qualification["qualification_version"] = 2
    qualification["decision_rule"] = V2_DECISION_RULE
    qualification["guard_thresholds"].pop("high_frequency_proxy_ratio")
    plan = freeze_v2(config)
    plan_path = tmp_path / "v2-plan.json"
    plan_path.write_text(json.dumps(plan))

    def evaluate_rows(rows, name):
        evidence, result = tmp_path / f"{name}-rows.json", tmp_path / f"{name}-result.json"
        evidence.write_text(json.dumps({"rows": rows}))
        _run("evaluate", "--plan", plan_path, "--rows", evidence, "--out", result)
        return json.loads(result.read_text())

    soft = _rows(plan)
    for row in soft:
        if row["type"] == "quality":
            diagnostics = dict(row["candidate_result"]["diagnostics"])
            diagnostics["high_frequency_proxy"] = 0.01
            row["candidate_result"] = {**row["candidate_result"], "diagnostics": diagnostics}
    assert evaluate_rows(soft, "soft")["status"] == "PASS"
    corrupted = _rows(plan)
    for row in corrupted:
        if row["type"] == "quality":
            diagnostics = dict(row["candidate_result"]["diagnostics"])
            diagnostics["dark_frame_fraction"] = 0.25
            row["candidate_result"] = {**row["candidate_result"], "diagnostics": diagnostics}
    assert evaluate_rows(corrupted, "black")["status"] == "FAIL"


def test_v3_guard_consumer_requires_real_envelopes_and_rejects_flicker_pixels():
    """Exercise the policy consumer with real decoded-pixel guard payloads."""
    config = _config()
    qualification = config["evaluator_qualification"]
    qualification["qualification_version"] = 3
    qualification["guard_thresholds"].pop("high_frequency_proxy_ratio")
    qualification["guard_thresholds"]["flicker_guard_v1"] = algorithm_id()
    plan = freeze_v2(config)

    clean_frames = np.full((4, 32, 32, 3), 100, dtype=np.uint8)
    for index in range(len(clean_frames)):
        clean_frames[index, :, index * 4 : index * 4 + 8] = 160
    clean = flicker_guard_envelope(clean_frames)
    flicker_frames = clean_frames.copy()
    flicker_frames[::2] = 255 - flicker_frames[::2]
    flicker = flicker_guard_envelope(flicker_frames)

    def rows_with(candidate_guard):
        rows = _rows(plan)
        for row in rows:
            if row["type"] == "quality":
                reference = dict(row["reference_result"])
                candidate = dict(row["candidate_result"])
                reference["diagnostics"] = {
                    **reference["diagnostics"],
                    "flicker_guard_v1": clean,
                }
                candidate["diagnostics"] = {
                    **candidate["diagnostics"],
                    **({"flicker_guard_v1": candidate_guard} if candidate_guard else {}),
                }
                row["reference_result"], row["candidate_result"] = reference, candidate
        return rows

    assert (
        __import__("forge.evaluation.launch_policy", fromlist=["evaluate"]).evaluate(
            plan, {"rows": rows_with(clean)}
        )["status"]
        == "PASS"
    )
    assert (
        __import__("forge.evaluation.launch_policy", fromlist=["evaluate"]).evaluate(
            plan, {"rows": rows_with(flicker)}
        )["status"]
        == "FAIL"
    )
    assert (
        __import__("forge.evaluation.launch_policy", fromlist=["evaluate"]).evaluate(
            plan, {"rows": rows_with(None)}
        )["status"]
        == "INCOMPLETE"
    )


def test_diversity_receipt_binds_candidate_report_and_three_frozen_seeds(tmp_path):
    """Filesystem/MP4 boundary with a simulated unavailable GPU provider."""
    hardware = {"provider": "fake-external-gpu"}
    dependencies = {"locked": "fake-contract"}
    config = _config()
    config["runtime"] = {
        "runtime_id": "f" * 64,
        "hardware_id": hash_canonical("forge/launch-hardware/v1", hardware),
        "dependency_lock_hash": hash_canonical("forge/launch-dependencies/v1", dependencies),
    }
    plan = freeze_v2(config)
    report = {
        "schema": "forge/panel/v1",
        "admission": "public_snapshot",
        "contract_hash": "a" * 64,
        "manifest_hash": plan["models"]["candidate"],
        "geometry": plan["geometry"],
        "runtime_id": plan["runtime"]["runtime_id"],
        "runtime": {"dependencies": dependencies},
        "hardware": hardware,
        "prompts": [],
        "observations": [],
    }
    video = tmp_path / "provider-output.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(video), (832, 480), fps=16, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p"
    )
    writer.send(None)
    for frame_index in range(81):
        frame = np.zeros((480, 832, 3), dtype=np.uint8)
        frame[:, (frame_index * 11) % 800 : (frame_index * 11) % 800 + 32] = 200
        writer.send(frame.tobytes())
    writer.close()
    for prompt_id in plan["diversity_prompt_ids"]:
        prompt = next(item for item in plan["prompts"] if item["prompt_id"] == prompt_id)
        seeds = diversity_seeds(plan, prompt_id)
        report["prompts"].append(
            {"prompt_id": prompt_id, "text": prompt["text"], "video_seeds": seeds}
        )
        for seed in seeds:
            report["observations"].append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "status": "OK",
                    "video": video.name,
                    "video_sha256": sha256_file(video),
                }
            )
    report_path = tmp_path / "provider-report.json"
    report_path.write_text(json.dumps(report))
    receipt = diversity_receipt(report=report_path, plan=plan, contract_hash="a" * 64)
    assert {row["prompt_id"] for row in receipt["rows"]} == set(plan["diversity_prompt_ids"])
    assert all(row["seeds"] == diversity_seeds(plan, row["prompt_id"]) for row in receipt["rows"])

    report["manifest_hash"] = "b" * 64
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="candidate runtime"):
        diversity_receipt(report=report_path, plan=plan, contract_hash="a" * 64)
    report["manifest_hash"] = plan["models"]["candidate"]
    report["prompts"][0]["video_seeds"] = list(reversed(report["prompts"][0]["video_seeds"]))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="prompt or seeds"):
        diversity_receipt(report=report_path, plan=plan, contract_hash="a" * 64)


def test_qualification_manifest_reads_real_video_files_but_synthetic_controls_cannot_qualify(
    tmp_path,
):
    sources = []
    for split in ("development", "confirmation"):
        for index, family in enumerate(("human", "animal", "object", "scenery")):
            path = tmp_path / f"{split}-{family}.mp4"
            _video(path, 20 + index + (40 if split == "confirmation" else 0))
            sources.append(
                {
                    "source_id": f"{split}-{family}",
                    "split": split,
                    "family": family,
                    "prompt": f"{family} prompt",
                    "path": str(path),
                }
            )
    manifest = source_manifest(sources, {"width": 32, "height": 32, "frames": 4, "fps": 8})
    assert manifest["source_eligibility"]["temporal_delta_floor"] == SOURCE_TEMPORAL_DELTA_FLOOR
    assert all(
        source["temporal_delta"] > SOURCE_TEMPORAL_DELTA_FLOOR for source in manifest["sources"]
    )
    with pytest.raises(ValueError, match="plan, raw and result"):
        replay_qualification_package(
            {
                "manifest": manifest,
                "evidence_origin": "real_gpu_calibrated",
                "detected": True,
                "qualification": "QUALIFIED",
            }
        )


def test_qualification_rejects_static_mp4_sources_before_control_scoring(tmp_path):
    sources = []
    for split in ("development", "confirmation"):
        for index, family in enumerate(("human", "animal", "object", "scenery")):
            path = tmp_path / f"static-{split}-{family}.mp4"
            _video(path, 20 + index + (40 if split == "confirmation" else 0), static=True)
            sources.append(
                {
                    "source_id": f"{split}-{family}",
                    "split": split,
                    "family": family,
                    "prompt": f"{family} prompt",
                    "path": str(path),
                }
            )
    controls = tmp_path / "static-controls"
    with pytest.raises(ValueError, match="temporal movement"):
        prepare_controls(sources, {"width": 32, "height": 32, "frames": 4, "fps": 8}, controls)
    assert not controls.exists()


def test_qualification_identity_copies_source_bytes_and_transcode_is_pinned(tmp_path):
    sources = []
    for split in ("development", "confirmation"):
        for index, family in enumerate(("human", "animal", "object", "scenery")):
            path = tmp_path / f"{split}-{family}.mp4"
            _video(path, 20 + index + (40 if split == "confirmation" else 0))
            sources.append(
                {
                    "source_id": f"{split}-{family}",
                    "split": split,
                    "family": family,
                    "prompt": f"{family} prompt",
                    "path": str(path),
                }
            )
    root = tmp_path / "controls"
    plan = prepare_controls(sources, {"width": 32, "height": 32, "frames": 4, "fps": 8}, root)
    source = sources[0]
    identity = next(
        item
        for item in plan["controls"]
        if item["control_id"] == f"{source['source_id']}--identity"
    )
    transcode = next(
        item
        for item in plan["controls"]
        if item["control_id"] == f"{source['source_id']}--transcode"
    )
    assert sha256_file(root / identity["path"]) == sha256_file(source["path"])
    assert sha256_file(root / transcode["path"]) != sha256_file(source["path"])
    assert plan["codec"]["name"] == "libx264" and plan["codec"]["crf"] == 18

    # Exercise the actual file/CLI replay boundary. Scores here emulate only the
    # unavailable paid scorer; media diagnostics come from decoded MP4s. Identical
    # semantic scores MUST fail qualification, even if a caller labels them good.
    spec, runtime = {"fixture": "external-scorer"}, {"device": "integration"}
    raw = {
        "schema": plan["schema"],
        "control_plan_id": plan["control_plan_id"],
        "evaluator": {
            "spec": spec,
            "runtime": runtime,
            "spec_hash": hash_canonical("forge/launch-evaluator-spec/v1", spec),
            "evaluator_id": hash_canonical("forge/videoreward-study-runtime/v1", spec, runtime),
        },
        "rows": [],
    }
    for control in plan["controls"]:
        row = {"control_id": control["control_id"]}
        if control["kind"] == "missing":
            row["status"] = "MISSING_REJECTED"
        elif control["kind"] == "hash_tamper":
            row["status"] = "HASH_MISMATCH_REJECTED"
        else:
            decoded = load_video(root / control["path"], plan["manifest"]["geometry"])
            row.update(
                status="OK",
                result={
                    "video_sha256": decoded.video_sha256,
                    "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                    "diagnostics": decoded.diagnostics,
                },
            )
        raw["rows"].append(row)
    raw["raw_receipt_id"] = hash_canonical("forge/evaluator-controls-score/v1", raw)
    tampered_plan = json.loads(json.dumps(plan))
    tampered_plan["manifest"]["sources"][0]["temporal_delta"] = 0.0
    manifest_body = {
        key: value for key, value in tampered_plan["manifest"].items() if key != "manifest_id"
    }
    tampered_plan["manifest"]["manifest_id"] = hash_canonical(
        "forge/evaluator-qualification/v1", manifest_body
    )
    plan_body = {key: value for key, value in tampered_plan.items() if key != "control_plan_id"}
    tampered_plan["control_plan_id"] = hash_canonical("forge/evaluator-controls/v1", plan_body)
    tampered_raw = json.loads(json.dumps(raw))
    tampered_raw["control_plan_id"] = tampered_plan["control_plan_id"]
    tampered_raw["raw_receipt_id"] = hash_canonical(
        "forge/evaluator-controls-score/v1",
        {key: value for key, value in tampered_raw.items() if key != "raw_receipt_id"},
    )
    assert finalize_qualification(tampered_plan, tampered_raw)["qualification"] == "INCOMPLETE"
    frozen, score_path, result_path = (
        tmp_path / name for name in ("plan.json", "raw.json", "result.json")
    )
    frozen.write_text(json.dumps(plan))
    score_path.write_text(json.dumps(raw))
    args = [
        sys.executable,
        "-m",
        "forge.evaluation.evaluator_qualification",
        "finalize",
        "--plan",
        str(frozen),
        "--raw",
        str(score_path),
        "--out",
        str(result_path),
    ]
    subprocess.run(args, cwd=ROOT, check=True, capture_output=True)
    result = json.loads(result_path.read_text())
    assert result["qualification"] == "FAILED"
    assert any(reason.endswith(":semantic_mismatch") for reason in result["reasons"])
    with pytest.raises(ValueError, match="not qualified"):
        replay_qualification_package({"plan": plan, "raw": raw, "result": result})
    with pytest.raises(ValueError, match="never overwrite"):
        prepare_controls(sources, plan["manifest"]["geometry"], root)
    # A rehashed duplicate raw row cannot claim complete control coverage, and the
    # CLI must refuse to overwrite the original immutable failed-result receipt.
    raw["rows"][-1] = raw["rows"][0]
    raw["raw_receipt_id"] = hash_canonical(
        "forge/evaluator-controls-score/v1",
        {key: value for key, value in raw.items() if key != "raw_receipt_id"},
    )
    score_path.write_text(json.dumps(raw))
    changed = subprocess.run(args, cwd=ROOT, capture_output=True)
    assert changed.returncode != 0 and b"immutable receipt changed" in changed.stderr
    assert json.loads(result_path.read_text()) == result
