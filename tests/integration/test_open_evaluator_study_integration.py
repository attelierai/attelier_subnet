"""Study CLI, archived-result contracts and real video codec boundaries.

Reward rows below are explicit GPU-boundary fixtures, not VideoReward inference.
No provider, GPU, model download or claims of quality are involved.
"""

import json
import subprocess
import sys
from pathlib import Path

import imageio
import numpy as np
import pytest

from forge.evaluation.evaluator_attacks import prepare
from forge.evaluation.evaluator_study import SCHEMA, seal
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.evaluation.vision_media import load_video
from forge.protocol.hashing import hash_canonical, sha256_file

ROOT = Path(__file__).parents[2]


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def command(module, *args, ok=True):
    result = subprocess.run(
        [sys.executable, "-m", "forge.evaluation." + module, *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if ok:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def receipt(plan, request, number):
    sample = next(s for s in plan["samples"] if s["sample_id"] == request["sample_id"])
    runtime = {"fixture": "external GPU response; no actual model execution"}
    return {
        "request_id": request["request_id"],
        "plan_id": plan["plan_id"],
        "evaluator_id": hash_canonical("forge/videoreward-study-runtime/v1", plan["spec"], runtime),
        "runtime": runtime,
        "status": "OK",
        "qualification": "prohibited",
        "video_sha256": sample["video_sha256"],
        "scores": {"VQ": number, "MQ": -number, "TA": 0.0},
        "elapsed_seconds": 1.5,
        "peak_cuda_bytes": 4096,
        "diagnostics": {"frozen_fraction": 0.0},
    }


def test_archived_pilot_to_comparison_report_and_rejection_boundaries(tmp_path):
    pilot_path = ROOT / "configs/evaluator-pilot-manifest-v1.json"
    work = pilot_worklist(json.loads(pilot_path.read_text()))
    sources = {r["job_id"]: r for r in work["renders"]}
    videos = {}
    for i, row in enumerate(work["renders"] + work["transforms"]):
        source = row if "prompt_text" in row else sources[row["reference_job_id"]]
        videos[row["job_id"]] = {
            "job_id": row["job_id"],
            "path": row["job_id"] + ".mp4",
            "video_sha256": f"{i:064x}",
            "prompt_text": source["prompt_text"],
            "seed": source["seed"],
        }
    for row in work["transforms"]:
        videos[row["job_id"]]["transformation"] = {
            "reference_job_id": row["reference_job_id"],
            "reference_video_sha256": videos[row["reference_job_id"]]["video_sha256"],
            "kind": row["kind"],
            "options": row["options"],
            "candidate_video_sha256": videos[row["job_id"]]["video_sha256"],
        }
    index = put(
        tmp_path / "index.json",
        {
            "schema": "forge/evaluator-pilot-media-index/v1",
            "status": "COMPLETE",
            "worklist_id": work["worklist_id"],
            "recipe": work["recipe"],
            "videos": list(videos.values()),
        },
    )
    legacy = put(
        tmp_path / "legacy.json",
        {
            "status": "COMPLETE",
            "qualification": "prohibited",
            "rows": [
                {
                    "slot_id": p["slot_id"],
                    "kind": p["kind"],
                    "variant": "base",
                    "luna": {"status": "OK", "axes": {"visual_quality": -0.25}},
                    "orders": [
                        {"reference_first": v, "axes": {"image_quality": -0.1}}
                        for v in (True, False)
                    ],
                }
                for p in work["comparisons"]
            ],
        },
    )
    path = tmp_path / "plan.json"
    command(
        "evaluator_study",
        "freeze-pilot",
        "--pilot",
        pilot_path,
        "--media-index",
        index,
        "--legacy",
        legacy,
        "--spec",
        ROOT / "configs/videoreward-study-v1.json",
        "--repeats",
        12,
        "--out",
        path,
    )
    plan = json.loads(path.read_text())
    assert len(plan["comparisons"]) == 96
    assert len(plan["samples"]) == 96  # 88 files, eight wrong-prompt/video combinations.
    assert len(plan["requests"]) == 108
    samples = {s["sample_id"]: s for s in plan["samples"]}
    for pair in plan["comparisons"]:
        assert (
            samples[pair["reference_sample"]]["prompt_text"]
            == (samples[pair["candidate_sample"]]["prompt_text"])
        )
        if pair["kind"] == "wrong_video":
            assert pair["candidate_sample"] != pair["reference_sample"]
    report = tmp_path / "missing.json"
    command("evaluator_study", "report", "--plan", path, "--out", report)
    missing = json.loads(report.read_text())
    assert missing["status"] == "INCOMPLETE" and missing["measurements"] == 0
    assert all(r["open_delta"] is None for r in missing["comparisons"])
    rows = tmp_path / "receipts"
    rows.mkdir()
    for i, request in enumerate(plan["requests"]):
        put(rows / f"{i}.json", receipt(plan, request, i / 100))
    output = tmp_path / "complete.json"
    command("evaluator_study", "report", "--plan", path, "--receipts", rows, "--out", output)
    preserved = output.read_bytes()
    command(
        "evaluator_study", "report", "--plan", path, "--receipts", rows, "--out", output, ok=False
    )
    assert output.read_bytes() == preserved
    complete = json.loads(output.read_text())
    assert complete["status"] == "COMPLETE" and complete["qualification"] == "prohibited"
    assert complete["quality_decision"] == "NOT_EMITTED"
    assert len(complete["repeatability"]) == 12
    for pair in complete["comparisons"]:
        if pair["kind"] in {"self", "static_scene"}:
            assert set(pair["open_delta"].values()) == {0.0}
    original = json.loads((rows / "0.json").read_text())
    for field, value in (
        ("plan_id", "b" * 64),
        ("video_sha256", "b" * 64),
        ("evaluator_id", "b" * 64),
        ("scores", {"VQ": float("nan"), "MQ": 0, "TA": 0}),
    ):
        put(rows / "0.json", {**original, field: value})
        command(
            "evaluator_study",
            "report",
            "--plan",
            path,
            "--receipts",
            rows,
            "--out",
            tmp_path / "invalid.json",
            ok=False,
        )
        assert not (tmp_path / "invalid.json").exists()
    put(rows / "0.json", original)
    put(rows / "duplicate.json", original)
    command(
        "evaluator_study",
        "report",
        "--plan",
        path,
        "--receipts",
        rows,
        "--out",
        tmp_path / "duplicate-report.json",
        ok=False,
    )
    # A newly edited plan cannot reuse pre-edit receipts, even if resealed.
    changed = dict(plan)
    changed.pop("plan_id")
    changed["samples"][0]["prompt_text"] = "Changed after closure"
    put(tmp_path / "changed.json", seal(changed))
    command(
        "evaluator_study",
        "report",
        "--plan",
        tmp_path / "changed.json",
        "--out",
        tmp_path / "changed-report.json",
        ok=False,
    )


def test_real_codec_attack_search_and_partial_selection_refusal(tmp_path):
    source = tmp_path / "source.mp4"
    frames = np.zeros((16, 32, 32, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        frame[:, (i * 2) % 24 : (i * 2) % 24 + 8] = 180
    with imageio.get_writer(str(source), fps=8, codec="libx264", macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)
    digest, prompt = sha256_file(source), "A bright shape moves smoothly across a dark scene."
    identity = hash_canonical(SCHEMA, digest, prompt)
    plan = seal(
        {
            "schema": SCHEMA,
            "split": "development",
            "qualification": "prohibited",
            "spec": {},
            "geometry": {"width": 32, "height": 32, "frames": 16, "fps": 8},
            "samples": [
                {
                    "sample_id": identity,
                    "path": "source.mp4",
                    "video_sha256": digest,
                    "prompt_text": prompt,
                    "source_family": "fixture-motion",
                }
            ],
            "requests": [{"request_id": identity + ":0", "sample_id": identity, "repeat": 0}],
            "comparisons": [
                {
                    "slot_id": "fixture",
                    "kind": "self",
                    "source_family": "fixture-motion",
                    "reference_sample": identity,
                    "candidate_sample": identity,
                }
            ],
        }
    )
    out = tmp_path / "attack"
    study = prepare(plan, tmp_path, out, families=1)
    assert len(study["comparisons"]) == 12
    # Original + matched codec control + neutral text control + 12 treatments.
    # Some encoders may byte-deduplicate an original and no-op control.
    assert 14 <= len(study["requests"]) <= 15
    by_id = {s["sample_id"]: s for s in study["samples"]}
    controls = {row["kind"]: row for row in study["controls"]}
    assert set(controls) == {"noop_reencode", "neutral_text"}
    assert all(row["kind"] not in controls for row in study["comparisons"])
    assert all(
        row["candidate_sample"]
        not in {
            controls["noop_reencode"]["sample_id"],
            controls["neutral_text"]["sample_id"],
        }
        for row in study["comparisons"]
    )
    for pair in study["comparisons"]:
        expected = "neutral_text" if pair["kind"] == "reward_text" else "noop_reencode"
        assert pair["reference_control"] == expected
        assert pair["reference_sample"] == controls[expected]["sample_id"]
    frozen = next(p for p in study["comparisons"] if p["kind"] == "freeze")
    decoded = load_video(out / by_id[frozen["candidate_sample"]]["path"], study["geometry"])
    assert decoded.diagnostics["frozen_fraction"] == 1
    with pytest.raises(FileExistsError):
        prepare(plan, tmp_path, out, families=1)
    receipts = tmp_path / "attack-receipts"
    receipts.mkdir()
    for i, request in enumerate(study["requests"][:-1]):
        put(receipts / f"{i}.json", receipt(study, request, i))
    command(
        "evaluator_attacks",
        "select",
        "--plan",
        out / "plan.json",
        "--receipts",
        receipts,
        "--out",
        tmp_path / "partial-selection.json",
        ok=False,
    )
    put(receipts / "last.json", receipt(study, study["requests"][-1], 99))
    selected = tmp_path / "selection.json"
    command(
        "evaluator_attacks",
        "select",
        "--plan",
        out / "plan.json",
        "--receipts",
        receipts,
        "--out",
        selected,
    )
    selection_bytes = selected.read_bytes()
    command(
        "evaluator_attacks",
        "select",
        "--plan",
        out / "plan.json",
        "--receipts",
        receipts,
        "--out",
        selected,
        ok=False,
    )
    assert selected.read_bytes() == selection_bytes
    result = json.loads(selected.read_text())
    assert result["qualification"] == "prohibited" and len(result["selected"]) == 4
    assert {r["human_quality_label"] for r in result["selected"]} == {"UNREVIEWED"}
    assert len(list(receipts.glob("*.json"))) == result["queries"]
    # A receipt for a different sealed plan cannot bind this search.
    changed = json.loads((receipts / "last.json").read_text())
    changed["plan_id"] = "f" * 64
    put(receipts / "last.json", changed)
    command(
        "evaluator_attacks",
        "select",
        "--plan",
        out / "plan.json",
        "--receipts",
        receipts,
        "--out",
        tmp_path / "foreign-selection.json",
        ok=False,
    )
    # CLI must reject local heavyweight work before importing any GPU dependencies.
    blocked = command(
        "video_reward",
        "measure",
        "--plan",
        out / "plan.json",
        "--assets",
        tmp_path,
        "--media-root",
        out,
        "--out",
        tmp_path / "gpu",
        "--deadline-unix",
        4_000_000_000,
        ok=False,
    )
    assert "/workspace/forge" in blocked.stderr and not (tmp_path / "gpu").exists()
