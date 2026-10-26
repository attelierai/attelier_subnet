"""Filesystem and replay coverage for the versioned evaluator qualification."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from forge.evaluation.evaluator_qualification import (
    CONTROL_KINDS,
    V2_DECISION_RULE,
    replay_qualification_package,
)
from forge.evaluation.launch_policy import freeze_policy_v2
from forge.protocol.hashing import hash_canonical

ROOT = Path(__file__).parents[2]
GEOMETRY = {"width": 32, "height": 32, "frames": 4, "fps": 8}


def _video(path: Path, value: int) -> None:
    frames = np.full((4, 32, 32, 3), value, dtype=np.uint8)
    for index, frame in enumerate(frames):
        frame[:, index * 4 : index * 4 + 8] = 255 - value
    writer = imageio_ffmpeg.write_frames(
        str(path), (32, 32), fps=8, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p"
    )
    writer.send(None)
    for frame in frames:
        writer.send(frame.tobytes())
    writer.close()


def _sources(root: Path) -> tuple[list[dict], list[str]]:
    old = [
        "ducks-swimming-pond",
        "baker-kneading-dough",
        "jug-pouring-glass",
        "waves-washing-sand",
        "horse-walking-field",
        "pendulum-swinging",
        "mist-drifting-vineyard",
    ]
    new = [
        ("animal", "dog-digging-garden"),
        ("animal", "squirrel-climbing-pine"),
        ("human", "drummer-playing-snare"),
        ("human", "dancer-pirouette-studio"),
        ("object", "toy-train-circling-track"),
        ("object", "metal-gears-meshing"),
        ("scenery", "waterfall-cascading-rocks"),
        ("scenery", "pine-branches-swaying-wind"),
    ]
    result = []
    for index, family in enumerate(("animal", "human", "object", "scenery")):
        path = root / f"development-{family}.mp4"
        _video(path, 30 + index * 10)
        result.append(
            {
                "source_id": f"development-{family}",
                "split": "development",
                "family": family,
                "prompt": f"human-reviewed moving {family}",
                "path": str(path),
                "lineage_id": f"development-{family}-new-lineage",
            }
        )
    for index, (family, lineage) in enumerate(new):
        path = root / f"confirmation-{index}.mp4"
        _video(path, 90 + index * 7)
        result.append(
            {
                "source_id": f"confirmation-{index}",
                "split": "confirmation",
                "family": family,
                "prompt": f"human-reviewed {lineage}",
                "path": str(path),
                "lineage_id": lineage,
            }
        )
    return result, old


def _raw(plan: dict) -> dict:
    base = {
        "dark_frame_fraction": 0.0,
        "magenta_frame_fraction": 0.0,
        "bright_frame_fraction": 0.0,
        "frozen_fraction": 0.0,
        "luma_acceleration_p95": 0.1,
        "high_frequency_proxy": 1.0,
        "temporal_delta": 0.1,
    }
    sources = {item["source_id"]: item for item in plan["manifest"]["sources"]}
    rows = []
    for control in plan["controls"]:
        kind, source = control["kind"], sources[control["source_id"]]
        if kind == "missing":
            rows.append({"control_id": control["control_id"], "status": "MISSING_REJECTED"})
            continue
        if kind == "hash_tamper":
            rows.append({"control_id": control["control_id"], "status": "HASH_MISMATCH_REJECTED"})
            continue
        diagnostics = dict(base, temporal_delta=source["temporal_delta"])
        scores = {"VQ": 0.0, "MQ": 0.0, "TA": 0.0}
        if kind == "freeze":
            diagnostics.update(frozen_fraction=1.0, temporal_delta=0.0)
        elif kind == "black_frame":
            diagnostics["dark_frame_fraction"] = 0.25
        elif kind == "magenta_frame":
            diagnostics["magenta_frame_fraction"] = 0.25
        elif kind == "segment_corruption":
            diagnostics["bright_frame_fraction"] = 0.5
        elif kind == "flicker":
            diagnostics["luma_acceleration_p95"] = 10.0
        elif kind == "blur":
            scores.update(VQ=-0.5, TA=-0.5)
        elif kind == "semantic_mismatch":
            # Mean -0.4 / n=8: the one-sided 95% upper bound passes -0.2,
            # while an accidental two-sided interpretation does not.
            scores["TA"] = (
                -0.66
                if source["split"] == "confirmation"
                and int(source["source_id"].rsplit("-", 1)[1]) % 2
                else -0.14
                if source["split"] == "confirmation"
                else -0.5
            )
        elif kind == "transcode":
            scores = {"VQ": 0.01, "MQ": -0.01, "TA": 0.01}
        rows.append(
            {
                "control_id": control["control_id"],
                "status": "OK",
                "result": {
                    "video_sha256": control["video_sha256"],
                    "scores": scores,
                    "diagnostics": diagnostics,
                },
            }
        )
    spec, runtime = {"pin": "fake-external-boundary"}, {"runtime": "fake"}
    evaluator = {
        "spec": spec,
        "runtime": runtime,
        "spec_hash": hash_canonical("forge/launch-evaluator-spec/v1", spec),
    }
    evaluator["evaluator_id"] = hash_canonical("forge/videoreward-study-runtime/v1", spec, runtime)
    body = {
        "schema": "forge/evaluator-qualification/v1",
        "control_plan_id": plan["control_plan_id"],
        "evaluator": evaluator,
        "rows": rows,
    }
    return {**body, "raw_receipt_id": hash_canonical("forge/evaluator-controls-score/v1", body)}


def _seal_raw(raw: dict) -> dict:
    body = {key: value for key, value in raw.items() if key != "raw_receipt_id"}
    return {**body, "raw_receipt_id": hash_canonical("forge/evaluator-controls-score/v1", body)}


def _finalize_cli(plan_path: Path, raw: dict, output: Path) -> dict:
    raw_path = output.with_suffix(".raw.json")
    raw_path.write_text(json.dumps(raw))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.evaluation.evaluator_qualification",
            "finalize",
            "--plan",
            str(plan_path),
            "--raw",
            str(raw_path),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text())


def test_v2_cli_prepares_a_lineage_disjoint_panel_and_replays_qualification(tmp_path):
    sources, excluded = _sources(tmp_path)
    source_path, geometry_path, excluded_path = (
        tmp_path / name for name in ("sources.json", "geometry.json", "excluded.json")
    )
    source_path.write_text(json.dumps(sources))
    geometry_path.write_text(json.dumps(GEOMETRY))
    excluded_path.write_text(json.dumps(excluded))
    controls, plan_path = tmp_path / "controls", tmp_path / "plan.json"
    command = [
        sys.executable,
        "-m",
        "forge.evaluation.evaluator_qualification",
        "prepare",
        "--sources",
        str(source_path),
        "--geometry",
        str(geometry_path),
        "--out",
        str(controls),
        "--plan-out",
        str(plan_path),
        "--qualification-version",
        "2",
        "--excluded-confirmation-lineages",
        str(excluded_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(plan_path.read_text())
    assert plan["qualification_version"] == 2 and plan["decision_rule"] == V2_DECISION_RULE
    assert len(plan["controls"]) == 12 * len(CONTROL_KINDS)
    assert plan["decision_rule"]["quality_margins"] == freeze_policy_v2()["quality_margins"]
    raw = _raw(plan)
    result = _finalize_cli(plan_path, raw, tmp_path / "qualified.json")
    assert result["qualification"] == "QUALIFIED", result
    assert "high_frequency_proxy_ratio" not in result["guard_thresholds"]
    assert replay_qualification_package({"plan": plan, "raw": raw, "result": result}) == result
    inconclusive = json.loads(json.dumps(raw))
    transcode_index = 0
    for row in inconclusive["rows"]:
        if row["control_id"].endswith("--transcode"):
            row["result"]["scores"]["VQ"] = 0.49 if transcode_index % 2 else -0.49
            transcode_index += 1
    assert (
        _finalize_cli(plan_path, _seal_raw(inconclusive), tmp_path / "inconclusive.json")[
            "qualification"
        ]
        == "INCONCLUSIVE"
    )
    failed = json.loads(json.dumps(raw))
    for row in failed["rows"]:
        if row["control_id"].endswith("--transcode"):
            row["result"]["scores"]["VQ"] = 0.36
    assert (
        _finalize_cli(plan_path, _seal_raw(failed), tmp_path / "failed.json")["qualification"]
        == "FAILED"
    )
    incomplete = json.loads(json.dumps(raw))
    incomplete["rows"].pop()
    assert (
        _finalize_cli(plan_path, _seal_raw(incomplete), tmp_path / "incomplete.json")[
            "qualification"
        ]
        == "INCOMPLETE"
    )
    collision = json.loads(source_path.read_text())
    collision[4]["lineage_id"] = excluded[0]
    collision_path = tmp_path / "collision.json"
    collision_path.write_text(json.dumps(collision))
    rejected = subprocess.run(
        [
            *command[:4],
            "--sources",
            str(collision_path),
            "--geometry",
            str(geometry_path),
            "--out",
            str(tmp_path / "collision-controls"),
            "--plan-out",
            str(tmp_path / "collision-plan.json"),
            "--qualification-version",
            "2",
            "--excluded-confirmation-lineages",
            str(excluded_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
