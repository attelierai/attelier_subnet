"""Receipt-to-baseline screening integration over real JSON files and SQLite."""

import json

from forge.evaluation.launch_baseline import ParentLaunchBaselineStore
from forge.evaluation.launch_screening import SCHEMA, _screening_rows
from forge.protocol.hashing import hash_canonical, sha256_file


def _hash(number):
    return f"{number:064x}"


def test_screening_rows_are_derived_from_receipts_and_selectable(tmp_path):
    root, anchor, candidate = tmp_path / "remote", _hash(1), _hash(2)
    prompt_rows = [
        {
            "prompt_id": f"prompt-{index}",
            "text": f"motion {index}",
            "seed": index + 10,
            "tags": ["motion_required"] if index else [],
        }
        for index in range(6)
    ]
    thresholds = {
        "dark_frame_fraction": 1.0,
        "magenta_frame_fraction": 1.0,
        "bright_frame_fraction": 1.0,
        "frozen_fraction": 0.95,
        "luma_acceleration_p95": 10.0,
        "high_frequency_proxy_ratio": 0.1,
        "near_freeze_temporal_max": 0.01,
        "near_freeze_ratio_max": 0.01,
        "near_freeze_reference_floor": 0.01,
    }
    screening_body = {
        "schema": SCHEMA,
        "contract_hash": _hash(3),
        "contract": {},
        "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
        "base": {"manifest": {}, "directory": "/workspace/forge/wan50", "manifest_hash": anchor},
        "candidates": [
            {
                "name": "Wan44",
                "manifest": {},
                "directory": "/workspace/forge/wan44",
                "manifest_hash": candidate,
            }
        ],
        "prompts": prompt_rows,
        "evaluator_qualification": {
            "evaluator_id": _hash(4),
            "spec_hash": _hash(5),
            "guard_thresholds": thresholds,
        },
    }
    screening = {**screening_body, "screening_id": hash_canonical(SCHEMA, screening_body)}
    root.mkdir()
    name = "Wan44"
    reports = {}
    for side, manifest in (("reference", anchor), ("candidate", candidate)):
        report = {
            "schema": "forge/panel/v1",
            "admission": "public_snapshot",
            "manifest_hash": manifest,
            "contract_hash": screening["contract_hash"],
            "geometry": screening["geometry"],
            "runtime_id": _hash(8),
            "runtime": {"pinned": "runtime"},
            "hardware": {"gpu_name": "fixture", "vram_capacity_bytes": 1},
            "recipe": {"steps": 50 if side == "reference" else 44, "guidance_scale": 5.0},
            "scheduler_recipe": {"kind": "original_unipc", "shift": None},
            "observations": [
                {
                    "prompt_id": p["prompt_id"],
                    "seed": p["seed"],
                    "status": "OK",
                    "video_sha256": _hash(100 + index + (20 if side == "candidate" else 0)),
                }
                for index, p in enumerate(prompt_rows)
            ],
        }
        path = root / name / side / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
        reports[side] = (report, path)
    timing = {
        "schema": "forge/screen-timing/v1",
        "frozen_plan_id": screening["screening_id"],
        "geometry": screening["geometry"],
        "contract_hash": screening["contract_hash"],
        "runtime_id": _hash(8),
        "runtime": {"pinned": "runtime"},
        "hardware": {"gpu_name": "fixture", "vram_capacity_bytes": 1},
        "reference_manifest_hash": anchor,
        "candidate_manifest_hash": candidate,
        "reference_admission": "public_snapshot",
        "candidate_admission": "public_snapshot",
        "design": {"name": "screening_ab", "order": ["reference", "candidate"]},
        "recipes": {
            "reference": {
                "steps": 50,
                "guidance_scale": 5.0,
                "kind": "original_unipc",
                "shift": None,
            },
            "candidate": {
                "steps": 44,
                "guidance_scale": 5.0,
                "kind": "original_unipc",
                "shift": None,
            },
        },
        "measurements": [
            {
                "prompt_id": p["prompt_id"],
                "side": side,
                "status": "OK",
                "wall_time_seconds": seconds,
            }
            for p in prompt_rows
            for side, seconds in (("reference", 2.0), ("candidate", 1.0))
        ],
    }
    timing_path = root / name / "screen-timing.json"
    timing_path.write_text(json.dumps(timing))
    diagnostics = {
        "dark_frame_fraction": 0.0,
        "magenta_frame_fraction": 0.0,
        "bright_frame_fraction": 0.0,
        "frozen_fraction": 0.1,
        "luma_acceleration_p95": 0.1,
        "high_frequency_proxy": 1.0,
        "temporal_delta": 1.0,
    }
    score_dir = root / "scores" / name
    score_dir.mkdir(parents=True)
    for side, (report, path) in reports.items():
        for observation in report["observations"]:
            value = {
                "schema": "forge/launch-score/v1",
                "plan_id": screening["screening_id"],
                "prompt_id": observation["prompt_id"],
                "seed": observation["seed"],
                "manifest_hash": report["manifest_hash"],
                "video_sha256": observation["video_sha256"],
                "evaluator_id": _hash(4),
                "spec_hash": _hash(5),
                "geometry": screening["geometry"],
                "source_report_sha256": sha256_file(path),
                "result": {
                    "video_sha256": observation["video_sha256"],
                    "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                    "diagnostics": diagnostics,
                },
            }
            (score_dir / f"{side}-{observation['prompt_id']}.json").write_text(json.dumps(value))
    receipt, sources = _screening_rows(root, screening, screening["candidates"][0], score_dir)
    assert all(row["status"] == "OK" and not row["vetoes"] for row in receipt["rows"])
    assert len(sources["bindings"]) == 6
    selection = ParentLaunchBaselineStore(tmp_path / "state.sqlite").select_once(
        [receipt], anchor_hash=anchor
    )
    assert selection["selected_candidate_hash"] == candidate
