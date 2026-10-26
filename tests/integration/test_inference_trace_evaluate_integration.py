"""Filesystem boundary coverage for the local trace-evidence analyzer."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(root: Path, name: str, values: list[float]) -> dict:
    path = root / f"{name}.npy"
    array = np.asarray(values, dtype=np.float32)
    np.save(path, array)
    return {
        "tensorPath": path.name,
        "sha256": _sha(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _run(
    root: Path, run_id: str, split: str, role: str, lineage: str, *, altered=False, output="a"
) -> dict:
    trace = []
    for step in range(20):
        values = [float(step), 1.0]
        if altered and step == 4:
            values[1] = 50.0
        trace.append({"step": step, "tensor": _tensor(root, f"{run_id}-{step}", values)})
    return {
        "schema": "forge/inference-trace-capture/v1",
        "runId": run_id,
        "split": split,
        "role": role,
        "condition": role,
        "lineageId": lineage,
        "inputHash": "f" * 64,
        "capture": {"runtime": "fixture"},
        "trace": trace,
        "finalLatent": _tensor(root, f"{run_id}-final", [1.0, 2.0]),
        "output": {
            "canonicalVideoSha256": output * 64,
            "canonicalAudioSha256": "b" * 64,
            "mediaSha256": "c" * 64,
        },
    }


def _write(root: Path, *, wrong_output: bool) -> None:
    runs = []
    for split, lineages in (("calibration", ("one", "two")), ("holdout", ("three",))):
        for lineage in lineages:
            runs += [
                _run(root, f"{split}-{lineage}-base", split, "baseline", lineage),
                _run(root, f"{split}-{lineage}-honest", split, "honest", lineage),
                _run(
                    root, f"{split}-{lineage}-attack", split, "adversarial", lineage, altered=True
                ),
            ]
    if wrong_output:
        runs.append(
            _run(root, "calibration-one-sidecar", "calibration", "adversarial", "one", output="d")
        )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "schema": "forge/inference-trace-experiment/v1",
                "frozenSplit": {
                    "calibrationLineages": ["one", "two"],
                    "holdoutLineages": ["three"],
                },
                "runs": runs,
            }
        )
    )


def _as_nested_capture_bindings(root: Path) -> None:
    experiment = json.loads((root / "experiment.json").read_text())
    bindings = []
    for source in experiment["runs"]:
        capture_dir = root / "captures" / source["runId"]
        capture_dir.mkdir(parents=True)

        def relocate(descriptor: dict, target_root: Path = capture_dir) -> dict:
            old = root / descriptor["tensorPath"]
            target = target_root / old.name
            shutil.copyfile(old, target)
            return {**descriptor, "tensorPath": target.name, "sha256": _sha(target)}

        trace = [
            {
                "step": row["step"],
                "totalSteps": 20,
                "sigma": 1.0,
                "stateCallback": [{"component": "video", **relocate(row["tensor"])}],
                "denoised": [],
            }
            for row in source["trace"]
        ]
        manifest = {
            key: value
            for key, value in source.items()
            if key not in {"role", "trace", "finalLatent", "output"}
        }
        manifest.update(
            trace=trace,
            finalLatent=[{"component": "video", **relocate(source["finalLatent"])}],
            decoded={},
            output={
                "runId": source["runId"],
                "status": "completed",
                "sha256": source["output"]["mediaSha256"],
                "video": {"sha256": source["output"]["canonicalVideoSha256"], "bytes": 1},
                "audio": {"sha256": source["output"]["canonicalAudioSha256"], "bytes": 1},
                "media": "/private/capture.mp4",
            },
        )
        (capture_dir / "manifest.json").write_text(json.dumps(manifest))
        bindings.append(
            {
                "role": source["role"],
                "capturePath": str((capture_dir / "manifest.json").relative_to(root)),
            }
        )
    experiment["runs"] = bindings
    (root / "experiment.json").write_text(json.dumps(experiment))


def test_trace_evidence_uses_frozen_lineages_and_reports_output_sidecar_bypass(tmp_path):
    _write(tmp_path, wrong_output=False)
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/inference_trace_evaluate.py",
            "--experiment",
            str(tmp_path),
            "--out",
            str(report),
        ],
        check=True,
    )
    clean = json.loads(report.read_text())
    assert clean["status"] == "PASS"
    assert clean["claim"] == "callback_trajectory_discrimination_only"
    _as_nested_capture_bindings(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "scripts/inference_trace_evaluate.py",
            "--experiment",
            str(tmp_path),
            "--out",
            str(report),
        ],
        check=True,
    )
    assert json.loads(report.read_text())["status"] == "PASS"
    _write(tmp_path, wrong_output=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/inference_trace_evaluate.py",
            "--experiment",
            str(tmp_path),
            "--out",
            str(report),
        ],
        check=True,
    )
    bypass = json.loads(report.read_text())
    assert bypass["status"] == "INCONCLUSIVE"
    assert bypass["reason"] == "trace_to_output_bypass_observed"
    assert bypass["bypasses"][0]["outputDiffersFromBaseline"] is True
