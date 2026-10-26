"""Controlled recurrence forgery must pass algebra only, never model binding."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from forge.inference.transition_diag import residual


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path, component: str, root: Path) -> dict[str, object]:
    value = np.load(path, allow_pickle=False)
    return {
        "component": component,
        "tensorPath": str(path.relative_to(root)),
        "sha256": _sha(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": path.stat().st_size,
    }


def _padded_fixture(root: Path) -> Path:
    run = root / "padded"
    tensors = run / "tensors"
    tensors.mkdir(parents=True)
    trace = []
    for step in range(4):
        states = []
        for offset, component in ((0.0, "video"), (10.0, "audio")):
            path = tensors / f"state-{step}-{component}.npy"
            np.save(path, np.array([step + offset, step + offset + 0.5], dtype=np.float32))
            states.append(_descriptor(path, component, run))
        trace.append(
            {
                "step": step,
                "totalSteps": 4,
                "sigma": 4.0 - step,
                "stateCallback": states,
                "denoised": states,
            }
        )
    final = []
    for value, component in (([99.0, 100.0], "video"), ([7.0, 8.0], "audio")):
        path = tensors / f"final-{component}.npy"
        np.save(path, np.array(value, dtype=np.float32))
        final.append(_descriptor(path, component, run))
    manifest = {
        "schema": "forge/inference-trace-capture/v1",
        "runId": "padded",
        "split": "calibration",
        "condition": "padded_steps_10",
        "lineageId": "fixed",
        "inputHash": "a" * 64,
        "capture": {"modelManifestHash": "m", "workflowHash": "w", "runtime": "r"},
        "declaredSteps": 4,
        "callbackSteps": 4,
        "inputNoise": trace[0]["stateCallback"],
        "trace": trace,
        "finalLatent": final,
        "decoded": {},
        "forward": [],
        "output": {
            "runId": "padded",
            "status": "completed",
            "sha256": "a" * 64,
            "video": {"sha256": "b" * 64},
            "audio": {"sha256": "c" * 64},
        },
    }
    path = run / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_recurrence_adversary_preserves_states_and_passes_only_transition_algebra(tmp_path):
    root = tmp_path / "experiment"
    padded = _padded_fixture(root)
    script = (
        Path(__file__).parents[2]
        / "scripts"
        / "verification"
        / "h3_trace_recurrence_adversary.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiment-root",
            str(root),
            "--padded-manifest",
            str(padded),
            "--output",
            str(root / "recurrence"),
        ],
        check=True,
    )
    manifest_path = root / "recurrence" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["runId"] != "padded"
    assert manifest["output"]["runId"] == manifest["runId"]
    assert manifest["condition"] == "honest_repeat"
    assert manifest["trace"][0]["stateCallback"][0]["tensorPath"].startswith("artifacts/raw/")
    assert manifest["trace"][0]["denoised"][0]["tensorPath"].startswith("derived/")
    assert residual(manifest_path)["maxRelativeRms"] < 2e-6

    by_component = {item["component"]: item for item in manifest["trace"][-1]["denoised"]}
    video = np.load(manifest_path.parent / by_component["video"]["tensorPath"], allow_pickle=False)
    audio = np.load(manifest_path.parent / by_component["audio"]["tensorPath"], allow_pickle=False)
    assert np.array_equal(video, np.array([99.0, 100.0], dtype=np.float32))
    assert np.array_equal(audio, np.array([28.0, 32.0], dtype=np.float32))
    assert "label" not in manifest
    assert (
        json.loads((root / "recurrence" / "operator-ground-truth.json").read_text())["cases"][
            "recurrence-forged"
        ]["label"]
        == "recurrence_consistent_model_unbound"
    )
