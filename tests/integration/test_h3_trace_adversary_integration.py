"""Filesystem-level controlled bypass fixtures, without GPU or tensor loading."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output(run_id: str, media: str, video: str, audio: str) -> dict[str, object]:
    return {
        "runId": run_id,
        "status": "completed",
        "sha256": media * 64,
        "video": {"sha256": video * 64},
        "audio": {"sha256": audio * 64},
    }


def _capture(root: Path, name: str, steps: int, output: dict) -> Path:
    directory = root / name
    (directory / "tensors").mkdir(parents=True)
    trace = []
    for step in range(steps):
        tensor = directory / "tensors" / f"{step}.npy"
        np.save(
            tensor, np.array([float(step), float(step + 1)], dtype=np.float32), allow_pickle=False
        )
        descriptor = {
            "component": "video",
            "tensorPath": f"tensors/{step}.npy",
            "sha256": _sha(tensor),
        }
        trace.append(
            {
                "step": step,
                "totalSteps": steps,
                "sigma": step / 10,
                "stateCallback": [descriptor],
                "denoised": [descriptor],
            }
        )
    retained = dict(trace[0]["stateCallback"][0])
    manifest = {
        "schema": "forge/inference-trace-capture/v1",
        "runId": name,
        "lineageId": "same",
        "inputHash": "x" * 64,
        "capture": {"modelManifestHash": "m", "workflowHash": "w", "runtime": "r"},
        "condition": "honest_repeat",
        "declaredSteps": steps,
        "callbackSteps": steps,
        "trace": trace,
        "inputNoise": [retained],
        "finalLatent": [retained],
        "forward": [{"tensorPath": retained["tensorPath"], "sha256": retained["sha256"]}],
        "decoded": {"audioWaveform": retained},
        "output": output,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_adversary_fixtures_copy_raw_blobs_and_keep_labels_out_of_detector_manifests(tmp_path):
    root = tmp_path / "experiment"
    honest = _capture(root, "honest", 20, _output("honest", "a", "b", "c"))
    short = _capture(root, "short", 10, _output("short", "a", "b", "c"))
    alternate = _capture(root, "alternate", 20, _output("alternate", "d", "e", "f"))
    script = Path(__file__).parents[2] / "scripts" / "h3_trace_adversary.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiment-root",
            str(root),
            "--honest-20",
            str(honest),
            "--actual-10",
            str(short),
            "--alternate-video",
            str(alternate),
            "--output",
            str(root / "adversary"),
        ],
        check=True,
    )
    padded = json.loads((root / "adversary" / "padded-10-as-20" / "manifest.json").read_text())
    sidecar = json.loads((root / "adversary" / "sidecar-output" / "manifest.json").read_text())
    assert padded["declaredSteps"] == padded["callbackSteps"] == len(padded["trace"]) == 20
    assert padded["trace"][0]["stateCallback"][0]["tensorPath"].startswith("derived/")
    assert (
        root
        / "adversary"
        / "padded-10-as-20"
        / padded["trace"][0]["stateCallback"][0]["tensorPath"]
    ).is_file()
    assert padded["finalLatent"][0]["tensorPath"].startswith("artifacts/raw/")
    assert padded["inputNoise"][0]["tensorPath"].startswith("artifacts/raw/")
    assert padded["decoded"]["audioWaveform"]["tensorPath"].startswith("artifacts/raw/")
    assert padded["runId"] != "short" and sidecar["runId"] != "honest"
    assert padded["output"]["runId"] == padded["runId"]
    assert sidecar["output"]["runId"] == sidecar["runId"]
    assert sidecar["output"]["sha256"] == "d" * 64
    assert "label" not in padded and "label" not in sidecar
    assert (
        json.loads((root / "adversary" / "operator-ground-truth.json").read_text())["cases"][
            "padded-10-as-20"
        ]["label"]
        == "padded_steps_10"
    )
