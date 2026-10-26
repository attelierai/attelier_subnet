"""The private trace graph uses Comfy's real graph contract without a GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_trace_graph_isolated_and_captures_sampler_and_vae_boundaries(tmp_path):
    root = Path(__file__).parents[2]
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0]}},
        "11": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["10", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0]}},
    }
    source, output = tmp_path / "base.json", tmp_path / "trace.json"
    source.write_text(json.dumps(graph))
    values = [
        sys.executable,
        str(root / "scripts" / "h3_trace_capture.py"),
        "prepare-graph",
        "--graph",
        str(source),
        "--output",
        str(output),
        "--trace-root",
        "/workspace/forge/trace-x",
        "--run-id",
        "run-a",
        "--split",
        "calibration",
        "--condition",
        "honest_repeat",
        "--lineage-id",
        "lineage-a",
        "--input-hash",
        "a" * 64,
        "--model-manifest-hash",
        "b" * 64,
        "--workflow-hash",
        "c" * 64,
        "--runtime-id",
        "comfy-b0f4b7",
        "--audio",
    ]
    result = subprocess.run(values, text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    prepared = json.loads(output.read_text())
    assert receipt["schema"] == "forge/inference-trace-capture/v1"
    assert graph["10"]["class_type"] == "SamplerCustomAdvanced", "input graph was never changed"
    assert prepared["10"]["class_type"] == "H3TraceSamplerCustomAdvanced"
    assert "skip_model_step" not in prepared["10"]["inputs"], "default graph remains unchanged"
    video = prepared[prepared["traceExperiment"]["videoNode"]]
    audio = prepared[prepared["traceExperiment"]["audioNode"]]
    assert video["inputs"]["images"] == ["11", 0]
    assert audio["inputs"]["audio"] == ["12", 0]
    assert prepared["13"]["inputs"] == {
        "images": [prepared["traceExperiment"]["videoNode"], 0],
        "audio": [prepared["traceExperiment"]["audioNode"], 0],
    }
    assert output.stat().st_mode & 0o077 == 0

    bundle = tmp_path / "forward.pt"
    bundle.write_bytes(b"fixture trace bundle")
    replay_graph = tmp_path / "replay.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "h3_trace_capture.py"),
            "prepare-replay",
            "--bundle",
            str(bundle),
            "--result",
            str(tmp_path / "replay-result.json"),
            "--output",
            str(replay_graph),
            "--unet",
            "approved.safetensors",
            "--candidate-label",
            "approved",
        ],
        check=True,
    )
    replay = json.loads(replay_graph.read_text())
    assert replay["2"]["class_type"] == "H3TraceForwardReplay"
    assert replay["2"]["inputs"]["model"] == ["1", 0]

    latent_manifest = tmp_path / "latent-manifest.json"
    latent_manifest.write_text(json.dumps({"schema": "forge/inference-trace-capture/v1"}))
    vae_graph = tmp_path / "vae-replay.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "h3_trace_capture.py"),
            "prepare-vae-replay",
            "--manifest",
            str(latent_manifest),
            "--output",
            str(vae_graph),
            "--audit-prefix",
            "attelier/audit/unique-run",
        ],
        check=True,
    )
    vae = json.loads(vae_graph.read_text())
    assert vae["1"]["class_type"] == "H3TraceFinalLatentLoader"
    assert vae["4"]["inputs"]["samples"] == ["1", 0]
    assert vae["6"]["inputs"]["fps"] == 24
    assert vae["7"]["inputs"]["codec"] == "h264"
