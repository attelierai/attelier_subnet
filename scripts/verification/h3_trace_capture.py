#!/usr/bin/env python3
"""Install H3 trace nodes and rewrite one private Comfy graph for an experiment.

The utility deliberately cannot modify a production graph in place.  The caller
supplies an output file and an experiment root on the remote worker.  It does
not render, contact a miner, upload trace data or make a verifier decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

SCHEMA = "forge/inference-trace-capture/v1"
CONDITIONS = {
    "honest_repeat",
    "steps_10",
    "steps_3",
    "alt_checkpoint",
    "lora",
    "wrong_output",
    "padded_steps_10",
    "sidecar_honest_wrong_output",
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def safe_absolute(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("trace root must be absolute")
    return str(path)


def next_id(graph: dict[str, Any]) -> str:
    numeric = [int(key) for key in graph if key.isdecimal()]
    return str(max(numeric, default=0) + 1)


def metadata(args: argparse.Namespace) -> dict[str, str]:
    if args.condition not in CONDITIONS:
        raise ValueError("unknown trace condition")
    values = {
        "trace_root": safe_absolute(args.trace_root),
        "run_id": args.run_id,
        "split": args.split,
        "condition": args.condition,
        "lineage_id": args.lineage_id,
        "input_hash": args.input_hash,
        "model_manifest_hash": args.model_manifest_hash,
        "workflow_hash": args.workflow_hash,
        "runtime_id": args.runtime_id,
    }
    if not all(values.values()) or any(".." in str(value) for value in values.values()):
        raise ValueError("trace metadata missing or unsafe")
    return values


def add_trace_graph(document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    graph = document.get("graph", document)
    if not isinstance(graph, dict) or not all(isinstance(value, dict) for value in graph.values()):
        raise ValueError("Comfy graph object required")
    for key in ("10", "11", "12", "13"):
        if key not in graph:
            raise ValueError(f"H3 workflow node {key} missing")
    if graph["10"].get("class_type") != "SamplerCustomAdvanced":
        raise ValueError("expected unmodified H3 sampler node")
    if (
        graph["11"].get("class_type") != "VAEDecodeTiled"
        or graph["12"].get("class_type") != "VAEDecodeAudio"
    ):
        raise ValueError("expected H3 VAE decode nodes")
    if graph["13"].get("class_type") != "CreateVideo":
        raise ValueError("expected H3 CreateVideo node")
    values = metadata(args)
    node_inputs = {key: value for key, value in values.items()}
    graph["10"] = {
        "class_type": "H3TraceSamplerCustomAdvanced",
        "inputs": {**graph["10"]["inputs"], **node_inputs},
    }
    images, audio = next_id(graph), None
    graph[images] = {"class_type": "H3TraceImages", "inputs": {"images": ["11", 0], **node_inputs}}
    graph["13"]["inputs"]["images"] = [images, 0]
    if args.audio:
        audio = next_id(graph)
        graph[audio] = {"class_type": "H3TraceAudio", "inputs": {"audio": ["12", 0], **node_inputs}}
        graph["13"]["inputs"]["audio"] = [audio, 0]
    document["traceExperiment"] = {
        "schema": SCHEMA,
        "runId": args.run_id,
        "inputHash": args.input_hash,
        "graphHash": sha256(canonical(graph)),
        "samplerNode": "10",
        "videoNode": images,
        "audioNode": audio,
    }
    return document


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite graph output")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(data)
    os.chmod(path, 0o600)


def command_prepare(args: argparse.Namespace) -> None:
    source, target = Path(args.graph), Path(args.output)
    if source.resolve() == target.resolve():
        raise ValueError("trace graph output must be distinct")
    document = json.loads(source.read_text())
    result = add_trace_graph(document, args)
    write_new(target, canonical(result) + b"\n")
    print(
        json.dumps(
            {"schema": SCHEMA, "graph": str(target), **result["traceExperiment"]}, sort_keys=True
        )
    )


def command_install(args: argparse.Namespace) -> None:
    source = Path(__file__).with_name("h3_trace_custom_node.py")
    comfy = Path(args.comfy_dir).resolve()
    if not (comfy / "main.py").is_file():
        raise ValueError("expected ComfyUI checkout")
    target = comfy / "custom_nodes" / "attelier_h3_trace" / "__init__.py"
    if target.exists() and not args.replace:
        raise FileExistsError(
            "trace custom node already exists; use --replace only after inspection"
        )
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_suffix(".partial")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(temporary, target)
    print(
        json.dumps(
            {"installed": str(target), "sha256": sha256(target.read_bytes())}, sort_keys=True
        )
    )


def command_record_output(args: argparse.Namespace) -> None:
    root = Path(safe_absolute(args.trace_root)).resolve()
    manifest_path = root / args.run_id / "manifest.json"
    media = Path(args.media).resolve()
    if not manifest_path.is_file() or not media.is_file():
        raise ValueError("trace manifest or rendered media missing")
    document = json.loads(manifest_path.read_text())
    if document.get("schema") != SCHEMA or document.get("runId") != args.run_id:
        raise ValueError("trace manifest identity invalid")
    output = {
        "mediaSha256": sha256(media.read_bytes()),
        "canonicalVideoSha256": args.canonical_video_sha256,
        "canonicalAudioSha256": args.canonical_audio_sha256,
    }
    if not all(
        len(value) == 64 and set(value) <= set("0123456789abcdef") for value in output.values()
    ):
        raise ValueError("output hashes must be lowercase SHA-256")
    if document.get("output") not in (None, output):
        raise ValueError("trace output already bound differently")
    document["output"] = output
    temporary = manifest_path.with_suffix(".partial")
    temporary.write_bytes(canonical(document))
    os.chmod(temporary, 0o600)
    os.replace(temporary, manifest_path)
    print(json.dumps({"schema": SCHEMA, "runId": args.run_id, "output": output}, sort_keys=True))


def command_prepare_replay(args: argparse.Namespace) -> None:
    bundle, result, output = (
        Path(args.bundle).resolve(),
        Path(args.result).resolve(),
        Path(args.output),
    )
    if not bundle.is_file() or result.exists() or not result.is_absolute():
        raise ValueError("bundle missing or replay result unsafe")
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": args.unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "H3TraceForwardReplay",
            "inputs": {
                "model": ["1", 0],
                "bundle_path": str(bundle),
                "result_path": str(result),
                "candidate_label": args.candidate_label,
            },
        },
    }
    write_new(output, canonical(graph) + b"\n")
    print(
        json.dumps(
            {"schema": "forge/inference-forward-replay/v1", "graph": str(output)}, sort_keys=True
        )
    )


def command_prepare_vae_replay(args: argparse.Namespace) -> None:
    manifest, output = Path(args.manifest).resolve(), Path(args.output)
    if not manifest.is_file() or not manifest.is_absolute():
        raise ValueError("trace manifest missing")
    if (
        not args.audit_prefix
        or ".." in Path(args.audit_prefix).parts
        or Path(args.audit_prefix).is_absolute()
    ):
        raise ValueError("audit prefix unsafe")
    graph = {
        "1": {"class_type": "H3TraceFinalLatentLoader", "inputs": {"manifest_path": str(manifest)}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": args.video_vae}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": args.audio_vae}},
        "4": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["1", 0],
                "vae": ["2", 0],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "5": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["1", 0], "vae": ["3", 0]}},
        "6": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["4", 0], "audio": ["5", 0], "fps": 24},
        },
        "7": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["6", 0],
                "filename_prefix": args.audit_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }
    write_new(output, canonical(graph) + b"\n")
    print(
        json.dumps(
            {"schema": "forge/inference-vae-replay/v1", "graph": str(output)}, sort_keys=True
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install")
    install.add_argument("--comfy-dir", required=True)
    install.add_argument("--replace", action="store_true")
    install.set_defaults(run=command_install)
    graph = commands.add_parser("prepare-graph")
    graph.add_argument("--graph", required=True)
    graph.add_argument("--output", required=True)
    graph.add_argument("--trace-root", required=True)
    graph.add_argument("--run-id", required=True)
    graph.add_argument("--split", choices=("calibration", "holdout"), required=True)
    graph.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    graph.add_argument("--lineage-id", required=True)
    graph.add_argument("--input-hash", required=True)
    graph.add_argument("--model-manifest-hash", required=True)
    graph.add_argument("--workflow-hash", required=True)
    graph.add_argument("--runtime-id", required=True)
    graph.add_argument("--audio", action="store_true")
    graph.set_defaults(run=command_prepare)
    recorded = commands.add_parser("record-output")
    recorded.add_argument("--trace-root", required=True)
    recorded.add_argument("--run-id", required=True)
    recorded.add_argument("--media", required=True)
    recorded.add_argument("--canonical-video-sha256", required=True)
    recorded.add_argument("--canonical-audio-sha256", required=True)
    recorded.set_defaults(run=command_record_output)
    replay = commands.add_parser("prepare-replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--result", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--unet", required=True)
    replay.add_argument("--candidate-label", required=True)
    replay.set_defaults(run=command_prepare_replay)
    vae = commands.add_parser("prepare-vae-replay")
    vae.add_argument("--manifest", required=True)
    vae.add_argument("--output", required=True)
    vae.add_argument("--audit-prefix", required=True)
    vae.add_argument("--video-vae", default="minimax_h3_video_vae_fp16.safetensors")
    vae.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
    vae.set_defaults(run=command_prepare_vae_replay)
    return result


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        parsed.run(parsed)
    except Exception as error:
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
