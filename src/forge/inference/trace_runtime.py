"""Concrete Comfy graphs for the restricted H3 trusted-trace runtime.

This is deliberately a graph adapter, not a verdict engine.  ``trace_protocol``
owns canonical request/receipt/profile validation; this module converts its
already-validated values into the exact pinned H3 graph.  It never accepts a
miner MP4 as evidence or as a renderer input.

The initial profile is intentionally narrow: native Blackwell, base 20-step
text-to-video at 832x480, five seconds, 24 fps and audio.  Expanding that set
requires a separately pinned profile because conditioning nodes and tensor
shapes are part of the replay contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_SCHEMA = "forge/h3-trusted-runtime/v1"
CAPTURE_SCHEMA = "forge/inference-trace-capture/v1"
PROFILE_ID = "h3-native-blackwell-20-v1"
UNET = "minimax_h3_fl2va_pruned_nvfp4.safetensors"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
STEPS = 20
FPS = 24
WIDTH = 832
HEIGHT = 480
DURATION_SECONDS = 5
LENGTH = 124  # MiniMax H3's 17-frame alignment for five seconds at 24 fps.
DEFAULT_MAX_BUNDLE_BYTES = 20 * 1024 * 1024 * 1024
RUNTIME_CODE = (
    "src/forge/inference/trace_protocol.py",
    "src/forge/inference/trace_runtime.py",
    "src/forge/inference/trace_eval.py",
    "src/forge/inference/transition_diag.py",
    "scripts/h3_trusted_runtime.py",
    "scripts/h3_trace_custom_node.py",
    "scripts/h3_transition_diagnostic.py",
    "scripts/h3_runtime_measure.py",
)


class RuntimeGraphError(ValueError):
    """A request cannot be safely represented by the pinned runtime graph."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _protocol_profile_hash(value: Mapping[str, Any]) -> str:
    """Use protocol canonical JSON for identities shared with the API."""
    from forge.inference.trace_protocol import sha256

    return sha256(dict(value))


def renderer_fingerprint(code: Mapping[str, str], vae: Mapping[str, str], ffmpeg: str) -> str:
    """Identity shared by startup measurement and the executing verifier."""
    return _protocol_profile_hash(
        {
            "runtime": code["scripts/h3_trusted_runtime.py"],
            "workflow": code["src/forge/inference/trace_runtime.py"],
            "vae": {key: vae[key] for key in ("videoVaeName", "audioVaeName")},
            "ffmpeg": ffmpeg,
            "width": WIDTH,
            "height": HEIGHT,
            "frames": FPS * DURATION_SECONDS,
            "fps": FPS,
            "audioSampleRate": 32000,
            "audioChannels": 2,
        }
    )


def _plain_name(value: str, field: str) -> str:
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise RuntimeGraphError(f"{field}_unsafe")
    return value


def _safe_output_prefix(value: str) -> str:
    """Comfy output prefixes are relative paths, never absolute/parent paths."""
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise RuntimeGraphError("filename_prefix_unsafe")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise RuntimeGraphError(f"{field}_invalid")
    return value


def _absolute_new(path: Path, field: str) -> Path:
    path = path.resolve()
    if not path.is_absolute() or path.exists():
        raise RuntimeGraphError(f"{field}_must_be_new_absolute_path")
    return path


@dataclass(frozen=True)
class NativeH3Profile:
    """The small profile surface that is semantically part of H3 replay."""

    profile_id: str
    profile_hash: str
    model_manifest_hash: str
    workflow_hash: str
    runtime_id: str
    unet: str = UNET
    encoder: str = ENCODER
    video_vae: str = VIDEO_VAE
    audio_vae: str = AUDIO_VAE

    @classmethod
    def from_validated(cls, value: Mapping[str, Any]) -> NativeH3Profile:
        """Construct from protocol-validated profile data, rejecting drift here too."""
        required = {
            "id",
            "modelManifestHash",
            "workflowHash",
            "runtimeHash",
            "runtime",
        }
        if not required <= set(value):
            raise RuntimeGraphError("profile_fields_missing")
        if value["id"] != PROFILE_ID:
            raise RuntimeGraphError("profile_not_supported")
        runtime = value["runtime"]
        if not isinstance(runtime, Mapping):
            raise RuntimeGraphError("profile_runtime_invalid")
        names = {
            "unet": runtime.get("unetName", UNET),
            "encoder": runtime.get("encoderName", ENCODER),
            "video_vae": runtime.get("videoVaeName", VIDEO_VAE),
            "audio_vae": runtime.get("audioVaeName", AUDIO_VAE),
        }
        if names != {
            "unet": UNET,
            "encoder": ENCODER,
            "video_vae": VIDEO_VAE,
            "audio_vae": AUDIO_VAE,
        }:
            raise RuntimeGraphError("checkpoint_names_not_pinned")
        return cls(
            profile_id=PROFILE_ID,
            # Profile identity is the canonical protocol hash of this body;
            # it cannot be stored inside itself without self-reference.
            profile_hash=_protocol_profile_hash(value),
            model_manifest_hash=_sha256(value["modelManifestHash"], "model_manifest_hash"),
            workflow_hash=_sha256(value["workflowHash"], "workflow_hash"),
            runtime_id=_plain_name(str(runtime.get("comfyRevision", "")), "runtime_id"),
        )


@dataclass(frozen=True)
class TrustedT2VInput:
    """Validator-owned values used to make conditioning and noise locally."""

    job_id: str
    nonce: str
    parameters_hash: str
    request_hash: str
    prompt: str
    effective_seed: int

    @classmethod
    def from_validated(cls, value: Mapping[str, Any]) -> TrustedT2VInput:
        required = {
            "jobId",
            "nonce",
            "parametersHash",
            "requestHash",
            "parameters",
            "effectiveSeed",
        }
        if not required <= set(value) or not isinstance(value["parameters"], Mapping):
            raise RuntimeGraphError("trusted_request_fields_missing")
        parameters = value["parameters"]
        # This is deliberately not an API compatibility layer.  References and
        # arbitrary resolutions require their own canonical input-bundle profile.
        if set(parameters) != {
            "modality",
            "prompt",
            "duration",
            "resolution",
            "aspectRatio",
            "audio",
            "seed",
            "profile",
        }:
            raise RuntimeGraphError("trusted_parameters_not_restricted_profile")
        if (
            not isinstance(parameters.get("prompt"), str)
            or not parameters["prompt"].strip()
            or parameters.get("duration") != DURATION_SECONDS
            or parameters.get("resolution") != "480p"
            or parameters.get("aspectRatio") != "16:9"
            or parameters.get("audio") is not True
            or parameters.get("profile") != "base"
            or parameters.get("modality") != "text-to-video"
        ):
            raise RuntimeGraphError("trusted_parameters_not_restricted_profile")
        seed = value["effectiveSeed"]
        if type(seed) is not int or not 0 <= seed < 2**53:
            raise RuntimeGraphError("effective_seed_invalid")
        if parameters["seed"] != seed:
            raise RuntimeGraphError("effective_seed_not_bound_to_parameters")
        return cls(
            job_id=_plain_name(str(value["jobId"]), "job_id"),
            nonce=str(value["nonce"]),
            parameters_hash=_sha256(value["parametersHash"], "parameters_hash"),
            request_hash=_sha256(value["requestHash"], "request_hash"),
            prompt=parameters["prompt"],
            effective_seed=seed,
        )


def _trace_metadata(
    request: TrustedT2VInput, profile: NativeH3Profile, trace_root: Path
) -> dict[str, Any]:
    """Metadata duplicated in the capture to bind it to validator-owned input."""
    return {
        "trace_root": str(trace_root.resolve()),
        "run_id": request.job_id,
        "split": "holdout",
        "condition": "honest_repeat",
        "lineage_id": request.request_hash,
        "input_hash": request.request_hash,
        "model_manifest_hash": profile.model_manifest_hash,
        "workflow_hash": profile.workflow_hash,
        "runtime_id": profile.runtime_id,
    }


def build_trusted_baseline_graph(
    request: TrustedT2VInput,
    profile: NativeH3Profile,
    *,
    trace_root: Path,
    filename_prefix: str,
) -> dict[str, Any]:
    """Build the validator's independent conditioning/noise and 20-step trace.

    ``RandomNoise`` receives only ``effective_seed``.  The protocol derives it
    from job id, nonce, requested seed and profile hash; no miner-provided seed
    or conditioning is accepted by this adapter.
    """
    trace_root = trace_root.resolve()
    if not trace_root.is_absolute():
        raise RuntimeGraphError("trace_root_not_absolute")
    metadata = _trace_metadata(request, profile, trace_root)
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": profile.unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": profile.encoder, "type": "minimax", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": profile.video_vae}},
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": request.prompt,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
            },
        },
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": request.effective_seed}},
        "7": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1},
        },
        "10": {
            "class_type": "H3TraceSamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["7", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["5", 1],
                "capture_all_forwards": True,
                **metadata,
            },
        },
        "17": {
            "class_type": "H3TraceSeal",
            "inputs": {
                "samples": ["10", 0],
                "trace_root": metadata["trace_root"],
                "run_id": metadata["run_id"],
            },
        },
    }
    return graph


def _candidate_forward_requests(
    manifest_path: Path,
    result_directory: Path,
    candidate_label: str,
    *,
    max_bundle_bytes: int,
) -> list[dict[str, str]]:
    """Validate bounded bundle custody before the Comfy node loads with weights_only."""
    manifest_path = manifest_path.resolve()
    result_directory = result_directory.resolve()
    if (
        not manifest_path.is_file()
        or not manifest_path.is_absolute()
        or not result_directory.is_absolute()
    ):
        raise RuntimeGraphError("candidate_paths_invalid")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeGraphError("candidate_manifest_invalid") from exc
    if document.get("schema") != CAPTURE_SCHEMA:
        raise RuntimeGraphError("candidate_manifest_schema_invalid")
    forwards = document.get("forward")
    if not isinstance(forwards, list) or len(forwards) != STEPS:
        raise RuntimeGraphError("candidate_forward_count_not_20")
    if max_bundle_bytes <= 0:
        raise RuntimeGraphError("bundle_limit_invalid")
    root = manifest_path.parent.resolve()
    results: list[dict[str, str]] = []
    total = 0
    for index, descriptor in enumerate(forwards):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "call",
            "tensorPath",
            "sha256",
            "bytes",
        }:
            raise RuntimeGraphError("candidate_forward_descriptor_invalid")
        if descriptor["call"] != index:
            raise RuntimeGraphError("candidate_forward_order_invalid")
        relative = Path(str(descriptor["tensorPath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeGraphError("candidate_forward_path_unsafe")
        bundle = (root / relative).resolve()
        try:
            bundle.relative_to(root)
        except ValueError as exc:
            raise RuntimeGraphError("candidate_forward_path_unsafe") from exc
        if not bundle.is_file() or type(descriptor["bytes"]) is not int or descriptor["bytes"] <= 0:
            raise RuntimeGraphError("candidate_forward_missing")
        if bundle.stat().st_size != descriptor["bytes"] or file_digest(bundle) != _sha256(
            descriptor["sha256"], "candidate_forward_sha256"
        ):
            raise RuntimeGraphError("candidate_forward_custody_invalid")
        total += descriptor["bytes"]
        if total > max_bundle_bytes:
            raise RuntimeGraphError("candidate_forward_bundles_too_large")
        results.append(
            {
                "bundle_path": str(bundle),
                "result_path": str(result_directory / f"forward-{index:03d}.json"),
                "candidate_label": candidate_label,
            }
        )
    if len({item["result_path"] for item in results}) != STEPS:
        raise RuntimeGraphError("candidate_replay_results_duplicate")
    return results


def build_full_forward_replay_graph(
    profile: NativeH3Profile,
    *,
    candidate_manifest: Path,
    result_directory: Path,
    candidate_label: str,
    max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
) -> dict[str, Any]:
    """Replay all 20 claimed fixed inputs with the pinned model, once resident."""
    _plain_name(candidate_label, "candidate_label")
    requests = _candidate_forward_requests(
        candidate_manifest,
        result_directory,
        candidate_label,
        max_bundle_bytes=max_bundle_bytes,
    )
    if any(Path(item["result_path"]).exists() for item in requests):
        raise RuntimeGraphError("candidate_replay_result_exists")
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": profile.unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "H3TraceForwardReplayBatch",
            "inputs": {
                "model": ["1", 0],
                "requests_json": json.dumps(requests, sort_keys=True, separators=(",", ":")),
            },
        },
    }


def build_validated_latent_render_graph(
    profile: NativeH3Profile,
    *,
    candidate_manifest: Path,
    filename_prefix: str,
) -> dict[str, Any]:
    """Render a protocol-validated candidate final latent under validator control.

    The caller must call this only after trace_protocol has bound the manifest's
    final latent to the accepted full replay.  This graph never reads a miner
    video path: its sole candidate input is the two bounded final-latent blobs.
    """
    manifest = candidate_manifest.resolve()
    if not manifest.is_file() or not manifest.is_absolute():
        raise RuntimeGraphError("candidate_manifest_path_invalid")
    _safe_output_prefix(filename_prefix)
    return {
        "1": {"class_type": "H3TraceFinalLatentLoader", "inputs": {"manifest_path": str(manifest)}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": profile.video_vae}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": profile.audio_vae}},
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
            "inputs": {"images": ["4", 0], "audio": ["5", 0], "fps": FPS},
        },
        "7": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["6", 0],
                "filename_prefix": filename_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def write_graph_new(path: Path, graph: Mapping[str, Any]) -> str:
    """Atomically write one controller-owned graph; never overwrite an intent."""
    path = _absolute_new(path, "graph_output")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(canonical(graph))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest(path.read_bytes())


def runtime_intent(
    request: TrustedT2VInput,
    profile: NativeH3Profile,
    *,
    phase: str,
    graph_sha256: str,
) -> dict[str, Any]:
    """Small immutable binding emitted by the controller beside each graph."""
    if phase not in {"trusted-baseline", "full-forward-replay", "validated-render"}:
        raise RuntimeGraphError("runtime_phase_invalid")
    return {
        "schema": RUNTIME_SCHEMA,
        "phase": phase,
        "jobId": request.job_id,
        "requestHash": request.request_hash,
        "parametersHash": request.parameters_hash,
        "profileId": profile.profile_id,
        "profileHash": profile.profile_hash,
        "modelManifestHash": profile.model_manifest_hash,
        "workflowHash": profile.workflow_hash,
        "effectiveSeed": request.effective_seed,
        "graphSha256": _sha256(graph_sha256, "graph_sha256"),
    }
