#!/usr/bin/env python3
"""Measure an installed, operator-controlled H3 runtime before accepting jobs.

Runs on the GPU host. A new profile is always UNQUALIFIED; this measurement
does not replace the independent honest/adversarial qualification campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from forge.inference.trace_protocol import (
    FIXED_PARAMETERS,
    PROFILE_SCHEMA,
    canonical,
    file_sha256,
    read_json,
    sha256,
)
from forge.inference.trace_runtime import RUNTIME_CODE, renderer_fingerprint

CHECKPOINTS = {
    "unetName": "diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors",
    "encoderName": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "videoVaeName": "vae/minimax_h3_video_vae_fp16.safetensors",
    "audioVaeName": "vae/minimax_h3_audio_vae_fp32.safetensors",
}
# These top-level runtime-data directories are not imported as Comfy source.
# They may be mounted or symlinked by the GPU host and are outside source attestation.
RUNTIME_DATA_DIRECTORIES = {"models", "input", "output", "temp", "user", ".git"}


def command(argv: list[str]) -> str:
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_name(name: str) -> bool:
    basename = PurePosixPath(name).name
    return name.endswith(".py") or (
        basename.startswith("requirements") and basename.endswith(".txt")
    )


def archive_source_tree(archive_path: Path, revision: str) -> tuple[dict[str, str], str]:
    """Hash source files directly from a pinned GitHub archive; never extract it."""
    prefix = f"ComfyUI-{revision}"
    source: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                name = member.name
                path = PurePosixPath(name)
                if name.startswith("/") or "\\" in name or ".." in path.parts:
                    raise RuntimeError("comfy_archive_path_unsafe")
                if path == PurePosixPath(prefix):
                    if not member.isdir():
                        raise RuntimeError("comfy_archive_prefix_invalid")
                    continue
                if not name.startswith(prefix + "/"):
                    raise RuntimeError("comfy_archive_prefix_invalid")
                relative = name[len(prefix) + 1 :]
                if not relative or relative.startswith("/") or not source_name(relative):
                    if not member.isdir() and not member.isfile():
                        raise RuntimeError("comfy_archive_member_unsafe")
                    continue
                if not member.isfile() or relative in source:
                    raise RuntimeError("comfy_archive_member_unsafe")
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("comfy_archive_member_unreadable")
                digest = hashlib.sha256()
                block = stream.read(1024 * 1024)
                while block:
                    digest.update(block)
                    block = stream.read(1024 * 1024)
                source[relative] = digest.hexdigest()
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("comfy_archive_unavailable") from exc
    if not source or "requirements.txt" not in source:
        raise RuntimeError("comfy_archive_source_incomplete")
    return source, file_sha256(archive_path)


def trace_node_relative(comfy: Path, trace_node: Path, expected_hash: str) -> str:
    root = Path(os.path.abspath(comfy))
    node = Path(os.path.abspath(trace_node))
    try:
        relative = node.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("installed_trace_node_outside_comfy") from exc
    if (
        not relative.parts
        or relative.parts[0] in RUNTIME_DATA_DIRECTORIES
        or ".." in relative.parts
        or node.is_symlink()
        or not node.is_file()
        or file_sha256(node) != expected_hash
    ):
        raise RuntimeError("installed_trace_node_changed")
    return relative.as_posix()


def installed_source_tree(comfy: Path) -> dict[str, str]:
    source: dict[str, str] = {}
    for root, directories, files in os.walk(comfy, followlinks=False):
        root_path = Path(root)
        if root_path != comfy and root_path.is_symlink():
            raise RuntimeError("comfy_installed_symlink")
        for directory in tuple(directories):
            candidate = root_path / directory
            if root_path == comfy and directory in RUNTIME_DATA_DIRECTORIES:
                directories.remove(directory)
            elif candidate.is_symlink():
                raise RuntimeError("comfy_installed_symlink")
        for name in files:
            candidate = root_path / name
            relative = candidate.relative_to(comfy).as_posix()
            if not source_name(relative):
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError("comfy_installed_symlink")
            source[relative] = file_sha256(candidate)
    return source


def validate_archive_source_tree(
    comfy: Path, archive: Path, lock: dict, trace_node: Path, trace_node_hash: str
) -> tuple[dict[str, str], str]:
    revision = lock["comfy"]["revision"]
    expected, archive_sha256 = archive_source_tree(archive, revision)
    observed = installed_source_tree(comfy)
    trace_node_name = trace_node_relative(comfy, trace_node, trace_node_hash)
    patch = lock["comfy"]["nvfp4ScalePatch"]
    target = PurePosixPath(patch["target"])
    if target.is_absolute() or ".." in target.parts or target.as_posix() not in expected:
        raise RuntimeError("quantization_patch_target_invalid")
    target_name = target.as_posix()
    if trace_node_name in expected or set(observed) != {*expected, trace_node_name}:
        raise RuntimeError("comfy_source_tree_changed")
    for name, digest in expected.items():
        if name == target_name:
            if observed[name] != patch["afterSha256"]:
                raise RuntimeError("quantization_patch_changed")
        elif observed[name] != digest:
            raise RuntimeError("comfy_source_tree_changed")
    if observed[trace_node_name] != trace_node_hash:
        raise RuntimeError("installed_trace_node_changed")
    return observed, archive_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("code-root", "comfy-dir", "checkpoint-lock", "trace-node", "out"):
        parser.add_argument("--" + argument, required=True, type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare-profile", type=Path)
    choice.add_argument("--expected-profile", type=Path)
    parser.add_argument("--recurrence-relative-rms", default="0.0000011514888827434288")
    parser.add_argument("--comfy-origin", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-source-archive", type=Path)
    args = parser.parse_args()
    import torch  # Only available in the explicitly installed GPU runtime.

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one_visible_cuda_device_required")
    if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise RuntimeError("native_blackwell_sm120_profile_required")
    comfy = args.comfy_dir.resolve()
    lock = read_json(args.checkpoint_lock)
    code = {relative: file_sha256(args.code_root / relative) for relative in RUNTIME_CODE}
    archive_sha256 = None
    if args.comfy_source_archive:
        revision = lock["comfy"]["revision"]
        comfy_code, archive_sha256 = validate_archive_source_tree(
            comfy,
            args.comfy_source_archive,
            lock,
            args.trace_node,
            code["scripts/verification/h3_trace_custom_node.py"],
        )
    else:
        revision = command(["git", "-C", str(comfy), "rev-parse", "HEAD"])
        if revision != lock["comfy"]["revision"]:
            raise RuntimeError("comfy_revision_changed")
    patch = lock["comfy"]["nvfp4ScalePatch"]
    if file_sha256(comfy / patch["target"]) != patch["afterSha256"]:
        raise RuntimeError("quantization_patch_changed")
    if file_sha256(comfy / "requirements.txt") != lock["comfy"]["requirementsSha256"]:
        raise RuntimeError("comfy_requirements_changed")
    files = {item["destination"]: item for item in lock["files"]}
    measured = {}
    for key, relative in CHECKPOINTS.items():
        expected = files[relative]
        path = comfy / "models" / relative
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError("checkpoint_size_changed:" + key)
        observed = file_sha256(path)
        if observed != expected["sha256"]:
            raise RuntimeError("checkpoint_bytes_changed:" + key)
        measured[key] = observed
    if file_sha256(args.trace_node) != code["scripts/verification/h3_trace_custom_node.py"]:
        raise RuntimeError("installed_trace_node_changed")
    if not args.comfy_source_archive:
        tracked = subprocess.run(
            ["git", "-C", str(comfy), "ls-files", "-z"], check=True, capture_output=True, timeout=30
        ).stdout.split(b"\0")
        python_files = [name.decode() for name in tracked if name.endswith(b".py")]
        comfy_code = {name: file_sha256(comfy / name) for name in sorted(python_files)}
    if not args.comfy_origin.startswith("http://127.0.0.1:"):
        raise RuntimeError("comfy_must_be_operator_loopback")
    with urllib.request.urlopen(
        args.comfy_origin.rstrip("/") + "/system_stats", timeout=10
    ) as response:
        running = json.load(response)
    launch = running.get("system", {}).get("argv")
    if not isinstance(launch, list) or "--cache-none" not in launch:
        raise RuntimeError("independent_preprocessing_requires_cache_none")
    facts = {
        "comfyRevision": revision,
        "comfyPythonTreeHash": sha256(comfy_code),
        **({"comfySourceArchiveSha256": archive_sha256} if archive_sha256 else {}),
        "comfyLaunchArgs": launch,
        "quantOpsSha256": patch["afterSha256"],
        "code": code,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "packages": sorted(
            [
                {"name": item.metadata["Name"], "version": item.version}
                for item in importlib.metadata.distributions()
            ],
            key=lambda item: (item["name"], item["version"]),
        ),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "ffmpeg": command(["ffmpeg", "-version"]).splitlines()[0],
    }
    model_hash = file_sha256(args.checkpoint_lock)
    runtime_hash = sha256(facts)
    workflow_hash = code["src/forge/inference/trace_runtime.py"]
    renderer_hash = renderer_fingerprint(code, measured, facts["ffmpeg"])
    profile = {
        "id": "h3-native-blackwell-20-v1",
        "modelManifestHash": model_hash,
        "workflowHash": workflow_hash,
        "runtimeHash": runtime_hash,
        "rendererHash": renderer_hash,
        "steps": 20,
        "parameters": FIXED_PARAMETERS,
        # This bound applies to floating-point sampler recurrence only.
        # All model inputs, predictions and final latents are compared exactly.
        "numerics": {"maxRelativeRms": args.recurrence_relative_rms},
        "limits": {"maxChunks": 256, "maxChunkBytes": 2 * 1024**3, "maxBundleBytes": 20 * 1024**3},
        "runtime": {
            "comfyRevision": revision,
            **{key: Path(name).name for key, name in CHECKPOINTS.items()},
            "checkpointSha256": measured,
            "numerics": {"mode": "bit_exact"},
            "measurement": facts,
        },
    }
    if args.prepare_profile:
        write_new(
            args.prepare_profile,
            {
                "schema": PROFILE_SCHEMA,
                "profile": profile,
                "qualification": {"status": "unqualified", "evidenceSha256": None, "expiresAt": 0},
            },
        )
    else:
        expected = read_json(args.expected_profile)
        if canonical(expected["profile"]) != canonical(profile):
            raise RuntimeError("measured_runtime_differs_from_profile")
    attestation = {
        "schema": "forge-h3-runtime-attestation/v2",
        "runtimeHash": runtime_hash,
        "rendererHash": renderer_hash,
        "modelManifestHash": model_hash,
        "comfyRevision": revision,
        "checkpoints": {key: Path(name).name for key, name in CHECKPOINTS.items()},
        "checkpointSha256": measured,
    }
    write_new(args.out, attestation)
    print(
        json.dumps(
            {
                "status": "MEASURED",
                "profileId": profile["id"],
                "profileHash": sha256(profile),
                "runtimeHash": runtime_hash,
                "attestationSha256": file_sha256(args.out),
                "qualified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
