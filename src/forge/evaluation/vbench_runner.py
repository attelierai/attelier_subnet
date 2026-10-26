"""Validator-owned VBench runner against a pinned, pre-provisioned local checkout.

Provision dependencies and metric weights on the authorized remote worker first.
Nothing here downloads candidate code or fits/calibrates quality thresholds.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from forge.evaluation.vbench import (
    VBENCH_COMMIT,
    normalize_native_score,
    validate_metric_evidence,
)
from forge.protocol.hashing import hash_canonical, sha256_file

# Native VBench names differ from the MVP's public metric names.
NATIVE_METRICS = {
    "subject_consistency": "subject_consistency",
    "background_consistency": "background_consistency",
    "imaging_quality": "imaging_quality",
    "motion_smoothness": "motion_smoothness",
    "dynamic_degree": "dynamic_degree",
}
WEIGHT_PATHS = {
    "subject_consistency": "dino_model/dino_vitbase16_pretrain.pth",
    "background_consistency": "clip_model/ViT-B-32.pt",
    "imaging_quality": "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
    "motion_smoothness": "amt_model/amt-s.pth",
    "dynamic_degree": "raft_model/models/raft-things.pth",
}


def _pinned_checkout(path: Path, revision: str) -> None:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("an immutable metric source revision is required")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != revision or dirty:
        raise ValueError("metric source checkout differs from pinned clean revision")


def run_vbench(
    *,
    video: Path,
    prompt: str,
    output: Path,
    checkout: Path,
    cache: Path,
    weight_sha256: dict[str, str],
    dino_revision: str,
    runner_image_digest: str,
    plan_id: str,
    prompt_id: str,
    seed: int,
    manifest_hash: str,
) -> dict[str, Any]:
    """Measure one real video, binding native scores to its exact bytes.

    Call only inside reserved GPU work. A new output directory is mandatory:
    an existing run, including a partial run, is never silently retried.
    """
    for digest in (plan_id, manifest_hash):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("frozen plan and model hashes are required")
    if not prompt_id or type(seed) is not int or seed < 0:
        raise ValueError("frozen prompt and seed are required")
    checkout, cache, video = checkout.resolve(), cache.resolve(), video.resolve()
    validate_metric_evidence(
        {
            "provenance": {
                "implementation_commit": VBENCH_COMMIT,
                "weight_sha256": weight_sha256,
                "runner_image_digest": runner_image_digest,
            },
            "scores": dict.fromkeys(NATIVE_METRICS, 0.0),
        }
    )
    _pinned_checkout(checkout, VBENCH_COMMIT)
    _pinned_checkout(cache / "dino_model/facebookresearch_dino_main", dino_revision)
    for metric, relative in WEIGHT_PATHS.items():
        weight = cache / relative
        if not weight.is_file() or sha256_file(weight) != weight_sha256[metric]:
            raise ValueError(f"missing or mismatched validator metric weight: {metric}")
    if not prompt.strip() or video.suffix.lower() != ".mp4" or not video.is_file():
        raise ValueError("a real MP4 and its prescribed prompt are required")
    if "vbench" in sys.modules:
        raise ValueError("run VBench in a fresh dedicated interpreter")
    output.mkdir(parents=True, exist_ok=False)
    video_hash = sha256_file(video)
    # All declared metric weights were checked before any metric loader runs.
    # These pinned VBench checkpoints use legacy torch.load serialization. Only
    # enable compatibility after all operator metric weight hashes are checked;
    # candidate model weights never enter this loader. This is a dedicated child.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["VBENCH_CACHE_DIR"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(checkout))
    sys.dont_write_bytecode = True
    module = importlib.import_module("vbench")
    if Path(module.__file__).resolve() != checkout / "vbench/__init__.py":
        raise ValueError("imported VBench is not the approved checkout")
    import torch

    background = importlib.import_module("vbench.background_consistency")
    native_background = background.background_consistency

    def background_float32(model, *args, **kwargs):
        # CLIP encodes in fp16 on CUDA. Normalize and compare the *same encoded
        # features* in fp32: fp16 cosine can exceed one by more than 1e-4 on
        # frozen clips. Do not widen the existing 1e-6 boundary-roundoff rule.
        class Float32Features:
            def encode_image(self, images):
                return model.encode_image(images).float()

        return native_background(Float32Features(), *args, **kwargs)

    background.background_consistency = background_float32

    # DINO's local hubconf still resolves pretrained weights through Torch Hub,
    # rather than the VBench dino_model directory. Bind that second cache to
    # the verified file so neither a download nor an unrelated global cache is used.
    hub = cache / "torch_hub"
    checkpoint = hub / "checkpoints" / "dino_vitbase16_pretrain.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists() and not checkpoint.is_symlink():
        checkpoint.symlink_to(cache / WEIGHT_PATHS["subject_consistency"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != weight_sha256["subject_consistency"]:
        raise ValueError("Torch Hub DINO checkpoint differs from verified metric weight")
    torch.hub.set_dir(str(hub))

    if not torch.cuda.is_available():
        raise RuntimeError("VBench MVP runner requires the reserved CUDA worker")
    started = time.monotonic()
    benchmark = module.VBench(
        torch.device("cuda"), str(checkout / "VBench_full_info.json"), str(output)
    )
    benchmark.evaluate(
        str(video),
        "metrics",
        prompt_list=[prompt],
        dimension_list=list(NATIVE_METRICS.values()),
        local=True,
        mode="custom_input",
    )
    raw = json.loads((output / "metrics_eval_results.json").read_text())
    scores = {}
    raw_scores = {}
    for metric, native in NATIVE_METRICS.items():
        aggregate, rows = raw[native]
        if len(rows) != 1 or Path(rows[0]["video_path"]).resolve() != video:
            raise ValueError("VBench output did not measure the exact requested video")
        # Imaging-quality aggregate is normalized by100 upstream; per-video value is not.
        raw_scores[metric] = float(aggregate)
        scores[metric] = normalize_native_score(raw_scores[metric])
    if sha256_file(video) != video_hash:
        raise ValueError("video changed during VBench execution")
    receipt = {
        "schema": "forge/vbench-run/v1",
        "video_sha256": video_hash,
        "plan_id": plan_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "manifest_hash": manifest_hash,
        "prompt_hash": hash_canonical("forge/mvp-prompt/v1", prompt),
        "provenance": {
            "implementation_commit": VBENCH_COMMIT,
            "weight_sha256": weight_sha256,
            "runner_image_digest": runner_image_digest,
        },
        "dino_revision": dino_revision,
        "runner_source_sha256": sha256_file(__file__),
        "metric_precision": "CLIP feature normalization and cosine in float32/v1",
        "scores": scores,
        "raw_normalized_scores": raw_scores,
        "normalization": "clip float32 boundary roundoff within 1e-6; reject larger excursions",
        "operational_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "pyiqa", "decord", "numpy")
        },
        "raw_sha256": sha256_file(output / "metrics_eval_results.json"),
        "calibrated": False,
    }
    validate_metric_evidence(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def _validate_isolated_receipt(
    receipt: object,
    *,
    video: Path,
    output: Path,
    prompt: str,
    weight_sha256: dict[str, str],
    dino_revision: str,
    runner_image_digest: str,
    plan_id: str,
    prompt_id: str,
    seed: int,
    manifest_hash: str,
) -> dict[str, Any]:
    """Accept only a child receipt bound to this exact frozen metric request."""

    if not isinstance(receipt, dict):
        raise ValueError("isolated VBench worker did not write a receipt object")
    validate_metric_evidence(receipt)
    expected = {
        "schema": "forge/vbench-run/v1",
        "video_sha256": sha256_file(video),
        "plan_id": plan_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "manifest_hash": manifest_hash,
        "prompt_hash": hash_canonical("forge/mvp-prompt/v1", prompt),
        "provenance": {
            "implementation_commit": VBENCH_COMMIT,
            "weight_sha256": weight_sha256,
            "runner_image_digest": runner_image_digest,
        },
        "dino_revision": dino_revision,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("isolated VBench receipt differs from the frozen metric request")
    raw_output = output / "metrics_eval_results.json"
    if (
        not raw_output.is_file()
        or not isinstance(receipt.get("raw_sha256"), str)
        or receipt["raw_sha256"] != sha256_file(raw_output)
    ):
        raise ValueError("isolated VBench receipt lacks the exact raw metric output")
    return receipt


def run_vbench_isolated(
    *,
    video: Path,
    prompt: str,
    output: Path,
    checkout: Path,
    cache: Path,
    weight_sha256: dict[str, str],
    dino_revision: str,
    runner_image_digest: str,
    interpreter: Path,
    plan_id: str,
    prompt_id: str,
    seed: int,
    manifest_hash: str,
    timeout_seconds: float = 1800,
) -> dict[str, Any]:
    """Run one VBench metric job in a fresh interpreter and verify its receipt.

    VBench mutates module/global CUDA state and deliberately rejects a second
    in-process invocation.  The validator therefore owns a short-lived child
    interpreter for every video; miner bytes never determine its command.
    """

    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 1800
    ):
        raise ValueError("metric timeout must be positive and at most 1800 seconds")
    output = output.resolve()
    if output.exists():
        raise ValueError("isolated VBench output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "video": str(video.resolve()),
        "prompt": prompt,
        "output": str(output),
        "checkout": str(checkout.resolve()),
        "cache": str(cache.resolve()),
        "weight_sha256": weight_sha256,
        "dino_revision": dino_revision,
        "runner_image_digest": runner_image_digest,
        "plan_id": plan_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "manifest_hash": manifest_hash,
    }
    # Preserve a venv Python symlink: resolving it selects the system environment.
    interpreter = interpreter.expanduser().absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("VBench metrics interpreter is not an executable file")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix="vbench-request-", suffix=".json", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(request, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-B",
                    "-m",
                    "forge.evaluation.vbench_worker",
                    "--request",
                    str(temporary),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            # TimeoutExpired carries bytes even when text=True. Preserve the
            # last diagnostic before the caller archives and closes the GPU.
            for suffix, content in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                with output.with_name(output.name + f".{suffix}.txt").open("x") as stream:
                    stream.write((content or "")[-65536:])
            raise
        for suffix, content in (("stdout", completed.stdout), ("stderr", completed.stderr)):
            with output.with_name(output.name + f".{suffix}.txt").open("x") as stream:
                stream.write(content[-65536:])
        if completed.returncode:
            raise RuntimeError("isolated VBench worker failed; see adjacent .stderr.txt diagnostic")
        receipt_path = output / "receipt.json"
        if not receipt_path.is_file():
            raise ValueError("isolated VBench worker did not persist a receipt")
        return _validate_isolated_receipt(
            json.loads(receipt_path.read_text()),
            video=video,
            output=output,
            prompt=prompt,
            weight_sha256=weight_sha256,
            dino_revision=dino_revision,
            runner_image_digest=runner_image_digest,
            plan_id=plan_id,
            prompt_id=prompt_id,
            seed=seed,
            manifest_hash=manifest_hash,
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
