"""Single-GPU executor for externally assigned diagnostic Wan jobs.

This module deliberately has no chain, committee, scoring, or signing logic.
It is the small, reproducible execution half of distributed evaluation: an
external, frozen assignment says exactly what to render and this worker records
what it actually rendered.  A diagnostic run can never be used as a reward
claim merely by changing a receipt field.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.models.artifact_manifest import ModelManifest
from forge.models.generation import GenerationRecipe, paired_reference_scheduler, timing_scope
from forge.models.lora_loading import load_candidate_lora
from forge.models.weights import validate_candidate_weights
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.runtime_preflight import verify_local

JOB_SCHEMA = "forge/distributed-worker-jobs/v1"
RECEIPT_SCHEMA = "forge/distributed-worker-receipt/v1"
RUN_SCHEMA = "forge/distributed-worker-run/v1"
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class DiagnosticJob:
    job_id: str
    kind: str
    prompt_id: str
    prompt: str
    seed: int
    model: ModelManifest
    model_dir: Path
    scheduler: str

    def identity(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "seed": self.seed,
            "model_manifest_hash": self.model.fingerprint(),
            "recipe": dict(self.model.generation_config),
            "scheduler": self.scheduler,
        }

    def job_hash(self) -> str:
        return hash_canonical("forge/distributed-worker-job/v1", self.identity())


@dataclass(frozen=True)
class DiagnosticJobBundle:
    validator_hotkey: str
    round_id: str
    contract: ChallengeContract
    base: ModelManifest
    base_dir: Path
    jobs: tuple[DiagnosticJob, ...]
    max_generation_seconds: float

    def bundle_hash(self) -> str:
        return hash_canonical(
            "forge/distributed-worker-bundle/v1",
            {
                "validator_hotkey": self.validator_hotkey,
                "round_id": self.round_id,
                "contract": self.contract.to_dict(),
                "base_manifest_hash": self.base.fingerprint(),
                "jobs": [job.job_hash() for job in self.jobs],
                "max_generation_seconds": self.max_generation_seconds,
            },
        )


def load_diagnostic_jobs(path: Path) -> DiagnosticJobBundle:
    """Read the exact, offline-safe job format before a CUDA import occurs."""
    raw = json.loads(path.read_text())
    required = {
        "schema",
        "diagnostic_only",
        "validator_hotkey",
        "round_id",
        "contract",
        "base_manifest",
        "base_dir",
        "max_generation_seconds",
        "jobs",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("distributed worker job bundle fields are not exact")
    if raw["schema"] != JOB_SCHEMA or raw["diagnostic_only"] is not True:
        raise ValueError("only explicit diagnostic_only distributed jobs are accepted")
    if not isinstance(raw["validator_hotkey"], str) or not raw["validator_hotkey"]:
        raise ValueError("validator hotkey is required")
    if not isinstance(raw["round_id"], str) or not raw["round_id"]:
        raise ValueError("round id is required")
    if (
        isinstance(raw["max_generation_seconds"], bool)
        or not isinstance(raw["max_generation_seconds"], (int, float))
        or not math.isfinite(raw["max_generation_seconds"])
        or raw["max_generation_seconds"] <= 0
    ):
        raise ValueError("max_generation_seconds must be positive")
    if not isinstance(raw["contract"], Mapping) or not isinstance(raw["base_manifest"], Mapping):
        raise ValueError("contract and base_manifest must be objects")
    if not isinstance(raw["base_dir"], str) or not raw["base_dir"]:
        raise ValueError("base_dir is required")
    if not isinstance(raw["jobs"], list) or not raw["jobs"]:
        raise ValueError("at least one explicit job is required")
    contract = ChallengeContract.from_dict(dict(raw["contract"]))
    base = ModelManifest.from_dict(raw["base_manifest"])
    if base.role != "pipeline" or (base.reference.repo_id, base.reference.revision) != (
        contract.base_model_id,
        contract.base_model_revision,
    ):
        raise ValueError("base manifest differs from the pinned contract")
    seen: set[str] = set()
    jobs: list[DiagnosticJob] = []
    for row in raw["jobs"]:
        if not isinstance(row, Mapping) or set(row) != {
            "job_id",
            "kind",
            "prompt_id",
            "prompt",
            "seed",
            "model_manifest",
            "model_dir",
            "scheduler",
        }:
            raise ValueError("distributed worker job fields are not exact")
        job_id = row["job_id"]
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id) or job_id in seen:
            raise ValueError("job_id must be unique and filesystem-safe")
        seen.add(job_id)
        if not all(
            isinstance(row[key], str) and row[key]
            for key in ("kind", "prompt_id", "prompt", "model_dir")
        ):
            raise ValueError("job text fields are required")
        if type(row["seed"]) is not int or not 0 <= row["seed"] < 2**63:
            raise ValueError("job seed must be a nonnegative 63-bit integer")
        if row["scheduler"] not in {"uniform_euler", "original_unipc"}:
            raise ValueError("job scheduler is invalid")
        if not isinstance(row["model_manifest"], Mapping):
            raise ValueError("job model_manifest must be an object")
        model = ModelManifest.from_dict(row["model_manifest"])
        if Path(row["model_dir"]).resolve() == Path(raw["base_dir"]).resolve():
            pinned = {item.path: item for item in base.files}
            if any(pinned.get(item.path) != item for item in model.files):
                raise ValueError("a shared base directory requires identical pinned file records")
        recipe = GenerationRecipe.from_dict(dict(model.generation_config))
        if recipe.steps > contract.max_steps:
            raise ValueError("job recipe exceeds the pinned max_steps")
        expected_scheduler = paired_reference_scheduler(model, base, recipe, contract)
        if (
            row["kind"] == "reference_control"
            and recipe.scheduler is None
            and model.role == "full_transformer"
            and model.reference == base.reference
            and recipe.steps == contract.reference_steps
            and recipe.guidance_scale == 5.0
        ):
            # Diagnostic equality control: exact public base transformer, original recipe.
            # Competitive duplicate admission still rejects this package.
            expected_scheduler = "original_unipc"
        if row["scheduler"] != expected_scheduler:
            raise ValueError("job scheduler differs from the pinned model/recipe lane")
        if model.role == "pipeline" and model.reference != base.reference:
            raise ValueError("only the pinned base pipeline may be rendered")
        if model.role != "pipeline" and model.base_reference != base.reference:
            raise ValueError("candidate must declare the pinned base")
        jobs.append(
            DiagnosticJob(
                job_id,
                row["kind"],
                row["prompt_id"],
                row["prompt"],
                row["seed"],
                model,
                Path(row["model_dir"]),
                row["scheduler"],
            )
        )
    return DiagnosticJobBundle(
        raw["validator_hotkey"],
        raw["round_id"],
        contract,
        base,
        Path(raw["base_dir"]),
        tuple(jobs),
        float(raw["max_generation_seconds"]),
    )


def inspect_diagnostic_jobs(path: Path) -> dict[str, object]:
    """CPU-only validation useful before allocating a GPU."""
    bundle = load_diagnostic_jobs(path)
    return {
        "schema": JOB_SCHEMA,
        "diagnostic_only": True,
        "bundle_hash": bundle.bundle_hash(),
        "validator_hotkey": bundle.validator_hotkey,
        "round_id": bundle.round_id,
        "jobs": [{"job_id": job.job_id, "job_hash": job.job_hash()} for job in bundle.jobs],
    }


def run_diagnostic_jobs(
    bundle: DiagnosticJobBundle, out: Path, *, warmup: bool = True
) -> dict[str, object]:
    """Execute a frozen bundle sequentially on one CUDA GPU.

    Completed exact receipts are resumed without regeneration. Failed or
    uncertain receipts are deliberately terminal: an operator must issue a new
    signed/committed diagnostic bundle rather than silently retrying it.
    """
    import gc

    import numpy as np
    import torch
    from diffusers import (
        AutoencoderKLWan,
        FlowMatchEulerDiscreteScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import export_to_video

    if not torch.cuda.is_available():
        raise RuntimeError("distributed Wan worker requires CUDA")
    verify_local(bundle.base, bundle.base_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_path = out / "run.json"
    started = time.monotonic()
    completed: list[str] = []
    skipped: list[str] = []
    incomplete: list[str] = []
    active_key: tuple[str, str, str] | None = None
    pipe: Any = None

    def close_pipeline() -> None:
        nonlocal pipe, active_key
        if pipe is not None:
            del pipe
            pipe = None
            active_key = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_pipeline(job: DiagnosticJob) -> Any:
        nonlocal pipe, active_key
        key = (job.model.fingerprint(), str(job.model_dir.resolve()), job.scheduler)
        if pipe is not None and active_key == key:
            return pipe
        close_pipeline()
        model_dir = job.model_dir
        if model_dir.resolve() != bundle.base_dir.resolve():
            verify_local(job.model, model_dir)
        validate_candidate_weights(job.model, model_dir, bundle.base, bundle.base_dir)
        vae = AutoencoderKLWan.from_pretrained(
            bundle.base_dir,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
            use_safetensors=True,
        )
        loaded = WanPipeline.from_pretrained(
            bundle.base_dir,
            vae=vae,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
        )
        if job.model.role == "full_transformer":
            transformer_dir = model_dir / "transformer"
            loaded.transformer = WanTransformer3DModel.from_pretrained(
                transformer_dir if transformer_dir.is_dir() else model_dir,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                use_safetensors=True,
            )
        elif job.model.role == "lora":
            weight = [item.path for item in job.model.files if item.path.endswith(".safetensors")]
            load_candidate_lora(loaded, model_dir, weight[0])
        if job.scheduler == "uniform_euler":

            class UniformFlowEuler(FlowMatchEulerDiscreteScheduler):
                def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
                    kwargs.setdefault(
                        "sigmas",
                        [1.0 - index / num_inference_steps for index in range(num_inference_steps)],
                    )
                    return super().set_timesteps(num_inference_steps, device=device, **kwargs)

            loaded.scheduler = UniformFlowEuler(
                num_train_timesteps=1000,
                shift=bundle.contract.scheduler_shift,
                use_dynamic_shifting=False,
            )
        loaded.to("cuda")
        loaded.vae.enable_tiling()
        loaded.set_progress_bar_config(disable=True)
        recipe = GenerationRecipe.from_dict(dict(job.model.generation_config))
        if warmup:
            with torch.inference_mode():
                loaded(
                    prompt=job.prompt,
                    generator=torch.Generator("cuda").manual_seed(0),
                    height=bundle.contract.height,
                    width=bundle.contract.width,
                    num_frames=bundle.contract.frames,
                    num_inference_steps=recipe.steps,
                    guidance_scale=recipe.guidance_scale,
                    negative_prompt="",
                    output_type="np",
                )
        torch.cuda.synchronize()
        pipe, active_key = loaded, key
        return loaded

    for job in bundle.jobs:
        receipt_path = out / f"{job.job_id}.json"
        video_path = out / f"{job.job_id}.mp4"
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text())
            if existing.get("job_hash") != job.job_hash():
                raise ValueError(f"existing receipt binds a different job: {job.job_id}")
            if (
                existing.get("status") != "OK"
                or not video_path.is_file()
                or existing.get("video_sha256") != sha256_file(video_path)
            ):
                raise RuntimeError(f"terminal receipt cannot be blindly retried: {job.job_id}")
            skipped.append(job.job_id)
            continue
        if time.monotonic() - started >= bundle.max_generation_seconds:
            incomplete.append(job.job_id)
            continue
        runtime_started = time.perf_counter()
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "diagnostic_only": True,
            "bundle_hash": bundle.bundle_hash(),
            "job_hash": job.job_hash(),
            "job": job.identity(),
            "validator_hotkey": bundle.validator_hotkey,
            "round_id": bundle.round_id,
            "geometry": {
                "height": bundle.contract.height,
                "width": bundle.contract.width,
                "frames": bundle.contract.frames,
                "fps": bundle.contract.fps,
            },
            "status": "RUNNING",
            "timing_scope": timing_scope(),
        }
        # Persist before any paid execution: a killed process must not look unstarted.
        with receipt_path.open("x") as handle:
            json.dump(receipt, handle, sort_keys=True, allow_nan=False)
        try:
            # The loop's previous local reference would otherwise keep the old
            # pipeline on CUDA while the next model is loaded and warmed up.
            loaded = None
            loaded = load_pipeline(job)
            recipe = GenerationRecipe.from_dict(dict(job.model.generation_config))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                frames = loaded(
                    prompt=job.prompt,
                    generator=torch.Generator("cuda").manual_seed(job.seed),
                    height=bundle.contract.height,
                    width=bundle.contract.width,
                    num_frames=bundle.contract.frames,
                    num_inference_steps=recipe.steps,
                    guidance_scale=recipe.guidance_scale,
                    negative_prompt="",
                    output_type="np",
                ).frames[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            expected = (bundle.contract.frames, bundle.contract.height, bundle.contract.width, 3)
            if frames.shape != expected or not np.isfinite(frames).all():
                raise ValueError("model produced malformed or nonfinite frames")
            export_to_video(frames, str(video_path), fps=bundle.contract.fps)
            receipt.update(
                {
                    "status": "OK" if elapsed <= bundle.max_generation_seconds else "MODEL_FAILURE",
                    "video": video_path.name,
                    "video_sha256": sha256_file(video_path),
                    "full_pipeline_seconds": elapsed,
                    "gpu_synchronized": True,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                    "device_total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "model_weight_content_hash": validate_candidate_weights(
                        job.model, job.model_dir, bundle.base, bundle.base_dir
                    ).content_fingerprint,
                    "runtime": {
                        "source_sha256": sha256_file(__file__),
                        "lora_loader_sha256": sha256_file(load_candidate_lora.__code__.co_filename),
                        "torch": importlib.metadata.version("torch"),
                        "diffusers": importlib.metadata.version("diffusers"),
                        "cuda": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "dtype": "bfloat16",
                        "vae_dtype": "float32",
                        "scheduler": job.scheduler,
                    },
                }
            )
            if receipt["status"] == "OK":
                completed.append(job.job_id)
            else:
                incomplete.append(job.job_id)
        except (ValueError, torch.cuda.OutOfMemoryError) as exc:
            receipt.update(status="MODEL_FAILURE", detail=type(exc).__name__)
            incomplete.append(job.job_id)
            close_pipeline()
        except (RuntimeError, OSError) as exc:
            receipt.update(status="INFRA_FAILURE", detail=type(exc).__name__)
            incomplete.append(job.job_id)
            close_pipeline()
        receipt["worker_elapsed_seconds"] = time.perf_counter() - runtime_started
        temporary = receipt_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(receipt_path)
    close_pipeline()
    result = {
        "schema": RUN_SCHEMA,
        "diagnostic_only": True,
        "bundle_hash": bundle.bundle_hash(),
        "validator_hotkey": bundle.validator_hotkey,
        "round_id": bundle.round_id,
        "completed_job_ids": completed,
        "skipped_completed_job_ids": skipped,
        "incomplete_job_ids": incomplete,
        "status": "COMPLETE" if not incomplete else "INCOMPLETE",
    }
    temporary = run_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(run_path)
    return result
