"""Generate and receipt a single validator-owned Wan panel."""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from forge.models.artifact_manifest import ModelManifest
from forge.models.artifacts import inspect_public_artifact
from forge.models.generation import (
    GenerationRecipe,
    paired_reference_scheduler,
    timing_scope,
    validate_prompts,
)
from forge.models.lora_loading import load_candidate_lora
from forge.models.weights import validate_candidate_weights, validate_lora_weights
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.runtime_preflight import verify_local


def run_panel(
    *,
    base: ModelManifest,
    base_dir: Path,
    candidate: ModelManifest,
    candidate_dir: Path,
    prompts: list[dict],
    contract: ChallengeContract,
    out: Path,
    experimental_adapter: Path | None = None,
    reference: bool = False,
    scheduler_kind: str | None = None,
) -> dict[str, Any]:
    """Run a prescribed panel and persist one receipt after every generated video.

    A local experimental adapter is useful for private training validation but
    its report is explicitly ineligible for public competition admission.
    """
    import numpy as np
    import torch
    from diffusers import (
        AutoencoderKLWan,
        FlowMatchEulerDiscreteScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import export_to_video

    validate_prompts(prompts)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty; evidence cannot be overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan validation requires a CUDA worker")
    if base.role != "pipeline" or (base.reference.repo_id, base.reference.revision) != (
        contract.base_model_id,
        contract.base_model_revision,
    ):
        raise ValueError("base snapshot differs from the pinned challenge contract")
    if candidate.role == "pipeline" and candidate.reference != base.reference:
        raise ValueError("only the pinned base may supply text encoder, tokenizer and VAE")
    if candidate.role != "pipeline" and candidate.base_reference != base.reference:
        raise ValueError("candidate must declare the pinned public base")
    recipe = GenerationRecipe.from_dict(dict(candidate.generation_config))
    committed_scheduler = paired_reference_scheduler(candidate, base, recipe, contract)
    if scheduler_kind is not None and scheduler_kind != committed_scheduler:
        raise ValueError("scheduler override differs from the committed model recipe")
    scheduler_kind = committed_scheduler
    limit = contract.reference_steps if reference else contract.max_steps
    if recipe.steps > limit:
        raise ValueError("candidate recipe exceeds the step limit")
    inspect_public_artifact(base.reference)
    inspect_public_artifact(candidate.reference)
    verify_local(base, base_dir)
    if candidate_dir.resolve() != base_dir.resolve():
        verify_local(candidate, candidate_dir)
    candidate_weights = validate_candidate_weights(candidate, candidate_dir, base, base_dir)
    if experimental_adapter:
        validate_lora_weights(
            experimental_adapter,
            [
                base_dir / f.path
                for f in base.files
                if f.path.startswith("transformer/") and f.path.endswith(".safetensors")
            ],
        )
    # Explicit classes and local-only paths avoid repository-supplied Python.
    vae = AutoencoderKLWan.from_pretrained(
        base_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
        use_safetensors=True,
    )
    pipe = WanPipeline.from_pretrained(
        base_dir,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
    )
    if candidate.role == "full_transformer":
        transformer_dir = candidate_dir / "transformer"
        if not transformer_dir.is_dir():
            transformer_dir = candidate_dir
        pipe.transformer = WanTransformer3DModel.from_pretrained(
            transformer_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
        )
    if candidate.role == "lora":
        files = [f.path for f in candidate.files if f.path.endswith(".safetensors")]
        if len(files) != 1:
            raise ValueError("a LoRA candidate must contain exactly one safetensors file")
        load_candidate_lora(pipe, candidate_dir, files[0])
    if experimental_adapter:
        if experimental_adapter.suffix != ".safetensors":
            raise ValueError("experimental adapters must use safetensors")
        load_candidate_lora(pipe, experimental_adapter.parent, experimental_adapter.name)

    class UniformFlowEuler(FlowMatchEulerDiscreteScheduler):
        def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
            if "sigmas" not in kwargs:
                kwargs["sigmas"] = [
                    1.0 - i / num_inference_steps for i in range(num_inference_steps)
                ]
            return super().set_timesteps(num_inference_steps, device=device, **kwargs)

    if scheduler_kind == "uniform_euler":
        pipe.scheduler = UniformFlowEuler(
            num_train_timesteps=1000,
            shift=contract.scheduler_shift,
            use_dynamic_shifting=False,
        )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "diffusers", "transformers", "peft", "safetensors", "accelerate")
    }
    runtime = {
        "schema": "forge/wan-runtime/v1",
        "source_sha256": sha256_file(__file__),
        "lora_loader_sha256": sha256_file(load_candidate_lora.__code__.co_filename),
        "dependencies": versions,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "vae_dtype": "float32",
        "vae_tiling": True,
        "scheduler": "UniformFlowEuler" if scheduler_kind == "uniform_euler" else "original_unipc",
        "scheduler_grid": (
            "uniform-flow-endpoints/v1" if scheduler_kind == "uniform_euler" else "model-default"
        ),
        "shift": contract.scheduler_shift if scheduler_kind == "uniform_euler" else None,
        "negative_prompt": "",
        "attention": "torch-sdpa",
        "compile": False,
        "timing_scope": timing_scope(),
    }
    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "forge/panel/v1",
        "contract_hash": contract.contract_hash(),
        "contract": contract.to_dict(),
        "runtime_id": hash_canonical("forge/runtime/v1", runtime),
        "runtime": runtime,
        "hardware": hardware,
        "manifest_hash": candidate.fingerprint(),
        "manifest": candidate.to_dict(),
        "weight_content_hash": candidate_weights.content_fingerprint,
        "prompts": prompts,
        "observations": [],
        "geometry": {
            "height": contract.height,
            "width": contract.width,
            "frames": contract.frames,
            "fps": contract.fps,
        },
        "admission": "private_experiment" if experimental_adapter else "public_snapshot",
        "quality": None,
    }
    if experimental_adapter:
        report["experimental_adapter_sha256"] = sha256_file(experimental_adapter)
        report["manifest_hash"] = hash_canonical(
            "forge/private-experiment/v1",
            candidate.fingerprint(),
            report["experimental_adapter_sha256"],
        )
    call = dict(
        height=contract.height,
        width=contract.width,
        num_frames=contract.frames,
        num_inference_steps=recipe.steps,
        guidance_scale=recipe.guidance_scale,
        negative_prompt="",
        output_type="np",
    )
    # One full unmeasured warm-up per model, same shape and recipe as the panel.
    with torch.inference_mode():
        pipe(prompt=prompts[0]["text"], generator=torch.Generator("cuda").manual_seed(0), **call)
    torch.cuda.synchronize()
    report["warmup_videos"] = 1
    for index, prompt in enumerate(prompts):
        for seed in prompt["video_seeds"]:
            events: list[tuple[Any, Any]] = []
            active: list[Any] = []

            def before_forward(_module, _args, active=active):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                active.append(event)

            def after_forward(_module, _args, _output, active=active, events=events):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events.append((active.pop(), event))

            pre = pipe.transformer.register_forward_pre_hook(before_forward)
            post = pipe.transformer.register_forward_hook(after_forward)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            observation: dict[str, Any] = {"prompt_id": prompt["prompt_id"], "seed": seed}
            try:
                with torch.inference_mode():
                    frames = pipe(
                        prompt=prompt["text"],
                        generator=torch.Generator("cuda").manual_seed(seed),
                        **call,
                    ).frames[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                expected_shape = (contract.frames, contract.height, contract.width, 3)
                if frames.shape != expected_shape or not np.isfinite(frames).all():
                    raise ValueError("model produced malformed or nonfinite frames")
                video = out / f"video-{index:04d}-{seed}.mp4"
                export_to_video(frames, str(video), fps=contract.fps)
                status = "OK"
                if elapsed > contract.wall_time_limit_seconds or peak > contract.vram_limit_bytes:
                    status = "MODEL_FAILURE"
                observation.update(
                    status=status,
                    video=video.name,
                    video_sha256=sha256_file(video),
                    wall_time_seconds=elapsed,
                    full_pipeline_seconds=elapsed,
                    gpu_synchronized=True,
                    denoise_seconds=sum(a.elapsed_time(b) for a, b in events) / 1000,
                    peak_vram_bytes=peak,
                    forward_passes=len(events),
                    steps=recipe.steps,
                    diagnostics={
                        "pixel_std": float(frames.std()),
                        "frame_delta": float(np.abs(np.diff(frames, axis=0)).mean()),
                    },
                )
            except (ValueError, torch.cuda.OutOfMemoryError) as error:
                observation.update(status="MODEL_FAILURE", detail=type(error).__name__)
            except (RuntimeError, OSError) as error:
                observation.update(status="INFRA_FAILURE", detail=type(error).__name__)
            finally:
                pre.remove()
                post.remove()
            report["observations"].append(observation)
            receipt = out / "report.json"
            temporary = out / "report.json.tmp"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(receipt)
            print(json.dumps(observation), flush=True)
    return report
