"""Execute clean paired Wan models with synchronized ABBA timing."""

from __future__ import annotations

import importlib.metadata
import json
import statistics
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
from forge.validator.runtime_preflight import effective_candidate_hash, verify_local


def run_paired_panel(
    *,
    base: ModelManifest,
    base_dir: Path,
    reference_model: ModelManifest,
    reference_dir: Path,
    candidate: ModelManifest,
    candidate_dir: Path,
    prompts: list[dict],
    contract: ChallengeContract,
    out: Path,
    experimental_adapter: Path | None = None,
    reference_scheduler_kind: str | None = None,
    timing_design: str = "confirmation_abba",
    frozen_plan_id: str | None = None,
    baseline_self_control: bool = False,
) -> dict[str, Any]:
    """Render screening AB or confirmation ABBA on one validator-owned Wan pipeline.

    Each invocation starts from a clean base transformer. A full transformer
    is copied from verified CPU tensors, while a LoRA is loaded onto that clean
    base immediately before the clock starts.  This supports LoRA↔LoRA,
    transformer↔transformer and mixed LoRA↔transformer timing without letting
    adapters accumulate across timing positions.
    """
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline
    from diffusers.utils import export_to_video

    validate_prompts(prompts)
    designs = {
        "screening_ab": ("forge/screen-timing/v1", ("reference", "candidate")),
        "confirmation_abba": (
            "forge/paired-timing/v1",
            ("reference", "candidate", "candidate", "reference"),
        ),
    }
    if timing_design not in designs:
        raise ValueError("timing design must be screening_ab or confirmation_abba")
    if timing_design == "screening_ab" and (
        not isinstance(frozen_plan_id, str) or not frozen_plan_id
    ):
        raise ValueError("screening timing requires a frozen plan id")
    receipt_schema, order = designs[timing_design]
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty; evidence cannot be overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan validation requires a CUDA worker")
    if base.role != "pipeline" or (base.reference.repo_id, base.reference.revision) != (
        contract.base_model_id,
        contract.base_model_revision,
    ):
        raise ValueError("base snapshot differs from the pinned challenge contract")

    def _check_model(model: ModelManifest, model_dir: Path, label: str) -> None:
        if model.role == "pipeline":
            if model.reference != base.reference:
                raise ValueError(f"{label} pipeline must be the pinned base")
        elif model.role in {"lora", "full_transformer"}:
            if model.base_reference != base.reference:
                raise ValueError(f"{label} weights must declare the pinned base")
        else:
            raise ValueError(
                f"paired timing does not support {label} role {model.role!r}; "
                "full-transformer duels require isolated-worker evidence"
            )
        inspect_public_artifact(model.reference)
        if model_dir.resolve() != base_dir.resolve():
            verify_local(model, model_dir)

    inspect_public_artifact(base.reference)
    verify_local(base, base_dir)
    _check_model(reference_model, reference_dir, "reference")
    _check_model(candidate, candidate_dir, "candidate")
    if experimental_adapter and candidate.role == "full_transformer":
        raise ValueError(
            "an experimental adapter must be evaluated on the pinned base, not a full transformer"
        )
    checked_weights = {
        "reference": validate_candidate_weights(reference_model, reference_dir, base, base_dir),
        "candidate": validate_candidate_weights(candidate, candidate_dir, base, base_dir),
    }
    if experimental_adapter:
        validate_lora_weights(
            experimental_adapter,
            [
                base_dir / f.path
                for f in base.files
                if f.path.startswith("transformer/") and f.path.endswith(".safetensors")
            ],
        )
    if baseline_self_control and (
        experimental_adapter
        or reference_model.fingerprint() != base.fingerprint()
        or candidate.fingerprint() != base.fingerprint()
        or dict(base.generation_config).get("steps") != contract.reference_steps
        or dict(base.generation_config).get("guidance_scale") != 5.0
    ):
        raise ValueError("baseline self-control requires the exact original Wan50 on both sides")
    if (
        reference_model.fingerprint() == candidate.fingerprint()
        and not experimental_adapter
        and not baseline_self_control
    ):
        raise ValueError("paired timing requires two distinct model manifests")
    reference_recipe = GenerationRecipe.from_dict(dict(reference_model.generation_config))
    candidate_recipe = GenerationRecipe.from_dict(dict(candidate.generation_config))
    committed_reference_scheduler = paired_reference_scheduler(
        reference_model, base, reference_recipe, contract
    )
    candidate_scheduler_kind = paired_reference_scheduler(
        candidate, base, candidate_recipe, contract
    )
    if (
        reference_scheduler_kind is not None
        and reference_scheduler_kind != committed_reference_scheduler
    ):
        raise ValueError("reference scheduler differs from the committed recipe")
    reference_scheduler_kind = committed_reference_scheduler
    original_reference = (
        reference_model.role == "pipeline"
        and reference_model.reference == base.reference
        and reference_recipe.steps == contract.reference_steps
        and reference_recipe.guidance_scale == 5.0
        and reference_scheduler_kind == "original_unipc"
    )
    if candidate_recipe.steps > contract.max_steps or (
        not original_reference and reference_recipe.steps > contract.max_steps
    ):
        raise ValueError("paired timing recipe exceeds its contract step limit")
    if not experimental_adapter and not baseline_self_control and effective_candidate_hash(
        reference_model, checked_weights["reference"], base, contract
    ) == effective_candidate_hash(candidate, checked_weights["candidate"], base, contract):
        raise ValueError("paired timing candidates duplicate exact effective tensors and recipe")

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

    from safetensors.torch import load_file

    base_state = {
        key: tensor.detach().cpu().clone() for key, tensor in pipe.transformer.state_dict().items()
    }

    def _transformer_state(model: ModelManifest, directory: Path) -> dict[str, Any]:
        if model.role != "full_transformer":
            return base_state
        state = {}
        for item in model.files:
            if item.path.startswith("transformer/") and item.path.endswith(".safetensors"):
                state.update(load_file(str(directory / item.path), device="cpu"))
        return state

    cpu_states = {
        "reference": _transformer_state(reference_model, reference_dir),
        "candidate": _transformer_state(candidate, candidate_dir),
    }

    def _lora_file(model: ModelManifest, directory: Path) -> tuple[Path, str] | None:
        if model.role != "lora":
            return None
        files = [file.path for file in model.files if file.path.endswith(".safetensors")]
        if len(files) != 1:
            raise ValueError("a LoRA candidate must contain exactly one safetensors file")
        return directory, files[0]

    lora_files = {
        "reference": _lora_file(reference_model, reference_dir),
        "candidate": (
            (experimental_adapter.parent, experimental_adapter.name)
            if experimental_adapter
            else _lora_file(candidate, candidate_dir)
        ),
    }
    if experimental_adapter and experimental_adapter.suffix != ".safetensors":
        raise ValueError("experimental adapters must use safetensors")

    class UniformFlowEuler(FlowMatchEulerDiscreteScheduler):
        def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):
            if "sigmas" not in kwargs:
                kwargs["sigmas"] = [
                    1.0 - index / num_inference_steps for index in range(num_inference_steps)
                ]
            return super().set_timesteps(num_inference_steps, device=device, **kwargs)

    uniform_scheduler = UniformFlowEuler(
        num_train_timesteps=1000,
        shift=contract.scheduler_shift,
        use_dynamic_shifting=False,
    )
    original_unipc_scheduler = pipe.scheduler
    schedulers = {
        "reference": (
            original_unipc_scheduler
            if reference_scheduler_kind == "original_unipc"
            else uniform_scheduler
        ),
        "candidate": (
            type(original_unipc_scheduler).from_config(original_unipc_scheduler.config)
            if candidate_scheduler_kind == "original_unipc"
            else uniform_scheduler
        ),
    }
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
        "scheduler": "paired-exact-recipes",
        "schedulers": {
            "reference": reference_scheduler_kind,
            "candidate": candidate_scheduler_kind,
        },
        "scheduler_grid": {
            "reference": (
                "model-default"
                if reference_scheduler_kind == "original_unipc"
                else "uniform-flow-endpoints/v1"
            ),
            "candidate": (
                "model-default"
                if candidate_scheduler_kind == "original_unipc"
                else "uniform-flow-endpoints/v1"
            ),
        },
        "shift": {
            "reference": (
                None if reference_scheduler_kind == "original_unipc" else contract.scheduler_shift
            ),
            "candidate": None
            if candidate_scheduler_kind == "original_unipc"
            else contract.scheduler_shift,
        },
        "negative_prompt": "",
        "attention": "torch-sdpa",
        "compile": False,
        "model_switch": "clean-base-plus-verified-transformer-or-single-lora",
        "timing_scope": timing_scope(),
    }
    runtime_id = hash_canonical("forge/runtime/v1", runtime)
    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    geometry = {
        "height": contract.height,
        "width": contract.width,
        "frames": contract.frames,
        "fps": contract.fps,
    }
    out.mkdir(parents=True, exist_ok=True)
    for side in ("reference", "candidate"):
        (out / side).mkdir(exist_ok=True)

    def progress(stage: str, **details: Any) -> None:
        checkpoint = out / "progress.json"
        temporary = checkpoint.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "schema": "forge/paired-progress/v1", "stage": stage,
            "updated_unix": time.time(), "runtime_id": runtime_id,
            "frozen_plan_id": frozen_plan_id, **details,
        }, allow_nan=False) + "\n")
        temporary.replace(checkpoint)

    def _report(model: ModelManifest, recipe: GenerationRecipe, admission: str) -> dict[str, Any]:
        return {
            "schema": "forge/panel/v1",
            "contract_hash": contract.contract_hash(),
            "contract": contract.to_dict(),
            "runtime_id": runtime_id,
            "runtime": runtime,
            "hardware": hardware,
            "manifest_hash": model.fingerprint(),
            "manifest": model.to_dict(),
            "prompts": prompts,
            "observations": [],
            "geometry": geometry,
            "admission": admission,
            "quality": None,
            "recipe": recipe.to_dict(),
            "scheduler_recipe": {
                "kind": reference_scheduler_kind
                if model is reference_model
                else candidate_scheduler_kind,
                "shift": (
                    None
                    if (
                        reference_scheduler_kind
                        if model is reference_model
                        else candidate_scheduler_kind
                    )
                    == "original_unipc"
                    else contract.scheduler_shift
                ),
            },
            "warmup_videos": 1,
        }

    reference_report = _report(
        reference_model, reference_recipe,
        "baseline_control" if baseline_self_control else "public_snapshot",
    )
    candidate_report = _report(
        candidate,
        candidate_recipe,
        "baseline_control" if baseline_self_control
        else "private_experiment" if experimental_adapter else "public_snapshot",
    )
    reference_report["weight_content_hash"] = checked_weights["reference"].content_fingerprint
    candidate_report["weight_content_hash"] = checked_weights["candidate"].content_fingerprint
    if experimental_adapter:
        candidate_report["experimental_adapter_sha256"] = sha256_file(experimental_adapter)
        candidate_report["manifest_hash"] = hash_canonical(
            "forge/private-experiment/v1",
            candidate.fingerprint(),
            candidate_report["experimental_adapter_sha256"],
        )

    adapter_loaded = False

    def _activate(side: str) -> None:
        """Materialize exactly one contender before an unmeasured invocation."""
        nonlocal adapter_loaded
        if adapter_loaded:
            pipe.unload_lora_weights()
            adapter_loaded = False
        pipe.transformer.load_state_dict(cpu_states[side], strict=True)
        lora = lora_files[side]
        if lora is not None:
            directory, filename = lora
            load_candidate_lora(pipe, directory, filename)
            adapter_loaded = True
        pipe.scheduler = schedulers[side]
        torch.cuda.synchronize()

    def _call(
        recipe: GenerationRecipe, prompt: str, seed: int
    ) -> tuple[dict[str, Any], Any | None]:
        events: list[tuple[Any, Any]] = []
        active: list[Any] = []

        def before_forward(_module, _args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            active.append(event)

        def after_forward(_module, _args, _output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append((active.pop(), event))

        pre = pipe.transformer.register_forward_pre_hook(before_forward)
        post = pipe.transformer.register_forward_hook(after_forward)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                frames = pipe(
                    prompt=prompt,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    height=contract.height,
                    width=contract.width,
                    num_frames=contract.frames,
                    num_inference_steps=recipe.steps,
                    guidance_scale=recipe.guidance_scale,
                    negative_prompt="",
                    output_type="np",
                ).frames[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            expected_shape = (contract.frames, contract.height, contract.width, 3)
            if frames.shape != expected_shape or not np.isfinite(frames).all():
                raise ValueError("model produced malformed or nonfinite frames")
            within_limits = (
                elapsed <= contract.wall_time_limit_seconds
                and torch.cuda.max_memory_allocated() <= contract.vram_limit_bytes
            )
            cfg_enabled = recipe.guidance_scale > 1
            expected_forwards = recipe.steps * (2 if cfg_enabled else 1)
            return {
                "status": (
                    "OK" if within_limits and len(events) == expected_forwards else "MODEL_FAILURE"
                ),
                "detail": None if len(events) == expected_forwards else "forward_passes_mismatch",
                "wall_time_seconds": elapsed,
                "full_pipeline_seconds": elapsed,
                "gpu_synchronized": True,
                "denoise_seconds": sum(begin.elapsed_time(end) for begin, end in events) / 1000,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "forward_passes": len(events),
                "expected_forward_passes": expected_forwards,
                "cfg_enabled": cfg_enabled,
                "steps": recipe.steps,
            }, frames
        except (ValueError, torch.cuda.OutOfMemoryError) as error:
            return {"status": "MODEL_FAILURE", "detail": type(error).__name__}, None
        except (RuntimeError, OSError) as error:
            return {"status": "INFRA_FAILURE", "detail": type(error).__name__}, None
        finally:
            pre.remove()
            post.remove()

    # Equal, unmeasured full calls establish kernel and allocator state for both recipes.
    for side, recipe in (("reference", reference_recipe), ("candidate", candidate_recipe)):
        progress("WARMUP", side=side)
        _activate(side)
        started = time.perf_counter()
        with torch.inference_mode():
            pipe(
                prompt=prompts[0]["text"],
                generator=torch.Generator("cuda").manual_seed(0),
                height=contract.height,
                width=contract.width,
                num_frames=contract.frames,
                num_inference_steps=recipe.steps,
                guidance_scale=recipe.guidance_scale,
                negative_prompt="",
                output_type="np",
            )
        torch.cuda.synchronize()
        progress("WARMUP_COMPLETE", side=side, elapsed_seconds=time.perf_counter() - started)

    reports = {"reference": reference_report, "candidate": candidate_report}
    models = {"reference": reference_recipe, "candidate": candidate_recipe}
    observations: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        "reference": {},
        "candidate": {},
    }
    ordered: list[dict[str, Any]] = []
    pair_index = 0
    for prompt_index, prompt in enumerate(prompts):
        for seed in prompt["video_seeds"]:
            pair_id = f"{prompt['prompt_id']}:{seed}"
            for position, side in enumerate(order):
                recipe = models[side]
                progress("GENERATING", side=side, prompt_id=prompt["prompt_id"],
                         seed=seed, run_order=position, completed_calls=len(ordered))
                _activate(side)
                measurement, frames = _call(recipe, prompt["text"], seed)
                measurement.update(
                    pair_id=pair_id,
                    pair_index=pair_index,
                    side=side,
                    run_order=position,
                    repeat_index=sum(
                        1 for item in ordered if item["pair_id"] == pair_id and item["side"] == side
                    ),
                    prompt_id=prompt["prompt_id"],
                    seed=seed,
                )
                slot = (prompt["prompt_id"], seed)
                if slot not in observations[side]:
                    video_fields: dict[str, Any] = {}
                    if frames is not None:
                        video = out / side / f"video-{prompt_index:04d}-{seed}.mp4"
                        try:
                            export_to_video(frames, str(video), fps=contract.fps)
                            video_fields = {
                                "video": video.name,
                                "video_sha256": sha256_file(video),
                                "diagnostics": {
                                    "pixel_std": float(frames.std()),
                                    "frame_delta": float(np.abs(np.diff(frames, axis=0)).mean()),
                                },
                            }
                        except (OSError, RuntimeError) as error:
                            measurement.update(status="INFRA_FAILURE", detail=type(error).__name__)
                    observation = {"prompt_id": prompt["prompt_id"], "seed": seed, **measurement}
                    observation["timing_repeats"] = [dict(measurement)]
                    observation.update(video_fields)
                    observations[side][slot] = observation
                    reports[side]["observations"].append(observation)
                else:
                    observations[side][slot]["timing_repeats"].append(dict(measurement))
                ordered.append(dict(measurement))
                progress("CALL_COMPLETE", side=side, prompt_id=prompt["prompt_id"],
                         seed=seed, completed_calls=len(ordered), status=measurement["status"])
                for report_side, report in reports.items():
                    receipt = out / report_side / "report.json"
                    temporary = receipt.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    temporary.replace(receipt)
            pair_index += 1

    # Confirmation has two samples per side; screening has one. Keep the raw
    # stage-specific order and never synthesize a second screening measurement.
    for side in ("reference", "candidate"):
        for observation in reports[side]["observations"]:
            repeats = observation["timing_repeats"]
            statuses = {item["status"] for item in repeats}
            if statuses != {"OK"}:
                observation["status"] = (
                    "INFRA_FAILURE" if "INFRA_FAILURE" in statuses else "MODEL_FAILURE"
                )
                continue
            for field in ("wall_time_seconds", "denoise_seconds"):
                observation[field] = statistics.median(item[field] for item in repeats)
            observation["peak_vram_bytes"] = max(item["peak_vram_bytes"] for item in repeats)
            observation["forward_passes"] = max(item["forward_passes"] for item in repeats)
        receipt = out / side / "report.json"
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(reports[side], indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt)

    paired = {
        "schema": receipt_schema,
        "contract_hash": contract.contract_hash(),
        "panel_hash": hash_canonical("forge/panel/v1", contract.contract_hash(), prompts, geometry),
        "frozen_plan_id": frozen_plan_id,
        "runtime_id": runtime_id,
        "runtime": runtime,
        "hardware": hardware,
        "geometry": geometry,
        "design": {
            "name": timing_design,
            "order": list(order),
            "warmups_per_side": 1,
            "clean_materialization_before_every_call": True,
            "timing_scope": timing_scope(),
            "reference_scheduler": reference_scheduler_kind,
            "candidate_scheduler": candidate_scheduler_kind,
        },
        "reference_manifest_hash": reference_report["manifest_hash"],
        "candidate_manifest_hash": candidate_report["manifest_hash"],
        "reference_admission": reference_report["admission"],
        "candidate_admission": candidate_report["admission"],
        "recipes": {
            "reference": reference_report["recipe"] | reference_report["scheduler_recipe"],
            "candidate": candidate_report["recipe"] | candidate_report["scheduler_recipe"],
        },
        "report_video_sha256": {
            side: {
                f"{item['prompt_id']}:{item['seed']}": item.get("video_sha256")
                for item in report["observations"]
            }
            for side, report in reports.items()
        },
        "measurements": ordered,
    }
    receipt_name = "screen-timing.json" if timing_design == "screening_ab" else "paired-timing.json"
    (out / receipt_name).write_text(json.dumps(paired, indent=2, allow_nan=False) + "\n")
    progress("COMPLETE", completed_calls=len(ordered))
    return {"reference": reference_report, "candidate": candidate_report, "paired_timing": paired}
