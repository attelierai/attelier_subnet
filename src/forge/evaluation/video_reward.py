"""Pinned VideoReward development runner; imports GPU dependencies only on execution.

Uses the published heads, preprocessing and normalization, with an explicitly
identified SDPA runtime. No commercial judge, miner code or automatic downloads
are involved in measurement. This adapter still needs a real GPU smoke test.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import signal
import sys
import time
import urllib.request
from pathlib import Path

from forge.evaluation.evaluator_study import (
    AXES,
    checked_receipts,
    read,
    validate_plan,
    write_new,
)
from forge.protocol.hashing import hash_canonical, sha256_file

REMOTE_ROOT = Path("/workspace/forge")


def remote_path(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(REMOTE_ROOT)
    return resolved


def verify_assets(spec: dict, assets: Path) -> None:
    if spec.get("schema") != "forge/videoreward-study-spec/v1":
        raise ValueError("unsupported VideoReward specification")
    for group in ("code", "base", "reward"):
        for relative, digest in spec[group]["files"].items():
            path = assets / group / relative
            path.resolve().relative_to((assets / group).resolve())
            if sha256_file(path) != digest:
                raise ValueError(f"changed evaluator asset: {group}/{relative}")
    expected_py = {name for name in spec["code"]["files"] if name.endswith(".py")}
    if {p.name for p in (assets / "code").glob("*.py")} != expected_py:
        raise ValueError("unexpected Python modules in evaluator checkout")


def download_assets(spec: dict, assets: Path) -> None:
    """Explicit remote preparation only; never invoked by measure or report."""
    assets = remote_path(assets)
    from huggingface_hub import snapshot_download
    code = assets / "code"
    code.mkdir(parents=True, exist_ok=True)
    for name, digest in spec["code"]["files"].items():
        target = code / name
        if target.exists():
            if sha256_file(target) != digest:
                raise ValueError("existing upstream file differs; use a new assets directory")
            continue
        url = (
            "https://raw.githubusercontent.com/KlingAIResearch/VideoAlign/"
            + spec["code"]["revision"] + "/" + name
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        import hashlib

        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("upstream code download hash mismatch")
        with target.open("xb") as stream:
            stream.write(payload)
    for group in ("base", "reward"):
        snapshot_download(
            repo_id=spec[group]["repo"], revision=spec[group]["revision"],
            allow_patterns=list(spec[group]["files"]), local_dir=assets / group, token=False,
        )
    verify_assets(spec, assets)


def load_inference(spec: dict, assets: Path):
    """Build the published model with local frozen assets and a strict state load."""
    import numpy as np
    import torch

    from forge.evaluation.vision_media import load_video

    if not torch.cuda.is_available():
        raise ValueError("a CUDA GPU is required")
    verify_assets(spec, assets)
    versions = {name: importlib.metadata.version(name) for name in spec["runtime_versions"]}
    if any(versions[name].split("+")[0] != version
           for name, version in spec["runtime_versions"].items()):
        raise ValueError("runtime versions differ from the frozen evaluator specification")
    settings = spec["inference"]
    if settings != {
        "fps": 2.0, "max_pixels": 200704, "use_norm": True, "dtype": "bfloat16",
        "attention": "sdpa", "batch_size": 1, "seed": 0,
    }:
        raise ValueError("this adapter only supports its frozen SDPA inference recipe")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(assets / "code"))
    from data import DataConfig
    from inference import VideoVLMRewardInference, load_configs_from_json
    from train_reward import create_model_and_processor
    from utils import ModelConfig, PEFTLoraConfig, TrainingConfig

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data, _, model, lora, normalization = load_configs_from_json(
        assets / "reward/model_config.json"
    )
    # Upstream's processor otherwise follows unpinned main independently of the model.
    model.update(model_name_or_path=str(assets / "base"), model_revision=spec["base"]["revision"])
    trained, processor, _ = create_model_and_processor(
        ModelConfig(**model), PEFTLoraConfig(**lora),
        TrainingConfig(output_dir="", gradient_checkpointing=False,
                       disable_flash_attn2=True, bf16=True, report_to=[]),
    )
    state = torch.load(assets / "reward/checkpoint-11352/model.pth",
                       map_location="cpu", weights_only=True)
    trained.load_state_dict(state, strict=True)
    del state
    # Reuse upstream preprocessing/reward exactly; bypass its mutable checkpoint lookup.
    inference = VideoVLMRewardInference.__new__(VideoVLMRewardInference)
    inference.model = trained.eval().to("cuda:0")
    inference.processor = processor
    inference.device = "cuda:0"
    inference.data_config = DataConfig(**data)
    inference.inference_config = normalization
    runtime = {
        "versions": versions, "cuda": torch.version.cuda,
        "installed_distributions": dict(sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        )),
        "python": sys.version,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "adapter_sha256": sha256_file(Path(__file__)),
        "study_sha256": sha256_file(Path(__file__).with_name("evaluator_study.py")),
        "media_decoder_sha256": sha256_file(Path(__file__).with_name("vision_media.py")),
    }
    evaluator_id = hash_canonical("forge/videoreward-study-runtime/v1", spec, runtime)

    def score(path: Path, prompt: str, geometry: dict) -> dict:
        decoded = load_video(path, geometry)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            values = inference.reward([str(path)], [prompt], fps=settings["fps"],
                                      max_pixels=settings["max_pixels"], use_norm=True)[0]
        torch.cuda.synchronize()
        return {
            "scores": {axis: float(values[axis]) for axis in AXES},
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "diagnostics": decoded.diagnostics,
            "video_sha256": decoded.video_sha256,
        }

    return score, evaluator_id, runtime


def measure(plan: dict, assets: Path, media_root: Path, output: Path, deadline: float,
            max_new_requests: int | None = None) -> dict:
    validate_plan(plan)
    if not deadline > time.time() or not deadline < float("inf"):
        raise ValueError("a finite future deadline is required")
    if max_new_requests is not None and (type(max_new_requests) is not int or max_new_requests < 1):
        raise ValueError("max_new_requests must be a positive integer")
    assets, media_root, output = map(remote_path, (assets, media_root, output))
    output.mkdir(parents=True, exist_ok=True)
    existing = checked_receipts(plan, [read(p) for p in sorted(output.glob("*.json"))])
    samples = {s["sample_id"]: s for s in plan["samples"]}
    paths = {}
    for key, sample in samples.items():
        path = (media_root / sample["path"]).resolve()
        path.relative_to(media_root)
        if sha256_file(path) != sample["video_sha256"]:
            raise ValueError("media does not match frozen study bytes")
        paths[key] = path
    if len(existing) == len(plan["requests"]):
        return {"status": "COMPLETE", "new_measurements": 0}
    score, evaluator_id, runtime = load_inference(plan["spec"], assets)
    if any(row["evaluator_id"] != evaluator_id for row in existing.values()):
        raise ValueError("cannot resume with a changed evaluator runtime")
    # Unmeasured warmup, never reused as a repeatability observation.
    first = plan["samples"][0]
    score(paths[first["sample_id"]], first["prompt_text"], plan["geometry"])
    count = 0
    for request in plan["requests"]:
        if request["request_id"] in existing:
            continue
        if time.time() >= deadline:
            break
        if max_new_requests is not None and count >= max_new_requests:
            break
        sample = samples[request["sample_id"]]
        path = paths[request["sample_id"]]
        result = score(path, sample["prompt_text"], plan["geometry"])
        if (result["video_sha256"] != sample["video_sha256"]
                or sha256_file(path) != result["video_sha256"]):
            raise ValueError("media changed during inference")
        row = {
            **result, "request_id": request["request_id"], "plan_id": plan["plan_id"],
            "status": "OK", "qualification": "prohibited",
            "evaluator_id": evaluator_id, "runtime": runtime,
        }
        checked_receipts(plan, [row])
        write_new(output / (request["request_id"].replace(":", "-") + ".json"), row)
        count += 1
    return {"status": "COMPLETE" if len(existing) + count == len(plan["requests"])
            else "INCOMPLETE", "new_measurements": count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-assets")
    download.add_argument("--spec", type=Path, required=True)
    download.add_argument("--assets", type=Path, required=True)
    run = commands.add_parser("measure")
    for name in ("plan", "assets", "media-root", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--deadline-unix", type=float, required=True)
    run.add_argument("--max-new-requests", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "download-assets":
            download_assets(read(args.spec), args.assets)
        else:
            if not 0 < args.deadline_unix - time.time() < float("inf"):
                raise ValueError("a finite future deadline is required")
            # Process deadline is additional to the provider lease/termination guard.
            signal.setitimer(signal.ITIMER_REAL, args.deadline_unix - time.time())
            result = measure(read(args.plan), args.assets, args.media_root, args.out,
                             args.deadline_unix, args.max_new_requests)
            signal.setitimer(signal.ITIMER_REAL, 0)
            print(result)
            return 0 if result["status"] == "COMPLETE" else 3
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"VideoReward study rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
