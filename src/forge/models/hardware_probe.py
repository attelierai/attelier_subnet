"""Prepare/run a bounded RTX 5090 evaluator probe using Forge's existing runtimes.

This never produces competitive evidence.  ``prepare`` is CPU-only. ``render``
invokes the existing diagnostic worker and paired ABBA runtime, and ``derive``
creates the eight frozen, hash-addressed control variants after rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from forge.evaluation.mvp_policy import CONTROL_KINDS, _sha
from forge.models.artifact_manifest import ModelManifest
from forge.models.distributed_worker import load_diagnostic_jobs, run_diagnostic_jobs
from forge.models.paired_runtime import run_paired_panel
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import sha256_file


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_new(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as stream:
        stream.write(encoded)


def _recipe(base: ModelManifest, *, steps: int, guidance_scale: float) -> ModelManifest:
    return replace(base, generation_config={"steps": steps, "guidance_scale": guidance_scale})


def _job_id(split: str, label: str) -> str:
    return f"{split}-{label}".replace("_", "-")


def prepare(
    *,
    probe: dict[str, Any],
    contract: ChallengeContract,
    base: ModelManifest,
    base_dir: Path,
    validator_hotkey: str,
    max_generation_seconds: float,
    out: Path,
) -> dict[str, object]:
    """Create immutable worker jobs and judging-pair metadata without CUDA."""

    if out.exists():
        raise ValueError("probe output directory must not already exist")
    if probe.get("schema") != "forge/rtx5090-evaluator-probe/v1":
        raise ValueError("unexpected probe manifest")
    if probe.get("promotion_eligible") is not False:
        raise ValueError("only an explicitly diagnostic probe may be prepared")
    if base.role != "pipeline":
        raise ValueError("probe requires the pinned Wan pipeline manifest")
    if contract.max_steps < 50:
        raise ValueError("probe contract must permit the 50-step Wan reference")
    out.mkdir(mode=0o700)
    recipes = {
        "wan50": _recipe(base, steps=50, guidance_scale=5.0),
        "fast40": _recipe(base, steps=40, guidance_scale=5.0),
        "poor1": _recipe(base, steps=1, guidance_scale=1.0),
    }
    jobs: list[dict[str, object]] = []
    job_ref: dict[tuple[str, str], str] = {}
    for source in probe["sources"]:
        split, prompt_id, prompt, seed = (
            source["split"],
            source["prompt_id"],
            source["text"],
            source["seed"],
        )
        for label, model, run_seed in (
            ("wan50", recipes["wan50"], seed),
            ("same-model-rerun", recipes["wan50"], seed),
            ("independent-seed", recipes["wan50"], seed + 10_000),
            ("fast40", recipes["fast40"], seed),
            ("poor1", recipes["poor1"], seed),
        ):
            job_id = _job_id(split, label)
            job_ref[(split, label)] = job_id
            jobs.append(
                {
                    "job_id": job_id,
                    "kind": "rtx5090_probe",
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": run_seed,
                    "model_manifest": model.to_dict(),
                    "model_dir": str(base_dir),
                    "scheduler": "original_unipc"
                    if label != "fast40" and label != "poor1"
                    else "uniform_euler",
                }
            )
    worker = {
        "schema": "forge/distributed-worker-jobs/v1",
        "diagnostic_only": True,
        "validator_hotkey": validator_hotkey,
        "round_id": probe["campaign_id"],
        "contract": contract.to_dict(),
        "base_manifest": base.to_dict(),
        "base_dir": str(base_dir),
        "max_generation_seconds": max_generation_seconds,
        "jobs": jobs,
    }
    # The worker keeps its current pipeline resident. Grouping recipes makes
    # the declared cost reproducible: Wan50, fast40 and poor1 each warm once.
    worker["jobs"].sort(key=lambda job: (job["scheduler"], _sha(job["model_manifest"])))
    transforms = []
    source_for_control = {row["kind"]: row["split"] for row in probe["control_observations"]}
    for kind, definition in probe["transform_definitions"].items():
        split = source_for_control[kind]
        transforms.append(
            {
                "kind": kind,
                "source_job_id": job_ref[(split, "wan50")],
                "replacement_job_id": (
                    job_ref[("development", "wan50")] if kind == "wrong_video" else None
                ),
                "definition": definition,
                "observable": None,
                "structurally_changed": None,
            }
        )
    controls = []
    for row in probe["control_observations"]:
        kind, split = row["kind"], row["split"]
        candidate = {
            "self": job_ref[(split, "wan50")],
            "same_model_new_run": job_ref[(split, "same-model-rerun")],
            "independent_seed": job_ref[(split, "independent-seed")],
            "static_scene": job_ref[(split, "same-model-rerun")],
        }.get(kind, f"derived-{kind}")
        controls.append(
            {
                **row,
                "reference_job_id": job_ref[(split, "wan50")],
                "candidate_id": candidate,
                "status": "PENDING_RENDER_AND_JUDGE",
            }
        )
    timing_prompts = [
        {
            "prompt_id": row["prompt_id"],
            "text": row["text"],
            "video_seeds": [row["seed"]],
        }
        for row in probe["sources"]
        if row["prompt_id"] in probe["timing"]["prompt_ids"]
    ]
    timing = {
        "reference_manifest": recipes["wan50"].to_dict(),
        "candidate_manifest": recipes["fast40"].to_dict(),
        "prompts": timing_prompts,
        "raw_measured_renders": len(timing_prompts) * 4,
        "warmups_per_side": 1,
        "warmups_not_in_raw_measured_renders": 2,
        "diagnostic_only": True,
    }
    _write_new(out / "quality-jobs.json", worker)
    _write_new(
        out / "transforms.json",
        {"schema": "forge/rtx5090-probe-transforms/v1", "items": transforms},
    )
    _write_new(
        out / "controls.json", {"schema": "forge/rtx5090-probe-controls/v1", "items": controls}
    )
    _write_new(out / "timing.json", timing)
    result = {
        "schema": "forge/rtx5090-probe-prepared/v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "probe_manifest_hash": _sha(probe),
        "quality_jobs": len(jobs),
        "quality_warmups": 3,
        "timing_measured_renders": timing["raw_measured_renders"],
        "timing_warmups": timing["warmups_not_in_raw_measured_renders"],
        "derived_transformations": len(transforms),
    }
    _write_new(out / "prepared.json", result)
    return result


def render(prepared: Path) -> dict[str, object]:
    """Run the existing worker and paired runtime; this is the only GPU action."""

    worker_path = prepared / "quality-jobs.json"
    timing = _read(prepared / "timing.json")
    bundle = load_diagnostic_jobs(worker_path)
    quality_dir, timing_dir = prepared / "quality", prepared / "timing"
    quality = run_diagnostic_jobs(bundle, quality_dir)
    if quality["status"] != "COMPLETE":
        return {"status": "INCOMPLETE", "quality": quality, "timing": "NOT_STARTED"}
    reference = ModelManifest.from_dict(timing["reference_manifest"])
    candidate = ModelManifest.from_dict(timing["candidate_manifest"])
    paired = run_paired_panel(
        base=bundle.base,
        base_dir=bundle.base_dir,
        reference_model=reference,
        reference_dir=bundle.base_dir,
        candidate=candidate,
        candidate_dir=bundle.base_dir,
        prompts=timing["prompts"],
        contract=bundle.contract,
        out=timing_dir,
    )
    result = {
        "schema": "forge/rtx5090-probe-render/v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "quality": quality,
        "paired_timing_path": str(timing_dir / "paired-timing.json"),
        "paired_timing_sha256": sha256_file(timing_dir / "paired-timing.json"),
        "timing_measurements": len(paired["paired_timing"]["measurements"]),
    }
    _write_new(prepared / "render.json", result)
    return result


def _transform_frames(frames, kind: str, options: dict[str, Any], seed: int):
    import numpy as np
    from PIL import Image, ImageFilter

    if kind == "freeze":
        return np.repeat(frames[[options["frame_index"]]], len(frames), axis=0)
    if kind in {"blur_mild", "blur_severe"}:
        return np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).filter(
                        ImageFilter.GaussianBlur(options["gaussian_radius_px"])
                    )
                )
                for frame in frames
            ]
        )
    if kind == "shuffle":
        order = np.random.default_rng(seed).permutation(len(frames))
        if np.array_equal(order, np.arange(len(frames))):
            order[[0, 1]] = order[[1, 0]]
        return frames[order]
    if kind == "detail_loss":
        height, width = frames.shape[1:3]
        small = (
            max(1, width // options["downscale_factor"]),
            max(1, height // options["downscale_factor"]),
        )
        return np.stack(
            [
                np.asarray(
                    Image.fromarray(frame)
                    .resize(small)
                    .resize((width, height), Image.Resampling.NEAREST)
                )
                for frame in frames
            ]
        )
    if kind.startswith("evaluator_attack_"):
        output = frames.astype(np.int16).copy()
        amount, period = options["temporal_luma_offset"], options["period_frames"]
        output[::period] = np.clip(output[::period] + amount, 0, 255)
        return output.astype(np.uint8)
    raise ValueError(f"unsupported derived transform {kind}")


def derive(prepared: Path) -> dict[str, object]:
    """Create eight non-overwriting derived videos and pending judge pairs."""

    import imageio
    import numpy as np

    plan = _read(prepared / "transforms.json")
    controls = _read(prepared / "controls.json")["items"]
    render_dir, out = prepared / "quality", prepared / "derived"
    out.mkdir(mode=0o700)

    def verified_video(job_id: str) -> Path:
        receipt_path, video = render_dir / f"{job_id}.json", render_dir / f"{job_id}.mp4"
        receipt = _read(receipt_path)
        if receipt.get("status") != "OK" or receipt.get("video_sha256") != sha256_file(video):
            raise ValueError(f"completed receipt does not verify source video: {job_id}")
        return video

    def decoded(path: Path):
        reader = imageio.get_reader(str(path), "ffmpeg")
        try:
            # Comprehension deliberately avoids list(reader), whose length hint
            # can make imageio preallocate without a bounded frame count.
            return np.stack([frame for frame in reader])
        finally:
            reader.close()

    results = []
    for item in plan["items"]:
        reference = verified_video(item["source_job_id"])
        source = reference
        target = out / f"derived-{item['kind']}.mp4"
        if item["kind"] == "wrong_video":
            source = verified_video(item["replacement_job_id"])
            shutil.copyfile(source, target)
        else:
            frames = decoded(source)
            changed = _transform_frames(frames, item["kind"], item["definition"]["options"], 91_700)
            writer = imageio.get_writer(str(target), fps=16, codec="libx264", macro_block_size=None)
            try:
                for frame in changed:
                    writer.append_data(frame)
            finally:
                writer.close()
        reference_hash, source_hash, output_hash = (
            sha256_file(reference),
            sha256_file(source),
            sha256_file(target),
        )
        structurally_changed = not np.array_equal(decoded(reference), decoded(target))
        results.append(
            {
                **item,
                "reference_video_sha256": reference_hash,
                "derivation_source_video_sha256": source_hash,
                "candidate_video_sha256": output_hash,
                "structurally_changed": structurally_changed,
                "observable": None,
                "video": target.name,
            }
        )
    pairs = []
    by_kind = {row["kind"]: row for row in results}
    for control in controls:
        candidate = control["candidate_id"]
        if candidate.startswith("derived-"):
            derived = by_kind[candidate.removeprefix("derived-")]
            candidate = derived["video"]
        pairs.append({**control, "candidate_id": candidate, "observable": None})
    receipt = {
        "schema": "forge/rtx5090-probe-derived/v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "transforms": results,
        "judge_pairs": pairs,
        "required_control_kinds": list(CONTROL_KINDS),
    }
    _write_new(prepared / "derived.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--probe", "--manifest", dest="probe", type=Path, required=True)
    prepare_command.add_argument("--contract", type=Path, required=True)
    prepare_command.add_argument("--base-manifest", type=Path, required=True)
    prepare_command.add_argument("--base-dir", type=Path, required=True)
    prepare_command.add_argument("--validator-hotkey", required=True)
    prepare_command.add_argument("--max-generation-seconds", type=float, required=True)
    prepare_command.add_argument("--out", type=Path, required=True)
    for name in ("render", "derive"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            probe=_read(args.probe),
            contract=ChallengeContract.from_dict(_read(args.contract)),
            base=ModelManifest.from_dict(_read(args.base_manifest)),
            base_dir=args.base_dir,
            validator_hotkey=args.validator_hotkey,
            max_generation_seconds=args.max_generation_seconds,
            out=args.out,
        )
    elif args.command == "render":
        result = render(args.prepared)
    else:
        result = derive(args.prepared)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
