"""Prepare the frozen pilot render bundle; execution requires an explicit GPU host."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path

from forge.evaluation.pilot_media_index import validate_pilot_media_index
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.evaluation.vision_media import load_video
from forge.models.artifact_manifest import ModelManifest
from forge.models.distributed_worker import DiagnosticJob, DiagnosticJobBundle, run_diagnostic_jobs
from forge.models.hardware_probe import _transform_frames
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import sha256_file


def remaining(deadline: float) -> float:
    if not math.isfinite(deadline) or deadline <= time.time():
        raise TimeoutError("INCOMPLETE: absolute pilot working deadline elapsed")
    return deadline - time.time()


def save_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def diagnostic_job(item: dict, manifest: ModelManifest, base_dir: Path) -> DiagnosticJob:
    return DiagnosticJob(
        item["job_id"],
        item["role"],
        item["source_id"],
        item["prompt_text"],
        item["seed"],
        manifest,
        base_dir,
        "original_unipc",
    )


def prepare(
    *, pilot: dict, manifest: ModelManifest, base_dir: Path, deadline_unix: float
) -> DiagnosticJobBundle:
    work = pilot_worklist(pilot)
    remaining(deadline_unix)
    recipe = dict(work["recipe"])
    if (
        manifest.role != "pipeline"
        or manifest.reference.repo_id != recipe["base_model_id"]
        or manifest.reference.revision != recipe["base_model_revision"]
        or dict(manifest.generation_config) != {"steps": 50, "guidance_scale": 5.0}
    ):
        raise ValueError("pilot requires the pinned original Wan-50 CFG-5 pipeline")
    contract = ChallengeContract(
        max_steps=50,
        max_forward_passes=100,
        width=recipe["width"],
        height=recipe["height"],
        frames=recipe["frames"],
        fps=recipe["fps"],
    )
    jobs = [
        diagnostic_job(item, manifest, base_dir) for item in [*work["warmups"], *work["renders"]]
    ]
    return DiagnosticJobBundle(
        "pilot-local",
        str(work["pilot_id"]),
        contract,
        manifest,
        base_dir,
        tuple(jobs),
        deadline_unix - time.time(),
    )


def derive_and_index(
    *, work: dict, manifest: ModelManifest, root: Path, deadline_unix: float
) -> dict:
    """Derive frozen controls after completed renders; never regenerates a source video."""
    import imageio
    import numpy as np

    renders = root / "renders"
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    videos = {}
    generation_receipts = []
    geometry = {key: work["recipe"][key] for key in ("width", "height", "frames", "fps")}
    for job in [*work["warmups"], *work["renders"]]:
        remaining(deadline_unix)
        path = renders / f"{job['job_id']}.mp4"
        receipt = renders / f"{job['job_id']}.json"
        if not receipt.exists() or not path.exists():
            raise RuntimeError("INCOMPLETE: required render is absent")
        digest = sha256_file(path)
        raw = json.loads(receipt.read_text())
        expected_job = diagnostic_job(job, manifest, root).job_hash()
        if (
            raw.get("status") != "OK"
            or raw.get("job_hash") != expected_job
            or raw.get("video_sha256") != digest
            or raw.get("round_id") != work["pilot_id"]
            or raw.get("diagnostic_only") is not True
            or raw.get("geometry") != geometry
        ):
            raise ValueError("render receipt differs from frozen job, model or video")
        generation_receipts.append({"job_id": job["job_id"], "sha256": sha256_file(receipt)})
        if job["role"] == "warmup":
            continue
        videos[job["job_id"]] = {
            "job_id": job["job_id"],
            "path": str(path.relative_to(root)),
            "video_sha256": digest,
            "prompt_text": job["prompt_text"],
            "seed": job["seed"],
            "transformation": None,
        }
    for job in work["transforms"]:
        remaining(deadline_unix)
        target = derived / f"{job['job_id']}.mp4"
        receipt_path = derived / f"{job['job_id']}.json"
        source = videos[job["reference_job_id"]]
        treatment = {
            "reference_job_id": job["reference_job_id"],
            "reference_video_sha256": source["video_sha256"],
            "kind": job["kind"],
            "options": job["options"],
        }
        if receipt_path.exists():
            raw = json.loads(receipt_path.read_text())
            if (
                raw.get("status") != "COMPLETE"
                or raw.get("worklist_id") != work["worklist_id"]
                or raw.get("treatment") != treatment
                or not target.is_file()
                or raw.get("candidate_video_sha256") != sha256_file(target)
            ):
                raise ValueError("partial or changed transformation cannot be retried")
        else:
            if target.exists():
                raise ValueError("unreceipted transformation cannot be adopted")
            raw = {"status": "RUNNING", "worklist_id": work["worklist_id"], "treatment": treatment}
            save_new(receipt_path, raw)
            decoded = load_video(root / source["path"], geometry)
            if decoded.video_sha256 != source["video_sha256"]:
                raise ValueError("source video changed before transformation")
            frames = decoded.frames
            options = dict(job["options"])
            if job["kind"] == "shuffle" and "frame_indices" in options:
                frames = frames[np.asarray(options["frame_indices"], dtype=int)]
            else:
                frames = _transform_frames(frames, job["kind"], options, 0)
            with imageio.get_writer(
                str(target), fps=geometry["fps"], codec="libx264", macro_block_size=None
            ) as writer:
                for frame in frames:
                    remaining(deadline_unix)
                    writer.append_data(frame)
            if sha256_file(root / source["path"]) != source["video_sha256"]:
                raise ValueError("source video changed during transformation")
            raw.update(status="COMPLETE", candidate_video_sha256=sha256_file(target))
            temporary = receipt_path.with_suffix(".complete.json")
            save_new(temporary, raw)
            temporary.replace(receipt_path)
        digest = sha256_file(target)
        videos[job["job_id"]] = {
            "job_id": job["job_id"],
            "path": str(target.relative_to(root)),
            "video_sha256": digest,
            "prompt_text": source["prompt_text"],
            "seed": source["seed"],
            "transformation": {
                "reference_job_id": job["reference_job_id"],
                "reference_video_sha256": source["video_sha256"],
                "kind": job["kind"],
                "options": job["options"],
                "candidate_video_sha256": digest,
            },
        }
    index = {
        "schema": "forge/evaluator-pilot-media-index/v1",
        "status": "COMPLETE",
        "worklist_id": work["worklist_id"],
        "base_manifest_hash": manifest.fingerprint(),
        "recipe": work["recipe"],
        "videos": list(videos.values()),
        "generation_receipts": generation_receipts,
        "executor_sha256": sha256_file(__file__),
    }
    validate_pilot_media_index(work, index)
    index_path = root / "media-index.json"
    if index_path.exists():
        if json.loads(index_path.read_text()) != index:
            raise ValueError("existing media index differs from current evidence")
    else:
        save_new(index_path, index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not str(args.out.resolve()).startswith("/workspace/forge/"):
        raise ValueError("pilot output must be below /workspace/forge")
    bundle = prepare(
        pilot=json.loads(args.pilot.read_text()),
        manifest=ModelManifest.from_dict(json.loads(args.base_manifest.read_text())),
        base_dir=args.base_dir,
        deadline_unix=args.deadline_unix,
    )
    if args.execute:
        args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (args.out / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = {
                "pilot_sha256": sha256_file(args.pilot),
                "base_manifest_hash": bundle.base.fingerprint(),
                "base_dir": str(args.base_dir.resolve()),
                "deadline_unix": args.deadline_unix,
            }
            state = args.out / "execution.json"
            if state.exists():
                if json.loads(state.read_text()) != identity:
                    raise ValueError("pilot identity or deadline changed on restart")
            else:
                save_new(state, identity)
            result = run_diagnostic_jobs(bundle, args.out / "renders", warmup=False)
            if result["status"] != "COMPLETE":
                return 2
            derive_and_index(
                work=pilot_worklist(json.loads(args.pilot.read_text())),
                manifest=bundle.base,
                root=args.out,
                deadline_unix=args.deadline_unix,
            )
    else:
        print(
            json.dumps(
                {
                    "jobs": len(bundle.jobs),
                    "bundle_hash": bundle.bundle_hash(),
                    "dispatch_authorized": False,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
