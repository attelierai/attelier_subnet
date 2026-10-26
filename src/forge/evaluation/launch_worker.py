"""Validator-owned generation and scoring receipts for the provisional V1 gate.

GPU commands run only on the designated remote workspace. A score binds the
exact completed generator report, prompt, seed, model and video bytes. These
files are evidence from a trusted worker, not an attestation of arbitrary files
uploaded by a miner. Chain signing and reward publication are separate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from forge.evaluation import launch_policy
from forge.evaluation.evaluator_study import write_new
from forge.evaluation.flicker_guard import envelope as flicker_guard_envelope
from forge.evaluation.launch_policy import diversity_seeds
from forge.evaluation.video_reward import load_inference, remote_path
from forge.evaluation.vision_media import load_video, media_identity
from forge.models.artifact_manifest import ModelManifest
from forge.models.paired_runtime import run_paired_panel
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def render(job: dict, output: Path) -> dict:
    """Execute an explicit frozen batch; never overwrite interrupted evidence."""
    output = remote_path(output)
    models = {name: ModelManifest.from_dict(job[name]) for name in ("base", "king", "candidate")}
    return run_paired_panel(
        base=models["base"],
        base_dir=remote_path(Path(job["base_dir"])),
        reference_model=models["king"],
        reference_dir=remote_path(Path(job["king_dir"])),
        candidate=models["candidate"],
        candidate_dir=remote_path(Path(job["candidate_dir"])),
        prompts=job["prompts"],
        contract=ChallengeContract.from_dict(job["contract"]),
        out=output,
        timing_design=job["timing_design"],
        frozen_plan_id=job["plan_id"],
        baseline_self_control=job.get("baseline_self_control", False),
    )


def score_reports(
    *,
    reports: list[Path],
    spec: dict,
    assets: Path,
    output: Path,
    plan_id: str,
    expected_evaluator_id: str | None = None,
) -> dict:
    """Score all first renders; immutable receipts make reruns skip finished work."""
    output, assets = remote_path(output), remote_path(assets)
    score, evaluator_id, runtime = load_inference(spec, assets)
    if expected_evaluator_id is not None and expected_evaluator_id != evaluator_id:
        raise ValueError("evaluator runtime changed after freezing the plan")
    spec_hash = hash_canonical("forge/launch-evaluator-spec/v1", spec)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"evaluator_id": evaluator_id, "spec_hash": spec_hash, "runtime": runtime}
    runtime_file = output / "evaluator-runtime.json"
    if runtime_file.exists():
        if read(runtime_file) != metadata:
            raise ValueError("output already belongs to another evaluator")
    else:
        write_new(runtime_file, metadata)
    completed = 0
    for report_path in reports:
        report_path = remote_path(report_path)
        report = read(report_path)
        if report.get("schema") != "forge/panel/v1" or report.get("admission") not in {
            "public_snapshot",
            "baseline_control",
        }:
            raise ValueError("only validator public-snapshot render reports are accepted")
        manifest = ModelManifest.from_dict(report["manifest"])
        if manifest.fingerprint() != report["manifest_hash"]:
            raise ValueError("render manifest changed")
        prompt_map = {p["prompt_id"]: p for p in report["prompts"]}
        seen = set()
        for observation in report["observations"]:
            if observation["status"] != "OK":
                raise ValueError("incomplete or failed generation cannot be scored as success")
            prompt = prompt_map[observation["prompt_id"]]
            identity = (observation["prompt_id"], observation["seed"])
            if identity in seen or observation["seed"] not in prompt["video_seeds"]:
                raise ValueError("duplicate or foreign generation observation")
            seen.add(identity)
            video = report_path.parent / observation["video"]
            if video.is_symlink() or video.resolve().parent != report_path.parent.resolve():
                raise ValueError("video must belong to its render report directory")
            if sha256_file(video) != observation["video_sha256"]:
                raise ValueError("rendered video bytes changed")
            binding = {
                "schema": "forge/launch-score/v1",
                "plan_id": plan_id,
                "prompt_id": observation["prompt_id"],
                "prompt_text": prompt["text"],
                "seed": observation["seed"],
                "manifest_hash": manifest.fingerprint(),
                "video_sha256": observation["video_sha256"],
                "evaluator_id": evaluator_id,
                "spec_hash": spec_hash,
                "geometry": report["geometry"],
                "source_report_sha256": sha256_file(report_path),
            }
            target = output / (hash_canonical("forge/launch-score-binding/v1", binding) + ".json")
            if target.exists():
                existing = read(target)
                if {k: v for k, v in existing.items() if k != "result"} != binding:
                    raise ValueError("existing score belongs to different evidence")
            else:
                result = score(video, prompt["text"], report["geometry"])
                if result["video_sha256"] != binding["video_sha256"]:
                    raise ValueError("video changed during scoring")
                diagnostics = result.get("diagnostics")
                if not isinstance(diagnostics, dict):
                    raise ValueError("scorer diagnostics are invalid")
                diagnostics["flicker_guard_v1"] = flicker_guard_envelope(
                    load_video(video, report["geometry"]).frames
                )
                if sha256_file(video) != binding["video_sha256"]:
                    raise ValueError("rendered video bytes changed during CPU guard decode")
                write_new(target, {**binding, "result": result})
            completed += 1
        expected = {(p["prompt_id"], seed) for p in report["prompts"] for seed in p["video_seeds"]}
        if seen != expected:
            raise ValueError("render report has incomplete prompt coverage")
    return {"status": "COMPLETE", "scores": completed, **metadata}


def diversity_receipt(*, report: Path, plan: Mapping[str, object], contract_hash: str) -> dict:
    """Bind V2 diversity to decoded pixels, never container/file hashes.

    The report is still validator-owned paired-runtime output.  This small
    adapter gives the bridge three actual seed renders per prescribed prompt;
    codecs may differ while equal decoded pixels still count as an exact copy.
    """
    report = Path(report)
    value = read(report)
    if value.get("schema") != "forge/panel/v1" or value.get("admission") != "public_snapshot":
        raise ValueError("only public-snapshot paired render reports are accepted")
    if plan.get("schema") != "forge/launch-policy/v2":
        raise ValueError("diversity requires a frozen V2 policy plan")
    launch_policy._check_plan_v2(dict(plan))
    plan_id = plan.get("plan_id")
    prompt_ids = plan.get("diversity_prompt_ids")
    if not isinstance(plan_id, str) or not isinstance(prompt_ids, list):
        raise ValueError("V2 diversity plan identity is invalid")
    geometry, prompts = (
        value.get("geometry"),
        {item["prompt_id"]: item for item in value.get("prompts", [])},
    )
    if (
        value.get("contract_hash") != contract_hash
        or value.get("geometry") != plan.get("geometry")
        or value.get("runtime_id") != plan.get("runtime", {}).get("runtime_id")
        or value.get("manifest_hash") != plan.get("models", {}).get("candidate")
        or hash_canonical("forge/launch-hardware/v1", value.get("hardware"))
        != plan.get("runtime", {}).get("hardware_id")
        or hash_canonical(
            "forge/launch-dependencies/v1", value.get("runtime", {}).get("dependencies")
        )
        != plan.get("runtime", {}).get("dependency_lock_hash")
    ):
        raise ValueError("diversity report does not bind candidate runtime or contract")
    if len(prompt_ids) != 4 or len(set(prompt_ids)) != 4 or set(prompt_ids) != set(prompts):
        raise ValueError("V2 diversity needs four prescribed prompt ids")
    observations: dict[str, list[dict]] = {prompt_id: [] for prompt_id in prompt_ids}
    for observation in value.get("observations", []):
        if observation.get("prompt_id") in observations and observation.get("status") == "OK":
            observations[observation["prompt_id"]].append(observation)
    rows = []
    for prompt_id in prompt_ids:
        seeds = diversity_seeds(plan, prompt_id)
        prompt = prompts[prompt_id]
        if (
            prompt.get("text")
            != next(
                item.get("text") for item in plan["prompts"] if item.get("prompt_id") == prompt_id
            )
            or prompt.get("video_seeds") != seeds
        ):
            raise ValueError("diversity report prompt or seeds differ from the frozen plan")
        selected = sorted(observations[prompt_id], key=lambda item: seeds.index(item["seed"]))
        if len(selected) != 3 or [item.get("seed") for item in selected] != seeds:
            raise ValueError("V2 diversity evidence does not cover the prescribed seeds")
        decoded = []
        source_videos = []
        for observation in selected:
            video = report.parent / observation["video"]
            if (
                not video.is_file()
                or video.is_symlink()
                or sha256_file(video) != observation["video_sha256"]
            ):
                raise ValueError("diversity video is absent or changed")
            source_videos.append(observation["video_sha256"])
            decoded.append(media_identity(load_video(video, geometry).frames))
        rows.append(
            {
                "type": "diversity",
                "plan_id": plan_id,
                "prompt_id": prompt_id,
                "seeds": seeds,
                "source_video_sha256": source_videos,
                "decoded_video_sha256": decoded,
            }
        )
    body = {
        "schema": "forge/launch-diversity/v2",
        "plan_id": plan_id,
        "source_report_sha256": sha256_file(report),
        "rows": rows,
    }
    return {**body, "receipt_id": hash_canonical("forge/launch-diversity/v2", body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("render")
    generation.add_argument("--job", type=Path, required=True)
    generation.add_argument("--out", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--report", action="append", type=Path, required=True)
    scoring.add_argument("--spec", type=Path, required=True)
    scoring.add_argument("--assets", type=Path, required=True)
    scoring.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--plan-id", required=True)
    scoring.add_argument("--evaluator-id")
    diversity = commands.add_parser("diversity")
    diversity.add_argument("--report", type=Path, required=True)
    diversity.add_argument("--plan", type=Path, required=True)
    diversity.add_argument("--contract-hash", required=True)
    diversity.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render":
        result = render(read(args.job), args.out)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "runtime_id": result["paired_timing"]["runtime_id"],
                }
            )
        )
    elif args.command == "score":
        result = score_reports(
            reports=args.report,
            spec=read(args.spec),
            assets=args.assets,
            output=args.out,
            plan_id=args.plan_id,
            expected_evaluator_id=args.evaluator_id,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "runtime"}))
    else:
        result = diversity_receipt(
            report=args.report, plan=read(args.plan), contract_hash=args.contract_hash
        )
        write_new(remote_path(args.out), result)
        print(json.dumps({"status": "COMPLETE", "receipt_id": result["receipt_id"]}))


if __name__ == "__main__":
    main()
