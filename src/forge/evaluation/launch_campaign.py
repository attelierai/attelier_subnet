"""Execute the frozen V2 panel on a validator-owned remote GPU.

Generation and scoring use separate Python environments. Each render has a
durable intent before execution; an interrupted render is never silently rerun.
Large outputs remain under /workspace/forge. Only immutable JSON receipts are
exported to the validator's custody directory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from forge.evaluation import launch_policy
from forge.evaluation.evaluator_qualification import replay_qualification_package
from forge.evaluation.video_reward import remote_path
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_service import write_immutable_json


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("campaign input must contain a JSON object")
    return value


def batches(
    plan: dict[str, Any], models: dict[str, Any], contract: ChallengeContract
) -> list[dict[str, Any]]:
    """Build the complete measured workload, including anchor and diversity."""
    launch_policy._check_plan_v2(plan)
    identities = {}
    for role in ("base", "king", "candidate", "anchor"):
        entry = models[role]
        manifest = ModelManifest.from_dict(entry["manifest"])
        remote_path(Path(entry["directory"]))
        identities[role] = manifest
    for role in ("king", "candidate", "anchor"):
        if identities[role].fingerprint() != plan["models"][role]:
            raise ValueError("campaign manifest differs from the frozen policy")
    if identities["base"].fingerprint() != identities["anchor"].fingerprint():
        raise ValueError("campaign base must be the original Wan50 quality anchor")
    if (
        identities["base"].role != "pipeline"
        or identities["base"].reference.repo_id != contract.base_model_id
        or identities["base"].reference.revision != contract.base_model_revision
        or plan["geometry"]
        != {key: getattr(contract, key) for key in ("width", "height", "frames", "fps")}
    ):
        raise ValueError("campaign base or geometry differs from its challenge")

    def prompts(ids: list[str], *, diversity: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "prompt_id": item["prompt_id"],
                "text": item["text"],
                "video_seeds": launch_policy.diversity_seeds(plan, item["prompt_id"])
                if diversity
                else [item["seed"]],
            }
            for item in plan["prompts"]
            if item["prompt_id"] in ids
        ]

    specs = [
        ("timing", "king", "confirmation_abba", prompts(plan["timing_prompt_ids"])),
        (
            "quality",
            "king",
            "screening_ab",
            prompts(
                [
                    item["prompt_id"]
                    for item in plan["prompts"]
                    if item["prompt_id"] not in plan["timing_prompt_ids"]
                ]
            ),
        ),
        (
            "diversity",
            "king",
            "screening_ab",
            prompts(plan["diversity_prompt_ids"], diversity=True),
        ),
    ]
    if plan["models"]["anchor"] not in {plan["models"]["king"], plan["models"]["candidate"]}:
        # Reuse the identical paired engine. Only the anchor side is imported;
        # these extra candidate renders are operating cost, not competitive data.
        specs.append(
            (
                "anchor",
                "anchor",
                "screening_ab",
                prompts([item["prompt_id"] for item in plan["prompts"]]),
            )
        )
    result = []
    for name, reference, design, panel in specs:
        result.append(
            {
                "name": name,
                "base": identities["base"].to_dict(),
                "king": identities[reference].to_dict(),
                "candidate": identities["candidate"].to_dict(),
                "base_dir": models["base"]["directory"],
                "king_dir": models[reference]["directory"],
                "candidate_dir": models["candidate"]["directory"],
                "contract": contract.to_dict(),
                "prompts": panel,
                "timing_design": design,
                "plan_id": plan["plan_id"],
            }
        )
    return result


def prepare(
    root: Path,
    *,
    plan: dict[str, Any],
    models: dict[str, Any],
    contract: ChallengeContract,
    qualification: dict[str, Any],
) -> dict[str, Any]:
    root = remote_path(root)
    if replay_qualification_package(qualification) != plan["evaluator_qualification"]:
        raise ValueError("campaign qualification differs from raw replay")
    if plan["evaluator_qualification"]["qualification"] != "QUALIFIED":
        raise ValueError("unqualified evaluator cannot start a competitive campaign")
    jobs = batches(plan, models, contract)
    body = {
        "schema": "forge/launch-campaign/v2",
        "plan_id": plan["plan_id"],
        "contract_hash": contract.contract_hash(),
        "jobs": jobs,
        "measured_generation_calls": sum(
            len(prompt["video_seeds"]) * (4 if job["timing_design"] == "confirmation_abba" else 2)
            for job in jobs
            for prompt in job["prompts"]
        ),
        "warmup_calls": len(jobs) * 2,
    }
    receipt = {**body, "campaign_hash": hash_canonical("forge/launch-campaign/v2", body)}
    root.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("launch-plan.json", plan),
        ("qualification-package.json", qualification),
        ("campaign.json", receipt),
    ):
        write_immutable_json(root / name, value)
    for job in jobs:
        write_immutable_json(root / f"{job['name']}-job.json", job)
    return receipt


def render_all(root: Path, *, custody_deadline: float) -> dict[str, Any]:
    if not math.isfinite(custody_deadline) or custody_deadline <= time.time():
        raise ValueError("custody deadline must be finite and in the future")
    from forge.evaluation.launch_worker import render

    root = remote_path(root)
    campaign = read(root / "campaign.json")
    body = {key: value for key, value in campaign.items() if key != "campaign_hash"}
    if campaign["campaign_hash"] != hash_canonical("forge/launch-campaign/v2", body):
        raise ValueError("campaign changed after preparation")
    for job in campaign["jobs"]:
        name = job["name"]
        if read(root / f"{name}-job.json") != job:
            raise ValueError("render job differs from its frozen campaign")
        intent, completed = root / f"{name}-intent.json", root / f"{name}-complete.json"
        job_hash = hash_canonical("forge/launch-render-job/v2", job)
        if completed.exists():
            receipt = read(completed)
            if receipt["job_hash"] != job_hash:
                raise ValueError("completed render belongs to another job")
            for relative, digest in receipt["files"].items():
                if sha256_file(root / relative) != digest:
                    raise ValueError("completed render evidence changed")
            _preserve_render(root, name, custody_deadline)
            continue
        if intent.exists():
            raise ValueError(
                f"render {name} was interrupted; retained evidence needs reconciliation"
            )
        write_immutable_json(intent, {"job_hash": job_hash, "status": "EXECUTION_UNKNOWN"})
        render(job, root / name)
        timing = (
            "paired-timing.json"
            if job["timing_design"] == "confirmation_abba"
            else "screen-timing.json"
        )
        files = [
            root / name / "reference/report.json",
            root / name / "candidate/report.json",
            root / name / timing,
        ]
        write_immutable_json(
            completed,
            {
                "job_hash": job_hash,
                "files": {str(path.relative_to(root)): sha256_file(path) for path in files},
            },
        )
        _preserve_render(root, name, custody_deadline)
    return {"status": "RENDERED", "campaign_hash": campaign["campaign_hash"]}


def _preserve_render(root: Path, name: str, deadline: float) -> None:
    from forge.evaluation.pilot_custody import preserve_batch

    paths = [p for p in (root / name).rglob("*") if p.suffix in {".json", ".mp4"}]
    paths += [root / f"{name}-{suffix}.json" for suffix in ("job", "complete")]
    paths += [root / file for file in ("campaign.json", "launch-plan.json")]
    preserve_batch(root, name, paths, deadline)


def preflight(root: Path) -> dict[str, Any]:
    """Verify downloaded tensor identity before permission to render is issued."""
    from forge.models.weights import validate_candidate_weights
    from forge.validator.runtime_preflight import (
        effective_candidate_hash,
        legacy_effective_candidate_hash,
        verify_local,
    )

    root = remote_path(root)
    campaign = read(root / "campaign.json")
    body = {key: value for key, value in campaign.items() if key != "campaign_hash"}
    if campaign["campaign_hash"] != hash_canonical("forge/launch-campaign/v2", body):
        raise ValueError("campaign changed before tensor preflight")
    job = campaign["jobs"][0]
    base = ModelManifest.from_dict(job["base"])
    candidate = ModelManifest.from_dict(job["candidate"])
    contract = ChallengeContract.from_dict(job["contract"])
    verify_local(base, Path(job["base_dir"]))
    if Path(job["candidate_dir"]).resolve() != Path(job["base_dir"]).resolve():
        verify_local(candidate, Path(job["candidate_dir"]))
    weights = validate_candidate_weights(
        candidate, Path(job["candidate_dir"]), base, Path(job["base_dir"])
    )
    receipt = {
        "schema": "forge/launch-effective-admission/v1",
        "plan_id": campaign["plan_id"],
        "contract_hash": contract.contract_hash(),
        "manifest_hash": candidate.fingerprint(),
        "effective_hash": effective_candidate_hash(candidate, weights, base, contract),
        "legacy_effective_hash": legacy_effective_candidate_hash(
            candidate, weights, base, contract
        ),
        "weight_content_hash": weights.content_fingerprint,
    }
    write_immutable_json(root / "effective-admission.json", receipt)
    return receipt


def score_all(
    root: Path, *, spec: dict[str, Any], assets: Path, custody_deadline: float
) -> dict[str, Any]:
    if not math.isfinite(custody_deadline) or custody_deadline <= time.time():
        raise ValueError("custody deadline must be finite and in the future")
    from forge.evaluation.launch_worker import diversity_receipt, score_reports

    root = remote_path(root)
    campaign, plan = read(root / "campaign.json"), read(root / "launch-plan.json")
    body = {key: value for key, value in campaign.items() if key != "campaign_hash"}
    if (
        campaign["campaign_hash"] != hash_canonical("forge/launch-campaign/v2", body)
        or campaign["plan_id"] != plan["plan_id"]
    ):
        raise ValueError("scoring campaign changed after preparation")
    launch_policy._check_plan_v2(plan)
    if any(not (root / f"{job['name']}-complete.json").is_file() for job in campaign["jobs"]):
        raise ValueError("all render shards must complete before scoring")
    for job in campaign["jobs"]:
        receipt = read(root / f"{job['name']}-complete.json")
        if receipt["job_hash"] != hash_canonical("forge/launch-render-job/v2", job):
            raise ValueError("completed render belongs to another frozen job")
        for relative, digest in receipt["files"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError("render evidence changed before scoring")
    reports = [
        root / name / side / "report.json"
        for name in ("timing", "quality")
        for side in ("reference", "candidate")
    ]
    if plan["models"]["anchor"] not in {plan["models"]["king"], plan["models"]["candidate"]}:
        reports.append(root / "anchor/reference/report.json")
    result = score_reports(
        reports=reports,
        spec=spec,
        assets=assets,
        output=root / "scores",
        plan_id=plan["plan_id"],
        expected_evaluator_id=plan["evaluator"]["evaluator_id"],
    )
    diversity = diversity_receipt(
        report=root / "diversity/candidate/report.json",
        plan=plan,
        contract_hash=campaign["contract_hash"],
    )
    write_immutable_json(root / "diversity-receipt.json", diversity)
    index = {
        "schema": "forge/launch-campaign-receipts/v2",
        "plan_id": plan["plan_id"],
        "campaign_hash": campaign["campaign_hash"],
        "panels": [str(path.relative_to(root)) for path in reports],
        "timing": ["timing/paired-timing.json", "quality/screen-timing.json"],
        "scores": [
            str(path.relative_to(root))
            for path in sorted((root / "scores").glob("*.json"))
            if read(path).get("schema") == "forge/launch-score/v1"
        ],
        "diversity": ["diversity-receipt.json"],
        "diversity_panels": ["diversity/candidate/report.json"],
        "qualification": ["qualification-package.json"],
    }
    paths = [item for value in index.values() if isinstance(value, list) for item in value]
    index["sha256"] = {relative: sha256_file(root / relative) for relative in paths}
    write_immutable_json(root / "receipt-index.json", index)
    from forge.evaluation.pilot_custody import preserve_batch

    preserve_batch(
        root,
        "scores",
        [root / relative for relative in paths]
        + list(root.glob("*.json"))
        + list((root / "scores").rglob("*.json")),
        custody_deadline,
    )
    return {"status": "SCORED", "scores": result["scores"], "plan_id": plan["plan_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "render", "score"):
        child = sub.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        if command in {"render", "score"}:
            child.add_argument("--custody-deadline", type=float, required=True)
        if command == "prepare":
            for name in ("plan", "models", "contract", "qualification"):
                child.add_argument(f"--{name}", type=Path, required=True)
        elif command == "score":
            child.add_argument("--spec", type=Path, required=True)
            child.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    if args.command in {"render", "score"} and (
        not math.isfinite(args.custody_deadline) or args.custody_deadline <= time.time()
    ):
        raise ValueError("custody deadline must be finite and in the future")
    if args.command == "prepare":
        result = prepare(
            args.root,
            plan=read(args.plan),
            models=read(args.models),
            contract=ChallengeContract.from_dict(read(args.contract)),
            qualification=read(args.qualification),
        )
    elif args.command == "render":
        result = render_all(args.root, custody_deadline=args.custody_deadline)
    elif args.command == "preflight":
        result = preflight(args.root)
    else:
        result = score_all(
            args.root,
            spec=read(args.spec),
            assets=args.assets,
            custody_deadline=args.custody_deadline,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
