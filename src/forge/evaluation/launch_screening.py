"""Bounded, receipt-derived remote baseline screening.

This adapter is deliberately smaller than the 32-prompt V2 confirmation.  It
freezes one six-prompt AB panel for Wan50 against at most three explicitly
named candidate manifests, then turns paired-runtime and VideoReward receipts
into the raw rows consumed by :mod:`launch_baseline`.  A screen can choose one
candidate for confirmation; it never asserts an authoritative policy PASS.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.evaluation.evaluator_qualification import replay_qualification_package
from forge.evaluation.launch_baseline import ParentLaunchBaselineStore
from forge.evaluation.launch_worker import render, score_reports
from forge.evaluation.video_reward import remote_path
from forge.models.artifact_manifest import ModelManifest
from forge.models.generation import GenerationRecipe, effective_generation_recipe
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_service import write_immutable_json

SCHEMA = "forge/launch-baseline-screening/v1"
PREDECLARED_CANDIDATES = frozenset({"Wan44", "Wan40", "LongCat1500"})


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("screening input must contain a JSON object")
    return value


def _wan50(manifest: ModelManifest, contract: ChallengeContract) -> None:
    recipe = effective_generation_recipe(
        GenerationRecipe.from_dict(dict(manifest.generation_config)), is_original_reference=True
    )
    if (
        manifest.role != "pipeline"
        or manifest.base_reference is not None
        or (manifest.reference.repo_id, manifest.reference.revision)
        != (contract.base_model_id, contract.base_model_revision)
        or recipe
        != {
            "steps": contract.reference_steps,
            "guidance_scale": 5.0,
            "scheduler": "original_unipc",
        }
    ):
        raise ValueError("base must be the original Wan50/UniPC contract pipeline")


def _named_candidate(manifest: ModelManifest, name: str, contract: ChallengeContract) -> None:
    base = (contract.base_model_id, contract.base_model_revision)
    recipe = effective_generation_recipe(
        GenerationRecipe.from_dict(dict(manifest.generation_config)), is_original_reference=False
    )
    if name in {"Wan44", "Wan40"}:
        steps = 44 if name == "Wan44" else 40
        if (
            manifest.role != "pipeline"
            or manifest.base_reference is not None
            or (manifest.reference.repo_id, manifest.reference.revision) != base
            or recipe != {"steps": steps, "guidance_scale": 5.0, "scheduler": "original_unipc"}
        ):
            raise ValueError(f"{name} must be its fixed public Wan/UniPC recipe")
        return
    if (
        manifest.role != "lora"
        or manifest.base_reference is None
        or (manifest.base_reference.repo_id, manifest.base_reference.revision) != base
        or (manifest.reference.backend, manifest.reference.repo_id, manifest.reference.revision)
        != (
            "hf",
            "lightx2v/Wan2.1-T2V-1.3B-longcat-step1500",
            "791dc13f1e672d4643c736b273d8c91a85da6b40",
        )
        or recipe != {"steps": 16, "guidance_scale": 4.5, "scheduler": "original_unipc"}
    ):
        raise ValueError("LongCat1500 must be the fixed public step1500 LoRA recipe")


def _candidate_entries(
    candidates: list[dict[str, Any]], contract: ChallengeContract
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("screening requires one to three declared candidates")
    names, hashes, result = set(), set(), []
    for entry in candidates:
        if not isinstance(entry, dict) or set(entry) != {"name", "manifest", "directory"}:
            raise ValueError("candidate requires name, manifest and directory")
        if entry["name"] not in PREDECLARED_CANDIDATES or entry["name"] in names:
            raise ValueError("candidate is not one of the predeclared baseline alternatives")
        manifest = ModelManifest.from_dict(entry["manifest"])
        if not isinstance(entry["directory"], str):
            raise ValueError("candidate directory is invalid")
        _named_candidate(manifest, entry["name"], contract)
        fingerprint = manifest.fingerprint()
        if fingerprint in hashes:
            raise ValueError("candidate manifests must be distinct")
        names.add(entry["name"])
        hashes.add(fingerprint)
        result.append({**entry, "manifest_hash": fingerprint})
    return result


def _prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(prompts, list) or len(prompts) != 6:
        raise ValueError("screening requires exactly six frozen prompts")
    ids, result = set(), []
    for item in prompts:
        if not isinstance(item, dict) or set(item) != {"prompt_id", "text", "seed", "tags"}:
            raise ValueError("screening prompt requires prompt_id, text, seed and tags")
        if (
            not isinstance(item["prompt_id"], str)
            or not item["prompt_id"]
            or item["prompt_id"] in ids
            or not isinstance(item["text"], str)
            or not item["text"].strip()
            or type(item["seed"]) is not int
            or not 0 <= item["seed"] < 2**63
            or not isinstance(item["tags"], list)
            or len(set(item["tags"])) != len(item["tags"])
            or any(not isinstance(tag, str) or not tag for tag in item["tags"])
        ):
            raise ValueError("screening prompt fields are invalid")
        ids.add(item["prompt_id"])
        result.append(dict(item))
    return sorted(result, key=lambda item: item["prompt_id"])


def prepare(
    root: Path,
    *,
    contract: ChallengeContract,
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    qualification: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the immutable remote workload before any GPU generation."""
    root = remote_path(root)
    qualified = replay_qualification_package(qualification)
    base_manifest = ModelManifest.from_dict(base["manifest"])
    if set(base) != {"manifest", "directory"} or not isinstance(base["directory"], str):
        raise ValueError("Wan50 base requires manifest and directory")
    _wan50(base_manifest, contract)
    frozen_prompts, alternatives = _prompts(prompts), _candidate_entries(candidates, contract)
    anchor_hash = base_manifest.fingerprint()
    if any(item["manifest_hash"] == anchor_hash for item in alternatives):
        raise ValueError("candidate cannot duplicate the Wan50 anchor")
    body = {
        "schema": SCHEMA,
        "contract_hash": contract.contract_hash(),
        "contract": contract.to_dict(),
        "geometry": {key: getattr(contract, key) for key in ("width", "height", "frames", "fps")},
        "base": {**base, "manifest_hash": anchor_hash},
        "candidates": alternatives,
        "prompts": frozen_prompts,
        "evaluator_qualification": qualified,
    }
    receipt = {**body, "screening_id": hash_canonical(SCHEMA, body)}
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root / "qualification-package.json", qualification)
    write_immutable_json(root / "screening.json", receipt)
    for candidate in alternatives:
        job = {
            "name": candidate["name"],
            "base": base["manifest"],
            "king": base["manifest"],
            "candidate": candidate["manifest"],
            "base_dir": base["directory"],
            "king_dir": base["directory"],
            "candidate_dir": candidate["directory"],
            "contract": contract.to_dict(),
            "prompts": [
                {
                    "prompt_id": prompt["prompt_id"],
                    "text": prompt["text"],
                    "video_seeds": [prompt["seed"]],
                }
                for prompt in frozen_prompts
            ],
            "timing_design": "screening_ab",
            "plan_id": receipt["screening_id"],
            "baseline_self_control": False,
        }
        write_immutable_json(root / f"{candidate['name']}-job.json", job)
    return receipt


def render_all(root: Path) -> dict[str, Any]:
    root = remote_path(root)
    screening = read(root / "screening.json")
    _verify_screening(screening)
    for candidate in screening["candidates"]:
        name, job_path = candidate["name"], root / f"{candidate['name']}-job.json"
        job, intent, completed = (
            read(job_path),
            root / f"{name}-intent.json",
            root / f"{name}-complete.json",
        )
        job_hash = hash_canonical("forge/launch-screening-render-job/v1", job)
        if completed.exists():
            receipt = read(completed)
            if receipt.get("job_hash") != job_hash:
                raise ValueError("completed screening render belongs to another job")
            for relative, digest in receipt.get("files", {}).items():
                if sha256_file(root / relative) != digest:
                    raise ValueError("completed screening evidence changed")
            continue
        if intent.exists():
            raise ValueError(f"screening render {name} is unknown; reconciliation is required")
        write_immutable_json(intent, {"job_hash": job_hash, "status": "EXECUTION_UNKNOWN"})
        render(job, root / name)
        files = [root / name / side / "report.json" for side in ("reference", "candidate")]
        files.append(root / name / "screen-timing.json")
        write_immutable_json(
            completed,
            {
                "job_hash": job_hash,
                "files": {str(path.relative_to(root)): sha256_file(path) for path in files},
            },
        )
    return {"status": "RENDERED", "screening_id": screening["screening_id"]}


def _verify_screening(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "screening_id"}
    if value.get("schema") != SCHEMA or value.get("screening_id") != hash_canonical(SCHEMA, body):
        raise ValueError("screening changed after preparation")


def _score_index(score_dir: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    found = {}
    for path in score_dir.glob("*.json"):
        value = read(path)
        if value.get("schema") != "forge/launch-score/v1":
            continue
        key = (value["manifest_hash"], value["prompt_id"], value["seed"], value["video_sha256"])
        if key in found or set(value.get("result", {})) < {"scores", "diagnostics", "video_sha256"}:
            raise ValueError("score receipt is ambiguous or incomplete")
        if value["result"]["video_sha256"] != value["video_sha256"]:
            raise ValueError("score receipt observed changed video bytes")
        found[key] = {"receipt": value, "sha256": sha256_file(path)}
    return found


def _screening_rows(
    root: Path, screening: dict[str, Any], candidate: dict[str, Any], score_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    from forge.evaluation.evaluator_qualification import _decoded_guard_hits

    reference_report = read(root / candidate["name"] / "reference" / "report.json")
    candidate_report = read(root / candidate["name"] / "candidate" / "report.json")
    timing = read(root / candidate["name"] / "screen-timing.json")
    expected_geometry, anchor_hash, candidate_hash = (
        screening["geometry"],
        screening["base"]["manifest_hash"],
        candidate["manifest_hash"],
    )
    if (
        timing.get("schema") != "forge/screen-timing/v1"
        or timing.get("frozen_plan_id") != screening["screening_id"]
    ):
        raise ValueError("screen timing is not bound to this frozen screening")
    if (
        timing.get("geometry") != expected_geometry
        or timing.get("contract_hash") != screening["contract_hash"]
        or timing.get("reference_manifest_hash") != anchor_hash
        or timing.get("candidate_manifest_hash") != candidate_hash
        or timing.get("reference_admission") != "public_snapshot"
        or timing.get("candidate_admission") != "public_snapshot"
        or timing.get("design", {}).get("name") != "screening_ab"
        or timing.get("design", {}).get("order") != ["reference", "candidate"]
    ):
        raise ValueError("screen timing binding changed")
    index = _score_index(score_dir)
    for report, expected_hash in (
        (reference_report, anchor_hash),
        (candidate_report, candidate_hash),
    ):
        if (
            report.get("schema") != "forge/panel/v1"
            or report.get("admission") != "public_snapshot"
            or report.get("manifest_hash") != expected_hash
            or report.get("geometry") != expected_geometry
            or report.get("contract_hash") != screening["contract_hash"]
            or report.get("runtime_id") != timing.get("runtime_id")
            or report.get("hardware") != timing.get("hardware")
            or report.get("runtime") != timing.get("runtime")
        ):
            raise ValueError("screen report binding changed")
    if timing.get("recipes", {}).get("reference") != reference_report.get(
        "recipe", {}
    ) | reference_report.get("scheduler_recipe", {}) or timing.get("recipes", {}).get(
        "candidate"
    ) != candidate_report.get("recipe", {}) | candidate_report.get("scheduler_recipe", {}):
        raise ValueError("screen timing recipe changed")
    prompt_map = {item["prompt_id"]: item for item in screening["prompts"]}
    reports = {"reference": reference_report, "candidate": candidate_report}
    observations = {
        side: {
            (item.get("prompt_id"), item.get("seed")): item
            for item in report.get("observations", [])
        }
        for side, report in reports.items()
    }
    source_paths = [
        root / candidate["name"] / "reference" / "report.json",
        root / candidate["name"] / "candidate" / "report.json",
        root / candidate["name"] / "screen-timing.json",
    ]
    panel_incomplete = any(
        len(observations[side]) != len(prompt_map)
        or any(item.get("status") != "OK" for item in observations[side].values())
        for side in observations
    ) or any(item.get("status") != "OK" for item in timing.get("measurements", []))
    rows, bindings = [], []
    for prompt_id, prompt in prompt_map.items():
        ref, cand = (
            observations["reference"].get((prompt_id, prompt["seed"])),
            observations["candidate"].get((prompt_id, prompt["seed"])),
        )
        statuses = [item.get("status") if item else "MISSING" for item in (ref, cand)]
        row = {"prompt_id": prompt_id, "tags": prompt["tags"], "run_statuses": statuses}
        if panel_incomplete or any(status != "OK" for status in statuses):
            row.update(
                status="INFRA_FAILURE"
                if "INFRA_FAILURE" in statuses or "MISSING" in statuses
                else "MODEL_FAILURE",
                vetoes=["infrastructure_incomplete"]
                if "INFRA_FAILURE" in statuses or "MISSING" in statuses
                else ["model_generation_failed"],
            )
            rows.append(row)
            continue
        measures = [
            item for item in timing.get("measurements", []) if item.get("prompt_id") == prompt_id
        ]
        if (
            len(measures) != 2
            or [item.get("side") for item in measures] != ["reference", "candidate"]
            or any(item.get("status") != "OK" for item in measures)
        ):
            row.update(status="INFRA_FAILURE", vetoes=["infrastructure_incomplete"])
            rows.append(row)
            continue
        scored = []
        for report, observation in ((reference_report, ref), (candidate_report, cand)):
            key = (
                report["manifest_hash"],
                prompt_id,
                prompt["seed"],
                observation.get("video_sha256"),
            )
            if key not in index:
                raise ValueError("completed render has no matching score receipt")
            source = index[key]["receipt"]
            if (
                source.get("plan_id") != screening["screening_id"]
                or source.get("evaluator_id")
                != screening["evaluator_qualification"]["evaluator_id"]
                or source.get("spec_hash") != screening["evaluator_qualification"]["spec_hash"]
                or source.get("geometry") != expected_geometry
                or source.get("source_report_sha256")
                != sha256_file(
                    root
                    / candidate["name"]
                    / ("reference" if report is reference_report else "candidate")
                    / "report.json"
                )
            ):
                raise ValueError("score receipt binding changed")
            scored.append(index[key])
        vetoes = sorted(
            guard
            for guard in _decoded_guard_hits(
                scored[0]["receipt"]["result"]["diagnostics"],
                scored[1]["receipt"]["result"]["diagnostics"],
                screening["evaluator_qualification"]["guard_thresholds"],
            )
            if not (
                guard == "blur"
                and screening["evaluator_qualification"].get("qualification_version") == 2
            )
            if not (guard == "freeze" and "motion_required" not in prompt["tags"])
            and not (guard == "flicker" and "steady_lighting" not in prompt["tags"])
        )
        row.update(
            status="OK",
            vetoes=vetoes,
            reference_seconds=measures[0]["wall_time_seconds"],
            candidate_seconds=measures[1]["wall_time_seconds"],
            reference_scores=scored[0]["receipt"]["result"]["scores"],
            candidate_scores=scored[1]["receipt"]["result"]["scores"],
        )
        rows.append(row)
        bindings.append(
            {
                "prompt_id": prompt_id,
                "reference_report_sha256": sha256_file(
                    root / candidate["name"] / "reference" / "report.json"
                ),
                "candidate_report_sha256": sha256_file(
                    root / candidate["name"] / "candidate" / "report.json"
                ),
                "timing_sha256": sha256_file(root / candidate["name"] / "screen-timing.json"),
                "score_sha256": [scored[0]["sha256"], scored[1]["sha256"]],
            }
        )
    receipt = {
        "candidate_hash": candidate_hash,
        "anchor_hash": anchor_hash,
        "rows": sorted(rows, key=lambda item: item["prompt_id"]),
    }
    sources = {
        "schema": "forge/launch-baseline-screening-sources/v1",
        "screening_id": screening["screening_id"],
        "candidate_hash": candidate_hash,
        "raw_sha256": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "bindings": bindings,
    }
    return receipt, sources


def score_all(root: Path, *, spec: dict[str, Any], assets: Path) -> dict[str, Any]:
    root, assets = remote_path(root), remote_path(assets)
    screening, package = read(root / "screening.json"), read(root / "qualification-package.json")
    _verify_screening(screening)
    if replay_qualification_package(package) != screening["evaluator_qualification"]:
        raise ValueError("raw qualification replay differs from frozen screening")
    outputs = []
    for candidate in screening["candidates"]:
        complete = root / f"{candidate['name']}-complete.json"
        if not complete.is_file():
            raise ValueError("all screening renders must complete before scoring")
        reports = [
            root / candidate["name"] / side / "report.json" for side in ("reference", "candidate")
        ]
        score_dir = root / "scores" / candidate["name"]
        raw_reports = [read(path) for path in reports]
        incomplete = any(
            item.get("status") != "OK"
            for report in raw_reports
            for item in report.get("observations", [])
        )
        result = (
            {"status": "INFRASTRUCTURE_INCOMPLETE", "scores": 0}
            if incomplete
            else score_reports(
                reports=reports,
                spec=spec,
                assets=assets,
                output=score_dir,
                plan_id=screening["screening_id"],
                expected_evaluator_id=screening["evaluator_qualification"]["evaluator_id"],
            )
        )
        receipt, sources = _screening_rows(root, screening, candidate, score_dir)
        write_immutable_json(root / f"{candidate['name']}-screening.json", receipt)
        write_immutable_json(root / f"{candidate['name']}-screening-sources.json", sources)
        outputs.append(
            {
                "name": candidate["name"],
                "status": "SCREENED",
                "scores": result["scores"],
                "screening_sha256": sha256_file(root / f"{candidate['name']}-screening.json"),
            }
        )
    return {"status": "SCORED", "screening_id": screening["screening_id"], "candidates": outputs}


def select_once(root: Path, db: Path) -> dict[str, Any]:
    root = remote_path(root)
    screening = read(root / "screening.json")
    _verify_screening(screening)
    receipts = [
        read(root / f"{candidate['name']}-screening.json") for candidate in screening["candidates"]
    ]
    return ParentLaunchBaselineStore(db).select_once(
        receipts, anchor_hash=screening["base"]["manifest_hash"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "render", "score", "select"):
        item = commands.add_parser(command)
        item.add_argument("--root", type=Path, required=True)
        if command == "prepare":
            for name in ("contract", "base", "candidates", "prompts", "qualification"):
                item.add_argument(f"--{name}", type=Path, required=True)
        elif command == "score":
            item.add_argument("--spec", type=Path, required=True)
            item.add_argument("--assets", type=Path, required=True)
        elif command == "select":
            item.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.root,
            contract=ChallengeContract.from_dict(read(args.contract)),
            base=read(args.base),
            candidates=json.loads(args.candidates.read_text()),
            prompts=json.loads(args.prompts.read_text()),
            qualification=read(args.qualification),
        )
    elif args.command == "render":
        result = render_all(args.root)
    elif args.command == "score":
        result = score_all(args.root, spec=read(args.spec), assets=args.assets)
    else:
        result = select_once(args.root, args.db)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
