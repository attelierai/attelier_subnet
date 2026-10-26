"""Trusted orchestration of blinded Luna comparisons over validator-rendered video.

This writes raw automatic evidence. Calibration and promotion decisions belong to
separate gates and are intentionally absent from this module.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.evaluation.raster_integrity import RULE_ID as RASTER_INTEGRITY_RULE_ID
from forge.evaluation.raster_integrity import check_frames
from forge.evaluation.vision_judge import ApiBudget, evaluator_id, evaluator_spec, judge_pair
from forge.evaluation.vision_media import (
    load_video,
    make_sheets,
    select_detail_crops,
    select_detail_frames,
    select_transition_frames,
)
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file

SCHEMA = "forge/automatic-quality/v1"


@dataclass(frozen=True)
class _Video:
    prompt_id: str
    seed: int
    prompt: str
    sha256: str
    path: Path


@dataclass(frozen=True)
class _Report:
    manifest_hash: str
    admission: str
    videos: tuple[_Video, ...]


def default_policy() -> dict[str, Any]:
    """Return the minimum automatic policy; it is not a calibration certificate."""

    body = {
        "evaluator_id": evaluator_id(),
        "min_prompts": 128,
        "min_slice_prompts": 24,
        "quality_margin": 0.02,
        "min_speedup": 1.1,
        "alpha": 0.05,
        "max_candidates": 16,
        "max_order_disagreement": 0.25,
        "quality_aggregation": "symmetric_mean_prompt_clustered",
    }
    return {
        "policy_id": hash_canonical("forge/auto-policy/v3", body),
        **body,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read JSON evidence {path.name}: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _geometry(plan: Mapping[str, Any]) -> dict[str, int]:
    contract_data = plan.get("contract")
    if not isinstance(contract_data, Mapping):
        raise ValueError("plan must contain its exact contract")
    contract = ChallengeContract.from_dict(dict(contract_data))
    if plan.get("contract_hash") != contract.contract_hash():
        raise ValueError("plan contract hash does not bind its contract")
    geometry = {
        "width": contract.width,
        "height": contract.height,
        "frames": contract.frames,
        "fps": contract.fps,
    }
    if "geometry" in plan and plan["geometry"] != geometry:
        raise ValueError("plan geometry differs from its contract")
    return geometry


def _prompts(plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    prompts = plan.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("plan prompts are required")
    slots: list[tuple[str, int]] = []
    seen_prompts: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("plan prompts must be objects")
        prompt_id, text, seeds = (
            prompt.get("prompt_id"),
            prompt.get("text"),
            prompt.get("video_seeds"),
        )
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or prompt_id in seen_prompts
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(seeds) != len(set(seeds))
        ):
            raise ValueError("plan prompts do not have unique exact slots")
        seen_prompts.add(prompt_id)
        slots.extend((prompt_id, seed) for seed in seeds)
    return prompts, slots


def _safe_video(report_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("video path must be a relative basename")
    candidate = report_path.parent / value
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("video must be a regular non-symlink file")
    try:
        candidate.resolve().relative_to(report_path.parent.resolve())
    except ValueError:
        raise ValueError("video path escapes its report directory") from None
    return candidate


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_abba(report: Mapping[str, Any], role: str) -> None:
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{role} is missing runtime evidence")
    if runtime.get("scheduler") == "UniformFlowEuler":
        # Historical reports used one fixed scheduler for both sides.
        pass
    elif runtime.get("scheduler") == "paired-exact-recipes":
        side = "reference" if role == "incumbent" else "candidate"
        schedulers = runtime.get("schedulers")
        recipe = report.get("scheduler_recipe")
        if (
            not isinstance(schedulers, Mapping)
            or not isinstance(recipe, Mapping)
            or recipe.get("kind") != schedulers.get(side)
            or recipe.get("kind") not in {"uniform_euler", "original_unipc"}
        ):
            raise ValueError(f"{role} paired runtime recipe is incomplete")
    else:
        raise ValueError(f"{role} must use a recognized ABBA runtime")
    expected_orders = (0, 3) if role == "incumbent" else (1, 2)
    for observation in report["observations"]:
        repeats = observation.get("timing_repeats")
        if not isinstance(repeats, list) or len(repeats) != 2:
            raise ValueError(f"{role} report is missing its two ABBA timing repeats")
        if tuple(item.get("run_order") for item in repeats) != expected_orders:
            raise ValueError(f"{role} report has an invalid ABBA run order")


def _require_candidate_recipe(
    report: Mapping[str, Any], manifest: ModelManifest, contract: ChallengeContract, role: str
) -> None:
    """Bind the quality videos to the manifest's bounded inference recipe."""
    config = manifest.generation_config
    steps, guidance = config.get("steps"), config.get("guidance_scale", 5.0)
    if (
        type(steps) is not int
        or not 1 <= steps <= contract.max_steps
        or isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
        or not 1 <= guidance <= 10
    ):
        raise ValueError(f"{role} manifest recipe is outside the contract")
    expected_forwards = steps * (2 if guidance > 1 else 1)
    for observation in report["observations"]:
        if (
            observation.get("steps") != steps
            or observation.get("forward_passes") != expected_forwards
        ):
            raise ValueError(f"{role} quality recipe differs from its manifest")


def _require_golden(
    report: Mapping[str, Any], manifest: ModelManifest, contract: ChallengeContract
) -> None:
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("scheduler") != "original_unipc":
        raise ValueError("golden must use the pinned original_unipc quality reference")
    if (
        manifest.role != "pipeline"
        or manifest.base_reference is not None
        or manifest.reference.repo_id != contract.base_model_id
        or manifest.reference.revision != contract.base_model_revision
    ):
        raise ValueError("golden manifest differs from the contract-pinned base")
    guidance_scale = manifest.generation_config.get("guidance_scale", 5.0)
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or not math.isfinite(guidance_scale)
        or guidance_scale < 1
    ):
        raise ValueError("golden guidance scale is invalid")
    expected_forwards = contract.reference_steps * (2 if guidance_scale > 1 else 1)
    for observation in report["observations"]:
        if (
            observation.get("steps") != contract.reference_steps
            or observation.get("forward_passes") != expected_forwards
        ):
            raise ValueError("golden forward passes differ from its pinned CFG recipe")


def _report(
    path: Path,
    *,
    role: str,
    plan: Mapping[str, Any],
    prompts: list[dict[str, Any]],
    slots: list[tuple[str, int]],
    geometry: Mapping[str, int],
) -> _Report:
    report = _read_json(path)
    if report.get("schema") != "forge/panel/v1":
        raise ValueError(f"{role} is not a forge/panel/v1 report")
    if (
        report.get("contract_hash") != plan["contract_hash"]
        or report.get("contract") != plan["contract"]
        or report.get("geometry") != geometry
        or report.get("prompts") != prompts
    ):
        raise ValueError(f"{role} report differs from the frozen plan")
    manifest_hash = _sha256(report.get("manifest_hash"), f"{role} manifest hash")
    try:
        manifest = ModelManifest.from_dict(report.get("manifest"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{role} manifest is invalid") from error
    admission = report.get("admission")
    if admission not in {"public_snapshot", "private_experiment"}:
        raise ValueError(f"{role} admission is invalid")
    if role == "golden" and admission != "public_snapshot":
        raise ValueError("golden quality reference must be a public snapshot")
    if admission == "public_snapshot":
        expected_manifest_hash = manifest.fingerprint()
    else:
        adapter_hash = _sha256(
            report.get("experimental_adapter_sha256"),
            f"{role} experimental adapter hash",
        )
        expected_manifest_hash = hash_canonical(
            "forge/private-experiment/v1",
            manifest.fingerprint(),
            adapter_hash,
        )
    if manifest_hash != expected_manifest_hash:
        raise ValueError(f"{role} manifest fingerprint does not match its report")
    observations = report.get("observations")
    if not isinstance(observations, list) or len(observations) != len(slots):
        raise ValueError(f"{role} observations do not exactly cover the prompt panel")
    contract = ChallengeContract.from_dict(dict(plan["contract"]))
    if role == "golden":
        _require_golden(report, manifest, contract)
    else:
        _require_abba(report, role)
        _require_candidate_recipe(report, manifest, contract, role)
    prompt_text = {prompt["prompt_id"]: prompt["text"] for prompt in prompts}
    videos: list[_Video] = []
    for expected, observation in zip(slots, observations, strict=True):
        if (
            not isinstance(observation, Mapping)
            or (
                observation.get("prompt_id"),
                observation.get("seed"),
            )
            != expected
        ):
            raise ValueError(f"{role} observations are not in prescribed order")
        if observation.get("status") != "OK":
            raise ValueError(f"{role} contains a failed render")
        video_path = _safe_video(path, observation.get("video"))
        expected_hash = _sha256(observation.get("video_sha256"), f"{role} video hash")
        actual_hash = sha256_file(video_path)
        if actual_hash != expected_hash:
            raise ValueError(f"{role} video hash does not match its report")
        loaded = load_video(video_path, geometry)
        if loaded.video_sha256 != expected_hash:
            raise ValueError(f"{role} video changed during validation")
        videos.append(
            _Video(
                prompt_id=expected[0],
                seed=expected[1],
                prompt=prompt_text[expected[0]],
                sha256=expected_hash,
                path=video_path,
            )
        )
    return _Report(manifest_hash=manifest_hash, admission=admission, videos=tuple(videos))


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError("quality output already exists; do not overwrite or blindly resume")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ValueError("quality temporary output already exists")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_golden_report(plan: dict, report_path: Path) -> None:
    """Use the same pinned-reference/media checks for calibration and evaluation."""
    geometry = _geometry(plan)
    prompts, slots = _prompts(plan)
    _report(report_path, role="golden", plan=plan, prompts=prompts, slots=slots, geometry=geometry)


def _persist(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ValueError("quality temporary output already exists")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _comparison(
    reference_role: str,
    reference: _Report,
    candidate: _Report,
) -> dict[str, Any]:
    return {
        "reference_role": reference_role,
        "reference_manifest_hash": reference.manifest_hash,
        "candidate_manifest_hash": candidate.manifest_hash,
        "reference_admission": reference.admission,
        "candidate_admission": candidate.admission,
        "observations": [],
    }


def _raster_checks(reports: Mapping[str, _Report], geometry: Mapping[str, int]) -> dict[str, dict]:
    """Run deterministic decoded-raster preflight once per unique video file hash."""
    checks: dict[str, dict] = {}
    for report in reports.values():
        for video in report.videos:
            if video.sha256 not in checks:
                media = load_video(video.path, geometry)
                checks[video.sha256] = check_frames(media.frames, video.sha256)
    return checks


def score_reports(
    plan: Path,
    golden: Path,
    incumbent: Path,
    candidate: Path,
    out: Path,
    budget: ApiBudget,
    env_file: Path | None,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Judge frozen video pairs with bounded concurrency and durable partial evidence."""

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in [1, 8]")

    plan_path = Path(plan)
    plan_data = _read_json(plan_path)
    geometry = _geometry(plan_data)
    prompts, slots = _prompts(plan_data)
    panel_hash = hash_canonical("forge/panel-prompts/v1", prompts)
    reports = {
        "golden": _report(
            Path(golden),
            role="golden",
            plan=plan_data,
            prompts=prompts,
            slots=slots,
            geometry=geometry,
        ),
        "incumbent": _report(
            Path(incumbent),
            role="incumbent",
            plan=plan_data,
            prompts=prompts,
            slots=slots,
            geometry=geometry,
        ),
        "candidate": _report(
            Path(candidate),
            role="candidate",
            plan=plan_data,
            prompts=prompts,
            slots=slots,
            geometry=geometry,
        ),
    }
    raster_checks = _raster_checks(reports, geometry)
    raster_anomalies = sorted(
        video_sha256
        for video_sha256, check in raster_checks.items()
        if check["status"] == "ANOMALY"
    )
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": plan_data.get("campaign_id", plan_data.get("round_id")),
        "created_unix": time.time(),
        "status": "RAW_UNCALIBRATED",
        "evaluator_id": evaluator_id(),
        "spec": evaluator_spec(),
        "contract_hash": plan_data["contract_hash"],
        "geometry": geometry,
        "panel_hash": panel_hash,
        "raster_integrity_rule_id": RASTER_INTEGRITY_RULE_ID,
        "raster_checks": raster_checks,
        "private_evidence": any(
            report.admission != "public_snapshot" for report in reports.values()
        ),
        "promotion_eligible": False,
        "comparisons": [
            _comparison("golden", reports["golden"], reports["candidate"]),
            _comparison("incumbent", reports["incumbent"], reports["candidate"]),
        ],
    }
    _write_new(Path(out), evidence)
    if raster_anomalies:
        evidence.update(
            status="INCOMPLETE",
            reasons=["RASTER_INTEGRITY_ANOMALY"],
            raster_anomaly_video_sha256=raster_anomalies,
        )
        _persist(Path(out), evidence)
        return evidence
    work: list[tuple[int, int, _Video, _Video]] = []
    for comparison_index, comparison in enumerate(evidence["comparisons"]):
        reference = reports[comparison["reference_role"]]
        for slot_index, (reference_video, candidate_video) in enumerate(
            zip(reference.videos, reports["candidate"].videos, strict=True)
        ):
            if (
                reference_video.prompt_id != candidate_video.prompt_id
                or reference_video.seed != candidate_video.seed
            ):
                raise ValueError("reports have mismatched prescribed video slots")
            work.append((comparison_index, slot_index, reference_video, candidate_video))

    def judge(work_item: tuple[int, int, _Video, _Video]) -> tuple[tuple[int, int], dict[str, Any]]:
        comparison_index, slot_index, reference_video, candidate_video = work_item
        reference_media = load_video(reference_video.path, geometry)
        candidate_media = load_video(candidate_video.path, geometry)
        detail_indices = select_detail_frames(len(reference_media.frames))
        detail_crops = select_detail_crops(reference_media.frames, detail_indices)
        transition_indices = select_transition_frames(len(reference_media.frames))
        result = judge_pair(
            reference_video.prompt,
            make_sheets(
                reference_media.frames,
                geometry["fps"],
                detail_indices=detail_indices,
                detail_crops=detail_crops,
                transition_indices=transition_indices,
            ),
            make_sheets(
                candidate_media.frames,
                geometry["fps"],
                detail_indices=detail_indices,
                detail_crops=detail_crops,
                transition_indices=transition_indices,
            ),
            budget=budget,
            env_file=env_file,
            pair_id=f"{comparison_index}:{reference_video.prompt_id}:{reference_video.seed}",
        )
        return (comparison_index, slot_index), {
            "prompt_id": reference_video.prompt_id,
            "seed": reference_video.seed,
            "judged_unix": time.time(),
            "reference_video_sha256": reference_video.sha256,
            "candidate_video_sha256": candidate_video.sha256,
            "detail_frame_indices": list(detail_indices),
            "detail_crop_regions": list(detail_crops),
            "transition_frame_indices": list(transition_indices),
            **result,
            "diagnostics": {
                "reference": reference_media.diagnostics,
                "candidate": candidate_media.diagnostics,
            },
        }

    completed: dict[tuple[int, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(work), workers):
            futures = [executor.submit(judge, item) for item in work[start : start + workers]]
            try:
                for future in as_completed(futures):
                    key, row = future.result()
                    completed[key] = row
                    for comparison_index, comparison in enumerate(evidence["comparisons"]):
                        comparison["observations"] = [
                            completed[(comparison_index, slot_index)]
                            for slot_index in range(len(slots))
                            if (comparison_index, slot_index) in completed
                        ]
                    _persist(Path(out), evidence)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    evidence["completed_unix"] = time.time()
    _persist(Path(out), evidence)
    return evidence
