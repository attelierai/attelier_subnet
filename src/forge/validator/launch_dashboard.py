"""Export validator-custodied Launch V1 evidence as a diagnostic dashboard snapshot.

This exporter is intentionally not a promotion path.  It replays the frozen
policy locally, verifies the score-to-render bindings, and always labels the
result as an operator experiment.  Missing capacity is visible as incomplete,
not replaced with estimated scores.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.evaluation.launch_policy import evaluate, freeze_policy
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.dashboard_v2 import _label, _model

SCHEMA = "forge/dashboard/v2"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encoded)


def _manifests(directory: Path) -> dict[str, ModelManifest]:
    found = {}
    for path in sorted(directory.glob("*.json")):
        manifest = ModelManifest.from_dict(_read(path))
        found[manifest.fingerprint()] = manifest
    if not found:
        raise ValueError("no pinned manifests supplied")
    return found


def _submissions(directory: Path | None) -> dict[str, dict[str, Any]]:
    found = {}
    if directory is None or not directory.exists():
        return found
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if not isinstance(value, dict) or value.get("schema") != "forge/testnet-publication/v1":
                continue
            if value.get("status") != "FINALIZED":
                continue
            manifest = ModelManifest.from_dict(value["manifest"])
            if value.get("manifest_hash") != manifest.fingerprint():
                raise ValueError("submission manifest hash changed")
            found[manifest.fingerprint()] = {
                "receipt": {
                    key: value.get(key)
                    for key in (
                        "status",
                        "network",
                        "netuid",
                        "commitment_block",
                        "finalized_block",
                        "extrinsic_id",
                    )
                },
                "hotkey": value.get("hotkey")
                if isinstance(value.get("hotkey"), str)
                else "operator-control",
            }
    return found


def _reports(
    candidate: Path, plan: dict[str, Any]
) -> tuple[dict[tuple[str, str, int], dict], list[dict]]:
    rows, media = {}, []
    for report_path in sorted(candidate.glob("*/**/report.json")):
        report = _read(report_path)
        if report.get("schema") != "forge/panel/v1" or report.get("geometry") != plan["geometry"]:
            raise ValueError("panel receipt schema or geometry changed")
        manifest = report.get("manifest_hash")
        try:
            manifest_object = ModelManifest.from_dict(report.get("manifest"))
        except (TypeError, ValueError) as error:
            raise ValueError("panel receipt manifest is invalid") from error
        if manifest != manifest_object.fingerprint() or manifest not in set(
            plan["models"].values()
        ):
            raise ValueError("panel receipt binds a foreign model")
        if report.get("runtime_id") != plan["runtime"]["runtime_id"] or report.get(
            "runtime_id"
        ) != hash_canonical("forge/runtime/v1", report.get("runtime")):
            raise ValueError("panel receipt runtime is not the frozen runtime")
        prompts = {row.get("prompt_id"): row for row in report.get("prompts", [])}
        frozen_prompts = {row["prompt_id"]: row for row in plan["prompts"]}
        if (
            not prompts
            or len(prompts) != len(report.get("prompts", []))
            or set(prompts) - set(frozen_prompts)
        ):
            raise ValueError("panel receipt has no prompts")
        for prompt_id, prompt in prompts.items():
            frozen = frozen_prompts[prompt_id]
            if prompt.get("text") != frozen["text"] or prompt.get("video_seeds") != [
                frozen["seed"]
            ]:
                raise ValueError("panel receipt prompt text or seed changed")
        for item in report.get("observations", []):
            if item.get("status") != "OK":
                raise ValueError("panel receipt has a failed render")
            prompt = prompts.get(item.get("prompt_id"))
            video_name, video_hash = item.get("video"), item.get("video_sha256")
            if (
                not isinstance(prompt, dict)
                or not isinstance(video_name, str)
                or Path(video_name).name != video_name
            ):
                raise ValueError("panel receipt video binding is invalid")
            video = report_path.parent / video_name
            if video.is_symlink() or not video.is_file() or sha256_file(video) != video_hash:
                raise ValueError("panel video bytes changed")
            if item.get("seed") != frozen_prompts[item["prompt_id"]]["seed"]:
                raise ValueError("panel receipt observation seed changed")
            key = (manifest, item["prompt_id"], item["seed"])
            if key in rows and rows[key]["video_sha256"] != video_hash:
                raise ValueError("conflicting panel video")
            rows[key] = {
                "video_sha256": video_hash,
                "report_sha256": sha256_file(report_path),
                "video": video,
            }
            media.append({"sha256": video_hash, "video": video})
    if not rows:
        raise ValueError("no panel receipts")
    return rows, media


def _measured(candidate: Path, plan: dict[str, Any], media_out: Path | None) -> dict[str, Any]:
    rows_path, result_path = candidate / "rows.json", candidate / "result.json"
    if not rows_path.exists() or not result_path.exists():
        raise ValueError("capacity or execution did not produce frozen rows and result")
    rows, stored = json.loads(rows_path.read_text()), _read(result_path)
    recomputed = evaluate(plan, rows)
    if stored != recomputed or recomputed["status"] == "INCOMPLETE":
        raise ValueError("stored policy result does not replay from frozen rows")
    panel, media = _reports(candidate, plan)
    timing_path = candidate / "timing" / "paired-timing.json"
    if not timing_path.exists():
        raise ValueError("raw ABBA timing receipt is absent")
    timing = _read(timing_path)
    if (
        timing.get("frozen_plan_id") != plan["plan_id"]
        or timing.get("runtime_id") != plan["runtime"]["runtime_id"]
        or hash_canonical("forge/launch-hardware/v1", timing.get("hardware"))
        != plan["runtime"]["hardware_id"]
        or hash_canonical(
            "forge/launch-dependencies/v1", timing.get("runtime", {}).get("dependencies")
        )
        != plan["runtime"]["dependency_lock_hash"]
    ):
        raise ValueError("raw timing receipt binds another frozen plan or runtime")
    grouped = {}
    frozen_seeds = {item["prompt_id"]: item["seed"] for item in plan["prompts"]}
    links = timing.get("report_video_sha256")
    if (
        timing.get("schema") != "forge/paired-timing/v1"
        or timing.get("reference_manifest_hash") != plan["models"]["king"]
        or timing.get("candidate_manifest_hash") != plan["models"]["candidate"]
        or not isinstance(links, dict)
        or set(links) != {"reference", "candidate"}
    ):
        raise ValueError("raw timing receipt bindings are incomplete")
    for item in timing.get("measurements", []):
        prompt_id = item.get("prompt_id")
        if item.get("status") != "OK" or item.get("seed") != frozen_seeds.get(prompt_id):
            raise ValueError("raw timing receipt has a foreign measurement")
        grouped.setdefault(item.get("prompt_id"), []).append(item)
    for row in rows:
        if row.get("type") != "timing":
            continue
        actual = grouped.get(row["prompt_id"], [])
        if [item.get("side") for item in actual] != [
            "reference",
            "candidate",
            "candidate",
            "reference",
        ]:
            raise ValueError("raw timing receipt is not ABBA")
        if [item.get("full_pipeline_seconds") for item in actual] != row["times"]:
            raise ValueError("policy timing row differs from raw timing receipt")
        pair_key = f"{row['prompt_id']}:{frozen_seeds[row['prompt_id']]}"
        for role, model in (
            ("reference", plan["models"]["king"]),
            ("candidate", plan["models"]["candidate"]),
        ):
            if links[role].get(pair_key) != panel.get(
                (model, row["prompt_id"], frozen_seeds[row["prompt_id"]]), {}
            ).get("video_sha256"):
                raise ValueError("raw timing receipt is not linked to panel video bytes")
    scores = {}
    for path in sorted((candidate / "scores").glob("*.json")):
        if path.name == "evaluator-runtime.json":
            continue
        score = _read(path)
        if (
            score.get("schema") != "forge/launch-score/v1"
            or score.get("plan_id") != plan["plan_id"]
            or score.get("evaluator_id") != plan["evaluator"]["evaluator_id"]
            or score.get("spec_hash") != plan["evaluator"]["spec_hash"]
            or score.get("geometry") != plan["geometry"]
        ):
            raise ValueError("score receipt binds another plan")
        key = (score.get("manifest_hash"), score.get("prompt_id"), score.get("seed"))
        prompt = next(
            (item for item in plan["prompts"] if item["prompt_id"] == score.get("prompt_id")), None
        )
        if (
            prompt is None
            or score.get("seed") != prompt["seed"]
            or score.get("prompt_text") != prompt["text"]
        ):
            raise ValueError("score receipt prompt text or seed changed")
        source = panel.get(key)
        result = score.get("result")
        if (
            source is None
            or score.get("source_report_sha256") != source["report_sha256"]
            or score.get("video_sha256") != source["video_sha256"]
            or not isinstance(result, dict)
            or result.get("video_sha256") != source["video_sha256"]
        ):
            raise ValueError("score receipt is not bound to verified panel bytes")
        scores[key] = result
    for row in rows:
        if row.get("type") != "quality":
            continue
        for side in ("reference", "candidate"):
            model = row["model_ids"][side]
            result = row[side + "_result"]
            expected = scores.get(
                (
                    model,
                    row["prompt_id"],
                    next(
                        prompt["seed"]
                        for prompt in plan["prompts"]
                        if prompt["prompt_id"] == row["prompt_id"]
                    ),
                )
            )
            if expected is None or any(
                result[key] != expected[key] for key in ("video_sha256", "scores", "diagnostics")
            ):
                raise ValueError("policy quality row is not backed by a score receipt")
    exported = []
    if media_out is not None:
        media_out.mkdir(parents=True, exist_ok=True)
        for item in {entry["sha256"]: entry for entry in media}.values():
            target = media_out / f"{item['sha256']}.mp4"
            if target.exists():
                if sha256_file(target) != item["sha256"]:
                    raise ValueError("public media target has different bytes")
            else:
                shutil.copyfile(item["video"], target)
            exported.append({"sha256": item["sha256"], "src": f"/media/{item['sha256']}.mp4"})
    videos = {}
    if media_out is not None:
        by_hash = {item["sha256"]: item for item in exported}
        for prompt in plan["prompts"]:
            seed = prompt["seed"]
            videos[prompt["prompt_id"]] = {
                role: by_hash[panel[(model, prompt["prompt_id"], seed)]["video_sha256"]]
                for role, model in (
                    ("wan50", plan["models"]["anchor"]),
                    ("king", plan["models"]["king"]),
                    ("challenger", plan["models"]["candidate"]),
                )
            }
    return {"policy": recomputed, "videos": videos}


def export(
    snapshot: dict[str, Any],
    campaign: Path,
    manifests: Path,
    submissions: Path | None = None,
    media_out: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if snapshot.get("schema") != SCHEMA:
        raise ValueError("previous snapshot must be forge/dashboard/v2")
    result = json.loads(json.dumps(snapshot))
    metadata = _read(campaign / "campaign-metadata.json")
    if (
        metadata.get("schema") != "forge/launch-dashboard-campaign/v1"
        or not isinstance(metadata.get("campaign_id"), str)
        or not isinstance(metadata.get("created_at"), str)
        or datetime.fromisoformat(metadata["created_at"].replace("Z", "+00:00")).tzinfo is None
    ):
        raise ValueError("campaign metadata needs a fixed id and ISO created_at")
    created_at, campaign_id = metadata["created_at"], metadata["campaign_id"]
    result["updated_at"] = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result.setdefault("contract", {})["launch_v1"] = freeze_policy()
    known, published = _manifests(manifests), _submissions(submissions)
    plans = {path.stem: _read(path) for path in sorted((campaign / "plans").glob("*.json"))}
    for plan in plans.values():
        candidate = plan["models"]["candidate"]
        if candidate not in known:
            raise ValueError("campaign plan candidate is absent from manifests")
    candidates = set(plans) | {path.stem for path in manifests.glob("*.json")}
    result["challengers"] = [
        row
        for row in result.setdefault("challengers", [])
        if not str(row.get("id", "")).startswith(f"launch-v1-{campaign_id}-")
    ]
    result["evaluations"] = [
        row
        for row in result.setdefault("evaluations", [])
        if not str(row.get("challenger_id", "")).startswith(f"launch-v1-{campaign_id}-")
    ]
    result["rounds"] = [
        row
        for row in result.setdefault("rounds", [])
        if row.get("round_id") != f"operator-experiment-{campaign_id}"
    ]
    rounds = result["rounds"]
    for name in sorted(candidates):
        manifest = ModelManifest.from_dict(_read(manifests / f"{name}.json"))
        fingerprint = manifest.fingerprint()
        if manifest.role == "pipeline":
            continue
        plan = next(
            (item for item in plans.values() if item["models"]["candidate"] == fingerprint), None
        )
        round_id = f"operator-experiment-{campaign_id}"
        if not any(row.get("round_id") == round_id for row in rounds):
            rounds.append(
                {
                    "round_id": round_id,
                    "stage": "final",
                    "origin": "operator_experiment",
                    "campaign_id": campaign_id,
                    "diagnostic": True,
                    "close_block": None,
                    "entropy_block": None,
                    "commit_close_block": None,
                    "created_at": created_at,
                    "committee": [],
                    "challengers": [],
                }
            )
        entry = next(row for row in rounds if row["round_id"] == round_id)
        challenger = {
            "id": f"launch-v1-{campaign_id}-{fingerprint[:12]}",
            "round_id": round_id,
            "label": _label(manifest),
            "hotkey": published.get(fingerprint, {}).get("hotkey", "operator-control"),
            "model": _model(manifest),
            "manifest_hash": fingerprint,
            "commit_block": published.get(fingerprint, {})
            .get("receipt", {})
            .get("commitment_block"),
            "verdict": "DIAGNOSTIC",
        }
        if fingerprint in published:
            challenger["submission"] = published[fingerprint]["receipt"]
        if plan is None:
            challenger.update(
                {"status": "incomplete", "reason": "no frozen GPU plan or capacity evidence"}
            )
            measured = None
        else:
            folder = campaign / name
            try:
                measured = _measured(folder, plan, media_out)
            except ValueError as error:
                challenger.update({"status": "incomplete", "reason": str(error)})
                measured = None
            else:
                challenger.update({"status": "diagnostic"})
        entry["challengers"].append(challenger["id"])
        result["challengers"].append(challenger)
        policy = measured["policy"] if measured else None
        open_quality = (
            None
            if plan is None
            else {
                "evaluator": "VideoReward",
                "policy_id": plan["policy"]["policy_id"],
                "evaluator_id": plan["evaluator"]["evaluator_id"],
                "units": "normalized_reward",
                "provisional": True,
                "status": policy["status"] if policy else "INCOMPLETE",
                "quality": policy["quality"] if policy else None,
                "speed": policy["speed"] if policy else None,
                "temporal_violations": policy["temporal_violations"] if policy else [],
            }
        )
        result["evaluations"].append(
            {
                "challenger_id": challenger["id"],
                "plan_hash": plan["plan_id"] if plan else None,
                "verdict": {
                    "status": "DIAGNOSTIC",
                    "speed_ratio": policy["speed"]["ratio"] if policy else None,
                    "speed_range": None,
                    "quality_vs_king": None,
                    "quality_vs_wan50": None,
                },
                "consensus": {
                    "rule": "operator diagnostic; no stake consensus",
                    "favorable_stake": 0,
                    "total_stake": 0,
                    "reports": [],
                },
                "open_quality": open_quality,
                "prompts": []
                if measured is None
                else [
                    {
                        "id": prompt["prompt_id"],
                        "text": prompt["text"],
                        "speed_ratio": None,
                        "quality_vs_king": None,
                        "quality_vs_wan50": None,
                        "axes": None,
                        "videos": measured["videos"].get(prompt["prompt_id"]),
                    }
                    for prompt in plan["prompts"]
                ],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--submissions", type=Path)
    parser.add_argument("--media-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _write_new(
        args.out,
        export(
            _read(args.snapshot), args.campaign, args.manifests, args.submissions, args.media_out
        ),
    )


if __name__ == "__main__":
    main()
