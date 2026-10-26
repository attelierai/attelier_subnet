"""Filesystem replay of external-GPU Launch V1 evidence into dashboard v2."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from forge.evaluation.launch_policy import evaluate, freeze
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.hashing import hash_canonical, sha256_file

ROOT = Path(__file__).parents[2]
REVISION = "a" * 40


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def manifest(steps, marker):
    reference = {"backend": "hf", "repo_id": f"fixture/model-{marker}", "revision": REVISION}
    return ModelManifest.from_dict(
        {
            "reference": reference,
            "role": "lora",
            "files": [
                {"path": "adapter_config.json", "sha256": marker * 64, "bytes": 10},
                {"path": "adapter_model.safetensors", "sha256": "b" * 64, "bytes": 20},
            ],
            "license": "mit",
            "generation_config": {"steps": steps, "guidance_scale": 5.0},
            "base_reference": {"backend": "hf", "repo_id": "fixture/base", "revision": REVISION},
        }
    )


def test_launch_dashboard_replays_measured_evidence_and_preserves_competitive_state(tmp_path):
    base, candidate, capacity_only = manifest(50, "1"), manifest(16, "2"), manifest(12, "3")
    manifests = tmp_path / "manifests"
    for name, item in (("base", base), ("candidate", candidate), ("capacity", capacity_only)):
        put(manifests / f"{name}.json", item.to_dict())
    prompts = [
        {
            "prompt_id": f"p{i}",
            "text": f"prompt {i}",
            "seed": i,
            "kind": "motion" if i < 8 else "static",
            "tags": ["motion_required"] if i == 0 else [],
        }
        for i in range(16)
    ]
    geometry = {"width": 32, "height": 32, "frames": 4, "fps": 4}
    hardware, dependencies = {"gpu_name": "fixture"}, {"torch": "fixture"}
    runtime = {"dependencies": dependencies}
    plan = freeze(
        {
            "models": {
                "candidate": candidate.fingerprint(),
                "king": base.fingerprint(),
                "anchor": base.fingerprint(),
            },
            "evaluator": {"evaluator_id": "4" * 64, "spec_hash": "5" * 64},
            "runtime": {
                "runtime_id": hash_canonical("forge/runtime/v1", runtime),
                "hardware_id": hash_canonical("forge/launch-hardware/v1", hardware),
                "dependency_lock_hash": hash_canonical(
                    "forge/launch-dependencies/v1", dependencies
                ),
            },
            "geometry": geometry,
            "prompts": prompts,
            "timing_prompt_ids": [f"p{i}" for i in range(8)],
            "assignments": [
                {
                    "prompt_id": p["prompt_id"],
                    "candidate": candidate.fingerprint(),
                    "king": base.fingerprint(),
                    "anchor": base.fingerprint(),
                }
                for p in prompts
            ],
        }
    )
    campaign, root = tmp_path / "campaign", tmp_path / "campaign" / "candidate"
    put(campaign / "plans" / "candidate.json", plan)
    capacity_plan = json.loads(json.dumps(plan))
    capacity_plan.pop("plan_id")
    capacity_plan["models"]["candidate"] = capacity_only.fingerprint()
    for assignment in capacity_plan["assignments"]:
        assignment["candidate"] = capacity_only.fingerprint()
    capacity_plan = freeze(
        {key: value for key, value in capacity_plan.items() if key != "policy" and key != "schema"}
    )
    put(campaign / "plans" / "capacity.json", capacity_plan)
    results = {}
    reports = {}
    for side, model in (("reference", base), ("candidate", candidate)):
        folder = root / "all" / side
        observations = []
        for prompt in prompts:
            video = folder / f"{prompt['prompt_id']}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes((side + prompt["prompt_id"]).encode())
            digest = sha256_file(video)
            observations.append(
                {
                    "status": "OK",
                    "prompt_id": prompt["prompt_id"],
                    "seed": prompt["seed"],
                    "video": video.name,
                    "video_sha256": digest,
                }
            )
            results[(model.fingerprint(), prompt["prompt_id"], prompt["seed"])] = {
                "video_sha256": digest,
                "scores": {"VQ": 0.0, "MQ": 0.0, "TA": 0.0},
                "diagnostics": {"frozen_fraction": 0.0, "temporal_delta": 0.02},
            }
        report = {
            "schema": "forge/panel/v1",
            "geometry": geometry,
            "manifest_hash": model.fingerprint(),
            "manifest": model.to_dict(),
            "runtime": runtime,
            "runtime_id": plan["runtime"]["runtime_id"],
            "prompts": [
                {"prompt_id": p["prompt_id"], "text": p["text"], "video_seeds": [p["seed"]]}
                for p in prompts
            ],
            "observations": observations,
        }
        report_path = put(folder / "report.json", report)
        reports[model.fingerprint()] = (report_path, sha256_file(report_path))
    rows = []
    for prompt in prompts:
        for reference in ("king", "anchor"):
            rows.append(
                {
                    "type": "quality",
                    "plan_id": plan["plan_id"],
                    "prompt_id": prompt["prompt_id"],
                    "reference": reference,
                    "model_ids": {
                        "reference": base.fingerprint(),
                        "candidate": candidate.fingerprint(),
                    },
                    "evaluator_id": plan["evaluator"]["evaluator_id"],
                    "geometry": geometry,
                    "reference_result": results[
                        (base.fingerprint(), prompt["prompt_id"], prompt["seed"])
                    ],
                    "candidate_result": results[
                        (candidate.fingerprint(), prompt["prompt_id"], prompt["seed"])
                    ],
                }
            )
    for index in range(8):
        rows.append(
            {
                "type": "timing",
                "plan_id": plan["plan_id"],
                "prompt_id": f"p{index}",
                "model_ids": {
                    "reference": base.fingerprint(),
                    "candidate": candidate.fingerprint(),
                },
                "runtime_id": plan["runtime"]["runtime_id"],
                "hardware_id": plan["runtime"]["hardware_id"],
                "geometry": geometry,
                "times": [12, 10, 10, 12],
            }
        )
    put(root / "rows.json", rows)
    put(root / "result.json", evaluate(plan, rows))
    put(
        root / "timing" / "paired-timing.json",
        {
            "frozen_plan_id": plan["plan_id"],
            "runtime_id": plan["runtime"]["runtime_id"],
            "hardware": hardware,
            "runtime": {"dependencies": dependencies},
            "schema": "forge/paired-timing/v1",
            "reference_manifest_hash": base.fingerprint(),
            "candidate_manifest_hash": candidate.fingerprint(),
            "report_video_sha256": {
                role: {
                    f"p{i}:{i}": results[(model.fingerprint(), f"p{i}", i)]["video_sha256"]
                    for i in range(8)
                }
                for role, model in (("reference", base), ("candidate", candidate))
            },
            "measurements": [
                {
                    "prompt_id": f"p{i}",
                    "seed": i,
                    "status": "OK",
                    "side": side,
                    "full_pipeline_seconds": seconds,
                }
                for i in range(8)
                for side, seconds in zip(
                    ("reference", "candidate", "candidate", "reference"),
                    (12, 10, 10, 12),
                    strict=True,
                )
            ],
        },
    )
    for key, result in results.items():
        model_hash, prompt_id, seed = key
        report_path, report_hash = reports[model_hash]
        put(
            root / "scores" / f"{model_hash[:4]}-{prompt_id}.json",
            {
                "schema": "forge/launch-score/v1",
                "plan_id": plan["plan_id"],
                "prompt_id": prompt_id,
                "prompt_text": prompts[seed]["text"],
                "seed": seed,
                "manifest_hash": model_hash,
                "video_sha256": result["video_sha256"],
                "evaluator_id": plan["evaluator"]["evaluator_id"],
                "spec_hash": plan["evaluator"]["spec_hash"],
                "geometry": geometry,
                "source_report_sha256": report_hash,
                "result": result,
            },
        )
    submissions = tmp_path / "submissions"
    put(
        submissions / "candidate.json",
        {
            "schema": "forge/testnet-publication/v1",
            "status": "FINALIZED",
            "network": "test",
            "netuid": 578,
            "manifest": candidate.to_dict(),
            "manifest_hash": candidate.fingerprint(),
            "commitment_block": 1,
            "finalized_block": 2,
            "extrinsic_id": "2-0",
        },
    )
    previous = {
        "schema": "forge/dashboard/v2",
        "updated_at": "2026-01-01T00:00:00Z",
        "network": {"name": "test", "netuid": 578},
        "base_model": {"name": "fixture", "steps": 50},
        "contract": {
            "min_speedup": 1.1,
            "quality_margin": 0.1,
            "axes": ["quality"],
            "consensus_rule": "stake",
        },
        "king": None,
        "crowns": [{"crown_id": "old"}],
        "validators": [],
        "rounds": [
            {
                "round_id": "chain-round",
                "stage": "final",
                "diagnostic": False,
                "close_block": 1,
                "entropy_block": 2,
                "commit_close_block": 1,
                "committee": [],
                "challengers": [],
            }
        ],
        "challengers": [],
        "evaluations": [],
    }
    snapshot, output = tmp_path / "previous.json", tmp_path / "snapshot.json"
    put(snapshot, previous)
    put(
        campaign / "campaign-metadata.json",
        {
            "schema": "forge/launch-dashboard-campaign/v1",
            "campaign_id": "fixture",
            "created_at": "2023-11-14T22:13:20Z",
        },
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.validator.launch_dashboard",
            "--snapshot",
            snapshot,
            "--campaign",
            campaign,
            "--manifests",
            manifests,
            "--submissions",
            submissions,
            "--media-out",
            tmp_path / "media",
            "--out",
            output,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    exported = json.loads(output.read_text())
    assert exported["king"] == previous["king"] and exported["crowns"] == previous["crowns"]
    assert exported["contract"]["launch_v1"]["speed"]["minimum_ratio"] == 1.10
    experiment = next(
        row for row in exported["rounds"] if row.get("origin") == "operator_experiment"
    )
    assert experiment["diagnostic"] is True
    assert (
        experiment["close_block"]
        is experiment["entropy_block"]
        is experiment["commit_close_block"]
        is None
    )
    measured = next(
        row for row in exported["challengers"] if row["manifest_hash"] == candidate.fingerprint()
    )
    assert measured["status"] == "diagnostic" and measured["verdict"] == "DIAGNOSTIC"
    assert measured["hotkey"] == "operator-control" and measured["model"]["url"].startswith(
        "https://huggingface.co/"
    )
    incomplete = next(
        row
        for row in exported["challengers"]
        if row["manifest_hash"] == capacity_only.fingerprint()
    )
    assert incomplete["status"] == "incomplete" and "capacity" in incomplete["reason"]
    incomplete_evaluation = next(
        row for row in exported["evaluations"] if row["challenger_id"] == incomplete["id"]
    )
    assert incomplete_evaluation["open_quality"]["status"] == "INCOMPLETE"
    assert incomplete_evaluation["open_quality"]["speed"] is None
    evaluation = next(
        row for row in exported["evaluations"] if row["challenger_id"] == measured["id"]
    )
    assert evaluation["open_quality"]["evaluator"] == "VideoReward"
    assert len(evaluation["prompts"]) == 16 and all(
        "challenger" in p["videos"] for p in evaluation["prompts"]
    )
    consumer_script = (
        "import {checkV2} from '../attelier_subnet_dashboard/scripts/v2-contract.mjs';"
        "import fs from 'node:fs';const bad=[];"
        "checkV2(JSON.parse(fs.readFileSync(process.argv[1])),(_,ok,x)=>{if(!ok)bad.push(x)});"
        "if(bad.length)throw Error(bad.join('\\n'))"
    )
    consumer_path = ROOT.parent / "attelier_subnet_dashboard" / "scripts" / "v2-contract.mjs"
    if not consumer_path.exists() or shutil.which("node") is None:
        pytest.skip("dashboard consumer is unavailable in this core-only checkout")
    consumer = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            consumer_script,
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert consumer.returncode == 0, consumer.stderr
    # Native timing receipts have per-measurement status, not an aggregate status.
    timing_path = root / "timing" / "paired-timing.json"
    timing = json.loads(timing_path.read_text())
    failed_timing = json.loads(timing_path.read_text())
    failed_timing["measurements"][0]["status"] = "INFRA_FAILURE"
    put(timing_path, failed_timing)
    failed_output = tmp_path / "failed-timing.json"
    failed_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.validator.launch_dashboard",
            "--snapshot",
            snapshot,
            "--campaign",
            campaign,
            "--manifests",
            manifests,
            "--out",
            failed_output,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert failed_run.returncode == 0, failed_run.stderr
    failed_candidate = next(
        row
        for row in json.loads(failed_output.read_text())["challengers"]
        if row["manifest_hash"] == candidate.fingerprint()
    )
    assert failed_candidate["status"] == "incomplete"
    assert "raw timing receipt" in failed_candidate["reason"]
    put(timing_path, timing)
    # Even after rewriting the receipt file, a semantic prompt mismatch is not accepted.
    score_path = next((root / "scores").glob("*.json"))
    altered = json.loads(score_path.read_text())
    altered["prompt_text"] = "different prompt"
    put(score_path, altered)
    tampered = tmp_path / "tampered.json"
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.validator.launch_dashboard",
            "--snapshot",
            snapshot,
            "--campaign",
            campaign,
            "--manifests",
            manifests,
            "--out",
            tampered,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode == 0
    assert (
        next(
            row
            for row in json.loads(tampered.read_text())["challengers"]
            if row["manifest_hash"] == candidate.fingerprint()
        )["status"]
        == "incomplete"
    )
    assert "/workspace/" not in output.read_text()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.validator.launch_dashboard",
            "--snapshot",
            snapshot,
            "--campaign",
            campaign,
            "--manifests",
            manifests,
            "--out",
            output,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert again.returncode != 0 and json.loads(output.read_text()) == exported
