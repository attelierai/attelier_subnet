"""Real signed distributed round artifacts become the public v2 dashboard snapshot.

Two sr25519 wallets sign partial reports through the CLI, the CLI aggregates them,
small real MP4 renders carry worker receipts, and `dashboard-export-v2` builds the
snapshot that must match the frontend fixtures' shape and the privacy rules.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

bt = pytest.importorskip("bittensor")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

from forge.evaluation.distributed_plan import (  # noqa: E402
    DistributedEvaluationPlan,
    allocate_work,
    assignments_for,
)
from forge.evaluation.mvp_policy import QUALITY_AXES  # noqa: E402
from forge.evaluation.synthetic_prompts import synthetic_panel  # noqa: E402
from forge.models.artifact_manifest import ModelManifest  # noqa: E402
from forge.protocol.distributed_batch import DiagnosticBatch  # noqa: E402
from forge.protocol.hashing import sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Copies of the attelier_subnet_dashboard fixtures/v2 the page renders; refresh them
# when that repository changes its fixtures.
FIXTURES = ROOT / "tests" / "fixtures" / "dashboard_v2"
REPO, REVISION = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"
GEOMETRY = {"height": 32, "width": 32, "frames": 2, "fps": 4}


def _cli(*args):
    completed = subprocess.run(
        [sys.executable, "-m", "forge.cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    return completed


def _manifest(role: str, steps: int) -> ModelManifest:
    reference = {"backend": "hf", "repo_id": REPO, "revision": REVISION}
    files = [{"path": "transformer/config.json", "sha256": "1" * 64, "bytes": 400}]
    if role == "full_transformer":
        files.append(
            {
                "path": "transformer/diffusion_pytorch_model.safetensors",
                "sha256": f"{steps:064x}",
                "bytes": 1000 + steps,
            }
        )
    return ModelManifest.from_dict(
        {
            "reference": reference,
            "role": role,
            "files": files,
            "license": "apache-2.0",
            "generation_config": {"steps": steps, "guidance_scale": 5.0},
            "base_reference": reference if role != "pipeline" else None,
        }
    )


def _shape(value, path="$", out=None):
    """path -> (set of JSON types, list of observed dict key sets)."""
    out = {} if out is None else out
    kind = (
        "null"
        if value is None
        else "number"
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else type(value).__name__
    )
    types, keysets = out.setdefault(path, (set(), []))
    types.add(kind)
    if isinstance(value, dict):
        keysets.append(frozenset(value))
        for key, item in value.items():
            _shape(item, f"{path}.{key}", out)
    elif isinstance(value, list):
        for item in value:
            _shape(item, f"{path}[]", out)
    return out


def test_signed_distributed_round_exports_public_v2_snapshot(tmp_path):
    wallets = []
    for name in ("validator-a", "validator-b"):
        wallet = bt.Wallet(name=name, hotkey="hot", path=str(tmp_path / "wallets"))
        wallet.create_new_hotkey(suppress=True, use_password=False, overwrite=False)
        wallets.append(wallet)
    hot_a, hot_b = (w.hotkey.ss58_address for w in wallets)
    miner_fast, miner_turbo = "5MinerFastHotkeyForDashboardIntegration0000000001", "5MinerTurbo2"
    committee = {hot_a: 104_289_154_491_905, hot_b: 1_918_128_626}
    wan50, king = _manifest("pipeline", 50), _manifest("full_transformer", 30)
    fast, turbo = _manifest("full_transformer", 40), _manifest("full_transformer", 12)
    context = {
        "network": "test",
        "netuid": 578,
        "close_block": 100,
        "close_block_hash": "0x" + "a" * 64,
        "entropy_block": 102,
        "entropy_block_hash": "0x" + "b" * 64,
    }
    genesis = DistributedEvaluationPlan(
        round_id="test-578-b100-e102",
        stage="diagnostic",
        entropy="0x" + "b" * 64,
        policy_id="diagnostic-no-calibration-no-rewards-v1",
        candidate_manifest_hash=fast.fingerprint(),
        king_manifest_hash=wan50.fingerprint(),
        wan50_manifest_hash=wan50.fingerprint(),
        committee=committee,
        prompts=tuple(synthetic_panel("e", round_id="r", phase="diagnostic", per_cell=1)[:6]),
        timing_prompt_ids=(),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
        audit_work_count=2,
        chain_context={**context, "candidate_hotkey": miner_fast, "candidate_commit_block": 90},
    )
    # Same panel and committee; validator B never reports this challenger.
    slow = replace(
        genesis,
        candidate_manifest_hash=turbo.fingerprint(),
        chain_context={**context, "candidate_hotkey": miner_turbo, "candidate_commit_block": 91},
    )
    # A later round shape with a non-baseline king (a batch freezes one king per round).
    crowned = replace(slow, king_manifest_hash=king.fingerprint())
    plans = {"genesis": genesis, "slow": slow, "crowned": crowned}
    for label, plan in plans.items():
        (tmp_path / f"plan-{label}.json").write_text(json.dumps(plan.to_dict()))
    publications = []
    for label, hotkey, manifest in (("fast", miner_fast, fast), ("turbo", miner_turbo, turbo)):
        path = tmp_path / f"model-{label}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "forge/testnet-publication/v1",
                    "hotkey": hotkey,
                    "manifest": manifest.to_dict(),
                    "manifest_hash": manifest.fingerprint(),
                }
            )
        )
        publications.append(path)

    # Worker renders: one real MP4 + receipt per (validator, prompt, model), on each GPU.
    gpu = {hot_a: "NVIDIA RTX 6000 Ada Generation", hot_b: "NVIDIA L40S"}
    seconds = {wan50.fingerprint(): 260.0, king.fingerprint(): 200.0}
    seconds |= {fast.fingerprint(): 208.0, turbo.fingerprint(): 100.0}
    receipt_dirs, hashes, shade = {}, {}, 0
    prompts = {row["prompt_id"]: row for row in genesis.prompts}
    for wallet in wallets:
        hotkey = wallet.hotkey.ss58_address
        folder = tmp_path / f"renders-{wallet.name}"
        folder.mkdir()
        receipt_dirs[hotkey] = folder
        for item in assignments_for(genesis, hotkey):
            prompt = prompts[item.prompt_id]
            for model in (wan50, king, fast, turbo):
                shade += 1
                job_id = f"{model.generation_config['steps']}-{item.prompt_id}"
                video = folder / f"{job_id}.mp4"
                writer = imageio_ffmpeg.write_frames(
                    str(video), (32, 32), fps=4, macro_block_size=1
                )
                writer.send(None)
                for _ in range(GEOMETRY["frames"]):
                    writer.send(bytes([shade % 256]) * (32 * 32 * 3))
                writer.close()
                hashes[(hotkey, item.prompt_id, model.fingerprint())] = sha256_file(video)
                elapsed = seconds[model.fingerprint()] * (1 if hotkey == hot_a else 1.5)
                (folder / f"{job_id}.json").write_text(
                    json.dumps(
                        {
                            "schema": "forge/distributed-worker-receipt/v1",
                            "diagnostic_only": True,
                            "job": {
                                "job_id": job_id,
                                "kind": "candidate",
                                "prompt_id": item.prompt_id,
                                "prompt": prompt["text"],
                                "seed": prompt["seed"],
                                "model_manifest_hash": model.fingerprint(),
                            },
                            "validator_hotkey": hotkey,
                            "round_id": genesis.round_id,
                            "geometry": GEOMETRY,
                            "status": "OK",
                            "video": video.name,
                            "video_sha256": hashes[(hotkey, item.prompt_id, model.fingerprint())],
                            # Canonical tagged float, as written by hashing-aware tools.
                            "full_pipeline_seconds": {"__forge_float__": repr(elapsed)},
                            "runtime": {"gpu_name": gpu[hotkey]},
                        }
                    )
                )
        (folder / "run.json").write_text(json.dumps({"schema": "forge/distributed-worker-run/v1"}))

    def rows(plan, hotkey, reference, delta):
        out = []
        for item in assignments_for(plan, hotkey):
            prompt = prompts[item.prompt_id]
            out.append(
                {
                    **{
                        k: prompt[k]
                        for k in (
                            "prompt_id",
                            "seed",
                            "subject_class",
                            "camera_type",
                            "style_family",
                        )
                    },
                    "evidence_kind": "diagnostic_quality",
                    "orders": [
                        {"reference_first": o, "axes": dict.fromkeys(QUALITY_AXES, delta)}
                        for o in (True, False)
                    ],
                    "reference_video_sha256": hashes[(hotkey, item.prompt_id, reference)],
                    "candidate_video_sha256": hashes[
                        (hotkey, item.prompt_id, plan.candidate_manifest_hash)
                    ],
                }
            )
        return out

    reports, resolved = [], []
    for label, plan, signers in (
        ("genesis", genesis, wallets),
        ("slow", slow, wallets[:1]),
        ("crowned", crowned, wallets[:1]),
    ):
        received = []
        for wallet in signers:
            hotkey = wallet.hotkey.ss58_address
            partial = tmp_path / f"partial-{label}-{wallet.name}.json"
            partial.write_text(
                json.dumps(
                    {
                        "quality_vs_king": rows(plan, hotkey, plan.king_manifest_hash, -0.04),
                        "quality_vs_wan50": rows(plan, hotkey, plan.wan50_manifest_hash, -0.02),
                        "timing": [],
                    }
                )
            )
            report = tmp_path / f"report-{label}-{wallet.name}.json"
            signed = _cli(
                "distributed-report",
                "--plan",
                tmp_path / f"plan-{label}.json",
                "--partial",
                partial,
                "--wallet",
                wallet.name,
                "--hotkey",
                "hot",
                "--wallet-path",
                tmp_path / "wallets",
                "--out",
                report,
            )
            assert signed.returncode == 0, signed.stderr
            received.append(report)
        aggregate = tmp_path / f"aggregate-{label}.json"
        args = [arg for path in received for arg in ("--report", path)]
        done = _cli(
            "distributed-aggregate",
            "--plan",
            tmp_path / f"plan-{label}.json",
            *args,
            "--out",
            aggregate,
        )
        assert done.returncode == 0, done.stderr
        reports += received
        resolved.append(aggregate)

    batch = tmp_path / "batch.json"
    frozen = _cli(
        "distributed-batch-plan",
        *(a for p in ("genesis", "slow") for a in ("--plan", tmp_path / f"plan-{p}.json")),
        "--commit-deadline-block",
        150,
        "--reveal-deadline-block",
        160,
        "--out",
        batch,
    )
    assert frozen.returncode == 0, frozen.stderr
    batch_hash = DiagnosticBatch.from_dict(json.loads(batch.read_text())).batch_hash()
    anchors = []
    for block, hotkey in ((140, hot_a), (141, hot_b)):
        path = tmp_path / f"anchor-{block}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "forge/diagnostic-batch-anchor/v1",
                    "hotkey": hotkey,
                    "batch_hash": batch_hash,
                    "status": "FINALIZED",
                    "commitment_block": block,
                }
            )
        )
        anchors.append(path)
    close = tmp_path / "close.json"
    close.write_text(json.dumps({"close_block": 100, "entropy_block": 102, "committee": committee}))
    validators = tmp_path / "validators.json"
    validators.write_text(
        json.dumps(
            {
                "schema": "forge/validator-snapshot/v1",
                "validators": [
                    {"uid": 0, "hotkey": hot_a, "stake_rao": committee[hot_a], "active": True},
                    {"uid": 3, "hotkey": hot_b, "stake_rao": committee[hot_b], "active": True},
                ],
            }
        )
    )
    uids = tmp_path / "chain.json"
    uids.write_text(json.dumps({"uids": {"6": miner_fast, "7": miner_turbo}}))
    media_out, out = tmp_path / "public" / "media", tmp_path / "public" / "dashboard.json"

    def export(labels, target, *extra):
        many = [
            *(a for p in labels for a in ("--plan", tmp_path / f"plan-{p}.json")),
            *(a for p in publications for a in ("--model-publication", p)),
            *(a for p in resolved for a in ("--resolved", p)),
            *(a for p in reports for a in ("--report", p)),
            *(a for p in anchors for a in ("--anchor", p)),
            *(a for p in receipt_dirs.values() for a in ("--worker-receipts", p)),
        ]
        return _cli(
            "dashboard-export-v2",
            *many,
            "--close",
            close,
            "--validators",
            validators,
            "--contract",
            ROOT / "configs" / "wan-pilot.json",
            *extra,
            "--uids",
            uids,
            "--media-out",
            media_out,
            "--out",
            target,
        )

    result = export(("genesis", "slow"), out, "--batch", batch)
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    snapshot = json.loads(text)

    # Shape: every path, type and object key set occurs in the frontend fixtures.
    expected: dict = {}
    for name in ("genesis", "promoted", "incomplete"):
        fixture = json.loads((FIXTURES / f"{name}.json").read_text())
        fixture.pop("_fixture")
        _shape(fixture, out=expected)
    # The additional download metadata is verified against the actual admitted
    # manifest. Historical fixtures still exercise the legacy projection.
    legacy_snapshot = json.loads(text)
    manifests = {m.fingerprint(): m for m in (fast, turbo)}
    for challenger in legacy_snapshot["challengers"]:
        model = challenger["model"]
        manifest = manifests[model["manifest_hash"]]
        assert model["backend"] == manifest.reference.backend
        assert model["license"] == manifest.license
        assert model["role"] == manifest.role
        assert model["files"] == [
            {
                "path": f.path, "sha256": f.sha256, "size_bytes": f.bytes,
                "url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/{f.path}",
            }
            for f in manifest.files
        ]
        assert model["base"]["revision"] == REVISION
        for field in ("backend", "manifest_hash", "license", "role", "files", "base"):
            model.pop(field)
    for path, (types, keysets) in _shape(legacy_snapshot).items():
        assert path in expected, path
        assert types <= expected[path][0], (path, types)
        assert all(keys in expected[path][1] for keys in keysets), path
    assert list(snapshot)[:5] == ["schema", "updated_at", "network", "base_model", "contract"]
    assert snapshot["base_model"] == {
        "name": "Wan2.1 T2V 1.3B",
        "repo": REPO,
        "revision": REVISION,
        "url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "steps": 50,
    }
    assert {c["model"]["url"] for c in snapshot["challengers"]} == {snapshot["base_model"]["url"]}
    assert snapshot["king"] is None and snapshot["crowns"] == []
    assert [c["id"][:5] for c in snapshot["challengers"]] == ["uid6-", "uid7-"]
    assert snapshot["rounds"][0]["commit_close_block"] == 150
    assert snapshot["rounds"][0]["stage"] == "consensus"  # validator B missed one report

    first, second = snapshot["evaluations"]
    assert first["verdict"]["status"] == second["verdict"]["status"] == "DIAGNOSTIC"
    assert [r["status"] for r in first["consensus"]["reports"]] == ["valid", "valid"]
    assert {r["anchor_block"] for r in first["consensus"]["reports"]} == {140, 141}
    for prompt in first["prompts"]:
        # King is the Wan2.1 50-step baseline: no king video or king quality.
        assert set(prompt["videos"]) == {"wan50", "challenger"}
        assert prompt["quality_vs_king"] is None and prompt["quality_vs_wan50"] == -0.02
        assert prompt["speed_ratio"] == pytest.approx(1.25)  # 260 s / 208 s on each GPU
        assert prompt["text"] == prompts[prompt["id"]]["text"]
        for clip in prompt["videos"].values():
            assert (media_out / Path(clip["src"]).name).is_file()
            assert (media_out / Path(clip["poster"]).name).is_file()
            assert sha256_file(media_out / Path(clip["src"]).name) == clip["sha256"]
        # Absolute seconds stay with the GPU that rendered them; one machine per prompt.
        assert prompt["videos"]["wan50"]["gpu"] == prompt["videos"]["challenger"]["gpu"]

    missing = {r["validator"]: r for r in second["consensus"]["reports"]}[hot_b]
    assert missing == {**missing, "status": "missing", "report_sha256": None, "anchor_block": None}
    only_b = {i.prompt_id for i in allocate_work(slow)} - {
        i.prompt_id for i in assignments_for(slow, hot_a)
    }
    assert only_b
    for prompt in second["prompts"]:
        if prompt["id"] in only_b:
            assert prompt["speed_ratio"] is prompt["videos"] is prompt["axes"] is None
        else:
            assert set(prompt["videos"]) == {"wan50", "challenger"}
            assert prompt["speed_ratio"] == pytest.approx(2.6)  # 260 s / 100 s, GPU A

    # A plan's comparator is not proof of coronation. Fail closed until explicit
    # finalized state is supplied (synthetic state here tests projection only).
    crowned_out = tmp_path / "public" / "crowned.json"
    refused_king = export(("crowned",), crowned_out)
    assert refused_king.returncode != 0 and "finalized king state" in refused_king.stderr
    assert not crowned_out.exists()
    from forge.protocol.consensus_types import FinalizedCompetitionState
    from forge.protocol.rewards import Crown

    state = replace(
        FinalizedCompetitionState.genesis("dashboard-fixture", wan50.fingerprint()),
        current_king_fingerprint=king.fingerprint(),
        crown_history=(Crown("a" * 64, "fixture-king", king.fingerprint(), 100),),
    )
    state_path = tmp_path / "finalized-king-state.json"
    state_path.write_text(json.dumps(state.to_dict()))
    assert export(("crowned",), crowned_out, "--king-state", state_path).returncode == 0
    crowned_snapshot = json.loads(crowned_out.read_text())
    assert crowned_snapshot["king"]["manifest_hash"] == king.fingerprint()
    assert crowned_snapshot["king"]["crown_id"] == "a" * 64
    (evaluation,) = crowned_snapshot["evaluations"]
    for prompt in evaluation["prompts"]:
        if prompt["id"] not in only_b:
            assert set(prompt["videos"]) == {"wan50", "king", "challenger"}
            assert prompt["quality_vs_king"] == -0.04
            assert prompt["speed_ratio"] == pytest.approx(2.0)  # king 200 s / turbo 100 s
            assert prompt["videos"]["king"]["seconds"] == 200.0

    # Privacy: no local paths, signatures, coldkeys, salts or peer URLs; the only
    # remote links are the pinned public model artifacts.
    for forbidden in (str(tmp_path), "signature", "coldkey", "salt", "http://", "/workspace"):
        assert forbidden not in text
    assert text.count("https://") == text.count("https://huggingface.co/")

    # Media is content-addressed; different bytes under a published name are refused.
    victim = media_out / Path(first["prompts"][0]["videos"]["wan50"]["src"]).name
    victim.write_bytes(b"tampered")
    refused = export(("genesis", "slow"), out, "--batch", batch)
    assert refused.returncode != 0 and "different bytes" in refused.stderr
    assert json.loads(out.read_text()) == snapshot  # previous snapshot left intact
