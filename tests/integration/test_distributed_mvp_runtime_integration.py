"""No-GPU integration for the assigned competitive runtime file/signature boundary."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from forge.evaluation.distributed import verify_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.fusion import evaluator_implementation_hash
from forge.evaluation.mvp_evidence import freeze_evaluation_plan
from forge.evaluation.mvp_policy import freeze_policy
from forge.evaluation.vbench_runner import run_vbench_isolated
from forge.evaluation.vision_judge import MODEL, judge_pair
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.contract import DEFAULT_AXES, ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.sim import HmacSigner, HmacVerifier
from forge.validator.budget import BudgetLedger
from forge.validator.distributed_mvp_runtime import (
    AssignedRuntime,
    VBenchConfig,
    execute_assigned_confirmation,
    runtime_identity,
)
from forge.validator.mvp_runtime import MVPWork


def _sha(value: int) -> str:
    return f"{value:064x}"


def _prompts() -> list[dict]:
    return [
        {
            "prompt_id": f"{subject}-{camera}-{style}-{number}",
            "text": f"{subject} {camera} {style} {number}",
            "seed": number,
            "subject_class": subject,
            "camera_type": camera,
            "style_family": style,
        }
        for subject in ("human", "animal", "object", "scenery")
        for camera in ("static", "moving")
        for style in ("natural", "stylized")
        for number in range(8)
    ]


def _model(ref: ArtifactReference, role: str, steps: int, base: ArtifactReference | None = None):
    return ModelManifest(
        ref,
        role,
        (
            ManifestFile(
                "adapter.safetensors" if role == "lora" else "model_index.json", _sha(7), 1
            ),
        ),
        "apache-2.0",
        {"steps": steps, "guidance_scale": 5.0},
        base,
    )


def test_assigned_runtime_uses_exact_frozen_prompts_and_signs_file_bound_partial(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FORGE_GPU_LOCK_DIR", str(tmp_path / "gpu-lock"))
    contract = ChallengeContract(max_steps=50, reference_steps=50, width=32, height=32, frames=5)
    base_ref = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    base = _model(base_ref, "pipeline", 50)
    candidate = _model(ArtifactReference("hf", "public/candidate", "a" * 40), "lora", 16, base_ref)
    incumbent = _model(ArtifactReference("hf", "public/king", "b" * 40), "lora", 16, base_ref)
    weights = {
        name: _sha(index + 50)
        for index, name in enumerate(
            (
                "subject_consistency",
                "background_consistency",
                "imaging_quality",
                "motion_smoothness",
                "dynamic_degree",
            )
        )
    }
    provenance = {
        "source_implementation_hash": evaluator_implementation_hash(),
        "vbench_model_hashes": hash_canonical("forge/vbench-weight-lock/v1", weights),
        "luna_evaluator_id": MODEL,
    }
    policy = freeze_policy(
        state="frozen", threshold_derivation_hash=_sha(9), evaluator_provenance=provenance
    )
    prompts = _prompts()
    mvp_plan = freeze_evaluation_plan(
        stage="confirmation",
        prompts=prompts,
        policy=policy,
        candidate_manifest_hash=candidate.fingerprint(),
        incumbent_manifest_hash=incumbent.fingerprint(),
        wan50_manifest_hash=base.fingerprint(),
    )
    timing_ids = tuple(row["prompt_id"] for row in prompts[:32])
    plan = DistributedEvaluationPlan(
        round_id="round",
        stage="confirmation",
        entropy="entropy",
        policy_id=policy.policy_id,
        candidate_manifest_hash=candidate.fingerprint(),
        king_manifest_hash=incumbent.fingerprint(),
        wan50_manifest_hash=base.fingerprint(),
        committee={"a": 1, "b": 1},
        prompts=tuple(prompts),
        timing_prompt_ids=timing_ids,
        quality_tolerance=0.1,
        timing_log_ratio_tolerance=0.1,
        audit_calibration_hash=_sha(10),
        audit_work_count=2,
        chain_context={"mvp_plan_id": mvp_plan["plan_id"]},
    )
    work = MVPWork(
        plan_id=mvp_plan["plan_id"],
        stage="confirmation",
        policy_id=policy.policy_id,
        candidate_manifest_hash=candidate.fingerprint(),
        incumbent_manifest_hash=incumbent.fingerprint(),
        wan50_manifest_hash=base.fingerprint(),
        contract_hash=contract.contract_hash(),
        quality_prompts=tuple(
            {"prompt_id": p["prompt_id"], "text": p["text"], "video_seeds": [p["seed"]]}
            for p in prompts
        ),
        timing_prompts=tuple(
            {"prompt_id": p["prompt_id"], "text": p["text"], "video_seeds": [p["seed"]]}
            for p in prompts[:32]
        ),
        binding_hash=_sha(11),
    )
    ledger = BudgetLedger(tmp_path / "budget.sqlite")
    operation_key = f"distributed-mvp:{plan.plan_hash()}:a"
    reservation = ledger.reserve(1, operation_key, 2.0)
    job = AssignedRuntime(
        plan,
        mvp_plan,
        work,
        provenance,
        contract,
        base,
        tmp_path,
        candidate,
        tmp_path,
        incumbent,
        tmp_path,
        base,
        tmp_path,
        _sha(12),
        VBenchConfig(
            tmp_path, tmp_path, Path(sys.executable), weights, "d" * 40, "sha256:" + "a" * 64
        ),
        __import__("forge.evaluation.vision_judge", fromlist=["ApiBudget"]).ApiBudget(
            tmp_path / "luna.json", 20.0
        ),
        None,
        ledger,
        reservation.id,
        1,
        operation_key,
        1.0,
    )

    # External render/VBench/Luna services are replaced, but bytes, journals,
    # fusion, allocation, durable reservation and signature remain real.
    def panel_runner(*, prompts, out, **_):
        out.mkdir(parents=True)
        observations = []
        for prompt in prompts:
            video = out / f"{prompt['prompt_id']}.mp4"
            video.write_bytes(f"{out.name}:{prompt['prompt_id']}".encode())
            observations.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "seed": prompt["video_seeds"][0],
                    "status": "OK",
                    "video": video.name,
                    "video_sha256": sha256_file(video),
                }
            )
        return {"observations": observations}

    def vbench_runner(*, video, prompt, plan_id, prompt_id, seed, manifest_hash, output, **_):
        output.mkdir(parents=True)
        raw = output / "metrics_eval_results.json"
        raw.write_text("{}")
        return {
            "plan_id": plan_id,
            "prompt_id": prompt_id,
            "seed": seed,
            "prompt_hash": hash_canonical("forge/mvp-prompt/v1", prompt),
            "video_sha256": sha256_file(video),
            "manifest_hash": manifest_hash,
            "runner_source_sha256": sha256_file(run_vbench_isolated.__code__.co_filename),
            "provenance": {
                "implementation_commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490",
                "weight_sha256": weights,
                "runner_image_digest": "sha256:" + "a" * 64,
            },
            "scores": {name: 0.5 for name in weights},
            "raw_sha256": sha256_file(raw),
        }

    calls = 0

    def provider_response(request, timeout):
        nonlocal calls
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert timeout == 180
        calls += 1
        response = {
            "id": f"response-{calls}",
            "model": MODEL,
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "axes": {
                                        axis: {"a": 2, "b": 2, "reason": "same"}
                                        for axis in DEFAULT_AXES
                                    },
                                    "uncertain_axes": [],
                                }
                            ),
                        }
                    ],
                }
            ],
        }
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-provider-key")
    monkeypatch.setattr("forge.evaluation.vision_judge.urllib.request.urlopen", provider_response)
    luna_judge = judge_pair
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(encoded, format="PNG")
    sheet = encoded.getvalue()

    def paired_runner(*, prompts, **_):
        measurements, videos = [], {"reference": {}, "candidate": {}}
        for prompt in prompts:
            for order, side in enumerate(("reference", "candidate", "candidate", "reference")):
                key = f"{prompt['prompt_id']}:{prompt['video_seeds'][0]}"
                videos[side][key] = _sha(1000 + len(measurements))
                measurements.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "seed": prompt["video_seeds"][0],
                        "side": side,
                        "run_order": order,
                        "status": "OK",
                        "gpu_synchronized": True,
                        "wall_time_seconds": 1.0,
                        "full_pipeline_seconds": 1.0,
                    }
                )
        return {
            "paired_timing": {
                "schema": "forge/paired-timing/v1",
                "runtime": {
                    "dtype": "bfloat16",
                    "timing_scope": {
                        "competitive_includes": ["text_encoding", "denoising", "vae_decoding"]
                    },
                },
                "recipes": {"reference": {"steps": 16}, "candidate": {"steps": 16}},
                "hardware": {"gpu_name": "fake"},
                "measurements": measurements,
                "report_video_sha256": videos,
            }
        }

    monkeypatch.setattr("forge.validator.distributed_mvp_runtime._sheets", lambda *_: [sheet])
    result = execute_assigned_confirmation(
        job,
        HmacSigner("a", b"key-a"),
        out=tmp_path / "out",
        panel_runner=panel_runner,
        paired_runner=paired_runner,
        vbench_runner=vbench_runner,
        luna_judge=luna_judge,
    )
    assert result["partial_report"]["validator_hotkey"] == "a"
    for field in ("quality_vs_king", "quality_vs_wan50", "timing"):
        assert all(
            row["runtime_id"] == runtime_identity(job) for row in result["partial_report"][field]
        )
    assert (
        verify_partial_report(plan, result["partial_report"], HmacVerifier({"a": b"key-a"}))[
            "validator_hotkey"
        ]
        == "a"
    )
    assert (tmp_path / "out" / "partial-report.json").is_file()
    assert calls == 2 * len(
        result["partial_report"]["quality_vs_king"] + result["partial_report"]["quality_vs_wan50"]
    )
    first_calls = calls

    def must_not_start(**_):
        raise AssertionError("complete evidence was regenerated")

    resumed = execute_assigned_confirmation(
        job,
        HmacSigner("a", b"key-a"),
        out=tmp_path / "out",
        panel_runner=must_not_start,
        paired_runner=must_not_start,
        vbench_runner=must_not_start,
        luna_judge=luna_judge,
    )
    assert resumed["partial_report"] == result["partial_report"]
    assert calls == first_calls
    raw = next((tmp_path / "out" / "vbench").glob("*/metrics_eval_results.json"))
    raw.write_text("corrupt")
    with pytest.raises(ValueError, match="raw VBench evidence"):
        execute_assigned_confirmation(
            job,
            HmacSigner("a", b"key-a"),
            out=tmp_path / "out",
            panel_runner=must_not_start,
            paired_runner=must_not_start,
            vbench_runner=must_not_start,
            luna_judge=luna_judge,
        )

    bad = AssignedRuntime(
        **{
            **job.__dict__,
            "mvp_plan": {
                **mvp_plan,
                "prompts": [*prompts[:-1], {**prompts[-1], "text": "tampered"}],
            },
        }
    )
    with pytest.raises(ValueError, match="canonical chain-bound"):
        execute_assigned_confirmation(
            bad, HmacSigner("a", b"key-a"), out=tmp_path / "bad", panel_runner=panel_runner
        )

    def must_not_start_gpu(**_):
        raise AssertionError("a frozen-input failure must happen before CUDA work")

    tampered_work = AssignedRuntime(
        **{
            **job.__dict__,
            "work": MVPWork(
                **{
                    **work.__dict__,
                    "quality_prompts": (
                        {
                            **work.quality_prompts[0],
                            "text": "substituted after assignment",
                        },
                        *work.quality_prompts[1:],
                    ),
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="quality panel differ"):
        execute_assigned_confirmation(
            tampered_work,
            HmacSigner("a", b"key-a"),
            out=tmp_path / "tampered-work",
            panel_runner=must_not_start_gpu,
        )

    for malformed_digest in ("sha256:image", "sha256:" + "g" * 64, "a" * 64):
        invalid_provenance = AssignedRuntime(
            **{
                **job.__dict__,
                "vbench": VBenchConfig(
                    **{**job.vbench.__dict__, "runner_image_digest": malformed_digest}
                ),
            }
        )
        with pytest.raises(ValueError, match="runner image digest"):
            execute_assigned_confirmation(
                invalid_provenance,
                HmacSigner("a", b"key-a"),
                out=tmp_path / "invalid-provenance",
                panel_runner=must_not_start_gpu,
                paired_runner=must_not_start_gpu,
            )
        assert not (tmp_path / "invalid-provenance").exists()

    stale_code = AssignedRuntime(
        **{
            **job.__dict__,
            "policy_provenance": {
                **job.policy_provenance,
                "source_implementation_hash": "old-fusion-only-hash",
            },
        }
    )
    with pytest.raises(ValueError, match="executing implementation"):
        execute_assigned_confirmation(
            stale_code,
            HmacSigner("a", b"key-a"),
            out=tmp_path / "stale-code",
            panel_runner=must_not_start_gpu,
            paired_runner=must_not_start_gpu,
        )
    assert not (tmp_path / "stale-code").exists()

    unbound_reservation = AssignedRuntime(**{**job.__dict__, "operation_key": "manual"})
    with pytest.raises(ValueError, match="reservation key"):
        execute_assigned_confirmation(
            unbound_reservation,
            HmacSigner("a", b"key-a"),
            out=tmp_path / "unbound-reservation",
            panel_runner=must_not_start_gpu,
        )

    # A provider timeout might have charged.  It blocks a fresh output directory
    # before any render instead of silently retrying the paid comparison.
    job.luna_budget.reserve()
    with pytest.raises(ValueError, match="charge is unsettled"):
        execute_assigned_confirmation(
            job,
            HmacSigner("a", b"key-a"),
            out=tmp_path / "unsettled-luna",
            panel_runner=must_not_start_gpu,
        )


def test_isolated_vbench_accepts_only_a_subprocess_receipt_bound_to_its_request(
    tmp_path, monkeypatch
):
    """CPU-only boundary check; the fake child cannot claim a real VBench run."""

    video = tmp_path / "video.mp4"
    video.write_bytes(b"frozen-video")
    weights = {
        name: _sha(index + 80)
        for index, name in enumerate(
            (
                "subject_consistency",
                "background_consistency",
                "imaging_quality",
                "motion_smoothness",
                "dynamic_degree",
            )
        )
    }
    fake_worker = tmp_path / "fake-vbench-worker.py"
    fake_worker.write_text(
        "#!" + sys.executable + "\n"
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "from forge.evaluation.vbench import VBENCH_COMMIT\n"
        "from forge.protocol.hashing import hash_canonical\n"
        "raw = json.loads(Path(sys.argv[-1]).read_text())\n"
        "output = Path(raw['output']); output.mkdir()\n"
        "receipt = {\n"
        " 'schema': 'forge/vbench-run/v1',\n"
        " 'video_sha256': hashlib.sha256(Path(raw['video']).read_bytes()).hexdigest(),\n"
        " 'plan_id': raw['plan_id'], 'prompt_id': raw['prompt_id'], 'seed': raw['seed'],\n"
        " 'manifest_hash': raw['manifest_hash'],\n"
        " 'prompt_hash': hash_canonical('forge/mvp-prompt/v1', raw['prompt']),\n"
        " 'provenance': {'implementation_commit': VBENCH_COMMIT,\n"
        "                'weight_sha256': raw['weight_sha256'],\n"
        "                'runner_image_digest': raw['runner_image_digest']},\n"
        " 'dino_revision': raw['dino_revision'],\n"
        " 'scores': {key: .5 for key in raw['weight_sha256']},\n"
        " 'raw_sha256': None,\n"
        "}\n"
        "raw_output = output / 'metrics_eval_results.json'\n"
        "raw_output.write_text('{}')\n"
        "receipt['raw_sha256'] = hashlib.sha256(raw_output.read_bytes()).hexdigest()\n"
        "(output / 'receipt.json').write_text(json.dumps(receipt))\n"
    )
    os.chmod(fake_worker, 0o700)
    receipt = run_vbench_isolated(
        video=video,
        prompt="a moving glass sculpture",
        output=tmp_path / "metric-output",
        checkout=tmp_path,
        cache=tmp_path,
        interpreter=fake_worker,
        weight_sha256=weights,
        dino_revision="d" * 40,
        runner_image_digest="sha256:" + "a" * 64,
        plan_id="a" * 64,
        prompt_id="prompt",
        seed=7,
        manifest_hash="b" * 64,
    )
    assert receipt["video_sha256"] == sha256_file(video)
    assert (tmp_path / "metric-output" / "receipt.json").is_file()


def test_isolated_vbench_preserves_a_distinct_metrics_venv_interpreter(tmp_path):
    """A venv Python is commonly a symlink; resolving it selects its base Python."""

    metrics_venv = tmp_path / "metrics-venv"
    subprocess.run([sys.executable, "-m", "venv", str(metrics_venv)], check=True)
    interpreter = metrics_venv / "bin" / "python"
    site = Path(
        subprocess.run(
            [str(interpreter), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    worker = site / "forge" / "evaluation" / "vbench_worker.py"
    worker.parent.mkdir(parents=True)
    (site / "forge" / "__init__.py").write_text("")
    (worker.parent / "__init__.py").write_text("")
    worker.write_text(
        textwrap.dedent(
            """\
            import hashlib
            import json
            import sys
            from pathlib import Path

            raw = json.loads(Path(sys.argv[-1]).read_text())
            def canonical(domain, value):
                body = {'domain': domain, 'parts': [value]}
                encoded = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
                return hashlib.sha256(encoded).hexdigest()

            output = Path(raw['output'])
            output.mkdir()
            raw_output = output / 'metrics_eval_results.json'
            raw_output.write_text('{}')
            receipt = {
                'schema': 'forge/vbench-run/v1',
                'video_sha256': hashlib.sha256(Path(raw['video']).read_bytes()).hexdigest(),
                'plan_id': raw['plan_id'],
                'prompt_id': raw['prompt_id'],
                'seed': raw['seed'],
                'manifest_hash': raw['manifest_hash'],
                'prompt_hash': canonical('forge/mvp-prompt/v1', raw['prompt']),
                'provenance': {
                    'implementation_commit': 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490',
                    'weight_sha256': raw['weight_sha256'],
                    'runner_image_digest': raw['runner_image_digest'],
                },
                'dino_revision': raw['dino_revision'],
                'scores': {key: .5 for key in raw['weight_sha256']},
                'raw_sha256': hashlib.sha256(raw_output.read_bytes()).hexdigest(),
                'metrics_interpreter_prefix': sys.prefix,
            }
            (output / 'receipt.json').write_text(json.dumps(receipt))
            """
        )
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"venv-interpreter")
    weights = {
        name: _sha(index + 110)
        for index, name in enumerate(
            (
                "subject_consistency",
                "background_consistency",
                "imaging_quality",
                "motion_smoothness",
                "dynamic_degree",
            )
        )
    }
    receipt = run_vbench_isolated(
        video=video,
        prompt="a bird flies through mist",
        output=tmp_path / "metric-venv-output",
        checkout=tmp_path,
        cache=tmp_path,
        interpreter=interpreter,
        weight_sha256=weights,
        dino_revision="e" * 40,
        runner_image_digest="sha256:" + "a" * 64,
        plan_id="c" * 64,
        prompt_id="venv-prompt",
        seed=9,
        manifest_hash="d" * 64,
    )
    assert receipt["metrics_interpreter_prefix"] == str(metrics_venv)
    assert not list((site / "forge").rglob("*.pyc"))
    assert (tmp_path / "metric-venv-output.stderr.txt").is_file()
    worker.write_text(
        "import sys, time\n"
        "print('metric started', flush=True)\n"
        "print('waiting for CUDA', file=sys.stderr, flush=True)\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_vbench_isolated(
            video=video,
            prompt="a bird flies through mist",
            output=tmp_path / "metric-timeout",
            checkout=tmp_path,
            cache=tmp_path,
            interpreter=interpreter,
            weight_sha256=weights,
            dino_revision="e" * 40,
            runner_image_digest="sha256:" + "a" * 64,
            plan_id="c" * 64,
            prompt_id="venv-prompt",
            seed=9,
            manifest_hash="d" * 64,
            timeout_seconds=1,
        )
    assert (tmp_path / "metric-timeout.stdout.txt").read_text() == "metric started\n"
    assert (tmp_path / "metric-timeout.stderr.txt").read_text() == "waiting for CUDA\n"
    assert not list(tmp_path.glob("vbench-request-*.json"))
