"""No-GPU integration coverage for the frozen MVP to runtime bridge."""

import pytest

from forge.evaluation.mvp_evidence import freeze_evaluation_plan
from forge.evaluation.mvp_policy import freeze_policy
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.budget import BudgetError, BudgetLedger
from forge.validator.mvp_runtime import (
    prepare_mvp_work,
    timing_rows_from_paired_receipt,
)
from forge.validator.runtime_preflight import PASS, SCHEMA


def _sha(number: int) -> str:
    return f"{number:064x}"


def _model(reference: ArtifactReference, role: str, steps: int, *, base=None) -> ModelManifest:
    return ModelManifest(
        reference,
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


def _prompts():
    return [
        {
            "prompt_id": f"{subject}-{camera}-{style}-{index}",
            "text": f"{subject} {camera} {style} scene {index}",
            "seed": index,
            "subject_class": subject,
            "camera_type": camera,
            "style_family": style,
        }
        for subject in ("human", "animal", "object", "scenery")
        for camera in ("static", "moving")
        for style in ("natural", "stylized")
        for index in range(8)
    ]


def test_frozen_confirmation_prepares_128_quality_and_exactly_32_timing_slots():
    contract = ChallengeContract(max_steps=50, reference_steps=50)
    base_ref = ArtifactReference("hf", contract.base_model_id, contract.base_model_revision)
    base = _model(base_ref, "pipeline", 50)
    candidate = _model(
        ArtifactReference("hf", "public/candidate", "a" * 40), "lora", 16, base=base_ref
    )
    incumbent = _model(
        ArtifactReference("hf", "public/incumbent", "b" * 40), "lora", 16, base=base_ref
    )
    policy = freeze_policy(
        state="frozen",
        threshold_derivation_hash="c" * 64,
        evaluator_provenance={
            "source_implementation_hash": "source@pin",
            "vbench_model_hashes": "weights@pin",
            "luna_evaluator_id": "gpt-6-luna@pin",
        },
    )
    plan = freeze_evaluation_plan(
        stage="confirmation",
        prompts=_prompts(),
        policy=policy,
        candidate_manifest_hash=candidate.fingerprint(),
        incumbent_manifest_hash=incumbent.fingerprint(),
        wan50_manifest_hash=base.fingerprint(),
    )
    preflight_plan = {
        "contract_hash": contract.contract_hash(),
        "admitted": [
            {
                "manifest_hash": candidate.fingerprint(),
                "manifest": candidate.to_dict(),
                "block": 10,
                "hotkey": "new-hotkey",
            }
        ],
    }
    receipt = {
        "schema": SCHEMA,
        "plan_hash": hash_canonical("forge/runtime-preflight-plan/v1", preflight_plan),
        "contract_hash": contract.contract_hash(),
        "base_manifest_hash": base.fingerprint(),
        "candidates": [
            {
                "manifest_hash": candidate.fingerprint(),
                "weight_content_hash": _sha(8),
                "effective_candidate_hash": _sha(9),
                "status": PASS,
            }
        ],
    }
    binding = {
        "candidate_manifest_hash": candidate.fingerprint(),
        "incumbent_manifest_hash": incumbent.fingerprint(),
        "evaluation_plan_id": plan["plan_id"],
        "admission_history_root": _sha(10),
        "prompt_history_root": _sha(11),
        "certified_state_hash": _sha(12),
    }
    work = prepare_mvp_work(
        plan=plan,
        policy=policy,
        contract=contract,
        preflight_plan=preflight_plan,
        preflight_receipt=receipt,
        candidate=candidate,
        incumbent=incumbent,
        wan50=base,
        certified_binding=binding,
        verify_certified_binding=lambda value: value == binding,
    )
    assert len(work.quality_prompts) == 128
    assert len(work.timing_prompts) == 32
    assert {row["prompt_id"] for row in work.timing_prompts} <= {
        row["prompt_id"] for row in work.quality_prompts
    }
    assert all(len(row["video_seeds"]) == 1 for row in work.timing_prompts)

    measurements = []
    videos = {"reference": {}, "candidate": {}}
    for index, prompt in enumerate(work.timing_prompts):
        for order, side in enumerate(("reference", "candidate", "candidate", "reference")):
            video = _sha(100_000 + index * 4 + order)
            videos[side][f"{prompt['prompt_id']}:{prompt['video_seeds'][0]}"] = video
            measurements.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "seed": prompt["video_seeds"][0],
                    "side": side,
                    "run_order": order,
                    "status": "OK",
                    "gpu_synchronized": True,
                    "wall_time_seconds": 2.0 if side == "candidate" else 4.0,
                    "full_pipeline_seconds": 2.0 if side == "candidate" else 4.0,
                }
            )
    rows = timing_rows_from_paired_receipt(
        work,
        {
            "schema": "forge/paired-timing/v1",
            "runtime": {
                "dtype": "bfloat16",
                "timing_scope": {
                    "competitive_includes": ["text_encoding", "denoising", "vae_decoding"]
                },
            },
            "recipes": {"reference": {"steps": 16}, "candidate": {"steps": 16}},
            "hardware": {"gpu_name": "test-gpu"},
            "measurements": measurements,
            "report_video_sha256": videos,
        },
        dependency_lock_hash=_sha(13),
    )
    assert len(rows) == 32 and all(len(row["runs"]) == 4 for row in rows)

    with pytest.raises(ValueError, match="certified competition binding"):
        prepare_mvp_work(
            plan=plan,
            policy=policy,
            contract=contract,
            preflight_plan=preflight_plan,
            preflight_receipt=receipt,
            candidate=candidate,
            incumbent=incumbent,
            wan50=base,
            certified_binding=None,
            verify_certified_binding=None,
        )


def test_runtime_budget_must_be_a_durable_active_reservation(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite")
    reservation = ledger.reserve(7, "mvp-runtime:plan", 3.0)
    assert (
        ledger.require_active(
            reservation.id, candidate_id=7, operation_key="mvp-runtime:plan", minimum_usd=2.0
        ).status
        == "RESERVED"
    )
    with pytest.raises(BudgetError, match="absent or insufficient"):
        ledger.require_active(
            reservation.id, candidate_id=7, operation_key="mvp-runtime:other", minimum_usd=2.0
        )
