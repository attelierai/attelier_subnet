"""Qualified distributed evidence reaches a one-hot intent only through replay.

The media and calibration rows are synthetic protocol fixtures.  They exercise
two independent signed validators, durable history and certificate boundaries;
they do not claim a Wan improvement.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest

from forge.evaluation.calibration_campaign import freeze_calibration_campaign
from forge.evaluation.capacity import CapacityInputs
from forge.evaluation.distributed import signed_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan, assignments_for
from forge.evaluation.mvp_calibration import (
    fit_policy_from_development,
    validate_qualified_calibration,
)
from forge.evaluation.mvp_evidence import freeze_evaluation_plan
from forge.evaluation.mvp_policy import CONTROL_AXES, LEGITIMATE_CONTROLS, QUALITY_AXES, _sha
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.consensus_types import ConsensusPolicy, FinalizedCompetitionState, RoundPlan
from forge.protocol.hashing import hash_canonical
from forge.protocol.preconsensus import build_certificate, commit_report, decide_round
from forge.protocol.store import Store
from forge.sim import HmacSigner, HmacVerifier
from forge.validator.admission import build_authenticated_history
from forge.validator.competitive_bridge import (
    DistributedExecutionReplayer,
    DistributedMVPCertificateVerifier,
    build_distributed_execution_bundle,
    execution_evidence_map,
)
from forge.validator.competitive_lifecycle import CompetitiveLifecycle


def _hash(number: int) -> str:
    return f"{number:064x}"


def _orders(kind: str | None = None) -> list[dict[str, object]]:
    axes = {axis: 0.0 for axis in QUALITY_AXES}
    if kind and kind not in LEGITIMATE_CONTROLS:
        for axis in CONTROL_AXES[kind]:
            axes[axis] = -0.5
    return [
        {"reference_first": True, "axes": dict(axes)},
        {"reference_first": False, "axes": dict(axes)},
    ]


def _v4_source_bank() -> list[dict[str, str]]:
    rows = []
    for mode in ("static", "motion"):
        cells = (
            [
                (subject, "static", style)
                for subject in ("human", "animal", "rigid_object", "scenery")
                for style in ("realistic", "stylized")
            ]
            if mode == "static"
            else list(
                product(
                    ("human", "animal", "rigid_object", "scenery"),
                    ("static", "moving"),
                    ("realistic", "stylized"),
                )
            )
        )
        repeats = 12 if mode == "static" else 6
        for cell_number, (subject, camera, style) in enumerate(cells):
            for repeat in range(repeats):
                index = cell_number * repeats + repeat
                rows.append(
                    {
                        "source_mode": mode,
                        "source_id": f"bridge-v4-{mode}-source-{index}",
                        "concept_id": f"bridge-v4-{mode}-concept-{index}",
                        "source_family": f"bridge-v4-{mode}-family-{index}",
                        "concept_family": f"bridge-v4-{mode}-concept-family-{index}",
                        "prompt_text": f"Bridge v4 {mode} semantic family {index}.",
                        "subject_class": subject,
                        "camera_type": camera,
                        "style_family": style,
                    }
                )
    return rows


def _calibration():
    provenance = {
        "source_implementation_hash": "fixture-source",
        "vbench_model_hashes": "fixture-weights",
        "luna_evaluator_id": "fixture-luna",
    }
    campaign = freeze_calibration_campaign(
        campaign_label="bridge-fixture", entropy="f" * 64, source_bank=_v4_source_bank()
    )
    rows = [
        {**slot, "observable": True, "orders": _orders(slot["kind"])} for slot in campaign["slots"]
    ]
    policy, derivation = fit_policy_from_development(
        [row for row in rows if row["split"] == "development"],
        evaluator_provenance=provenance,
    )
    receipt = {
        "policy_id": policy.policy_id,
        "campaign": campaign,
        "rows": rows,
        "evaluator_provenance": provenance,
        "threshold_derivation": derivation,
    }
    receipt["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": rows,
        }
    )
    return (
        policy,
        receipt,
        validate_qualified_calibration(receipt, policy)["binding"]["calibration_hash"],
    )


def _qualification_evidence(policy, calibration_hash, committee, prompts):
    workers = {
        hotkey: {
            "runtime_id": _hash(7_000),
            "hardware": {
                "gpu": "fixture",
                "precision": "bf16",
                "dependency_lock_hash": _hash(8_000),
            },
        }
        for hotkey in sorted(committee)
    }
    provenance = dict(policy.evaluator_provenance or {})
    prompts = [
        {
            **prompt,
            "text": f"Independent qualification scene {index}",
            "source_family": f"panel-family-{index}",
            "concept_family": f"panel-concept-{index}",
        }
        for index, prompt in enumerate(prompts)
    ]
    panel_validation = {
        "schema": "forge/independent-panel-validation/v1",
        "policy_id": policy.policy_id,
        "calibration_hash": calibration_hash,
        "evaluator_provenance": provenance,
        "worker_bindings": workers,
        "panels": {
            kind: {
                "prompts": [
                    {**prompt, "source_family": f"panel-family-{index}"}
                    for index, prompt in enumerate(prompts)
                ],
                "quality_vs_king": [
                    {
                        key: prompt[key]
                        for key in (
                            "prompt_id",
                            "seed",
                            "source_family",
                            "concept_family",
                            "subject_class",
                            "camera_type",
                            "style_family",
                        )
                    }
                    | {
                        "prompt_hash": hash_canonical("forge/mvp-prompt/v1", prompt["text"]),
                        "orders": _orders(
                            None
                            if kind == "same_model"
                            else ("blur_mild" if kind == "blur_mild" else "wrong_video")
                        ),
                    }
                    for prompt in prompts
                ],
                "source_family_ids": [f"panel-family-{index}" for index in range(128)],
            }
            for kind in ("same_model", "blur_mild", "wrong_video")
        },
    }
    worker_tolerances = {
        "schema": "forge/cross-worker-tolerance/v1",
        "policy_id": policy.policy_id,
        "calibration_hash": calibration_hash,
        "evaluator_provenance": provenance,
        "development": [],
        "independent": [],
        "worker_bindings": deepcopy(workers),
        "quality_tolerance": 0.0,
        "timing_log_ratio_tolerance": 0.0,
        "tolerance_derivation_hash": "",
    }
    for split in ("development", "independent"):
        for index in range(8):
            worker_tolerances[split].append(
                {
                    "family_id": f"worker-{split}-{index}",
                    "workers": {
                        hotkey: {
                            "quality_axes": {axis: 0.0 for axis in QUALITY_AXES},
                            "runs": [
                                {"model": model, "seconds": 2.0}
                                for model in ("reference", "candidate", "candidate", "reference")
                            ],
                        }
                        for hotkey in committee
                    },
                }
            )
    worker_tolerances["tolerance_derivation_hash"] = hash_canonical(
        "forge/cross-worker-tolerance-development/v1", worker_tolerances["development"]
    )
    return panel_validation, worker_tolerances


def _prompts() -> list[dict[str, object]]:
    return [
        {
            "prompt_id": f"{subject}-{camera}-{style}-{number}",
            "text": f"fixture {subject} {camera} {style} {number}",
            "seed": index,
            "subject_class": subject,
            "camera_type": camera,
            "style_family": style,
        }
        for index, (subject, camera, style, number) in enumerate(
            (subject, camera, style, number)
            for subject in ("human", "animal", "rigid_object", "scenery")
            for camera in ("static", "moving")
            for style in ("realistic", "stylized")
            for number in range(8)
        )
    ]


def _partial(plan, mvp_plan, signer, prompts, candidate, king):
    quality, timing = [], []
    slots = {prompt["prompt_id"]: prompt for prompt in prompts}
    for item in assignments_for(plan, signer.validator_hotkey):
        prompt = slots[item.prompt_id]
        if item.kind == "quality":
            quality.append(
                {
                    **{
                        key: prompt[key]
                        for key in (
                            "prompt_id",
                            "seed",
                            "subject_class",
                            "camera_type",
                            "style_family",
                        )
                    },
                    "plan_id": mvp_plan["plan_id"],
                    "prompt_hash": hash_canonical("forge/mvp-prompt/v1", prompt["text"]),
                    "candidate_manifest_hash": candidate,
                    "reference_manifest_hash": king,
                    "candidate_video_sha256": _hash(1_000 + prompt["seed"]),
                    "reference_video_sha256": _hash(2_000 + prompt["seed"]),
                    "raw_evidence_hash": _hash(3_000 + prompt["seed"]),
                    "runtime_id": _hash(7_000),
                    "orders": _orders(),
                }
            )
        else:
            runs = [
                {"model": "reference", "seconds": 4.0},
                {"model": "candidate", "seconds": 2.0},
                {"model": "candidate", "seconds": 2.0},
                {"model": "reference", "seconds": 4.0},
            ]
            for index, run in enumerate(runs):
                run.update(
                    gpu_synchronized=True,
                    video_sha256=_hash(4_000 + prompt["seed"] * 4 + index),
                    recipe_hash=_hash(5_000 + prompt["seed"] * 4 + index),
                    full_pipeline_seconds=run["seconds"],
                )
            timing.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "seed": prompt["seed"],
                    "candidate_manifest_hash": candidate,
                    "reference_manifest_hash": king,
                    "runtime_id": _hash(7_000),
                    "gpu_synchronized": True,
                    "timing_scope": "text_encode+denoise+decode",
                    "hardware": {
                        "gpu": "fixture",
                        "precision": "bf16",
                        "dependency_lock_hash": _hash(8_000),
                    },
                    "recipes": {"reference": "wan50", "candidate": "candidate"},
                    "warmup": {"reference": True, "candidate": True},
                    "runs": runs,
                }
            )
    return signed_partial_report(
        plan,
        signer,
        quality_vs_king=quality,
        quality_vs_wan50=deepcopy(quality),
        timing=timing,
    )


def test_two_signed_validators_replay_a_full_panel_before_certificate_and_wta_intent(tmp_path):
    keys = {"validator-a": b"a", "validator-b": b"b"}
    signers = {hotkey: HmacSigner(hotkey, key) for hotkey, key in keys.items()}
    verifier = HmacVerifier(keys)
    policy, calibration, calibration_hash = _calibration()
    king, candidate = _hash(2), _hash(3)
    state = FinalizedCompetitionState.genesis("wan", king)
    prompts = _prompts()
    mvp_plan = freeze_evaluation_plan(
        stage="confirmation",
        prompts=prompts,
        policy=policy,
        candidate_manifest_hash=candidate,
        incumbent_manifest_hash=king,
        wan50_manifest_hash=king,
    )
    history = build_authenticated_history(
        network="test",
        netuid=9,
        through_block=100,
        certified_state_hash=state.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    round_plan = RoundPlan(
        "test",
        9,
        "bridge-r1",
        state.state_hash(),
        "wan",
        king,
        candidate,
        "miner",
        _hash(4),
        {
            "admission_history_root": history.history_root,
            "prompt_history_root": _hash(5),
            "evaluation_plan_id": mvp_plan["plan_id"],
        },
        {hotkey: 1 for hotkey in signers},
        100,
        110,
        120,
        130,
        ConsensusPolicy.STAKE_MAJORITY_V2,
    )
    distributed = DistributedEvaluationPlan(
        round_id=round_plan.round_id,
        stage="confirmation",
        entropy="e" * 64,
        policy_id=policy.policy_id,
        candidate_manifest_hash=candidate,
        king_manifest_hash=king,
        wan50_manifest_hash=king,
        committee=dict(round_plan.committee),
        prompts=tuple(prompts),
        timing_prompt_ids=tuple(prompt["prompt_id"] for prompt in prompts[:32]),
        quality_tolerance=0.0,
        timing_log_ratio_tolerance=0.0,
        audit_calibration_hash=calibration_hash,
        chain_context={
            "network": "test",
            "netuid": 9,
            "close_block": 100,
            "mvp_plan_id": mvp_plan["plan_id"],
            "round_plan_hash": round_plan.plan_hash(),
        },
    )
    reports = [
        _partial(distributed, mvp_plan, signer, prompts, candidate, king)
        for signer in signers.values()
    ]
    panel_validation, worker_tolerances = _qualification_evidence(
        policy, calibration_hash, round_plan.committee, prompts
    )
    bundle = build_distributed_execution_bundle(
        round_plan=round_plan,
        distributed_plan=distributed,
        reports=reports,
        mvp_plan=mvp_plan,
        policy=policy,
        calibration=calibration,
        panel_validation=panel_validation,
        worker_tolerances=worker_tolerances,
        public_custody={"manifest_hash": candidate, "uri": "https://example.invalid/candidate"},
    )
    replayer = DistributedExecutionReplayer(verifier)
    assert replayer.replay(bundle, round_plan)["status"] == "PROMOTE_ELIGIBLE"
    malformed = []
    missing = deepcopy(bundle)
    missing.pop("panel_validation")
    malformed.append(missing)
    mismatch = deepcopy(bundle)
    mismatch["panel_validation"]["panels"]["same_model"]["quality_vs_king"][0]["seed"] += 1
    malformed.append(mismatch)
    reused = deepcopy(bundle)
    reused["worker_tolerances"]["independent"][0]["family_id"] = reused["worker_tolerances"][
        "development"
    ][0]["family_id"]
    malformed.append(reused)
    outside = deepcopy(bundle)
    outside["worker_tolerances"]["independent"][0]["workers"]["validator-a"]["runs"][0][
        "seconds"
    ] = 8.0
    malformed.append(outside)
    degradation_missing = deepcopy(bundle)
    for row in degradation_missing["panel_validation"]["panels"]["blur_mild"]["quality_vs_king"]:
        row["orders"] = _orders()
    malformed.append(degradation_missing)
    wrong_runtime = deepcopy(bundle)
    first = wrong_runtime["reports"][0]
    first["quality_vs_king"][0]["runtime_id"] = _hash(999)
    wrong_runtime["reports"][0] = signed_partial_report(
        distributed,
        signers[first["validator_hotkey"]],
        quality_vs_king=first["quality_vs_king"],
        quality_vs_wan50=first["quality_vs_wan50"],
        timing=first["timing"],
    )
    malformed.append(wrong_runtime)
    for bad in malformed:
        with pytest.raises(ValueError):
            replayer.replay(bad, round_plan)
    local_reports = [
        replayer.local_report(
            round_plan,
            hotkey,
            bundle,
            verify_public_copy=lambda fingerprint, custody: fingerprint == custody["manifest_hash"],
        )
        for hotkey in signers
    ]
    commits, reveals = zip(
        *[
            commit_report(round_plan, report, 109, salt=f"salt-{index}")
            for index, report in enumerate(local_reports)
        ],
        strict=True,
    )
    certificate = build_certificate(
        round_plan,
        decide_round(round_plan, commits, reveals),
        activation_chain_time=1,
        signers=list(signers.values()),
        commits=commits,
        reveals=reveals,
    )
    verifier_bridge = DistributedMVPCertificateVerifier(
        verifier,
        evidence=execution_evidence_map(bundle),
        replayer=replayer,
        verify_public_copy=lambda fingerprint, custody: fingerprint == custody["manifest_hash"],
    )
    with Store(tmp_path / "competitive.sqlite") as store:
        lifecycle = CompetitiveLifecycle(store, verifier)
        lifecycle.open_round(
            round_plan,
            state,
            create_signed_history(history, list(signers.values())),
            distributed_plan=distributed,
            capacity_inputs=CapacityInputs(
                king_seconds=0.001,
                wan50_seconds=0.001,
                candidate_seconds=0.001,
                gpu_usd_hour=0.01,
                chain_block_seconds=0.1,
                commit_delivery_seconds=0.1,
                reveal_delivery_seconds=0.1,
                network_remaining_usd=100.0,
                validator_candidate_budget_usd=100.0,
                node_overhead_seconds=0.0,
                node_overhead_usd=0.0,
                metric_seconds_per_video=0.0,
                judge_usd_per_quality_comparison=0.0,
                storage_usd_per_video=0.0,
                validator_screening_spent_usd=0.0,
            ),
        )
        delivered = lifecycle.certify_and_prepare_weight_intent(
            plan=round_plan,
            state=state,
            certificate=certificate,
            mvp_verifier=verifier_bridge,
            runtime_hash=_hash(6),
            registered_uids={7: "burn", 8: "miner"},
            public_custody={candidate: _hash(7)},
            verify_public_custody=lambda *_args: True,
            burn_hotkey="burn",
            verify_burn_qualification=lambda _hotkey: _hash(8),
        )
    assert delivered.intent.status == "WINNER"
    assert delivered.intent.uid_weights == {8: 1.0}


def test_competitive_bundle_fails_closed_without_independent_panel_or_worker_tolerances():
    policy, calibration, calibration_hash = _calibration()
    committee = {"validator-a": 1, "validator-b": 1}
    panel, tolerances = _qualification_evidence(policy, calibration_hash, committee, _prompts())
    plan = DistributedEvaluationPlan(
        round_id="receipt-only",
        stage="confirmation",
        entropy="e" * 64,
        policy_id=policy.policy_id,
        candidate_manifest_hash=_hash(31),
        king_manifest_hash=_hash(30),
        wan50_manifest_hash=_hash(30),
        committee=committee,
        prompts=tuple(_prompts()),
        timing_prompt_ids=tuple(prompt["prompt_id"] for prompt in _prompts()[:32]),
        quality_tolerance=1.0,
        timing_log_ratio_tolerance=1.0,
        audit_calibration_hash=calibration_hash,
        chain_context={},
    )
    with pytest.raises(Exception, match="receipt fields|tolerance"):
        # Exercise the receipt parser directly: no missing evidence can turn a
        # calibration-only campaign into a competitive qualification.
        from forge.validator.competitive_bridge import _qualification_evidence as validate

        validate(
            panel_validation={**panel, "status": "MISSING"},
            worker_tolerances=tolerances,
            policy=policy,
            calibration=calibration,
            calibration_hash=calibration_hash,
            distributed_plan=plan,
        )
