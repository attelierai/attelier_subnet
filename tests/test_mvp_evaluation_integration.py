"""Synthetic contract fixtures only; they never certify a real video result."""

import json
import subprocess
import sys

import pytest

from forge.cli import main
from forge.evaluation.mvp import evaluate_confirmation, evaluate_mvp, evaluate_screening
from forge.evaluation.mvp_calibration import fit_policy_from_development
from forge.evaluation.mvp_evidence import _text_hash, freeze_evaluation_plan
from forge.evaluation.mvp_policy import (
    CONTROL_KINDS,
    QUALITY_AXES,
    Decision,
    EvaluationPolicy,
    _sha,
    freeze_policy,
)
from forge.protocol.hashing import hash_canonical


def _prompts(count):
    values = []
    for subject in ("human", "animal", "object", "scenery"):
        for camera in ("static", "moving"):
            for style in ("natural", "stylized"):
                for number in range(8):
                    values.append(
                        {
                            "prompt_id": f"{subject}-{camera}-{style}-{number}",
                            "text": f"synthetic {subject} {camera} {style} {number}",
                            "seed": number,
                            "subject_class": subject,
                            "camera_type": camera,
                            "style_family": style,
                        }
                    )
    return values[:count]


def _quality(prompts, delta=0.0):
    return [
        {
            **prompt,
            "orders": _orders(delta),
        }
        for prompt in prompts
    ]


def _orders(delta):
    return [
        {"reference_first": True, "axes": {axis: delta for axis in QUALITY_AXES}},
        {"reference_first": False, "axes": {axis: delta for axis in QUALITY_AXES}},
    ]


def _policy():
    return fit_policy_from_development(
        [row for row in _control_rows() if row["split"] == "development"],
        evaluator_provenance=_provenance(),
    )[0]


def _provenance():
    return {
        "source_implementation_hash": "source@pin",
        "vbench_model_hashes": "weights@pin",
        "luna_evaluator_id": "gpt-6-luna@pin",
    }


def _timing(count, reference=12.0, candidate=8.0):
    return [
        {
            "prompt_id": _prompts(count)[number]["prompt_id"],
            "runs": [
                {"model": "reference", "seconds": reference + number / 1_000},
                {"model": "candidate", "seconds": candidate + number / 1_000},
                {"model": "candidate", "seconds": candidate + number / 1_000},
                {"model": "reference", "seconds": reference + number / 1_000},
            ],
        }
        for number in range(count)
    ]


def _bound_quality(prompts, reference_manifest, candidate_manifest):
    rows = _quality(prompts)
    for index, row in enumerate(rows):
        row.update(
            seed=prompts[index]["seed"],
            prompt_hash=_text_hash(prompts[index]["text"]),
            reference_manifest_hash=reference_manifest,
            candidate_manifest_hash=candidate_manifest,
            reference_video_sha256=f"{index + 1:064x}",
            candidate_video_sha256=f"{index + 10_000:064x}",
            evidence_kind="validator_fused_quality",
            raw_evidence_hash=f"{index + 40_000:064x}",
        )
    return rows


def _bound_timing(prompts, reference_manifest, candidate_manifest):
    rows = _timing(32)
    for index, row in enumerate(rows):
        row.update(
            candidate_manifest_hash=candidate_manifest,
            reference_manifest_hash=reference_manifest,
            gpu_synchronized=True,
            timing_scope="text_encode+denoise+decode",
            hardware={"gpu": "synthetic", "precision": "fp16", "dependency_lock_hash": "x"},
            recipes={"reference": "ref", "candidate": "cand"},
            warmup={"reference": True, "candidate": True},
        )
        for run_index, run in enumerate(row["runs"]):
            run.update(
                gpu_synchronized=True,
                video_sha256=f"{index * 10 + run_index + 20_000:064x}",
                recipe_hash=f"{index * 10 + run_index + 30_000:064x}",
                full_pipeline_seconds=run["seconds"],
            )
    return rows


def _screen_timing(reference=12.0, candidate=8.0):
    return [
        {
            "prompt_id": f"screen-time-{number}",
            "reference_seconds": reference + number / 1_000,
            "candidate_seconds": candidate + number / 1_000,
        }
        for number in range(16)
    ]


def _control_rows(*, freeze_detected=True):
    rows = []
    for split in ("development", "calibration", "independent"):
        for kind in CONTROL_KINDS:
            for number in range(32):
                rows.append(
                    {
                        "split": split,
                        "kind": kind,
                        "source_id": f"{split}-source-{kind}-{number}",
                        "concept_id": f"{split}-concept-{kind}-{number}",
                        "source_family": f"{split}-source-family-{kind}-{number}",
                        "concept_family": f"{split}-concept-family-{kind}-{number}",
                        "observable": True,
                        "orders": _orders(
                            -0.5
                            if kind
                            not in {
                                "self",
                                "same_model_new_run",
                                "independent_seed",
                                "static_scene",
                            }
                            and freeze_detected
                            else 0.0
                        ),
                    }
                )
    return rows


def _calibration(policy, *, self_detected=False, freeze_detected=True):
    rows = _control_rows(freeze_detected=freeze_detected)
    _, derivation = fit_policy_from_development(
        [row for row in _control_rows() if row["split"] == "development"],
        evaluator_provenance=_provenance(),
    )
    return {
        "policy_id": policy.policy_id,
        "rows": rows,
        "evaluator_provenance": dict(policy.evaluator_provenance or {}),
        "threshold_derivation": derivation,
        "independent_check_hash": _sha(
            {
                "schema": "forge/mvp-independent-calibration/v1",
                "policy_id": policy.policy_id,
                "threshold_derivation_hash": policy.threshold_derivation_hash,
                "rows": rows,
            }
        ),
    }


def test_complete_independent_calibration_and_bounded_panel_can_promote():
    policy, prompts = _policy(), _prompts(128)
    result = evaluate_confirmation(
        prompts,
        _quality(prompts),
        _timing(32),
        _calibration(policy),
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert result["status"] == Decision.PROMOTE_ELIGIBLE
    assert result["speed_lower_ci"] >= 1.1


@pytest.mark.parametrize("invalid_seconds", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_timing_cannot_promote_otherwise_valid_confirmation(invalid_seconds):
    policy, prompts = _policy(), _prompts(128)
    timing = _timing(32)
    timing[0]["runs"][0]["seconds"] = invalid_seconds
    result = evaluate_confirmation(
        prompts,
        _quality(prompts),
        timing,
        _calibration(policy),
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert result["status"] == Decision.INCOMPLETE
    json.dumps(result, allow_nan=False)


def test_threshold_is_fitted_before_independent_check_and_check_hash_is_recomputed():
    seed_policy = _policy()
    development_rows = [
        row for row in _calibration(seed_policy)["rows"] if row["split"] == "development"
    ]
    policy, derivation = fit_policy_from_development(
        development_rows, evaluator_provenance=seed_policy.evaluator_provenance or {}
    )
    assert policy.state == "frozen"
    assert policy.quality_margin == 0.0
    assert policy.threshold_derivation_hash == derivation["derivation_hash"]
    receipt = _calibration(policy)
    receipt["independent_check_hash"] = "0" * 64
    result = evaluate_confirmation(
        _prompts(128),
        _quality(_prompts(128)),
        _timing(32),
        receipt,
        policy,
        wan50_quality_rows=_quality(_prompts(128)),
    )
    assert result["status"] == Decision.INCOMPLETE


def test_cli_writes_only_a_development_policy(tmp_path):
    destination = tmp_path / "policy.json"
    assert main(["mvp-policy", "--out", str(destination)]) == 0
    assert EvaluationPolicy.from_dict(json.loads(destination.read_text())).state == "development"


@pytest.mark.parametrize(
    "attack", ["margin", "method", "source_family", "concept_family", "missing_legitimate"]
)
def test_calibration_cli_rejects_rehashed_thresholds_and_related_controls(tmp_path, attack):
    policy = _policy()
    receipt = _calibration(policy)
    if attack in {"margin", "method", "missing_legitimate"}:
        derivation = receipt["threshold_derivation"]
        if attack == "margin":
            derivation["quality_margin"] = 0.25
        elif attack == "method":
            derivation["method"] = "hand-picked-after-holdout"
        else:
            receipt["rows"] = [
                row
                for row in receipt["rows"]
                if not (row["split"] == "development" and row["kind"] == "independent_seed")
            ]
            derivation["development_rows_hash"] = _sha(
                [row for row in receipt["rows"] if row["split"] == "development"]
            )
        derivation["derivation_hash"] = _sha(
            {key: value for key, value in derivation.items() if key != "derivation_hash"}
        )
        body = policy.body()
        body["quality_margin"] = derivation["quality_margin"]
        body["threshold_derivation_hash"] = derivation["derivation_hash"]
        policy = EvaluationPolicy.from_dict({**body, "policy_id": _sha(body)})
        receipt["policy_id"] = policy.policy_id
        reason = "THRESHOLD_DERIVATION_NOT_REPRODUCIBLE"
    else:
        # Distinct file IDs, but all 32 controls are siblings of one source/concept.
        for row in receipt["rows"]:
            if row["split"] == "independent" and row["kind"] == "blur_mild":
                row[attack] = "independent-shared-family"
        reason = "CALIBRATION_DEPENDENT_CONTROLS"
    receipt["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": receipt["rows"],
        }
    )
    policy_path, receipt_path, result_path = [
        tmp_path / name for name in ("policy.json", "receipt.json", "result.json")
    ]
    policy_path.write_text(json.dumps(policy.to_dict()))
    receipt_path.write_text(json.dumps(receipt))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.cli",
            "mvp-calibrate",
            "--policy",
            str(policy_path),
            "--evidence",
            str(receipt_path),
            "--out",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2, process.stderr
    result = json.loads(result_path.read_text())
    assert result["status"] == Decision.INCOMPLETE
    assert result["reasons"] == [reason]
    prompts = _prompts(128)
    decision = evaluate_confirmation(
        prompts,
        _quality(prompts),
        _timing(32),
        receipt,
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert decision["status"] == Decision.INCOMPLETE
    assert reason in decision["reasons"]


def test_mvp_rejects_swapped_seed_category_video_or_source_family_alias():
    policy, prompts = _policy(), _prompts(128)
    candidate, incumbent, wan50 = "a" * 64, "b" * 64, "c" * 64
    plan = freeze_evaluation_plan(
        stage="confirmation",
        prompts=prompts,
        policy=policy,
        candidate_manifest_hash=candidate,
        incumbent_manifest_hash=incumbent,
        wan50_manifest_hash=wan50,
    )
    quality = _bound_quality(prompts, incumbent, candidate)
    wan_quality = _bound_quality(prompts, wan50, candidate)
    for rows in (quality, wan_quality):
        for row in rows:
            row["plan_id"] = plan["plan_id"]
    timing = _bound_timing(prompts, incumbent, candidate)
    calibration = _calibration(policy)
    for field, value in (
        ("seed", 99),
        ("subject_class", "wrong"),
        ("candidate_video_sha256", "bad"),
    ):
        broken = [dict(row) for row in quality]
        broken[0][field] = value
        result = evaluate_mvp(
            plan=plan,
            quality_rows=broken,
            timing_rows=timing,
            policy=policy,
            calibration=calibration,
            wan50_quality_rows=wan_quality,
        )
        assert result["status"] == Decision.INCOMPLETE
    calibration["rows"][0]["source_family"] = calibration["rows"][-1]["source_family"]
    calibration["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": calibration["rows"],
        }
    )
    result = evaluate_mvp(
        plan=plan,
        quality_rows=quality,
        timing_rows=timing,
        policy=policy,
        calibration=calibration,
        wan50_quality_rows=wan_quality,
    )
    assert result["status"] == Decision.INCOMPLETE


def test_exact_bootstrap_alias_reuses_one_canonical_128_row_evidence_set():
    policy, prompts = _policy(), _prompts(128)
    candidate, bootstrap = "a" * 64, "b" * 64
    plan = freeze_evaluation_plan(
        stage="confirmation",
        prompts=prompts,
        policy=policy,
        candidate_manifest_hash=candidate,
        incumbent_manifest_hash=bootstrap,
        wan50_manifest_hash=bootstrap,
    )
    rows = _bound_quality(prompts, bootstrap, candidate)
    for row in rows:
        row["plan_id"] = plan["plan_id"]
        identity = {
            key: row[key]
            for key in (
                "plan_id",
                "prompt_id",
                "seed",
                "prompt_hash",
                "candidate_manifest_hash",
                "reference_manifest_hash",
                "candidate_video_sha256",
                "reference_video_sha256",
                "raw_evidence_hash",
            )
        }
        row["reference_roles"] = ["wan50", "incumbent"]
        row["comparison_id"] = hash_canonical("forge/mvp-bootstrap-comparison/v1", identity)
    result = evaluate_mvp(
        plan=plan,
        quality_rows=rows,
        timing_rows=_bound_timing(prompts, bootstrap, candidate),
        policy=policy,
        calibration=_calibration(policy),
        wan50_quality_rows=rows,
    )
    assert result["status"] == Decision.PROMOTE_ELIGIBLE


def test_missing_or_insensitive_calibration_cannot_reward():
    policy, prompts = _policy(), _prompts(128)
    missing = evaluate_confirmation(
        prompts, _quality(prompts), _timing(32), None, policy, wan50_quality_rows=_quality(prompts)
    )
    insensitive = evaluate_confirmation(
        prompts,
        _quality(prompts),
        _timing(32),
        _calibration(policy, freeze_detected=False),
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert missing["status"] == Decision.INCOMPLETE
    assert insensitive["status"] == Decision.INCOMPLETE


def test_development_margin_and_unobservable_attack_controls_cannot_reward():
    prompts = _prompts(128)
    development = freeze_policy()
    development_result = evaluate_confirmation(
        prompts,
        _quality(prompts),
        _timing(32),
        _calibration(development),
        development,
        wan50_quality_rows=_quality(prompts),
    )
    assert development_result["status"] == Decision.INCOMPLETE

    policy = _policy()
    calibration = _calibration(policy)
    calibration["rows"][0]["observable"] = False
    result = evaluate_confirmation(
        prompts,
        _quality(prompts),
        _timing(32),
        calibration,
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert result["status"] == Decision.INCOMPLETE


def test_protected_category_regression_fails_and_abba_is_mandatory():
    policy, prompts = _policy(), _prompts(128)
    quality = _quality(prompts)
    for row in quality[:8]:
        for axis in QUALITY_AXES:
            for order in row["orders"]:
                order["axes"][axis] = -0.5
    failed = evaluate_confirmation(
        prompts,
        quality,
        _timing(32),
        _calibration(policy),
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert failed["status"] == Decision.FAIL
    bad_timing = _timing(32)
    bad_timing[0]["runs"][0]["model"] = "candidate"
    incomplete = evaluate_confirmation(
        prompts,
        _quality(prompts),
        bad_timing,
        _calibration(policy),
        policy,
        wan50_quality_rows=_quality(prompts),
    )
    assert incomplete["status"] == Decision.INCOMPLETE


def test_screening_uses_16_and_rejects_no_speed_potential():
    policy, prompts = _policy(), _prompts(16)
    result = evaluate_screening(_quality(prompts), _screen_timing(10.0, 10.0), policy)
    assert result["status"] == Decision.SCREEN_FAIL
    assert result["reasons"] == ["SCREEN_NO_SPEED_POTENTIAL"]
