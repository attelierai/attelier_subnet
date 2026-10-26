"""Native dashboard metrics replay signed diagnostic evidence end to end."""

from __future__ import annotations

import json

import pytest

bt = pytest.importorskip("bittensor")

from forge.chain import Sr25519Signer  # noqa: E402
from forge.evaluation.diagnostic_fusion import fuse_diagnostic_quality  # noqa: E402
from forge.evaluation.distributed import signed_partial_report  # noqa: E402
from forge.evaluation.distributed_plan import DistributedEvaluationPlan  # noqa: E402
from forge.protocol.distributed_transport import report_digest  # noqa: E402
from forge.validator.dashboard_native_metrics import attach_native_metrics  # noqa: E402


def _plan(wallets) -> DistributedEvaluationPlan:
    prompts = tuple(
        {
            "prompt_id": f"p{number}",
            "text": f"diagnostic prompt {number}",
            "seed": number,
            "subject_class": "object",
            "camera_type": "moving",
            "style_family": "natural",
        }
        for number in range(2)
    )
    return DistributedEvaluationPlan(
        round_id="native-metrics-fixture",
        stage="diagnostic",
        entropy="fixture-entropy",
        policy_id="diagnostic-only",
        candidate_manifest_hash="candidate",
        king_manifest_hash="wan50",
        wan50_manifest_hash="wan50",
        committee={wallet.hotkey.ss58_address: 1 for wallet in wallets},
        prompts=prompts,
        timing_prompt_ids=(),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
        audit_work_count=2,
    )


def _raw(plan, prompt, hotkey, delta):
    def receipt(model, video):
        return {
            "status": "OK",
            "diagnostic_only": True,
            "round_id": plan.round_id,
            "validator_hotkey": hotkey,
            "video_sha256": video,
            "job": {
                "prompt_id": prompt["prompt_id"],
                "seed": prompt["seed"],
                "prompt": prompt["text"],
                "model_manifest_hash": model,
            },
        }

    axes = {
        axis: {"a": 3, "b": 3}
        for axis in (
            "prompt_adherence",
            "motion_adherence",
            "camera_adherence",
            "motion_quality",
        )
    }
    calls = [
        {
            "reference_is_a": first,
            "response_id": f"{hotkey}-{prompt['prompt_id']}-{first}",
            "answer": {"axes": axes},
        }
        for first in (True, False)
    ]
    reference, candidate = receipt("wan50", "a" * 64), receipt("candidate", "b" * 64)
    return {
        "pair": {
            "reference_receipt": reference,
            "candidate_receipt": candidate,
            "reference_sha256": reference["video_sha256"],
            "candidate_sha256": candidate["video_sha256"],
            "judgment": {"status": "OK", "uncertain_axes": [], "calls": calls},
        },
        "reference_metrics": {
            "subject_consistency": 0.8,
            "background_consistency": 0.8,
            "imaging_quality": 60.0,
            "motion_smoothness": 0.8,
            "dynamic_degree": 0.5,
        },
        "candidate_metrics": {
            "subject_consistency": 0.8,
            "background_consistency": 0.8,
            "imaging_quality": 60.0 + delta,
            "motion_smoothness": 0.8,
            "dynamic_degree": 0.5,
        },
    }


def test_native_musiq_is_replayed_per_prompt_before_the_dashboard_mean(tmp_path):
    wallets = []
    for name in ("a", "b"):
        wallet = bt.Wallet(name=name, hotkey="validator", path=str(tmp_path / "wallets"))
        wallet.create_new_hotkey(suppress=True, use_password=False, overwrite=False)
        wallets.append(wallet)
    plan = _plan(wallets)
    reports, raw_dir = [], tmp_path / "raw"
    raw_dir.mkdir()
    for wallet in wallets:
        rows = []
        for prompt, delta in zip(plan.prompts, (-10.0, -2.0), strict=True):
            raw = _raw(plan, prompt, wallet.hotkey.ss58_address, delta)
            row = fuse_diagnostic_quality(plan, prompt_id=prompt["prompt_id"], **raw)
            rows.append(row)
            (raw_dir / f"{row['raw_evidence_hash']}.json").write_text(json.dumps(raw))
        reports.append(
            signed_partial_report(
                plan, Sr25519Signer(wallet), quality_vs_king=rows, quality_vs_wan50=rows
            )
        )
    snapshot = {
        "evaluations": [
            {
                "plan_hash": plan.plan_hash(),
                "consensus": {
                    "reports": [
                        {
                            "validator": wallet.hotkey.ss58_address,
                            "status": "valid",
                            "report_sha256": report_digest(report),
                        }
                        for wallet, report in zip(wallets, reports, strict=True)
                    ]
                },
                "prompts": [{"id": prompt["prompt_id"]} for prompt in plan.prompts],
            }
        ]
    }
    augmented = attach_native_metrics(snapshot, [plan], reports, [raw_dir])
    native = augmented["evaluations"][0]["native_metrics"]
    assert native == {
        "metric": "MUSIQ",
        "unit": "points",
        "reference": "wan50",
        "mean_delta": -6.0,
        "min_delta": -10.0,
        "max_delta": -2.0,
        "unique_prompts": 2,
        "aggregation": "equal_prompt_mean_after_replica_mean",
    }
    assert [row["native_quality_delta"] for row in augmented["evaluations"][0]["prompts"]] == [
        -10.0,
        -2.0,
    ]
    with pytest.raises(ValueError, match="matching raw evidence"):
        attach_native_metrics(snapshot, [plan], reports, [])
