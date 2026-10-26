"""Synthetic local receipt validation; this does not prove GPU or API execution."""

import copy
import json

import pytest

from forge.evaluation import vbench_runner
from forge.evaluation.fusion import (
    evaluator_implementation_hash,
    fuse_bootstrap_alias_row,
    fuse_quality_row,
)
from forge.evaluation.vbench import normalize_native_score
from forge.protocol.hashing import hash_canonical, sha256_file


def _hash(number: int) -> str:
    return f"{number:064x}"


def _prompt():
    return {
        "prompt_id": "sealed-1",
        "text": "a red fox runs through snow",
        "seed": 7,
        "subject_class": "animal",
        "camera_type": "moving",
        "style_family": "natural",
    }


def _metrics(prompt, video, manifest):
    weights = {
        name: _hash(index + 20)
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
    return {
        "plan_id": "plan-1",
        "prompt_id": prompt["prompt_id"],
        "seed": prompt["seed"],
        "prompt_hash": hash_canonical("forge/mvp-prompt/v1", prompt["text"]),
        "video_sha256": video,
        "manifest_hash": manifest,
        "runner_source_sha256": sha256_file(vbench_runner.__file__),
        "provenance": {
            "implementation_commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490",
            "weight_sha256": weights,
            "runner_image_digest": "sha256:" + "a" * 64,
        },
        "scores": {name: 0.5 for name in weights},
    }


def _luna():
    axes = {
        axis: {"a": 2, "b": 2}
        for axis in ("prompt_adherence", "motion_adherence", "camera_adherence", "motion_quality")
    }
    return {
        "status": "OK",
        "uncertain_axes": [],
        "asserted_delta": 1.0,
        "calls": [
            {
                "response_id": "response-a",
                "reference_is_a": True,
                "model": "gpt-6-luna@pin",
                "answer": {"axes": axes},
            },
            {
                "response_id": "response-b",
                "reference_is_a": False,
                "model": "gpt-6-luna@pin",
                "answer": {"axes": axes},
            },
        ],
    }


def _inputs():
    prompt = _prompt()
    plan = {
        "plan_id": "plan-1",
        "prompts": [prompt],
        "candidate_manifest_hash": _hash(1),
        "incumbent_manifest_hash": _hash(2),
        "wan50_manifest_hash": _hash(3),
    }
    reference, candidate = (
        _metrics(prompt, _hash(4), _hash(2)),
        _metrics(prompt, _hash(5), _hash(1)),
    )
    provenance = {
        "source_implementation_hash": evaluator_implementation_hash(),
        "vbench_model_hashes": hash_canonical(
            "forge/vbench-weight-lock/v1", reference["provenance"]["weight_sha256"]
        ),
        "luna_evaluator_id": "gpt-6-luna@pin",
    }
    return plan, prompt, reference, candidate, provenance


def test_fusion_recomputes_raw_scores_and_binds_authenticated_slot_context():
    plan, prompt, reference, candidate, provenance = _inputs()
    contexts = []
    row = fuse_quality_row(
        plan=plan,
        prompt=prompt,
        reference_role="incumbent",
        reference_video_sha256=_hash(4),
        candidate_video_sha256=_hash(5),
        luna=_luna(),
        reference_metrics=reference,
        candidate_metrics=candidate,
        evaluator_provenance=provenance,
        verify_journal=lambda call, context: contexts.append((call, context)) is None,
    )
    assert row["orders"][0]["axes"]["prompt_adherence"] == 0.0
    assert row["prompt_hash"] == hash_canonical("forge/mvp-prompt/v1", prompt["text"])
    assert len(contexts) == 2 and contexts[0][1]["candidate_video_sha256"] == _hash(5)


def test_native_roundoff_receipt_survives_storage_and_quality_fusion(tmp_path):
    plan, prompt, reference, candidate, provenance = _inputs()
    # Observed in the real frozen-watch pilot. Preserve the native value while
    # feeding bounded producer scores through the ordinary consumer boundary.
    for receipt in (reference, candidate):
        receipt["raw_normalized_scores"] = {"subject_consistency": 1.0000001192092896}
        receipt["scores"]["subject_consistency"] = normalize_native_score(
            receipt["raw_normalized_scores"]["subject_consistency"]
        )
    path = tmp_path / "metric-receipts.json"
    path.write_text(json.dumps([reference, candidate]))
    reference, candidate = json.loads(path.read_text())
    inputs = dict(
        plan=plan,
        prompt=prompt,
        reference_role="incumbent",
        reference_video_sha256=_hash(4),
        candidate_video_sha256=_hash(5),
        luna=_luna(),
        reference_metrics=reference,
        candidate_metrics=candidate,
        evaluator_provenance=provenance,
        verify_journal=lambda call, context: True,
    )
    row = fuse_quality_row(**inputs)
    assert row["orders"][0]["axes"]["subject_consistency"] == 0.0
    assert candidate["raw_normalized_scores"]["subject_consistency"] > 1.0
    # Normalization belongs to the trusted producer; supplied out-of-range
    # receipts are still rejected by the consumer, not silently repaired.
    candidate["scores"]["subject_consistency"] = 1.0000001192092896
    with pytest.raises(ValueError):
        fuse_quality_row(**inputs)


def test_fusion_rejects_slot_swaps_and_luna_response_reuse():
    plan, prompt, reference, candidate, provenance = _inputs()

    def journal(call, context):
        return call["response_id"] == context["response_id"]

    bad_seed = copy.deepcopy(reference)
    bad_seed["seed"] = 8
    with pytest.raises(ValueError, match="frozen prompt slot"):
        fuse_quality_row(
            plan=plan,
            prompt=prompt,
            reference_role="incumbent",
            reference_video_sha256=_hash(4),
            candidate_video_sha256=_hash(5),
            luna=_luna(),
            reference_metrics=bad_seed,
            candidate_metrics=candidate,
            evaluator_provenance=provenance,
            verify_journal=journal,
        )
    used = set()
    kwargs = dict(
        plan=plan,
        prompt=prompt,
        reference_role="incumbent",
        reference_video_sha256=_hash(4),
        candidate_video_sha256=_hash(5),
        luna=_luna(),
        reference_metrics=reference,
        candidate_metrics=candidate,
        evaluator_provenance=provenance,
        verify_journal=journal,
        used_response_ids=used,
    )
    fuse_quality_row(**kwargs)
    with pytest.raises(ValueError, match="reused"):
        fuse_quality_row(**kwargs)


def test_bootstrap_alias_fuses_once_and_refuses_nonidentical_reference():
    plan, prompt, reference, candidate, provenance = _inputs()
    plan["wan50_manifest_hash"] = plan["incumbent_manifest_hash"]

    def journal(call, context):
        return call["response_id"] == context["response_id"]

    row = fuse_bootstrap_alias_row(
        plan=plan,
        prompt=prompt,
        reference_role="wan50",
        reference_video_sha256=_hash(4),
        candidate_video_sha256=_hash(5),
        luna=_luna(),
        reference_metrics=reference,
        candidate_metrics=candidate,
        evaluator_provenance=provenance,
        verify_journal=journal,
        used_response_ids=set(),
    )
    assert row["reference_roles"] == ["wan50", "incumbent"]
    assert row["comparison_id"]
    plan["wan50_manifest_hash"] = _hash(9)
    with pytest.raises(ValueError, match="identical"):
        fuse_bootstrap_alias_row(
            plan=plan,
            prompt=prompt,
            reference_video_sha256=_hash(4),
            candidate_video_sha256=_hash(5),
            luna=_luna(),
            reference_metrics=reference,
            candidate_metrics=candidate,
            evaluator_provenance=provenance,
            verify_journal=journal,
        )
