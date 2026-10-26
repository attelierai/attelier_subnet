"""Freeze and verify the evaluator-pilot pre-score observability boundary.

This is deliberately evidence plumbing, not a perceptual evaluator.  A PASS
only says every frozen comparison has a pre-score media review declaring it
observable under its declared treatment.  It never establishes that the
control is perceptually meaningful or that the fusion will separate it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forge.evaluation.calibration_campaign import pilot_execution_ledger
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.protocol.hashing import hash_canonical

PROTOCOL_SCHEMA = "forge/evaluator-pilot-observability-protocol/v1"
RECEIPT_SCHEMA = "forge/evaluator-pilot-observability-receipt/v1"

# Frozen instructions for a reviewer who has no score access. These are not
# claims that a real clip passes its rule.
PERCEPTUAL_RULES: dict[str, dict[str, str]] = {
    "self": {
        "applicability": "same frozen media job",
        "observable_when": "media identity is confirmed",
        "reject_when": "a distinct job or digest is substituted",
    },
    "same_model_new_run": {
        "applicability": "same prompt and seed, separate run",
        "observable_when": "both clips remain equivalent on protected quality axes",
        "reject_when": "a real quality difference is treated as evaluator noise",
    },
    "independent_seed": {
        "applicability": "same prompt and preregistered distinct seed",
        "observable_when": (
            "independent review establishes equivalent protected quality despite composition "
            "differences; sharing a model or prompt alone does not establish equivalence"
        ),
        "reject_when": (
            "either clip has a meaningful prompt, motion or quality difference; retain it as "
            "generation variation, not an equivalence control or evaluator false alarm"
        ),
    },
    "static_scene": {
        "applicability": "static-camera source with no requested action",
        "observable_when": "absence of movement remains legitimate",
        "reject_when": "static content is penalized for lacking motion",
    },
    "freeze": {
        "applicability": "reference requests meaningful subject or camera motion",
        "observable_when": "visible reference progression is removed by the candidate",
        "reject_when": "reference lacks meaningful motion or pixels alone are evidence",
    },
    "blur_mild": {
        "applicability": "frozen same-content mild Gaussian treatment",
        "observable_when": "blur is perceptible and content remains comparable",
        "reject_when": "a changed digest without perceptible blur is evidence",
    },
    "blur_severe": {
        "applicability": "frozen same-content severe Gaussian treatment",
        "observable_when": "blur is perceptible and content remains comparable",
        "reject_when": "a changed digest without perceptible blur is evidence",
    },
    "shuffle": {
        "applicability": "meaningful requested motion and frozen nonidentity permutation",
        "observable_when": "temporal progression has perceptible order disruption",
        "reject_when": "no-motion source or permutation alone is evidence",
    },
    "wrong_video": {
        "applicability": "bound distinct source and donor lineages",
        "observable_when": "donor is perceptibly inconsistent with the source prompt",
        "reject_when": "distinct lineage alone is evidence",
    },
    "detail_loss": {
        "applicability": "frozen same-content downscale/upscale treatment",
        "observable_when": "visible detail loss with comparable content",
        "reject_when": "transform or pixels alone are evidence",
    },
    "evaluator_attack_mild": {
        "applicability": "frozen mild temporal-luma treatment",
        "observable_when": "temporal photometric artifact is perceptible",
        "reject_when": "numeric luma offset or pixels alone are evidence",
    },
    "evaluator_attack_severe": {
        "applicability": "frozen severe temporal-luma treatment",
        "observable_when": "temporal photometric artifact is perceptible",
        "reject_when": "numeric luma offset or pixels alone are evidence",
    },
}


def _sha(body: object) -> str:
    return hash_canonical("forge/evaluator-pilot-observability/v1", body)


def freeze_pilot_observability_protocol(
    pilot: Mapping[str, Any], worklist: Mapping[str, Any]
) -> dict[str, object]:
    """Bind all 96 pre-score reviews to the frozen pilot and worklist."""

    ledger = pilot_execution_ledger(pilot)
    expected_worklist = pilot_worklist(pilot)
    if dict(worklist) != expected_worklist:
        raise ValueError("observability requires the exact frozen pilot worklist")
    slot_by_id = {slot["slot_id"]: slot for slot in pilot["slots"]}
    requirements = []
    for pair in worklist["comparisons"]:
        slot = slot_by_id.get(pair["slot_id"])
        if slot is None:
            raise ValueError("worklist comparison is outside the frozen pilot")
        lineage = {
            "source_id": slot["source_id"],
            "source_family": slot["source_family"],
            "concept_id": slot["concept_id"],
            "concept_family": slot["concept_family"],
            "donor_source_id": slot.get("donor_source_id"),
            "donor_source_family": slot.get("donor_source_family"),
            "donor_concept_id": slot.get("donor_concept_id"),
            "donor_concept_family": slot.get("donor_concept_family"),
        }
        treatment = {
            "kind": slot["kind"],
            "control_spec": slot["control_spec"],
            "reference_job_id": pair["reference_job_id"],
            "candidate_job_id": pair["candidate_job_id"],
        }
        requirements.append(
            {
                "slot_id": slot["slot_id"],
                "kind": slot["kind"],
                "reference_job_id": pair["reference_job_id"],
                "candidate_job_id": pair["candidate_job_id"],
                "lineage_hash": _sha(lineage),
                "treatment_hash": _sha(treatment),
                "perceptual_rule_id": _sha(
                    {"kind": slot["kind"], "rule": PERCEPTUAL_RULES[slot["kind"]]}
                ),
            }
        )
    if len(requirements) != 96 or len({row["slot_id"] for row in requirements}) != 96:
        raise ValueError("observability protocol needs every frozen pilot comparison")
    body = {
        "schema": PROTOCOL_SCHEMA,
        "pilot_id": ledger["pilot_id"],
        "worklist_id": worklist["worklist_id"],
        "review_stage": "PRE_SCORE_MEDIA_REVIEW",
        "score_access_boundary": "before_luna_fusion_or_metric_score_access",
        "perceptual_rules": PERCEPTUAL_RULES,
        "requirements": requirements,
        "judging_permitted_only_after": "complete_observable_receipt",
        "qualification": "prohibited",
    }
    return {**body, "protocol_id": _sha(body)}


def validate_pilot_observability_receipt(
    protocol: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    pilot: Mapping[str, Any] | None = None,
    worklist: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Fail closed unless every planned comparison has reviewable media.

    Review rows are intentionally score-free.  They bind media digests and the
    frozen lineage/treatment hashes before Luna or fused scores are available.
    """

    if not isinstance(protocol, Mapping):
        raise ValueError("observability protocol is not canonically frozen")
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != _sha(protocol_body)
        or protocol_body.get("review_stage") != "PRE_SCORE_MEDIA_REVIEW"
        or protocol_body.get("score_access_boundary") != "before_luna_fusion_or_metric_score_access"
        or protocol_body.get("perceptual_rules") != PERCEPTUAL_RULES
        or protocol_body.get("qualification") != "prohibited"
        or not isinstance(protocol_body.get("requirements"), list)
    ):
        raise ValueError("observability protocol is not canonically frozen")
    if (pilot is None) != (worklist is None):
        raise ValueError("pilot and worklist must be supplied together for protocol recomputation")
    if pilot is not None and dict(protocol) != freeze_pilot_observability_protocol(pilot, worklist):
        raise ValueError("observability protocol differs from the frozen pilot worklist")
    requirements = {row.get("slot_id"): row for row in protocol_body["requirements"]}
    if len(requirements) != 96 or None in requirements:
        raise ValueError("observability protocol has incomplete comparison coverage")
    if not isinstance(receipt, Mapping):
        raise ValueError("observability receipt is not canonically frozen")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    required_fields = {
        "schema",
        "protocol_id",
        "pilot_id",
        "worklist_id",
        "media",
        "reviews",
        "judging_permitted",
        "qualification",
    }
    if (
        set(body) != required_fields
        or receipt.get("receipt_id") != _sha(body)
        or body["schema"] != RECEIPT_SCHEMA
        or body["protocol_id"] != protocol["protocol_id"]
        or body["pilot_id"] != protocol["pilot_id"]
        or body["worklist_id"] != protocol["worklist_id"]
        or body["qualification"] != "prohibited"
        or body["judging_permitted"] is not True
        or not isinstance(body["media"], list)
        or not isinstance(body["reviews"], list)
    ):
        raise ValueError("observability receipt is not canonically frozen")
    media_by_job: dict[str, str] = {}
    for media in body["media"]:
        if not isinstance(media, Mapping) or set(media) != {"job_id", "sha256"}:
            raise ValueError("observability media fields are not exact")
        job_id, digest = media.get("job_id"), media.get("sha256")
        if (
            not isinstance(job_id, str)
            or job_id in media_by_job
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("observability media is invalid")
        media_by_job[job_id] = digest
    reviews_by_slot: dict[str, Mapping[str, Any]] = {}
    review_fields = {
        "slot_id",
        "kind",
        "reference_job_id",
        "candidate_job_id",
        "lineage_hash",
        "treatment_hash",
        "perceptual_rule_id",
        "reference_sha256",
        "candidate_sha256",
        "observable",
        "review_status",
    }
    for review in body["reviews"]:
        if not isinstance(review, Mapping) or set(review) != review_fields:
            raise ValueError("observability review fields are not exact")
        slot_id = review.get("slot_id")
        expected = requirements.get(slot_id)
        if expected is None or slot_id in reviews_by_slot:
            raise ValueError("observability review is outside the frozen protocol")
        for field in (
            "kind",
            "reference_job_id",
            "candidate_job_id",
            "lineage_hash",
            "treatment_hash",
            "perceptual_rule_id",
        ):
            if review.get(field) != expected[field]:
                raise ValueError("observability review differs from frozen lineage or treatment")
        if (
            review.get("review_status") != "PRE_SCORE_COMPLETE"
            or review.get("observable") is not True
            or review.get("reference_sha256") != media_by_job.get(review["reference_job_id"])
            or review.get("candidate_sha256") != media_by_job.get(review["candidate_job_id"])
        ):
            raise ValueError("observability review lacks verified observable media")
        reviews_by_slot[slot_id] = review
    if set(reviews_by_slot) != set(requirements):
        raise ValueError("observability receipt does not cover every frozen comparison")
    required_jobs = {
        job_id
        for requirement in requirements.values()
        for job_id in (requirement["reference_job_id"], requirement["candidate_job_id"])
    }
    if set(media_by_job) != required_jobs:
        raise ValueError("observability receipt media does not match frozen comparisons")
    return {
        "status": "PASS",
        "pilot_id": protocol["pilot_id"],
        "protocol_id": protocol["protocol_id"],
        "observable_comparisons": len(reviews_by_slot),
        "judging_permitted": True,
        "qualification": "prohibited",
    }
