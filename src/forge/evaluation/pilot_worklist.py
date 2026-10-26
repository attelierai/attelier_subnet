"""Concrete, deterministic development-pilot jobs; no rendering or provider calls."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from forge.evaluation.calibration_campaign import pilot_execution_ledger
from forge.evaluation.vbench import METRICS
from forge.protocol.hashing import hash_canonical


def pilot_worklist(pilot: Mapping[str, Any], *, warmup_generations: int = 2) -> dict[str, Any]:
    """Materialize all jobs so the preflight counts executable identities.

    The base identity survives inclusion in development. A new execution uses
    the same seed with a different job ID; a new seed is derived from the base,
    never selected after judging. Warmups are explicitly outside observations.
    """
    ledger = pilot_execution_ledger(pilot)
    if type(warmup_generations) is not int or warmup_generations < 1:
        raise ValueError("pilot needs a positive, explicit warmup count")
    bases = {row["source_id"]: row for row in pilot["slots"]}
    renders: dict[str, dict[str, Any]] = {}
    references: dict[str, str] = {}

    def render(source: Mapping[str, Any], role: str) -> str:
        seed = source["seed"]
        if role == "independent_seed":
            seed = int(
                hash_canonical("forge/pilot-independent-seed/v1", source["base_id"])[:15], 16
            )
            if seed == source["seed"]:
                seed = (seed + 1) % (2**60)
        body = {
            "base_id": source["base_id"],
            "source_id": source["source_id"],
            "source_family": source["source_family"],
            "prompt_text": source["prompt_text"],
            "prompt_hash": source["prompt_hash"],
            "seed": seed,
            "role": role,
        }
        job_id = hash_canonical("forge/evaluator-pilot-render/v1", body)
        renders.setdefault(job_id, {"job_id": job_id, **body})
        return job_id

    for source_id, source in sorted(bases.items()):
        references[source_id] = render(source, "reference")
    transforms, comparisons, calls = [], [], []
    for slot in pilot["slots"]:
        kind, reference = slot["kind"], references[slot["source_id"]]
        if kind in {"self", "static_scene"}:
            candidate = reference
        elif kind in {"same_model_new_run", "independent_seed"}:
            candidate = render(slot, kind)
        elif kind == "wrong_video":
            candidate = references[slot["donor_source_id"]]
        else:
            options = dict(slot["control_spec"])
            if kind == "shuffle":
                indices = list(range(81))
                random.Random(int(reference[:16], 16)).shuffle(indices)
                if indices == list(range(81)):
                    indices = indices[1:] + indices[:1]
                options["frame_indices"] = indices
            body = {"reference_job_id": reference, "kind": kind, "options": options}
            candidate = hash_canonical("forge/evaluator-pilot-transform/v1", body)
            transforms.append({"job_id": candidate, **body})
        comparisons.append(
            {
                "slot_id": slot["slot_id"],
                "kind": kind,
                "reference_job_id": reference,
                "candidate_job_id": candidate,
                "observability": "REQUIRES_PRE_SCORE_MEDIA_REVIEW",
            }
        )
        for reference_first in (True, False):
            body = {
                "slot_id": slot["slot_id"],
                "reference_first": reference_first,
                "a_job_id": reference if reference_first else candidate,
                "b_job_id": candidate if reference_first else reference,
            }
            calls.append({"call_id": hash_canonical("forge/evaluator-pilot-call/v1", body), **body})
    if (len(renders), len(transforms), len(comparisons), len(calls)) != (32, 56, 96, 192):
        raise ValueError("materialized pilot jobs disagree with the declared work")
    first_reference = renders[next(iter(references.values()))]
    warmups = [
        {
            **first_reference,
            "job_id": hash_canonical(
                "forge/evaluator-pilot-warmup/v1", first_reference["job_id"], index
            ),
            "role": "warmup",
            "measurement_eligible": False,
        }
        for index in range(warmup_generations)
    ]
    body = {
        "schema": "forge/evaluator-pilot-worklist/v1",
        "pilot_id": pilot["pilot_id"],
        "recipe": {
            "base_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "base_model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
            "steps": 50,
            "scheduler": "original_unipc",
            "guidance_scale": 5.0,
            "width": 832,
            "height": 480,
            "frames": 81,
            "fps": 16,
            "scheduler_shift": 3.0,
        },
        "ledger": ledger,
        "renders": list(renders.values()),
        "warmups": warmups,
        "transforms": transforms,
        "comparisons": comparisons,
        "blind_calls": calls,
        "metric_jobs": [
            {"video_job_id": job["job_id"], "metrics": list(METRICS)}
            for job in [*renders.values(), *transforms]
        ],
        "dispatch_authorized": False,
        "qualification": "prohibited",
    }
    return {**body, "worklist_id": hash_canonical("forge/evaluator-pilot-worklist/v1", body)}
