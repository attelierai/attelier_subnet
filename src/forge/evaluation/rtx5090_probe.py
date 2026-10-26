"""Create a small, explicitly non-qualifying Wan evaluator probe manifest.

The manifest is deliberately cheap enough for a hardware smoke on one RTX 5090.
It covers the evaluator's known control families, but has one source per split
and control: it can discover a broken runner or judge and can *never* validate
the 32-independent-family calibration required for promotion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from forge.evaluation.mvp_policy import CONTROL_AXES, CONTROL_KINDS, QUALITY_AXES, _sha

_SOURCES = (
    (
        "development",
        "A ceramicist turns a small blue bowl slowly on a wheel, close camera follows the hands.",
        "person-action",
    ),
    (
        "calibration",
        "A red tram stands at a quiet rain-slick station; locked wide camera, "
        "only rain ripples move.",
        "static-legitimate",
    ),
    (
        "independent",
        "A snow leopard crosses a rocky ridge as the camera pans from its paws to its face.",
        "animal-motion",
    ),
)

_TRANSFORMS = {
    "freeze": {
        "severity": "severe",
        "options": {"frame_index": 0},
        "eligible": "requested_motion",
        "protected_axes": ("action_adherence", "temporal_smoothness", "dynamic_degree"),
    },
    "blur_mild": {
        "severity": "mild",
        "options": {"gaussian_radius_px": 1.0},
        "eligible": "any",
        "protected_axes": ("image_quality",),
    },
    "blur_severe": {
        "severity": "severe",
        "options": {"gaussian_radius_px": 4.0},
        "eligible": "any",
        "protected_axes": ("image_quality",),
    },
    "shuffle": {
        "severity": "severe",
        "options": {"permutation": "seeded_nonidentity"},
        "eligible": "requested_motion",
        "protected_axes": ("temporal_smoothness",),
    },
    "wrong_video": {
        "severity": "severe",
        "options": {"replacement": "different_frozen_prompt"},
        "eligible": "different_prompt",
        "protected_axes": ("prompt_adherence",),
    },
    "detail_loss": {
        "severity": "severe",
        "options": {"downscale_factor": 4, "upscale": "nearest"},
        "eligible": "any",
        "protected_axes": ("image_quality", "subject_consistency"),
    },
    "evaluator_attack_mild": {
        "severity": "mild",
        "options": {"temporal_luma_offset": 3, "period_frames": 2},
        "eligible": "any",
        "protected_axes": ("prompt_adherence", "image_quality"),
    },
    "evaluator_attack_severe": {
        "severity": "severe",
        "options": {"temporal_luma_offset": 16, "period_frames": 2},
        "eligible": "any",
        "protected_axes": ("prompt_adherence", "image_quality"),
    },
}


def _id(campaign_id: str, split: str, label: str) -> str:
    return _sha(
        {
            "domain": "forge/rtx5090-probe-slot/v1",
            "campaign_id": campaign_id,
            "split": split,
            "label": label,
        }
    )[:20]


def build_probe_manifest(campaign_id: str) -> dict[str, object]:
    """Return a sealed-work manifest for 47 media files and two ABBA prompts."""

    if not campaign_id or not isinstance(campaign_id, str):
        raise ValueError("campaign_id is required")
    sources = []
    controls = []
    for number, (split, text, family) in enumerate(_SOURCES):
        source_id = _id(campaign_id, split, "source")
        concept_id = _id(campaign_id, split, "concept")
        sources.append(
            {
                "split": split,
                "prompt_id": _id(campaign_id, split, "prompt"),
                "source_id": source_id,
                "concept_id": concept_id,
                "source_family": f"probe-{split}-{family}",
                "concept_family": f"probe-{split}-{family}",
                "text": text,
                "seed": 91_700 + number,
                "subject_class": ("human", "object", "animal")[number],
                "camera_type": ("moving", "static", "moving")[number],
                "style_family": "natural",
            }
        )
    source_by_split = {row["split"]: row for row in sources}
    control_sources = {
        "self": "development",
        "same_model_new_run": "development",
        "independent_seed": "independent",
        "static_scene": "calibration",
        "freeze": "development",
        "blur_mild": "development",
        "blur_severe": "independent",
        "shuffle": "independent",
        "wrong_video": "independent",
        "detail_loss": "development",
        "evaluator_attack_mild": "independent",
        "evaluator_attack_severe": "development",
    }
    for kind in CONTROL_KINDS:
        source = source_by_split[control_sources[kind]]
        controls.append(
            {
                "split": source["split"],
                "kind": kind,
                "source_id": source["source_id"],
                "concept_id": source["concept_id"],
                "source_family": source["source_family"],
                "concept_family": source["concept_family"],
                "expected_protected_axes": sorted(CONTROL_AXES.get(kind, set())),
                "observable": None,
                "observability_rule": (
                    "judge after rendering; a pixel hash difference alone is not observability"
                ),
            }
        )
    return {
        "schema": "forge/rtx5090-evaluator-probe/v1",
        "campaign_id": campaign_id,
        "promotion_eligible": False,
        "expected_gate": "INCONCLUSIVE",
        "reason": (
            "One source per split/control is a runner and sensitivity probe, not "
            "the 32 independent source/concept families per control required by mvp_calibration."
        ),
        "hardware": {
            "required_gpu": "RTX 5090 (32 GB)",
            "concurrency": 1,
            "timing_rule": (
                "no concurrent render during ABBA; synchronize before every measured run"
            ),
            "quality_rule": (
                "record text encode, denoise, VAE decode, video SHA-256 and exact recipe"
            ),
        },
        "evaluator": {
            "quality_axes": list(QUALITY_AXES),
            "blind_orders_per_comparison": 2,
            "required_native_metrics": [
                "subject_consistency",
                "background_consistency",
                "imaging_quality",
                "motion_smoothness",
                "dynamic_degree",
            ],
        },
        "sources": sources,
        "control_observations": controls,
        "render_recipes": {
            "wan50": {"steps": 50, "scheduler": "original_unipc", "guidance_scale": 5.0},
            "fast40": {"steps": 40, "scheduler": "uniform_euler", "guidance_scale": 5.0},
            "poor1": {"steps": 1, "scheduler": "uniform_euler", "guidance_scale": 1.0},
        },
        "transform_definitions": _TRANSFORMS,
        "media_budget": {
            "quality_media_files": 39,
            "quality_gpu_renders": 15,
            "derived_transformations": 8,
            "timing_media_files": 8,
            "total_media_files": 31,
            "explanation": (
                "Fifteen quality renders (five per source), eight targeted derived transformations "
                "and two separate ABBA timing quartets. Warmups are recorded separately "
                "by the runtime."
            ),
        },
        "timing": {
            "prompt_ids": [sources[0]["prompt_id"], sources[2]["prompt_id"]],
            "runs_per_prompt": ["reference", "candidate", "candidate", "reference"],
            "diagnostic_only": True,
            "reason": "Two prompts cannot establish the 32-prompt speed lower confidence bound.",
        },
        "acceptance": {
            "must_show": [
                "identical self comparison does not create a systematic regression",
                "legitimately static scene is not called frozen motion",
                "at least one deliberately degraded control is flagged on its protected axis",
                "all two-order Luna comparisons and VBench provenance are retained",
            ],
            "must_not_claim": [
                "calibrated sensitivity or false-alarm rate",
                "quality preservation of fast40",
                "promotion eligibility or a new king",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_probe_manifest(args.campaign_id)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with os.fdopen(os.open(args.out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as stream:
        stream.write(encoded)
    print(json.dumps({"manifest_hash": _sha(payload), "promotion_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
