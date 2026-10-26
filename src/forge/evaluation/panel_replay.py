"""Conditional power diagnostic from preregistered source-level controls."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping

from forge.evaluation.calibration_campaign import bind_calibration_rows
from forge.evaluation.mvp_calibration import validate_calibration
from forge.evaluation.mvp_evidence import _quality_rows
from forge.evaluation.mvp_policy import CONFIRMATION_PROMPTS, EvaluationPolicy, _sha


def conditional_panel_replay(
    calibration: Mapping, policy: EvaluationPolicy, *, draws: int = 64
) -> dict:
    """Replay the existing quality rule, preserving within-source axis correlation.

    Resampled copies are Monte Carlo inputs, never new independent observations.
    This estimates neither two-reference joint power nor promotion eligibility.
    """
    if type(draws) is not int or not 1 <= draws <= 1024:
        raise ValueError("conditional replay needs 1..1024 diagnostic draws")
    if calibration.get("policy_id") != policy.policy_id:
        raise ValueError("conditional replay policy differs from control evidence")
    campaign, rows = calibration.get("campaign"), calibration.get("rows")
    if not isinstance(campaign, Mapping) or campaign.get("schema") not in {
        "forge/mvp-calibration-campaign/v3",
        "forge/mvp-calibration-campaign/v4",
    }:
        raise ValueError("conditional replay requires preregistered v3 or v4 categories")
    binding = bind_calibration_rows(campaign, rows)
    source_calibration = validate_calibration(calibration, policy)
    if source_calibration["status"] == "INCOMPLETE":
        raise ValueError("conditional replay requires complete, hash-bound control evidence")
    if any(row.get("observable") is not True for row in rows):
        raise ValueError("unobservable controls cannot estimate conditional panel power")
    grouped = defaultdict(lambda: defaultdict(list))
    labels = ("subject_class", "camera_type", "style_family")
    for row in rows:
        if row["split"] in {"calibration", "independent"}:
            grouped[(row["split"], row["kind"])][tuple(row[name] for name in labels)].append(row)
    seed = int(_sha({"binding": binding, "policy_id": policy.policy_id})[:16], 16)
    generator = random.Random(seed)
    reports = []
    for (split, kind), cells in sorted(grouped.items()):
        if len(cells) != 16 or any(len(sources) != 2 for sources in cells.values()):
            raise ValueError("conditional replay requires two frozen sources per protected cell")
        outcomes = Counter()
        for draw in range(draws):
            panel = []
            for cell, sources in sorted(cells.items()):
                for position, row in enumerate(generator.choices(sources, k=8)):
                    panel.append(
                        {
                            **row,
                            "prompt_id": f"simulation:{draw}:{cell}:{position}",
                        }
                    )
            means, cell_means, lower, cell_lower = _quality_rows(panel, CONFIRMATION_PROMPTS)
            threshold = -policy.quality_margin
            mean_values = [
                *means.values(),
                *(v for axes in cell_means.values() for v in axes.values()),
            ]
            lower_values = [
                *lower.values(),
                *(v for axes in cell_lower.values() for v in axes.values()),
            ]
            if min(mean_values) < threshold:
                outcomes["FAIL"] += 1
            elif min(lower_values) < threshold:
                outcomes["INCONCLUSIVE"] += 1
            else:
                outcomes["QUALITY_CLEAR"] += 1
        reports.append(
            {
                "split": split,
                "kind": kind,
                "independent_source_families": 32,
                "sources_per_cell": 2,
                "simulated_draws": draws,
                "outcomes": {
                    key: outcomes[key] for key in ("FAIL", "INCONCLUSIVE", "QUALITY_CLEAR")
                },
                "conditional_quality_clear_fraction": outcomes["QUALITY_CLEAR"] / draws,
            }
        )
    return {
        "campaign_id": campaign["campaign_id"],
        "evidence_hash": binding["evidence_hash"],
        "seed": seed,
        "draws_per_control": draws,
        "controls": reports,
        "source_calibration": source_calibration,
        "promotion_enabled": False,
        "panel_power_established": False,
        "assumptions": [
            "Empirical within-cell distribution from only two independent source families.",
            "Whole-source axes are resampled together; cells are sampled independently.",
            "Repeated draws are not independent measured panels or additional source families.",
            "Conditional diagnostic of one reference comparison; no joint king/Wan-50 power claim.",
            "No threshold fitting, speed decision or certification follows this report.",
        ],
    }
