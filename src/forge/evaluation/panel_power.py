"""Offline adequacy checks for the frozen 128-prompt confirmation panel.

This is intentionally a diagnostic.  It never changes a policy, a threshold or
a candidate decision; it makes missing variance and multiplicity evidence
visible before a paid confirmation is scheduled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from scipy.stats import t

from forge.evaluation.mvp_policy import CONFIRMATION_PROMPTS, QUALITY_AXES


def confirmation_power_preflight(
    *,
    quality_margin: float,
    family_alpha: float = 0.05,
    references: int = 2,
) -> dict[str, float | int | str]:
    """Expose the precision the existing panel can and cannot support.

    The current decision inspects every global axis and every axis in each of
    the 16 protected cells against both references: 2 × 8 × (1 + 16) = 272
    lower bounds. It is an intersection-union noninferiority test: promotion
    requires every bound, so alpha=0.05 per bound already controls false
    promotion when any protected axis is truly below margin. The many bounds
    instead make false inconclusiveness likely near the margin; this preflight
    quantifies that power limitation without changing the policy.
    """

    if not 0 <= quality_margin < 1 or not 0 < family_alpha < 1 or references < 1:
        raise ValueError("panel power inputs are invalid")
    cells, per_cell = 16, CONFIRMATION_PROMPTS // 16
    bounds = references * len(QUALITY_AXES) * (1 + cells)
    per_bound_alpha = family_alpha
    # At the noninferiority boundary, one valid continuous test passes
    # with probability alpha, not 1-alpha. This illustration assumes all
    # bounds sit at that boundary and are independent; neither is measured.
    boundary_any_bound_failure = 1 - per_bound_alpha**bounds
    uncorrected_t = float(t.ppf(1 - per_bound_alpha, per_cell - 1))
    return {
        "confirmation_prompts": CONFIRMATION_PROMPTS,
        "protected_cells": cells,
        "prompts_per_cell": per_cell,
        "quality_bounds_examined": bounds,
        "historical_per_bound_alpha": per_bound_alpha,
        "intersection_union_false_promotion_alpha": per_bound_alpha,
        "all_bounds_at_boundary_independent_inconclusive_probability": (boundary_any_bound_failure),
        "boundary_probability_assumptions": (
            "Every true mean equals the negative margin; continuous valid tests; "
            "independent bounds. Illustrative only, not measured panel power."
        ),
        "historical_t_critical_df7": uncorrected_t,
        "max_cell_sd_for_zero_mean_to_clear_margin_historical": quality_margin
        * math.sqrt(per_cell)
        / uncorrected_t,
        "verdict": "INFORMATIONAL_ONLY_NO_POLICY_CHANGE",
    }


def probe_quality_information(probe: Mapping[str, Any]) -> dict[str, object]:
    """Describe what a small probe can tell us without inventing panel power."""

    controls = probe.get("controls")
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)):
        raise ValueError("probe controls are required")
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for row in controls:
        if isinstance(row, Mapping) and isinstance(row.get("kind"), str):
            by_kind.setdefault(row["kind"], []).append(row)
    blindspots = []
    disagreements = []
    for kind, rows in sorted(by_kind.items()):
        axes = [row.get("axes") for row in rows if isinstance(row.get("axes"), Mapping)]
        if axes and all(all(float(value) == 0 for value in item.values()) for item in axes):
            blindspots.append(kind)
        disagreements.extend(
            float(row["order_disagreement"])
            for row in rows
            if isinstance(row.get("order_disagreement"), (int, float))
        )
    return {
        "control_kinds_observed": sorted(by_kind),
        "largest_rows_per_kind": max((len(rows) for rows in by_kind.values()), default=0),
        "zero_signal_control_kinds": blindspots,
        "mean_order_disagreement": mean(disagreements) if disagreements else None,
        "panel_power_established": False,
        "reason": (
            "The probe has at most one observation per control kind and no eight-prompt "
            "protected-category sample; it cannot estimate confirmation variance or power."
        ),
    }
