#!/usr/bin/env python3
"""Describe decision-unit limits in an existing diagnostic outcome; no fitting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from forge.protocol.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.outcome.read_text())
    if raw.get("schema") != "forge/5090-probe-outcome/v1" or not isinstance(
        raw.get("controls"), list
    ):
        raise ValueError("requires a forge 5090 diagnostic outcome")
    controls = []
    for row in raw["controls"]:
        if not isinstance(row, dict) or not isinstance(row.get("axes"), dict):
            raise ValueError("diagnostic control is malformed")
        axes = row["axes"]
        native = row.get("native_metric_deltas")
        if not isinstance(native, dict):
            raise ValueError("diagnostic control lacks native metrics")
        controls.append(
            {
                "split": row.get("split"),
                "kind": row.get("kind"),
                "luna_min_axis_delta": min(axes.values()),
                "luna_max_absolute_axis_delta": max(abs(value) for value in axes.values()),
                "native_imaging_quality_delta": native.get("imaging_quality"),
                # Pinned VBench imaging_quality.py divides its per-video mean by100.
                "normalized_imaging_quality_delta": (
                    float(native["imaging_quality"]) / 100
                    if native.get("imaging_quality") is not None
                    else None
                ),
                "observable": row.get("observable"),
            }
        )
    development_legitimate = [
        row
        for row in controls
        if row["split"] == "development" and row["kind"] in {"self", "same_model_new_run"}
    ]
    mechanical_margin = max(
        (row["luna_max_absolute_axis_delta"] for row in development_legitimate), default=None
    )
    result = {
        "schema": "forge/calibration-decision-unit-diagnostic/v1",
        "outcome_sha256": sha256_file(args.outcome),
        "outcome_status": raw.get("status"),
        "evidence_limit": (
            "diagnostic controls have observable=null and one family per kind; no policy is fit"
        ),
        "candidate_gate_unit": {
            "rows": 128,
            "protected_cells": 16,
            "rows_per_cell": 8,
            "statistic": "one-sided t lower bound of prompt-level symmetric blind deltas",
        },
        "calibration_gate_unit": {
            "statistic": "per-control-row threshold crossing then binomial count",
            "required_held_out_rows_per_kind": 32,
            "current_rows_per_kind": dict(Counter(row["kind"] for row in controls)),
        },
        "development_legitimate_rows_present": development_legitimate,
        "invalid_mechanical_luna_margin_from_present_development_rows": mechanical_margin,
        "observed_controls": controls,
        "supported_findings": [
            (
                "The outcome cannot fit or qualify calibration: all controls are observable=null "
                "and each kind has one family."
            ),
            (
                "The candidate gate aggregates prompt-level deltas by global and protected-cell "
                "lower bounds, while calibration counts per-row crossings; their operating "
                "characteristics are not established as equivalent."
            ),
            (
                "Independent-seed variation is only in the independent diagnostic split and "
                "cannot be moved into development to set a margin."
            ),
            (
                "Mild blur had zero Luna axis deltas while its native imaging-quality delta was "
                "negative. Dividing the native MUSIQ delta by100 gives the pinned VBench "
                "image-quality axis delta, not a percentage of perceptual loss or a complete "
                "fused calibration receipt."
            ),
            (
                "More rows of a control that Luna does not observe do not establish sensitivity "
                "to that defect."
            ),
            (
                "The current fusion records Luna physical_plausibility, visual_quality, "
                "temporal_stability and Luna subject_consistency but does not put them in the "
                "eight protected candidate axes; VBench supplies proxies except physical "
                "plausibility."
            ),
        ],
        "not_identifiable_from_this_outcome": [
            (
                "Whether a single maximum legitimate margin would hide a mild systematic "
                "candidate regression on the candidate gate scale."
            ),
            (
                "A numerical conversion from the reported native imaging-quality deltas to the "
                "fused candidate image_quality axis."
            ),
            "128-prompt protected-cell power or a qualified false-alarm/sensitivity rate.",
        ],
        "decision_changing_experiment": {
            "freeze_first": (
                "reviewed explicit source bank, disjoint development/calibration/independent "
                "families, evaluator provenance and fusion code"
            ),
            "development": (
                "measure legitimate same-model/new-run/independent-seed controls and each "
                "degradation level on the candidate fused axis scale; fit before holdouts"
            ),
            "holdouts": (
                "run 32 observable families per control kind in calibration and independent "
                "splits, with the same two-order Luna plus VBench fusion used by candidates"
            ),
            "panel_alignment": (
                "include enough frozen confirmation-like prompts to estimate global and each "
                "8-prompt protected-cell lower-bound behavior; report inconclusive power rather "
                "than infer it from row hits"
            ),
            "go_no_go": (
                "enable qualification only if held-out controls separate on fused axes and the "
                "confirmation-like analysis shows the declared gate can distinguish legitimate "
                "variation from the targeted regression; otherwise remain disabled"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
