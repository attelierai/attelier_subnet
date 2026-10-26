"""Bind frozen panels, quality evidence and timing receipts to model identities."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, stdev
from typing import Any

from scipy.stats import t

from forge.evaluation.mvp_policy import (
    CONFIRMATION_PROMPTS,
    QUALITY_AXES,
    SCREEN_PROMPTS,
    EvaluationPolicy,
    _is_hash,
    _sha,
)
from forge.protocol.hashing import hash_canonical


def freeze_evaluation_plan(
    *,
    stage: str,
    prompts: Sequence[Mapping[str, Any]],
    policy: EvaluationPolicy,
    candidate_manifest_hash: str | None = None,
    incumbent_manifest_hash: str | None = None,
    wan50_manifest_hash: str | None = None,
) -> dict[str, object]:
    """Freeze a panel before its video, quality, or timing evidence exists."""

    policy.validate()
    if stage == "screening":
        if len(prompts) != SCREEN_PROMPTS:
            raise ValueError("screening plan must contain exactly 16 prompts")
    elif stage == "confirmation":
        validate_confirmation_panel(prompts)
    else:
        raise ValueError("MVP stage must be screening or confirmation")
    if any(not isinstance(row.get("prompt_id"), str) or not row["prompt_id"] for row in prompts):
        raise ValueError("evaluation plan prompts require ids")
    body = {
        "schema": "forge/mvp-evaluation-plan/v1",
        "stage": stage,
        "policy_id": policy.policy_id,
        "prompts": list(prompts),
        "candidate_manifest_hash": candidate_manifest_hash,
        "incumbent_manifest_hash": incumbent_manifest_hash,
        "wan50_manifest_hash": wan50_manifest_hash,
    }
    return {**body, "plan_id": _sha(body)}


def _as_records(records: Sequence[Mapping[str, Any]], label: str) -> list[Mapping[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError(f"{label} is required")
    return list(records)


def _text_hash(text: str) -> str:
    return hash_canonical("forge/mvp-prompt/v1", text)


def _plan_slots(prompts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    slots: dict[str, Mapping[str, Any]] = {}
    for prompt in prompts:
        prompt_id = prompt.get("prompt_id")
        if (
            not isinstance(prompt_id, str)
            or not isinstance(prompt.get("text"), str)
            or type(prompt.get("seed")) is not int
            or prompt_id in slots
        ):
            raise ValueError("MVP plan prompts require unique id, text and integer seed")
        slots[prompt_id] = prompt
    return slots


def _validate_quality_binding(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    *,
    plan_id: str,
    candidate_manifest_hash: str,
    reference_manifest_hash: str,
    bootstrap_alias: bool = False,
) -> None:
    slots = _plan_slots(prompts)
    if len(rows) != len(slots):
        raise ValueError("quality evidence does not cover the frozen prompt panel")
    seen: set[str] = set()
    for row in rows:
        prompt_id = row.get("prompt_id")
        prompt = slots.get(prompt_id)
        if prompt is None or prompt_id in seen:
            raise ValueError("quality row is not a unique frozen prompt slot")
        seen.add(prompt_id)
        if (
            row.get("seed") != prompt["seed"]
            or row.get("prompt_hash") != _text_hash(prompt["text"])
            or row.get("plan_id") != plan_id
            or any(
                row.get(key) != prompt.get(key)
                for key in ("subject_class", "camera_type", "style_family")
            )
            or row.get("candidate_manifest_hash") != candidate_manifest_hash
            or row.get("reference_manifest_hash") != reference_manifest_hash
            or not _is_hash(row.get("candidate_video_sha256"))
            or not _is_hash(row.get("reference_video_sha256"))
            or row.get("evidence_kind") != "validator_fused_quality"
            or not _is_hash(row.get("raw_evidence_hash"))
        ):
            raise ValueError("quality row differs from frozen prompt or model evidence")
        if bootstrap_alias:
            identity = {
                key: row.get(key)
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
            if row.get("reference_roles") != ["wan50", "incumbent"] or row.get(
                "comparison_id"
            ) != hash_canonical("forge/mvp-bootstrap-comparison/v1", identity):
                raise ValueError("bootstrap evidence alias is not a canonical signed comparison")
        elif "reference_roles" in row or "comparison_id" in row:
            raise ValueError("reference alias is only valid for exact bootstrap identity")


def _validate_timing_receipts(
    rows: Sequence[Mapping[str, Any]],
    allowed_prompt_ids: set[str],
    *,
    candidate_manifest_hash: str,
    reference_manifest_hash: str,
) -> None:
    for row in rows:
        if row.get("prompt_id") not in allowed_prompt_ids:
            raise ValueError("timing prompt is outside the frozen evaluation panel")
        if (
            row.get("candidate_manifest_hash") != candidate_manifest_hash
            or row.get("reference_manifest_hash") != reference_manifest_hash
            or row.get("gpu_synchronized") is not True
            or row.get("timing_scope") != "text_encode+denoise+decode"
            or not isinstance(row.get("hardware"), Mapping)
            or set(row["hardware"]) != {"gpu", "precision", "dependency_lock_hash"}
            or not isinstance(row.get("recipes"), Mapping)
            or set(row["recipes"]) != {"reference", "candidate"}
            or not isinstance(row.get("warmup"), Mapping)
            or row["warmup"].get("reference") is not True
            or row["warmup"].get("candidate") is not True
        ):
            raise ValueError("timing receipt lacks validator-owned runtime evidence")
        for run in row.get("runs", []):
            if (
                not isinstance(run, Mapping)
                or run.get("gpu_synchronized") is not True
                or not _is_hash(run.get("video_sha256"))
                or not _is_hash(run.get("recipe_hash"))
                or run.get("full_pipeline_seconds") != run.get("seconds")
            ):
                raise ValueError("timing run is not a synchronized full-pipeline receipt")


def validate_confirmation_panel(prompts: Sequence[Mapping[str, Any]]) -> None:
    """Require the frozen 4 x 2 x 2 geometry, eight prompts per cell."""

    rows = _as_records(prompts, "confirmation prompts")
    if len(rows) != CONFIRMATION_PROMPTS:
        raise ValueError("confirmation panel must contain exactly 128 prompts")
    cells: dict[tuple[str, str, str], int] = defaultdict(int)
    ids: set[str] = set()
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in ids:
            raise ValueError("confirmation prompt ids must be unique")
        ids.add(prompt_id)
        key = tuple(row.get(name) for name in ("subject_class", "camera_type", "style_family"))
        if any(not isinstance(part, str) or not part for part in key):
            raise ValueError("confirmation prompts need subject, camera and style categories")
        cells[key] += 1
    subjects = {key[0] for key in cells}
    cameras = {key[1] for key in cells}
    styles = {key[2] for key in cells}
    expected_cells = {
        (subject, camera, style) for subject in subjects for camera in cameras for style in styles
    }
    if (
        len(subjects) != 4
        or len(cameras) != 2
        or len(styles) != 2
        or set(cells) != expected_cells
        or set(cells.values()) != {8}
    ):
        raise ValueError("confirmation panel must have eight prompts in each of 16 cells")


def symmetric_axes(orders: object) -> dict[str, float]:
    """Recompute a blind comparison from its two opposing orientations."""

    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)) or len(orders) != 2:
        raise ValueError("two blinded score orders are required")
    if {row.get("reference_first") for row in orders if isinstance(row, Mapping)} != {True, False}:
        raise ValueError("both score orientations are required")
    result: dict[str, float] = {}
    for axis in QUALITY_AXES:
        values = []
        for row in orders:
            axes = row.get("axes") if isinstance(row, Mapping) else None
            value = axes.get(axis) if isinstance(axes, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not -1 <= value <= 1
            ):
                raise ValueError("blind score axis is invalid")
            values.append(float(value))
        result[axis] = mean(values)
    return result


def _quality_rows(
    rows: Sequence[Mapping[str, Any]], expected_count: int
) -> tuple[
    dict[str, float],
    dict[tuple[str, str, str], dict[str, float]],
    dict[str, float],
    dict[tuple[str, str, str], dict[str, float]],
]:
    rows = _as_records(rows, "quality judgments")
    if len(rows) != expected_count:
        raise ValueError(f"quality evidence needs exactly {expected_count} rows")
    totals = {axis: [] for axis in QUALITY_AXES}
    cells: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {axis: [] for axis in QUALITY_AXES}
    )
    prompt_ids: set[str] = set()
    for row in rows:
        prompt_id = row.get("prompt_id")
        orders = row.get("orders")
        cell = tuple(row.get(name) for name in ("subject_class", "camera_type", "style_family"))
        if not isinstance(prompt_id, str) or prompt_id in prompt_ids:
            raise ValueError("quality rows need unique prompt ids and axis deltas")
        if any(not isinstance(part, str) or not part for part in cell):
            raise ValueError("quality rows need protected category labels")
        if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)) or len(orders) != 2:
            raise ValueError("quality rows need two blind orders")
        if {order.get("reference_first") for order in orders if isinstance(order, Mapping)} != {
            True,
            False,
        }:
            raise ValueError("quality rows need both blind orientations")
        prompt_ids.add(prompt_id)
        axes = symmetric_axes(orders)
        for axis, value in axes.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not -1 <= value <= 1
            ):
                raise ValueError("quality delta is outside [-1, 1]")
            totals[axis].append(float(value))
            cells[cell][axis].append(float(value))

    def lower(values: list[float]) -> float:
        if len(set(values)) == 1:
            return values[0]
        return mean(values) - t.ppf(0.95, len(values) - 1) * stdev(values) / math.sqrt(len(values))

    return (
        {axis: mean(values) for axis, values in totals.items()},
        {
            key: {axis: mean(values) for axis, values in values.items()}
            for key, values in cells.items()
        },
        {axis: lower(values) for axis, values in totals.items()},
        {
            key: {axis: lower(values) for axis, values in values.items()}
            for key, values in cells.items()
        },
    )
