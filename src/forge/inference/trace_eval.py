"""Evidence-only analysis of captured H3 denoise tensors.

This module deliberately tests a narrow claim: whether captured callback-state
trajectories on a fixed input separate known honest repeats from supplied
adversarial runs.
It neither trusts a worker's labels nor establishes that a trace produced the
served media, that a checkpoint ran remotely, or that a GPU is owned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "forge/inference-trace-experiment/v1"
REPORT_SCHEMA = "forge/inference-trace-analysis/v1"


class TraceEvidenceError(ValueError):
    """Evidence that is malformed or cannot safely be interpreted."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceEvidenceError("experiment_json_invalid") from exc
    if not isinstance(value, dict):
        raise TraceEvidenceError("experiment_must_be_object")
    return value


def _tensor(root: Path, item: dict[str, Any]) -> np.ndarray:
    required = {"tensorPath", "sha256", "shape", "dtype"}
    if not required <= set(item) or not isinstance(item["tensorPath"], str):
        raise TraceEvidenceError("tensor_descriptor_invalid")
    path = (root / item["tensorPath"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise TraceEvidenceError("tensor_path_escapes_experiment") from exc
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise TraceEvidenceError("tensor_custody_invalid")
    try:
        if path.suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif path.suffix == ".pt":
            try:
                import torch
            except ImportError as exc:
                raise TraceEvidenceError("torch_required_for_pt_trace") from exc
            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, torch.Tensor):
                observed_dtype = str(value.dtype).removeprefix("torch.")
                if observed_dtype != item["dtype"] or not torch.isfinite(value).all().item():
                    raise TraceEvidenceError("tensor_dtype_or_values_invalid")
                # NumPy cannot represent bfloat16.  Preserve and check the raw
                # dtype above, then convert only for numerical diagnostics.
                value = value.detach().to(torch.float32).numpy()
        else:
            raise TraceEvidenceError("tensor_format_unsupported")
    except TraceEvidenceError:
        raise
    except Exception as exc:
        raise TraceEvidenceError("tensor_load_invalid") from exc
    if not isinstance(value, np.ndarray) or list(value.shape) != item["shape"]:
        raise TraceEvidenceError("tensor_shape_invalid")
    if path.suffix != ".pt" and (str(value.dtype) != item["dtype"] or not np.isfinite(value).all()):
        raise TraceEvidenceError("tensor_dtype_or_values_invalid")
    return value.astype(np.float64, copy=False)


def _components(root: Path, values: Any, step: int) -> dict[str, np.ndarray]:
    if not isinstance(values, list) or not values:
        raise TraceEvidenceError("capture_components_invalid")
    result: dict[str, np.ndarray] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("component"), str):
            raise TraceEvidenceError("capture_component_invalid")
        key = f"{step}:{value['component']}"
        if key in result:
            raise TraceEvidenceError("capture_component_duplicate")
        result[key] = _tensor(root, value)
    return result


def _capture_from_binding(root: Path, item: Any) -> tuple[dict[str, Any], Path]:
    """Read an immutable capture manifest with analyst-side role/output metadata."""
    if not isinstance(item, dict) or set(item) not in (
        {"role", "capturePath"},
        {"role", "capturePath", "output"},
    ):
        return item, root
    if item["role"] not in {"baseline", "honest", "adversarial"}:
        raise TraceEvidenceError("capture_role_invalid")
    path = (root / str(item["capturePath"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise TraceEvidenceError("capture_path_escapes_experiment") from exc
    capture = _read_json(path)
    merged = {**capture, "role": item["role"]}
    if "output" in item:
        merged["output"] = item["output"]
    return merged, path.parent


def _output(value: Any, run_id: str) -> dict[str, str]:
    """Normalize flat evidence and the Comfy campaign's captured media receipt."""
    flat = {"canonicalVideoSha256", "canonicalAudioSha256", "mediaSha256"}
    if isinstance(value, dict) and set(value) == flat:
        result = value
    elif isinstance(value, dict) and {"runId", "status", "sha256", "video", "audio"} <= set(value):
        if value["runId"] != run_id or value["status"] != "completed":
            raise TraceEvidenceError("capture_output_identity_invalid")
        if not isinstance(value["video"], dict) or not isinstance(value["audio"], dict):
            raise TraceEvidenceError("capture_output_invalid")
        result = {
            "canonicalVideoSha256": value["video"].get("sha256"),
            "canonicalAudioSha256": value["audio"].get("sha256"),
            "mediaSha256": value.get("sha256"),
        }
    else:
        raise TraceEvidenceError("capture_output_invalid")
    if not all(isinstance(item, str) and len(item) == 64 for item in result.values()):
        raise TraceEvidenceError("capture_output_invalid")
    return result


def _run(root: Path, item: Any) -> dict[str, Any]:
    item, capture_root = _capture_from_binding(root, item)
    required = {
        "schema",
        "runId",
        "split",
        "role",
        "condition",
        "lineageId",
        "inputHash",
        "capture",
        "trace",
        "finalLatent",
        "output",
    }
    if (
        not isinstance(item, dict)
        or not required <= set(item)
        or item["schema"] != "forge/inference-trace-capture/v1"
    ):
        raise TraceEvidenceError("capture_schema_invalid")
    if item["split"] not in {"calibration", "holdout"} or item["role"] not in {
        "baseline",
        "honest",
        "adversarial",
    }:
        raise TraceEvidenceError("capture_role_invalid")
    if not all(
        isinstance(item[k], str) and item[k]
        for k in ("runId", "condition", "lineageId", "inputHash")
    ):
        raise TraceEvidenceError("capture_identity_invalid")
    if (
        not isinstance(item["capture"], dict)
        or not isinstance(item["trace"], list)
        or not item["trace"]
    ):
        raise TraceEvidenceError("capture_trace_invalid")
    seen: set[int] = set()
    tensors: dict[str | int, np.ndarray] = {}
    for step in item["trace"]:
        if not isinstance(step, dict) or type(step.get("step")) is not int:
            raise TraceEvidenceError("capture_step_invalid")
        if step["step"] in seen:
            raise TraceEvidenceError("capture_step_duplicate")
        seen.add(step["step"])
        if set(step) == {"step", "tensor"}:
            tensors[step["step"]] = _tensor(capture_root, step["tensor"])
        elif "stateCallback" in step:
            tensors.update(_components(capture_root, step["stateCallback"], step["step"]))
        else:
            raise TraceEvidenceError("capture_step_tensor_missing")
    # The final latent is retained separately so a later decoder-binding experiment
    # can use it.  This evaluator intentionally does not claim that binding exists.
    if isinstance(item["finalLatent"], dict):
        _tensor(capture_root, item["finalLatent"])
    else:
        _components(capture_root, item["finalLatent"], -1)
    item = {**item, "output": _output(item["output"], item["runId"])}
    return {"raw": item, "tensors": tensors}


def _metrics(reference: dict[int, np.ndarray], candidate: dict[int, np.ndarray]) -> dict[str, Any]:
    common = sorted(set(reference) & set(candidate))
    rows: list[dict[str, Any]] = []
    for step in common:
        left, right = reference[step], candidate[step]
        if left.shape != right.shape:
            return {"comparable": False, "reason": "trajectory_shape_diverged", "steps": rows}
        diff = right - left
        scale = max(float(np.linalg.norm(left.ravel())), 1e-12)
        max_scale = max(float(np.max(np.abs(left))), 1e-12)
        rows.append(
            {
                "step": step,
                "relL2": float(np.linalg.norm(diff.ravel()) / scale),
                "maxAbsRel": float(np.max(np.abs(diff)) / max_scale),
                "exact": bool(np.array_equal(left, right)),
            }
        )
    missing = sorted(set(reference) ^ set(candidate))
    if missing:
        return {
            "comparable": False,
            "reason": "trace_step_set_diverged",
            "steps": rows,
            "missingSteps": missing,
        }
    return {
        "comparable": True,
        "steps": rows,
        "maxRelL2": max(row["relL2"] for row in rows),
        "maxAbsRel": max(row["maxAbsRel"] for row in rows),
        "firstNonExactStep": next((row["step"] for row in rows if not row["exact"]), None),
    }


def _below(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return bool(metrics.get("comparable")) and (
        metrics["maxRelL2"] <= thresholds["maxRelL2"]
        and metrics["maxAbsRel"] <= thresholds["maxAbsRel"]
    )


def analyze(root: Path) -> dict[str, Any]:
    """Analyze a frozen experiment directory without modifying its raw evidence."""
    root = root.resolve()
    raw = _read_json(root / "experiment.json")
    if set(raw) != {"schema", "frozenSplit", "runs"} or raw["schema"] != SCHEMA:
        raise TraceEvidenceError("experiment_schema_invalid")
    split = raw["frozenSplit"]
    if (
        set(split) != {"calibrationLineages", "holdoutLineages"}
        or not all(
            isinstance(split[key], list)
            and split[key]
            and all(isinstance(x, str) for x in split[key])
            for key in split
        )
        or set(split["calibrationLineages"]) & set(split["holdoutLineages"])
    ):
        raise TraceEvidenceError("frozen_split_invalid")
    runs = [_run(root, item) for item in raw["runs"]]
    if len({entry["raw"]["runId"] for entry in runs}) != len(runs):
        raise TraceEvidenceError("run_id_duplicate")
    allowed = {
        "calibration": set(split["calibrationLineages"]),
        "holdout": set(split["holdoutLineages"]),
    }
    if any(entry["raw"]["lineageId"] not in allowed[entry["raw"]["split"]] for entry in runs):
        raise TraceEvidenceError("run_outside_frozen_split")

    references: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in runs:
        r = entry["raw"]
        if r["role"] == "baseline":
            key = (r["split"], r["lineageId"])
            if key in references:
                raise TraceEvidenceError("baseline_duplicate")
            references[key] = entry
    if set(references) != {
        (kind, lineage) for kind, values in allowed.items() for lineage in values
    }:
        raise TraceEvidenceError("baseline_missing")

    comparisons: list[dict[str, Any]] = []
    for entry in runs:
        r = entry["raw"]
        if r["role"] == "baseline":
            continue
        reference = references[(r["split"], r["lineageId"])]["raw"]
        if r["inputHash"] != reference["inputHash"]:
            raise TraceEvidenceError("fixed_input_mismatch")
        metrics = _metrics(references[(r["split"], r["lineageId"])]["tensors"], entry["tensors"])
        output_differs = r["output"] != reference["output"]
        comparisons.append(
            {
                "runId": r["runId"],
                "split": r["split"],
                "role": r["role"],
                "condition": r["condition"],
                "lineageId": r["lineageId"],
                "metrics": metrics,
                "outputDiffersFromBaseline": output_differs,
            }
        )

    honest = [
        row for row in comparisons if row["split"] == "calibration" and row["role"] == "honest"
    ]
    adversarial = [
        row for row in comparisons if row["split"] == "calibration" and row["role"] == "adversarial"
    ]
    if (
        len(honest) < 2
        or len({row["lineageId"] for row in honest}) < 2
        or not all(row["metrics"].get("comparable") for row in honest)
    ):
        return {
            "schema": REPORT_SCHEMA,
            "status": "INCONCLUSIVE",
            "claim": "none",
            "reason": "honest_calibration_insufficient",
            "comparisons": comparisons,
        }
    thresholds = {
        "maxRelL2": max(row["metrics"]["maxRelL2"] for row in honest),
        "maxAbsRel": max(row["metrics"]["maxAbsRel"] for row in honest),
    }
    bypasses = [
        row
        for row in adversarial
        if _below(row["metrics"], thresholds) and row["outputDiffersFromBaseline"]
    ]
    separable = [row for row in adversarial if not _below(row["metrics"], thresholds)]
    if not adversarial or bypasses or len(separable) != len(adversarial):
        reason = (
            "trace_to_output_bypass_observed" if bypasses else "adversarial_trace_not_separated"
        )
        return {
            "schema": REPORT_SCHEMA,
            "status": "INCONCLUSIVE",
            "claim": "none",
            "reason": reason,
            "thresholds": thresholds,
            "comparisons": comparisons,
            "bypasses": bypasses,
        }
    holdout = [row for row in comparisons if row["split"] == "holdout"]
    if not holdout or any(
        (row["role"] == "honest" and not _below(row["metrics"], thresholds))
        or (row["role"] == "adversarial" and _below(row["metrics"], thresholds))
        for row in holdout
    ):
        return {
            "schema": REPORT_SCHEMA,
            "status": "INCONCLUSIVE",
            "claim": "none",
            "reason": "heldout_separation_not_demonstrated",
            "thresholds": thresholds,
            "comparisons": comparisons,
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "claim": "callback_trajectory_discrimination_only",
        "limits": [
            "no_trace_to_served_video_binding",
            "no_checkpoint_or_gpu_ownership_proof",
            "not_a_population_estimate",
        ],
        "thresholds": thresholds,
        "comparisons": comparisons,
    }
