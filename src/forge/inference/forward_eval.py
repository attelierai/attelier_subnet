"""Frozen diagnostic analysis for captured fixed-input H3 forwards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "forge/inference-trace-plan/v1"
RECEIPT_SCHEMA = "forge/inference-forward-replay/v1"
CALIBRATION_SCHEMA = "forge/inference-forward-calibration/v1"
REPORT_SCHEMA = "forge/inference-forward-report/v1"


class ForwardEvidenceError(ValueError):
    pass


def _read(p: Path) -> dict[str, Any]:
    try:
        v = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ForwardEvidenceError("json_invalid") from e
    if not isinstance(v, dict):
        raise ForwardEvidenceError("json_object_required")
    return v


def _hash(v: Any) -> str:
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _plan(root: Path):
    p = _read(root / "experiment-plan.json")
    names = [x.get("name") for x in p.get("variants", []) if isinstance(x, dict)]
    need = {"honest-1", "honest-2", "steps10", "steps3", "weights-fp8", "lora20"}
    if (
        p.get("schema") != PLAN_SCHEMA
        or set(names) != need
        or len(names) != 6
        or not isinstance(p.get("lineages"), list)
    ):
        raise ForwardEvidenceError("plan_schema_invalid")
    ids = set()
    for x in p["lineages"]:
        if (
            not isinstance(x, dict)
            or x.get("split") not in {"calibration", "holdout"}
            or not isinstance(x.get("id"), str)
            or not isinstance(x.get("parameters"), dict)
            or x["id"] in ids
        ):
            raise ForwardEvidenceError("lineage_invalid")
        ids.add(x["id"])
    if {x["split"] for x in p["lineages"]} != {"calibration", "holdout"}:
        raise ForwardEvidenceError("plan_split_invalid")
    return p, names, p["lineages"]


def _run(root: Path, run_id: str):
    try:
        m = _read(root / "captures" / run_id / "manifest.json")
    except ForwardEvidenceError:
        return {"status": "MISSING"}
    fw = m.get("forward")
    expected = {
        f"canonical-{int(x['call']):03d}": x
        for x in fw or []
        if isinstance(x, dict) and type(x.get("call")) is int and isinstance(x.get("sha256"), str)
    }
    if not expected or len(expected) != len(fw):
        return {"status": "INCONCLUSIVE", "reason": "forward_manifest_invalid"}
    actual = {x.parent.name: x for x in (root / "replays" / run_id).glob("canonical-*/result.json")}
    if set(actual) != set(expected):
        return {"status": "MISSING"}
    receipts = {}
    values = []
    for label, path in actual.items():
        v = _read(path)
        o = v.get("outcome")
        if (
            v.get("schema") != RECEIPT_SCHEMA
            or v.get("candidateLabel") != "canonical"
            or not isinstance(o, dict)
        ):
            raise ForwardEvidenceError("receipt_schema_invalid")
        if v.get("bundleSha256") != expected[label]["sha256"]:
            return {"status": "INCONCLUSIVE", "reason": "bundle_binding_mismatch"}
        if o.get("status") == "INCONCLUSIVE":
            return {"status": "INCONCLUSIVE", "reason": str(o.get("reason", "replay_inconclusive"))}
        if o.get("status") != "MEASURED" or not all(
            isinstance(o.get(k), (int, float)) and math.isfinite(o[k]) and o[k] >= 0
            for k in ("maxAbs", "rms", "relativeRms")
        ):
            raise ForwardEvidenceError("receipt_measurement_invalid")
        receipts[label] = {
            "receiptSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bundleSha256": v["bundleSha256"],
            "relativeRms": o["relativeRms"],
        }
        values.append(float(o["relativeRms"]))
    return {"status": "MEASURED", "relativeRms": max(values), "receipts": receipts}


def calibrate(root: Path, out: Path, *, floor: float = 1e-6):
    if not math.isfinite(floor) or floor <= 0:
        raise ForwardEvidenceError("floor_invalid")
    p, _, lines = _plan(root)
    values = []
    evidence = {}
    for x in lines:
        if x["split"] == "calibration":
            for name in ("honest-1", "honest-2"):
                r = _run(root, f"{x['id']}-{name}")
                if r["status"] != "MEASURED":
                    raise ForwardEvidenceError("calibration_receipt_" + r["status"].lower())
                values.append(r["relativeRms"])
                evidence[f"{x['id']}-{name}"] = r["receipts"]
    if len(values) < 2:
        raise ForwardEvidenceError("calibration_honest_insufficient")
    a = {
        "schema": CALIBRATION_SCHEMA,
        "planSha256": _hash(p),
        "calibrationReceipts": evidence,
        "honestCount": len(values),
        "relativeRmsThreshold": max(values) * 2 + floor,
        "floor": floor,
        "claim": "exploratory_forward_threshold_not_statistical_certification",
    }
    if out.exists():
        raise ForwardEvidenceError("calibration_artifact_exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(a, sort_keys=True, separators=(",", ":")) + "\n")
    return a


def report(root: Path, calibration_path: Path):
    p, names, lines = _plan(root)
    c = _read(calibration_path)
    threshold = c.get("relativeRmsThreshold")
    if (
        c.get("schema") != CALIBRATION_SCHEMA
        or c.get("planSha256") != _hash(p)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ForwardEvidenceError("calibration_binding_invalid")
    for run_id, frozen in c.get("calibrationReceipts", {}).items():
        if _run(root, run_id).get("receipts") != frozen:
            raise ForwardEvidenceError("calibration_evidence_changed")
    rows = []
    for x in lines:
        if x["split"] == "holdout":
            for name in names:
                r = _run(root, f"{x['id']}-{name}")
                row = {"lineageId": x["id"], "variant": name, "runId": f"{x['id']}-{name}", **r}
                if r["status"] == "MEASURED":
                    row["accepted"] = r["relativeRms"] <= threshold
                rows.append(row)
    measured = [x for x in rows if x["status"] == "MEASURED"]
    unresolved = [x for x in rows if x["status"] != "MEASURED"]
    return {
        "schema": REPORT_SCHEMA,
        "status": "INCONCLUSIVE" if unresolved else "MEASURED",
        "claim": "forward_checkpoint_diagnostic_only",
        "threshold": threshold,
        "rows": rows,
        "summary": {
            "missing": sum(x["status"] == "MISSING" for x in rows),
            "inconclusive": sum(x["status"] == "INCONCLUSIVE" for x in rows),
            "honestRejected": sum(
                x["variant"] in {"honest-1", "honest-2"} and not x["accepted"] for x in measured
            ),
            "checkpointFalseAccepts": sum(
                x["variant"] in {"weights-fp8", "lora20"} and x["accepted"] for x in measured
            ),
            "stepVariantsMeasured": sum(x["variant"] in {"steps10", "steps3"} for x in measured),
        },
        "limits": [
            "step_variants_are_not_expected_to_fail_forward_gate",
            "no_trajectory_or_step_count_verification",
            "no_video_output_binding",
            "no_gpu_ownership_proof",
            "ground_truth_variant_labels_are_report_only",
        ],
    }
