"""Immutable V3 composition of a replayed V2 package and CPU flicker evidence."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.evaluation.evaluator_qualification import finalize_qualification
from forge.evaluation.flicker_guard import RULE, RULE_ID, algorithm_id, decide, validate_envelope
from forge.evaluation.vision_media import __file__ as vision_media_file
from forge.protocol.hashing import hash_canonical, sha256_file

SCHEMA = "forge/evaluator-qualification-composite/v3"
VERSION = 3
FRESH_KINDS = ("identity", "transcode", "pan", "zoom", "ramp", "inversion", "luminance", "chroma")
REGRESSION_KINDS = ("identity", "transcode", "flicker")
FALSE_KINDS = frozenset(("identity", "transcode", "pan", "zoom", "ramp"))


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _package_hash(package: Mapping[str, Any]) -> str:
    return hash_canonical("forge/evaluator-qualification-v2-package/v1", dict(package))


def _v2(package: Mapping[str, Any]) -> dict[str, Any]:
    if set(package) != {"plan", "raw", "result"} or not all(
        isinstance(package[key], Mapping) for key in package
    ):
        raise ValueError("v2 package must contain plan, raw and result")
    if package["plan"].get("qualification_version") != 2:
        raise ValueError("composite base package is not v2")
    recorded = package["result"]
    if (
        recorded.get("schema") != "forge/evaluator-qualification/v1"
        or recorded.get("qualification_version") != 2
    ):
        raise ValueError("composite base result is not v2")
    recorded_body = {key: value for key, value in recorded.items() if key != "receipt_id"}
    if recorded.get("receipt_id") != hash_canonical(
        "forge/evaluator-qualification-result/v2", recorded_body
    ):
        raise ValueError("composite base v2 receipt hash is invalid")
    replayed = finalize_qualification(package["plan"], package["raw"])
    replayed_body = {key: value for key, value in replayed.items() if key != "receipt_id"}
    if not _portable_v2_equal(recorded_body, replayed_body):
        raise ValueError("v2 package does not replay exactly")
    # The hashed immutable V2 result remains the evidence.  A portable replay
    # only permits numerical round-off in its derived evidence leaves.
    return dict(recorded)


def _portable_v2_equal(expected: object, actual: object, path: tuple[str, ...] = ()) -> bool:
    """Compare a V2 replay without treating a float rounding change as science.

    V2's receipt is first verified above.  The only permitted cross-runtime
    difference is a finite, sub-nanounit float under its derived evidence;
    decisions, raw bindings, thresholds and all non-float values remain exact.
    """
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return False
        if set(expected) != set(actual):
            return False
        return all(
            _portable_v2_equal(expected[key], actual[key], (*path, str(key)))
            for key in expected
        )
    if isinstance(expected, list) or isinstance(actual, list):
        return (
            isinstance(expected, list)
            and isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                _portable_v2_equal(left, right, path)
                for left, right in zip(expected, actual, strict=False)
            )
        )
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, float) and path and path[0] == "evidence":
        return math.isfinite(expected) and math.isfinite(actual) and abs(expected - actual) <= 1e-9
    return expected == actual


def _geometry(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"width", "height", "frames", "fps"}:
        raise ValueError("composite geometry fields are invalid")
    if any(type(value[key]) is not int or value[key] <= 0 for key in value):
        raise ValueError("composite geometry values are invalid")
    return dict(value)


def _guard_runtime() -> dict[str, str]:
    import imageio_ffmpeg
    import numpy as np

    module = Path(__file__).with_name("flicker_guard.py")
    return {
        "module_sha256": sha256_file(module),
        "media_decoder_sha256": sha256_file(Path(vision_media_file)),
        "decoder_binary_sha256": sha256_file(Path(imageio_ffmpeg.get_ffmpeg_exe())),
        "numpy_version": np.__version__,
    }


def _check_guard_runtime(value: object, algorithm: str, rule: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "module_sha256",
        "media_decoder_sha256",
        "decoder_binary_sha256",
        "numpy_version",
    }:
        raise ValueError("composite guard runtime fields are invalid")
    for key in ("module_sha256", "media_decoder_sha256", "decoder_binary_sha256"):
        _sha(value[key], f"composite guard runtime {key}")
    if not isinstance(value["numpy_version"], str) or not value["numpy_version"]:
        raise ValueError("composite guard runtime numpy version is invalid")
    _sha(rule, "composite rule id")
    if rule != RULE_ID:
        raise ValueError("composite rule id does not bind the guard rule")
    expected = hash_canonical(
        "forge/flicker-guard-algorithm/v1",
        RULE,
        value["module_sha256"],
        value["media_decoder_sha256"],
        value["decoder_binary_sha256"],
        value["numpy_version"],
    )
    if algorithm != expected:
        raise ValueError("composite algorithm does not bind declared runtime")


def _cell_fields() -> set[str]:
    return {"source_id", "kind", "video_sha256", "decoded_sha256"}


def _entry(value: object, *, source: bool = False) -> dict[str, Any]:
    fields = (
        {"source_id", "lineage_id", "slices", "video_sha256", "decoded_sha256"}
        if source
        else _cell_fields()
    )
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("composite corpus entry fields are invalid")
    if not isinstance(value["source_id"], str) or not value["source_id"]:
        raise ValueError("composite source id is invalid")
    _sha(value["video_sha256"], "composite video hash")
    _sha(value["decoded_sha256"], "composite decoded hash")
    if source:
        if not isinstance(value["lineage_id"], str) or not value["lineage_id"]:
            raise ValueError("composite lineage is invalid")
        if not isinstance(value["slices"], list) or not all(
            isinstance(item, str) and item for item in value["slices"]
        ):
            raise ValueError("composite descriptive slices are invalid")
    elif value["kind"] not in FRESH_KINDS + REGRESSION_KINDS:
        raise ValueError("composite control kind is invalid")
    return value


def _manifest(value: object, v2_package: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"sources", "controls", "regression_controls"}:
        raise ValueError("composite corpus manifest fields are invalid")
    sources, controls, regression = (
        value["sources"],
        value["controls"],
        value["regression_controls"],
    )
    if not all(isinstance(rows, list) for rows in (sources, controls, regression)):
        raise ValueError("composite corpus rows are invalid")
    sources = [_entry(row, source=True) for row in sources]
    controls = [_entry(row) for row in controls]
    regression = [_entry(row) for row in regression]
    if len(sources) != 8 or len(controls) != 64 or len(regression) != 36:
        raise ValueError("composite corpus matrix size is invalid")
    source_ids = {row["source_id"] for row in sources}
    if (
        len(source_ids) != 8
        or len({row["lineage_id"] for row in sources}) != 8
        or len({row["decoded_sha256"] for row in sources}) != 8
    ):
        raise ValueError("composite natural sources are not independent")
    expected_fresh = {(source_id, kind) for source_id in source_ids for kind in FRESH_KINDS}
    if {(row["source_id"], row["kind"]) for row in controls} != expected_fresh:
        raise ValueError("composite fresh control matrix is incomplete")
    if any(row["source_id"] not in source_ids for row in controls):
        raise ValueError("composite fresh control source is unknown")
    by_source = {row["source_id"]: row for row in sources}
    for row in controls:
        if row["kind"] == "identity" and any(
            row[key] != by_source[row["source_id"]][key]
            for key in ("video_sha256", "decoded_sha256")
        ):
            raise ValueError("composite identity control does not bind fresh source bytes")
    v2_sources = {row["source_id"] for row in v2_package["plan"]["manifest"]["sources"]}
    v2_evidence = v2_package["plan"]["manifest"]["sources"]
    if (
        source_ids & v2_sources
        or {row["lineage_id"] for row in sources} & {row["lineage_id"] for row in v2_evidence}
        or {row["decoded_sha256"] for row in sources}
        & {row["decoded_sha256"] for row in v2_evidence}
    ):
        raise ValueError("fresh composite sources overlap v2 evidence")
    expected_regression = {
        (source_id, kind) for source_id in v2_sources for kind in REGRESSION_KINDS
    }
    if {(row["source_id"], row["kind"]) for row in regression} != expected_regression:
        raise ValueError("composite regression matrix is incomplete")
    v2_controls = {
        (row["source_id"], row["kind"]): row
        for row in v2_package["plan"]["controls"]
        if row.get("kind") in REGRESSION_KINDS
    }
    for row in regression:
        bound = v2_controls.get((row["source_id"], row["kind"]))
        if not bound or any(
            row[key] != bound.get(key) for key in ("video_sha256", "decoded_sha256")
        ):
            raise ValueError("composite regression controls do not bind v2 bytes")
    return {"sources": sources, "controls": controls, "regression_controls": regression}


def freeze_plan(
    v2_package: Mapping[str, Any], corpus_manifest: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the expected 64 fresh and 36 retained control bytes before CPU measurement."""
    v2_result = _v2(v2_package)
    geometry = _geometry(geometry)
    if geometry != v2_package["plan"]["manifest"]["geometry"]:
        raise ValueError("composite geometry differs from base v2")
    corpus = _manifest(dict(corpus_manifest), v2_package)
    body = {
        "schema": SCHEMA,
        "qualification_version": VERSION,
        "v2_package": copy.deepcopy(dict(v2_package)),
        "v2_package_sha256": _package_hash(v2_package),
        "v2_receipt_id": _sha(v2_result.get("receipt_id"), "v2 receipt id"),
        "algorithm_id": algorithm_id(),
        "rule_id": RULE_ID,
        "guard_runtime": _guard_runtime(),
        "geometry": geometry,
        "corpus_manifest": corpus,
    }
    return {**body, "plan_id": hash_canonical(SCHEMA, body)}


def _expected(kind: str) -> bool:
    return kind not in FALSE_KINDS


def _raw(plan: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "plan_id",
        "algorithm_id",
        "geometry",
        "cells",
        "raw_receipt_id",
    }:
        raise ValueError("composite raw fields are invalid")
    if raw["plan_id"] != plan["plan_id"] or raw["algorithm_id"] != plan["algorithm_id"]:
        raise ValueError("composite raw identity differs from frozen plan")
    if raw["geometry"] != plan["geometry"] or not isinstance(raw["cells"], list):
        raise ValueError("composite raw geometry or cells differ from frozen plan")
    expected = {
        (row["source_id"], row["kind"]): row
        for group in (
            plan["corpus_manifest"]["controls"],
            plan["corpus_manifest"]["regression_controls"],
        )
        for row in group
    }
    cells = raw["cells"]
    if len(cells) != 100:
        raise ValueError("composite raw matrix size is invalid")
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != _cell_fields() | {"envelope"}:
            raise ValueError("composite raw cell fields are invalid")
        key = (cell["source_id"], cell["kind"])
        if key in observed or key not in expected:
            raise ValueError("composite raw cell is duplicated or foreign")
        bound = expected[key]
        if any(cell[name] != bound[name] for name in ("video_sha256", "decoded_sha256")):
            raise ValueError("composite raw cell bytes differ from frozen plan")
        diagnostics = validate_envelope(
            cell["envelope"], expected_algorithm_id=plan["algorithm_id"]
        )
        if diagnostics["frame_count"] != plan["geometry"]["frames"]:
            raise ValueError("composite envelope frame count differs from frozen geometry")
        actual = decide(diagnostics, steady_lighting=True)
        if actual != _expected(cell["kind"]):
            failures.append(f"{cell['source_id']}:{cell['kind']}:flicker_guard")
        observed[key] = cell
    if set(observed) != set(expected):
        raise ValueError("composite raw matrix coverage is incomplete")
    body = {key: value for key, value in raw.items() if key != "raw_receipt_id"}
    if raw["raw_receipt_id"] != hash_canonical(
        "forge/evaluator-qualification-composite-raw/v3", body
    ):
        raise ValueError("composite raw receipt hash is invalid")
    return cells, failures


def finalize_composite(plan: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return V3 without relabelling its immutable V2 result."""
    try:
        if (
            not isinstance(plan, Mapping)
            or plan.get("schema") != SCHEMA
            or plan.get("qualification_version") != VERSION
        ):
            raise ValueError("composite plan is invalid")
        body = {key: value for key, value in plan.items() if key != "plan_id"}
        if plan.get("plan_id") != hash_canonical(SCHEMA, body):
            raise ValueError("composite plan hash is invalid")
        v2 = _v2(plan["v2_package"])
        if plan.get("v2_package_sha256") != _package_hash(plan["v2_package"]) or plan.get(
            "v2_receipt_id"
        ) != v2.get("receipt_id"):
            raise ValueError("composite v2 package binding is invalid")
        _sha(plan.get("algorithm_id"), "composite algorithm id")
        _sha(plan.get("rule_id"), "composite rule id")
        _check_guard_runtime(plan.get("guard_runtime"), plan["algorithm_id"], plan["rule_id"])
        geometry = _geometry(plan["geometry"])
        if geometry != plan["v2_package"]["plan"]["manifest"]["geometry"]:
            raise ValueError("composite geometry differs from base v2")
        _manifest(plan["corpus_manifest"], plan["v2_package"])
        _, guard_failures = _raw(plan, raw)
        allowed = v2["qualification"] == "QUALIFIED" or (
            v2["qualification"] == "FAILED"
            and bool(v2["reasons"])
            and all(
                isinstance(reason, str) and reason.endswith(":flicker") for reason in v2["reasons"]
            )
        )
        if not allowed:
            status, reasons = "FAILED", ["v2 qualification has non-flicker rejection"]
        elif guard_failures:
            status, reasons = "FAILED", guard_failures
        else:
            status, reasons = "QUALIFIED", []
        thresholds = dict(v2["guard_thresholds"])
        thresholds["flicker_guard_v1"] = plan["algorithm_id"]
        result = {
            "schema": "forge/evaluator-qualification/v1",
            "qualification": status,
            "qualification_version": VERSION,
            "v2_receipt_id": v2["receipt_id"],
            "v2_package_sha256": plan["v2_package_sha256"],
            "plan_id": plan["plan_id"],
            "raw_receipt_id": raw.get("raw_receipt_id") if isinstance(raw, Mapping) else None,
            "algorithm_id": plan["algorithm_id"],
            "rule_id": plan["rule_id"],
            "manifest_id": v2["manifest_id"],
            "control_plan_id": v2["control_plan_id"],
            "evaluator_id": v2["evaluator_id"],
            "spec_hash": v2["spec_hash"],
            "evidence": copy.deepcopy(v2["evidence"]),
            "guard_thresholds": thresholds,
            "reasons": reasons,
        }
        return {**result, "receipt_id": hash_canonical(SCHEMA, result)}
    except RuntimeError as error:
        return _incomplete(plan, raw, str(error), failed=True)
    except (KeyError, TypeError, ValueError) as error:
        return _incomplete(plan, raw, str(error))


def _incomplete(
    plan: Mapping[str, Any], raw: Mapping[str, Any], reason: str, *, failed: bool = False
) -> dict[str, Any]:
    body = {
        # Results deliberately retain the V2 envelope so launch-policy V2
        # consumers stay narrow.  The explicit qualification version selects
        # this composite replay path.
        "schema": "forge/evaluator-qualification/v1",
        "qualification": "FAILED" if failed else "INCOMPLETE",
        "qualification_version": VERSION,
        "plan_id": plan.get("plan_id") if isinstance(plan, Mapping) else None,
        "raw_receipt_id": raw.get("raw_receipt_id") if isinstance(raw, Mapping) else None,
        "reasons": [reason],
    }
    return {**body, "receipt_id": hash_canonical(SCHEMA, body)}


def replay_composite_package(package: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(package, Mapping) or set(package) != {"plan", "raw", "result"}:
        raise ValueError("composite package must contain plan, raw and result")
    result = finalize_composite(package["plan"], package["raw"])
    if result != package["result"]:
        raise ValueError("composite package result does not replay")
    return result


def main() -> None:
    import argparse
    import json

    from forge.validator.launch_service import write_immutable_json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--v2-package", type=Path, required=True)
    freeze.add_argument("--corpus-manifest", type=Path, required=True)
    freeze.add_argument("--geometry", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--raw", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze_plan(
            json.loads(args.v2_package.read_text()),
            json.loads(args.corpus_manifest.read_text()),
            json.loads(args.geometry.read_text()),
        )
    else:
        value = finalize_composite(
            json.loads(args.plan.read_text()), json.loads(args.raw.read_text())
        )
    write_immutable_json(args.out, value)


if __name__ == "__main__":
    main()
