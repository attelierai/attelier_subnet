"""Bounded evaluator qualification controls with real media-file boundaries.

The harness proves that a particular evaluator receipt reacts to pre-registered
controls.  It does *not* prove product quality or authorize a GPU campaign by
itself.  In particular, generated controls and unknown signals stay
``NOT_QUALIFIED`` until a separately pinned real-GPU calibration supplies the
same immutable receipt format.
"""

from __future__ import annotations

import copy
import math
import shutil
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from scipy.stats import t as student_t

from forge.evaluation.flicker_guard import decide as decide_flicker
from forge.evaluation.flicker_guard import validate_envelope
from forge.evaluation.vision_media import degrade_frames, load_video, media_identity
from forge.protocol.hashing import hash_canonical, sha256_file

SCHEMA = "forge/evaluator-qualification/v1"
QUALIFICATION_V2 = 2
SOURCE_ELIGIBILITY_SCHEMA = "forge/evaluator-source-eligibility/v1"
SOURCE_TEMPORAL_DELTA_FLOOR = 0.1 / 255
CONTROL_KINDS = (
    "identity",
    "transcode",
    "freeze",
    "black_frame",
    "magenta_frame",
    "segment_corruption",
    "flicker",
    "blur",
    "semantic_mismatch",
    "copy_diversity",
    "hash_tamper",
    "missing",
)
SEVERE_NEGATIVES = frozenset(CONTROL_KINDS) - {"identity", "transcode"}

# This rule is deliberately local rather than imported from launch_policy: both
# artifacts are independently hash-bound and importing the policy here would
# make the qualification replay depend on the launch-service module graph.
V2_DECISION_RULE = {
    "schema": "forge/evaluator-qualification-rule/v2",
    "quality_margins": {"VQ": 0.35, "MQ": 0.20, "TA": 0.20},
    "transcode": {"test": "paired_tost", "confidence": 0.90},
    "blur": {"axes": ["VQ", "TA"], "confidence": 0.975},
    "semantic_mismatch": {"axis": "TA", "confidence": 0.95},
    "blur_family_slice": "no_family_positive_on_vq_and_ta",
    "confirmation_sources": 8,
}


def source_manifest(
    sources: list[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    *,
    qualification_version: int = 1,
    excluded_confirmation_lineages: list[str] | None = None,
) -> dict[str, Any]:
    """Read and bind the versioned, human-labelled source panel."""
    if qualification_version not in {1, QUALIFICATION_V2}:
        raise ValueError("unsupported qualification version")
    count = 8 if qualification_version == 1 else 12
    if len(sources) != count:
        raise ValueError(f"qualification v{qualification_version} requires exactly {count} sources")
    seen, decoded, splits, families, bound = (
        set(),
        set(),
        {"development": set(), "confirmation": set()},
        {},
        [],
    )
    for source in sources:
        fields = {"source_id", "split", "family", "prompt", "path"}
        if qualification_version == QUALIFICATION_V2:
            fields.add("lineage_id")
        if set(source) != fields:
            raise ValueError("source fields are invalid")
        source_id, split, family, prompt, path = (
            source["source_id"],
            source["split"],
            source["family"],
            source["prompt"],
            source["path"],
        )
        if not isinstance(source_id, str) or source_id in seen or split not in splits:
            raise ValueError("source identities/splits are invalid")
        lineage_id = source.get("lineage_id")
        if not all(isinstance(value, str) and value for value in (family, prompt, path)):
            raise ValueError("source family, prompt and path are required")
        if qualification_version == QUALIFICATION_V2 and (
            not isinstance(lineage_id, str) or not lineage_id.strip()
        ):
            raise ValueError("v2 source lineage_id must be a nonempty human-reviewed label")
        video_path = Path(path)
        loaded = load_video(video_path, geometry)
        temporal_delta = loaded.diagnostics.get("temporal_delta")
        if (
            type(temporal_delta) not in (int, float)
            or not np.isfinite(temporal_delta)
            or temporal_delta <= SOURCE_TEMPORAL_DELTA_FLOOR
        ):
            raise ValueError(
                "qualification source lacks enough temporal movement for the freeze control"
            )
        decoded_identity = media_identity(loaded.frames)
        if decoded_identity in decoded:
            raise ValueError("qualification sources must have distinct decoded video identities")
        decoded.add(decoded_identity)
        seen.add(source_id)
        splits[split].add(family)
        families.setdefault(family, set()).add(split)
        entry = {
            "source_id": source_id,
            "split": split,
            "family": family,
            "prompt": prompt,
            "video_sha256": sha256_file(video_path),
            "decoded_sha256": decoded_identity,
            "temporal_delta": float(temporal_delta),
        }
        if qualification_version == QUALIFICATION_V2:
            entry["lineage_id"] = lineage_id
        bound.append(entry)
    if qualification_version == 1:
        if any(len(splits[split]) != 4 for split in splits) or any(
            len(value) != 2 for value in families.values()
        ):
            raise ValueError("each split needs one source from each of four shared families")
    else:
        excluded = excluded_confirmation_lineages or []
        if (
            not isinstance(excluded, list)
            or not excluded
            or any(not isinstance(value, str) or not value for value in excluded)
            or len(set(excluded)) != len(excluded)
        ):
            raise ValueError("v2 requires unique excluded confirmation lineages")
        expected = {"development": 4, "confirmation": 8}
        split_counts = {
            split: sum(1 for item in bound if item["split"] == split) for split in splits
        }
        if split_counts != expected:
            raise ValueError("v2 requires four development and eight confirmation sources")
        by_family = {
            (split, family): sum(
                1 for item in bound if item["split"] == split and item["family"] == family
            )
            for split in splits
            for family in families
        }
        if len(families) != 4 or any(
            by_family.get(("development", family)) != 1
            or by_family.get(("confirmation", family)) != 2
            for family in families
        ):
            raise ValueError("v2 needs one development and two confirmation sources per family")
        lineages = [item["lineage_id"] for item in bound]
        if len(set(lineages)) != len(lineages):
            raise ValueError("v2 source lineage labels must be unique")
        confirmation_lineages = {
            item["lineage_id"] for item in bound if item["split"] == "confirmation"
        }
        if confirmation_lineages.intersection(excluded):
            raise ValueError("v2 confirmation source reuses an excluded semantic lineage")
    body = {
        "schema": SCHEMA,
        "geometry": dict(geometry),
        "source_eligibility": {
            "schema": SOURCE_ELIGIBILITY_SCHEMA,
            "temporal_delta_floor": SOURCE_TEMPORAL_DELTA_FLOOR,
        },
        "sources": sorted(bound, key=lambda x: x["source_id"]),
    }
    if qualification_version == QUALIFICATION_V2:
        body.update(
            qualification_version=QUALIFICATION_V2,
            excluded_confirmation_lineages=sorted(excluded_confirmation_lineages or []),
            decision_rule=copy.deepcopy(V2_DECISION_RULE),
        )
    return {**body, "manifest_id": hash_canonical(SCHEMA, body)}


def controlled_frames(frames: np.ndarray, kind: str) -> np.ndarray:
    """Produce bounded severe controls over decoded RGB frames."""
    if kind in {"freeze", "blur"}:
        return degrade_frames(frames, kind)
    result = frames.copy()
    if kind == "black_frame":
        result[len(result) // 2] = 0
    elif kind == "magenta_frame":
        result[len(result) // 2] = np.array([255, 0, 255], dtype=np.uint8)
    elif kind == "segment_corruption":
        result[len(result) // 3 : 2 * len(result) // 3] = 255
    elif kind == "flicker":
        result[::2] = 255 - result[::2]
    elif kind not in {
        "identity",
        "transcode",
        "semantic_mismatch",
        "copy_diversity",
        "hash_tamper",
        "missing",
    }:
        raise ValueError("unknown qualification control")
    return result


def prepare_controls(
    sources: list[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    output: Path,
    *,
    qualification_version: int = 1,
    excluded_confirmation_lineages: list[str] | None = None,
) -> dict[str, Any]:
    """Materialize the bounded source/control bank through real MP4 files.

    This is CPU-only preparation.  Later scoring calls the pinned VideoReward
    worker over these exact files; it never accepts miner-rendered control data.
    """
    manifest = source_manifest(
        sources,
        geometry,
        qualification_version=qualification_version,
        excluded_confirmation_lineages=excluded_confirmation_lineages,
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("control output must be new or empty; never overwrite evidence")
    output.mkdir(parents=True, exist_ok=True)
    by_id = {item["source_id"]: item for item in sources}
    controls = []
    for source in manifest["sources"]:
        original = by_id[source["source_id"]]
        loaded = load_video(original["path"], geometry)
        for kind in CONTROL_KINDS:
            control_id = f"{source['source_id']}--{kind}"
            if kind == "missing":
                controls.append(
                    {
                        "control_id": control_id,
                        "source_id": source["source_id"],
                        "kind": kind,
                        "path": f"{control_id}.missing.mp4",
                        "video_sha256": None,
                        "decoded_sha256": None,
                    }
                )
                continue
            frames = (
                loaded.frames
                if kind in {"identity", "transcode"}
                else controlled_frames(loaded.frames, kind)
            )
            if kind == "semantic_mismatch":
                # Real media boundary, but the score request uses the paired source's prompt.
                other = next(
                    item
                    for item in manifest["sources"]
                    if item["split"] == source["split"] and item["family"] != source["family"]
                )
                frames = load_video(by_id[other["source_id"]]["path"], geometry).frames
            if kind == "copy_diversity":
                frames = loaded.frames
            path = output / f"{control_id}.mp4"
            if kind == "identity":
                shutil.copyfile(original["path"], path)
                if sha256_file(path) != source["video_sha256"]:
                    raise ValueError("identity control bytes differ from source")
                decoded = load_video(path, geometry)
                controls.append(
                    {
                        "control_id": control_id,
                        "source_id": source["source_id"],
                        "kind": kind,
                        "path": path.name,
                        "video_sha256": source["video_sha256"],
                        "decoded_sha256": media_identity(decoded.frames),
                    }
                )
                continue
            writer = imageio_ffmpeg.write_frames(
                str(path),
                (geometry["width"], geometry["height"]),
                fps=geometry["fps"],
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                output_params=["-crf", "18", "-preset", "medium"],
            )
            writer.send(None)
            for frame in frames:
                writer.send(frame)
            writer.close()
            decoded = load_video(path, geometry)
            entry = {
                "control_id": control_id,
                "source_id": source["source_id"],
                "kind": kind,
                "path": path.name,
                "video_sha256": sha256_file(path),
                "decoded_sha256": media_identity(decoded.frames),
            }
            if kind == "copy_diversity":
                entry["decoded_seed_sha256"] = [entry["decoded_sha256"]] * 3
            if kind == "hash_tamper":
                entry["video_sha256"] = "0" * 64
            controls.append(entry)
    body = {
        "schema": SCHEMA,
        "manifest": manifest,
        "controls": controls,
        "codec": {
            "name": "libx264",
            "pix_fmt": "yuv420p",
            "crf": 18,
            "preset": "medium",
            "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
        },
    }
    if qualification_version == QUALIFICATION_V2:
        body.update(
            qualification_version=QUALIFICATION_V2,
            decision_rule=copy.deepcopy(V2_DECISION_RULE),
        )
    return {**body, "control_plan_id": hash_canonical("forge/evaluator-controls/v1", body)}


_CONTROL_AXIS = {
    "freeze": "MQ",
    "black_frame": "VQ",
    "magenta_frame": "VQ",
    "segment_corruption": "VQ",
    "flicker": "MQ",
    "blur": "VQ",
    "semantic_mismatch": "TA",
}


def _decoded_guard_hits(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> set[str]:
    """The all-frame guard replay shared by controls and competitive candidates."""
    keys = {
        "dark_frame_fraction",
        "magenta_frame_fraction",
        "bright_frame_fraction",
        "frozen_fraction",
        "luma_acceleration_p95",
        "high_frequency_proxy",
        "temporal_delta",
    }
    for diagnostics in (reference, candidate):
        if not isinstance(diagnostics, Mapping) or not keys.issubset(diagnostics):
            raise ValueError("all-frame decoded diagnostics are incomplete")
        if any(
            type(diagnostics[key]) not in (int, float)
            or not np.isfinite(diagnostics[key])
            or diagnostics[key] < 0
            for key in keys
        ):
            raise ValueError("all-frame decoded diagnostics must be finite and nonnegative")
    base_threshold_keys = (keys - {"high_frequency_proxy", "temporal_delta"}) | {
        "high_frequency_proxy_ratio",
        "near_freeze_temporal_max",
        "near_freeze_ratio_max",
        "near_freeze_reference_floor",
    }
    v2_threshold_keys = base_threshold_keys - {"high_frequency_proxy_ratio"}
    v3_threshold_keys = v2_threshold_keys | {"flicker_guard_v1"}
    numeric_thresholds = {
        key: value for key, value in thresholds.items() if key != "flicker_guard_v1"
    }
    if set(thresholds) not in (base_threshold_keys, v2_threshold_keys, v3_threshold_keys) or any(
        type(value) not in (int, float) or not np.isfinite(value) or value < 0
        for value in numeric_thresholds.values()
    ):
        raise ValueError("qualified guard thresholds are invalid")
    v3 = set(thresholds) == v3_threshold_keys
    if v3 and (
        not isinstance(thresholds["flicker_guard_v1"], str)
        or len(thresholds["flicker_guard_v1"]) != 64
        or set(thresholds["flicker_guard_v1"]) - set("0123456789abcdef")
    ):
        raise ValueError("qualified v3 flicker guard marker is invalid")
    hits = set()
    if candidate["dark_frame_fraction"] > thresholds["dark_frame_fraction"]:
        hits.add("black_frame")
    if candidate["magenta_frame_fraction"] > thresholds["magenta_frame_fraction"]:
        hits.add("magenta_frame")
    if candidate["bright_frame_fraction"] > thresholds["bright_frame_fraction"]:
        hits.add("segment_corruption")
    if (
        candidate["frozen_fraction"] >= thresholds["frozen_fraction"]
        and reference["frozen_fraction"] < thresholds["frozen_fraction"]
    ):
        hits.add("freeze")
    if (
        reference["temporal_delta"] > thresholds["near_freeze_reference_floor"]
        and candidate["temporal_delta"] <= thresholds["near_freeze_temporal_max"]
        and candidate["temporal_delta"]
        <= thresholds["near_freeze_ratio_max"] * reference["temporal_delta"]
    ):
        hits.add("freeze")
    if v3:
        # V3 replaces the former unqualified global-brightness branch. Both
        # envelopes are required so every retained comparison has detector
        # provenance, but only the candidate can be the flicker target.
        marker = thresholds["flicker_guard_v1"]
        validate_envelope(reference.get("flicker_guard_v1"), expected_algorithm_id=marker)
        candidate_flicker = validate_envelope(
            candidate.get("flicker_guard_v1"), expected_algorithm_id=marker
        )
        if decide_flicker(candidate_flicker, steady_lighting=True):
            hits.add("flicker")
    elif candidate["luma_acceleration_p95"] > max(
        thresholds["luma_acceleration_p95"], 3 * reference["luma_acceleration_p95"]
    ):
        hits.add("flicker")
    if "high_frequency_proxy_ratio" in thresholds and candidate["high_frequency_proxy"] < (
        thresholds["high_frequency_proxy_ratio"] * reference["high_frequency_proxy"]
    ):
        hits.add("blur")
    return hits


def _finalize_qualification_v1(plan: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Replay pinned raw scores and derive thresholds only from development positives."""
    try:
        if plan.get("schema") != SCHEMA or not isinstance(plan.get("manifest"), Mapping):
            raise ValueError("frozen control plan is invalid")
        body = {key: value for key, value in plan.items() if key != "control_plan_id"}
        if plan.get("control_plan_id") != hash_canonical("forge/evaluator-controls/v1", body):
            raise ValueError("control plan hash is invalid")
        manifest = plan["manifest"]
        manifest_body = {key: value for key, value in manifest.items() if key != "manifest_id"}
        if manifest.get("manifest_id") != hash_canonical(SCHEMA, manifest_body):
            raise ValueError("source manifest hash is invalid")
        eligibility = manifest.get("source_eligibility")
        if eligibility != {
            "schema": SOURCE_ELIGIBILITY_SCHEMA,
            "temporal_delta_floor": SOURCE_TEMPORAL_DELTA_FLOOR,
        }:
            raise ValueError("source eligibility gate is missing or changed")
        sources_list = manifest.get("sources")
        if not isinstance(sources_list, list) or len(sources_list) != 8:
            raise ValueError("qualification needs exactly eight independently bound sources")
        required_source_fields = {
            "source_id",
            "split",
            "family",
            "prompt",
            "video_sha256",
            "decoded_sha256",
            "temporal_delta",
        }
        if any(
            not isinstance(source, Mapping)
            or set(source) != required_source_fields
            or type(source["temporal_delta"]) not in (int, float)
            or not np.isfinite(source["temporal_delta"])
            or source["temporal_delta"] <= SOURCE_TEMPORAL_DELTA_FLOOR
            for source in sources_list
        ):
            raise ValueError("qualification source eligibility does not replay")
        if (
            len({item["source_id"] for item in sources_list}) != 8
            or len({item["decoded_sha256"] for item in sources_list}) != 8
        ):
            raise ValueError("qualification source identities are duplicated")
        families = {item["family"] for item in sources_list}
        if len(families) != 4 or {(item["split"], item["family"]) for item in sources_list} != {
            (split, family) for split in ("development", "confirmation") for family in families
        }:
            raise ValueError("qualification source split/family matrix is incomplete")
        raw_body = {key: value for key, value in raw.items() if key != "raw_receipt_id"}
        if (
            raw.get("schema") != SCHEMA
            or raw.get("control_plan_id") != plan["control_plan_id"]
            or raw.get("raw_receipt_id")
            != hash_canonical("forge/evaluator-controls-score/v1", raw_body)
        ):
            raise ValueError("raw score receipt is invalid or belongs to another plan")
        evaluator = raw.get("evaluator")
        if not isinstance(evaluator, Mapping) or not all(
            isinstance(evaluator.get(key), str) and evaluator[key]
            for key in ("evaluator_id", "spec_hash")
        ):
            raise ValueError("raw receipt lacks pinned evaluator/spec provenance")
        spec, runtime = evaluator.get("spec"), evaluator.get("runtime")
        if (
            not isinstance(spec, dict)
            or not isinstance(runtime, dict)
            or evaluator["spec_hash"] != hash_canonical("forge/launch-evaluator-spec/v1", spec)
            or evaluator["evaluator_id"]
            != hash_canonical("forge/videoreward-study-runtime/v1", spec, runtime)
        ):
            raise ValueError("raw evaluator identity does not replay its pinned spec and runtime")
        expected = {control["control_id"]: control for control in plan["controls"]}
        expected_pairs = {
            (source["source_id"], kind) for source in sources_list for kind in CONTROL_KINDS
        }
        if (
            len(plan["controls"]) != 96
            or len(expected) != 96
            or {(control["source_id"], control["kind"]) for control in plan["controls"]}
            != expected_pairs
        ):
            raise ValueError("frozen control matrix is incomplete or duplicated")
        sources = {source["source_id"]: source for source in sources_list}
        for source_id, source in sources.items():
            identity = expected[f"{source_id}--identity"]
            if (
                identity.get("video_sha256") != source["video_sha256"]
                or identity.get("decoded_sha256") != source["decoded_sha256"]
            ):
                raise ValueError("identity control no longer binds its eligible source bytes")
        rows = raw.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 96
            or {row.get("control_id") for row in rows if isinstance(row, Mapping)} != set(expected)
        ):
            raise ValueError("raw score receipt does not cover the frozen controls")
        scored = {}
        for row in rows:
            control = expected[row["control_id"]]
            if control["kind"] == "missing":
                if row != {"control_id": control["control_id"], "status": "MISSING_REJECTED"}:
                    raise ValueError("procedural control changed")
                continue
            if control["kind"] == "hash_tamper":
                if row != {"control_id": control["control_id"], "status": "HASH_MISMATCH_REJECTED"}:
                    raise ValueError("procedural control changed")
                continue
            result = row.get("result")
            if (
                row.get("status") != "OK"
                or not isinstance(result, Mapping)
                or result.get("video_sha256") != control["video_sha256"]
                or not isinstance(result.get("scores"), Mapping)
            ):
                raise ValueError("raw score row does not bind frozen control bytes")
            if control["kind"] == "identity":
                diagnostics = result.get("diagnostics")
                if (
                    not isinstance(diagnostics, Mapping)
                    or diagnostics.get("temporal_delta")
                    != sources[control["source_id"]]["temporal_delta"]
                ):
                    raise ValueError("identity diagnostics do not replay source eligibility")
            scores = result["scores"]
            if set(scores) != {"VQ", "MQ", "TA"} or any(
                type(scores[key]) not in (int, float) or not np.isfinite(scores[key])
                for key in scores
            ):
                raise ValueError("raw score values are invalid")
            scored[control["control_id"]] = {key: float(scores[key]) for key in scores}
        thresholds = {}
        for axis in ("VQ", "MQ", "TA"):
            deltas = []
            for source in sources.values():
                if source["split"] != "development":
                    continue
                identity, transcode = (
                    scored[f"{source['source_id']}--{kind}"][axis]
                    for kind in ("identity", "transcode")
                )
                deltas.append(abs(identity - transcode))
            if len(deltas) != 4:
                raise ValueError("development positives are incomplete")
            thresholds[axis] = max(deltas) + 0.01
        guard_thresholds = {
            "dark_frame_fraction": 0.0,
            "magenta_frame_fraction": 0.0,
            "bright_frame_fraction": 0.0,
            "frozen_fraction": 0.95,
            "luma_acceleration_p95": 0.0,
            "high_frequency_proxy_ratio": 0.5,
            "near_freeze_temporal_max": 0.1 / 255,
            "near_freeze_ratio_max": 0.01,
            "near_freeze_reference_floor": 0.1 / 255,
        }
        # Diagnostics come from all decoded frames, unlike VideoReward's sparse
        # sampling.  Positive transcodes establish the tolerated baseline.
        positive_diagnostics = []
        for source in sources.values():
            if source["split"] == "development":
                for kind in ("identity", "transcode"):
                    control_id = f"{source['source_id']}--{kind}"
                    row = next(item for item in rows if item["control_id"] == control_id)
                    positive_diagnostics.append(row["result"]["diagnostics"])
        for key in ("dark_frame_fraction", "magenta_frame_fraction", "bright_frame_fraction"):
            guard_thresholds[key] = max(float(item[key]) for item in positive_diagnostics) + 1e-9
        guard_thresholds["luma_acceleration_p95"] = (
            max(float(item["luma_acceleration_p95"]) for item in positive_diagnostics) + 1e-6
        )
        detections, failures = [], []
        for source_id, source in sources.items():
            identity = scored[f"{source_id}--identity"]
            identity_row = next(
                item for item in rows if item["control_id"] == f"{source_id}--identity"
            )
            identity_diagnostics = identity_row["result"]["diagnostics"]
            for kind in CONTROL_KINDS:
                expected_detect = kind in _CONTROL_AXIS or kind in {
                    "copy_diversity",
                    "hash_tamper",
                    "missing",
                }
                if kind in {"identity", "transcode"}:
                    row = next(
                        item for item in rows if item["control_id"] == f"{source_id}--{kind}"
                    )
                    detected = bool(
                        _decoded_guard_hits(
                            identity_diagnostics, row["result"]["diagnostics"], guard_thresholds
                        )
                    )
                    for axis in ("VQ", "MQ", "TA"):
                        detected |= (
                            abs(identity[axis] - scored[f"{source_id}--{kind}"][axis])
                            > thresholds[axis]
                        )
                elif kind == "copy_diversity":
                    control = expected[f"{source_id}--{kind}"]
                    copies = control.get("decoded_seed_sha256")
                    detected = (
                        isinstance(copies, list) and len(copies) == 3 and len(set(copies)) == 1
                    )
                elif kind in {"hash_tamper", "missing"}:
                    detected = True
                else:
                    axis = _CONTROL_AXIS[kind]
                    detected = (
                        identity[axis] - scored[f"{source_id}--{kind}"][axis] > thresholds[axis]
                    )
                    row = next(
                        item for item in rows if item["control_id"] == f"{source_id}--{kind}"
                    )
                    diagnostics = row["result"]["diagnostics"]
                    if kind in {
                        "freeze",
                        "black_frame",
                        "magenta_frame",
                        "segment_corruption",
                        "flicker",
                        "blur",
                    }:
                        detected = kind in _decoded_guard_hits(
                            identity_diagnostics, diagnostics, guard_thresholds
                        )
                detections.append(
                    {
                        "source_id": source_id,
                        "split": source["split"],
                        "kind": kind,
                        "detected": detected,
                    }
                )
                if detected != expected_detect:
                    failures.append(f"{source_id}:{kind}")
        status = "QUALIFIED" if not failures else "FAILED"
        result = {
            "schema": SCHEMA,
            "qualification": status,
            "manifest_id": plan["manifest"]["manifest_id"],
            "control_plan_id": plan["control_plan_id"],
            "raw_receipt_id": raw["raw_receipt_id"],
            "evaluator_id": evaluator["evaluator_id"],
            "spec_hash": evaluator["spec_hash"],
            "thresholds": thresholds,
            "guard_thresholds": guard_thresholds,
            "detections": detections,
            "reasons": failures,
        }
        return {
            **result,
            "receipt_id": hash_canonical("forge/evaluator-qualification-result/v1", result),
        }
    except (KeyError, TypeError, ValueError) as error:
        return _result("INCOMPLETE", [str(error)], None, None)


def _interval(values: list[float], confidence: float) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("at least two independent confirmation sources are required")
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / math.sqrt(len(values))
    critical = float(student_t.ppf((1 + confidence) / 2, df=len(values) - 1))
    return {
        "n": len(values),
        "mean": mean,
        "lower": mean - critical * error,
        "upper": mean + critical * error,
        "confidence": confidence,
    }


def _upper_bound(values: list[float], confidence: float) -> dict[str, float | int]:
    """Return an explicitly one-sided upper Student-t bound."""
    if len(values) < 2:
        raise ValueError("at least two independent confirmation sources are required")
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "standard_error": error,
        "upper": mean + float(student_t.ppf(confidence, df=len(values) - 1)) * error,
        "one_sided_confidence": confidence,
    }


def _v2_guard_thresholds(
    rows: list[Mapping[str, Any]], sources: dict[str, Mapping[str, Any]]
) -> dict[str, float]:
    positives = []
    for source in sources.values():
        if source["split"] != "development":
            continue
        for kind in ("identity", "transcode"):
            row = next(
                item for item in rows if item["control_id"] == f"{source['source_id']}--{kind}"
            )
            positives.append(row["result"]["diagnostics"])
    if len(positives) != 8:
        raise ValueError("v2 decoded guard calibration is incomplete")
    thresholds = {
        key: max(float(item[key]) for item in positives) + 1e-9
        for key in ("dark_frame_fraction", "magenta_frame_fraction", "bright_frame_fraction")
    }
    thresholds["frozen_fraction"] = 0.95
    thresholds["luma_acceleration_p95"] = (
        max(float(item["luma_acceleration_p95"]) for item in positives) + 1e-6
    )
    thresholds.update(
        near_freeze_temporal_max=0.1 / 255,
        near_freeze_ratio_max=0.01,
        near_freeze_reference_floor=0.1 / 255,
    )
    return thresholds


def _finalize_qualification_v2(plan: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the v2 rule without changing historical v1 replay semantics."""
    try:
        if (
            plan.get("schema") != SCHEMA
            or plan.get("qualification_version") != QUALIFICATION_V2
            or plan.get("decision_rule") != V2_DECISION_RULE
            or not isinstance(plan.get("manifest"), Mapping)
        ):
            raise ValueError("v2 frozen control plan is invalid")
        body = {key: value for key, value in plan.items() if key != "control_plan_id"}
        if plan.get("control_plan_id") != hash_canonical("forge/evaluator-controls/v1", body):
            raise ValueError("v2 control plan hash is invalid")
        manifest = plan["manifest"]
        manifest_body = {key: value for key, value in manifest.items() if key != "manifest_id"}
        if (
            manifest.get("manifest_id") != hash_canonical(SCHEMA, manifest_body)
            or manifest.get("qualification_version") != QUALIFICATION_V2
            or manifest.get("decision_rule") != V2_DECISION_RULE
            or not isinstance(manifest.get("excluded_confirmation_lineages"), list)
            or manifest.get("source_eligibility")
            != {
                "schema": SOURCE_ELIGIBILITY_SCHEMA,
                "temporal_delta_floor": SOURCE_TEMPORAL_DELTA_FLOOR,
            }
        ):
            raise ValueError("v2 source manifest is invalid")
        sources_list = manifest.get("sources")
        required_source = {
            "source_id",
            "split",
            "family",
            "prompt",
            "video_sha256",
            "decoded_sha256",
            "temporal_delta",
            "lineage_id",
        }
        if (
            not isinstance(sources_list, list)
            or len(sources_list) != 12
            or any(
                not isinstance(source, Mapping)
                or set(source) != required_source
                or source["split"] not in {"development", "confirmation"}
                or not all(
                    isinstance(source[key], str) and source[key]
                    for key in (
                        "source_id",
                        "family",
                        "prompt",
                        "video_sha256",
                        "decoded_sha256",
                        "lineage_id",
                    )
                )
                or type(source["temporal_delta"]) not in (int, float)
                or not np.isfinite(source["temporal_delta"])
                or source["temporal_delta"] <= SOURCE_TEMPORAL_DELTA_FLOOR
                for source in sources_list
            )
        ):
            raise ValueError("v2 source evidence is incomplete")
        sources = {source["source_id"]: source for source in sources_list}
        if (
            len(sources) != 12
            or len({source["decoded_sha256"] for source in sources.values()}) != 12
        ):
            raise ValueError("v2 source identities are duplicated")
        families = {source["family"] for source in sources.values()}
        if len(families) != 4 or any(
            sum(
                source["split"] == "development" and source["family"] == family
                for source in sources.values()
            )
            != 1
            or sum(
                source["split"] == "confirmation" and source["family"] == family
                for source in sources.values()
            )
            != 2
            for family in families
        ):
            raise ValueError("v2 source split/family matrix is incomplete")
        lineages = [source["lineage_id"] for source in sources.values()]
        excluded = manifest["excluded_confirmation_lineages"]
        if (
            len(set(lineages)) != 12
            or len(set(excluded)) != len(excluded)
            or not excluded
            or any(not isinstance(item, str) or not item for item in excluded)
            or {
                source["lineage_id"]
                for source in sources.values()
                if source["split"] == "confirmation"
            }.intersection(excluded)
        ):
            raise ValueError("v2 semantic lineage declaration is invalid")
        raw_body = {key: value for key, value in raw.items() if key != "raw_receipt_id"}
        if (
            raw.get("schema") != SCHEMA
            or raw.get("control_plan_id") != plan["control_plan_id"]
            or raw.get("raw_receipt_id")
            != hash_canonical("forge/evaluator-controls-score/v1", raw_body)
        ):
            raise ValueError("v2 raw score receipt is invalid")
        evaluator = raw.get("evaluator")
        if not isinstance(evaluator, Mapping) or not all(
            isinstance(evaluator.get(key), str) and evaluator[key]
            for key in ("evaluator_id", "spec_hash")
        ):
            raise ValueError("v2 raw receipt lacks evaluator provenance")
        spec, runtime = evaluator.get("spec"), evaluator.get("runtime")
        if (
            not isinstance(spec, dict)
            or not isinstance(runtime, dict)
            or evaluator["spec_hash"] != hash_canonical("forge/launch-evaluator-spec/v1", spec)
            or evaluator["evaluator_id"]
            != hash_canonical("forge/videoreward-study-runtime/v1", spec, runtime)
        ):
            raise ValueError("v2 evaluator identity does not replay")
        expected = {control["control_id"]: control for control in plan.get("controls", [])}
        expected_pairs = {(source_id, kind) for source_id in sources for kind in CONTROL_KINDS}
        if (
            len(plan.get("controls", [])) != 144
            or len(expected) != 144
            or {(item.get("source_id"), item.get("kind")) for item in expected.values()}
            != expected_pairs
        ):
            raise ValueError("v2 frozen control matrix is incomplete")
        rows = raw.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 144
            or {row.get("control_id") for row in rows if isinstance(row, Mapping)} != set(expected)
        ):
            raise ValueError("v2 raw score receipt does not cover frozen controls")
        by_id = {row["control_id"]: row for row in rows}
        for source_id, source in sources.items():
            identity = expected[f"{source_id}--identity"]
            identity_row = by_id[f"{source_id}--identity"]
            identity_result = identity_row.get("result")
            if (
                identity.get("video_sha256") != source["video_sha256"]
                or identity.get("decoded_sha256") != source["decoded_sha256"]
                or not isinstance(identity_result, Mapping)
                or not isinstance(identity_result.get("diagnostics"), Mapping)
                or identity_result["diagnostics"].get("temporal_delta") != source["temporal_delta"]
            ):
                raise ValueError("v2 identity control does not bind its eligible source")
        scores: dict[str, dict[str, float]] = {}
        for control_id, control in expected.items():
            row = by_id[control_id]
            if control["kind"] == "missing":
                if row != {"control_id": control_id, "status": "MISSING_REJECTED"}:
                    raise ValueError("v2 missing control changed")
                continue
            if control["kind"] == "hash_tamper":
                if row != {"control_id": control_id, "status": "HASH_MISMATCH_REJECTED"}:
                    raise ValueError("v2 hash control changed")
                continue
            result = row.get("result")
            if (
                row.get("status") != "OK"
                or not isinstance(result, Mapping)
                or result.get("video_sha256") != control.get("video_sha256")
                or set(result.get("scores", {})) != {"VQ", "MQ", "TA"}
                or not isinstance(result.get("diagnostics"), Mapping)
                or any(
                    type(result["scores"][axis]) not in (int, float)
                    or not np.isfinite(result["scores"][axis])
                    for axis in ("VQ", "MQ", "TA")
                )
            ):
                raise ValueError("v2 scored control is invalid")
            scores[control_id] = {
                axis: float(result["scores"][axis]) for axis in ("VQ", "MQ", "TA")
            }
        thresholds = _v2_guard_thresholds(rows, sources)
        deterministic = ("freeze", "black_frame", "magenta_frame", "segment_corruption", "flicker")
        failures: list[str] = []
        for source_id in sources:
            identity = by_id[f"{source_id}--identity"]["result"]["diagnostics"]
            for kind in ("identity", "transcode"):
                diagnostics = by_id[f"{source_id}--{kind}"]["result"]["diagnostics"]
                if _decoded_guard_hits(identity, diagnostics, thresholds):
                    failures.append(f"{source_id}:{kind}:decoded_guard")
            for kind in deterministic:
                diagnostics = by_id[f"{source_id}--{kind}"]["result"]["diagnostics"]
                if kind not in _decoded_guard_hits(identity, diagnostics, thresholds):
                    failures.append(f"{source_id}:{kind}")
            copies = expected[f"{source_id}--copy_diversity"].get("decoded_seed_sha256")
            if not (isinstance(copies, list) and len(copies) == 3 and len(set(copies)) == 1):
                failures.append(f"{source_id}:copy_diversity")
        confirmation = [source for source in sources.values() if source["split"] == "confirmation"]
        evidence: dict[str, Any] = {"transcode": {}, "blur": {}, "semantic_mismatch": {}}
        inconclusive = False
        inconclusive_reasons: list[str] = []
        for axis, margin in V2_DECISION_RULE["quality_margins"].items():
            deltas = [
                scores[f"{source['source_id']}--transcode"][axis]
                - scores[f"{source['source_id']}--identity"][axis]
                for source in confirmation
            ]
            estimate = _interval(deltas, V2_DECISION_RULE["transcode"]["confidence"])
            evidence["transcode"][axis] = estimate
            if estimate["mean"] < -margin or estimate["mean"] > margin:
                failures.append(f"transcode:{axis}:point_outside_equivalence_margin")
            elif estimate["lower"] < -margin or estimate["upper"] > margin:
                inconclusive = True
                inconclusive_reasons.append(f"transcode:{axis}:equivalence_ci")
            evidence["transcode"][axis]["family_means"] = {
                family: statistics.fmean(
                    scores[f"{source['source_id']}--transcode"][axis]
                    - scores[f"{source['source_id']}--identity"][axis]
                    for source in confirmation
                    if source["family"] == family
                )
                for family in sorted(families)
            }
        blur_pass = False
        for axis in V2_DECISION_RULE["blur"]["axes"]:
            margin = V2_DECISION_RULE["quality_margins"][axis]
            deltas = [
                scores[f"{source['source_id']}--blur"][axis]
                - scores[f"{source['source_id']}--identity"][axis]
                for source in confirmation
            ]
            estimate = _upper_bound(deltas, V2_DECISION_RULE["blur"]["confidence"])
            evidence["blur"][axis] = estimate
            if estimate["upper"] < -margin:
                blur_pass = True
        family_means = {}
        for family in sorted(families):
            family_means[family] = {
                axis: statistics.fmean(
                    scores[f"{source['source_id']}--blur"][axis]
                    - scores[f"{source['source_id']}--identity"][axis]
                    for source in confirmation
                    if source["family"] == family
                )
                for axis in ("VQ", "TA")
            }
            if family_means[family]["VQ"] > 0 and family_means[family]["TA"] > 0:
                failures.append(f"blur:{family}:positive_both_axes")
        evidence["blur"]["family_means"] = family_means
        if not blur_pass:
            if all(
                evidence["blur"][axis]["mean"] >= -V2_DECISION_RULE["quality_margins"][axis]
                for axis in V2_DECISION_RULE["blur"]["axes"]
            ):
                failures.append("blur:point_not_materially_inferior")
            else:
                inconclusive = True
                inconclusive_reasons.append("blur:panel_ci")
        semantic = [
            scores[f"{source['source_id']}--semantic_mismatch"]["TA"]
            - scores[f"{source['source_id']}--identity"]["TA"]
            for source in confirmation
        ]
        semantic_estimate = _upper_bound(
            semantic, V2_DECISION_RULE["semantic_mismatch"]["confidence"]
        )
        evidence["semantic_mismatch"]["TA"] = semantic_estimate
        if semantic_estimate["mean"] >= -V2_DECISION_RULE["quality_margins"]["TA"]:
            failures.append("semantic_mismatch:TA:point_not_materially_inferior")
        elif semantic_estimate["upper"] >= -V2_DECISION_RULE["quality_margins"]["TA"]:
            inconclusive = True
            inconclusive_reasons.append("semantic_mismatch:TA:panel_ci")
        status = "FAILED" if failures else "INCONCLUSIVE" if inconclusive else "QUALIFIED"
        result = {
            "schema": SCHEMA,
            "qualification": status,
            "qualification_version": QUALIFICATION_V2,
            "decision_rule": copy.deepcopy(V2_DECISION_RULE),
            "manifest_id": manifest["manifest_id"],
            "control_plan_id": plan["control_plan_id"],
            "raw_receipt_id": raw["raw_receipt_id"],
            "evaluator_id": evaluator["evaluator_id"],
            "spec_hash": evaluator["spec_hash"],
            "guard_thresholds": thresholds,
            "evidence": evidence,
            "reasons": failures + inconclusive_reasons,
        }
        return {
            **result,
            "receipt_id": hash_canonical("forge/evaluator-qualification-result/v2", result),
        }
    except (KeyError, TypeError, ValueError) as error:
        body = {
            "schema": SCHEMA,
            "qualification": "INCOMPLETE",
            "qualification_version": QUALIFICATION_V2,
            "decision_rule": copy.deepcopy(V2_DECISION_RULE),
            "manifest_id": None,
            "evaluator_id": None,
            "reasons": [str(error)],
        }
        return {
            **body,
            "receipt_id": hash_canonical("forge/evaluator-qualification-result/v2", body),
        }


def finalize_qualification(plan: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch a frozen package without changing the historical v1 path."""
    if plan.get("schema") == "forge/evaluator-qualification-composite/v3":
        from forge.evaluation.qualification_composite import finalize_composite

        return finalize_composite(plan, raw)
    if plan.get("qualification_version") == QUALIFICATION_V2:
        return _finalize_qualification_v2(plan, raw)
    if plan.get("qualification_version") is None:
        return _finalize_qualification_v1(plan, raw)
    return _result("INCOMPLETE", ["unsupported qualification version"], None, None)


def replay_qualification_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the only qualification summary accepted by bridge/publication."""
    if set(package) != {"plan", "raw", "result"}:
        raise ValueError("qualification package must contain plan, raw and result")
    if not all(isinstance(package[key], Mapping) for key in package):
        raise ValueError("qualification package members must be objects")
    if package["plan"].get("schema") == "forge/evaluator-qualification-composite/v3":
        from forge.evaluation.qualification_composite import replay_composite_package

        actual = replay_composite_package(package)
        if actual.get("qualification") != "QUALIFIED":
            raise ValueError("qualification package is not qualified")
        return actual
    actual = finalize_qualification(package["plan"], package["raw"])
    if actual != package["result"]:
        raise ValueError("qualification package result does not replay")
    if actual.get("qualification") != "QUALIFIED":
        raise ValueError("qualification package is not qualified")
    return actual


def _result(
    status: str, reasons: list[str], manifest_id: str | None, evaluator_id: str | None
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "qualification": status,
        "manifest_id": manifest_id,
        "evaluator_id": evaluator_id,
        "reasons": reasons,
    }
    return {**body, "receipt_id": hash_canonical("forge/evaluator-qualification-result/v1", body)}


def score_controls(
    plan: Mapping[str, Any], control_root: Path, spec: Mapping[str, Any], assets: Path
) -> dict[str, Any]:
    """Run the pinned local VideoReward evaluator on prepared control MP4s."""
    if (
        plan.get("schema") != SCHEMA
        or not isinstance(plan.get("manifest"), Mapping)
        or not isinstance(plan.get("controls"), list)
    ):
        raise ValueError("invalid frozen control plan")
    body = {key: value for key, value in plan.items() if key != "control_plan_id"}
    if plan.get("control_plan_id") != hash_canonical("forge/evaluator-controls/v1", body):
        raise ValueError("control plan hash changed before scoring")
    for control in plan["controls"]:
        relative = Path(control["path"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
            raise ValueError("control media path must stay inside its frozen directory")
    from forge.evaluation.video_reward import load_inference

    score, evaluator_id, runtime = load_inference(dict(spec), assets)
    prompts = {item["source_id"]: item["prompt"] for item in plan["manifest"]["sources"]}
    rows = []
    for control in plan["controls"]:
        if control["kind"] == "missing":
            if (control_root / control["path"]).exists():
                raise ValueError("missing control unexpectedly exists")
            rows.append({"control_id": control["control_id"], "status": "MISSING_REJECTED"})
            continue
        if control["kind"] == "hash_tamper":
            path = control_root / control["path"]
            if not path.is_file() or sha256_file(path) == control["video_sha256"]:
                raise ValueError("hash tamper control did not mismatch")
            rows.append({"control_id": control["control_id"], "status": "HASH_MISMATCH_REJECTED"})
            continue
        path = control_root / control["path"]
        if not path.is_file() or sha256_file(path) != control["video_sha256"]:
            raise ValueError("prepared control is missing or changed")
        result = score(path, prompts[control["source_id"]], plan["manifest"]["geometry"])
        if result.get("video_sha256") != control["video_sha256"]:
            raise ValueError("scorer observed changed control bytes")
        rows.append({"control_id": control["control_id"], "status": "OK", "result": result})
    body = {
        "schema": SCHEMA,
        "control_plan_id": plan["control_plan_id"],
        "evaluator": {
            "evaluator_id": evaluator_id,
            "spec": dict(spec),
            "runtime": runtime,
            "spec_hash": hash_canonical("forge/launch-evaluator-spec/v1", dict(spec)),
        },
        "rows": rows,
    }
    return {**body, "raw_receipt_id": hash_canonical("forge/evaluator-controls-score/v1", body)}


def main() -> None:
    import argparse
    import json

    from forge.validator.launch_service import write_immutable_json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sources", type=Path, required=True)
    prepare.add_argument("--geometry", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--plan-out", type=Path, required=True)
    prepare.add_argument("--qualification-version", type=int, choices=(1, 2), default=1)
    prepare.add_argument("--excluded-confirmation-lineages", type=Path)
    score = commands.add_parser("score")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--control-root", type=Path, required=True)
    score.add_argument("--spec", type=Path, required=True)
    score.add_argument("--assets", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--raw", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        excluded = (
            json.loads(args.excluded_confirmation_lineages.read_text())
            if args.excluded_confirmation_lineages is not None
            else None
        )
        value = prepare_controls(
            json.loads(args.sources.read_text()),
            json.loads(args.geometry.read_text()),
            args.out,
            qualification_version=args.qualification_version,
            excluded_confirmation_lineages=excluded,
        )
        write_immutable_json(args.plan_out, value)
    elif args.command == "score":
        value = score_controls(
            json.loads(args.plan.read_text()),
            args.control_root,
            json.loads(args.spec.read_text()),
            args.assets,
        )
        write_immutable_json(args.out, value)
    else:
        value = finalize_qualification(
            json.loads(args.plan.read_text()), json.loads(args.raw.read_text())
        )
        write_immutable_json(args.out, value)


if __name__ == "__main__":
    main()
