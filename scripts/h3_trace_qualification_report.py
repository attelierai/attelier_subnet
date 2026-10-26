#!/usr/bin/env python3
"""Analyze closed H3 research receipts; not a remote-proof or reward module.

This helper only consumes already-captured local evidence.  A PASS means that
the supplied receipts meet this frozen, closed-profile research rule; it does
not establish miner identity, GPU ownership, or service of a user request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA = "forge/h3-trace-qualification-report/v1"
CALIBRATION_SCHEMA = "forge/h3-trace-qualification-calibration/v1"
HONEST_LABELS = {"honest-1", "honest-2"}
FULL_STEPS = set(range(20))
MEDIA_SCHEMAS = {"forge/h3-media-audit/v2", "forge/h3-media-audit/v3"}
DELIVERED_AUDIO_SAMPLES = 165888


class EvidenceError(ValueError):
    pass


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvidenceError("path_outside_root") from exc
    return path


def read_json(
    root: Path, value: str | Path, sources: dict[str, str]
) -> tuple[Path, dict[str, Any]]:
    path = contained(root, value)
    if not path.is_file():
        raise EvidenceError("missing_artifact")
    sources[str(path.relative_to(root))] = digest(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceError("invalid_json") from exc
    if not isinstance(data, dict):
        raise EvidenceError("artifact_not_object")
    return path, data


def finite(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise EvidenceError("nonfinite_or_negative_metric")
    return float(value)


def add_check(checks: dict[str, dict[str, Any]], name: str, status: str, **detail: Any) -> None:
    checks[name] = {"status": status, **detail}


def measured_metric(
    root: Path,
    path: Path,
    metric: str,
    sources: dict[str, str],
) -> float:
    _, receipt = read_json(root, path, sources)
    if receipt.get("status") != "MEASURED":
        raise EvidenceError("metric_not_measured")
    return finite(receipt.get(metric))


def media_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Return the evaluator/crop contract, not a candidate-controlled score."""
    if result.get("schema") not in MEDIA_SCHEMAS:
        raise EvidenceError("media_schema_unknown")
    audio = result.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("trimPolicy"), str):
        raise EvidenceError("media_crop_contract_missing")
    samples = {}
    for field in ("expectedSamples", "sourceSamples", "deliveredSamples"):
        value = audio.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvidenceError("media_sample_contract_invalid")
        samples[field] = value
    if (
        audio["trimPolicy"] != "pinned_comfy_video_duration"
        or samples["expectedSamples"] != 165334
        or samples["sourceSamples"] != 165600
    ):
        raise EvidenceError("media_crop_profile_mismatch")
    if samples["deliveredSamples"] < samples["expectedSamples"]:
        raise EvidenceError("media_crop_contract_invalid")
    if samples["deliveredSamples"] != DELIVERED_AUDIO_SAMPLES:
        raise EvidenceError("media_delivered_samples_profile_mismatch")
    return {
        "schema": result["schema"],
        "trimPolicy": audio["trimPolicy"],
        **samples,
    }


def verdict(*, complete: bool, failed: bool) -> str:
    if failed:
        return "FAIL"
    if complete and not failed:
        return "PASS"
    return "INCONCLUSIVE"


def inspect_case(
    root: Path,
    row: dict[str, Any],
    thresholds: dict[str, float] | None,
    expected_media_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return independent run-level evidence. Labels are deliberately unused here."""
    required = {
        "id",
        "lineageId",
        "split",
        "label",
        "manifest",
        "trustedManifest",
        "numericDirectory",
        "challengeReceipt",
        "mediaDirectory",
    }
    if not isinstance(row, dict) or not required.issubset(row):
        raise EvidenceError("case_schema")
    sources: dict[str, str] = {}
    checks: dict[str, dict[str, Any]] = {}
    base_failures = False
    full_model_failures = False
    sampled_model_failures = False
    missing = False
    valid_replays = 0
    full_replays = False
    partial_replays = False
    partial_coverage = False
    all_predictions_match = False
    schedule_ok = False
    numeric_ok = False
    media_ok = False
    observed_media_contract: dict[str, Any] | None = None

    try:
        manifest_path, manifest = read_json(root, row["manifest"], sources)
        trusted_path, trusted = read_json(root, row["trustedManifest"], sources)
        candidate_trace = manifest.get("trace")
        trusted_trace = trusted.get("trace")
        if not isinstance(candidate_trace, list) or not isinstance(trusted_trace, list):
            raise EvidenceError("trace_missing")
        if len(candidate_trace) != 20 or len(trusted_trace) != 20:
            add_check(
                checks,
                "schedule",
                "FAIL",
                reason="approved_20_step_schedule_required",
                candidateSteps=len(candidate_trace),
                trustedSteps=len(trusted_trace),
            )
            base_failures = True
        elif manifest.get("inputHash") != trusted.get("inputHash"):
            add_check(checks, "schedule", "FAIL", reason="input_hash_mismatch")
            base_failures = True
        else:
            candidate_sigmas = [
                finite(x.get("sigma")) for x in candidate_trace if isinstance(x, dict)
            ]
            trusted_sigmas = [finite(x.get("sigma")) for x in trusted_trace if isinstance(x, dict)]
            if (
                len(candidate_sigmas) != 20
                or len(trusted_sigmas) != 20
                or candidate_sigmas != trusted_sigmas
            ):
                add_check(checks, "schedule", "FAIL", reason="sigma_schedule_mismatch")
                base_failures = True
            else:
                add_check(checks, "schedule", "PASS", steps=20)
                schedule_ok = True
    except EvidenceError as exc:
        add_check(checks, "schedule", "MISSING", reason=str(exc))
        missing = True
        manifest_path = trusted_path = None
        manifest = trusted = {}

    transition = endpoint = None
    try:
        numeric_dir = contained(root, row["numericDirectory"])
        transition = measured_metric(
            root, numeric_dir / "transition.json", "maxRelativeRms", sources
        )
        endpoint = measured_metric(root, numeric_dir / "endpoint.json", "relativeRms", sources)
        over = thresholds and (
            transition > thresholds["numericMaxRelativeRms"]
            or endpoint > thresholds["numericMaxRelativeRms"]
        )
        if over:
            add_check(checks, "numeric", "FAIL", transition=transition, endpoint=endpoint)
            base_failures = True
        else:
            add_check(checks, "numeric", "PASS", transition=transition, endpoint=endpoint)
            numeric_ok = True
    except EvidenceError as exc:
        add_check(checks, "numeric", "MISSING", reason=str(exc))
        missing = True

    try:
        media_dir = contained(root, row["mediaDirectory"])
        _, result = read_json(root, media_dir / "result.json", sources)
        _, completion = read_json(root, media_dir / "completion.json", sources)
        if manifest_path is None:
            raise EvidenceError("manifest_unavailable")
        output = manifest.get("output")
        if (
            not isinstance(output, dict)
            or not isinstance(output.get("media"), str)
            or not isinstance(output.get("sha256"), str)
        ):
            raise EvidenceError("manifest_media_missing")
        media_path = contained(root, output["media"])
        media_hash = digest(media_path)
        sources[str(media_path.relative_to(root))] = media_hash
        manifest_hash = digest(manifest_path)
        if (
            output["sha256"] != media_hash
            or result.get("status") != "MEASURED"
            or result.get("mediaSha256") != media_hash
            or completion.get("manifestSha256") != manifest_hash
            or completion.get("mediaSha256") != media_hash
            or completion.get("resultSha256") != digest(media_dir / "result.json")
        ):
            raise EvidenceError("media_binding_mismatch")
        observed_media_contract = media_contract(result)
        if (
            expected_media_contract is not None
            and observed_media_contract != expected_media_contract
        ):
            raise EvidenceError("media_evaluator_contract_mismatch")
        video = finite(result.get("video", {}).get("maxFrameRmse"))
        audio = finite(result.get("audio", {}).get("relativeRms"))
        over = thresholds and (
            video > thresholds["videoMaxFrameRmse"] or audio > thresholds["audioRelativeRms"]
        )
        if over:
            add_check(checks, "media", "FAIL", video=video, audio=audio)
            base_failures = True
        else:
            add_check(checks, "media", "PASS", video=video, audio=audio)
            media_ok = True
    except (EvidenceError, AttributeError, KeyError, TypeError) as exc:
        add_check(checks, "media", "MISSING", reason=str(exc))
        missing = True

    replay_by_step: dict[int, bool] = {}
    measured_replay_steps: set[int] = set()
    sample_steps: set[int] = set()
    receipt_steps: set[int] = set()
    try:
        receipt_path, receipt = read_json(root, row["challengeReceipt"], sources)
        if manifest_path is None or trusted_path is None:
            raise EvidenceError("manifest_unavailable")
        if (
            receipt.get("manifestSha256") != digest(manifest_path)
            or receipt.get("conditioningManifestSha256") != digest(trusted_path)
            or receipt.get("inputHash") != manifest.get("inputHash")
        ):
            raise EvidenceError("challenge_binding_mismatch")
        if not isinstance(receipt.get("auditPolicy"), str):
            raise EvidenceError("challenge_policy_missing")
        raw_steps = receipt.get("steps")
        raw_sample = receipt.get("comparisonSampleSteps")
        bundles = receipt.get("bundles")
        if (
            not isinstance(raw_steps, list)
            or not all(isinstance(step, int) and step in FULL_STEPS for step in raw_steps)
            or len(set(raw_steps)) != len(raw_steps)
            or not isinstance(raw_sample, list)
            or not all(isinstance(step, int) and step in FULL_STEPS for step in raw_sample)
            or len(set(raw_sample)) != len(raw_sample)
            or not isinstance(bundles, list)
        ):
            raise EvidenceError("challenge_steps_invalid")
        receipt_steps = set(raw_steps)
        sample_steps = set(raw_sample)
        if len(sample_steps) != 3 or not sample_steps.issubset(receipt_steps):
            raise EvidenceError("comparison_sample_steps_invalid")
        seen: set[int] = set()
        for bundle in bundles:
            if (
                not isinstance(bundle, dict)
                or not isinstance(bundle.get("step"), int)
                or bundle["step"] not in receipt_steps
                or bundle["step"] in seen
                or not isinstance(bundle.get("path"), str)
                or not isinstance(bundle.get("sha256"), str)
            ):
                raise EvidenceError("challenge_bundle_invalid")
            seen.add(bundle["step"])
            bundle_path = contained(receipt_path.parent, bundle["path"])
            # Challenge bundles are part of the case evidence and must remain under root.
            bundle_path.relative_to(root)
            if digest(bundle_path) != bundle["sha256"]:
                raise EvidenceError("challenge_bundle_custody")
            sources[str(bundle_path.relative_to(root))] = bundle["sha256"]
            replay_path = receipt_path.parent / "replay" / str(bundle["step"]) / "result.json"
            try:
                _, replay = read_json(root, replay_path, sources)
                outcome = replay.get("outcome")
                if (
                    replay.get("bundleSha256") != bundle["sha256"]
                    or replay.get("candidateLabel") != "canonicalchallenge"
                    or not isinstance(outcome, dict)
                    or outcome.get("status") != "MEASURED"
                ):
                    raise EvidenceError("replay_binding_mismatch")
                value = finite(outcome.get("relativeRms"))
                valid_replays += 1
                measured_replay_steps.add(bundle["step"])
                if thresholds and value > thresholds["numericMaxRelativeRms"]:
                    add_check(checks, f"replay:{bundle['step']}", "FAIL", relativeRms=value)
                    full_model_failures = True
                    sampled_model_failures |= bundle["step"] in sample_steps
                    replay_by_step[bundle["step"]] = False
                else:
                    add_check(checks, f"replay:{bundle['step']}", "PASS", relativeRms=value)
                    replay_by_step[bundle["step"]] = True
            except EvidenceError as exc:
                add_check(checks, f"replay:{bundle['step']}", "MISSING", reason=str(exc))
                missing = True
        if seen != receipt_steps:
            raise EvidenceError("challenge_bundles_do_not_cover_steps")
        partial_coverage = sample_steps.issubset(measured_replay_steps)
        full_replays = receipt_steps == FULL_STEPS and FULL_STEPS.issubset(measured_replay_steps)
        partial_replays = partial_coverage and all(
            replay_by_step.get(step) for step in sample_steps
        )
        all_predictions_match = full_replays and all(
            replay_by_step.get(step) for step in FULL_STEPS
        )
        add_check(
            checks,
            "challenge",
            "PASS" if all_predictions_match else "FAIL" if full_replays else "MISSING",
            steps=sorted(receipt_steps),
            comparisonSampleSteps=sorted(sample_steps),
            validReplays=valid_replays,
        )
        if not full_replays:
            missing = True
    except EvidenceError as exc:
        add_check(checks, "challenge", "MISSING", reason=str(exc), validReplays=valid_replays)
        missing = True

    partial_complete = schedule_ok and numeric_ok and media_ok and partial_replays
    full_complete = partial_complete and all_predictions_match
    return {
        "id": row["id"],
        "lineageId": row["lineageId"],
        "split": row["split"],
        "label": row["label"],
        "checks": checks,
        "status": verdict(complete=full_complete, failed=base_failures or full_model_failures),
        "partialStatus": verdict(
            complete=partial_complete, failed=base_failures or sampled_model_failures
        ),
        "coverage": {
            "validReplays": valid_replays,
            "fullTwentyReplayCoverage": full_replays,
            "allPredictionsMatch": all_predictions_match,
            "comparisonSampleCoverage": partial_coverage,
            "allComparisonPredictionsMatch": partial_replays,
            "missingEvidence": missing,
        },
        "mediaContract": observed_media_contract,
        "sources": dict(sorted(sources.items())),
    }


def load_cases(root: Path, cases_path: Path) -> tuple[Path, list[dict[str, Any]], str]:
    path = contained(root, cases_path)
    data = json.loads(path.read_text())
    rows = data["cases"] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise EvidenceError("cases_schema")
    identifiers = [row.get("id") for row in rows]
    if len(identifiers) != len(set(identifiers)) or not all(
        isinstance(value, str) for value in identifiers
    ):
        raise EvidenceError("case_ids_invalid")
    calibration_lineages: set[str] = set()
    holdout_lineages: set[str] = set()
    capture_ids: set[str] = set()
    capture_hashes: set[str] = set()
    for row in rows:
        lineage = row.get("lineageId")
        if not isinstance(lineage, str) or not lineage:
            raise EvidenceError("case_lineage_invalid")
        if row.get("split") == "holdout":
            holdout_lineages.add(lineage)
        if row.get("split") != "calibration":
            continue
        calibration_lineages.add(lineage)
        if row.get("label") not in HONEST_LABELS:
            continue
        # Row IDs and renamed paths cannot turn one stored capture into six
        # honest repetitions. Check this on report as well as calibration so
        # a previously frozen contaminated set cannot bypass the gate.
        manifest_path, manifest = read_json(root, row["manifest"], {})
        capture_id = manifest.get("runId")
        capture_hash = digest(manifest_path)
        if not isinstance(capture_id, str) or not capture_id:
            raise EvidenceError("calibration_capture_identity_missing")
        if capture_id in capture_ids or capture_hash in capture_hashes:
            raise EvidenceError("calibration_capture_reused")
        capture_ids.add(capture_id)
        capture_hashes.add(capture_hash)
    if calibration_lineages & holdout_lineages:
        raise EvidenceError("calibration_holdout_lineage_overlap")
    return path, rows, digest(path)


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def derive_frozen_media_contract(
    root: Path, rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Derive the legacy evaluator contract only from freeze-hash-bound receipts."""
    case_ids = calibration.get("caseIds")
    source_hashes = calibration.get("caseSourceHashes")
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != 6
        or len(set(case_ids)) != 6
        or not isinstance(source_hashes, dict)
    ):
        raise EvidenceError("calibration_media_sources_missing")
    rows_by_id = {row.get("id"): row for row in rows}
    contracts = []
    for case_id in case_ids:
        row = rows_by_id.get(case_id)
        frozen = source_hashes.get(case_id)
        if (
            not isinstance(row, dict)
            or row.get("split") != "calibration"
            or row.get("label") not in HONEST_LABELS
            or not isinstance(frozen, dict)
        ):
            raise EvidenceError("calibration_media_case_invalid")
        result_path = contained(root, row["mediaDirectory"]) / "result.json"
        relative = str(result_path.relative_to(root))
        if frozen.get(relative) != digest(result_path):
            raise EvidenceError("calibration_media_receipt_changed")
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as exc:
            raise EvidenceError("calibration_media_receipt_invalid") from exc
        contracts.append(media_contract(result))
    if any(contract != contracts[0] for contract in contracts):
        raise EvidenceError("calibration_media_contract_mismatch")
    stored = calibration.get("mediaContract")
    if stored is not None:
        if not isinstance(stored, dict):
            raise EvidenceError("calibration_media_contract_invalid")
        try:
            normalized = media_contract({"schema": stored.get("schema"), "audio": stored})
        except EvidenceError as exc:
            raise EvidenceError("calibration_media_contract_invalid") from exc
        if normalized != contracts[0]:
            raise EvidenceError("calibration_media_contract_changed")
    return contracts[0]


def calibrate(root: Path, cases_path: Path, out: Path) -> dict[str, Any]:
    cases_file, rows, cases_sha = load_cases(root, cases_path)
    selected = [
        row
        for row in rows
        if row.get("split") == "calibration" and row.get("label") in HONEST_LABELS
    ]
    if len(selected) != 6 or {row["lineageId"] for row in selected}.__len__() != 3:
        raise EvidenceError("requires_6_honest_cases_across_3_lineages")
    by_lineage: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        by_lineage.setdefault(row["lineageId"], []).append(row)
    if any(
        sorted(row["label"] for row in group) != sorted(HONEST_LABELS)
        for group in by_lineage.values()
    ):
        raise EvidenceError("requires_two_honest_variants_per_lineage")
    inspected = [inspect_case(root, row, thresholds=None) for row in selected]
    if any(item["status"] != "PASS" for item in inspected):
        raise EvidenceError("calibration_requires_6_complete_full_receipts")
    contracts = [item["mediaContract"] for item in inspected]
    if any(contract is None or contract != contracts[0] for contract in contracts):
        raise EvidenceError("calibration_media_contract_mismatch")
    numeric = []
    for item in inspected:
        numeric.extend(
            [item["checks"]["numeric"]["transition"], item["checks"]["numeric"]["endpoint"]]
        )
        numeric.extend(
            check["relativeRms"]
            for name, check in item["checks"].items()
            if name.startswith("replay:")
        )
    video = [item["checks"]["media"]["video"] for item in inspected]
    audio = [item["checks"]["media"]["audio"] for item in inspected]
    result = {
        "schema": CALIBRATION_SCHEMA,
        "status": "CALIBRATED",
        "claim": "frozen_heuristic_thresholds_not_statistical_certification",
        "casesPath": str(cases_file.relative_to(root)),
        "casesSha256": cases_sha,
        "caseSourceHashes": {item["id"]: item["sources"] for item in inspected},
        "caseIds": [item["id"] for item in inspected],
        "lineageIds": sorted(by_lineage),
        "mediaContract": contracts[0],
        "thresholds": {
            "numericMaxRelativeRms": max(numeric) * 2 + 1e-6,
            "videoMaxFrameRmse": max(video) * 1.5 + 0.001,
            "audioRelativeRms": max(audio) * 1.5 + 0.005,
        },
    }
    write_exclusive(contained(root, out), result)
    return result


def report(root: Path, cases_path: Path, calibration_path: Path, out: Path) -> dict[str, Any]:
    _, rows, cases_sha = load_cases(root, cases_path)
    calibration_file, calibration = read_json(root, calibration_path, {})
    if calibration.get("schema") != CALIBRATION_SCHEMA or calibration.get("status") != "CALIBRATED":
        raise EvidenceError("calibration_schema")
    raw_thresholds = calibration.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise EvidenceError("calibration_thresholds")
    if calibration.get("casesSha256") != cases_sha:
        raise EvidenceError("calibration_cases_changed")
    expected_media_contract = derive_frozen_media_contract(root, rows, calibration)
    thresholds = {
        name: finite(raw_thresholds.get(name))
        for name in ("numericMaxRelativeRms", "videoMaxFrameRmse", "audioRelativeRms")
    }
    inspected = [inspect_case(root, row, thresholds, expected_media_contract) for row in rows]
    frozen_sources = calibration.get("caseSourceHashes")
    if not isinstance(frozen_sources, dict):
        raise EvidenceError("calibration_sources_missing")
    inspected_by_id = {item["id"]: item for item in inspected}
    for case_id in calibration.get("caseIds", []):
        if (
            case_id not in inspected_by_id
            or frozen_sources.get(case_id) != inspected_by_id[case_id]["sources"]
        ):
            raise EvidenceError("calibration_sources_changed")
    counts: dict[str, int] = {}
    counts_by_label: dict[str, dict[str, int]] = {}
    for item in inspected:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        label_counts = counts_by_label.setdefault(item["label"], {})
        label_counts[item["status"]] = label_counts.get(item["status"], 0) + 1
    result = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "claim": "closed_trusted_research_receipt_analysis_not_remote_proof_or_reward",
        "casesSha256": cases_sha,
        "calibrationSha256": digest(calibration_file),
        "mediaContract": expected_media_contract,
        "thresholds": thresholds,
        "rows": inspected,
        "counts": counts,
        "countsByLabel": counts_by_label,
        "independentLineages": {
            "all": len({item["lineageId"] for item in inspected}),
            "bySplit": {
                split: len({item["lineageId"] for item in inspected if item["split"] == split})
                for split in sorted({item["split"] for item in inspected})
            },
        },
        "limits": [
            "one run and lineage is one unit; calls and frames are not independent samples",
            "labels are postscore reporting groups only and never choose a row verdict",
            "partialStatus covers only three comparisonSampleSteps and is not full coverage",
            "no GPU ownership, remote-service, or reward claim",
        ],
    }
    write_exclusive(contained(root, out), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "report"):
        command = subcommands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--cases", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "report":
            command.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = (
            calibrate(root, args.cases, args.out)
            if args.command == "calibrate"
            else report(root, args.cases, args.calibration, args.out)
        )
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "INCONCLUSIVE", "reason": str(exc)}))
        raise SystemExit(2) from exc
    print(json.dumps({"status": result["status"], "out": str(contained(root, args.out))}))


if __name__ == "__main__":
    main()
