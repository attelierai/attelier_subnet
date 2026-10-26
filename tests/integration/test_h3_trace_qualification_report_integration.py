from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def case(root: Path, run_id: str, lineage: str, split: str, label: str, *, steps: int = 20) -> dict:
    trace = [{"sigma": float(20 - index)} for index in range(steps)]
    trusted_trace = [{"sigma": float(20 - index)} for index in range(20)]
    media = root / "media" / f"{run_id}.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(run_id.encode())
    manifest = root / "captures" / run_id / "manifest.json"
    trusted = root / "trusted" / lineage / "manifest.json"
    write(
        manifest,
        {
            "runId": run_id,
            "inputHash": f"input-{lineage}",
            "trace": trace,
            "output": {"media": str(media.relative_to(root)), "sha256": sha(media)},
        },
    )
    write(trusted, {"inputHash": f"input-{lineage}", "trace": trusted_trace})
    numeric = root / "numerics" / run_id
    write(numeric / "transition.json", {"status": "MEASURED", "maxRelativeRms": 1e-8})
    write(numeric / "endpoint.json", {"status": "MEASURED", "relativeRms": 2e-8})
    audit = root / "media-audits" / run_id
    result = audit / "result.json"
    write(
        result,
        {
            "schema": "forge/h3-media-audit/v2",
            "status": "MEASURED",
            "mediaSha256": sha(media),
            "video": {"maxFrameRmse": 0.0001},
            "audio": {
                "relativeRms": 0.0002,
                "trimPolicy": "pinned_comfy_video_duration",
                "expectedSamples": 165334,
                "sourceSamples": 165600,
                "deliveredSamples": 165888,
            },
        },
    )
    write(
        audit / "completion.json",
        {
            "manifestSha256": sha(manifest),
            "mediaSha256": sha(media),
            "resultSha256": sha(result),
        },
    )
    challenge = root / "postcommit" / f"full-{run_id}" / "receipt.json"
    bundles = []
    for step in range(20):
        bundle = challenge.parent / f"candidate-{step:03d}.pt"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(f"{run_id}-{step}".encode())
        bundles.append({"step": step, "path": bundle.name, "sha256": sha(bundle)})
        write(
            challenge.parent / "replay" / str(step) / "result.json",
            {
                "bundleSha256": sha(bundle),
                "candidateLabel": "canonicalchallenge",
                "outcome": {"status": "MEASURED", "relativeRms": 1e-8},
            },
        )
    write(
        challenge,
        {
            "manifestSha256": sha(manifest),
            "conditioningManifestSha256": sha(trusted),
            "inputHash": f"input-{lineage}",
            "auditPolicy": "full-20",
            "steps": list(range(20)),
            "comparisonSampleSteps": [0, 10, 19],
            "bundles": bundles,
        },
    )
    return {
        "id": run_id,
        "lineageId": lineage,
        "split": split,
        "label": label,
        "manifest": str(manifest.relative_to(root)),
        "trustedManifest": str(trusted.relative_to(root)),
        "numericDirectory": str(numeric.relative_to(root)),
        "challengeReceipt": str(challenge.relative_to(root)),
        "mediaDirectory": str(audit.relative_to(root)),
    }


def test_qualification_report_freezes_complete_calibration_and_flags_native_step_profile(tmp_path):
    cases = []
    for lineage in ("lineage-a", "lineage-b", "lineage-c"):
        cases.append(case(tmp_path, f"{lineage}-h1", lineage, "calibration", "honest-1"))
        cases.append(case(tmp_path, f"{lineage}-h2", lineage, "calibration", "honest-2"))
    cases.append(case(tmp_path, "holdout-forged", "lineage-forged", "holdout", "forged"))
    cases.append(
        case(tmp_path, "holdout-steps10", "lineage-holdout", "holdout", "steps10", steps=10)
    )
    (tmp_path / "postcommit" / "full-holdout-steps10" / "receipt.json").unlink()
    cases_path = tmp_path / "cases.json"
    write(cases_path, {"cases": cases})
    calibration = tmp_path / "calibration.json"
    script = "scripts/verification/h3_trace_qualification_report.py"
    calibrate = [
        sys.executable,
        script,
        "calibrate",
        "--root",
        str(tmp_path),
        "--cases",
        str(cases_path),
        "--out",
        str(calibration),
    ]
    assert subprocess.run(calibrate, capture_output=True, text=True).returncode == 0
    frozen = json.loads(calibration.read_text())
    assert frozen["status"] == "CALIBRATED"
    assert len(frozen["caseSourceHashes"]) == 6
    assert frozen["mediaContract"]["schema"] == "forge/h3-media-audit/v2"
    assert subprocess.run(calibrate, capture_output=True, text=True).returncode != 0

    report_path = tmp_path / "report.json"
    report = [
        sys.executable,
        script,
        "report",
        "--root",
        str(tmp_path),
        "--cases",
        str(cases_path),
        "--calibration",
        str(calibration),
        "--out",
        str(report_path),
    ]
    assert subprocess.run(report, capture_output=True, text=True).returncode == 0
    rows = {row["id"]: row for row in json.loads(report_path.read_text())["rows"]}
    assert rows["lineage-a-h1"]["status"] == "PASS"
    bad = rows["holdout-steps10"]
    assert bad["status"] == "FAIL"
    assert bad["checks"]["schedule"]["reason"] == "approved_20_step_schedule_required"
    assert bad["coverage"]["validReplays"] == 0

    legacy_calibration = tmp_path / "legacy-calibration.json"
    legacy = dict(frozen)
    legacy.pop("mediaContract")
    write(legacy_calibration, legacy)
    legacy_report = tmp_path / "legacy-report.json"
    legacy_command = [
        *report[:-4],
        "--calibration",
        str(legacy_calibration),
        "--out",
        str(legacy_report),
    ]
    assert subprocess.run(legacy_command, capture_output=True, text=True).returncode == 0
    legacy_rows = {row["id"]: row for row in json.loads(legacy_report.read_text())["rows"]}
    assert {key: value["status"] for key, value in legacy_rows.items()} == {
        key: value["status"] for key, value in rows.items()
    }
    assert legacy_rows["lineage-a-h1"]["mediaContract"] == rows["lineage-a-h1"]["mediaContract"]

    forged_media = tmp_path / "media-audits" / "holdout-forged"
    result = json.loads((forged_media / "result.json").read_text())
    result["schema"] = "forge/h3-media-audit/v3"
    write(forged_media / "result.json", result)
    write(
        forged_media / "completion.json",
        {
            "manifestSha256": sha(tmp_path / "captures" / "holdout-forged" / "manifest.json"),
            "mediaSha256": result["mediaSha256"],
            "resultSha256": sha(forged_media / "result.json"),
        },
    )
    version_report = tmp_path / "version-report.json"
    version_command = [*report[:-1], str(version_report)]
    assert subprocess.run(version_command, capture_output=True, text=True).returncode == 0
    rows = {row["id"]: row for row in json.loads(version_report.read_text())["rows"]}
    assert rows["holdout-forged"]["status"] == "INCONCLUSIVE"
    assert (
        rows["holdout-forged"]["checks"]["media"]["reason"] == "media_evaluator_contract_mismatch"
    )
    result["schema"] = "forge/h3-media-audit/v2"
    write(forged_media / "result.json", result)
    write(
        forged_media / "completion.json",
        {
            "manifestSha256": sha(tmp_path / "captures" / "holdout-forged" / "manifest.json"),
            "mediaSha256": result["mediaSha256"],
            "resultSha256": sha(forged_media / "result.json"),
        },
    )

    forged_replay = tmp_path / "postcommit" / "full-holdout-forged" / "replay" / "7" / "result.json"
    replay = json.loads(forged_replay.read_text())
    replay["outcome"]["relativeRms"] = 0.1
    write(forged_replay, replay)
    outside_sample_report = tmp_path / "outside-sample-report.json"
    outside_sample = [*report[:-1], str(outside_sample_report)]
    assert subprocess.run(outside_sample, capture_output=True, text=True).returncode == 0
    rows = {row["id"]: row for row in json.loads(outside_sample_report.read_text())["rows"]}
    assert rows["holdout-forged"]["status"] == "FAIL"
    assert rows["holdout-forged"]["partialStatus"] == "PASS"
    assert rows["holdout-forged"]["coverage"]["fullTwentyReplayCoverage"] is True
    assert rows["holdout-forged"]["coverage"]["allPredictionsMatch"] is False

    forged_receipt = tmp_path / "postcommit" / "full-holdout-forged" / "receipt.json"
    receipt = json.loads(forged_receipt.read_text())
    receipt["comparisonSampleSteps"] = [7, 10, 19]
    write(forged_receipt, receipt)
    selected_sample_report = tmp_path / "selected-sample-report.json"
    selected_sample = [*report[:-1], str(selected_sample_report)]
    assert subprocess.run(selected_sample, capture_output=True, text=True).returncode == 0
    rows = {row["id"]: row for row in json.loads(selected_sample_report.read_text())["rows"]}
    assert rows["holdout-forged"]["status"] == "FAIL"
    assert rows["holdout-forged"]["partialStatus"] == "FAIL"

    mixed_result = tmp_path / "media-audits" / "lineage-a-h1" / "result.json"
    mixed = json.loads(mixed_result.read_text())
    mixed["schema"] = "forge/h3-media-audit/v3"
    write(mixed_result, mixed)
    mixed_completion = tmp_path / "media-audits" / "lineage-a-h1" / "completion.json"
    write(
        mixed_completion,
        {
            "manifestSha256": sha(tmp_path / "captures" / "lineage-a-h1" / "manifest.json"),
            "mediaSha256": mixed["mediaSha256"],
            "resultSha256": sha(mixed_result),
        },
    )
    mixed_calibration = tmp_path / "mixed-calibration.json"
    mixed_command = [*calibrate[:-1], str(mixed_calibration)]
    assert subprocess.run(mixed_command, capture_output=True, text=True).returncode != 0
    changed_receipt_report = tmp_path / "changed-receipt-report.json"
    changed_receipt_command = [*report[:-1], str(changed_receipt_report)]
    assert subprocess.run(changed_receipt_command, capture_output=True, text=True).returncode != 0


@pytest.mark.parametrize("contamination", ["reused_capture", "holdout_lineage"])
def test_calibration_rejects_recounted_captures_and_holdout_leakage(tmp_path, contamination):
    cases = [
        case(tmp_path, f"{lineage}-{variant}", lineage, "calibration", variant)
        for lineage in ("one", "two", "three")
        for variant in ("honest-1", "honest-2")
    ]
    if contamination == "reused_capture":
        # Six distinct row IDs and three declared lineages, but one capture.
        # Copying the manifest to a new pathname must not create another sample.
        for row in cases[1:]:
            for field in (
                "trustedManifest", "numericDirectory", "challengeReceipt", "mediaDirectory"
            ):
                row[field] = cases[0][field]
            (tmp_path / row["manifest"]).write_bytes((tmp_path / cases[0]["manifest"]).read_bytes())
    else:
        cases.append(case(tmp_path, "held-out", "one", "holdout", "honest-1"))
    write(tmp_path / "cases.json", {"cases": cases})
    output = tmp_path / "calibration.json"
    completed = subprocess.run(
        [sys.executable, "scripts/verification/h3_trace_qualification_report.py", "calibrate",
         "--root", str(tmp_path), "--cases", str(tmp_path / "cases.json"),
         "--out", str(output)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["status"] == "INCONCLUSIVE"
    assert not output.exists()
