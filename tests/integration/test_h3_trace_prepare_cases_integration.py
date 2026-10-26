"""The case preparer reconciles an indexed lineage without GPU work."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_preparer_uses_cases_as_the_complete_index_and_preserves_complete_receipts(tmp_path):
    root, lineage = tmp_path / "evidence", "lineage-01"
    trusted = root / "trusted" / "manifest.json"
    _write(trusted, {"inputHash": "input-01"})
    rows = []
    suffixes = (
        "honest-1",
        "honest-2",
        "weights-fp8",
        "lora20",
        "steps10",
        "steps3",
        "padded-10-as-20",
        "recurrence-forged",
        "sidecar-output",
        "skip7",
    )
    for suffix in suffixes:
        case_id = f"{lineage}-{suffix}"
        manifest = root / "captures" / case_id / "manifest.json"
        _write(manifest, {"inputHash": "input-01"})
        numeric = root / "numeric" / case_id
        _write(numeric / "transition.json", {"status": "MEASURED"})
        _write(numeric / "endpoint.json", {"status": "MEASURED"})
        receipt = root / "postcommit" / case_id / "receipt.json"
        _write(
            receipt,
            {
                "manifestSha256": _sha(manifest),
                "conditioningManifestSha256": _sha(trusted),
                "inputHash": "input-01",
            },
        )
        rows.append(
            {
                "id": case_id,
                "lineageId": lineage,
                "label": suffix,
                "manifest": str(manifest.relative_to(root)),
                "trustedManifest": str(trusted.relative_to(root)),
                "numericDirectory": str(numeric.relative_to(root)),
                "challengeReceipt": str(receipt.relative_to(root)),
                "mediaDirectory": f"media/{case_id}",
            }
        )
    adversaries = root / "adversaries-v2" / lineage
    for relative in (
        "padded-10-as-20/manifest.json",
        "sidecar-output/manifest.json",
        "recurrence-forged/manifest.json",
        "operator-ground-truth.json",
        "recurrence-forged/operator-ground-truth.json",
    ):
        _write(adversaries / relative, {})
    cases = tmp_path / "qualification-cases.json"
    _write(cases, {"cases": rows})
    comfy = tmp_path / "comfy"
    (comfy / "main.py").parent.mkdir(parents=True)
    (comfy / "main.py").write_text("")
    script = Path(__file__).parents[2] / "scripts" / "verification" / "h3_trace_prepare_cases.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--lineage",
            lineage,
            "--cases",
            str(cases),
            "--comfy-dir",
            str(comfy),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    output = json.loads(result.stdout)
    assert output["lineage"] == lineage
    assert output["caseIds"] == [row["id"] for row in rows]
    assert len(output["challengeReceipts"]) == 8
    assert all(not path.is_absolute() for path in map(Path, output["challengeReceipts"]))
