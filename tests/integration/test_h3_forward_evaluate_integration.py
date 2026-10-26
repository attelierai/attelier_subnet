from __future__ import annotations

import json
import subprocess
import sys

NAMES = ("honest-1", "honest-2", "steps10", "steps3", "weights-fp8", "lora20")


def test_real_plan_three_receipts_cli(tmp_path):
    lines = [
        {"id": "cal", "split": "calibration", "parameters": {}},
        {"id": "hold", "split": "holdout", "parameters": {}},
    ]
    (tmp_path / "experiment-plan.json").write_text(
        json.dumps(
            {
                "schema": "forge/inference-trace-plan/v1",
                "variants": [{"name": x, "condition": x} for x in NAMES],
                "lineages": lines,
                "extension": True,
            }
        )
    )
    for line in lines:
        for name in NAMES:
            run = f"{line['id']}-{name}"
            forwards = [{"call": x, "sha256": str(x) * 64} for x in (0, 10, 19)]
            p = tmp_path / "captures" / run / "manifest.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"forward": forwards}))
            for f in forwards:
                q = tmp_path / "replays" / run / f"canonical-{f['call']:03d}" / "result.json"
                q.parent.mkdir(parents=True)
                v = 0.2 if name in {"weights-fp8", "lora20"} else 0.002
                q.write_text(
                    json.dumps(
                        {
                            "schema": "forge/inference-forward-replay/v1",
                            "bundleSha256": f["sha256"],
                            "candidateLabel": "canonical",
                            "outcome": {
                                "status": "MEASURED",
                                "maxAbs": v,
                                "rms": v,
                                "relativeRms": v,
                                "expectedSha256": "a" * 64,
                                "actualSha256": "b" * 64,
                            },
                        }
                    )
                )
    artifact = tmp_path / "frozen.json"
    c = [
        sys.executable,
        "scripts/verification/h3_forward_evaluate.py",
        "calibrate",
        "--root",
        str(tmp_path),
        "--out",
        str(artifact),
    ]
    assert subprocess.run(c, capture_output=True, text=True).stdout.strip() == "CALIBRATED"
    out = tmp_path / "report.json"
    c = [
        sys.executable,
        "scripts/verification/h3_forward_evaluate.py",
        "report",
        "--root",
        str(tmp_path),
        "--calibration",
        str(artifact),
        "--out",
        str(out),
    ]
    assert subprocess.run(c, capture_output=True, text=True).stdout.strip() == "MEASURED"
    assert json.loads(out.read_text())["summary"]["stepVariantsMeasured"] == 2
