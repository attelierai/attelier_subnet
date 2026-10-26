#!/usr/bin/env python3
"""Reconcile one closed H3 qualification lineage without rendering or networking.

The utility only derives local CPU evidence from complete capture manifests. It
never creates a GPU job, fetches media, retries a partial adversary directory,
or makes a verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("path outside qualification root") from error
    return path


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error


def rows(value: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("cases", value.get("rows"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must contain case rows")
    return value


def category(row: dict[str, Any]) -> str:
    value = row.get("id")
    if not isinstance(value, str):
        raise ValueError("qualification case id invalid")
    for suffix, name in (
        ("-honest-1", "honest1"),
        ("-honest-2", "honest2"),
        ("-weights-fp8", "fp8"),
        ("-lora20", "lora"),
        ("-steps10", "steps10"),
        ("-steps3", "steps3"),
        ("-padded-10-as-20", "padded"),
        ("-recurrence-forged", "recurrence"),
        ("-sidecar-output", "sidecar"),
        ("-skip7", "skip7"),
    ):
        if value.endswith(suffix):
            return name
    raise ValueError(f"unrecognized qualification case id: {value!r}")


def run_python(args: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run([sys.executable, *args], check=True, env=environment)


def require_manifest(root: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    required = {
        "id",
        "label",
        "manifest",
        "trustedManifest",
        "numericDirectory",
        "challengeReceipt",
        "mediaDirectory",
    }
    if not required <= set(row):
        raise ValueError("actual index row schema invalid")
    path = contained(root, row["manifest"])
    manifest = read_json(path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputHash"), str):
        raise ValueError("case manifest invalid")
    return path, manifest


def complete_adversary(directory: Path) -> bool:
    required = (
        directory / "padded-10-as-20" / "manifest.json",
        directory / "sidecar-output" / "manifest.json",
        directory / "recurrence-forged" / "manifest.json",
        directory / "operator-ground-truth.json",
        directory / "recurrence-forged" / "operator-ground-truth.json",
    )
    return all(path.is_file() for path in required)


def prepare_adversaries(
    root: Path, lineage: str, by_kind: dict[str, dict[str, Any]], scripts: Path
) -> None:
    required = {"honest1", "steps10", "steps3"}
    if not required <= set(by_kind):
        raise ValueError("adversary construction needs honest1, steps10 and steps3")
    directory = root / "adversaries-v2" / lineage
    if directory.exists():
        if complete_adversary(directory):
            return
        raise RuntimeError("existing adversary directory is partial; reconcile manually")
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_python(
        [
            str(scripts / "h3_trace_adversary.py"),
            "--experiment-root",
            str(root),
            "--honest-20",
            str(contained(root, by_kind["honest1"]["manifest"])),
            "--actual-10",
            str(contained(root, by_kind["steps10"]["manifest"])),
            "--alternate-video",
            str(contained(root, by_kind["steps3"]["manifest"])),
            "--output",
            str(directory),
        ]
    )
    # If this second derivation fails, the now-partial directory is deliberately
    # left for manual reconciliation rather than blindly retried.
    run_python(
        [
            str(scripts / "h3_trace_recurrence_adversary.py"),
            "--experiment-root",
            str(root),
            "--padded-manifest",
            str(directory / "padded-10-as-20" / "manifest.json"),
            "--output",
            str(directory / "recurrence-forged"),
        ]
    )
    if not complete_adversary(directory):
        raise RuntimeError("adversary helpers did not produce a complete artifact")


def receipt_valid(receipt: Path, manifest: Path, trusted: Path, input_hash: str) -> bool:
    value = read_json(receipt)
    return isinstance(value, dict) and (
        value.get("manifestSha256") == digest(manifest)
        and value.get("conditioningManifestSha256") == digest(trusted)
        and value.get("inputHash") == input_hash
    )


def numerical_receipts(
    root: Path,
    row: dict[str, Any],
    manifest: Path,
    trusted: Path,
    scripts: Path,
    source_dir: Path | None,
) -> None:
    numeric = contained(root, row["numericDirectory"])
    if numeric.exists() and not numeric.is_dir():
        raise ValueError("numeric directory is not a directory")
    numeric.mkdir(parents=True, exist_ok=True, mode=0o700)
    transition, endpoint = numeric / "transition.json", numeric / "endpoint.json"
    environment = os.environ.copy()
    if source_dir is not None:
        environment["PYTHONPATH"] = str(source_dir) + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
    if not transition.exists():
        run_python(
            [
                str(scripts / "h3_transition_diagnostic.py"),
                "--manifest",
                str(manifest),
                "--out",
                str(transition),
            ],
            environment=environment,
        )
    if not endpoint.exists():
        run_python(
            [
                str(scripts / "h3_trace_endpoint_binding.py"),
                "--manifest",
                str(manifest),
                "--trusted-manifest",
                str(trusted),
                "--out",
                str(endpoint),
            ],
            environment=environment,
        )


def postcommit(
    root: Path,
    row: dict[str, Any],
    manifest: Path,
    trusted: Path,
    input_hash: str,
    comfy: Path,
    scripts: Path,
) -> Path | None:
    kind = category(row)
    if kind in {"steps3", "steps10"}:
        return None
    count = 20 if kind in {"honest1", "honest2", "skip7"} else 3
    receipt = contained(root, row["challengeReceipt"])
    destination = receipt.parent
    if receipt.exists():
        if not receipt.is_file() or not receipt_valid(receipt, manifest, trusted, input_hash):
            raise RuntimeError("existing challenge receipt is invalid or belongs to other evidence")
        return receipt
    if destination.exists():
        raise RuntimeError("existing challenge directory is partial; reconcile manually")
    run_python(
        [
            str(scripts / "h3_postcommit_derive.py"),
            "--comfy-dir",
            str(comfy),
            "--manifest",
            str(manifest),
            "--conditioning-manifest",
            str(trusted),
            "--count",
            str(count),
            "--out",
            str(destination),
        ]
    )
    if not receipt.is_file() or not receipt_valid(receipt, manifest, trusted, input_hash):
        raise RuntimeError("derived challenge receipt failed binding validation")
    return receipt


def main(args: argparse.Namespace) -> None:
    root, cases_path, comfy = (
        Path(args.root).resolve(),
        Path(args.cases).resolve(),
        Path(args.comfy_dir).resolve(),
    )
    scripts = Path(args.scripts_dir).resolve() if args.scripts_dir else SCRIPTS
    source_dir = Path(args.source_dir).resolve() if args.source_dir else None
    required_helpers = (
        "h3_trace_adversary.py",
        "h3_trace_recurrence_adversary.py",
        "h3_transition_diagnostic.py",
        "h3_trace_endpoint_binding.py",
        "h3_postcommit_derive.py",
    )
    if (
        not root.is_dir()
        or not cases_path.is_file()
        or not (comfy / "main.py").is_file()
        or not scripts.is_dir()
        or any(not (scripts / helper).is_file() for helper in required_helpers)
        or source_dir is not None
        and not source_dir.is_dir()
    ):
        raise ValueError("root, cases, or Comfy checkout missing")
    index = rows(read_json(cases_path), "qualification cases")
    selected = [row for row in index if row.get("lineageId") == args.lineage]
    if not selected or len({row.get("id") for row in selected}) != len(selected):
        raise ValueError("lineage cases missing or duplicate")
    by_kind: dict[str, dict[str, Any]] = {}
    for row in selected:
        kind = category(row)
        if kind in by_kind:
            raise ValueError("duplicate case category in lineage")
        by_kind[kind] = row
    prepare_adversaries(root, args.lineage, by_kind, scripts)

    receipts: list[str] = []
    for row in selected:
        manifest, value = require_manifest(root, row)
        trusted = contained(root, row["trustedManifest"])
        if not trusted.is_file():
            raise ValueError("trusted manifest missing")
        numerical_receipts(root, row, manifest, trusted, scripts, source_dir)
        receipt = postcommit(root, row, manifest, trusted, value["inputHash"], comfy, scripts)
        if receipt is not None:
            receipts.append(str(receipt.relative_to(root)))
    print(
        json.dumps(
            {
                "lineage": args.lineage,
                "caseIds": [row["id"] for row in selected],
                "challengeReceipts": receipts,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", required=True)
    result.add_argument("--lineage", required=True)
    result.add_argument("--cases", required=True)
    result.add_argument("--comfy-dir", required=True)
    result.add_argument(
        "--scripts-dir", help="installed helper directory; defaults to this script directory"
    )
    result.add_argument("--source-dir", help="optional source directory prepended to PYTHONPATH")
    result.set_defaults(run=main)
    return result


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        parsed.run(parsed)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None
