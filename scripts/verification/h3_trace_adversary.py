#!/usr/bin/env python3
"""Construct private, controlled trace-bypass fixtures from captured H3 runs.

The output is test evidence, never a miner response.  It deliberately makes two
adversarial inputs that a detector must reject from their contents alone:

* a genuine 10-step trace padded to a claimed 20-step public schedule; and
* an honest trace paired with media hashes from another captured output.

Ground-truth labels live only in ``operator-ground-truth.json``.  They are not
present in either generated manifest, so an evaluator cannot accept/reject by
reading a test label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SCHEMA = "forge/inference-trace-adversary/v1"
TRACE_SCHEMA = "forge/inference-trace-capture/v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def contained(root: Path, value: Path) -> Path:
    resolved = value.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("capture path escapes experiment root") from error
    return resolved


def read_manifest(root: Path, value: str) -> tuple[Path, dict[str, Any]]:
    path = contained(root, Path(value))
    document = json.loads(path.read_text())
    if document.get("schema") != TRACE_SCHEMA or not isinstance(document.get("trace"), list):
        raise ValueError("unsupported captured trace manifest")
    if not isinstance(document.get("lineageId"), str) or not isinstance(
        document.get("inputHash"), str
    ):
        raise ValueError("capture lineage/input missing")
    return path, document


def same_input(*documents: dict[str, Any]) -> None:
    pairs = {(document["lineageId"], document["inputHash"]) for document in documents}
    if len(pairs) != 1:
        raise ValueError("all captures must share exact lineage and input hash")


def descriptor_paths(value: Any):
    if isinstance(value, dict):
        if "tensorPath" in value:
            yield value
        for child in value.values():
            yield from descriptor_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptor_paths(child)


def copy_blobs(
    root: Path, manifest_path: Path, document: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Copy raw evidence under a new root and rewrite only relative paths."""
    copied = json.loads(json.dumps(document))
    source_root = manifest_path.parent
    prefix = Path("artifacts") / "raw"
    for descriptor in descriptor_paths(copied):
        relative = Path(str(descriptor["tensorPath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("trace descriptor path unsafe")
        source = contained(root, source_root / relative)
        if not source.is_file() or sha256(source) != descriptor.get("sha256"):
            raise ValueError("trace descriptor custody mismatch")
        target_relative = prefix / relative
        target = destination / target_relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.exists():
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
        if sha256(target) != descriptor["sha256"]:
            raise ValueError("copied descriptor custody mismatch")
        descriptor["tensorPath"] = str(target_relative)
    return copied


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite adversarial artifact")
    path.write_bytes(canonical(value) + b"\n")
    os.chmod(path, 0o600)


def claimed_steps(document: dict[str, Any]) -> int:
    value = document.get("declaredSteps")
    if not isinstance(value, int) or value < 1:
        raise ValueError("captured declared steps invalid")
    return value


def _tensor(path: Path):
    if path.suffix == ".npy":
        import numpy as np

        return "npy", np.load(path, allow_pickle=False)
    if path.suffix == ".pt":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to interpolate .pt trace tensors") from error
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError("trace tensor file is not a dense tensor")
        return "pt", value
    raise ValueError("trace tensor format must be .npy or .pt")


def _save_interpolated(path: Path, left, right, alpha: float, kind: str) -> dict[str, Any]:
    if kind == "npy":
        import numpy as np

        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not np.issubdtype(left.dtype, np.floating)
        ):
            raise ValueError("incompatible .npy trace tensors")
        value = (
            left.astype(np.float64) + (right.astype(np.float64) - left.astype(np.float64)) * alpha
        ).astype(left.dtype)
        np.save(path, value, allow_pickle=False)
    else:
        import torch

        if (
            tuple(left.shape) != tuple(right.shape)
            or left.dtype != right.dtype
            or not left.is_floating_point()
        ):
            raise ValueError("incompatible .pt trace tensors")
        value = (left.float() + (right.float() - left.float()) * alpha).to(left.dtype)
        with path.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    return {
        "tensorPath": str(path),
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "bytes": path.stat().st_size,
    }


def _bracket(
    records: list[dict[str, Any]], sigma: float
) -> tuple[dict[str, Any], dict[str, Any], float]:
    ordered = sorted(records, key=lambda item: float(item["sigma"]))
    if sigma <= float(ordered[0]["sigma"]):
        return ordered[0], ordered[0], 0.0
    if sigma >= float(ordered[-1]["sigma"]):
        return ordered[-1], ordered[-1], 0.0
    for left, right in zip(ordered, ordered[1:], strict=True):
        low, high = float(left["sigma"]), float(right["sigma"])
        if low <= sigma <= high:
            return left, right, (sigma - low) / (high - low)
    raise ValueError("could not bracket public sigma in shortcut schedule")


def _interpolate_components(
    destination: Path,
    step: int,
    field: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    by_component = {entry.get("component"): entry for entry in right}
    if len(by_component) != len(right) or {entry.get("component") for entry in left} != set(
        by_component
    ):
        raise ValueError("shortcut trace components differ")
    outputs = []
    for item in left:
        other = by_component[item["component"]]
        left_path, right_path = destination / item["tensorPath"], destination / other["tensorPath"]
        left_kind, left_tensor = _tensor(left_path)
        right_kind, right_tensor = _tensor(right_path)
        if left_kind != right_kind:
            raise ValueError("shortcut trace tensor formats differ")
        target = (
            destination / "derived" / f"{step:03d}-{field}-{item['component']}{left_path.suffix}"
        )
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = _save_interpolated(target, left_tensor, right_tensor, alpha, left_kind)
        descriptor["component"] = item["component"]
        descriptor["tensorPath"] = str(target.relative_to(destination))
        outputs.append(descriptor)
    return outputs


def padded_trace(
    honest: dict[str, Any], actual: dict[str, Any], destination: Path
) -> dict[str, Any]:
    reference, shortcut = honest["trace"], actual["trace"]
    if len(reference) < 2 or len(shortcut) < 2 or claimed_steps(honest) != len(reference):
        raise ValueError("trace counts do not support controlled padding")
    result = json.loads(json.dumps(actual))
    rewritten = []
    for target, public in enumerate(reference):
        left, right, alpha = _bracket(shortcut, float(public["sigma"]))
        # Values come only from genuine shortcut tensors. The honest capture
        # supplies public sigma positions/count, never values or raw files.
        rewritten.append(
            {
                "step": target,
                "totalSteps": len(reference),
                "sigma": public["sigma"],
                "stateCallback": _interpolate_components(
                    destination,
                    target,
                    "state",
                    left["stateCallback"],
                    right["stateCallback"],
                    alpha,
                ),
                "denoised": _interpolate_components(
                    destination, target, "denoised", left["denoised"], right["denoised"], alpha
                ),
            }
        )
    result["trace"] = rewritten
    result["declaredSteps"] = len(reference)
    result["callbackSteps"] = len(reference)
    return result


def repacked_id(*sources: Path) -> str:
    digest = hashlib.sha256("\0".join(str(path.resolve()) for path in sources).encode()).hexdigest()
    return f"run-{digest[:24]}"


def rebind_output_identity(output: Any, run_id: str) -> Any:
    """Keep a forged outer receipt internally coherent without losing hashes."""
    copied = json.loads(json.dumps(output))
    if isinstance(copied, dict) and "runId" in copied:
        copied["runId"] = run_id
    return copied


def make(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root).resolve()
    if not root.is_dir():
        raise ValueError("experiment root missing")
    output = contained(root, Path(args.output))
    if output.exists():
        raise FileExistsError("adversarial output directory must be fresh")
    honest_path, honest = read_manifest(root, args.honest_20)
    short_path, short = read_manifest(root, args.actual_10)
    other_path, other = read_manifest(root, args.alternate_video)
    same_input(honest, short, other)
    if claimed_steps(honest) <= claimed_steps(short):
        raise ValueError("honest trace must claim more steps than shortcut trace")
    if not isinstance(other.get("output"), dict):
        raise ValueError("alternate capture needs bound media output")
    output.mkdir(parents=True, mode=0o700)
    padded_dir, sidecar_dir = output / "padded-10-as-20", output / "sidecar-output"
    padded_dir.mkdir(mode=0o700)
    sidecar_dir.mkdir(mode=0o700)

    padded_short = copy_blobs(root, short_path, short, padded_dir)
    padded = padded_trace(honest, padded_short, padded_dir)
    # Claim the authorized profile/capture while retaining shortcut bytes.
    padded["capture"] = json.loads(json.dumps(honest["capture"]))
    padded["condition"] = honest.get("condition")
    padded["runId"] = repacked_id(honest_path, short_path)
    padded["output"] = rebind_output_identity(short.get("output"), padded["runId"])
    write_json(padded_dir / "manifest.json", padded)

    sidecar = copy_blobs(root, honest_path, honest, sidecar_dir)
    sidecar["runId"] = repacked_id(honest_path, other_path)
    sidecar["output"] = rebind_output_identity(other["output"], sidecar["runId"])
    write_json(sidecar_dir / "manifest.json", sidecar)

    ground_truth = {
        "schema": SCHEMA,
        "lineageId": honest["lineageId"],
        "inputHash": honest["inputHash"],
        "cases": {
            "padded-10-as-20": {
                "label": "padded_steps_10",
                "honestScheduleManifestSha256": sha256(honest_path),
                "genuineShortcutManifestSha256": sha256(short_path),
                "outputProvenance": {
                    "sourceManifestSha256": sha256(short_path),
                    "sourceOutput": short.get("output"),
                },
            },
            "sidecar-output": {
                "label": "sidecar_honest_wrong_output",
                "honestTraceManifestSha256": sha256(honest_path),
                "alternateOutputManifestSha256": sha256(other_path),
                "outputProvenance": {
                    "sourceManifestSha256": sha256(other_path),
                    "sourceOutput": other.get("output"),
                },
            },
        },
    }
    write_json(output / "operator-ground-truth.json", ground_truth)
    print(
        json.dumps(
            {"schema": SCHEMA, "output": str(output), "cases": sorted(ground_truth["cases"])},
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment-root", required=True)
    result.add_argument("--honest-20", required=True)
    result.add_argument("--actual-10", required=True)
    result.add_argument("--alternate-video", required=True)
    result.add_argument("--output", required=True)
    result.set_defaults(run=make)
    return result


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        parsed.run(parsed)
    except Exception as error:
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
