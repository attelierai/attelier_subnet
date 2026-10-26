#!/usr/bin/env python3
"""Forge a recurrence-consistent trace from a controlled padded H3 fixture.

This is private red-team evidence, not a miner response and not a proof that a
model ran.  It demonstrates why an algebraic Res Multistep recurrence check
must be paired with a fixed-input model forward replay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from h3_trace_adversary import (
    TRACE_SCHEMA,
    claimed_steps,
    contained,
    copy_blobs,
    read_manifest,
    rebind_output_identity,
    repacked_id,
    sha256,
    write_json,
)


def _descriptor_path(root: Path, descriptor: dict[str, Any]) -> Path:
    required = {"component", "tensorPath", "sha256", "shape", "dtype", "bytes"}
    if not required <= set(descriptor):
        raise ValueError("trace tensor descriptor invalid")
    relative = Path(str(descriptor["tensorPath"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("trace tensor path unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("trace tensor path escapes fixture") from error
    if (
        not path.is_file()
        or sha256(path) != descriptor["sha256"]
        or path.stat().st_size != descriptor["bytes"]
    ):
        raise ValueError("trace tensor custody mismatch")
    return path


def _load_float32(root: Path, descriptor: dict[str, Any]) -> tuple[str, Any]:
    path = _descriptor_path(root, descriptor)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
        if (
            not isinstance(value, np.ndarray)
            or list(value.shape) != descriptor["shape"]
            or str(value.dtype) != descriptor["dtype"]
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid numpy trace tensor")
        return "npy", value.astype(np.float32, copy=False)
    if path.suffix == ".pt":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for .pt trace tensors") from error
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(value, torch.Tensor)
            or list(value.shape) != descriptor["shape"]
            or str(value.dtype).removeprefix("torch.") != descriptor["dtype"]
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("invalid torch trace tensor")
        return "pt", value.to(dtype=torch.float32, device="cpu")
    raise ValueError("trace tensor format unsupported")


def _write_float32(path: Path, kind: str, value: Any, component: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError("recurrence tensor output exists")
    if kind == "npy":
        value = np.asarray(value, dtype=np.float32)
        np.save(path, value, allow_pickle=False)
        dtype = str(value.dtype)
    elif kind == "pt":
        import torch

        value = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
        with path.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        dtype = str(value.dtype).removeprefix("torch.")
    else:
        raise ValueError("trace tensor format unsupported")
    os.chmod(path, 0o600)
    return {
        "component": component,
        "tensorPath": str(path),
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": dtype,
        "bytes": path.stat().st_size,
    }


def _components(root: Path, values: Any) -> dict[str, tuple[dict[str, Any], str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError("trace components missing")
    result = {}
    for descriptor in values:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("component"), str):
            raise ValueError("trace component invalid")
        name = descriptor["component"]
        if name in result:
            raise ValueError("trace component duplicate")
        kind, value = _load_float32(root, descriptor)
        result[name] = (descriptor, kind, value)
    return result


def _previous_coefficients(previous_sigma: np.float32, sigma: np.float32, next_sigma: np.float32):
    t = -math.log(float(sigma))
    h = -math.log(float(next_sigma)) - t
    c2 = (-math.log(float(previous_sigma)) - t) / h
    phi1 = math.expm1(-h) / (-h)
    phi2 = (phi1 - 1) / (-h)
    return h, phi1 - phi2 / c2, phi2 / c2


def _solve_euler(xi: Any, next_state: Any, sigma: np.float32, next_sigma: np.float32) -> Any:
    return xi - (next_state - xi) * float(sigma) / (float(next_sigma) - float(sigma))


def _solve_multistep(
    xi: Any,
    next_state: Any,
    previous_denoised: Any,
    previous_sigma: np.float32,
    sigma: np.float32,
    next_sigma: np.float32,
) -> Any:
    h, b1, b2 = _previous_coefficients(previous_sigma, sigma, next_sigma)
    if h == 0 or b1 == 0:
        raise ValueError("degenerate Res Multistep coefficients")
    return (next_state - math.exp(-h) * xi - h * b2 * previous_denoised) / (h * b1)


def _stored_descriptor(destination: Path, step: int, component: str, kind: str, value: Any):
    suffix = ".npy" if kind == "npy" else ".pt"
    target = destination / "derived" / f"{step:03d}-recurrence-denoised-{component}{suffix}"
    descriptor = _write_float32(target, kind, value, component)
    descriptor["tensorPath"] = str(target.relative_to(destination))
    return descriptor


def forge_recurrence(padded: dict[str, Any], destination: Path) -> dict[str, Any]:
    result = json.loads(json.dumps(padded))
    trace = result.get("trace")
    if not isinstance(trace, list) or len(trace) < 2 or claimed_steps(result) != len(trace):
        raise ValueError("padded trace is incomplete")
    if result.get("callbackSteps") != len(trace):
        raise ValueError("padded callback count invalid")
    sigmas = [np.float32(row.get("sigma")) for row in trace]
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in sigmas) or any(
        left <= right for left, right in zip(sigmas, sigmas[1:], strict=False)
    ):
        raise ValueError("padded sigmas must strictly decrease and remain nonzero")

    states = [_components(destination, row.get("stateCallback")) for row in trace]
    components = set(states[0])
    if components != {"video", "audio"} or any(set(row) != components for row in states):
        raise ValueError("expected stable video/audio callback states")
    final = _components(destination, result.get("finalLatent"))
    if set(final) != components:
        raise ValueError("final latent components differ")

    generated: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for index, _row in enumerate(trace):
        descriptors = []
        for component in ("video", "audio"):
            _, kind, xi = states[index][component]
            if index == len(trace) - 1:
                _, final_kind, final_value = final[component]
                if final_kind != kind:
                    raise ValueError("final latent tensor format differs from callback state")
                value = final_value if component == "video" else final_value * 4.0
            else:
                _, _, next_state = states[index + 1][component]
                if index == 0:
                    value = _solve_euler(xi, next_state, sigmas[index], sigmas[index + 1])
                else:
                    assert previous is not None
                    value = _solve_multistep(
                        xi,
                        next_state,
                        previous[component],
                        sigmas[index - 1],
                        sigmas[index],
                        sigmas[index + 1],
                    )
            descriptors.append(_stored_descriptor(destination, index, component, kind, value))
        generated.append(descriptors)
        previous = {item["component"]: _load_float32(destination, item)[1] for item in descriptors}

    for row, descriptors in zip(trace, generated, strict=True):
        row["denoised"] = descriptors
    result["condition"] = "honest_repeat"
    return result


def run(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root).resolve()
    if not root.is_dir():
        raise ValueError("experiment root missing")
    padded_path, padded = read_manifest(root, args.padded_manifest)
    if padded.get("schema") != TRACE_SCHEMA:
        raise ValueError("padded capture schema invalid")
    output = contained(root, Path(args.output))
    if output.exists():
        raise FileExistsError("recurrence adversary output must be fresh")
    output.mkdir(parents=True, mode=0o700)
    copied = copy_blobs(root, padded_path, padded, output)
    forged = forge_recurrence(copied, output)
    forged["runId"] = repacked_id(padded_path, output)
    forged["output"] = rebind_output_identity(padded.get("output"), forged["runId"])
    write_json(output / "manifest.json", forged)
    write_json(
        output / "operator-ground-truth.json",
        {
            "schema": "forge/inference-trace-recurrence-adversary/v1",
            "lineageId": forged["lineageId"],
            "inputHash": forged["inputHash"],
            "cases": {
                "recurrence-forged": {
                    "label": "recurrence_consistent_model_unbound",
                    "paddedManifestSha256": sha256(padded_path),
                    "outputProvenance": {
                        "sourceManifestSha256": sha256(padded_path),
                        "sourceOutput": padded.get("output"),
                    },
                }
            },
        },
    )
    print(json.dumps({"manifest": str(output / "manifest.json")}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment-root", required=True)
    result.add_argument("--padded-manifest", required=True)
    result.add_argument("--output", required=True)
    result.set_defaults(run=run)
    return result


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        parsed.run(parsed)
    except Exception as error:
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
