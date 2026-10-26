"""Versioned CPU guard for severe repeated photometric flicker.

This guard works only on canonical decoded RGB frames.  It is intentionally
separate from VideoAlign and its scores: a rejected frame sequence is an
observable media defect, not a claim about perceptual-model calibration.

The rule detects repeated oscillation in either global luminance or the
spatial/chromatic residual.  It does not reject large motion alone.  Its fixed
cutoffs describe the deliberately severe controls used for qualification; a
fresh held-out natural corpus is still required before any production use.
Camera cuts, intentional strobing, and unusual lighting are outside this
guard's claim and must be handled by separately declared prompt contracts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np

from forge.evaluation.vision_media import load_video, media_identity
from forge.protocol.hashing import hash_canonical, sha256_file

SCHEMA = "forge/flicker-guard/v1"
RULE = {
    "schema": SCHEMA,
    "version": 1,
    "minimum_frames": 3,
    "global_luma_acceleration_p95_min": 32.0,
    "brightness_direction_flip_fraction_min": 0.75,
    "spatial_opposition_fraction_min": 0.75,
    "chroma_opposition_fraction_min": 0.75,
    "opposition_correlation_max": -0.70,
    "residual_rms_floor_8bit": 4.0,
    "steady_lighting_required": True,
}
RULE_ID = hash_canonical("forge/flicker-guard-rule/v1", RULE)


def algorithm_id() -> str:
    """Bind the fixed rule, implementation source and actual decoder binary."""
    return hash_canonical(
        "forge/flicker-guard-algorithm/v1",
        RULE,
        sha256_file(Path(__file__)),
        sha256_file(Path(__file__).with_name("vision_media.py")),
        sha256_file(Path(imageio_ffmpeg.get_ffmpeg_exe())),
        np.__version__,
    )


def _frames(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB")
    if value.ndim != 4 or value.shape[0] < RULE["minimum_frames"] or value.shape[-1] != 3:
        raise ValueError("frames must have at least three RGB frames")
    return np.ascontiguousarray(value)


def _opposition_fraction(values: np.ndarray) -> float:
    """Count opposite correlations only when both residuals exceed the physical floor."""
    left, right = values[:-1].reshape(len(values) - 1, -1), values[1:].reshape(len(values) - 1, -1)
    left_rms = np.sqrt(np.mean(np.square(left), axis=1))
    right_rms = np.sqrt(np.mean(np.square(right), axis=1))
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    usable = (
        (left_rms >= RULE["residual_rms_floor_8bit"])
        & (right_rms >= RULE["residual_rms_floor_8bit"])
        & (denominator > 1e-12)
    )
    correlations = np.ones(len(left), dtype=np.float64)
    correlations[usable] = (left[usable] * right[usable]).sum(axis=1) / denominator[usable]
    return float(np.mean(usable & (correlations <= RULE["opposition_correlation_max"])))


def diagnostics(frames: np.ndarray) -> dict[str, float | int]:
    """Return deterministic signals; no threshold is fitted from this input."""
    checked = _frames(frames).astype(np.float64)
    luma = checked @ np.array([0.2126, 0.7152, 0.0722])
    brightness = luma.mean(axis=(1, 2))
    acceleration = np.abs(np.diff(brightness, n=2))
    deltas = np.diff(brightness)
    meaningful = np.abs(deltas) >= 4.0
    directions = np.sign(deltas[meaningful])
    flips = float(np.mean(directions[1:] * directions[:-1] < 0)) if len(directions) > 1 else 0.0
    spatial = luma - luma.mean(axis=(1, 2), keepdims=True)
    chroma = checked - checked.mean(axis=3, keepdims=True)
    return {
        "frame_count": int(len(checked)),
        "global_luma_acceleration_p95": float(np.quantile(acceleration, 0.95)),
        "brightness_direction_flip_fraction": flips,
        "spatial_opposition_fraction": _opposition_fraction(spatial),
        "chroma_opposition_fraction": _opposition_fraction(chroma),
    }


def envelope(frames: np.ndarray) -> dict[str, Any]:
    """The portable detector payload for future composition."""
    return {"algorithm_id": algorithm_id(), "diagnostics": diagnostics(frames)}


def validate_envelope(
    value: dict[str, Any], *, expected_algorithm_id: str | None = None
) -> dict[str, float | int]:
    """Fail closed on malformed diagnostics or an unexpected producer identity.

    By default the local implementation is required.  A composite receipt
    verifier can instead supply the frozen producer identifier, allowing a
    receipt made in its pinned numerical runtime to be replayed elsewhere.
    """
    required = {
        "frame_count",
        "global_luma_acceleration_p95",
        "brightness_direction_flip_fraction",
        "spatial_opposition_fraction",
        "chroma_opposition_fraction",
    }
    if not isinstance(value, dict) or set(value) != {"algorithm_id", "diagnostics"}:
        raise ValueError("flicker guard envelope fields are invalid")
    expected = algorithm_id() if expected_algorithm_id is None else expected_algorithm_id
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("expected flicker guard algorithm identity is invalid")
    if value["algorithm_id"] != expected:
        raise ValueError("flicker guard algorithm identity differs")
    measured = value["diagnostics"]
    if not isinstance(measured, dict) or set(measured) != required:
        raise ValueError("flicker guard diagnostics fields are invalid")
    if type(measured["frame_count"]) is not int or measured["frame_count"] < RULE["minimum_frames"]:
        raise ValueError("flicker guard frame count is invalid")
    fractions = {
        "brightness_direction_flip_fraction",
        "spatial_opposition_fraction",
        "chroma_opposition_fraction",
    }
    for key in required - {"frame_count"}:
        item = measured[key]
        if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
            raise ValueError("flicker guard diagnostics must be finite and nonnegative")
        if key in fractions and item > 1:
            raise ValueError("flicker guard fractions must not exceed one")
    return {
        key: int(item) if key == "frame_count" else float(item) for key, item in measured.items()
    }


def decide(value: dict[str, float | int], *, steady_lighting: bool = True) -> bool:
    """Return true only for repeated severe oscillation under a steady-light tag."""
    if not steady_lighting:
        return False
    numeric = validate_envelope({"algorithm_id": algorithm_id(), "diagnostics": value})
    global_oscillation = (
        numeric["global_luma_acceleration_p95"] >= RULE["global_luma_acceleration_p95_min"]
        and numeric["brightness_direction_flip_fraction"]
        >= RULE["brightness_direction_flip_fraction_min"]
    )
    residual_inversion = (
        numeric["spatial_opposition_fraction"] >= RULE["spatial_opposition_fraction_min"]
        or numeric["chroma_opposition_fraction"] >= RULE["chroma_opposition_fraction_min"]
    )
    return global_oscillation or residual_inversion


def _write_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".flicker-", suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_video(
    video: Path, geometry: dict[str, Any], *, steady_lighting: bool
) -> dict[str, Any]:
    loaded = load_video(video, geometry)
    measured = envelope(loaded.frames)
    body = {
        "schema": SCHEMA,
        "rule": RULE,
        "rule_id": RULE_ID,
        "algorithm_id": measured["algorithm_id"],
        "video_sha256": sha256_file(video),
        "decoded_sha256": media_identity(loaded.frames),
        "geometry": dict(geometry),
        "steady_lighting": steady_lighting,
        "diagnostics": measured["diagnostics"],
        "severe_flicker": decide(measured["diagnostics"], steady_lighting=steady_lighting),
    }
    return {**body, "receipt_id": hash_canonical("forge/flicker-guard-receipt/v1", body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steady-lighting", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    geometry = json.loads(args.geometry.read_text())
    _write_new(args.out, inspect_video(args.video, geometry, steady_lighting=args.steady_lighting))


if __name__ == "__main__":
    main()
