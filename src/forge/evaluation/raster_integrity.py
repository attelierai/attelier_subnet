"""Deterministic decoded-raster integrity check, separate from perceptual quality."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from forge.protocol.hashing import hash_canonical

SCHEMA = "forge/raster-integrity/v1"
RULE = {
    "patch_pixels": 32,
    "modes": ["nyquist_x", "nyquist_y", "nyquist_checker"],
    "channels": "max_rgb",
    "phase_handling": "absolute_correlation_all_parities_equivalent",
    "tile_score_threshold_ppm": 100_000,
    "frame_flagged_tiles_fraction_ppm": 250_000,
    "video_flagged_frames_fraction_ppm": 250_000,
}
RULE_ID = hash_canonical("forge/raster-integrity-rule/v1", RULE)


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("video sha256 must be lowercase SHA-256")
    return value


def _check_body(value: Mapping[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "check_hash"}


def check_frames(frames: np.ndarray, video_sha256: str) -> dict[str, object]:
    """Check every decoded RGB frame for large repeated Nyquist raster energy."""
    _sha256(video_sha256)
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or not all(dimension > 0 for dimension in frames.shape[:3])
    ):
        raise ValueError("frames must be a non-empty uint8 [frame,height,width,rgb] array")
    frame_count, height, width, _ = frames.shape
    patch = RULE["patch_pixels"]
    rows, columns = height // patch, width // patch
    if rows == 0 or columns == 0:
        raise ValueError("decoded geometry is smaller than one raster patch")
    cropped = frames[:, : rows * patch, : columns * patch].astype(np.float32)
    tiles = cropped.reshape(frame_count, rows, patch, columns, patch, 3).transpose(0, 1, 3, 2, 4, 5)
    tiles -= tiles.mean(axis=(3, 4), keepdims=True)
    coordinates = np.arange(patch, dtype=np.float32)
    alternating = 1 - 2 * (coordinates.astype(np.int8) & 1)
    masks = np.stack(
        (
            np.broadcast_to(alternating[None, :], (patch, patch)),
            np.broadcast_to(alternating[:, None], (patch, patch)),
            alternating[:, None] * alternating[None, :],
        )
    )
    scores = np.abs(np.einsum("frcijq,mij->fmrcq", tiles, masks, optimize=True))
    return _receipt(scores, video_sha256, frame_count, height, width, rows * columns, patch)


def _receipt(
    scores: np.ndarray,
    video_sha256: str,
    frame_count: int,
    height: int,
    width: int,
    tiles_per_frame: int,
    patch: int,
) -> dict[str, object]:
    # ``einsum`` retains the tile dimensions; normalize by pixels and RGB range.
    scores = scores / (patch * patch * 255)
    tile_scores = scores.max(axis=(1, 4))
    threshold = RULE["tile_score_threshold_ppm"]
    frame_metrics = []
    flagged_frames = []
    for index, values in enumerate(tile_scores):
        flagged_tiles = int(np.count_nonzero(values * 1_000_000 >= threshold))
        fraction_ppm = flagged_tiles * 1_000_000 // tiles_per_frame
        metric = {
            "frame_index": index,
            "max_tile_score_ppm": int(round(float(values.max()) * 1_000_000)),
            "flagged_tiles": flagged_tiles,
            "tiles": tiles_per_frame,
            "flagged_tiles_fraction_ppm": fraction_ppm,
        }
        frame_metrics.append(metric)
        if fraction_ppm >= RULE["frame_flagged_tiles_fraction_ppm"]:
            flagged_frames.append(index)
    flagged_fraction_ppm = len(flagged_frames) * 1_000_000 // frame_count
    body: dict[str, object] = {
        "schema": SCHEMA,
        "rule": RULE,
        "rule_id": RULE_ID,
        "video_sha256": video_sha256,
        "geometry": {"frames": frame_count, "height": height, "width": width, "channels": 3},
        "status": "ANOMALY"
        if flagged_fraction_ppm >= RULE["video_flagged_frames_fraction_ppm"]
        else "CLEAR",
        "flagged_frames": flagged_frames,
        "flagged_frames_fraction_ppm": flagged_fraction_ppm,
        "frame_metrics": frame_metrics,
    }
    return {**body, "check_hash": hash_canonical("forge/raster-integrity-check/v1", body)}


def verify_check(value: object, video_sha256: str, geometry: Mapping[str, object]) -> bool:
    """Verify an immutable receipt binding without treating it as perceptual quality."""
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        return False
    if value.get("rule") != RULE or value.get("rule_id") != RULE_ID:
        return False
    if value.get("video_sha256") != video_sha256 or value.get("status") not in {"CLEAR", "ANOMALY"}:
        return False
    expected_geometry = {
        "frames": geometry.get("frames"),
        "height": geometry.get("height"),
        "width": geometry.get("width"),
        "channels": 3,
    }
    if value.get("geometry") != expected_geometry:
        return False
    frame_metrics = value.get("frame_metrics")
    frame_count = expected_geometry["frames"]
    patch = RULE["patch_pixels"]
    tiles = (expected_geometry["height"] // patch) * (expected_geometry["width"] // patch)
    if not isinstance(frame_count, int) or not isinstance(tiles, int) or tiles < 1:
        return False
    if not isinstance(frame_metrics, list) or len(frame_metrics) != frame_count:
        return False
    derived_flagged: list[int] = []
    for index, metric in enumerate(frame_metrics):
        if not isinstance(metric, Mapping) or set(metric) != {
            "frame_index",
            "max_tile_score_ppm",
            "flagged_tiles",
            "tiles",
            "flagged_tiles_fraction_ppm",
        }:
            return False
        if any(type(metric[key]) is not int for key in metric):
            return False
        flagged_tiles = metric["flagged_tiles"]
        if (
            metric["frame_index"] != index
            or metric["max_tile_score_ppm"] < 0
            or not 0 <= flagged_tiles <= tiles
            or metric["tiles"] != tiles
            or metric["flagged_tiles_fraction_ppm"] != flagged_tiles * 1_000_000 // tiles
            or (
                flagged_tiles == 0
                and metric["max_tile_score_ppm"] >= RULE["tile_score_threshold_ppm"]
            )
            or (
                flagged_tiles > 0
                and metric["max_tile_score_ppm"] < RULE["tile_score_threshold_ppm"]
            )
        ):
            return False
        if metric["flagged_tiles_fraction_ppm"] >= RULE["frame_flagged_tiles_fraction_ppm"]:
            derived_flagged.append(index)
    if value.get("flagged_frames") != derived_flagged:
        return False
    fraction = len(derived_flagged) * 1_000_000 // frame_count
    if value.get("flagged_frames_fraction_ppm") != fraction:
        return False
    expected_status = (
        "ANOMALY" if fraction >= RULE["video_flagged_frames_fraction_ppm"] else "CLEAR"
    )
    if value.get("status") != expected_status:
        return False
    return value.get("check_hash") == hash_canonical(
        "forge/raster-integrity-check/v1", _check_body(value)
    )
