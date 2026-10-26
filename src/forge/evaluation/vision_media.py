"""Trusted decoding, review sheets and deterministic negative media controls.

The diagnostics here describe decoded pixels; they are deliberately not a
calibrated assessment of video quality.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from forge.protocol.hashing import sha256_bytes, sha256_file

_SHEET_COLUMNS = 3
_SHEET_ROWS = 9
_SHEET_CAPACITY = _SHEET_COLUMNS * _SHEET_ROWS
_THUMB_WIDTH = 416
_LABEL_HEIGHT = 24


@dataclass(frozen=True)
class LoadedVideo:
    frames: np.ndarray
    video_sha256: str
    diagnostics: dict[str, float | int]


def _checked_frames(frames: np.ndarray) -> np.ndarray:
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("frames must be a uint8 numpy array")
    if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (count, height, width, 3)")
    if frames.shape[1] < 1 or frames.shape[2] < 1:
        raise ValueError("frames must have positive dimensions")
    return np.ascontiguousarray(frames)


def _geometry(geometry: Mapping[str, Any]) -> tuple[int, int, int, float]:
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be a mapping")
    expected = {"width", "height", "frames", "fps"}
    if set(geometry) != expected:
        raise ValueError("geometry must contain exactly width, height, frames and fps")
    width, height, count, fps = (
        geometry["width"],
        geometry["height"],
        geometry["frames"],
        geometry["fps"],
    )
    invalid_integer = any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (width, height, count)
    )
    if invalid_integer:
        raise ValueError("geometry width, height and frames must be positive integers")
    invalid_fps = (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    )
    if invalid_fps:
        raise ValueError("geometry fps must be a positive finite number")
    return width, height, count, float(fps)


def load_video(path: str | Path, geometry: Mapping[str, Any]) -> LoadedVideo:
    """Decode every RGB frame and reject any mismatch with prescribed geometry."""

    video = Path(path)
    if not video.is_file():
        raise ValueError("video path must be a regular file")
    width, height, expected_count, expected_fps = _geometry(geometry)
    try:
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            source_size = tuple(metadata.get("source_size") or ())
            source_fps = metadata.get("fps")
            if source_size != (width, height):
                raise ValueError(f"video resolution is {source_size}, expected {(width, height)}")
            invalid_source_fps = not isinstance(source_fps, (int, float)) or (
                abs(float(source_fps) - expected_fps) > 1e-6
            )
            if invalid_source_fps:
                raise ValueError(f"video fps is {source_fps!r}, expected {expected_fps}")
            raw_frames = []
            for raw in reader:
                if len(raw_frames) == expected_count:
                    raise ValueError(f"video has more than {expected_count} frames")
                raw_frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
        finally:
            reader.close()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"unable to decode video: {type(error).__name__}") from error
    if len(raw_frames) != expected_count:
        raise ValueError(f"video has {len(raw_frames)} frames, expected {expected_count}")
    frames = _checked_frames(np.stack(raw_frames, axis=0))
    return LoadedVideo(
        frames=frames,
        video_sha256=sha256_file(video),
        diagnostics=video_diagnostics(frames),
    )


def media_identity(frames: np.ndarray) -> str:
    """SHA-256 of canonical decoded RGB bytes, independent of container encoding."""

    return sha256_bytes(_checked_frames(frames).tobytes(order="C"))


def video_diagnostics(frames: np.ndarray) -> dict[str, float | int]:
    """Return deterministic pixel diagnostics, never a quality score."""

    checked = _checked_frames(frames)
    values = checked.astype(np.float32) / 255.0
    if len(values) == 1:
        temporal_delta, frozen_fraction = 0.0, 1.0
    else:
        transitions = np.abs(np.diff(values, axis=0))
        temporal_delta = float(transitions.mean())
        frozen_fraction = float(np.mean(np.all(np.diff(checked, axis=0) == 0, axis=(1, 2, 3))))
    gray = values.mean(axis=3)
    if gray.shape[1] < 3 or gray.shape[2] < 3:
        high_frequency = 0.0
    else:
        center = gray[:, 1:-1, 1:-1]
        laplacian = (
            4 * center
            - gray[:, :-2, 1:-1]
            - gray[:, 2:, 1:-1]
            - gray[:, 1:-1, :-2]
            - gray[:, 1:-1, 2:]
        )
        high_frequency = float(np.abs(laplacian).mean())
    return {
        "frame_count": int(checked.shape[0]),
        "temporal_delta": temporal_delta,
        "high_frequency_proxy": high_frequency,
        "frozen_fraction": frozen_fraction,
        "dark_frame_fraction": float(np.mean(values.mean(axis=(1, 2, 3)) <= 0.01)),
        "bright_frame_fraction": float(np.mean(values.mean(axis=(1, 2, 3)) >= 0.99)),
        "magenta_frame_fraction": float(
            np.mean(
                (values[..., 0].mean(axis=(1, 2)) >= 0.9)
                & (values[..., 1].mean(axis=(1, 2)) <= 0.1)
                & (values[..., 2].mean(axis=(1, 2)) >= 0.9)
            )
        ),
        **(temporal_diagnostics(checked) if len(checked) >= 3 else {}),
    }


def temporal_diagnostics(frames: np.ndarray) -> dict[str, float]:
    """All-frame signals, not quality decisions or calibrated defect thresholds.

    Brightness acceleration separates rapid photometric changes from a gradual
    exposure ramp. Spatial changes are measured after removing each frame's mean
    brightness. Cuts, intentional flashes and camera motion still need context.
    Values use an 8-bit luminance scale and preserve the original frame order.
    """
    checked = _checked_frames(frames)
    if len(checked) < 3:
        raise ValueError("temporal diagnostics require at least three frames")
    thumbnails = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((32, 32), Image.Resampling.BOX))
            for frame in checked
        ]
    ).astype(np.float64)
    gray = thumbnails @ np.array([0.2126, 0.7152, 0.0722])
    brightness = gray.mean(axis=(1, 2))
    brightness_acceleration = np.abs(np.diff(brightness, n=2))
    centered = gray - brightness[:, None, None]
    content_changes = np.abs(np.diff(centered, axis=0)).mean(axis=(1, 2))
    return {
        "luma_acceleration_median": float(np.median(brightness_acceleration)),
        "luma_acceleration_p95": float(np.quantile(brightness_acceleration, 0.95)),
        "centered_transition_median": float(np.median(content_changes)),
        "centered_transition_p95": float(np.quantile(content_changes, 0.95)),
    }


def degrade_frames(frames: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Construct deterministic negative controls from decoded frames."""

    checked = _checked_frames(frames)
    if kind == "freeze":
        return np.repeat(checked[:1], len(checked), axis=0)
    if kind == "black":
        return np.zeros_like(checked)
    if kind == "shuffle":
        order = np.random.default_rng(seed).permutation(len(checked))
        if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
            order = order[::-1]
        return checked[order].copy()
    if kind == "blur":
        return np.stack(
            [
                np.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=8)))
                for frame in checked
            ],
            axis=0,
        ).astype(np.uint8, copy=False)
    raise ValueError("unknown degradation kind; expected freeze, blur, shuffle or black")


def _sheet(frame_entries: list[tuple[int, np.ndarray]], fps: float) -> bytes:
    if not frame_entries:
        raise ValueError("a sheet needs at least one frame")
    first = frame_entries[0][1]
    thumb_height = max(1, round(first.shape[0] * _THUMB_WIDTH / first.shape[1]))
    cell_height = thumb_height + _LABEL_HEIGHT
    canvas = Image.new(
        "RGB",
        (_SHEET_COLUMNS * _THUMB_WIDTH, _SHEET_ROWS * cell_height),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    for position, (index, frame) in enumerate(frame_entries):
        row, column = divmod(position, _SHEET_COLUMNS)
        thumb = Image.fromarray(frame).resize(
            (_THUMB_WIDTH, thumb_height), Image.Resampling.LANCZOS
        )
        x, y = column * _THUMB_WIDTH, row * cell_height
        canvas.paste(thumb, (x, y))
        draw.text((x + 5, y + thumb_height + 4), f"frame {index}  {index / fps:.3f}s", fill="white")
    from io import BytesIO

    encoded = BytesIO()
    canvas.save(encoded, format="PNG", optimize=False)
    return encoded.getvalue()


def _native_strip(frame_entries: list[tuple[int, np.ndarray]], fps: float) -> bytes:
    """A spatial-detail sheet without resizing any selected decoded pixels."""

    first = frame_entries[0][1]
    height, width = first.shape[:2]
    rows = math.ceil(len(frame_entries) / 3)
    canvas = Image.new("RGB", (width * 3, rows * (height + _LABEL_HEIGHT)), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for position, (index, frame) in enumerate(frame_entries):
        row, column = divmod(position, 3)
        x, y = column * width, row * (height + _LABEL_HEIGHT)
        canvas.paste(Image.fromarray(frame), (x, y))
        draw.text(
            (x + 5, y + height + 4),
            f"frame {index}  {index / fps:.3f}s",
            fill="white",
        )
    from io import BytesIO

    encoded = BytesIO()
    canvas.save(encoded, format="PNG", optimize=False)
    return encoded.getvalue()


def select_detail_frames(frame_count: int) -> tuple[int, ...]:
    """Choose one native-detail frame per temporal stratum, after rendering.

    This entropy must never be a function of a public video seed or prompt. A
    fixed first/middle/last strip admitted a measured checkerboard blind spot.
    The same recorded selection is applied to both sides of a comparison.
    """
    if type(frame_count) is not int or not 1 <= frame_count <= 81:
        raise ValueError("detail selection requires between one and 81 frames")
    count = min(frame_count, 9)
    return tuple(
        secrets.choice(range(index * frame_count // count, (index + 1) * frame_count // count))
        for index in range(count)
    )


def select_detail_crops(
    frames: np.ndarray, detail_indices: tuple[int, ...]
) -> tuple[dict[str, int], ...]:
    """Choose three post-render, shared 4x spatial crops from native-detail frames.

    Selection is deliberately secret until persisted in the evaluation receipt.  The
    returned coordinates are sufficient to replay precisely the same evidence for
    both sides of a pair; they are never derived from a public prompt or video seed.
    """

    checked = _checked_frames(frames)
    if len(detail_indices) < 3 or len(set(detail_indices)) != len(detail_indices):
        raise ValueError("crop selection requires at least three distinct detail frames")
    height, width = checked.shape[1:3]
    crop_width, crop_height = width // 4, height // 4
    if crop_width < 1 or crop_height < 1:
        raise ValueError("decoded geometry is too small for 4x detail crops")
    chosen = list(detail_indices)
    selected: list[int] = []
    while len(selected) < 3:
        index = chosen.pop(secrets.randbelow(len(chosen)))
        selected.append(index)
    return tuple(
        {
            "frame_index": index,
            "x": secrets.randbelow(width - crop_width + 1),
            "y": secrets.randbelow(height - crop_height + 1),
            "width": crop_width,
            "height": crop_height,
        }
        for index in selected
    )


def _checked_detail_crops(
    crops: tuple[dict[str, int], ...], detail_indices: tuple[int, ...], width: int, height: int
) -> None:
    if len(crops) != 3 or len({crop.get("frame_index") for crop in crops}) != 3:
        raise ValueError("detail crops must select three distinct native detail frames")
    expected_width, expected_height = width // 4, height // 4
    for crop in crops:
        if set(crop) != {"frame_index", "x", "y", "width", "height"} or any(
            type(value) is not int for value in crop.values()
        ):
            raise ValueError("detail crops must have integer replay coordinates")
        if crop["frame_index"] not in detail_indices:
            raise ValueError("detail crop frame must be one of the native detail frames")
        if (crop["width"], crop["height"]) != (expected_width, expected_height):
            raise ValueError("detail crops must be quarter-resolution source regions")
        if (
            not 0 <= crop["x"] <= width - crop["width"]
            or not 0 <= crop["y"] <= height - crop["height"]
        ):
            raise ValueError("detail crop coordinates are outside the decoded frame")


def select_transition_frames(frame_count: int) -> tuple[int, ...]:
    """Secret post-render adjacent-frame starts spanning temporal strata.

    The same persisted selection is supplied to both blinded sides. It is not
    derived from a public seed, so miners cannot target fixed transition slots.
    """

    if type(frame_count) is not int or not 2 <= frame_count <= 81:
        raise ValueError("transition selection requires between two and 81 frames")
    transitions = frame_count - 1
    count = min(transitions, 9)
    return tuple(
        secrets.choice(range(index * transitions // count, (index + 1) * transitions // count))
        for index in range(count)
    )


def _transition_strip(frames: np.ndarray, indices: tuple[int, ...], fps: float) -> bytes:
    """Show adjacent frames and their absolute pixel delta as temporal evidence.

    This visualizes discontinuities for the judge. It is deliberately not a
    scalar quality metric: large deltas can be either valid motion or a defect.
    """

    height, width = frames.shape[1:3]
    thumb_width = 256
    thumb_height = max(1, round(height * thumb_width / width))
    cell_height = thumb_height + _LABEL_HEIGHT
    canvas = Image.new("RGB", (thumb_width * 3, len(indices) * cell_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        x, y = 0, row * cell_height
        previous, following = frames[index], frames[index + 1]
        delta = np.abs(following.astype(np.int16) - previous.astype(np.int16)).astype(np.uint8)
        for column, image in enumerate((previous, following, delta)):
            canvas.paste(
                Image.fromarray(image).resize(
                    (thumb_width, thumb_height), Image.Resampling.LANCZOS
                ),
                (x + column * thumb_width, y),
            )
        draw.text(
            (x + 5, y + thumb_height + 4),
            f"t={index} -> {index + 1}  {index / fps:.3f}s",
            fill="white",
        )
    from io import BytesIO

    encoded = BytesIO()
    canvas.save(encoded, format="PNG", optimize=False)
    return encoded.getvalue()


def _detail_crop_strip(frames: np.ndarray, crops: tuple[dict[str, int], ...], fps: float) -> bytes:
    """Render three labelled nearest-neighbour 4x crops at native frame size."""

    height, width = frames.shape[1:3]
    canvas = Image.new("RGB", (width * 3, height + _LABEL_HEIGHT), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for column, crop in enumerate(crops):
        index = crop["frame_index"]
        image = Image.fromarray(
            frames[
                index, crop["y"] : crop["y"] + crop["height"], crop["x"] : crop["x"] + crop["width"]
            ]
        ).resize((width, height), Image.Resampling.NEAREST)
        x = column * width
        canvas.paste(image, (x, 0))
        draw.text(
            (x + 5, height + 4),
            f"frame {index}  {index / fps:.3f}s  crop ({crop['x']},{crop['y']})",
            fill="white",
        )
    from io import BytesIO

    encoded = BytesIO()
    canvas.save(encoded, format="PNG", optimize=False)
    return encoded.getvalue()


def make_sheets(
    frames: np.ndarray,
    fps: int | float,
    *,
    detail_indices: tuple[int, ...],
    detail_crops: tuple[dict[str, int], ...],
    transition_indices: tuple[int, ...] | None = None,
) -> list[bytes]:
    """Create pages, spatial evidence, and a symmetric adjacent-transition strip."""

    checked = _checked_frames(frames)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("fps must be positive")
    if len(checked) > _SHEET_CAPACITY * 3:
        raise ValueError("at most 81 frames can be represented in five review images")
    pages = [
        _sheet(
            [
                (index, checked[index])
                for index in range(start, min(start + _SHEET_CAPACITY, len(checked)))
            ],
            float(fps),
        )
        for start in range(0, len(checked), _SHEET_CAPACITY)
    ]
    count = min(len(checked), 9)
    if len(detail_indices) != count or any(
        type(value) is not int
        or not index * len(checked) // count <= value < (index + 1) * len(checked) // count
        for index, value in enumerate(detail_indices)
    ):
        raise ValueError("native detail indices must cover all nine temporal strata")
    pages.append(_native_strip([(index, checked[index]) for index in detail_indices], float(fps)))
    _checked_detail_crops(detail_crops, detail_indices, checked.shape[2], checked.shape[1])
    pages.append(_detail_crop_strip(checked, detail_crops, float(fps)))
    transitions = transition_indices or select_transition_frames(len(checked))
    transition_count = min(len(checked) - 1, 9)
    if (
        len(transitions) != transition_count
        or len(set(transitions)) != transition_count
        or any(
            type(value) is not int
            or not index * (len(checked) - 1) // transition_count
            <= value
            < (index + 1) * (len(checked) - 1) // transition_count
            for index, value in enumerate(transitions)
        )
    ):
        raise ValueError("transition indices must cover all temporal strata")
    pages.append(_transition_strip(checked, transitions, float(fps)))
    return pages
