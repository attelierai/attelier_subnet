"""Real MP4/CLI coverage for the isolated deterministic flicker guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from forge.evaluation.flicker_guard import algorithm_id, validate_envelope

ROOT = Path(__file__).parents[2]
GEOMETRY = {"width": 64, "height": 48, "frames": 9, "fps": 8}


def _base_frames() -> np.ndarray:
    height, width, count = GEOMETRY["height"], GEOMETRY["width"], GEOMETRY["frames"]
    y, x = np.mgrid[:height, :width]
    frames = []
    for index in range(count):
        frame = np.stack(
            ((x * 3 + index * 5) % 256, (y * 5 + index * 3) % 256, ((x + y) * 2) % 256), axis=2
        ).astype(np.uint8)
        left = (index * 5) % (width - 12)
        frame[16:32, left : left + 12] = (255, 32, 128)
        frames.append(frame)
    return np.stack(frames)


def _write(path: Path, frames: np.ndarray) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (GEOMETRY["width"], GEOMETRY["height"]),
        fps=GEOMETRY["fps"],
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)
    for frame in frames:
        writer.send(frame.tobytes())
    writer.close()


def _inspect(
    tmp_path: Path, name: str, frames: np.ndarray, *, steady_lighting: bool = True
) -> dict:
    video, geometry, output = (
        tmp_path / f"{name}.{suffix}" for suffix in ("mp4", "json", "out.json")
    )
    _write(video, frames)
    geometry.write_text(json.dumps(GEOMETRY))
    command = [
        sys.executable,
        "-m",
        "forge.evaluation.flicker_guard",
        "--video",
        str(video),
        "--geometry",
        str(geometry),
        "--out",
        str(output),
    ]
    if not steady_lighting:
        command.append("--no-steady-lighting")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text())


def test_flicker_guard_cli_separates_repeated_photometric_corruption_from_motion(tmp_path):
    base = _base_frames()
    pan = np.stack([np.roll(frame, index * 3, axis=1) for index, frame in enumerate(base)])
    zoom = np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((72, 54), Image.Resampling.BILINEAR))[
                3:51, 4:68
            ]
            for frame in base
        ]
    )
    ramp = np.stack(
        [
            np.clip(frame.astype(np.int16) + index * 8, 0, 255).astype(np.uint8)
            for index, frame in enumerate(base)
        ]
    )
    gray = np.stack(
        [
            np.repeat(
                np.clip(
                    (frame.mean(axis=2, keepdims=True).astype(np.int16) + index * 5), 0, 255
                ).astype(np.uint8),
                3,
                axis=2,
            )
            for index, frame in enumerate(pan)
        ]
    )
    for name, frames in {
        "motion": base,
        "pan": pan,
        "zoom": zoom,
        "ramp": ramp,
        "near-gray-motion-ramp": gray,
    }.items():
        assert _inspect(tmp_path, name, frames)["severe_flicker"] is False
    luminance = np.stack(
        [
            np.clip(frame.astype(np.int16) + (70 if index % 2 else -70), 0, 255).astype(np.uint8)
            for index, frame in enumerate(base)
        ]
    )
    inversion = np.stack([255 - frame if index % 2 else frame for index, frame in enumerate(base)])
    chroma = []
    for index, frame in enumerate(base):
        if index % 2:
            mean = frame.astype(np.float64).mean(axis=2, keepdims=True)
            chroma.append(np.clip(2 * mean - frame, 0, 255).astype(np.uint8))
        else:
            chroma.append(frame)
    for name, frames in {
        "luminance": luminance,
        "inversion": inversion,
        "chroma": np.stack(chroma),
    }.items():
        receipt = _inspect(tmp_path, name, frames)
        assert receipt["severe_flicker"] is True
        assert receipt["video_sha256"] != receipt["decoded_sha256"]
        assert (
            receipt["rule_id"]
            and receipt["receipt_id"]
            and receipt["algorithm_id"] == algorithm_id()
        )
        payload = {"algorithm_id": receipt["algorithm_id"], "diagnostics": receipt["diagnostics"]}
        assert validate_envelope(payload) == receipt["diagnostics"]
        assert (
            validate_envelope(payload, expected_algorithm_id=receipt["algorithm_id"])
            == receipt["diagnostics"]
        )
        with pytest.raises(ValueError, match="identity differs"):
            validate_envelope(payload, expected_algorithm_id="0" * 64)
    assert (
        _inspect(tmp_path, "unsteady", luminance, steady_lighting=False)["severe_flicker"] is False
    )
