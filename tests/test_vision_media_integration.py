from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from forge.evaluation.vision_media import (
    degrade_frames,
    load_video,
    make_sheets,
    media_identity,
    select_detail_crops,
    select_detail_frames,
    select_transition_frames,
    video_diagnostics,
)


def _write_tiny_video(path: Path, frames: np.ndarray, fps: int) -> None:
    writer = imageio_ffmpeg.write_frames(str(path), (frames.shape[2], frames.shape[1]), fps=fps)
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()


def test_video_decode_controls_diagnostics_and_review_sheets(tmp_path):
    frame_count, height, width, fps = 81, 16, 16, 5
    frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    for index in range(frame_count):
        frames[index, :, :, 0] = (index * 3) % 256
        frames[index, :, index % width, 1] = 255
    video = tmp_path / "tiny.mp4"
    _write_tiny_video(video, frames, fps)

    loaded = load_video(
        video, {"width": width, "height": height, "frames": frame_count, "fps": fps}
    )
    with pytest.raises(ValueError, match="more than 80 frames"):
        load_video(video, {"width": width, "height": height, "frames": 80, "fps": fps})
    assert loaded.frames.shape == frames.shape
    assert len(loaded.video_sha256) == 64
    assert loaded.diagnostics["frame_count"] == frame_count
    assert loaded.diagnostics["temporal_delta"] > 0
    assert media_identity(loaded.frames) == media_identity(loaded.frames.copy())

    controls = {
        kind: degrade_frames(loaded.frames, kind, seed=7)
        for kind in ("freeze", "blur", "shuffle", "black")
    }
    assert np.array_equal(controls["freeze"][0], controls["freeze"][-1])
    assert not controls["black"].any()
    assert not np.array_equal(controls["shuffle"], loaded.frames)
    assert video_diagnostics(controls["freeze"])["frozen_fraction"] == 1.0

    indices = select_detail_frames(frame_count)
    crops = select_detail_crops(loaded.frames, indices)
    sheets = make_sheets(loaded.frames, fps, detail_indices=indices, detail_crops=crops)
    assert len(sheets) == 6
    assert sheets[-1] != sheets[0]
    transition_indices = select_transition_frames(frame_count)
    assert len(transition_indices) == 9
    assert all(
        index * 80 // 9 <= value < (index + 1) * 80 // 9
        for index, value in enumerate(transition_indices)
    )
    assert all(sheet.startswith(b"\x89PNG") for sheet in sheets)
    with Image.open(__import__("io").BytesIO(sheets[0])) as sheet:
        assert sheet.width == 1_248
    with Image.open(__import__("io").BytesIO(sheets[-3])) as strip:
        assert strip.size == (width * 3, (height + 24) * 3)
        pixels = np.asarray(strip)
        for position, index in enumerate(indices):
            row, column = divmod(position, 3)
            x, y = column * width, row * (height + 24)
            assert np.array_equal(pixels[y : y + height, x : x + width], loaded.frames[index])
    with Image.open(__import__("io").BytesIO(sheets[-2])) as crops_sheet:
        assert crops_sheet.size == (width * 3, height + 24)
        pixels = np.asarray(crops_sheet)
        for position, crop in enumerate(crops):
            region = loaded.frames[
                crop["frame_index"],
                crop["y"] : crop["y"] + crop["height"],
                crop["x"] : crop["x"] + crop["width"],
            ]
            expected = np.asarray(
                Image.fromarray(region).resize((width, height), Image.Resampling.NEAREST)
            )
            assert np.array_equal(
                pixels[:height, position * width : (position + 1) * width], expected
            )
    with pytest.raises(ValueError, match="all nine temporal strata"):
        make_sheets(loaded.frames, fps, detail_indices=(0, 40, 80), detail_crops=crops)


def test_encoded_temporal_signals_separate_flicker_from_ramps_and_motion(tmp_path):
    """Exercise actual video encoding/decoding, including legitimate changes."""
    count, size, fps = 81, 64, 16
    base = np.full((count, size, size, 3), 100, dtype=np.uint8)
    moving = base.copy()
    for index in range(count):
        left = index // 2
        moving[index, 16:32, left:left + 16] = 180
    ramp = np.clip(base.astype(np.int16) + np.arange(count)[:, None, None, None], 0, 255)
    cases = {"static": base, "ramp": ramp.astype(np.uint8), "motion": moving}
    for name, strength in (("mild", 3), ("severe", 16)):
        offset = np.where(np.arange(count) % 2 == 0, strength, 0)
        cases[name] = np.clip(
            moving.astype(np.int16) + offset[:, None, None, None], 0, 255
        ).astype(np.uint8)
    cases["shuffle"] = moving[np.random.default_rng(91).permutation(count)]
    observed = {}
    for name, frames in cases.items():
        path = tmp_path / f"{name}.mp4"
        _write_tiny_video(path, frames, fps)
        observed[name] = load_video(path, {
            "width": size, "height": size, "frames": count, "fps": fps,
        }).diagnostics
    assert observed["severe"]["luma_acceleration_median"] > (
        observed["mild"]["luma_acceleration_median"]
    ) > max(observed[name]["luma_acceleration_median"] for name in ("static", "ramp", "motion"))
    assert observed["shuffle"]["centered_transition_median"] > (
        observed["motion"]["centered_transition_median"]
    )
