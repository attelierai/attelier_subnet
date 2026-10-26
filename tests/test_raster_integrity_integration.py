from pathlib import Path

import imageio_ffmpeg
import numpy as np

from forge.evaluation.raster_integrity import RULE_ID, check_frames, verify_check
from forge.evaluation.vision_media import load_video
from forge.protocol.hashing import sha256_file


def _video(path: Path, frames: np.ndarray) -> None:
    writer = imageio_ffmpeg.write_frames(str(path), (frames.shape[2], frames.shape[1]), fps=8)
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()


def test_decoded_all_frame_raster_preflight_detects_checker_attack_but_not_clean_video(tmp_path):
    clean = np.full((81, 64, 64, 3), 128, dtype=np.uint8)
    attacked = clean.copy()
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 2 - 1).astype(np.int16) * 64
    for index in range(len(attacked)):
        if index not in {0, 40, 80}:
            attacked[index] = np.clip(128 + checker[..., None], 0, 255).astype(np.uint8)
    clean_path, attacked_path = tmp_path / "clean.mp4", tmp_path / "attacked.mp4"
    _video(clean_path, clean)
    _video(attacked_path, attacked)
    geometry = {"width": 64, "height": 64, "frames": 81, "fps": 8}
    clean_media, attacked_media = (
        load_video(clean_path, geometry),
        load_video(attacked_path, geometry),
    )
    clean_check = check_frames(clean_media.frames, sha256_file(clean_path))
    attacked_check = check_frames(attacked_media.frames, sha256_file(attacked_path))

    assert clean_check["status"] == "CLEAR"
    assert attacked_check["status"] == "ANOMALY"
    assert len(attacked_check["flagged_frames"]) >= 78
    assert attacked_check["rule_id"] == RULE_ID
    assert verify_check(attacked_check, sha256_file(attacked_path), geometry)


def test_receipt_rejects_a_hash_consistent_contradictory_summary(tmp_path):
    """A trusted receipt still must describe its metrics consistently."""
    from forge.evaluation.raster_integrity import verify_check
    from forge.protocol.hashing import hash_canonical

    frames = np.full((4, 64, 64, 3), 128, dtype=np.uint8)
    video = tmp_path / "clear.mp4"
    _video(video, frames)
    loaded = load_video(video, {"width": 64, "height": 64, "frames": 4, "fps": 8})
    receipt = check_frames(loaded.frames, loaded.video_sha256)
    assert verify_check(receipt, loaded.video_sha256, {"width": 64, "height": 64, "frames": 4})

    contradictory = dict(receipt)
    contradictory["status"] = "ANOMALY"
    body = {key: value for key, value in contradictory.items() if key != "check_hash"}
    contradictory["check_hash"] = hash_canonical("forge/raster-integrity-check/v1", body)
    assert not verify_check(
        contradictory, loaded.video_sha256, {"width": 64, "height": 64, "frames": 4}
    )
