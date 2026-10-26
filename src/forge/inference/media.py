"""Independent decoding of miner results. Run beside remote artifact storage."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def canonical_decode(path: Path) -> dict[str, Any]:
    """Hash decoded streams, avoiding container metadata as the replay oracle."""
    with __import__("tempfile").TemporaryDirectory() as temporary:
        root = Path(temporary)
        video, audio = root / "video.yuv", root / "audio.f32le"
        for selector, output, arguments in (
            ("0:v:0", video, ["-pix_fmt", "yuv420p", "-f", "rawvideo"]),
            ("0:a:0", audio, ["-ac", "2", "-ar", "48000", "-f", "f32le"]),
        ):
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(path),
                    "-map",
                    selector,
                    *arguments,
                    str(output),
                ],
                capture_output=True,
                timeout=180,
            )
            if result.returncode:
                raise ValueError("canonical_decode_failed")

        def digest(value: Path) -> str:
            if not value.is_file() or not value.stat().st_size:
                raise ValueError("empty_decoded_stream")
            with value.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()

        value = {"canonicalVideoSha256": digest(video), "canonicalAudioSha256": digest(audio)}
        value["canonicalMediaSha256"] = hashlib.sha256(
            json.dumps(
                {"audio": value["canonicalAudioSha256"], "video": value["canonicalVideoSha256"]},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        return value


def inspect_video(path: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    if path.stat().st_size > 512 * 1024 * 1024 or not path.stat().st_size:
        raise ValueError("result_size")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        timeout=60,
    )
    info = json.loads(probe.stdout)
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if len(video) != 1 or (parameters.get("audio") and not audio):
        raise ValueError("missing_stream")
    # The pilot has one explicitly shared geometry on both architectures.
    if parameters["resolution"] != "480p" or parameters["aspectRatio"] != "16:9":
        raise ValueError("unqualified_geometry")
    if (video[0]["width"], video[0]["height"]) != (832, 480):
        raise ValueError("wrong_dimensions")
    if Fraction(video[0].get("avg_frame_rate", "0/1")) != 24:
        raise ValueError("wrong_frame_rate")
    duration = float(info["format"]["duration"])
    if not math.isfinite(duration) or abs(duration - parameters["duration"]) > 0.15:
        raise ValueError("wrong_duration")
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=120,
    )
    if decoded.returncode:
        raise ValueError("decode_failed")
    return {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "width": 832,
        "height": 480,
        "duration": duration,
        "audio": bool(audio),
        "decoded": True,
        **canonical_decode(path),
    }
