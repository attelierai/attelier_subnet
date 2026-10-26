from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def test_bounded_aac_decode_accepts_honest_and_rejects_appended_tail(tmp_path):
    source = tmp_path / "honest.m4a"
    tail = tmp_path / "tail.m4a"
    joined = tmp_path / "tail-appended.m4a"
    offset = tmp_path / "tail-appended-offset.m4a"
    listing = tmp_path / "segments.txt"
    run(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=32000",
        "-t",
        "5.175",
        "-ac",
        "2",
        "-c:a",
        "aac",
        str(source),
    )
    run(
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=32000:cl=stereo",
        "-t",
        "2",
        "-c:a",
        "aac",
        str(tail),
    )
    listing.write_text(f"file '{source}'\nfile '{tail}'\n")
    run("-f", "concat", "-safe", "0", "-copyts", "-i", str(listing), "-c", "copy", str(joined))
    run("-itsoffset", "10", "-i", str(joined), "-c", "copy", str(offset))

    node_path = Path("scripts/verification/h3_media_audit_node.py")
    spec = importlib.util.spec_from_file_location("h3_media_audit_node", node_path)
    assert spec and spec.loader
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    allowed = 165334 + 4096
    honest = node.bounded_audio_samples(source, 32000, allowed)
    assert 165334 <= len(honest) <= allowed
    with pytest.raises(ValueError, match="Delivered audio length differs"):
        node.bounded_audio_samples(joined, 32000, allowed)
    with pytest.raises(ValueError, match="Delivered audio length differs"):
        node.bounded_audio_samples(offset, 32000, allowed)
