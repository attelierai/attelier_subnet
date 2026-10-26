from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tail_adversary_stream_copies_video_and_keeps_decoded_audio_prefix(tmp_path):
    media = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(media),
        ],
        check=True,
    )
    manifest = tmp_path / "captures" / "honest" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "forge/inference-trace-capture/v1",
                "runId": "honest",
                "lineageId": "lineage",
                "inputHash": "input",
                "trace": [],
                "output": {"media": str(media), "sha256": sha(media)},
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verification/h3_media_tail_adversary.py",
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest.relative_to(tmp_path)),
            "--out",
            "tail-adversary",
            "--expected-samples",
            "48000",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
    proof = json.loads((tmp_path / "tail-adversary" / "operator-ground-truth.json").read_text())
    assert proof["sourcePrefixSha256"] == proof["adversarialPrefixSha256"]
    assert proof["sourceVideoPacketSha256"] == proof["adversarialVideoPacketSha256"]
    assert proof["audioTailSeconds"] > 1
