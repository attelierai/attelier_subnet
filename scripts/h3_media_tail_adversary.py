#!/usr/bin/env python3
"""Create a private AAC-tail adversary without re-encoding the source media.

The output is diagnostic evidence only.  It preserves copied trace tensors and
rebinds them to a new MP4 whose video packets are stream-copied and whose audio
has an appended AAC silence segment.  The artifact is emitted only after the
decoded audio prefix and compressed video packet payloads are proven equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from h3_trace_adversary import contained, copy_blobs, read_manifest, repacked_id, sha256, write_json
from h3_trace_campaign import media_digest

SCHEMA = "forge/h3-media-tail-adversary/v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def under(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    return contained(root, raw.resolve() if raw.is_absolute() else (root / raw).resolve())


def command(args: list[str], *, timeout: int = 120) -> bytes:
    return subprocess.run(args, check=True, capture_output=True, timeout=timeout).stdout


def stream_duration(media: Path, kind: str) -> float:
    payload = command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            f"{kind}:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(media),
        ]
    )
    streams = json.loads(payload).get("streams", [])
    if len(streams) != 1 or streams[0].get("duration") in (None, "N/A"):
        raise ValueError(f"{kind}_duration_missing")
    duration = float(streams[0]["duration"])
    if duration <= 0:
        raise ValueError(f"{kind}_duration_invalid")
    return duration


def video_packet_digest(media: Path) -> str:
    payload = command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=size,data_hash",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(media),
        ]
    )
    packets = json.loads(payload).get("packets", [])
    if not packets or any("data_hash" not in packet for packet in packets):
        raise ValueError("video_packet_hash_missing")
    return hashlib.sha256(canonical(packets)).hexdigest()


def audio_prefix(media: Path, expected_samples: int, rate: int) -> bytes:
    raw = command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            str(rate),
            "-t",
            str((expected_samples + 4096) / rate),
            "-f",
            "f32le",
            "-",
        ]
    )
    width = expected_samples * 2 * 4
    if len(raw) < width:
        raise ValueError("audio_prefix_short")
    return raw[:width]


def write_failure(out: Path, reason: str) -> None:
    write_json(
        out / "diagnostic.json",
        {"schema": SCHEMA, "status": "INCONCLUSIVE", "reason": reason},
    )


def append_tail(source: Path, destination: Path, work: Path, rate: int, seconds: float) -> None:
    """Copy original AAC packets into an m4a segment, concat a new AAC segment, then remux."""
    prefix = work / "prefix.m4a"
    tail = work / "tail.m4a"
    joined = work / "joined.m4a"
    listing = work / "segments.txt"
    command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(prefix),
        ]
    )
    command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={rate}:cl=stereo",
            "-t",
            str(seconds),
            "-c:a",
            "aac",
            str(tail),
        ]
    )
    listing.write_text(f"file '{prefix}'\nfile '{tail}'\n")
    os.chmod(listing, 0o600)
    command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-copyts",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(joined),
        ]
    )
    command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-i",
            str(joined),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise ValueError("experiment_root_missing")
    output = under(root, args.out)
    if output.exists():
        raise FileExistsError("outdir_must_be_fresh")
    manifest_path, source_manifest = read_manifest(root, str(under(root, args.manifest)))
    source_output = source_manifest.get("output")
    if not isinstance(source_output, dict) or not isinstance(source_output.get("media"), str):
        raise ValueError("source_output_media_missing")
    source_media = under(root, source_output["media"])
    if not source_media.is_file() or sha256(source_media) != source_output.get("sha256"):
        raise ValueError("source_media_custody")
    output.mkdir(parents=True, mode=0o700)
    work = output / "work"
    work.mkdir(mode=0o700)
    target_media = output / "media" / "tail-appended.mp4"
    target_media.parent.mkdir(mode=0o700)
    try:
        prefix = audio_prefix(source_media, args.expected_samples, args.sample_rate)
        source_video = video_packet_digest(source_media)
        append_tail(source_media, target_media, work, args.sample_rate, args.tail_seconds)
        target_prefix = audio_prefix(target_media, args.expected_samples, args.sample_rate)
        target_video = video_packet_digest(target_media)
        video_seconds = stream_duration(target_media, "v")
        audio_seconds = stream_duration(target_media, "a")
        if target_prefix != prefix:
            raise ValueError("decoded_audio_prefix_changed")
        if target_video != source_video:
            raise ValueError("compressed_video_packets_changed")
        if audio_seconds <= video_seconds + 1.0:
            raise ValueError("audio_tail_not_over_one_second")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        write_failure(output, str(exc))
        print(json.dumps({"schema": SCHEMA, "status": "INCONCLUSIVE", "reason": str(exc)}))
        raise SystemExit(2) from None

    copied = copy_blobs(root, manifest_path, source_manifest, output)
    run_id = repacked_id(manifest_path, target_media)
    copied["runId"] = run_id
    output_receipt = dict(source_output)
    output_receipt.update(
        {
            "runId": run_id,
            "media": str(target_media),
            "sha256": sha256(target_media),
            "video": media_digest(target_media, "video"),
            "audio": media_digest(target_media, "audio"),
            "verificationClaim": "controlled_aac_tail_adversary_not_checkpoint_proof",
        }
    )
    copied["output"] = output_receipt
    write_json(output / "manifest.json", copied)
    write_json(
        output / "operator-ground-truth.json",
        {
            "schema": SCHEMA,
            "status": "MEASURED",
            "label": "audio_tail_after_verified_prefix",
            "sourceManifestSha256": sha256(manifest_path),
            "sourceMediaSha256": sha256(source_media),
            "adversarialMediaSha256": sha256(target_media),
            "expectedSamples": args.expected_samples,
            "sampleRate": args.sample_rate,
            "sourcePrefixSha256": hashlib.sha256(prefix).hexdigest(),
            "adversarialPrefixSha256": hashlib.sha256(target_prefix).hexdigest(),
            "sourceVideoPacketSha256": source_video,
            "adversarialVideoPacketSha256": target_video,
            "videoSeconds": video_seconds,
            "audioSeconds": audio_seconds,
            "audioTailSeconds": audio_seconds - video_seconds,
            "limits": [
                "controlled diagnostic only",
                "video packet payloads and decoded prefix are compared",
                "does not prove miner behavior or VAE provenance",
            ],
        },
    )
    shutil.rmtree(work)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "MEASURED",
                "manifest": str(output / "manifest.json"),
                "audioTailSeconds": audio_seconds - video_seconds,
            }
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", required=True)
    result.add_argument("--manifest", required=True, help="Path relative to --root")
    result.add_argument("--out", required=True, help="Fresh path relative to --root")
    result.add_argument("--expected-samples", type=int, default=165334)
    result.add_argument("--sample-rate", type=int, default=48000)
    result.add_argument("--tail-seconds", type=float, default=2.0)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.expected_samples < 1 or parsed.sample_rate < 1 or parsed.tail_seconds <= 1:
        parser().error("positive expected samples/rate and tail >1 second required")
    run(parsed)
