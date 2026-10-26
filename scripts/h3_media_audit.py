#!/usr/bin/env python3
"""Bounded private VAE-to-delivered-media measurements; never rewards a miner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from h3_trace_campaign import digest_file, request, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--audit-subdirectory", default="media-audits")
    parser.add_argument("--origin", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    if not args.origin.startswith("http://127.0.0.1:"):
        parser.error("Private loopback Comfy origin required")
    if Path(args.audit_subdirectory).name != args.audit_subdirectory or args.audit_subdirectory in (
        "",
        ".",
        "..",
    ):
        parser.error("Plain audit directory name required")
    for path in args.manifests:
        path = path.resolve()
        path.relative_to(args.root.resolve())
        manifest = json.loads(path.read_text())
        run_id = manifest["runId"]
        if Path(run_id).name != run_id or run_id in ("", ".", ".."):
            raise ValueError("Plain run identifier required")
        media = Path(manifest["output"]["media"]).resolve()
        media.relative_to(args.root.resolve())
        manifest_hash = digest_file(path)
        media_hash = digest_file(media)
        if media_hash != manifest["output"]["sha256"]:
            raise ValueError("Delivered media custody mismatch")
        directory = args.root / args.audit_subdirectory / run_id
        result = directory / "result.json"
        if (directory / "completion.json").exists():
            old = json.loads((directory / "completion.json").read_text())
            if old["manifestSha256"] != manifest_hash or old["mediaSha256"] != media_hash:
                raise ValueError("Existing audit belongs to different evidence")
            if old["resultSha256"] != digest_file(result):
                raise ValueError("Existing audit result changed")
            print(json.dumps({"runId": run_id, "status": "already_measured"}), flush=True)
            continue
        if (directory / "intent.json").exists():
            raise RuntimeError("Unresolved media audit intent; reconcile before retry")
        if time.time() + 600 >= args.deadline_unix:
            raise RuntimeError("Insufficient lease for bounded VAE audit")
        graph = {
            "1": {"class_type": "H3TraceFinalLatentLoader", "inputs": {"manifest_path": str(path)}},
            "2": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
            },
            "4": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["1", 0],
                    "vae": ["2", 0],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            },
            "5": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["1", 0], "vae": ["3", 0]}},
            "6": {
                "class_type": "H3TraceMediaAudit",
                "inputs": {
                    "images": ["4", 0],
                    "audio": ["5", 0],
                    "media_path": str(media),
                    "result_path": str(result),
                },
            },
        }
        write_new(
            directory / "intent.json",
            {"graph": graph, "manifestSha256": manifest_hash, "mediaSha256": media_hash},
        )
        started = time.monotonic()
        submission = request(
            args.origin, "/prompt", {"prompt": graph, "client_id": "media-audit-" + run_id}
        )
        write_new(directory / "submission.json", submission)
        history = None
        while time.monotonic() - started < 600:
            history = request(args.origin, "/history/" + submission["prompt_id"]).get(
                submission["prompt_id"]
            )
            if history:
                break
            time.sleep(2)
        write_new(directory / "history.json", history)
        if not history or history.get("status", {}).get("status_str") != "success":
            raise RuntimeError("Media audit infrastructure failed; inspect private history")
        receipt = json.loads(result.read_text())
        if (
            receipt["mediaSha256"] != media_hash
            or digest_file(path) != manifest_hash
            or digest_file(media) != media_hash
        ):
            raise ValueError("Media evidence changed during audit")
        write_new(
            directory / "completion.json",
            {
                "manifestSha256": manifest_hash,
                "mediaSha256": media_hash,
                "resultSha256": digest_file(result),
                "seconds": time.monotonic() - started,
            },
        )
        print(
            json.dumps(
                {
                    "runId": run_id,
                    "status": receipt["status"],
                    "video": receipt["video"],
                    "audio": receipt["audio"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
