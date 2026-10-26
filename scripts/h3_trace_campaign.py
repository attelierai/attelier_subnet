#!/usr/bin/env python3
"""Run a bounded private Comfy trace batch; never publishes rewards or retries jobs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from h3_trace_capture import add_trace_graph


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def media_digest(path, kind):
    args = ["ffmpeg", "-v", "error", "-i", str(path)]
    args += (
        ["-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        if kind == "video"
        else ["-map", "0:a:0", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"]
    )
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        digest = hashlib.sha256()
        size = 0
        while data := process.stdout.read(1024 * 1024):
            size += len(data)
            digest.update(data)
        if process.wait(timeout=120) != 0 or not size:
            raise RuntimeError("Media decode failed")
    return {"sha256": digest.hexdigest(), "bytes": size}


def request(origin, route, payload=None):
    body = None if payload is None else canonical(payload)
    req = urllib.request.Request(
        origin + route, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def run(args, batch):
    root = args.root.resolve()
    template = json.loads(args.template.read_text())
    for job in batch["jobs"]:
        if time.time() >= batch["deadlineUnix"] - 600:
            raise RuntimeError("Insufficient lease time for a new job")
        run_id = job["runId"]
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("Unsafe run id")
        directory = root / "jobs" / run_id
        # An intent without completion is an uncertainty to reconcile, not permission to retry.
        if (directory / "intent.json").exists():
            if (directory / "completion.json").exists():
                print(json.dumps({"runId": run_id, "status": "already_completed"}), flush=True)
                continue
            raise RuntimeError(f"Unresolved job intent: {run_id}")
        document = copy.deepcopy(template)
        graph = document["graph"]
        params = job["parameters"]
        graph["5"]["inputs"]["prompt"] = params["prompt"]
        graph["6"]["inputs"]["noise_seed"] = params["seed"]
        graph["9"]["inputs"]["steps"] = job.get("actualSteps", 20)
        graph["14"]["inputs"]["filename_prefix"] = "trace/" + run_id
        if job.get("checkpoint"):
            graph["1"]["inputs"]["unet_name"] = job["checkpoint"]
        if job.get("lora"):
            graph["16"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["1", 0], "lora_name": job["lora"], "strength_model": 1},
            }
            graph["7"]["inputs"]["model"] = ["16", 0]
            graph["9"]["inputs"]["model"] = ["16", 0]
        if job.get("sampler"):
            graph["8"]["inputs"]["sampler_name"] = job["sampler"]
        input_hash = hashlib.sha256(canonical(params)).hexdigest()
        metadata = SimpleNamespace(
            trace_root=str(root / "captures"),
            run_id=run_id,
            split=job["split"],
            condition=job["condition"],
            lineage_id=job["lineageId"],
            input_hash=input_hash,
            model_manifest_hash=batch["declaredModelManifestHash"],
            workflow_hash=batch["declaredWorkflowHash"],
            runtime_id=batch["runtimeId"],
            audio=True,
        )
        document = add_trace_graph(document, metadata)
        if "skipModelStep" in job:
            skip = job["skipModelStep"]
            if type(skip) is not int or not 1 <= skip < 20:
                raise ValueError("Controlled skip must be an interior model-call index")
            document["graph"]["10"]["inputs"]["skip_model_step"] = skip
            document["traceExperiment"]["graphHash"] = hashlib.sha256(
                canonical(document["graph"])
            ).hexdigest()
        write_new(directory / "graph.json", document)
        write_new(
            directory / "intent.json",
            {
                "job": job,
                "inputHash": input_hash,
                "graphSha256": digest_file(directory / "graph.json"),
                "at": time.time(),
            },
        )
        started = time.monotonic()
        submitted = request(
            args.origin,
            "/prompt",
            {"prompt": document["graph"], "client_id": "attelier-trace-" + run_id},
        )
        write_new(directory / "submission.json", submitted)
        prompt_id = submitted["prompt_id"]
        history = None
        while time.monotonic() - started < 600:
            item = request(args.origin, "/history/" + prompt_id).get(prompt_id)
            if item:
                history = item
                break
            time.sleep(2)
        if history is None:
            raise RuntimeError("Comfy job timed out; reconcile before another submission")
        write_new(directory / "history.json", history)
        if history.get("status", {}).get("status_str") != "success":
            raise RuntimeError(f"Comfy generation failed: {run_id}; inspect private history")
        media = []
        for node in history.get("outputs", {}).values():
            for key in ("videos", "images", "gifs"):
                for file in node.get(key, []):
                    if str(file.get("filename", "")).endswith(".mp4"):
                        media.append(
                            args.output_directory / file.get("subfolder", "") / file["filename"]
                        )
        if len(media) != 1:
            raise RuntimeError("Expected exactly one generated MP4")
        video = media[0].resolve()
        video.relative_to(args.output_directory.resolve())
        generation_seconds = time.monotonic() - started
        result = {
            "runId": run_id,
            "status": "completed",
            "instrumentedGenerationSeconds": generation_seconds,
            "media": str(video),
            "sha256": digest_file(video),
            "video": media_digest(video, "video"),
            "audio": media_digest(video, "audio"),
            "verificationClaim": "capture_only_not_checkpoint_proof",
        }
        manifest_path = root / "captures" / run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["output"] = result
        temporary = manifest_path.with_suffix(".new")
        write_new(temporary, manifest)
        temporary.replace(manifest_path)
        result["captureManifestSha256"] = digest_file(manifest_path)
        write_new(directory / "completion.json", result)
        print(
            json.dumps(
                {"runId": run_id, "status": "completed", "seconds": round(generation_seconds, 3)}
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--origin", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    if not args.origin.startswith("http://127.0.0.1:"):
        parser.error("Private loopback Comfy origin required")
    run(args, json.loads(args.batch.read_text()))


if __name__ == "__main__":
    main()
