#!/usr/bin/env python3
"""Read-only sanitized audit of ComfyUI /history for a serving interval."""

import argparse
import json
import urllib.parse
import urllib.request

EXECUTION_SAMPLERS = {"SamplerCustomAdvanced", "SamplerCustom", "KSampler", "KSamplerAdvanced"}


def _timestamp(messages, name):
    for event in messages:
        if (
            isinstance(event, list)
            and len(event) > 1
            and event[0] == name
            and isinstance(event[1], dict)
        ):
            value = event[1].get("timestamp")
            if isinstance(value, (int, float)):
                return int(value)
    return None


def sanitize(history, start_ms, end_ms):
    result = []
    for prompt_id, row in history.items():
        if not isinstance(prompt_id, str) or not isinstance(row, dict):
            continue
        status = row.get("status", {})
        messages = status.get("messages", []) if isinstance(status, dict) else []
        started, ended = (
            _timestamp(messages, "execution_start"),
            _timestamp(messages, "execution_success"),
        )
        if started is None or not (start_ms <= started < end_ms):
            continue
        prompt = row.get("prompt", [])
        graph = (
            prompt[2]
            if isinstance(prompt, list) and len(prompt) > 2 and isinstance(prompt[2], dict)
            else {}
        )
        samplers = []
        seeds = []
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            kind = node.get("class_type", "")
            inputs = node.get("inputs", {})
            if (
                kind == "RandomNoise"
                and isinstance(inputs, dict)
                and isinstance(inputs.get("noise_seed"), (int, float))
            ):
                seeds.append(int(inputs["noise_seed"]))
            if kind in EXECUTION_SAMPLERS:
                samplers.append({"id": str(node_id), "class": kind})
        cached = []
        for event in messages:
            if (
                isinstance(event, list)
                and len(event) > 1
                and event[0] == "execution_cached"
                and isinstance(event[1], dict)
            ):
                nodes = event[1].get("nodes", [])
                if isinstance(nodes, list):
                    cached.extend(str(x) for x in nodes)
        cached = sorted(set(cached))
        sampler_ids = {x["id"] for x in samplers}
        result.append(
            {
                "prompt_id": prompt_id,
                "executionStartMs": started,
                "executionEndMs": ended,
                "seeds": seeds,
                "samplers": samplers,
                "cachedNodeIds": cached,
                "samplerReused": bool(sampler_ids & set(cached)),
                "samplerExecuted": bool(sampler_ids - set(cached)),
                "executionSuccess": status.get("status_str") == "success",
            }
        )
    return sorted(result, key=lambda x: (x["executionStartMs"], x["prompt_id"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--startms", type=int, required=True)
    p.add_argument("--endms", type=int, required=True)
    p.add_argument("--origin", default="http://127.0.0.1:8188")
    args = p.parse_args()
    parsed = urllib.parse.urlparse(args.origin)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("origin must be loopback HTTP")
    if args.endms <= args.startms:
        raise SystemExit("endms must exceed startms")
    with urllib.request.urlopen(args.origin.rstrip("/") + "/history", timeout=15) as response:
        history = json.load(response)
    print(
        json.dumps(
            {
                "schema": "attelier/h3-comfy-cache-audit/v1",
                "runs": sanitize(history, args.startms, args.endms),
                "limitation": (
                    "History is local execution telemetry, not a checkpoint or authority proof."
                ),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
