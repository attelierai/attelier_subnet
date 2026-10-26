#!/usr/bin/env python3
"""CPU H3 FLOW_AV endpoint diagnostic; never a certificate or media proof."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forge.inference.trace_eval import _tensor

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--trusted-manifest", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--audio-scale", type=float, default=4.0)
p.add_argument("--noise-scale", type=float, default=1.0)
a = p.parse_args()


def dig(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(path):
    m = json.loads(path.read_text())
    root = path.parent

    def parts(items):
        out = {}
        if [d.get("component") for d in items] != ["video", "audio"]:
            raise ValueError("expected_video_audio_components")
        for d in items:
            out[d["component"]] = _tensor(root, d)
        return out

    return m, parts


try:
    if a.audio_scale != 4 or a.noise_scale != 1:
        raise ValueError("unsupported_pinned_profile")
    m, parts = run(a.manifest)
    t, tparts = run(a.trusted_manifest)
    if (
        m.get("inputHash") != t.get("inputHash")
        or len(t.get("trace", [])) != 20
        or len(m.get("trace", [])) != 20
    ):
        raise ValueError("schema_or_schedule_inconclusive")
    if [float(x["sigma"]) for x in m["trace"]] != [float(x["sigma"]) for x in t["trace"]]:
        raise ValueError("sigma_schedule_mismatch")
    noise, latent = parts(m["inputNoise"]), parts(m["inputLatent"])
    tnoise, tlatent = tparts(t["inputNoise"]), tparts(t["inputLatent"])
    if any(
        not np.array_equal(noise[k], tnoise[k]) or not np.array_equal(latent[k], tlatent[k])
        for k in noise
    ):
        raise ValueError("trusted_input_mismatch")
    s = float(m["trace"][0]["sigma"])
    first = parts(m["trace"][0]["stateCallback"])
    metrics = []
    for k in first:
        e = s * a.noise_scale * noise[k] + (1 - s) * latent[k] - first[k]
        metrics.append(e)
    last = parts(m["trace"][-1]["denoised"])
    final = parts(m["finalLatent"])
    for k in final:
        want = last[k] / a.audio_scale if k == "audio" else last[k]
        metrics.append(want - final[k])
    flat = np.concatenate([x.ravel() for x in metrics])
    rms = float(np.sqrt(np.mean(flat * flat)))
    out = {
        "schema": "forge/inference-endpoint-binding/v1",
        "status": "MEASURED",
        "maxAbs": float(np.max(np.abs(flat))),
        "rms": rms,
        "relativeRms": rms
        / max(
            float(np.sqrt(np.mean(np.concatenate([x.ravel() for x in first.values()]) ** 2))), 1e-12
        ),
        "profile": {"audioScale": a.audio_scale, "noiseScale": a.noise_scale},
        "claim": "endpoint_latent_diagnostic_only",
        "limits": ["requires_trusted_inputs", "no_forward_or_media_binding", "not_certificate"],
    }
except Exception as e:
    out = {
        "schema": "forge/inference-endpoint-binding/v1",
        "status": "INCONCLUSIVE",
        "reason": type(e).__name__,
    }
a.out.write_text(json.dumps(out, indent=2) + "\n")
print(out["status"])
