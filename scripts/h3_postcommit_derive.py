#!/usr/bin/env python3
"""Closed-research post-commit audits with validator-owned conditioning.

Only the pinned BasicGuider CFG=1/no-inpaint profile is supported. This creates
challenges; canonical replay and endpoint checks remain separately required.
"""

from __future__ import annotations

import argparse
import copy
import secrets
import sys
from pathlib import Path

from h3_trace_campaign import digest_file, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comfy-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--conditioning-manifest", type=Path, required=True)
    p.add_argument("--steps", help="Explicit indices for reconstruction control ONLY")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.count <= 20:
        p.error("Audit count must be between 1 and 20")
    sys.path.insert(0, str(a.comfy_dir))
    import json
    import os

    import comfy.utils
    import torch

    m = json.loads(a.manifest.read_text())
    c = json.loads(a.conditioning_manifest.read_text())
    if m.get("inputHash") != c.get("inputHash") or not m.get("output"):
        raise ValueError("Complete candidate and same trusted request required")
    trace, trusted_trace = m["trace"], c["trace"]
    if len(trace) != 20 or len(trusted_trace) != 20:
        raise ValueError("Approved 20-step profile required")
    if [r["sigma"] for r in trace] != [r["sigma"] for r in trusted_trace]:
        raise ValueError("Candidate schedule differs from trusted profile")

    def load(root, descriptor):
        q = (root / descriptor["tensorPath"]).resolve()
        q.relative_to(root.resolve())
        if digest_file(q) != descriptor["sha256"]:
            raise ValueError("Tensor custody mismatch")
        value = torch.load(q, map_location="cpu", weights_only=True)
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite tensor")
            if list(value.shape) != descriptor["shape"]:
                raise ValueError("Tensor shape mismatch")
        return value

    # The template is authority held by the validator, never a miner assertion.
    base = load(a.conditioning_manifest.parent, c["forward"][0])
    source = {
        "manifestSha256": digest_file(a.manifest),
        "conditioningManifestSha256": digest_file(a.conditioning_manifest),
        "conditioningBundleSha256": c["forward"][0]["sha256"],
        "inputHash": m["inputHash"],
        "schema": "forge/h3-postcommit-challenge/v1",
    }
    a.out.mkdir(parents=True, exist_ok=False)
    write_new(a.out / "commitment.json", source)
    steps = (
        [int(x) for x in a.steps.split(",")]
        if a.steps
        else sorted(secrets.SystemRandom().sample(range(20), a.count))
    )
    if not steps or len(set(steps)) != len(steps) or any(i not in range(20) for i in steps):
        raise ValueError("Invalid step selection")
    challenge = {
        **source,
        "steps": steps,
        "selection": "manual_reconstruction_control" if a.steps else "postcommit_system_random",
        "auditPolicy": "all_steps" if len(steps) == 20 else "sampled_steps",
        "comparisonSampleSteps": sorted(secrets.SystemRandom().sample(steps, min(3, len(steps)))),
    }
    write_new(a.out / "challenge.json", challenge)
    bundles = []
    for i in steps:
        row = trace[i]
        parts = []
        for kind in ("stateCallback", "denoised"):
            items = row[kind]
            if [x["component"] for x in items] != ["video", "audio"]:
                raise ValueError("Approved video/audio order required")
            parts.append(
                comfy.utils.pack_latents([load(a.manifest.parent, item) for item in items])[0]
            )
        conditioning = copy.deepcopy(base["conditioning"])
        timestep = torch.full_like(base["timestep"], float(trusted_trace[i]["sigma"]))
        # sample_sigmas is the full schedule; only sigmas is the current step.
        conditioning["transformer_options"]["sigmas"] = timestep.clone()
        bundle = {
            "input": parts[0],
            "timestep": timestep,
            "conditioning": conditioning,
            "prediction": parts[1],
        }
        q = a.out / f"candidate-{i:03d}.pt"
        with q.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        bundles.append({"step": i, "path": q.name, "sha256": digest_file(q)})
    if source["manifestSha256"] != digest_file(a.manifest) or source[
        "conditioningManifestSha256"
    ] != digest_file(a.conditioning_manifest):
        raise ValueError("Source changed during challenge")
    write_new(
        a.out / "receipt.json",
        {
            **challenge,
            "bundles": bundles,
            "limits": [
                "trusted conditioning required",
                "single pinned profile",
                "not an initial seed or final video proof",
            ],
        },
    )
    print("DERIVED")


if __name__ == "__main__":
    main()
