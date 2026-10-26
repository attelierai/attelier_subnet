#!/usr/bin/env python3
"""CPU diagnostic binding captured Comfy callback states to forward bundles.

Only valid for the closed controlled BasicGuider CFG=1/no-inpaint profile.
It does not establish that conditioning is trustworthy outside that baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--comfy-dir", type=Path, required=True)
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    x = a.parse_args()
    sys.path.insert(0, str(x.comfy_dir))
    import comfy.utils
    import torch

    m = json.loads(x.manifest.read_text())
    root = x.manifest.parent

    def load(d):
        p = (root / d["tensorPath"]).resolve()
        p.relative_to(root.resolve())
        if digest(p) != d["sha256"]:
            raise ValueError("tensor_custody")
        return torch.load(p, map_location="cpu", weights_only=True)

    def pack(items):
        return comfy.utils.pack_latents([load(v) for v in items])[0]

    if m.get("cfg") not in (None, 1, 1.0) or m.get("denoiseMask") not in (None, False):
        raise ValueError("profile_not_cfg1_noinpaint")
    rows = []
    forwards = {int(v["call"]): v for v in m.get("forward", [])}
    if len(forwards) != 3:
        raise ValueError("exactly_three_captured_forwards_required")
    bundles = {}
    for call, descriptor in forwards.items():
        path = (root / descriptor["tensorPath"]).resolve()
        path.relative_to(root.resolve())
        if digest(path) != descriptor["sha256"]:
            raise ValueError("forward_bundle_custody")
        bundles[call] = torch.load(path, map_location="cpu", weights_only=True)
    first = None
    for i, row in enumerate(m.get("trace", [])):
        sx, sd = pack(row["stateCallback"]), pack(row["denoised"])
        sigma = torch.tensor(float(row["sigma"]), dtype=torch.float32)
        matching = []
        for call in forwards:
            b = bundles[call]
            if torch.equal(b["input"], sx) and torch.equal(
                b["timestep"].float().cpu(),
                torch.full_like(b["timestep"].float().cpu(), sigma.item()),
            ):
                matching.append(call)
                rows.append(
                    {
                        "step": i,
                        "call": call,
                        "status": "VALID" if torch.equal(b["prediction"], sd) else "MISMATCH",
                        "inputMatch": True,
                        "timestepMatch": True,
                        "predictionMatch": bool(torch.equal(b["prediction"], sd)),
                    }
                )
                first = first or b
        if not matching:
            rows.append(
                {
                    "step": i,
                    "status": "UNSELECTED_OR_MISMATCH",
                    "inputSha256": hashlib.sha256(sx.float().numpy().tobytes()).hexdigest(),
                }
            )
    derivable = []
    if first is not None:
        for i, row in enumerate(m["trace"]):
            derivable.append(
                {
                    "step": i,
                    "inputFromCallback": True,
                    "timestepFromSigma": float(row["sigma"]),
                    "conditioningSource": "first captured bundle; only trusted in closed baseline",
                    "requiresTransformerOptionsSampleSigmas": True,
                }
            )
    out = {
        "schema": "forge/inference-callback-binding/v1",
        "status": "MEASURED",
        "selected": rows,
        "summary": {
            "expected": 3,
            "matched": sum("call" in r for r in rows),
            "valid": sum(r.get("status") == "VALID" for r in rows),
        },
        "derivablePostCommitBundles": derivable,
        "limits": [
            "cfg1_noinpaint_only",
            "conditioning trust is external",
            "does_not_bind_initial_seed_or_final_video",
        ],
    }
    x.out.write_text(json.dumps(out, indent=2) + "\n")
    print(out["status"])


if __name__ == "__main__":
    main()
