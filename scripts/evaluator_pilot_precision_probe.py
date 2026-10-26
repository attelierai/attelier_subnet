#!/usr/bin/env python3
"""Diagnostic only: compare native and float32 cosine on identical CLIP features.

Run under the pinned metrics interpreter on the owned remote worker. This never
changes evaluator normalization, calls a judge, or emits a quality receipt.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

from forge.evaluation.pilot_media_index import validate_pilot_media_index
from forge.evaluation.pilot_worklist import pilot_worklist
from forge.evaluation.vbench import VBENCH_COMMIT, normalize_native_score
from forge.evaluation.vbench_runner import WEIGHT_PATHS, _pinned_checkout
from forge.protocol.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "media-index", "media-root", "vbench-config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.deadline_unix) or not time.time() < args.deadline_unix:
        raise ValueError("a future deadline and external process timeout are required")
    root, output = args.media_root.resolve(), args.out.resolve()
    for path in (root, output):
        path.relative_to(Path("/workspace/forge"))
    if output.exists():
        raise ValueError("preserve existing diagnostic output")
    work = pilot_worklist(json.loads(args.pilot.read_text()))
    index = json.loads(args.media_index.read_text())
    videos = validate_pilot_media_index(work, index)["videos"]
    if not any(
        p["kind"] == "freeze" and p["candidate_job_id"] == args.job_id for p in work["comparisons"]
    ):
        raise ValueError("this diagnostic requires a frozen freeze-control job")
    row = videos[args.job_id]
    video = root / row["path"]
    video.resolve().relative_to(root)
    if video.is_symlink() or sha256_file(video) != row["video_sha256"]:
        raise ValueError("video differs from frozen media")
    config = json.loads(args.vbench_config.read_text())
    checkout, cache = Path(config["checkout"]), Path(config["cache"])
    _pinned_checkout(checkout, VBENCH_COMMIT)
    weight = cache / WEIGHT_PATHS["background_consistency"]
    if sha256_file(weight) != config["weight_sha256"]["background_consistency"]:
        raise ValueError("CLIP weights differ from the pinned manifest")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(checkout))
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    module = importlib.import_module("vbench.background_consistency")
    if Path(module.__file__).resolve() != checkout.resolve() / "vbench/background_consistency.py":
        raise ValueError("unapproved metric source import")
    import clip
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("reserved CUDA worker required")
    model, preprocess = clip.load(str(weight), device="cuda")

    class IdenticalFeatures:
        features = None
        images = None
        use_float32 = False

        def encode_image(self, images):
            if self.features is None:
                self.images = images.clone()
                self.features = model.encode_image(images).detach()
            elif not torch.equal(images, self.images):
                raise ValueError("precision comparison changed input frames")
            return self.features.float() if self.use_float32 else self.features

    encoded = IdenticalFeatures()
    with torch.no_grad():
        native, _ = module.background_consistency(encoded, preprocess, [str(video)], "cuda", False)
        encoded.use_float32 = True
        fp32, _ = module.background_consistency(encoded, preprocess, [str(video)], "cuda", False)
    output.mkdir(parents=True, mode=0o700)
    np.save(output / "features.npy", encoded.features.cpu().numpy(), allow_pickle=False)
    checks = {}
    for name, value in (("native", native), ("float32", fp32)):
        try:
            normalize_native_score(value)
            checks[name] = "WITHIN_EXISTING_BOUND"
        except ValueError:
            checks[name] = "OUTSIDE_EXISTING_BOUND"
    receipt = {
        "schema": "forge/pilot-precision-diagnostic/v1",
        "qualification": "prohibited",
        "worklist_id": work["worklist_id"],
        "job_id": args.job_id,
        "video_sha256": row["video_sha256"],
        "weight_sha256": sha256_file(weight),
        "implementation_commit": VBENCH_COMMIT,
        "probe_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.vbench_config),
        "features_sha256": sha256_file(output / "features.npy"),
        "feature_dtype": str(encoded.features.dtype),
        "feature_shape": list(encoded.features.shape),
        "native": native,
        "float32": fp32,
        "checks": checks,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "same_encoded_features": True,
        "api_calls": 0,
        "prescore_observability_evidence": False,
    }
    (output / "diagnostic.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "DIAGNOSTIC_COMPLETE", "qualification": "prohibited"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
