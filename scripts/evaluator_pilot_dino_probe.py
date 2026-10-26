#!/usr/bin/env python3
"""Exercise the exact pinned DINO cache on the owned remote CUDA worker."""

import argparse
import json
import os
import time
from pathlib import Path

from forge.evaluation.vbench_runner import _pinned_checkout
from forge.protocol.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path("/workspace/forge")
    if Path.cwd().resolve() != root:
        raise ValueError("DINO probe only runs on the reserved remote worker")
    if args.out.exists():
        raise ValueError("DINO evidence must not be overwritten")
    host_probe = root / "cuda-host-probe.json"
    host = json.loads(host_probe.read_text())
    if (
        host.get("status") != "PASS"
        or not host.get("expected_gpu_name")
        or host.get("expected_gpu_name") != host.get("gpu_name")
        or host.get("cuda_device_count") != 1
        or host.get("cuda_driver_api", 0) < 13000
    ):
        raise ValueError("DINO requires the successful frozen GPU/driver preflight")
    cache = root / "metric-cache"
    expected = json.loads((root / "subnet/configs/vbench-weights.json").read_text())[
        "dino_model/dino_vitbase16_pretrain.pth"
    ]
    checkpoint = cache / "dino_model/dino_vitbase16_pretrain.pth"
    if sha256_file(checkpoint) != expected["sha256"]:
        raise ValueError("DINO checkpoint does not match the approved weights")
    source = cache / "dino_model/facebookresearch_dino_main"
    revision = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
    _pinned_checkout(source, revision)
    import sys

    sys.dont_write_bytecode = True
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != host["expected_gpu_name"]:
        raise ValueError("DINO CUDA device differs from the reserved preflight runtime")
    hub = cache / "torch_hub"
    target = hub / "checkpoints/dino_vitbase16_pretrain.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and not target.is_symlink():
        target.symlink_to(checkpoint)
    if not target.is_file() or sha256_file(target) != expected["sha256"]:
        raise ValueError("Torch Hub cache is not the verified DINO checkpoint")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    torch.hub.set_dir(str(hub))
    started = time.monotonic()
    model = (
        torch.hub.load(str(source), "dino_vitb16", source="local", pretrained=True)
        .eval()
        .to("cuda")
    )
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 224, 224), device="cuda"))
    torch.cuda.synchronize()
    if tuple(output.shape) != (1, 768) or not torch.isfinite(output).all().item():
        raise ValueError("DINO CUDA cache probe returned malformed features")
    receipt = {
        "schema": "forge/evaluator-pilot-dino-cache-probe/v1",
        "status": "PASS",
        "checkpoint_sha256": sha256_file(target),
        "dino_revision": revision,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_host_probe_sha256": sha256_file(host_probe),
        "feature_shape": list(output.shape),
        "seconds": time.monotonic() - started,
        "qualification": "prohibited",
    }
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "qualification": "prohibited"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
