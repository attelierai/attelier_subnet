#!/usr/bin/env python3
"""Check the actual CUDA driver before installing the frozen cu130 environments.

Only the Python standard library is needed, so this runs before a worker venv
exists. A PASS proves driver initialization, not Torch kernels or DINO behavior.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-gpu-name", default="NVIDIA GeForce RTX 5090")
    args = parser.parse_args()
    receipt = {"schema": "forge/evaluator-pilot-cuda-host/v1", "status": "INCOMPLETE"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        rows = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError("require exactly one visible GPU")
        name, uuid, driver = (value.strip() for value in rows[0])
        receipt.update(gpu_name=name, gpu_uuid=uuid, driver_version=driver)
        receipt["expected_gpu_name"] = args.expected_gpu_name
        if name != args.expected_gpu_name:
            raise ValueError("visible GPU differs from the explicitly reserved model")
        if int(driver.split(".")[0]) < 580:
            raise ValueError("cu130 requires NVIDIA driver branch580 or newer")
        cuda = ctypes.CDLL("libcuda.so.1")
        version, count = ctypes.c_int(), ctypes.c_int()
        for function, arguments in (
            (cuda.cuInit, (ctypes.c_uint(0),)),
            (cuda.cuDriverGetVersion, (ctypes.byref(version),)),
            (cuda.cuDeviceGetCount, (ctypes.byref(count),)),
        ):
            code = function(*arguments)
            if code != 0:
                raise ValueError(f"{function.__name__} failed with CUDA code{code}")
        receipt.update(cuda_driver_api=version.value, cuda_device_count=count.value)
        if version.value < 13000 or count.value != 1:
            raise ValueError("require initialized CUDA13+ driver and exactly one device")
        receipt["status"] = "PASS"
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
    payload = json.dumps(receipt, sort_keys=True, allow_nan=False)
    if args.out:
        with args.out.open("x") as stream:
            stream.write(payload + "\n")
    print(payload)
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
