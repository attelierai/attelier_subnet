"""Opt-in real local Docker boundary; uses an existing image, never pulls one."""

import importlib.util
import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest


def test_real_container_identity_and_substitution():
    image = os.environ.get("FORGE_TEST_LOCAL_IMAGE")
    if not image:
        pytest.skip("set FORGE_TEST_LOCAL_IMAGE to an already installed digest-pinned image")
    script = Path(__file__).parents[2] / "scripts/evaluator_pilot_container.py"
    spec = importlib.util.spec_from_file_location("pilot_container", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
    name = "forge-pilot-integration-" + uuid.uuid4().hex
    container_id = subprocess.check_output(
        [
            "docker",
            "run",
            "--pull=never",
            "--network=none",
            "-d",
            "--name",
            name,
            "--entrypoint",
            "sleep",
            image,
            "120",
        ],
        text=True,
    ).strip()
    try:
        receipt = module.verify(["docker"], name, image, inspected["Id"])
        assert receipt["container_id"] == container_id
        with pytest.raises(ValueError, match="does not match"):
            module.verify(["docker"], name, image, "sha256:" + "0" * 64)
        subprocess.run(
            ["docker", "stop", "--time", "1", container_id], check=True, capture_output=True
        )
        with pytest.raises(ValueError, match="does not match"):
            module.verify(["docker"], name, image, inspected["Id"])
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], check=True, capture_output=True)
