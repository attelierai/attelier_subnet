#!/usr/bin/env python3
"""Inspect an owned Docker workload; this never pulls or starts a container."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def verify(docker: list[str], container: str, image: str, config_digest: str) -> dict:
    if not re.fullmatch(r"forge-pilot-[a-z0-9-]+", container):
        raise ValueError("container must have the scoped pilot name")
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("image must be pinned by registry digest")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest):
        raise ValueError("expected image configuration digest required")

    def inspect(kind: str, target: str) -> dict:
        result = subprocess.run(
            [*docker, kind, "inspect", target],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        rows = json.loads(result.stdout)
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("ambiguous Docker inspect response")
        return rows[0]

    runtime = inspect("container", container)
    loaded = inspect("image", runtime["Image"])
    manifest_digest = image.rsplit("@", 1)[1]
    if runtime["Image"] == manifest_digest:
        # Docker29's containerd store identifies images by manifest. Bind the
        # actual locally stored manifest/config blobs, not just a template label.
        blobs = Path(
            "/var/lib/docker/containerd/daemon/io.containerd.content.v1.content/blobs/sha256"
        )
        manifest_bytes = (blobs / manifest_digest.removeprefix("sha256:")).read_bytes()
        config_bytes = (blobs / config_digest.removeprefix("sha256:")).read_bytes()
        if (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest
            or json.loads(manifest_bytes)["config"]["digest"] != config_digest
            or "sha256:" + hashlib.sha256(config_bytes).hexdigest() != config_digest
        ):
            raise ValueError("stored manifest/config does not match frozen image")
    elif runtime["Image"] != config_digest:
        raise ValueError("running container does not match frozen image")
    if (
        runtime["Name"] != "/" + container
        or runtime["Config"]["Image"] != image
        or loaded["Id"] != runtime["Image"]
        or image not in loaded.get("RepoDigests", [])
        or runtime["State"].get("Running") is not True
    ):
        raise ValueError("running container does not match frozen image")
    return {
        "schema": "forge/evaluator-pilot-container/v1",
        "status": "VERIFIED",
        "container_id": runtime["Id"],
        "container_name": container,
        "image": image,
        "image_config_digest": config_digest,
        "started_at": runtime["State"]["StartedAt"],
        "limits": "Local Docker inspection; not hardware or hostile-host attestation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(
        ["docker", "--host", "unix:///run/forge-pilot-docker.sock"],
        args.container,
        args.image,
        args.config_digest,
    )
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
