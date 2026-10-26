"""Controlled H3 replay command; inputs on stdin, native service credentials only.

Invoke only after every candidate output for the challenge has been captured.
The reference Comfy process must run with caching disabled. The trusted control
host retains each distinct challenge's reference; both miners may be compared
against that same independently generated result.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from forge.inference.chain import atomic_json
from forge.inference.reference import sha256
from forge.inference.service import Worker


def main() -> None:
    request = json.load(sys.stdin)
    config = json.loads(Path(os.environ["H3_REFERENCE_CONFIG_FILE"]).read_text())
    if config.get("cacheDisabled") is not True:
        raise ValueError("reference_compute_cache_must_be_disabled")
    for field in ("modelManifestHash", "workflowHash"):
        if request[field] != config[field]:
            raise ValueError("reference_profile_mismatch")
    parameters = request["parameters"]
    key = sha256(
        {
            "parameters": parameters,
            "modelManifestHash": config["modelManifestHash"],
            "workflowHash": config["workflowHash"],
            "campaign": config["campaign"],
        }
    )
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_file = directory / (key + ".json")
    target = directory / (key + ".mp4")
    state = (
        json.loads(state_file.read_text())
        if state_file.exists()
        else {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "attelier-reference:" + key)),
            "started": time.time(),
            "deadline": time.time() + 600,
            "status": "prepared",
            "parametersHash": sha256(parameters),
            "workerId": config["workerId"],
        }
    )
    if state["workerId"] != config["workerId"]:
        raise ValueError("reference_worker_changed")
    atomic_json(state_file, state)
    worker = Worker(config)
    while time.time() < state["deadline"] or state["status"] == "succeeded":
        if state["status"] == "succeeded":
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != state["sha256"]:
                    raise ValueError("reference_custody_changed")
            print(
                json.dumps(
                    {
                        "path": str(target),
                        "workerId": config["workerId"],
                        "sha256": state["sha256"],
                        "jobId": state["id"],
                    }
                )
            )
            return
        job = worker.call("/jobs/" + state["id"])
        if job is None:
            quote = state.get("quote")
            if not quote or quote["expiresAt"] <= time.time() * 1000:
                quote = worker.call("/quotes", parameters)
                if quote["amountMicros"] > config["ceilingMicros"]:
                    raise ValueError("reference_quote_exceeds_ceiling")
                state["quote"] = quote
            state["status"] = "submitting"
            atomic_json(state_file, state)
            # Persisted job ID and quote survive a lost acknowledgement. The
            # caller may reconcile this exact command, never another worker.
            worker.call("/jobs/" + state["id"], {"parameters": parameters, "quote": quote})
            state["status"] = "running"
        elif job["status"] == "succeeded":
            partial = target.with_suffix(".partial")
            if partial.exists():
                partial.unlink()
            advertised = worker.call("/jobs/" + state["id"] + "/result", destination=partial)
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if advertised != digest:
                raise ValueError("reference_transport_hash")
            partial.replace(target)
            state.update(status="succeeded", sha256=digest, completed=time.time())
        elif job["status"] in ("failed", "cancelled"):
            raise ValueError("reference_terminal_failure")
        atomic_json(state_file, state)
        if state["status"] != "succeeded":
            time.sleep(2)
    raise ValueError("reference_deadline")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Worker/control credentials and remote response bodies never leave this
        # process. The verifier records a reference infrastructure failure.
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
