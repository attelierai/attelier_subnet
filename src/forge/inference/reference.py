"""Reference-replay evidence for the fixed H3 pilot profile.

This is deliberately a full output replay, not a claim of a cryptographic proof
of remote GPU execution.  The reference command is owned by the validator and
must emit a JSON object naming a freshly rendered MP4.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from forge.inference.media import canonical_decode

SCHEMA = "forge-h3-reference-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode()


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def verify_receipt(receipt: dict[str, Any], expected: dict[str, str], public_key: str) -> None:
    """Verify the worker's Ed25519 receipt with the system OpenSSL provider.

    `public_key` is base64 of the 32 raw Ed25519 bytes.  It is a pinned public
    value, never a wallet key.  The subprocess keeps this project dependency-free.
    """
    required = {
        "schema",
        "jobId",
        "workerId",
        "nonce",
        "parametersHash",
        "modelManifestHash",
        "workflowHash",
        "resultSha256",
        "signature",
    }
    if set(receipt) != required or receipt.get("schema") != SCHEMA:
        raise ValueError("receipt_schema")
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("receipt_binding")
    if any(
        not isinstance(receipt[k], str) or len(receipt[k]) != 64
        for k in ("parametersHash", "modelManifestHash", "workflowHash", "resultSha256")
    ):
        raise ValueError("receipt_hash")
    try:
        raw_key = base64.b64decode(public_key, validate=True)
        signature = base64.b64decode(receipt["signature"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("receipt_encoding") from exc
    if len(raw_key) != 32 or len(signature) != 64:
        raise ValueError("receipt_key_or_signature")
    signed = {k: v for k, v in receipt.items() if k != "signature"}
    # RFC 8410 SubjectPublicKeyInfo wrapper for raw Ed25519 public key.
    pem_der = bytes.fromhex("302a300506032b6570032100") + raw_key
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "pub.der").write_bytes(pem_der)
        (root / "message").write_bytes(canonical(signed))
        (root / "signature").write_bytes(signature)
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-keyform",
                "DER",
                "-inkey",
                str(root / "pub.der"),
                "-rawin",
                "-in",
                str(root / "message"),
                "-sigfile",
                str(root / "signature"),
            ],
            capture_output=True,
            timeout=15,
        )
    if result.returncode:
        raise ValueError("receipt_signature")


def replay(directory: Path, reference: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Run the configured full reference renderer and canonically decode its media."""
    command = reference.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError("reference_command")
    started = __import__("time").monotonic()
    completed = subprocess.run(
        command,
        input=canonical(request),
        capture_output=True,
        timeout=int(reference.get("timeoutSeconds", 600)),
        cwd=directory,
    )
    elapsed = __import__("time").monotonic() - started
    if completed.returncode:
        raise ValueError("reference_failed")
    try:
        value = json.loads(completed.stdout)
        path = Path(value["path"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("reference_protocol") from exc
    if not path.is_absolute():
        path = directory / path
    if not path.is_file():
        raise ValueError("reference_missing_result")
    evidence = canonical_decode(path)
    evidence["generationSeconds"] = elapsed
    evidence["path"] = str(path)
    return evidence
