"""Bounded custody and request binding for the H3 trace serving protocol.

This module verifies declarations, signatures and bytes. Only the independent
GPU runtime can decide whether those bytes represent a valid execution.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import stat
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "forge-h3-request/v1"
RECEIPT_SCHEMA = "forge-h3-trace/v1"
PROFILE_SCHEMA = "forge/h3-profile/v1"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_SAFE_INTEGER = 2**53 - 1
FIXED_PARAMETERS = {
    "modality": "text-to-video",
    "duration": 5,
    "resolution": "480p",
    "aspectRatio": "16:9",
    "audio": True,
    "profile": "base",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}\Z")


class TraceProtocolError(ValueError):
    def __init__(self, reason: str, category: str = "invalid"):
        super().__init__(reason)
        self.reason = reason
        self.category = category


def _require(condition: bool, reason: str, category: str = "invalid") -> None:
    if not condition:
        raise TraceProtocolError(reason, category)


def canonical(value: Any) -> bytes:
    """Cross-language canonical JSON for *protocol* data, not tensor metadata.

    Keys are ASCII and numbers are safe integers. Numeric tolerances in profiles are
    decimal strings, avoiding Python/JavaScript float serialization differences.
    """

    def check(item: Any, depth: int = 0) -> None:
        _require(depth <= 24, "json_too_deep")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int):
            _require(abs(item) <= MAX_SAFE_INTEGER, "json_integer_range")
        elif isinstance(item, str):
            try:
                item.encode("utf8")
            except UnicodeEncodeError as exc:
                raise TraceProtocolError("json_unicode") from exc
        elif isinstance(item, list):
            _require(len(item) <= 4096, "json_array_size")
            for child in item:
                check(child, depth + 1)
        elif isinstance(item, dict):
            _require(len(item) <= 1024, "json_object_size")
            for key, child in item.items():
                _require(isinstance(key, str) and key.isascii() and not key.isdecimal(), "json_key")
                check(child, depth + 1)
        else:
            raise TraceProtocolError("json_type_or_float")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    try:
        _require(path.stat().st_size <= maximum, "json_too_large")
        with path.open("rb") as stream:
            data = stream.read(maximum + 1)
        _require(len(data) <= maximum, "json_too_large")
        result = json.loads(
            data,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(TraceProtocolError("json_nonfinite")),
        )
        _require(isinstance(result, dict), "json_object_required")
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TraceProtocolError("json_unavailable_or_invalid") from exc


def effective_seed(job_id: str, nonce: str, requested_seed: int, profile_hash: str) -> int:
    digest = hashlib.sha256(
        canonical(
            {
                "domain": "forge-h3-seed/v1",
                "jobId": job_id,
                "nonce": nonce,
                "requestedSeed": requested_seed,
                "profileHash": profile_hash,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") >> 11


def validate_request(request: dict[str, Any]) -> None:
    _require(
        set(request)
        == {"schema", "jobId", "workerId", "nonce", "requestedSeed", "profile", "parameters"},
        "request_fields",
    )
    _require(request["schema"] == REQUEST_SCHEMA, "request_schema")
    for key in ("jobId", "workerId"):
        _require(
            isinstance(request[key], str) and bool(_ID.fullmatch(request[key])), "request_identity"
        )
    _require(
        isinstance(request["nonce"], str) and bool(_HASH.fullmatch(request["nonce"])),
        "request_nonce",
    )
    seed = request["requestedSeed"]
    _require(type(seed) is int and 0 <= seed <= 2**32 - 1, "requested_seed")
    identity = request["profile"]
    _require(isinstance(identity, dict) and set(identity) == {"id", "hash"}, "profile_identity")
    _require(isinstance(identity["id"], str) and bool(_ID.fullmatch(identity["id"])), "profile_id")
    _require(
        isinstance(identity["hash"], str) and bool(_HASH.fullmatch(identity["hash"])),
        "profile_hash",
    )
    params = request["parameters"]
    _require(
        isinstance(params, dict) and set(params) == {*FIXED_PARAMETERS, "prompt", "seed"},
        "parameters_fields",
    )
    _require(
        all(type(params[k]) is type(v) and params[k] == v for k, v in FIXED_PARAMETERS.items()),
        "unsupported_parameters",
    )
    prompt = params["prompt"]
    _require(
        isinstance(prompt, str) and bool(prompt.strip()) and len(prompt.encode("utf8")) <= 32000,
        "prompt_invalid",
    )
    _require(
        type(params["seed"]) is int
        and params["seed"]
        == effective_seed(request["jobId"], request["nonce"], seed, identity["hash"]),
        "effective_seed_binding",
    )
    canonical(request)


def load_profile(
    profile_root: Path,
    identity: dict[str, str],
    *,
    now: float | None = None,
    allow_unqualified: bool = False,
) -> dict[str, Any]:
    return load_profile_envelope(
        profile_root, identity, now=now, allow_unqualified=allow_unqualified
    )["profile"]


def load_profile_envelope(
    profile_root: Path,
    identity: dict[str, str],
    *,
    now: float | None = None,
    allow_unqualified: bool = False,
) -> dict[str, Any]:
    """Read an operator-owned profile. Miner evidence cannot qualify a profile."""
    _require(
        isinstance(identity.get("id"), str) and bool(_ID.fullmatch(identity["id"])), "profile_id"
    )
    path = profile_root / (identity["id"] + ".json")
    try:
        _require(not path.is_symlink(), "profile_symlink", "inconclusive")
        envelope = read_json(path)
        profile = envelope["profile"]
        _require(
            envelope.get("schema") == PROFILE_SCHEMA and isinstance(profile, dict),
            "profile_schema",
            "inconclusive",
        )
        _require(
            profile["id"] == identity["id"] and sha256(profile) == identity["hash"],
            "profile_changed",
            "inconclusive",
        )
        for field in ("modelManifestHash", "workflowHash", "runtimeHash", "rendererHash"):
            _require(
                isinstance(profile[field], str) and bool(_HASH.fullmatch(profile[field])),
                "profile_fingerprint",
                "inconclusive",
            )
        _require(
            type(profile["steps"]) is int and profile["steps"] == 20,
            "profile_steps",
            "inconclusive",
        )
        _require(
            canonical(profile["parameters"]) == canonical(FIXED_PARAMETERS),
            "profile_parameters",
            "inconclusive",
        )
        threshold = profile["numerics"]["maxRelativeRms"]
        _require(
            isinstance(threshold, str)
            and math.isfinite(float(threshold))
            and 0 <= float(threshold) <= 0.01,
            "profile_tolerance",
            "inconclusive",
        )
        limits = profile["limits"]
        for key, ceiling in (
            ("maxChunks", 512),
            ("maxChunkBytes", 2 * 1024**3),
            ("maxBundleBytes", 20 * 1024**3),
        ):
            _require(
                type(limits[key]) is int and 0 < limits[key] <= ceiling,
                "profile_limits",
                "inconclusive",
            )
        qualification = envelope["qualification"]
        _require(isinstance(qualification, dict), "qualification_invalid", "inconclusive")
        if not allow_unqualified:
            _require(
                qualification.get("status") == "qualified", "profile_unqualified", "inconclusive"
            )
            _require(
                isinstance(qualification.get("evidenceSha256"), str)
                and bool(_HASH.fullmatch(qualification["evidenceSha256"])),
                "qualification_evidence_missing",
                "inconclusive",
            )
            _require(
                type(qualification.get("expiresAt")) is int
                # Unix seconds through year 9999; reject accidental milliseconds.
                and qualification["expiresAt"] <= 253_402_300_799
                and qualification["expiresAt"] > (time.time() if now is None else now),
                "qualification_expired",
                "inconclusive",
            )
        return envelope
    except TraceProtocolError as exc:
        raise TraceProtocolError(exc.reason, "inconclusive") from exc
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise TraceProtocolError("profile_unavailable_or_invalid", "inconclusive") from exc


def qualified_lane(
    profile: dict[str, Any], qualification: dict[str, Any], hotkey: str, worker_id: str
) -> dict[str, Any]:
    """Resolve an exact operator-qualified lane; declarations never confer trust."""
    _require(
        isinstance(hotkey, str) and bool(hotkey) and isinstance(worker_id, str) and bool(worker_id),
        "trace_lane_identity",
        "inconclusive",
    )
    lane = {
        "hotkey": hotkey,
        "workerId": worker_id,
        "profileId": profile["id"],
        "profileHash": sha256(profile),
        "rendererHash": profile["rendererHash"],
    }
    lanes = qualification.get("lanes")
    _require(isinstance(lanes, list) and lane in lanes, "trace_lane_unqualified", "inconclusive")
    return lane


def _verify_signature(receipt: dict[str, Any], public_key: str) -> None:
    try:
        key = base64.b64decode(public_key, validate=True)
        signature = base64.b64decode(receipt["signature"], validate=True)
    except (ValueError, TypeError) as exc:
        raise TraceProtocolError("receipt_signature_encoding") from exc
    _require(len(key) == 32 and len(signature) == 64, "receipt_signature_encoding")
    message = canonical({k: v for k, v in receipt.items() if k != "signature"})
    with tempfile.TemporaryDirectory(prefix="h3-signature-") as temporary:
        root = Path(temporary)
        (root / "public.der").write_bytes(bytes.fromhex("302a300506032b6570032100") + key)
        (root / "signature").write_bytes(signature)
        (root / "message").write_bytes(message)
        try:
            result = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    str(root / "public.der"),
                    "-rawin",
                    "-in",
                    str(root / "message"),
                    "-sigfile",
                    str(root / "signature"),
                ],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TraceProtocolError("signature_verifier_unavailable", "inconclusive") from exc
    _require(result.returncode == 0, "receipt_signature")


def _checked_path(root: Path, name: str) -> Path:
    parts = _chunk_name_parts(name)
    target = root
    for part in parts:
        target = target / part
        _require(not target.is_symlink(), "chunk_symlink")
    _require(stat.S_ISREG(target.stat().st_mode), "chunk_not_regular")
    return target


def _chunk_name_parts(name: Any) -> list[str]:
    """Validate the portable relative name before any evidence path is made."""
    _require(isinstance(name, str) and bool(_NAME.fullmatch(name)), "chunk_name")
    parts = name.split("/")
    _require(
        all(part and part not in (".", "..") and not part.startswith(".") for part in parts),
        "chunk_name",
    )
    return parts


def check_tensor_archive(path: Path, maximum: int) -> None:
    """Bound a PyTorch zip before the runtime's weights_only CPU load.

    No extraction occurs. Large tensor allocations still run in the isolated,
    memory-limited worker, never in the gateway process.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            _require(0 < len(entries) <= 4096, "tensor_archive_entries")
            _require(
                sum(item.file_size for item in entries) <= maximum, "tensor_archive_expanded_size"
            )
            _require(
                len({item.filename for item in entries}) == len(entries), "tensor_archive_duplicate"
            )
            for item in entries:
                _require(
                    not item.flag_bits & 1 and item.compress_type == zipfile.ZIP_STORED,
                    "tensor_archive_compression",
                )
                _require(
                    not item.filename.startswith("/") and ".." not in item.filename.split("/"),
                    "tensor_archive_name",
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise TraceProtocolError("tensor_archive_invalid") from exc


def validate_receipt_metadata(
    request: dict[str, Any],
    receipt: dict[str, Any],
    profile: dict[str, Any],
    worker_public_key: str,
) -> dict[str, Any]:
    """Validate all signed receipt declarations before fetching any evidence bytes.

    The returned descriptors are still untrusted content declarations.  Callers
    must enforce their exact ``bytes`` limits while transporting and subsequently
    use :func:`verify_bundle` to bind stored bytes, capture metadata and tensors.
    """
    validate_request(request)
    expected_keys = {
        "schema",
        "jobId",
        "workerId",
        "nonce",
        "requestHash",
        "parametersHash",
        "profile",
        "modelManifestHash",
        "workflowHash",
        "steps",
        "trace",
        "finalLatent",
        "signature",
    }
    _require(
        isinstance(receipt, dict)
        and set(receipt) == expected_keys
        and receipt["schema"] == RECEIPT_SCHEMA,
        "receipt_fields",
    )
    _require(len(canonical(receipt)) <= MAX_JSON_BYTES, "receipt_too_large")
    for key in ("jobId", "workerId", "nonce", "profile"):
        _require(receipt[key] == request[key], "receipt_request_binding")
    _require(
        receipt["requestHash"] == sha256(request)
        and receipt["parametersHash"] == sha256(request["parameters"]),
        "receipt_parameters_binding",
    )
    _require(sha256(profile) == request["profile"]["hash"], "profile_binding", "inconclusive")
    _require(
        all(receipt[key] == profile[key] for key in ("modelManifestHash", "workflowHash", "steps")),
        "receipt_profile_binding",
    )
    _verify_signature(receipt, worker_public_key)

    trace = receipt["trace"]
    _require(
        isinstance(trace, dict)
        and set(trace) == {"algorithm", "chunks", "root"}
        and trace["algorithm"] == "sha256-manifest-v1",
        "trace_fields",
    )
    chunks = trace["chunks"]
    limits = profile["limits"]
    _require(
        isinstance(chunks, list) and 3 <= len(chunks) <= limits["maxChunks"], "trace_chunk_count"
    )
    _require(trace["root"] == sha256(chunks), "trace_root")
    names, total = set(), 0
    for index, descriptor in enumerate(chunks):
        _require(
            isinstance(descriptor, dict)
            and set(descriptor) == {"index", "name", "sha256", "bytes"},
            "chunk_fields",
        )
        _require(type(descriptor["index"]) is int and descriptor["index"] == index, "chunk_order")
        name, size = descriptor["name"], descriptor["bytes"]
        _chunk_name_parts(name)
        _require(name not in names, "chunk_duplicate")
        _require(type(size) is int and 0 < size <= limits["maxChunkBytes"], "chunk_size")
        _require(
            isinstance(descriptor["sha256"], str) and bool(_HASH.fullmatch(descriptor["sha256"])),
            "chunk_hash",
        )
        total += size
        _require(total <= limits["maxBundleBytes"], "bundle_size")
        if name == "capture.json":
            _require(size <= MAX_JSON_BYTES, "capture_too_large")
        else:
            _require(
                name.startswith(("tensors/", "forward/")) and name.endswith(".pt"),
                "unsupported_chunk",
            )
        names.add(name)
    _require("capture.json" in names, "capture_missing")

    final = receipt["finalLatent"]
    _require(isinstance(final, list) and len(final) == 2, "final_latent_components")
    final_indices = set()
    for component, descriptor in zip(("video", "audio"), final, strict=True):
        _require(
            isinstance(descriptor, dict)
            and set(descriptor) == {"component", "chunkIndex"}
            and descriptor["component"] == component,
            "final_latent_components",
        )
        index = descriptor["chunkIndex"]
        _require(
            type(index) is int
            and 0 <= index < len(chunks)
            and chunks[index]["name"].startswith("tensors/"),
            "final_latent_chunk",
        )
        final_indices.add(index)
    _require(len(final_indices) == 2, "final_latent_alias")
    return {"chunks": chunks, "totalBytes": total}


def verify_bundle(
    request: dict[str, Any],
    receipt: dict[str, Any],
    evidence_root: Path,
    profile: dict[str, Any],
    worker_public_key: str,
) -> dict[str, Any]:
    """Validate the signed bundle against trusted request/profile and stored bytes.

    The gateway must download into a private, single-writer job directory and seal
    it before calling this. A successful return establishes custody, not execution.
    """
    metadata = validate_receipt_metadata(request, receipt, profile, worker_public_key)
    chunks = metadata["chunks"]
    limits = profile["limits"]
    paths, names = [], set()
    for descriptor in chunks:
        name, size = descriptor["name"], descriptor["bytes"]
        names.add(name)
        try:
            path = _checked_path(evidence_root, name)
            _require(
                path.stat().st_size == size and file_sha256(path) == descriptor["sha256"],
                "chunk_custody",
            )
            if name == "capture.json":
                _require(size <= MAX_JSON_BYTES, "capture_too_large")
            else:
                _require(
                    name.startswith(("tensors/", "forward/")) and name.endswith(".pt"),
                    "unsupported_chunk",
                )
                check_tensor_archive(path, limits["maxChunkBytes"])
        except OSError as exc:
            raise TraceProtocolError("evidence_unavailable", "inconclusive") from exc
        paths.append(path)
    _require("capture.json" in names, "capture_missing")
    capture_path = paths[next(i for i, d in enumerate(chunks) if d["name"] == "capture.json")]
    capture = read_json(capture_path)
    descriptors = {item["name"]: item for item in chunks}

    def bound_references(value: Any, depth: int = 0) -> None:
        _require(depth <= 24, "capture_too_deep")
        if isinstance(value, dict):
            if "tensorPath" in value:
                name = value["tensorPath"]
                _require(isinstance(name, str), "capture_uncommitted_tensor")
                descriptor = descriptors.get(name)
                _require(
                    descriptor is not None and value.get("sha256") == descriptor["sha256"],
                    "capture_uncommitted_tensor",
                )
                _require(
                    "bytes" not in value or value["bytes"] == descriptor["bytes"],
                    "capture_tensor_size",
                )
            for child in value.values():
                bound_references(child, depth + 1)
        elif isinstance(value, list):
            _require(len(value) <= 4096, "capture_array_size")
            for child in value:
                bound_references(child, depth + 1)

    bound_references(capture)
    final = receipt["finalLatent"]
    _require(isinstance(final, list) and len(final) == 2, "final_latent_components")
    bound = {}
    for component, descriptor in zip(("video", "audio"), final, strict=True):
        _require(
            isinstance(descriptor, dict)
            and set(descriptor) == {"component", "chunkIndex"}
            and descriptor["component"] == component,
            "final_latent_components",
        )
        index = descriptor["chunkIndex"]
        _require(
            type(index) is int
            and 0 <= index < len(paths)
            and chunks[index]["name"].startswith("tensors/"),
            "final_latent_chunk",
        )
        bound[component] = paths[index]
    _require(bound["video"] != bound["audio"], "final_latent_alias")
    capture_final = capture.get("finalLatent")
    _require(isinstance(capture_final, list) and len(capture_final) == 2, "capture_final_latent")
    for component, descriptor in zip(("video", "audio"), capture_final, strict=True):
        _require(
            isinstance(descriptor, dict)
            and descriptor.get("component") == component
            and evidence_root / str(descriptor.get("tensorPath")) == bound[component],
            "capture_final_binding",
        )
    return {
        "capture": capture,
        "capturePath": capture_path,
        "chunks": paths,
        "finalLatent": bound,
        "receiptHash": sha256(receipt),
    }
