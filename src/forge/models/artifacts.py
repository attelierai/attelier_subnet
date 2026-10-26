"""Fetch anonymous immutable model artifacts and verify their bytes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from forge.models.artifact_manifest import (
    HF_BASE_URL,
    HF_DOWNLOAD_DEADLINE_SECONDS,
    MAX_HF_READ_TIMEOUT_SECONDS,
    MAX_INLINE_HASH_BYTES,
    MAX_METADATA_BYTES,
    ArtifactError,
    ArtifactInspection,
    ArtifactReference,
    ArtifactUnavailable,
    ManifestFile,
    ModelManifest,
    Role,
    _safe_path,
)


def _json_get(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "forge-subnet/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - immutable HTTPS URL
            payload = response.read(MAX_METADATA_BYTES + 1)
    except HTTPError as exc:
        raise ArtifactUnavailable(f"registry rejected public artifact ({exc.code})") from exc
    except URLError as exc:
        raise ArtifactUnavailable(f"registry is unavailable: {exc.reason}") from exc
    if len(payload) > MAX_METADATA_BYTES:
        raise ArtifactUnavailable("registry metadata exceeds size limit")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ArtifactUnavailable("registry returned invalid JSON metadata") from exc
    if not isinstance(decoded, dict):
        raise ArtifactUnavailable("registry metadata must be a JSON object")
    return decoded


def _hf_metadata(
    reference: ArtifactReference, *, base_url: str, timeout_seconds: float
) -> dict[str, Any]:
    root = base_url.rstrip("/")
    repo = quote(reference.repo_id, safe="/")
    url = f"{root}/api/models/{repo}/revision/{reference.revision}?blobs=true"
    metadata = _json_get(url, timeout_seconds=timeout_seconds)
    if metadata.get("private") or metadata.get("gated") or metadata.get("disabled"):
        raise ArtifactUnavailable("artifact must be public, ungated, and enabled")
    if metadata.get("sha") != reference.revision:
        raise ArtifactUnavailable("registry did not resolve the requested immutable revision")
    if not isinstance(metadata.get("siblings"), list):
        raise ArtifactUnavailable("registry did not return a file list")
    return metadata


def _hf_inspect(
    reference: ArtifactReference, *, base_url: str, timeout_seconds: float
) -> ArtifactInspection:
    metadata = _hf_metadata(reference, base_url=base_url, timeout_seconds=timeout_seconds)
    siblings = metadata["siblings"]
    files: list[str] = []
    for item in siblings:
        if not isinstance(item, Mapping) or not isinstance(item.get("rfilename"), str):
            raise ArtifactUnavailable("registry returned an invalid file entry")
        files.append(item["rfilename"])
    card_data = metadata.get("cardData")
    license_value = card_data.get("license") if isinstance(card_data, Mapping) else None
    return ArtifactInspection(
        reference, license_value if isinstance(license_value, str) else None, tuple(files)
    )


def _hippius_inspect(reference: ArtifactReference, *, endpoint: str | None) -> ArtifactInspection:
    result = _hippius_bridge(
        "inspect", reference, endpoint=endpoint, payload={}, timeout_seconds=20.0
    )
    names = result.get("files")
    license_value = result.get("license")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ArtifactUnavailable("Hippius bridge returned an invalid file list")
    return ArtifactInspection(
        reference, license_value if isinstance(license_value, str) else None, tuple(names)
    )


def _hippius_bridge(
    operation: str,
    reference: ArtifactReference,
    *,
    endpoint: str | None,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    interpreter = os.environ.get("FORGE_HIPPIUS_PYTHON")
    if not interpreter:
        raise ArtifactUnavailable(
            "Hippius requires isolated hippius-hub==0.6.0; set FORGE_HIPPIUS_PYTHON"
        )
    executable = Path(interpreter)
    if not executable.is_absolute() or not executable.is_file():
        raise ArtifactUnavailable(
            "FORGE_HIPPIUS_PYTHON must be an absolute trusted interpreter path"
        )
    request = {
        "operation": operation,
        "repo_id": reference.repo_id,
        "revision": reference.revision,
        "endpoint": endpoint,
        **dict(payload),
    }
    environment = dict(os.environ)
    environment.pop("FORGE_HIPPIUS_PYTHON", None)
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            [str(executable), "-m", "forge.models.hippius_bridge"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
        response = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable("isolated Hippius bridge failed") from exc
    if completed.returncode or not isinstance(response, Mapping) or not response.get("ok"):
        raise ArtifactUnavailable("isolated Hippius bridge rejected the public artifact")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ArtifactUnavailable("isolated Hippius bridge returned invalid JSON")
    return result


def inspect_public_artifact(
    reference: ArtifactReference,
    *,
    hf_base_url: str = HF_BASE_URL,
    hippius_endpoint: str | None = None,
    timeout_seconds: float = 20.0,
) -> ArtifactInspection:
    """Resolve a public immutable artifact without credentials or model downloads."""

    if timeout_seconds <= 0:
        raise ArtifactError("timeout_seconds must be positive")
    if reference.backend == "hf":
        return _hf_inspect(reference, base_url=hf_base_url, timeout_seconds=timeout_seconds)
    return _hippius_inspect(reference, endpoint=hippius_endpoint)


def build_manifest(
    reference: ArtifactReference,
    *,
    role: Role,
    files: Sequence[ManifestFile],
    license: str,
    generation_config: Mapping[str, Any],
    base_reference: ArtifactReference | None = None,
    inspection: ArtifactInspection | None = None,
    hf_base_url: str = HF_BASE_URL,
    hippius_endpoint: str | None = None,
) -> ModelManifest:
    """Validate a declared manifest against a public immutable registry listing."""

    resolved = inspection or inspect_public_artifact(
        reference, hf_base_url=hf_base_url, hippius_endpoint=hippius_endpoint
    )
    if resolved.reference != reference:
        raise ArtifactError("inspection does not belong to this artifact reference")
    normalized_license = license.strip().lower()
    if resolved.license is None:
        raise ArtifactUnavailable("public artifact must expose a verifiable license")
    if resolved.license.strip().lower() != normalized_license:
        raise ArtifactError("manifest license must match the public registry metadata")
    listed = set(resolved.files)
    missing = [item.path for item in files if item.path not in listed]
    if missing:
        raise ArtifactError(f"manifest files are absent from public artifact: {missing}")
    return ModelManifest(
        reference, role, tuple(files), normalized_license, generation_config, base_reference
    )


def _hf_resolve_url(reference: ArtifactReference, path: str, *, hf_base_url: str) -> str:
    return (
        f"{hf_base_url.rstrip('/')}/{quote(reference.repo_id, safe='/')}/resolve/"
        f"{reference.revision}/{quote(path, safe='/')}"
    )


def _small_hf_file_hash(
    reference: ArtifactReference,
    path: str,
    expected_bytes: int,
    *,
    hf_base_url: str,
    timeout_seconds: float,
) -> str:
    if expected_bytes > MAX_INLINE_HASH_BYTES:
        raise ArtifactError(f"small config file exceeds {MAX_INLINE_HASH_BYTES} bytes: {path}")
    request = Request(
        _hf_resolve_url(reference, path, hf_base_url=hf_base_url),
        headers={"User-Agent": "forge-subnet/0.1"},
    )
    digest = hashlib.sha256()
    received = 0
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > expected_bytes:
                    raise ArtifactError(f"config file exceeds declared size: {path}")
                digest.update(chunk)
    except HTTPError as exc:
        raise ArtifactUnavailable(f"config file is not publicly readable ({exc.code})") from exc
    if received != expected_bytes:
        raise ArtifactError(f"config file size mismatch: {path}")
    return digest.hexdigest()


def _runtime_path(role: Role, path: str) -> bool:
    try:
        safe = _safe_path(path)
    except ArtifactError:
        return False
    # V2 used this top-level declaration to describe a miner-provided
    # manifest.  V3 derives its manifest from registry metadata, and neither
    # runtime loads this declaration nor should its optional presence change
    # the candidate byte set.
    if safe == PurePosixPath("forge-manifest.json"):
        return False
    suffix = safe.suffix.lower()
    if role == "pipeline":
        return suffix in {".json", ".safetensors"} or (
            safe.parts[0] == "tokenizer" and suffix == ".model"
        )
    if role == "full_transformer":
        return safe.parts[0] == "transformer" and suffix in {".json", ".safetensors"}
    return suffix in {".json", ".safetensors"} and len(safe.parts) <= 2


def _manifest_file_from_hf_sibling(
    reference: ArtifactReference,
    sibling: Mapping[str, Any],
    *,
    hf_base_url: str,
    timeout_seconds: float,
) -> ManifestFile:
    path = sibling.get("rfilename")
    size = sibling.get("size")
    if not isinstance(path, str) or not isinstance(size, int) or size <= 0:
        raise ArtifactUnavailable("registry returned file metadata without a positive size")
    lfs = sibling.get("lfs")
    if isinstance(lfs, Mapping):
        checksum = lfs.get("sha256")
        lfs_size = lfs.get("size")
        if not isinstance(checksum, str) or lfs_size != size:
            raise ArtifactUnavailable(f"registry returned invalid LFS metadata for {path}")
        return ManifestFile(path, checksum, size)
    if PurePosixPath(path).suffix.lower() not in {".json", ".txt"}:
        raise ArtifactUnavailable(f"runtime file has no immutable SHA-256 metadata: {path}")
    checksum = _small_hf_file_hash(
        reference, path, size, hf_base_url=hf_base_url, timeout_seconds=timeout_seconds
    )
    return ManifestFile(path, checksum, size)


def snapshot_manifest(
    reference: ArtifactReference,
    *,
    role: Role,
    generation_config: Mapping[str, Any],
    base_reference: ArtifactReference | None = None,
    hf_base_url: str = HF_BASE_URL,
    timeout_seconds: float = 20.0,
) -> ModelManifest:
    """Build a pinned HF runtime manifest without downloading model weights.

    LFS SHA-256 metadata supplies large-file identities.  Only small JSON/TXT
    runtime configuration files lacking LFS metadata are fetched to calculate
    their SHA-256 hashes.
    """

    if reference.backend != "hf":
        raise ArtifactError("snapshot_manifest currently supports Hugging Face artifacts only")
    if timeout_seconds <= 0:
        raise ArtifactError("timeout_seconds must be positive")
    metadata = _hf_metadata(reference, base_url=hf_base_url, timeout_seconds=timeout_seconds)
    card_data = metadata.get("cardData")
    license_value = card_data.get("license") if isinstance(card_data, Mapping) else None
    if not isinstance(license_value, str) or not license_value.strip():
        raise ArtifactUnavailable("public registry metadata must declare a license")
    selected = [
        item
        for item in metadata["siblings"]
        if isinstance(item, Mapping)
        and isinstance(item.get("rfilename"), str)
        and _runtime_path(role, item["rfilename"])
    ]
    if not selected:
        raise ArtifactUnavailable(f"artifact has no runtime files for role {role}")
    files = tuple(
        _manifest_file_from_hf_sibling(
            reference, item, hf_base_url=hf_base_url, timeout_seconds=timeout_seconds
        )
        for item in selected
    )
    return ModelManifest(
        reference, role, files, license_value.lower(), generation_config, base_reference
    )


def _download_hf_file(
    reference: ArtifactReference,
    file: ManifestFile,
    destination: Path,
    *,
    hf_base_url: str,
    timeout_seconds: float,
    deadline_seconds: float,
) -> None:
    deadline = time.monotonic() + deadline_seconds
    url = _hf_resolve_url(reference, file.path, hf_base_url=hf_base_url)
    request = Request(url, headers={"User-Agent": "forge-subnet/0.1"})
    digest = hashlib.sha256()
    received = 0
    fd, temporary = tempfile.mkstemp(prefix=".forge-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            response = None
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ArtifactUnavailable("artifact download exceeded its deadline")
                response = urlopen(
                    request, timeout=min(timeout_seconds, MAX_HF_READ_TIMEOUT_SECONDS, remaining)
                )  # noqa: S310
            except HTTPError as exc:
                raise ArtifactUnavailable(
                    f"artifact file is not publicly readable ({exc.code})"
                ) from exc
            except TimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise ArtifactUnavailable("artifact download exceeded its deadline") from exc
                raise ArtifactUnavailable("artifact download timed out") from exc
            except (URLError, HTTPException, ConnectionError, OSError) as exc:
                raise ArtifactUnavailable("artifact download transport failed") from exc
            try:
                length = response.headers.get("Content-Length")
                if length and int(length) != file.bytes:
                    raise ArtifactError(f"unexpected content length for {file.path}")
                while True:
                    try:
                        chunk = response.read1(1024 * 1024)
                    except TimeoutError as exc:
                        if time.monotonic() >= deadline:
                            raise ArtifactUnavailable(
                                "artifact download exceeded its deadline"
                            ) from exc
                        raise ArtifactUnavailable("artifact download timed out") from exc
                    except (URLError, HTTPException, ConnectionError, OSError) as exc:
                        raise ArtifactUnavailable("artifact download transport failed") from exc
                    if not chunk:
                        break
                    if time.monotonic() > deadline:
                        raise ArtifactUnavailable("artifact download exceeded its deadline")
                    received += len(chunk)
                    if received > file.bytes:
                        raise ArtifactError(f"download exceeds declared bytes for {file.path}")
                    digest.update(chunk)
                    output.write(chunk)
            finally:
                if response is not None:
                    response.close()
        if received != file.bytes or digest.hexdigest() != file.sha256:
            raise ArtifactError(f"hash or size mismatch for {file.path}")
        os.replace(temporary, destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _download_hippius_file(
    reference: ArtifactReference, file: ManifestFile, destination: Path, *, endpoint: str | None
) -> None:
    relative = _safe_path(file.path)
    download_root = destination
    for _ in relative.parts:
        download_root = download_root.parent
    if not download_root.is_dir():
        raise ArtifactError("Hippius download root must exist")
    expected = (download_root / relative).resolve()
    if expected != destination.resolve():
        raise ArtifactUnavailable("Hippius client returned an unexpected download path")
    result = _hippius_bridge(
        "fetch",
        reference,
        endpoint=endpoint,
        payload={
            "path": file.path,
            "destination": str(download_root),
            "bytes": file.bytes,
            "sha256": file.sha256,
        },
        timeout_seconds=300.0,
    )
    if (
        result.get("path") != str(expected)
        or not destination.is_file()
        or (destination.stat().st_size != file.bytes or _sha256_path(destination) != file.sha256)
    ):
        destination.unlink(missing_ok=True)
        raise ArtifactError(f"hash or size mismatch for {file.path}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_verified(
    manifest: ModelManifest,
    destination: str | Path,
    *,
    hf_base_url: str = HF_BASE_URL,
    hippius_endpoint: str | None = None,
    timeout_seconds: float = 60.0,
    download_deadline_seconds: float = HF_DOWNLOAD_DEADLINE_SECONDS,
) -> Path:
    """Download exactly the manifest files and verify every byte before use."""

    if timeout_seconds <= 0 or not 0 < download_deadline_seconds <= HF_DOWNLOAD_DEADLINE_SECONDS:
        raise ArtifactError("download timeouts are outside the bounded range")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ArtifactError("destination must be a directory")
    root = root.resolve()
    for item in manifest.files:
        relative = _safe_path(item.path)
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ArtifactError(f"unsafe output path: {item.path!r}")
        if target.exists():
            if not target.is_file():
                raise ArtifactError(f"existing cache target is not a regular file: {item.path}")
            if target.stat().st_size != item.bytes or _sha256_path(target) != item.sha256:
                raise ArtifactError(f"existing cache file does not match manifest: {item.path}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if manifest.reference.backend == "hf":
            _download_hf_file(
                manifest.reference,
                item,
                target,
                hf_base_url=hf_base_url,
                timeout_seconds=timeout_seconds,
                deadline_seconds=download_deadline_seconds,
            )
        else:
            _download_hippius_file(manifest.reference, item, target, endpoint=hippius_endpoint)
    return root
