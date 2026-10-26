"""Isolated Hippius operations for environments that run an older Hub stack.

This module is invoked with ``python -m forge.models.hippius_bridge`` by a dedicated
interpreter containing only the ``hippius`` extra. Its stdin/stdout protocol is
JSON so the GPU runtime never imports hippius-hub or Hugging Face Hub >=1.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "forge-manifest.json"
MAX_DECLARATION_BYTES = 1024 * 1024
MAX_LICENSE_BYTES = 64 * 1024
_SPDX = re.compile(r"^\s*SPDX-License-Identifier:\s*([A-Za-z0-9.-]+)\s*$", re.MULTILINE)
_LICENSE_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")


def _client():
    import hippius_hub

    if hippius_hub.__version__ != "0.6.0":
        raise RuntimeError("trusted Hippius bridge requires hippius-hub==0.6.0")
    return hippius_hub


def _kwargs(request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"revision": request["revision"], "token": False}
    endpoint = request.get("endpoint")
    if endpoint is not None:
        kwargs["endpoint"] = endpoint
    return kwargs


def _safe_relative(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        raise ValueError("unsafe relative path")
    return candidate


def _declared_file_size(info: Any, name: str, *, limit: int, label: str) -> None:
    matches = [
        item for item in getattr(info, "siblings", ()) if getattr(item, "rfilename", None) == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"registry did not return one {label} entry")
    size = getattr(matches[0], "size", None)
    if type(size) is not int or size <= 0 or size > limit:
        raise RuntimeError(f"{label} exceeds size limit")


def _verify_fetch_metadata(
    info: Any, name: str, expected_size: object, expected_sha256: object
) -> None:
    """Bind a requested byte range to the immutable OCI manifest before download."""
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("fetch bytes must be a positive integer")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("fetch sha256 must be lowercase SHA-256")
    matches = [
        item for item in getattr(info, "siblings", ()) if getattr(item, "rfilename", None) == name
    ]
    if len(matches) != 1:
        raise RuntimeError("registry did not return exactly one requested file")
    item = matches[0]
    if getattr(item, "size", None) != expected_size:
        raise RuntimeError("registry file size differs from pinned manifest")
    if getattr(item, "blob_id", None) != f"sha256:{expected_sha256}":
        raise RuntimeError("registry blob digest differs from pinned manifest")


def _read_bounded(path: Path, *, limit: int, label: str) -> bytes:
    with path.open("rb") as handle:
        payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise RuntimeError(f"{label} exceeds size limit")
    return payload


def _public_license(text: str) -> str | None:
    match = _SPDX.search(text)
    if match is not None:
        return match.group(1).lower()
    lowered = text.lower()
    if "apache license" in lowered and "version 2.0" in lowered:
        return "apache-2.0"
    if "mit license" in lowered:
        return "mit"
    if "redistribution and use in source and binary forms" in lowered:
        return "bsd-3-clause" if "neither the name" in lowered else "bsd-2-clause"
    if "creative commons attribution-sharealike 4.0 international" in lowered:
        return "cc-by-sa-4.0"
    if "creative commons attribution 4.0 international" in lowered:
        return "cc-by-4.0"
    return None


def _inspect(request: dict[str, Any]) -> dict[str, Any]:
    hub = _client()
    kwargs = _kwargs(request)
    info = hub.model_info(request["repo_id"], **kwargs)
    if getattr(info, "private", None) is True or getattr(info, "gated", False):
        raise RuntimeError("artifact is not public and ungated")
    names = [getattr(item, "rfilename", None) for item in getattr(info, "siblings", ())]
    if not names or any(not isinstance(item, str) for item in names):
        raise RuntimeError("registry did not return a file list")
    available_license_files = [name for name in _LICENSE_FILES if name in names]
    if not available_license_files:
        raise RuntimeError("public Hippius artifact must contain a standard license file")
    license_name = available_license_files[0]
    _declared_file_size(info, license_name, limit=MAX_LICENSE_BYTES, label=license_name)
    with tempfile.TemporaryDirectory(prefix="forge-hippius-probe-") as temporary:
        snapshot = Path(
            hub.snapshot_download(
                request["repo_id"],
                local_dir=temporary,
                allow_patterns=[license_name],
                max_workers=1,
                **kwargs,
            )
        ).resolve()
        license_path = (snapshot / license_name).resolve()
        if snapshot not in license_path.parents or not license_path.is_file():
            raise RuntimeError("public Hippius artifact license file is unreadable")
        payload = _read_bounded(license_path, limit=MAX_LICENSE_BYTES, label=license_name)
    try:
        license_text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("LICENSE.txt must be UTF-8") from exc
    license_value = _public_license(license_text)
    if license_value is None:
        raise RuntimeError("public Hippius artifact license is not a supported reusable license")
    card = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    card_license = card.get("license") if isinstance(card, dict) else None
    if card_license is not None and card_license.lower() != license_value:
        raise RuntimeError("registry and LICENSE.txt licenses disagree")
    return {"files": names, "license": license_value}


def _declaration(request: dict[str, Any]) -> dict[str, Any]:
    hub = _client()
    info = hub.model_info(request["repo_id"], **_kwargs(request))
    _declared_file_size(info, MANIFEST_NAME, limit=MAX_DECLARATION_BYTES, label=MANIFEST_NAME)
    with tempfile.TemporaryDirectory(prefix="forge-hippius-manifest-") as temporary:
        snapshot = Path(
            hub.snapshot_download(
                request["repo_id"],
                local_dir=temporary,
                allow_patterns=[MANIFEST_NAME],
                max_workers=1,
                **_kwargs(request),
            )
        ).resolve()
        path = (snapshot / MANIFEST_NAME).resolve()
        if snapshot not in path.parents or not path.is_file():
            raise RuntimeError("snapshot is missing forge-manifest.json")
        payload = _read_bounded(path, limit=MAX_DECLARATION_BYTES, label=MANIFEST_NAME)
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise RuntimeError("forge-manifest.json must be a JSON object")
    return {"declaration": decoded}


def _fetch(request: dict[str, Any]) -> dict[str, Any]:
    hub = _client()
    relative = _safe_relative(request["path"])
    expected_size = request["bytes"]
    expected_sha256 = request["sha256"]
    # `info.sha` is the OCI config digest, not the repository manifest digest.
    # The pinned revision is supplied separately; per-file blob_id is what binds
    # this logical path to the requested bytes before the downloader allocates.
    info = hub.model_info(request["repo_id"], files_metadata=True, **_kwargs(request))
    _verify_fetch_metadata(info, request["path"], expected_size, expected_sha256)
    root = Path(request["destination"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / relative).resolve()
    if root not in target.parents:
        raise ValueError("unsafe destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hub.hf_hub_download(
            request["repo_id"], request["path"], local_dir=str(root), **_kwargs(request)
        )
    ).resolve()
    if downloaded != target or not target.is_file():
        raise RuntimeError("client returned an unexpected download path")
    digest_state = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    if target.stat().st_size != expected_size or digest != expected_sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError("downloaded file does not match declared hash or size")
    return {"path": str(target)}


def main() -> int:
    os.environ.pop("FORGE_HIPPIUS_PYTHON", None)
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict) or request.get("operation") not in {
            "inspect",
            "declaration",
            "fetch",
        }:
            raise ValueError("invalid bridge request")
        operation = request["operation"]
        # hippius-hub 0.6.0 can print cold-download progress to stdout. The
        # bridge protocol reserves stdout for exactly one machine-readable JSON.
        with contextlib.redirect_stdout(sys.stderr):
            result = {"inspect": _inspect, "declaration": _declaration, "fetch": _fetch}[operation](
                request
            )
        json.dump({"ok": True, "result": result}, sys.stdout, separators=(",", ":"))
        return 0
    except Exception as exc:
        json.dump({"ok": False, "error": str(exc)}, sys.stdout, separators=(",", ":"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
