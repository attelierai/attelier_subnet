"""Low-level IO, strict JSON, Ed25519 verification and feed fetching for SN126 validators."""

from __future__ import annotations

import base64
import importlib.resources
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from forge.inference.prelaunch import canonical

MAX_RESPONSE_BYTES = 1_048_576
STATUS_SCHEMA = "attelier/light-status/v1"


def _burn_only_sequence(hour: int) -> int:
    """Reserve a negative, per-hour journal sequence for an operator pause."""
    if hour < 0:
        raise LightError("light_burn_sequence")
    return -(hour * 2 + 1)


def _outage_burn_sequence(hour: int) -> int:
    """Keep outage fallbacks disjoint from burn-only renewals."""
    if hour < 0:
        raise LightError("light_burn_sequence")
    return -(hour * 2 + 2)


class LightError(ValueError):
    """A rejected input; callers must never turn this into a payout action."""


def _strict_json(raw: bytes) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise LightError("light_json_duplicate_key")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(LightError("light_json_constant")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LightError("light_json") from error
    if not isinstance(parsed, dict):
        raise LightError("light_envelope_schema")
    return parsed


def _raw_public_key(value: str) -> bytes:
    try:
        key = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as error:
        raise LightError("light_public_key") from error
    if len(key) != 32:
        raise LightError("light_public_key")
    return key


def final_allocation_public_key(path: Path) -> str:
    """Read an explicitly pinned Ed25519 verifier key without a wallet."""
    try:
        raw = path.read_bytes().strip()
        if raw.startswith(b"-----BEGIN"):
            result = subprocess.run(
                ["openssl", "pkey", "-pubin", "-pubcheck", "-outform", "DER", "-in", str(path)],
                check=True,
                capture_output=True,
                timeout=15,
            ).stdout
            raw = base64.b64encode(result[-32:])
        value = raw.decode("ascii")
        _raw_public_key(value)
        return value
    except (OSError, UnicodeDecodeError, ValueError, subprocess.SubprocessError) as error:
        raise LightError("final_allocation_public_key") from error


def _verify_ed25519(key: bytes, message: bytes, signature: str) -> bool:
    try:
        raw_signature = base64.b64decode(signature, validate=True)
        if len(raw_signature) != 64:
            return False
        with tempfile.TemporaryDirectory(prefix="attelier-light-ed25519-") as temporary:
            root = Path(temporary)
            # RFC 8410 SubjectPublicKeyInfo wrapper for a raw Ed25519 public key.
            (root / "key.der").write_bytes(bytes.fromhex("302a300506032b6570032100") + key)
            (root / "message").write_bytes(message)
            (root / "signature").write_bytes(raw_signature)
            result = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    str(root / "key.der"),
                    "-rawin",
                    "-in",
                    str(root / "message"),
                    "-sigfile",
                    str(root / "signature"),
                ],
                capture_output=True,
                check=False,
                timeout=15,
            )
            return result.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise urllib.error.HTTPError(args[0], 302, "redirect refused", {}, None)


def fetch_feed(
    url: str,
    *,
    allow_http_fixture: bool = False,
    timeout_seconds: int = 15,
    availability: bool = False,
    production_path: str | None = None,
) -> bytes:
    """Fetch exactly the reviewed HTTPS endpoint; HTTP is test-fixture only."""
    parsed = urllib.parse.urlparse(url)
    fixture = (
        allow_http_fixture and parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
    )
    production = (
        parsed.scheme == "https"
        and parsed.hostname == "inference.attelier.ai"
        and parsed.path
        == (
            production_path
            or "/v1/subnet/126/light/" + ("availability" if availability else "latest")
        )
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )
    if not (fixture or production):
        raise LightError("light_feed_url")
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "attelier-light/1"}
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise LightError("light_feed_status")
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise LightError("light_feed_fetch") from error
    if len(data) > MAX_RESPONSE_BYTES:
        raise LightError("light_feed_too_large")
    return data


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical(value) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise LightError("light_artifact_write") from error


def default_public_key() -> str:
    """Load the release-pinned public verifier key from package data."""
    try:
        raw = importlib.resources.files("forge.inference").joinpath("defaults.json").read_bytes()
        value = _strict_json(raw)
        if set(value) != {"lightPublicKey"} or not isinstance(value["lightPublicKey"], str):
            raise LightError("light_defaults")
        _raw_public_key(value["lightPublicKey"])
        return value["lightPublicKey"]
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError, LightError) as error:
        raise LightError("light_defaults") from error
