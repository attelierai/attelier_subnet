"""Bounded public transport for a signed gateway master envelope.

The payload is only a distribution channel.  Callers must still verify its
signature against their separately pinned local policy and, where applicable,
its on-chain commitment.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_MASTER_BYTES = 1_048_576
MASTER_TIMEOUT_SECONDS = 10


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _https_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("gateway_master_url_https")


def parse_gateway_master_bytes(
    url: str, raw: bytes, content_length: str | None = None
) -> dict[str, Any]:
    """Validate transport bounds and decode one JSON object, without trust."""
    _https_url(url)
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise ValueError("gateway_master_content_length") from exc
        if declared < 0 or declared > MAX_MASTER_BYTES:
            raise ValueError("gateway_master_too_large")
    if len(raw) > MAX_MASTER_BYTES:
        raise ValueError("gateway_master_too_large")
    try:
        value = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("gateway_master_json") from exc
    if not isinstance(value, dict):
        raise ValueError("gateway_master_json_object")
    return value


def fetch_gateway_master(url: str) -> dict[str, Any]:
    """Fetch one HTTPS object with no redirects, bounded to one MiB/10 seconds."""
    _https_url(url)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Attelier-LightValidator/1.0",
        },
    )
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=MASTER_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_MASTER_BYTES + 1)
            return parse_gateway_master_bytes(url, raw, response.headers.get("Content-Length"))
    except HTTPError as exc:
        raise ValueError(f"gateway_master_http_{exc.code}") from exc
    except OSError as exc:
        raise ValueError("gateway_master_fetch") from exc
