"""Restrict HTTP redirects for provider requests and configured report peers."""

from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def http_origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    if (
        parts.scheme not in {"https", "http"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError("invalid HTTP origin")
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("invalid HTTP origin") from error
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.scheme, parts.hostname, port


class _SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if http_origin(request.full_url) != http_origin(newurl):
            return None
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def open_same_origin(request: Request, *, timeout: int):
    """Open an HTTP(S) request without forwarding it to another origin."""
    http_origin(request.full_url)
    return build_opener(_SameOriginRedirect()).open(request, timeout=timeout)
