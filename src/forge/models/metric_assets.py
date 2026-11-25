"""Provision validator-owned metric weights before any timed GPU work."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from forge.http_transport import http_origin
from forge.protocol.hashing import sha256_file


class _NoDowngradeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        scheme, _, _ = http_origin(newurl)
        if request.type == "https" and scheme != "https":
            return None
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _http_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("metric asset URL must be a string")
    try:
        http_origin(value)
    except ValueError as error:
        raise ValueError("metric asset URL must be an HTTP(S) URL without credentials") from error
    return value


def provision(manifest: dict, cache: Path) -> dict[str, str]:
    """Download trusted operator assets, verifying every byte before use.

    Existing complete files are reverified. An unfinished download is retained
    and requires operator inspection rather than a silent replacement/retry.
    """
    cache.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, spec in sorted(manifest.items(), key=lambda item: "url" not in item[1]):
        if not isinstance(spec, dict):
            raise ValueError("metric asset specification must be an object")
        expected_bytes = spec.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError("metric asset bytes must be a non-negative integer")
        download_url = _http_url(spec["url"]) if "url" in spec else None
        target = cache / name
        if target.is_symlink() or not target.resolve().is_relative_to(cache.resolve()):
            raise ValueError("metric asset must remain inside cache")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with partial.open("xb") as output:
                if "url" in spec:
                    request = urllib.request.Request(download_url)
                    opener = urllib.request.build_opener(_NoDowngradeRedirect)
                    with opener.open(request, timeout=60) as response:
                        copied = 0
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            copied += len(chunk)
                            if copied > expected_bytes:
                                raise ValueError("metric asset exceeds declared bytes")
                            output.write(chunk)
                else:
                    archive = spec["archive"]
                    if archive not in verified:
                        raise ValueError("metric archive must be verified before extraction")
                    with (
                        zipfile.ZipFile(cache / archive) as zipped,
                        zipped.open(spec["member"]) as source,
                    ):
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            if partial.stat().st_size != spec["bytes"] or sha256_file(partial) != spec["sha256"]:
                raise ValueError(f"downloaded metric bytes differ: {name}")
            # Do not overwrite a concurrently provisioned destination.
            target.hardlink_to(partial)
            partial.unlink()
        if target.stat().st_size != spec["bytes"] or sha256_file(target) != spec["sha256"]:
            raise ValueError(f"cached metric bytes differ: {name}")
        verified[name] = spec["sha256"]
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    verified = provision(json.loads(args.manifest.read_text()), args.cache)
    print(json.dumps({"verified_files": len(verified), "sha256": verified}))


if __name__ == "__main__":
    main()
