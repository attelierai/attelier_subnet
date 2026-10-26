"""Provision validator-owned metric weights before any timed GPU work."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from forge.protocol.hashing import sha256_file


def provision(manifest: dict, cache: Path) -> dict[str, str]:
    """Download trusted operator assets, verifying every byte before use.

    Existing complete files are reverified. An unfinished download is retained
    and requires operator inspection rather than a silent replacement/retry.
    """
    cache.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, spec in sorted(manifest.items(), key=lambda item: "url" not in item[1]):
        target = cache / name
        if target.is_symlink() or not target.resolve().is_relative_to(cache.resolve()):
            raise ValueError("metric asset must remain inside cache")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with partial.open("xb") as output:
                if "url" in spec:
                    with urllib.request.urlopen(spec["url"], timeout=60) as response:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
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
