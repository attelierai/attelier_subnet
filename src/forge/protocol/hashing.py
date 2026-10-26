"""Canonical encoding and content identity helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class CanonicalEncodingError(ValueError):
    """Raised when a value cannot be represented canonically."""


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalEncodingError("non-finite floats are forbidden")
        # JSON number rendering has differed across implementations. A tagged
        # repr preserves Python's exact, shortest round-trippable value.
        return {"__forge_float__": repr(value)}
    if isinstance(value, bytes):
        return {"__forge_bytes__": value.hex()}
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalEncodingError("canonical mapping keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, (set, frozenset)):
        items = [_normalize(item) for item in value]
        return sorted(items, key=lambda item: canonical_json_bytes(item))
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _normalize(float(value))
    raise CanonicalEncodingError(f"unsupported canonical type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value as stable UTF-8 JSON.

    Floats are finite and encoded as tagged ``repr`` strings, avoiding reliance
    on a JSON implementation's number formatting.
    """

    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_canonical(domain: str, *parts: Any) -> str:
    if not domain.startswith("forge/"):
        raise ValueError("domain-separation tags must start with 'forge/'")
    return sha256_bytes(canonical_json_bytes({"domain": domain, "parts": parts}))
