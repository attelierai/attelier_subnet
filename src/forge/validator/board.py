"""Off-chain publication of round plans, reveals and certificates.

The chain only carries short commitments; the documents they bind live here.
``FileBoard`` is a shared directory, enough for a single operator on testnet.
A multi-operator deployment needs a replicated store (object storage or IPFS)
behind the same three methods.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

KINDS = ("plan", "reveal", "certificate")


class FileBoard:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, kind: str, round_id: str, name: str) -> Path:
        if kind not in KINDS or "/" in round_id or "/" in name:
            raise ValueError("invalid board key")
        return self.root / round_id / kind / f"{name}.json"

    def put(self, kind: str, round_id: str, name: str, payload: dict[str, Any]) -> bool:
        """Write once. Re-writing identical content is a no-op; different content fails."""

        path = self._path(kind, round_id, name)
        encoded = json.dumps(payload, sort_keys=True, indent=2)
        if path.exists():
            if path.read_text() != encoded:
                raise FileExistsError(f"{kind}/{name} for {round_id} already published")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(encoded)
        os.replace(tmp, path)
        return True

    def get(self, kind: str, round_id: str, name: str) -> dict[str, Any] | None:
        path = self._path(kind, round_id, name)
        return json.loads(path.read_text()) if path.exists() else None

    def list(self, kind: str, round_id: str) -> dict[str, dict[str, Any]]:
        folder = self.root / round_id / kind
        if not folder.exists():
            return {}
        return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))}
