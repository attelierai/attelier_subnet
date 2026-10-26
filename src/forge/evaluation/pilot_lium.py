"""Lium HTTP boundary for the owned pilot's closure supervisor.

The key is loaded only from the consuming project's ignored env. This module
never rents resources; delete requires the exact frozen ID, name and executor.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values


class LiumHTTPError(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"Lium API returned HTTP{status}")


class LiumClient:
    def __init__(self, env_file: Path, *, base_url: str = "https://lium.io/api"):
        self.key = dotenv_values(env_file).get("LIUM_API_KEY")
        if not self.key:
            raise ValueError("LIUM_API_KEY is missing from the project env")
        if base_url != "https://lium.io/api" and not base_url.startswith("http://127.0.0.1:"):
            raise ValueError("only the official API or a local integration fixture is allowed")
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, body: dict | None = None):
        if not path.startswith("/") or ".." in path or "://" in path:
            raise ValueError("invalid API path")
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-API-Key": self.key, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                data = response.read()
                value = json.loads(data) if data else {}
        except urllib.error.HTTPError as error:
            raise LiumHTTPError(error.code) from None
        except (OSError, ValueError):
            raise RuntimeError("Lium API unavailable or returned invalid JSON") from None
        return value

    def pods(self) -> list[dict]:
        value = self.request("GET", "/pods")
        # /pods contains active pods only; DELETING may already be omitted.
        if not isinstance(value, list) or any(
            not isinstance(row, dict) or not row.get("id") or not row.get("pod_name")
            for row in value
        ):
            raise ValueError("Lium pods response is not a complete pod array")
        return value

    def owned_pod(self, ownership: dict) -> dict:
        pod_id = str(UUID(ownership["pod_id"]))
        value = self.request("GET", f"/pods/{pod_id}")
        if (
            not isinstance(value, dict)
            or value.get("id") != pod_id
            or value.get("pod_name") != ownership["name"]
            or value.get("executor_id") != ownership["executor_id"]
            or not ownership["name"].startswith("attelier-evaluator-pilot-")
        ):
            raise ValueError("Lium provider ownership mismatch")
        return value

    def running_container(self, ownership: dict, manifest_digest: str) -> dict | None:
        """Check the owned pod's executor inventory against the pinned image.

        Inventories can predate pod creation. Absence is pending, never proof;
        the available-node catalog cannot recover it once the host is rented.
        """
        pod = self.owned_pod(ownership)
        node = pod.get("executor", {})
        if node.get("id") != ownership["executor_id"]:
            raise ValueError("executor inventory ownership mismatch")
        containers = node.get("specs", {}).get("docker", {}).get("containers", [])
        matches = [row for row in containers if row.get("name") == "pod_" + ownership["pod_id"]]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("ambiguous running container")
        container = matches[0]
        if not container.get("digest"):
            return None
        if container["digest"] != manifest_digest or not container.get("container_id"):
            raise ValueError("actual running container image differs from pinned manifest")
        return {key: container[key] for key in ("name", "digest", "container_id")}

    def provider_command(self, command: str, pod_id: str | None, ownership: dict):
        if command == "list":
            rows = self.pods()
            if not any(row["id"] == ownership["pod_id"] for row in rows):
                try:
                    rows.append(self.owned_pod(ownership))
                except LiumHTTPError as error:
                    if error.status != 404:
                        raise
            return [{"id": r["id"], "name": r["pod_name"]} for r in rows]
        if pod_id != ownership["pod_id"]:
            raise ValueError("command targets a different pod")
        value = self.owned_pod(ownership)
        if command == "get":
            return {"id": value["id"], "name": value["pod_name"]}
        if command == "delete":
            if value.get("status") != "DELETING":
                self.request("DELETE", f"/pods/{value['id']}")
            return {}  # The supervisor must still verify absence in the full list.
        raise ValueError("unsupported provider command")
