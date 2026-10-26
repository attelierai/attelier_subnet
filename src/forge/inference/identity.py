"""Hotkey proof binds a private miner endpoint to a durable worker ledger."""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.inference.service import canonical

DOMAIN = b"forge/inference-miner/v1\0"


def attest(wallet: Any, miner: dict[str, Any], netuid: int, expires: float) -> dict[str, Any]:
    if not time.time() < expires <= time.time() + 86400:
        raise ValueError("Worker attestations expire within 24 hours")
    signer = Sr25519Signer(wallet)
    if signer.validator_hotkey != miner["hotkey"]:
        raise ValueError("Miner hotkey mismatch")
    payload = {
        "schema": "forge/inference-miner/v1",
        "network": "test",
        "netuid": netuid,
        "mechanism": 1,
        "hotkey": miner["hotkey"],
        "id": miner["id"],
        "workerId": miner["workerId"],
        "url": miner["url"],
        "modelManifestHash": miner["modelManifestHash"],
        "expires": expires,
    }
    # Reference mode binds the worker service verifier key and exact Comfy graph
    # to the hotkey statement, rather than trusting a mutable validator config.
    if "workflowHash" in miner or "verificationPublicKey" in miner:
        if not isinstance(miner.get("workflowHash"), str) or not isinstance(
            miner.get("verificationPublicKey"), str
        ):
            raise ValueError("Reference attestation requires workflow and verification key")
        payload["workflowHash"] = miner["workflowHash"]
        payload["verificationPublicKey"] = miner["verificationPublicKey"]
    return {
        "payload": payload,
        "signature": signer.sign(DOMAIN + canonical(payload).encode()).hex(),
    }


def verify(miner: dict[str, Any], netuid: int) -> dict[str, Any]:
    signed = json.loads(Path(miner["attestationFile"]).read_text())
    payload = signed["payload"]
    url = urllib.parse.urlparse(payload["url"])
    if (
        payload["schema"] != "forge/inference-miner/v1"
        or payload["network"] != "test"
        or payload["netuid"] != netuid
        or payload["mechanism"] != 1
        or payload["expires"] <= time.time()
        or payload["expires"] > time.time() + 86400
        or url.username
        or url.password
        or url.query
        or url.fragment
        or any(
            payload[key] != miner[key]
            for key in ("id", "workerId", "hotkey", "url", "modelManifestHash")
        )
        or (
            "verificationPublicKey" in miner
            and (
                payload.get("workflowHash") != miner.get("workflowHash")
                or payload.get("verificationPublicKey") != miner.get("verificationPublicKey")
            )
        )
    ):
        raise ValueError("Invalid or expired miner attestation")
    if not Sr25519Verifier().verify(
        miner["hotkey"], DOMAIN + canonical(payload).encode(), bytes.fromhex(signed["signature"])
    ):
        raise ValueError("Miner signature failed")
    return {
        "hotkey": miner["hotkey"],
        "workerId": miner["workerId"],
        "modelManifestHash": miner["modelManifestHash"],
        **(
            {
                "workflowHash": miner["workflowHash"],
                "verificationPublicKey": miner["verificationPublicKey"],
            }
            if "verificationPublicKey" in miner
            else {}
        ),
        "attestationExpires": payload["expires"],
    }


def bind_registry(registry: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if (
        config["network"] != "test"
        or config["netuid"] != registry["netuid"]
        or config["mechanism"] != 1
    ):
        raise ValueError("Configuration domain mismatch")
    result = dict(registry)
    result["workers"] = {}
    for miner in config["miners"]:
        if registry["uids"].get(str(miner["uid"])) != miner["hotkey"]:
            raise ValueError("Miner is not registered at the finalized block")
        result["workers"][miner["id"]] = verify(miner, registry["netuid"])
    return result
