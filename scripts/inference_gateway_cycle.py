#!/usr/bin/env python3
"""Run one resumable, single-writer M1 gateway publication cycle.

This deliberately has no chain or wallet implementation of its own.  It only
orders the already guarded ``inference_gateway_epoch.py`` commands and records
which *revealed* window may be followed by the next one.  It is intended for a
systemd timer owned by the M1 operator, never for a miner or the gateway.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"object required in {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf8") as output:
        os.chmod(temporary, 0o600)
        json.dump(value, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _body_digest(body: dict[str, Any]) -> str:
    # The protocol canonical form rejects floats and uses these JSON separators.
    # Gateway bodies have already been schema-checked by gateway-sign.
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _config(path: Path) -> dict[str, Any]:
    config = _read(path)
    required = {
        "network",
        "netuid",
        "mechanism",
        "epochCommand",
        "closedBodyDirectory",
        "workDirectory",
        "policy",
        "master",
        "light",
        "publicationAuthorityDirectory",
        "gatewaySealCommand",
    }
    signing = {key for key in ("gatewayPrivateKey", "gatewaySignCommand") if key in config}
    if (
        set(config) - signing != required
        or len(signing) != 1
        or config["network"] != "test"
        or config["netuid"] != 578
        or config["mechanism"] != 1
    ):
        raise ValueError("gateway_cycle_namespace")
    if (
        not isinstance(config["epochCommand"], list)
        or not config["epochCommand"]
        or not all(isinstance(item, str) and item for item in config["epochCommand"])
    ):
        raise ValueError("gateway_cycle_command")
    if (
        not isinstance(config["gatewaySealCommand"], list)
        or not config["gatewaySealCommand"]
        or not all(isinstance(item, str) and item for item in config["gatewaySealCommand"])
    ):
        raise ValueError("gateway_cycle_seal_command")
    if "gatewaySignCommand" in config:
        command = config["gatewaySignCommand"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
            or sum(item.count("{body}") for item in command) != 1
            or sum(item.count("{out}") for item in command) != 1
        ):
            raise ValueError("gateway_cycle_sign_command")
    for key in ("closedBodyDirectory", "workDirectory", "policy", "publicationAuthorityDirectory"):
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError("gateway_cycle_paths")
    if "gatewayPrivateKey" in config and (
        not isinstance(config["gatewayPrivateKey"], str)
        or not Path(config["gatewayPrivateKey"]).is_absolute()
    ):
        raise ValueError("gateway_cycle_paths")
    for role in ("master", "light"):
        fields = config[role]
        expected = {"walletName", "walletHotkey", "walletPath"} | (
            {"publisherHotkey"} if role == "master" else set()
        )
        if (
            not isinstance(fields, dict)
            or set(fields) != expected
            or not all(isinstance(value, str) and value for value in fields.values())
            or not Path(fields["walletPath"]).is_absolute()
        ):
            raise ValueError("gateway_cycle_wallet")
    policy = _read(Path(config["policy"]))
    workers = policy.get("workers")
    if (
        policy.get("network") != "test"
        or policy.get("netuid") != 578
        or policy.get("mechanism") != 1
        or policy.get("masterPublisherHotkey") != config["master"]["publisherHotkey"]
        or policy.get("masterPublisherHotkey") == policy.get("m0PublisherHotkey")
        or not isinstance(workers, list)
        or any(
            not isinstance(worker, dict) or worker.get("trustMode") != "operator_whitelist"
            for worker in workers
        )
    ):
        raise ValueError("gateway_cycle_trusted_policy")
    return config


def _run(config: dict[str, Any], *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [*config["epochCommand"], *arguments], check=True, text=True, capture_output=True
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("gateway epoch command did not return one JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeError("gateway epoch command did not return an object")
    return value


def _closed_bodies(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not directory.is_dir():
        raise ValueError("gateway_cycle_closed_directory")
    bodies: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        body = _read(path)
        if (
            body.get("schema") != "attelier/gateway-window/v1"
            or body.get("network") != "test"
            or body.get("netuid") != 578
            or body.get("mechanism") != 1
            or type(body.get("sequence")) is not int
        ):
            raise ValueError(f"gateway_cycle_body:{path.name}")
        bodies.append((path, body))
    return sorted(bodies, key=lambda item: item[1]["sequence"])


def _next_body(config: dict[str, Any], state_path: Path) -> tuple[Path, dict[str, Any]] | None:
    state = _read(state_path) if state_path.exists() else None
    expected_sequence = 1
    expected_previous: str | None = None
    if state is not None:
        if (
            set(state)
            != {"network", "netuid", "mechanism", "sequence", "gatewayBodyDigest", "masterDigest"}
            or state.get("network") != "test"
            or state.get("netuid") != 578
            or state.get("mechanism") != 1
            or type(state.get("sequence")) is not int
            or not isinstance(state.get("gatewayBodyDigest"), str)
            or not isinstance(state.get("masterDigest"), str)
        ):
            raise ValueError("gateway_cycle_state")
        expected_sequence = state["sequence"] + 1
        expected_previous = state["gatewayBodyDigest"]
    bodies = _closed_bodies(Path(config["closedBodyDirectory"]))
    later = [(path, body) for path, body in bodies if body["sequence"] >= expected_sequence]
    if not later:
        return None
    path, body = later[0]
    if body["sequence"] != expected_sequence or body.get("previousHash") != expected_previous:
        raise ValueError("gateway_cycle_noncontiguous_window")
    return path, body


def _paths(config: dict[str, Any], body: dict[str, Any]) -> dict[str, Path]:
    work = Path(config["workDirectory"])
    sequence = body["sequence"]
    stem = f"m1-{sequence:012d}-{_body_digest(body)[:16]}"
    authority = Path(config["publicationAuthorityDirectory"])
    return {
        "gateway": work / "signed" / f"{stem}.gateway.json",
        "master": work / "master" / f"{stem}.master.json",
        "masterReceipt": work / "master" / f"{stem}.receipt.json",
        "lightReceipt": authority / f"{stem}.light-receipt.json",
        "lightState": authority / "gateway-light-state.json",
        "cycleState": work / "cycle-state.json",
    }


def _write_result(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and _read(path) != value:
        raise ValueError(f"gateway_cycle_artifact_conflict:{path.name}")
    if not path.exists():
        _atomic(path, value)


def _gateway(
    config: dict[str, Any], body_path: Path, body: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    if paths["gateway"].exists():
        envelope = _read(paths["gateway"])
    elif "gatewaySignCommand" in config:
        # A root-owned SSH wrapper may sign on the isolated CPU key host. It
        # receives exactly a body path and an output path, never wallet inputs.
        paths["gateway"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = [
            part.replace("{body}", str(body_path)).replace("{out}", str(paths["gateway"]))
            for part in config["gatewaySignCommand"]
        ]
        subprocess.run(command, check=True)
        envelope = _read(paths["gateway"])
    else:
        envelope = _run(
            config,
            "gateway-sign",
            "--body",
            str(body_path),
            "--private-key",
            config["gatewayPrivateKey"],
        )
        _write_result(paths["gateway"], envelope)
    if envelope.get("body") != body or envelope.get("digest") != _body_digest(body):
        raise ValueError("gateway_cycle_gateway_binding")
    return envelope


def _master(
    config: dict[str, Any], paths: dict[str, Path], gateway: dict[str, Any]
) -> dict[str, Any] | None:
    master = _read(paths["master"]) if paths["master"].exists() else None
    if master is None:
        paths["master"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        wallet = config["master"]
        master = _run(
            config,
            "master-live-sign",
            "--envelope",
            str(paths["gateway"]),
            "--policy",
            config["policy"],
            "--wallet-name",
            wallet["walletName"],
            "--wallet-hotkey",
            wallet["walletHotkey"],
            "--wallet-path",
            wallet["walletPath"],
            "--preflight-out",
            str(paths["master"].with_suffix(".preflight.json")),
        )
        _write_result(paths["master"], master)
    # Master binds the complete signed envelope (including its signature), while
    # previousHash chains the canonical gateway *body* digest. They are distinct.
    if master.get("body", {}).get("sourceGatewayEnvelopeHash") != _body_digest(gateway):
        raise ValueError("gateway_cycle_master_binding")
    receipt = paths["masterReceipt"]
    if not receipt.exists():
        wallet = config["master"]
        _run(
            config,
            "master-commit",
            "--master-envelope",
            str(paths["master"]),
            "--policy",
            config["policy"],
            "--receipt",
            str(receipt),
            "--wallet-name",
            wallet["walletName"],
            "--wallet-hotkey",
            wallet["walletHotkey"],
            "--wallet-path",
            wallet["walletPath"],
            "--execute",
        )
    status = _read(receipt).get("status")
    if status != "FINALIZED":
        _run(config, "master-reconcile", "--receipt", str(receipt))
        status = _read(receipt).get("status")
    if status == "FINALIZED":
        return master
    if status in {"SUBMITTING", "SUBMITTED", "PENDING", "UNKNOWN_AFTER_SUBMIT"}:
        return None
    raise RuntimeError(f"gateway master receipt is {status!r}")


def _light(config: dict[str, Any], paths: dict[str, Path], master: dict[str, Any]) -> bool:
    # gateway_publication requires both receipt and replay state directly below
    # this durable authority, including on a cold start.
    paths["lightReceipt"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = paths["lightReceipt"]
    if not receipt.exists():
        wallet = config["light"]
        _run(
            config,
            "light-publish",
            "--master-envelope",
            str(paths["master"]),
            "--policy",
            config["policy"],
            "--receipt",
            str(receipt),
            "--state",
            str(paths["lightState"]),
            "--wallet-name",
            wallet["walletName"],
            "--wallet-hotkey",
            wallet["walletHotkey"],
            "--wallet-path",
            wallet["walletPath"],
            "--publication-authority-directory",
            config["publicationAuthorityDirectory"],
            "--execute",
        )
    status = _read(receipt).get("status")
    if status != "REVEALED" and status != "NOOP":
        _run(
            config,
            "light-reconcile",
            "--master-envelope",
            str(paths["master"]),
            "--policy",
            config["policy"],
            "--receipt",
            str(receipt),
            "--state",
            str(paths["lightState"]),
            "--publication-authority-directory",
            config["publicationAuthorityDirectory"],
        )
        status = _read(receipt).get("status")
    if status == "NOOP":
        return False
    if status != "REVEALED":
        if status in {
            "SUBMITTING",
            "SUBMITTED",
            "PENDING",
            "REVEAL_PENDING",
            "UNKNOWN_AFTER_SUBMIT",
        }:
            return False
        raise RuntimeError(f"gateway light receipt is {status!r}")
    revealed = _read(paths["lightState"])
    if revealed.get("sequence") != master.get("body", {}).get("sequence") or revealed.get(
        "masterDigest"
    ) != master.get("digest"):
        raise RuntimeError("gateway light revealed state missing")
    return True


def cycle(config_path: Path) -> dict[str, Any]:
    config = _config(config_path)
    # The issuer/API owns ledger settlement and atomically writes closed bodies.
    # It is an argv list, never shell text, and receives no wallet arguments.
    subprocess.run(config["gatewaySealCommand"], check=True)
    state_path = Path(config["workDirectory"]) / "cycle-state.json"
    candidate = _next_body(config, state_path)
    if candidate is None:
        return {"status": "IDLE"}
    body_path, body = candidate
    paths = _paths(config, body)
    gateway = _gateway(config, body_path, body, paths)
    master = _master(config, paths, gateway)
    if master is None:
        return {"status": "MASTER_PENDING", "sequence": body["sequence"]}
    if not _light(config, paths, master):
        return {"status": "LIGHT_PENDING", "sequence": body["sequence"]}
    state = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "sequence": body["sequence"],
        "gatewayBodyDigest": gateway["digest"],
        "masterDigest": master["digest"],
    }
    _atomic(paths["cycleState"], state)
    return {"status": "REVEALED", "sequence": body["sequence"], "masterDigest": master["digest"]}


def locked_cycle(config_path: Path) -> dict[str, Any]:
    """Keep a manual invocation from racing the systemd timer's CR owner."""
    config = _config(config_path)
    lock_path = Path(config["workDirectory"]) / "cycle.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a+", encoding="utf8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("gateway cycle already running") from exc
        return cycle(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(locked_cycle(args.config), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
