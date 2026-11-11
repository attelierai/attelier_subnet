"""Bounded miner availability payments, distinct from delivered-work rewards."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from forge.inference.light import (
    LightError,
    LightRunner,
    _atomic_write,
    _raw_public_key,
    _strict_json,
    _verify_ed25519,
    verify_feed,
)
from forge.inference.light_publication import publication_weights
from forge.inference.prelaunch import GENESIS, HOUR_MS, canonical, digest, simulate

DOMAIN = b"attelier/light-availability-input/v1\0"


def clear_continuity(runner: LightRunner, *, reset_selection: bool = True) -> None:
    with runner._connection() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS continuity_selection("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, selection TEXT NOT NULL)"
        )
        latest = connection.execute(
            "SELECT selection FROM continuity_selection ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if reset_selection and latest is not None and latest[0] != "{}":
            connection.execute("INSERT INTO continuity_selection(selection) VALUES('{}')")
    _atomic_write(
        runner.state_dir / "continuity-status.json",
        {
            "schema": "attelier/light-continuity-status/v1",
            "status": "inactive",
        },
    )


def continuity_pending_stop(runner: LightRunner) -> bool:
    with runner._connection() as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='continuity_selection'"
        ).fetchone()
        if not exists:
            return False
        row = connection.execute(
            "SELECT selection FROM continuity_selection ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row is not None and row[0] != "{}"


def stop_continuity(runner: LightRunner, body: dict[str, Any]) -> dict[str, Any]:
    """Publish a new burn after continuity, even if this empty feed was sent before.

    Reusing a previously REVEALED empty-feed intent would leave the later
    continuity weights on chain until a new hour. Never reset the live journal.
    """
    if not continuity_pending_stop(runner):
        return body
    stopped = {
        **body,
        "schema": "attelier/light-continuity-stop/v1",
        "interruptedInputDigest": digest(body),
        "ledger": {"window": body["ledger"]["window"]},
    }
    stopped["ledger"]["sequence"] = _selection_sequence(runner, stopped)
    return stopped


def _selection_sequence(runner: LightRunner, body: dict[str, Any]) -> int:
    with runner._connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS continuity_selection("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, selection TEXT NOT NULL)"
        )
        latest = connection.execute(
            "SELECT id,selection FROM continuity_selection ORDER BY id DESC LIMIT 1"
        ).fetchone()
        selection = canonical(body).decode("ascii")
        if latest is None or latest[1] != selection:
            cursor = connection.execute(
                "INSERT INTO continuity_selection(selection) VALUES(?)", (selection,)
            )
            counter = cursor.lastrowid
        else:
            counter = latest[0]
        connection.execute("COMMIT")
    return -1_000_000_000_000 - counter


def verify_availability(raw: bytes, key: str, now: int) -> dict[str, Any]:
    if len(raw) > 1_048_576:
        raise LightError("light_availability_size")
    envelope = _strict_json(raw)
    public = _raw_public_key(key)
    body = envelope.get("body")
    if (
        set(envelope) != {"schema", "keyId", "digest", "signature", "body"}
        or envelope.get("schema") != "attelier/light-availability/v1"
        or envelope.get("keyId") != hashlib.sha256(public).hexdigest()
        or not isinstance(body, dict)
        or envelope.get("digest") != digest(body)
        or not isinstance(envelope.get("signature"), str)
        or not _verify_ed25519(public, DOMAIN + canonical(body), envelope["signature"])
    ):
        raise LightError("light_availability_signature")
    if (
        set(body)
        != {
            "schema",
            "network",
            "genesis",
            "netuid",
            "mechanism",
            "issuedAtMs",
            "expiresAtMs",
            "registry",
            "workers",
        }
        or body.get("schema") != "attelier/light-availability-input/v1"
        or body.get("network") != "finney"
        or body.get("genesis") != GENESIS
        or body.get("netuid") != 126
        or body.get("mechanism") != 0
        or type(body.get("issuedAtMs")) is not int
        or type(body.get("expiresAtMs")) is not int
        or body["expiresAtMs"] - body["issuedAtMs"] != 120_000
        or not body["issuedAtMs"] <= now < body["expiresAtMs"]
        or not isinstance(body.get("registry"), dict)
        or not isinstance(body.get("workers"), list)
        or len(body["workers"]) > 128
    ):
        raise LightError("light_availability_schema_or_expiry")
    registry = body["registry"]
    if (
        registry.get("schema") != "attelier/prelaunch-registry/v1"
        or registry.get("network") != "finney"
        or registry.get("genesis") != GENESIS
        or registry.get("netuid") != 126
        or registry.get("mechanism") != 0
        or not isinstance(registry.get("miners"), list)
        or len(registry["miners"]) > 128
    ):
        raise LightError("light_availability_registry")
    seen: set[tuple[int, str]] = set()
    for worker in body["workers"]:
        if (
            not isinstance(worker, dict)
            or set(worker) != {"uid", "hotkey", "workerId", "checkedAtMs"}
            or type(worker.get("uid")) is not int
            or not 0 < worker["uid"] < 65_536
            or not isinstance(worker.get("hotkey"), str)
            or not isinstance(worker.get("workerId"), str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", worker["workerId"])
            or type(worker.get("checkedAtMs")) is not int
            or not 0 <= body["issuedAtMs"] - worker["checkedAtMs"] <= 90_000
            or (worker["uid"], worker["hotkey"]) in seen
            or not any(
                m.get("uid") == worker["uid"]
                and m.get("hotkey") == worker["hotkey"]
                and m.get("coldkey") != registry.get("subnetOwnerColdkey")
                for m in registry["miners"]
                if isinstance(m, dict)
            )
        ):
            raise LightError("light_availability_worker")
        seen.add((worker["uid"], worker["hotkey"]))
    return body


def select_continuity(
    runner: LightRunner,
    raw: bytes,
    *,
    now: int,
    max_hours: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Never renew the historical source deadline with a heartbeat or restart.

    Preserve frozen bid discounts. An unavailable worker's share returns to
    burn; remaining workers never inherit that share. The accepted feed table
    and its accounting artifacts are left untouched.
    """
    if max_hours not in {2, 6}:
        raise LightError("light_continuity_limit")
    availability = verify_availability(raw, runner.public_key, now)
    if not availability["workers"]:
        clear_continuity(runner, reset_selection=False)
        return None
    runner.registry_verifier(availability["registry"])
    healthy = {(w["uid"], w["hotkey"], w["workerId"]) for w in availability["workers"]}
    current_identities = {
        (m["uid"], m["hotkey"], m["coldkey"]) for m in availability["registry"]["miners"]
    }
    with runner._connection() as connection:
        rows = connection.execute("SELECT raw FROM feed ORDER BY sequence DESC LIMIT 24").fetchall()
    for (encoded,) in rows:
        # Verify historical authority at its original issue time. This is
        # explicitly continuity, never a claim that the old feed is fresh.
        old = _strict_json(encoded)["body"]
        end = old["ledger"]["window"]["endMs"]
        if not end <= now < end + max_hours * HOUR_MS:
            continue
        source = verify_feed(encoded, runner.public_key, now_ms=old["issuedAtMs"], allow_live=True)
        if source["mode"] != "live" or source["policy"]["schema"] not in {
            "attelier/prelaunch-policy/v4",
            "attelier/prelaunch-policy/v5",
        }:
            continue
        artifact = simulate(source["ledger"], source["policy"], source["registry"])
        weights = publication_weights(source, artifact)
        if set(weights) == {0}:
            continue
        # Use the most recent actual positive allocation, not a succession of
        # older distributions chosen to find an offline miner to pay.
        miners = {m["uid"]: m for m in source["registry"]["miners"]}
        workers = {(w["minerHotkey"], w["workerId"]) for w in source["policy"]["workers"]}
        retained = {
            uid: weight
            for uid, weight in weights.items()
            if uid != 0
            and (uid, miners[uid]["hotkey"], miners[uid]["coldkey"]) in current_identities
            and any(
                (miners[uid]["hotkey"], worker) in workers
                for u, hotkey, worker in healthy
                if u == uid and hotkey == miners[uid]["hotkey"]
            )
        }
        if not retained:
            clear_continuity(runner, reset_selection=False)
            return None
        retained[0] = 65_535 - sum(retained.values())
        hour = now // HOUR_MS
        body = {
            "schema": "attelier/light-continuity-input/v1",
            "sourceDigest": digest(source),
            "sourceWindow": source["ledger"]["window"],
            "expiresAtMs": end + max_hours * HOUR_MS,
            "policy": source["policy"],
            "registry": source["registry"],
            "ledger": {"window": {"startMs": (hour - 1) * HOUR_MS, "endMs": hour * HOUR_MS}},
            "relativeWeights": [[uid, weight] for uid, weight in sorted(retained.items())],
        }
        # Persist selection transitions, including A -> B -> A. Hashing only
        # the vector would incorrectly reuse a receipt after a later burn.
        body["ledger"]["sequence"] = _selection_sequence(runner, body)
        result = {
            **artifact,
            "schema": "attelier/light-continuity-artifact/v1",
            "miners": [m for m in artifact["miners"] if m["uid"] in retained],
            "unsignedProposedWeights": {
                "total": 65_535,
                "entries": [
                    {"kind": "burn", "weight": retained[0]},
                    *[
                        {
                            "kind": "miner",
                            "uid": uid,
                            "hotkey": miners[uid]["hotkey"],
                            "weight": weight,
                        }
                        for uid, weight in sorted(retained.items())
                        if uid
                    ],
                ],
            },
        }
        if sum(retained.values()) != 65_535 or sum(v for u, v in retained.items() if u) > 3276:
            raise LightError("light_continuity_budget")
        _atomic_write(
            runner.state_dir / "continuity-status.json",
            {
                "schema": "attelier/light-continuity-status/v1",
                "status": "selected",
                "checkedAtMs": now,
                "input": body,
                "availability": _strict_json(raw),
            },
        )
        return body, result
    clear_continuity(runner, reset_selection=False)
    return None
