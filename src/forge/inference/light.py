"""Authenticated SN126 light review runner; shadow by default, live only when explicit."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import importlib.resources
import json
import os
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.inference.light_registry import RegistryError, verify_finalized_registry
from forge.inference.prelaunch import DAY_MS, GENESIS, HOUR_MS, canonical, digest, simulate

DEFAULT_URL = "https://inference.attelier.ai/v1/subnet/126/light/latest"
ENVELOPE_SCHEMA = "attelier/light-feed/v1"
BODY_SCHEMA = "attelier/light-input/v1"
DOMAIN = b"attelier/light-input/v1\0"
MAX_RESPONSE_BYTES = 1_048_576
MAX_BUDGET_BPS = 1_000
STATUS_SCHEMA = "attelier/light-status/v1"


class LightError(ValueError):
    """A rejected input; callers must never turn this into a payout action."""


def _strict_json(raw: bytes) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise LightError("light_json_duplicate_key")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(LightError("light_json_constant")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LightError("light_json") from error
    if not isinstance(parsed, dict):
        raise LightError("light_envelope_schema")
    return parsed


def _raw_public_key(value: str) -> bytes:
    try:
        key = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as error:
        raise LightError("light_public_key") from error
    if len(key) != 32:
        raise LightError("light_public_key")
    return key


def _verify_ed25519(key: bytes, message: bytes, signature: str) -> bool:
    try:
        raw_signature = base64.b64decode(signature, validate=True)
        if len(raw_signature) != 64:
            return False
        with tempfile.TemporaryDirectory(prefix="attelier-light-ed25519-") as temporary:
            root = Path(temporary)
            # RFC 8410 SubjectPublicKeyInfo wrapper for a raw Ed25519 public key.
            (root / "key.der").write_bytes(bytes.fromhex("302a300506032b6570032100") + key)
            (root / "message").write_bytes(message)
            (root / "signature").write_bytes(raw_signature)
            result = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    str(root / "key.der"),
                    "-rawin",
                    "-in",
                    str(root / "message"),
                    "-sigfile",
                    str(root / "signature"),
                ],
                capture_output=True,
                check=False,
                timeout=15,
            )
            return result.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise urllib.error.HTTPError(args[0], 302, "redirect refused", {}, None)


def fetch_feed(url: str, *, allow_http_fixture: bool = False, timeout_seconds: int = 15) -> bytes:
    """Fetch exactly the reviewed HTTPS endpoint; HTTP is test-fixture only."""
    parsed = urllib.parse.urlparse(url)
    fixture = (
        allow_http_fixture and parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
    )
    production = (
        parsed.scheme == "https"
        and parsed.hostname == "inference.attelier.ai"
        and parsed.path == "/v1/subnet/126/light/latest"
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )
    if not (fixture or production):
        raise LightError("light_feed_url")
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "attelier-light/1"}
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise LightError("light_feed_status")
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise LightError("light_feed_fetch") from error
    if len(data) > MAX_RESPONSE_BYTES:
        raise LightError("light_feed_too_large")
    return data


def verify_feed(
    raw: bytes,
    public_key: str,
    *,
    now_ms: int | None = None,
    allow_live: bool = False,
) -> dict[str, Any]:
    """Authenticate and bind a gateway feed to one fresh SN126 M0 window."""
    if len(raw) > MAX_RESPONSE_BYTES:
        raise LightError("light_feed_too_large")
    envelope = _strict_json(raw)
    if (
        set(envelope) != {"schema", "keyId", "body", "digest", "signature"}
        or envelope.get("schema") != ENVELOPE_SCHEMA
    ):
        raise LightError("light_envelope_schema")
    body = envelope.get("body")
    if not isinstance(body, dict):
        raise LightError("light_body_schema")
    required = {
        "schema",
        "network",
        "genesis",
        "netuid",
        "mechanism",
        "mode",
        "issuedAtMs",
        "expiresAtMs",
        "policy",
        "ledger",
        "registry",
    }
    if set(body) != required or body.get("schema") != BODY_SCHEMA:
        raise LightError("light_body_schema")
    if (
        body.get("network") != "finney"
        or body.get("genesis") != GENESIS
        or body.get("netuid") != 126
        or body.get("mechanism") != 0
    ):
        raise LightError("light_namespace")
    if body.get("mode") not in {"shadow", "live"} or (body["mode"] == "live" and not allow_live):
        raise LightError("light_mode")
    if any(not isinstance(body[key], dict) for key in ("policy", "ledger", "registry")):
        raise LightError("light_input_schema")
    key = _raw_public_key(public_key)
    if envelope.get("keyId") != hashlib.sha256(key).hexdigest() or envelope.get("digest") != digest(
        body
    ):
        raise LightError("light_digest_or_key")
    if not isinstance(envelope.get("signature"), str) or not _verify_ed25519(
        key, DOMAIN + canonical(body), envelope["signature"]
    ):
        raise LightError("light_signature")
    issued, expires = body.get("issuedAtMs"), body.get("expiresAtMs")
    if type(issued) is not int or type(expires) is not int or issued < 0 or expires <= issued:
        raise LightError("light_timing")
    ledger, policy = body["ledger"], body["policy"]
    simulation = policy.get("simulation") if isinstance(policy, dict) else None
    window = ledger.get("window") if isinstance(ledger, dict) else None
    window_ms = policy.get("windowMs", DAY_MS) if isinstance(policy, dict) else None
    if (
        not isinstance(window, dict)
        or type(window_ms) is not int
        or set(window) != {"startMs", "endMs"}
        or type(window.get("startMs")) is not int
        or type(window.get("endMs")) is not int
    ):
        raise LightError("light_timing")
    if window_ms not in {HOUR_MS, DAY_MS}:
        raise LightError("light_window")
    # Hourly accounting is the live rollout contract.  Retain daily shadow
    # receipts so historical review artifacts remain readable, but do not let a
    # daily live feed silently change the actual publication cadence.
    if body["mode"] == "live" and window_ms != HOUR_MS:
        raise LightError("light_live_window")
    if body["mode"] == "live" and simulation.get("serviceBudgetBps") != 500:
        raise LightError("light_live_budget")
    if issued != ledger.get("closedAtMs") or issued < window.get("endMs"):
        raise LightError("light_timing")
    if expires != window["endMs"] + window_ms or issued < window["endMs"]:
        raise LightError("light_timing")
    current = int(time.time() * 1000) if now_ms is None else now_ms
    if current < issued or current >= expires:
        raise LightError("light_expired_or_not_yet_valid")
    if (
        not isinstance(simulation, dict)
        or type(simulation.get("serviceBudgetBps")) is not int
        or not 0 < simulation["serviceBudgetBps"] <= MAX_BUDGET_BPS
    ):
        raise LightError("light_budget")
    if simulation.get("allocationMode", "target") != "fixed":
        raise LightError("light_policy_mode")
    return body


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical(value) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise LightError("light_artifact_write") from error


def outage_blocks_service(body: dict[str, Any], outage: dict[str, Any]) -> bool:
    """A conservative burn is final for the closed hour that selected it."""
    burned = outage.get("burnedThroughEndMs")
    ledger = body.get("ledger")
    window = ledger.get("window") if isinstance(ledger, dict) else None
    return type(burned) is int and isinstance(window, dict) and window.get("endMs") <= burned


class LightRunner:
    """Stateful processor that only produces the existing unsigned simulation."""

    def __init__(
        self,
        state_dir: Path,
        public_key: str,
        *,
        registry_verifier: Callable[[dict[str, Any]], None] = verify_finalized_registry,
        allow_live: bool = False,
        writer_lock: bool = False,
    ) -> None:
        _raw_public_key(public_key)
        self.state_dir, self.public_key, self.registry_verifier, self.allow_live = (
            state_dir,
            public_key,
            registry_verifier,
            allow_live,
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = None
        if writer_lock:
            # SQLite serializes transactions in one directory.  This lock makes
            # productive writer ownership explicit; unsigned review remains
            # safe to run concurrently because it cannot sign or submit.
            self._lock = (state_dir / "light.lock").open("a+", encoding="utf8")
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self._lock.close()
                raise LightError("light_writer_active") from error
        self.database = state_dir / "light.sqlite3"
        schema = (
            "CREATE TABLE IF NOT EXISTS feed ("
            "sequence INTEGER PRIMARY KEY, digest TEXT NOT NULL UNIQUE, "
            "raw BLOB NOT NULL, artifact BLOB NOT NULL)"
        )
        with sqlite3.connect(self.database) as connection:
            connection.execute(schema)
        self._rebuild_latest()

    def _connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=30, isolation_level=None)

    def _latest(self, connection: sqlite3.Connection) -> tuple[int, str, bytes] | None:
        return connection.execute(
            "SELECT sequence, digest, artifact FROM feed ORDER BY sequence DESC LIMIT 1"
        ).fetchone()

    def _rebuild_latest(self) -> None:
        """Restore the cached file from durable state after an interrupted write."""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                latest = self._latest(connection)
                if latest is None:
                    connection.execute("COMMIT")
                    return
                artifact = json.loads(latest[2])
                path = self.state_dir / "latest-artifact.json"
                expected = canonical(artifact) + b"\n"
                if not path.is_file() or path.read_bytes() != expected:
                    _atomic_write(path, artifact)
                connection.execute("COMMIT")
        except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError) as error:
            raise LightError("light_state_unavailable") from error

    def unavailable(self, reason: str) -> None:
        """Expose current unavailability without treating an old artifact as fresh."""
        status = {"schema": STATUS_SCHEMA, "status": "unavailable", "reason": reason}
        with contextlib.suppress(LightError):
            _atomic_write(self.state_dir / "status.json", status)

    def _ready(self, sequence: int, feed_digest: str) -> None:
        # A second process may have committed a newer window since this one
        # processed its input. Publish only the current durable head.
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = self._latest(connection)
            if latest is not None and (sequence, feed_digest) == latest[:2]:
                _atomic_write(
                    self.state_dir / "status.json",
                    {
                        "schema": STATUS_SCHEMA,
                        "status": "ready",
                        "sequence": sequence,
                        "digest": feed_digest,
                    },
                )
            connection.execute("COMMIT")

    def process(self, raw: bytes, *, now_ms: int | None = None) -> dict[str, Any]:
        body = verify_feed(raw, self.public_key, now_ms=now_ms, allow_live=self.allow_live)
        sequence, feed_digest = body["ledger"]["sequence"], digest(body)
        try:
            # Strict input validation is deliberately complete before any RPC.
            artifact = simulate(body["ledger"], body["policy"], body["registry"])
        except (TypeError, ValueError, KeyError) as error:
            raise LightError("light_simulation_input") from error
        if (
            artifact.get("simulationOnly") is not True
            or artifact.get("publishable") is not False
            or artifact.get("chainCalls") != []
        ):
            raise LightError("light_unsigned_invariant")
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                latest = self._latest(connection)
                if latest is not None:
                    if sequence < latest[0] or (sequence == latest[0] and feed_digest != latest[1]):
                        raise LightError("light_replay_or_rollback")
                    if sequence == latest[0]:
                        connection.execute("COMMIT")
                        self._ready(sequence, feed_digest)
                        return json.loads(latest[2])
                try:
                    self.registry_verifier(body["registry"])
                except (RegistryError, OSError, RuntimeError, ValueError, KeyError) as error:
                    raise LightError("light_registry_unavailable") from error
                encoded = canonical(artifact)
                _atomic_write(self.state_dir / "latest-artifact.json", artifact)
                connection.execute(
                    "INSERT INTO feed(sequence, digest, raw, artifact) VALUES (?, ?, ?, ?)",
                    (sequence, feed_digest, raw, encoded),
                )
                connection.execute("COMMIT")
        except LightError:
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            raise LightError("light_state_unavailable") from error
        self._ready(sequence, feed_digest)
        return artifact


def default_public_key() -> str:
    """Load the release-pinned public verifier key from package data."""
    try:
        raw = importlib.resources.files("forge.inference").joinpath("defaults.json").read_bytes()
        value = _strict_json(raw)
        if set(value) != {"lightPublicKey"} or not isinstance(value["lightPublicKey"], str):
            raise LightError("light_defaults")
        _raw_public_key(value["lightPublicKey"])
        return value["lightPublicKey"]
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError, LightError) as error:
        raise LightError("light_defaults") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("/var/lib/attelier-light"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--live", action="store_true", help="Enable the hourly Finney SN126 M0 writer"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate a live feed and record its vector only"
    )
    parser.add_argument("--wallet", help="Explicit Bittensor wallet name; required with --live")
    parser.add_argument("--hotkey", help="Explicit Bittensor hotkey name; required with --live")
    parser.add_argument(
        "--wallet-path",
        help="Existing Bittensor wallet directory; required by non-default wallet layouts",
    )
    parser.add_argument("--allow-http-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.allow_http_fixture and os.environ.get("PYTEST_CURRENT_TEST") is None:
        parser.error("--allow-http-fixture is restricted to an explicit test fixture")
    if args.dry_run and not args.live:
        parser.error("--dry-run requires --live")
    if args.live and not args.dry_run and (not args.wallet or not args.hotkey):
        parser.error("--live requires explicit --wallet and --hotkey")
    try:
        # A fresh signed live feed is also valid unsigned review input. Only
        # --live takes the writer lock and constructs wallet/chain boundaries.
        runner = LightRunner(
            args.state_dir, default_public_key(), allow_live=True, writer_lock=args.live
        )
    except (LightError, OSError, sqlite3.Error, ValueError) as error:
        reason = str(error) if isinstance(error, LightError) else "light_unavailable"
        print(reason, flush=True)
        raise SystemExit(1) from error
    publisher = None
    if args.live and not args.dry_run:
        try:
            import bittensor as bt

            from forge.chain import BittensorChain
            from forge.inference.light_publication import LiveJournal, LivePublisher, SdkM0Publisher

            wallet = bt.Wallet(
                name=args.wallet,
                hotkey=args.hotkey,
                path=args.wallet_path or str(Path.home() / ".bittensor" / "wallets"),
            )
            chain = BittensorChain("finney", 126)
            publisher = LivePublisher(
                LiveJournal(args.state_dir / "light.sqlite3"),
                SdkM0Publisher(
                    chain,
                    wallet,
                    wallet.hotkey.ss58_address,
                    lock_dir=Path("/var/lib/attelier-light/publisher-locks"),
                ),
            )
        except Exception as error:
            runner.unavailable("light_live_initialization")
            raise SystemExit(1) from error
    missing_path = args.state_dir / "live-feed-missing.json"
    try:
        outage_state = _strict_json(missing_path.read_bytes()) if missing_path.exists() else {}
    except (OSError, LightError):
        outage_state = {}
    missing = outage_state.get("missingSinceMs")
    if type(missing) is not int:
        missing = None
    while True:
        try:
            raw = fetch_feed(args.url, allow_http_fixture=args.allow_http_fixture)
            artifact = runner.process(raw)
            body = verify_feed(raw, default_public_key(), allow_live=True) if args.live else None
        except (LightError, OSError, RuntimeError, sqlite3.Error, ValueError) as error:
            reason = str(error) if isinstance(error, LightError) else "light_unavailable"
            runner.unavailable(reason)
            print(reason, flush=True)
            # Only a failed fetch/authentication/registry stage starts the
            # outage clock. A live SDK rate-limit or RPC issue is not a missing
            # closed feed and must never select a burn vector.
            now_ms = int(time.time() * 1000)
            if args.live and not args.dry_run:
                if missing is None:
                    missing = now_ms
                    _atomic_write(
                        missing_path,
                        {
                            "schema": "attelier/light-feed-missing/v1",
                            "missingSinceMs": missing,
                            "burnedThroughEndMs": outage_state.get("burnedThroughEndMs"),
                        },
                    )
                if now_ms - missing >= 600_000 and publisher is not None:
                    from forge.inference.light_publication import BURN_OWNER_COLDKEY

                    hour = now_ms // HOUR_MS
                    outage = {
                        "ledger": {
                            "sequence": -hour,
                            "window": {"startMs": (hour - 1) * HOUR_MS, "endMs": hour * HOUR_MS},
                        },
                        "registry": {"subnetOwnerColdkey": BURN_OWNER_COLDKEY, "miners": []},
                    }
                    with contextlib.suppress(ValueError):
                        burned = publisher.process(outage, {"miners": []})
                        if burned["status"] in {"COMMITTED", "REVEAL_PENDING", "REVEALED"}:
                            outage_state = {
                                "schema": "attelier/light-feed-missing/v1",
                                "missingSinceMs": missing,
                                "burnedThroughEndMs": outage["ledger"]["window"]["endMs"],
                            }
                            _atomic_write(missing_path, outage_state)
            if args.once:
                raise SystemExit(1) from error
        else:
            missing = None
            if args.live and not args.dry_run:
                outage_state = {
                    "schema": "attelier/light-feed-missing/v1",
                    "missingSinceMs": None,
                    "burnedThroughEndMs": outage_state.get("burnedThroughEndMs"),
                }
                _atomic_write(missing_path, outage_state)
            if args.live:
                try:
                    assert body is not None
                    if args.dry_run:
                        from forge.inference.light_publication import (
                            eligible_workers,
                            relative_vector,
                        )

                        _atomic_write(
                            args.state_dir / "live-dry-run.json",
                            {
                                "bodyDigest": digest(body),
                                "relativeWeights": relative_vector(eligible_workers(artifact)),
                            },
                        )
                    elif not outage_blocks_service(body, outage_state):
                        assert publisher is not None
                        publisher.process(body, artifact)
                except (OSError, RuntimeError, sqlite3.Error, ValueError) as error:
                    runner.unavailable("light_publication_unavailable")
                    print("light_publication_unavailable", flush=True)
                    if args.once:
                        raise SystemExit(1) from error
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
