"""Signed validator-report ingestion for the dashboard.

This is an observability boundary.  It stores reports that already bind a
frozen local plan; it neither resolves a round nor authorizes a crown, payout,
or weight update.  A deployment operator registers plans before accepting
traffic.  Live chain and raw-evidence verification intentionally remain in the
distributed resolver.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Verifier
from forge.evaluation.distributed import verify_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.protocol.distributed_transport import MAX_REPORT_BYTES, report_digest

SCHEMA = "forge/dashboard-ingestion/v1"
MAX_REQUEST_BYTES = MAX_REPORT_BYTES


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_loads(raw: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    value = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    _validate_json_numbers(value)
    return value


def _validate_json_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_numbers(item)


def load_registered_plans(paths: list[Path]) -> dict[str, DistributedEvaluationPlan]:
    """Load the immutable operator registry.

    Several challengers legitimately share a round's entropy, panel and
    committee.  Their candidate-specific plan hashes remain separate keys.
    """
    plans: dict[str, DistributedEvaluationPlan] = {}
    for path in paths:
        raw = _json_loads(Path(path).read_bytes())
        plan = DistributedEvaluationPlan.from_dict(raw)
        digest = plan.plan_hash()
        if digest in plans and plans[digest].to_dict() != plan.to_dict():
            raise ValueError("registered plans share a hash but differ")
        plans[digest] = plan
    if not plans:
        raise ValueError("at least one registered plan is required")
    return plans


class DashboardReportStore:
    def __init__(
        self,
        database: Path,
        plans: Mapping[str, DistributedEvaluationPlan],
        plan_directory: Path | None = None,
    ):
        self.database = Path(database)
        self.plans = dict(plans)
        self.plan_directory = plan_directory
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.refresh_plans()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS dashboard_reports (
                    plan_hash TEXT NOT NULL,
                    validator_hotkey TEXT NOT NULL,
                    report_hash TEXT NOT NULL,
                    round_id TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    PRIMARY KEY (plan_hash, validator_hotkey)
                )"""
            )

    def refresh_plans(self) -> None:
        """Discover operator-installed plans, retaining accepted history across restarts.

        Directory files must be installed atomically. HTTP clients cannot register
        plans or reveal future panels through this private registry.
        """
        plans = dict(self.plans)
        if self.plan_directory is not None:
            paths = sorted(self.plan_directory.glob("*.json"))
            if paths:
                plans.update(load_registered_plans(paths))
        with closing(self._connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS dashboard_plans "
                "(plan_hash TEXT PRIMARY KEY, plan_json TEXT NOT NULL)"
            )
            connection.execute("BEGIN IMMEDIATE")
            for digest, plan in plans.items():
                connection.execute(
                    "INSERT OR IGNORE INTO dashboard_plans VALUES (?, ?)",
                    (digest, json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False)),
                )
            rows = connection.execute("SELECT plan_hash, plan_json FROM dashboard_plans").fetchall()
            restored = {}
            for digest, encoded in rows:
                plan = DistributedEvaluationPlan.from_dict(json.loads(encoded))
                if plan.plan_hash() != digest:
                    raise ValueError("stored dashboard plan hash mismatch")
                restored[digest] = plan
            connection.execute("COMMIT")
        self.plans = restored

    def ingest(self, report: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        plan_hash = report.get("plan_hash")
        if isinstance(plan_hash, str) and plan_hash not in self.plans:
            self.refresh_plans()
        if not isinstance(plan_hash, str) or plan_hash not in self.plans:
            raise ValueError("report references an unregistered plan")
        plan = self.plans[plan_hash]
        body = verify_partial_report(plan, report, Sr25519Verifier())
        encoded = json.dumps(dict(report), sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = report_digest(report)
        received_at = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT report_hash, received_at FROM dashboard_reports "
                "WHERE plan_hash = ? AND validator_hotkey = ?",
                (plan_hash, body["validator_hotkey"]),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO dashboard_reports (plan_hash, validator_hotkey, report_hash, "
                    "round_id, received_at, report_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        plan_hash,
                        body["validator_hotkey"],
                        digest,
                        plan.round_id,
                        received_at,
                        encoded,
                    ),
                )
                state = "stored"
            elif row[0] == digest:
                received_at = row[1]
                state = "duplicate"
            else:
                connection.execute("ROLLBACK")
                raise FileExistsError(
                    "a different final report already exists for this plan and validator"
                )
            connection.execute("COMMIT")
        return state, self._public_row(plan, body, digest, received_at)

    @staticmethod
    def _public_row(
        plan: DistributedEvaluationPlan, body: Mapping[str, Any], digest: str, received_at: str
    ) -> dict[str, Any]:
        return {
            "round_id": plan.round_id,
            "plan_hash": plan.plan_hash(),
            "validator_hotkey": body["validator_hotkey"],
            "report_hash": digest,
            "quality_prompts": len(body["quality_vs_king"]),
            "timing_prompts": len(body["timing"]),
            "received_at": received_at,
            "signature_verified": True,
            "chain_verified": False,
            "evidence_replayed": False,
            "status": "RECEIVED",
        }

    def public_snapshot(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT plan_hash, report_hash, received_at, report_json FROM dashboard_reports "
                "ORDER BY received_at, report_hash"
            ).fetchall()
        reports = []
        for plan_hash, digest, received_at, encoded in rows:
            plan = self.plans.get(plan_hash)
            if plan is None:
                continue
            report = json.loads(encoded)
            body = {key: value for key, value in report.items() if key != "signature"}
            reports.append(self._public_row(plan, body, digest, received_at))
        return {"schema": SCHEMA, "updated_at": _now(), "reports": reports}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, store: DashboardReportStore, snapshot: Path | None):
        self.store, self.snapshot = store, snapshot
        super().__init__(address, handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(15)
        return request, address


class DashboardIngestionHandler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _send(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _body(self) -> bytes:
        if self.headers.get_all("Transfer-Encoding"):
            raise ValueError("transfer encoding is not accepted")
        lengths = self.headers.get_all("Content-Length")
        if lengths is None or len(lengths) != 1 or not lengths[0].isdigit():
            raise ValueError("request body is missing or exceeds size limit")
        length = int(lengths[0])
        if length > MAX_REQUEST_BYTES:
            raise ValueError("request body is missing or exceeds size limit")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("request body ended before its declared length")
        return body

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/v1/validator-reports":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            state, row = self.server.store.ingest(_json_loads(self._body()))
        except FileExistsError as exc:
            self._send(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except sqlite3.Error:
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "storage unavailable"})
        else:
            self._send(HTTPStatus.OK, {"status": state, "report": row})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/v1/validator-reports":
            self._send(HTTPStatus.OK, self.server.store.public_snapshot())
            return
        if self.path == "/data/status.json" and self.server.snapshot is not None:
            try:
                snapshot = _json_loads(self.server.snapshot.read_bytes())
                if snapshot.get("schema") != "forge/dashboard/v2" or "_fixture" in snapshot:
                    raise ValueError("snapshot is not a current dashboard artifact")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "snapshot unavailable"})
            else:
                self._send(HTTPStatus.OK, snapshot)
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})


def serve(
    database: Path,
    plans: list[Path],
    host: str,
    port: int,
    snapshot: Path | None = None,
    plan_directory: Path | None = None,
) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("dashboard ingestion only binds loopback; put a reverse proxy in front")
    if not plans and plan_directory is None:
        raise ValueError("registered plan files or a plan directory are required")
    if plan_directory is not None and not plan_directory.is_dir():
        raise ValueError("registered plan directory does not exist")
    store = DashboardReportStore(
        database, load_registered_plans(plans) if plans else {}, plan_directory
    )
    server = _Server((host, port), DashboardIngestionHandler, store, snapshot)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
