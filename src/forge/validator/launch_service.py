"""Durable serial campaign state, separate from GPU workers and signers.

One SQLite journal owns admission selection and frozen worker requests. A worker
must reconcile an existing request by its hash after a restart; the service never
interprets process failure as a scientific result or a reason to draw new prompts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from forge.protocol.hashing import hash_canonical
from forge.validator.budget import BudgetLedger, BudgetPolicy
from forge.validator.launch_config import LaunchConfig, require_hash, verify_selection

TERMINAL = ("COMPLETE", "EXPIRED", "INCOMPLETE")
_ACTIVE = ("WAITING_ENTROPY", "READY", "REQUESTED", "AWAITING_CONSENSUS", "CERTIFIED")


def _storage_json_bytes(value: object) -> bytes:
    """Stable, browser/SQLite JSON; canonical encoding is reserved for hashes.

    ``canonical_json_bytes`` represents floats as tagged objects so hashes stay
    unambiguous.  Service journals must instead preserve their JSON value shape:
    frozen V2 plans and evaluation results contain numeric thresholds and scores.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    """Compare JSON values on replay, preserving the original persisted bytes."""
    encoded = _storage_json_bytes(value)
    if path.exists():
        if _storage_json_bytes(json.loads(path.read_text())) != encoded:
            raise ValueError(f"immutable receipt changed: {path.name}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_storage_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class LaunchServiceStore:
    def __init__(self, path: Path, config: LaunchConfig):
        self.path, self.config = Path(path), config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = BudgetLedger(
            path,
            BudgetPolicy(
                candidate_limit_usd=config.campaign_limit_usd - config.closure_reserve_usd,
                session_limit_usd=config.campaign_limit_usd - config.closure_reserve_usd,
                campaign_limit_usd=config.campaign_limit_usd,
            ),
        )
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS launch_campaign (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    config_hash TEXT NOT NULL, config_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS launch_service_rounds (
                    id INTEGER PRIMARY KEY, round_id TEXT UNIQUE NOT NULL,
                    close_block INTEGER UNIQUE NOT NULL, status TEXT NOT NULL,
                    selection_json TEXT NOT NULL, entropy_hash TEXT,
                    plan_json TEXT, plan_hash TEXT, reservation_id INTEGER,
                    request_hash TEXT, request_json TEXT, result_json TEXT,
                    certification_json TEXT, publication_json TEXT, terminal_reason TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS launch_single_active_round
                    ON launch_service_rounds((1))
                    WHERE status NOT IN ('COMPLETE','EXPIRED','INCOMPLETE');
                CREATE TABLE IF NOT EXISTS launch_effective_admissions (
                    round_id TEXT PRIMARY KEY,
                    receipt_json TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL,
                    effective_hash TEXT NOT NULL,
                    legacy_effective_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS launch_effective_admission_aliases (
                    effective_hash TEXT PRIMARY KEY,
                    round_id TEXT NOT NULL,
                    alias_kind TEXT NOT NULL CHECK(alias_kind IN ('effective','legacy'))
                );
                CREATE TABLE IF NOT EXISTS launch_budget_subjects (
                    round_id TEXT PRIMARY KEY,
                    candidate_id INTEGER NOT NULL UNIQUE
                );
                """
            )
            self._migrate_round_columns(db)
            prior = db.execute("SELECT config_hash FROM launch_campaign").fetchone()
            if prior and prior[0] != config.config_hash():
                raise ValueError("service state belongs to another frozen campaign")
            db.execute(
                "INSERT OR IGNORE INTO launch_campaign VALUES (1, ?, ?)",
                (config.config_hash(), _storage_json_bytes(config.to_dict()).decode()),
            )

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def select(self, selection: dict[str, Any], *, finalized_block: int) -> str:
        verify_selection(
            self.config,
            selection,
            entropy_block=selection["entropy_block"],
            entropy_hash="0" * 64,
        )
        encoded = _storage_json_bytes(selection).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT selection_json FROM launch_service_rounds WHERE round_id=?",
                (selection["round_id"],),
            ).fetchone()
            if prior:
                if prior[0] != encoded:
                    raise ValueError("round selection changed")
                return selection["round_id"]
            if not selection["close_block"] <= finalized_block < selection["entropy_block"]:
                raise ValueError("new selection must be persisted before future entropy")
            try:
                db.execute(
                    "INSERT INTO launch_service_rounds "
                    "(round_id,close_block,status,selection_json) VALUES (?,?,'WAITING_ENTROPY',?)",
                    (selection["round_id"], selection["close_block"], encoded),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "another round is active or this close was already consumed"
                ) from error
        return selection["round_id"]

    def freeze(
        self,
        round_id: str,
        *,
        finalized_block: int,
        entropy_hash: str,
        plan: dict[str, Any],
    ) -> str:
        """The trusted planner supplies a replay-validated policy plan."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, round_id)
            selection = json.loads(row["selection_json"])
            if finalized_block < selection["entropy_block"]:
                raise ValueError("entropy is not finalized")
            entropy_hash = verify_selection(
                self.config,
                selection,
                entropy_block=selection["entropy_block"],
                entropy_hash=entropy_hash,
            )
            from forge.evaluation.launch_policy import validate_plan

            validate_plan(
                plan,
                expected_entropy_hash=entropy_hash,
                expected_round_id=selection["round_id"],
            )
            if (
                plan.get("models", {}).get("candidate") != selection["candidate_manifest_hash"]
                or plan.get("schema") != self.config.policy_schema
            ):
                raise ValueError("worker plan differs from the selected candidate or policy")
            encoded = _storage_json_bytes(plan).decode()
            digest = hash_canonical("forge/launch-service-plan/v1", plan)
            if row["plan_json"] is not None:
                if row["plan_json"] != encoded or row["entropy_hash"] != entropy_hash:
                    raise ValueError("frozen panel cannot change on resume")
                return digest
            if row["status"] != "WAITING_ENTROPY":
                raise ValueError("only a pending selection can freeze")
            db.execute(
                "UPDATE launch_service_rounds SET entropy_hash=?, plan_json=?, plan_hash=?, "
                "status='READY' WHERE round_id=?",
                (entropy_hash, encoded, digest, round_id),
            )
            return digest

    def request_work(
        self, round_id: str, *, reserve_usd: float, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist a budgeted idempotency key before any external GPU dispatch."""
        with self._db() as db:
            row = self._row(db, round_id)
        if row["status"] not in {"READY", "REQUESTED"}:
            raise ValueError("round is not ready for GPU work")
        if request.get("plan_hash") != row["plan_hash"] or request.get("round_id") != round_id:
            raise ValueError("worker request is not bound to its frozen plan")
        digest = hash_canonical("forge/launch-worker-request/v1", request)
        operation = f"launch:{round_id}"
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            subject = db.execute(
                "SELECT candidate_id FROM launch_budget_subjects WHERE round_id=?", (round_id,)
            ).fetchone()
            existing = db.execute(
                "SELECT id,candidate_id FROM budget_reservations WHERE operation_key=?",
                (operation,),
            ).fetchall()
            if len(existing) > 1:
                raise ValueError("multiple durable reservations claim this launch round")
            if subject is None:
                if existing:
                    candidate_id = int(existing[0]["candidate_id"])
                else:
                    maximum = db.execute(
                        "SELECT MAX(candidate_id) FROM ("
                        "SELECT candidate_id FROM budget_reservations UNION ALL "
                        "SELECT candidate_id FROM launch_budget_subjects)"
                    ).fetchone()[0]
                    candidate_id = int(maximum or 0) + 1
                db.execute(
                    "INSERT INTO launch_budget_subjects(round_id,candidate_id) VALUES (?,?)",
                    (round_id, candidate_id),
                )
            else:
                candidate_id = int(subject[0])
            if existing and int(existing[0]["candidate_id"]) != candidate_id:
                raise ValueError("launch budget subject differs from its durable reservation")
            if row["reservation_id"] is not None and (
                not existing or int(existing[0]["id"]) != row["reservation_id"]
            ):
                raise ValueError("launch round references another budget reservation")
            reservation = db.execute(
                "SELECT id FROM budget_reservations WHERE candidate_id=? AND operation_key=?",
                (candidate_id, operation),
            ).fetchone()
        if reservation:
            reserved = self.budget.require_active(
                reservation[0],
                candidate_id=candidate_id,
                operation_key=operation,
                minimum_usd=reserve_usd,
            )
            if reserved.amount_usd != reserve_usd:
                raise ValueError("frozen reservation cannot change")
        else:
            reserved = self.budget.reserve(candidate_id, operation, reserve_usd)
        encoded = _storage_json_bytes(request).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._row(db, round_id)
            if current["request_hash"] is not None and current["request_hash"] != digest:
                raise ValueError("external request changed; reconcile the original")
            if current["status"] not in {"READY", "REQUESTED"}:
                raise ValueError("round ended before dispatch; preserve reservation")
            db.execute(
                "UPDATE launch_service_rounds SET status='REQUESTED',reservation_id=?,"
                "request_hash=?,request_json=? WHERE round_id=?",
                (reserved.id, digest, encoded, round_id),
            )
        return {"request_hash": digest, "reservation_id": reserved.id, "request": request}

    def work_request(self, round_id: str) -> dict[str, Any]:
        """Recover the original reserved envelope after a process interruption."""
        with self._db() as db:
            row = self._row(db, round_id)
        if row["request_json"] is None or row["reservation_id"] is None:
            raise ValueError("round has no reserved worker request")
        request = json.loads(row["request_json"])
        if hash_canonical("forge/launch-worker-request/v1", request) != row["request_hash"]:
            raise ValueError("stored worker request hash changed")
        return {
            "request_hash": row["request_hash"],
            "reservation_id": row["reservation_id"],
            "request": request,
        }

    def record_effective_admission(self, round_id: str, receipt: dict[str, Any]) -> bool:
        """Reserve the candidate's effective weights before GPU dispatch.

        The launch service has its own durable alias registry because a runtime
        preflight must reject duplicate effective weights before it can spend
        campaign budget.  This is deliberately separate from the historical
        ``AdmissionLedger`` schema and remains across closed launch windows.
        """

        required = {
            "schema",
            "plan_id",
            "contract_hash",
            "manifest_hash",
            "effective_hash",
            "legacy_effective_hash",
            "weight_content_hash",
        }
        if (
            set(receipt) != required
            or receipt.get("schema") != "forge/launch-effective-admission/v1"
        ):
            raise ValueError("effective admission receipt fields are invalid")
        evidence = dict(receipt)
        for name in (
            "plan_id",
            "contract_hash",
            "manifest_hash",
            "effective_hash",
            "weight_content_hash",
        ):
            evidence[name] = require_hash(evidence[name], name.replace("_", " "))
        legacy = evidence["legacy_effective_hash"]
        if legacy is not None:
            evidence["legacy_effective_hash"] = require_hash(legacy, "legacy effective hash")
        encoded = _storage_json_bytes(evidence).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, round_id)
            if row["status"] != "REQUESTED":
                raise ValueError("effective admission requires a requested GPU round")
            plan = json.loads(row["plan_json"]) if row["plan_json"] is not None else None
            if (
                not isinstance(plan, dict)
                or evidence["plan_id"] != plan.get("plan_id")
                or evidence["contract_hash"] != self.config.contract_hash
                or evidence["manifest_hash"] != plan.get("models", {}).get("candidate")
            ):
                raise ValueError("effective admission differs from the frozen launch plan")
            existing = db.execute(
                "SELECT receipt_json FROM launch_effective_admissions WHERE round_id=?", (round_id,)
            ).fetchone()
            if existing is not None:
                if existing["receipt_json"] != encoded:
                    raise ValueError("effective admission receipt cannot change on resume")
                return False
            aliases = [(evidence["effective_hash"], "effective")]
            if (
                evidence["legacy_effective_hash"] is not None
                and evidence["legacy_effective_hash"] != evidence["effective_hash"]
            ):
                aliases.append((evidence["legacy_effective_hash"], "legacy"))
            for alias, _ in aliases:
                prior = db.execute(
                    "SELECT round_id FROM launch_effective_admission_aliases "
                    "WHERE effective_hash=?",
                    (alias,),
                ).fetchone()
                if prior is not None:
                    raise ValueError(
                        "effective candidate is already admitted in another launch round"
                    )
            db.execute(
                "INSERT INTO launch_effective_admissions "
                "(round_id,receipt_json,manifest_hash,effective_hash,legacy_effective_hash) "
                "VALUES (?,?,?,?,?)",
                (
                    round_id,
                    encoded,
                    evidence["manifest_hash"],
                    evidence["effective_hash"],
                    evidence["legacy_effective_hash"],
                ),
            )
            for alias, kind in aliases:
                db.execute(
                    "INSERT INTO launch_effective_admission_aliases "
                    "(effective_hash,round_id,alias_kind) VALUES (?,?,?)",
                    (alias, round_id, kind),
                )
        return True

    def effective_admission(self, round_id: str) -> dict[str, Any] | None:
        """Return the immutable pre-dispatch admission receipt, if recorded."""

        with self._db() as db:
            self._row(db, round_id)
            row = db.execute(
                "SELECT receipt_json FROM launch_effective_admissions WHERE round_id=?", (round_id,)
            ).fetchone()
        return None if row is None else json.loads(row["receipt_json"])

    def record_evaluation(
        self, round_id: str, *, request_hash: str, result: dict[str, Any]
    ) -> None:
        """Keep evaluation and consensus separate; a PASS never creates a crown."""
        require_hash(request_hash, "request hash")
        if result.get("status") not in {"PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE"}:
            raise ValueError("unknown evaluation outcome")
        encoded = _storage_json_bytes(result).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, round_id)
            if row["request_hash"] != request_hash:
                raise ValueError("worker result belongs to another request")
            if row["result_json"] is not None:
                if row["result_json"] != encoded:
                    raise ValueError("completed evidence cannot be replaced")
                return
            if row["status"] != "REQUESTED":
                raise ValueError("round is not awaiting worker evidence")
            target = "INCOMPLETE" if result["status"] == "INCOMPLETE" else "AWAITING_CONSENSUS"
            db.execute(
                "UPDATE launch_service_rounds SET status=?,result_json=? WHERE round_id=?",
                (target, encoded, round_id),
            )

    def settle_work(self, round_id: str, *, actual_usd: float) -> None:
        """Settle the reserved worker cost after a terminal worker receipt.

        This records provider cost only; it neither certifies the evaluation nor
        permits another external request for the same frozen round.
        """

        with self._db() as db:
            row = self._row(db, round_id)
        if row["status"] not in {"AWAITING_CONSENSUS", "INCOMPLETE"}:
            raise ValueError("worker cost can settle only after a stored evaluation")
        if row["reservation_id"] is None:
            raise ValueError("round has no durable worker reservation")
        reservation = self.budget.settle(row["reservation_id"], actual_usd)
        if reservation.status != "CAPTURED":  # defensive against future ledger changes
            raise ValueError("worker cost was not captured")

    def record_certification(
        self,
        round_id: str,
        *,
        certificate_hash: str,
        certified_state_hash: str,
        plan_hash: str,
    ) -> None:
        """Persist an already externally verified certificate reference.

        The service has no signer or committee verifier and deliberately does
        not claim that these hashes constitute certification by themselves.
        Its caller must have completed that authority-bearing verification.
        """

        evidence = {
            "certificate_hash": require_hash(certificate_hash, "certificate hash"),
            "certified_state_hash": require_hash(certified_state_hash, "certified state hash"),
            "plan_hash": require_hash(plan_hash, "certification plan hash"),
        }
        encoded = _storage_json_bytes(evidence).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, round_id)
            if row["certification_json"] is not None:
                if row["certification_json"] != encoded:
                    raise ValueError("certification evidence cannot change on resume")
                db.commit()
                return
            if row["status"] != "AWAITING_CONSENSUS":
                raise ValueError("only a completed evaluation can await certification")
            if evidence["plan_hash"] != row["plan_hash"]:
                raise ValueError("certification does not bind the frozen service plan")
            db.execute(
                "UPDATE launch_service_rounds SET status='CERTIFIED', certification_json=? "
                "WHERE round_id=?",
                (encoded, round_id),
            )

    def record_publication(self, round_id: str, *, receipt: dict[str, Any]) -> None:
        """Persist an external publication/readback receipt without submitting it."""

        allowed = {"SUBMITTED", "FINALIZED"}
        if receipt.get("status") not in allowed:
            raise ValueError("publication receipt status is unsupported")
        required = {"status", "certificate_hash", "intent_hash", "submitted_extrinsic_hash"}
        if receipt["status"] == "FINALIZED":
            required |= {"finalized_block", "readback_hash"}
        if set(receipt) != required:
            raise ValueError("publication receipt fields are invalid")
        evidence = dict(receipt)
        for name in (
            "certificate_hash",
            "intent_hash",
            "submitted_extrinsic_hash",
            "readback_hash",
        ):
            if name in evidence:
                evidence[name] = require_hash(evidence[name], name)
        if "finalized_block" in evidence and (
            type(evidence["finalized_block"]) is not int or evidence["finalized_block"] < 0
        ):
            raise ValueError("finalized block is invalid")
        encoded = _storage_json_bytes(evidence).decode()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, round_id)
            if row["publication_json"] is not None:
                prior = json.loads(row["publication_json"])
                if prior == evidence:
                    db.commit()
                    return
                if not (
                    prior.get("status") == "SUBMITTED"
                    and evidence["status"] == "FINALIZED"
                    and prior.get("intent_hash") == evidence["intent_hash"]
                    and prior.get("submitted_extrinsic_hash")
                    == evidence["submitted_extrinsic_hash"]
                ):
                    raise ValueError("publication receipt cannot change on resume")
            if row["status"] not in {"CERTIFIED", "COMPLETE"}:
                raise ValueError("publication requires externally verified certification")
            certification = json.loads(row["certification_json"])
            if evidence["certificate_hash"] != certification["certificate_hash"]:
                raise ValueError("publication receipt belongs to another certificate")
            target = "COMPLETE" if evidence["status"] == "FINALIZED" else "CERTIFIED"
            db.execute(
                "UPDATE launch_service_rounds SET status=?, publication_json=? WHERE round_id=?",
                (target, encoded, round_id),
            )

    def expire(self, round_id: str, *, finalized_block: int, reason: str) -> None:
        """Close an uncompleted round without manufacturing evaluation or consensus."""

        if (
            type(finalized_block) is not int
            or finalized_block < 0
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError("finalized expiry block and reason are required")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, round_id)
            if row["status"] == "EXPIRED":
                if row["terminal_reason"] != reason:
                    raise ValueError("expired round reason cannot change on resume")
                db.commit()
                return
            if row["status"] not in _ACTIVE:
                raise ValueError("only an active round can expire")
            db.execute(
                "UPDATE launch_service_rounds SET status='EXPIRED', terminal_reason=? "
                "WHERE round_id=?",
                (reason, round_id),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM launch_service_rounds ORDER BY id").fetchall()
            exposure = db.execute(
                "SELECT COALESCE(SUM(amount_usd),0) FROM budget_reservations "
                "WHERE status IN ('RESERVED','CAPTURED','UNCERTAIN')"
            ).fetchone()[0]
        return {
            "schema": "forge/launch-service/v1",
            "network": self.config.network,
            "netuid": self.config.netuid,
            "config_hash": self.config.config_hash(),
            "budget": {
                "reserved_or_captured_usd": exposure,
                "limit_usd": self.config.campaign_limit_usd,
                "closure_reserve_usd": self.config.closure_reserve_usd,
                "warning_at_80_percent": exposure >= 0.8 * self.config.campaign_limit_usd,
            },
            "rounds": [
                {
                    "round_id": row["round_id"],
                    "status": row["status"],
                    "selection": json.loads(row["selection_json"]),
                    "plan_hash": row["plan_hash"],
                    "request_hash": row["request_hash"],
                    "evaluation": json.loads(row["result_json"]) if row["result_json"] else None,
                    "certification": json.loads(row["certification_json"])
                    if row["certification_json"]
                    else None,
                    "publication": json.loads(row["publication_json"])
                    if row["publication_json"]
                    else None,
                    "terminal_reason": row["terminal_reason"],
                }
                for row in rows
            ],
        }

    @staticmethod
    def _row(db: sqlite3.Connection, round_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM launch_service_rounds WHERE round_id=?", (round_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown campaign round")
        return row

    @staticmethod
    def _migrate_round_columns(db: sqlite3.Connection) -> None:
        """Additive local journal migration; never rewrites frozen rows."""

        columns = {row["name"] for row in db.execute("PRAGMA table_info(launch_service_rounds)")}
        for name in ("certification_json", "publication_json", "terminal_reason"):
            if name not in columns:
                db.execute(f"ALTER TABLE launch_service_rounds ADD COLUMN {name} TEXT")
