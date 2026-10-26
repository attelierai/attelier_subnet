"""Per-candidate validator budget reservation with uncertain-charge containment."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


class BudgetError(ValueError):
    pass


class UncertainChargeError(BudgetError):
    pass


@dataclass(frozen=True)
class BudgetPolicy:
    """Bound one validator and the whole explicitly authorized campaign.

    Ten USD remains the conservative per-validator default.  A larger local
    allocation is valid only when the caller records a distinct campaign cap;
    this prevents an accidental higher CLI value from becoming spending
    authority.
    """

    candidate_limit_usd: float = 10.0
    session_limit_usd: float = 10.0
    session_quota_usd: float | None = None
    campaign_limit_usd: float | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_limit_usd, "candidate budget"),
            (self.session_limit_usd, "session budget"),
        ):
            if (
                type(value) not in {int, float}
                or isinstance(value, bool)
                or not isfinite(value)
                or value <= 0
            ):
                raise BudgetError(f"{label} must be a finite positive number")
        if self.campaign_limit_usd is not None and (
            type(self.campaign_limit_usd) not in {int, float}
            or isinstance(self.campaign_limit_usd, bool)
            or not isfinite(self.campaign_limit_usd)
            or self.campaign_limit_usd <= 0
        ):
            raise BudgetError("campaign budget must be a finite positive number")
        if self.candidate_limit_usd > 10.0 and self.campaign_limit_usd is None:
            raise BudgetError(
                "a validator budget above the 10 USD default requires an explicit campaign budget"
            )
        if (
            self.campaign_limit_usd is not None
            and self.candidate_limit_usd > self.campaign_limit_usd
        ):
            raise BudgetError("validator budget cannot exceed the authorized campaign budget")
        if self.session_quota_usd is not None and (
            type(self.session_quota_usd) not in {int, float}
            or isinstance(self.session_quota_usd, bool)
            or not isfinite(self.session_quota_usd)
            or self.session_quota_usd < 0
        ):
            raise BudgetError("session quota must be a finite nonnegative number")

    @property
    def effective_campaign_limit_usd(self) -> float:
        """Apply every declared global cap; a campaign cannot expand a legacy cap."""
        if self.campaign_limit_usd is None:
            return self.session_limit_usd
        return min(self.session_limit_usd, self.campaign_limit_usd)


@dataclass(frozen=True)
class Reservation:
    id: int
    candidate_id: int
    operation_key: str
    amount_usd: float
    status: str
    provider_receipt_id: str | None


@dataclass(frozen=True)
class BudgetSummary:
    candidate_reserved_usd: float
    session_reserved_usd: float
    candidate_limit_usd: float
    session_limit_usd: float
    warning_at_80_percent: bool
    campaign_limit_usd: float | None = None


class BudgetLedger:
    def __init__(self, path: Path | str, policy: BudgetPolicy | None = None) -> None:
        self.path, self.policy = Path(path), policy or BudgetPolicy()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget_reservations ("
                "id INTEGER PRIMARY KEY, candidate_id INTEGER NOT NULL, "
                "operation_key TEXT NOT NULL, "
                "amount_usd REAL NOT NULL, status TEXT NOT NULL, provider_receipt_id TEXT, "
                "UNIQUE(candidate_id, operation_key))"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def reserve(self, candidate_id: int, operation_key: str, amount_usd: float) -> Reservation:
        if (
            type(candidate_id) is not int
            or candidate_id <= 0
            or not operation_key
            or type(amount_usd) not in {int, float}
            or isinstance(amount_usd, bool)
            or not isfinite(amount_usd)
            or amount_usd <= 0
        ):
            raise BudgetError("candidate, operation key, and positive reservation are required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM budget_reservations WHERE candidate_id=? AND operation_key=?",
                (candidate_id, operation_key),
            ).fetchone()
            if existing:
                db.commit()
                if abs(existing["amount_usd"] - amount_usd) > 1e-9:
                    raise BudgetError("operation key cannot change its reserved amount")
                if existing["status"] == "UNCERTAIN":
                    raise UncertainChargeError(
                        "charge outcome is uncertain; do not retry this operation"
                    )
                raise BudgetError("operation was already captured; do not retry it")
            uncertain = db.execute(
                "SELECT 1 FROM budget_reservations WHERE candidate_id=? AND status='UNCERTAIN'",
                (candidate_id,),
            ).fetchone()
            if uncertain:
                raise UncertainChargeError(
                    "candidate has an uncertain charge; reconcile it before additional work"
                )
            total = db.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM budget_reservations "
                "WHERE candidate_id=? AND status IN ('RESERVED', 'CAPTURED', 'UNCERTAIN')",
                (candidate_id,),
            ).fetchone()[0]
            if total + amount_usd > self.policy.candidate_limit_usd + 1e-9:
                raise BudgetError("candidate budget insufficient before additional work")
            session_total = db.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM budget_reservations "
                "WHERE status IN ('RESERVED', 'CAPTURED', 'UNCERTAIN')"
            ).fetchone()[0]
            session_cap = self.policy.effective_campaign_limit_usd
            if self.policy.session_quota_usd is not None:
                session_cap = min(session_cap, self.policy.session_quota_usd)
            if session_total + amount_usd > session_cap + 1e-9:
                raise BudgetError("session budget or quota insufficient before additional work")
            cursor = db.execute(
                "INSERT INTO budget_reservations(candidate_id, operation_key, amount_usd, status) "
                "VALUES (?, ?, ?, 'RESERVED')",
                (candidate_id, operation_key, amount_usd),
            )
            row = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (cursor.lastrowid,)
            ).fetchone()
            db.commit()
            return Reservation(**dict(row))

    def require_active(
        self, reservation_id: int, *, candidate_id: int, operation_key: str, minimum_usd: float
    ) -> Reservation:
        """Read a durable reservation immediately before validator work starts."""
        if (
            type(reservation_id) is not int
            or type(candidate_id) is not int
            or candidate_id <= 0
            or not isinstance(operation_key, str)
            or not operation_key
            or type(minimum_usd) not in {int, float}
            or isinstance(minimum_usd, bool)
            or not isfinite(minimum_usd)
            or minimum_usd <= 0
        ):
            raise BudgetError("active reservation identity and finite minimum are required")
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
        if (
            row is None
            or row["status"] != "RESERVED"
            or row["candidate_id"] != candidate_id
            or row["operation_key"] != operation_key
            or not isfinite(row["amount_usd"])
            or row["amount_usd"] < minimum_usd
        ):
            raise BudgetError("durable active reservation is absent or insufficient")
        return Reservation(**dict(row))

    def settle(self, reservation_id: int, actual_usd: float) -> Reservation:
        if (
            type(actual_usd) not in {int, float}
            or isinstance(actual_usd, bool)
            or not isfinite(actual_usd)
            or actual_usd < 0
        ):
            raise BudgetError("actual charge must be finite and nonnegative")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            if row is None or row["status"] != "RESERVED":
                raise BudgetError("only a reserved charge can settle")
            if actual_usd > row["amount_usd"] + 1e-9:
                raise BudgetError("actual charge exceeds its pre-reserved amount")
            db.execute(
                "UPDATE budget_reservations SET amount_usd=?, status='CAPTURED' WHERE id=?",
                (actual_usd, reservation_id),
            )
            result = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            db.commit()
            return Reservation(**dict(result))

    def reconcile_uncertain(
        self, reservation_id: int, actual_usd: float, *, provider_receipt_id: str
    ) -> Reservation:
        if not isinstance(provider_receipt_id, str) or not provider_receipt_id:
            raise BudgetError("provider reconciliation receipt is required")
        if (
            type(actual_usd) not in {int, float}
            or isinstance(actual_usd, bool)
            or not isfinite(actual_usd)
            or actual_usd < 0
        ):
            raise BudgetError("actual charge must be finite and nonnegative")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            if row is None or row["status"] != "UNCERTAIN":
                raise BudgetError("only an uncertain charge can reconcile")
            if actual_usd > row["amount_usd"] + 1e-9:
                raise BudgetError("reconciled charge exceeds its pre-reserved amount")
            db.execute(
                "UPDATE budget_reservations SET amount_usd=?, status='CAPTURED', "
                "provider_receipt_id=? WHERE id=?",
                (actual_usd, provider_receipt_id, reservation_id),
            )
            result = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            db.commit()
            return Reservation(**dict(result))

    def summary(self, candidate_id: int) -> BudgetSummary:
        with self._connect() as db:
            candidate = db.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM budget_reservations "
                "WHERE candidate_id=? AND status IN ('RESERVED', 'CAPTURED', 'UNCERTAIN')",
                (candidate_id,),
            ).fetchone()[0]
            session = db.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM budget_reservations "
                "WHERE status IN ('RESERVED', 'CAPTURED', 'UNCERTAIN')"
            ).fetchone()[0]
        session_cap = self.policy.effective_campaign_limit_usd
        if self.policy.session_quota_usd is not None:
            session_cap = min(session_cap, self.policy.session_quota_usd)
        return BudgetSummary(
            candidate,
            session,
            self.policy.candidate_limit_usd,
            session_cap,
            candidate >= 0.8 * self.policy.candidate_limit_usd or session >= 0.8 * session_cap,
            self.policy.effective_campaign_limit_usd,
        )

    def mark_uncertain(self, reservation_id: int) -> Reservation:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            if row is None or row["status"] != "RESERVED":
                raise BudgetError("only a reserved charge can become uncertain")
            db.execute(
                "UPDATE budget_reservations SET status='UNCERTAIN' WHERE id=?", (reservation_id,)
            )
            result = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            db.commit()
            return Reservation(**dict(result))

    def release(self, reservation_id: int) -> Reservation:
        """Release a reservation only before an external provider receives work."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            if row is None or row["status"] != "RESERVED":
                raise BudgetError("only an unstarted reservation can release")
            db.execute(
                "UPDATE budget_reservations SET status='RELEASED' WHERE id=?", (reservation_id,)
            )
            result = db.execute(
                "SELECT * FROM budget_reservations WHERE id=?", (reservation_id,)
            ).fetchone()
            db.commit()
            return Reservation(**dict(result))
