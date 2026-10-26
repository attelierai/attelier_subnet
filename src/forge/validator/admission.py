"""Fail-closed, persistent FIFO admission for public model commitments.

The local SQLite database is an operational queue, never evidence that a
hotkey has not participated before.  Each admission call therefore requires a
separately authenticated certified-history snapshot.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from forge.protocol.hashing import hash_canonical
from forge.validator.admission_storage import (
    migrate_admission_windows,
    migrate_full_evaluation_claims,
)


class AdmissionError(ValueError):
    """A candidate cannot enter or move through the validator queue."""


_HEX = set("0123456789abcdef")
_TERMINAL = {"REJECTED", "FAILED", "INCONCLUSIVE", "CERTIFIED", "INCOMPLETE"}
_STATUSES = {"QUEUED", "SCREENING", "FULL_EVALUATION", "INFRASTRUCTURE_FAILED", *_TERMINAL}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdmissionError(f"{name} is required")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise AdmissionError(f"{name} must be a lowercase SHA-256")
    return value


def _block(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AdmissionError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class FinalizedCommitment:
    """A model commitment read from finalized chain state, before GPU work."""

    hotkey: str
    block: int
    extrinsic_index: int
    commitment_hash: str
    artifact_hash: str
    recipe_hash: str
    panel_hash: str

    def __post_init__(self) -> None:
        _text(self.hotkey, "hotkey")
        _block(self.block, "commitment block")
        _block(self.extrinsic_index, "commitment extrinsic index")
        for field in ("commitment_hash", "artifact_hash", "recipe_hash", "panel_hash"):
            _hash(getattr(self, field), field)


@dataclass(frozen=True)
class AuthenticatedAdmissionHistory:
    """An external certified history checkpoint, not a local database export.

    ``consumed_hotkeys`` contains the known lineage decision at its finalized
    height.  A caller must pass an independent verifier to ``admit``; carrying
    this object alone deliberately grants no authority.
    """

    network: str
    netuid: int
    through_block: int
    certified_state_hash: str
    history_root: str
    consumed_hotkeys: tuple[str, ...]
    registered_hotkeys: tuple[str, ...]
    lineage_edges: tuple[tuple[str, str], ...] = ()
    window_id: str | None = None
    window_open_block: int | None = None
    window_close_block: int | None = None

    def __post_init__(self) -> None:
        _text(self.network, "network")
        _block(self.netuid, "netuid")
        _block(self.through_block, "history through block")
        _hash(self.certified_state_hash, "certified state hash")
        for values, label in (
            (self.consumed_hotkeys, "certified consumed hotkeys"),
            (self.registered_hotkeys, "certified registered hotkeys"),
        ):
            if tuple(sorted(values)) != values or len(values) != len(set(values)):
                raise AdmissionError(f"{label} must be sorted and unique")
            for hotkey in values:
                _text(hotkey, label)
        if tuple(sorted(self.lineage_edges)) != self.lineage_edges or len(
            self.lineage_edges
        ) != len(set(self.lineage_edges)):
            raise AdmissionError("certified lineage edges must be sorted and unique")
        for old, new in self.lineage_edges:
            _text(old, "lineage old hotkey")
            _text(new, "lineage new hotkey")
            if old == new:
                raise AdmissionError("lineage edge cannot point to itself")
        if self.history_root != self.recomputed_root():
            raise AdmissionError("history root does not match its canonical checkpoint")
        scoped = (self.window_id, self.window_open_block, self.window_close_block)
        if any(value is not None for value in scoped):
            if not isinstance(self.window_id, str) or not self.window_id:
                raise AdmissionError("admission window id is required")
            _block(self.window_open_block, "admission window open block")
            _block(self.window_close_block, "admission window close block")
            if self.window_open_block >= self.window_close_block:
                raise AdmissionError("admission window must have a positive block range")
            if self.through_block < self.window_close_block:
                raise AdmissionError("certified history predates its admission window close")

    def recomputed_root(self) -> str:
        payload = {
            "network": self.network,
            "netuid": self.netuid,
            "through_block": self.through_block,
            "certified_state_hash": self.certified_state_hash,
            "consumed_hotkeys": self.consumed_hotkeys,
            "registered_hotkeys": self.registered_hotkeys,
            "lineage_edges": self.lineage_edges,
        }
        if self.window_id is not None:
            payload.update(
                {
                    "window_id": self.window_id,
                    "window_open_block": self.window_open_block,
                    "window_close_block": self.window_close_block,
                }
            )
        return hash_canonical(
            "forge/validator-admission-history/v2"
            if self.window_id is not None
            else "forge/validator-admission-history/v1",
            payload,
        )

    @property
    def is_windowed(self) -> bool:
        return self.window_id is not None

    def lineage(self, hotkey: str) -> frozenset[str]:
        """Return the authenticated hotkey component across swaps/re-registration."""
        _text(hotkey, "hotkey")
        neighbors: dict[str, set[str]] = {hotkey: set()}
        for old, new in self.lineage_edges:
            neighbors.setdefault(old, set()).add(new)
            neighbors.setdefault(new, set()).add(old)
        pending, result = [hotkey], set()
        while pending:
            item = pending.pop()
            if item not in result:
                result.add(item)
                pending.extend(neighbors.get(item, ()))
        return frozenset(result)


def build_authenticated_history(
    *,
    network: str,
    netuid: int,
    through_block: int,
    certified_state_hash: str,
    consumed_hotkeys: tuple[str, ...],
    registered_hotkeys: tuple[str, ...],
    lineage_edges: tuple[tuple[str, str], ...] = (),
    window_id: str | None = None,
    window_open_block: int | None = None,
    window_close_block: int | None = None,
) -> AuthenticatedAdmissionHistory:
    """Build the canonical checkpoint payload before its external authentication."""
    payload: dict[str, object] = {
        "network": network,
        "netuid": netuid,
        "through_block": through_block,
        "certified_state_hash": certified_state_hash,
        "consumed_hotkeys": consumed_hotkeys,
        "registered_hotkeys": registered_hotkeys,
        "lineage_edges": lineage_edges,
    }
    if window_id is not None:
        payload.update(
            {
                "window_id": window_id,
                "window_open_block": window_open_block,
                "window_close_block": window_close_block,
            }
        )
    return AuthenticatedAdmissionHistory(
        network,
        netuid,
        through_block,
        certified_state_hash,
        hash_canonical(
            "forge/validator-admission-history/v2"
            if window_id is not None
            else "forge/validator-admission-history/v1",
            payload,
        ),
        consumed_hotkeys,
        registered_hotkeys,
        lineage_edges,
        window_id,
        window_open_block,
        window_close_block,
    )


@dataclass(frozen=True)
class Admission:
    id: int
    queue_sequence: int
    hotkey: str
    block: int
    extrinsic_index: int
    commitment_hash: str
    artifact_hash: str
    recipe_hash: str
    panel_hash: str
    certified_state_hash: str
    history_root: str
    status: str


class AdmissionLedger:
    """SQLite queue whose admission rights are keyed only by hotkey lineage."""

    def __init__(self, path: Path | str, *, network: str, netuid: int) -> None:
        self.path = Path(path)
        self.network, self.netuid = _text(network, "network"), _block(netuid, "netuid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS admission_history (
                    network TEXT NOT NULL, netuid INTEGER NOT NULL,
                    through_block INTEGER NOT NULL, certified_state_hash TEXT NOT NULL,
                    history_root TEXT NOT NULL, PRIMARY KEY (network, netuid)
                );
                CREATE TABLE IF NOT EXISTS consumed_hotkeys (
                    network TEXT NOT NULL, netuid INTEGER NOT NULL, hotkey TEXT NOT NULL,
                    consumed_by INTEGER, source TEXT NOT NULL,
                    PRIMARY KEY (network, netuid, hotkey)
                );
                CREATE TABLE IF NOT EXISTS admissions (
                    id INTEGER PRIMARY KEY, network TEXT NOT NULL, netuid INTEGER NOT NULL,
                    queue_sequence INTEGER NOT NULL, hotkey TEXT NOT NULL, block INTEGER NOT NULL,
                    extrinsic_index INTEGER NOT NULL,
                    commitment_hash TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    recipe_hash TEXT NOT NULL,
                    panel_hash TEXT NOT NULL,
                    effective_hash TEXT,
                    certified_state_hash TEXT NOT NULL,
                    history_root TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resume_status TEXT,
                    UNIQUE(network, netuid, queue_sequence),
                    UNIQUE(network, netuid, hotkey),
                    UNIQUE(network, netuid, effective_hash)
                );
                CREATE TABLE IF NOT EXISTS full_evaluation_rounds (
                    network TEXT NOT NULL, netuid INTEGER NOT NULL, round_id TEXT NOT NULL,
                    admission_id INTEGER NOT NULL, PRIMARY KEY(network, netuid, round_id)
                );
                CREATE TABLE IF NOT EXISTS effective_hash_aliases (
                    network TEXT NOT NULL, netuid INTEGER NOT NULL,
                    effective_hash TEXT NOT NULL, admission_id INTEGER NOT NULL,
                    scheme TEXT NOT NULL,
                    PRIMARY KEY(network, netuid, effective_hash)
                );
                """
            )
            migrate_admission_windows(db)
            migrate_full_evaluation_claims(db)
            # Historical rows predate scheduler-aware v2.  Retain their v1
            # identities as aliases; never guess a v2 scheduler from absent
            # evidence.
            db.execute(
                "INSERT OR IGNORE INTO effective_hash_aliases "
                "SELECT network, netuid, effective_hash, id, 'legacy_v1' FROM admissions "
                "WHERE effective_hash IS NOT NULL"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _authenticate(
        self,
        history: AuthenticatedAdmissionHistory | None,
        verify_history: Callable[[AuthenticatedAdmissionHistory], bool] | None,
    ) -> AuthenticatedAdmissionHistory:
        if history is None or verify_history is None or verify_history(history) is not True:
            raise AdmissionError(
                "independently authenticated certified admission history is required"
            )
        if (history.network, history.netuid) != (self.network, self.netuid):
            raise AdmissionError("certified admission history belongs to another subnet")
        return history

    def _record_history(
        self, db: sqlite3.Connection, history: AuthenticatedAdmissionHistory
    ) -> None:
        prior = db.execute(
            "SELECT through_block, certified_state_hash, history_root FROM admission_history "
            "WHERE network = ? AND netuid = ?",
            (self.network, self.netuid),
        ).fetchone()
        if prior and history.through_block < prior["through_block"]:
            raise AdmissionError("certified admission history moved backwards")
        if (
            prior
            and history.through_block == prior["through_block"]
            and (
                history.certified_state_hash != prior["certified_state_hash"]
                or history.history_root != prior["history_root"]
            )
        ):
            raise AdmissionError("certified admission history conflicts at the same height")
        db.execute(
            "INSERT INTO admission_history VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(network, netuid) DO UPDATE SET through_block=excluded.through_block, "
            "certified_state_hash=excluded.certified_state_hash, "
            "history_root=excluded.history_root",
            (
                self.network,
                self.netuid,
                history.through_block,
                history.certified_state_hash,
                history.history_root,
            ),
        )
        if not history.is_windowed:
            db.executemany(
                "INSERT OR IGNORE INTO consumed_hotkeys("
                "network, netuid, hotkey, consumed_by, source) "
                "VALUES (?, ?, ?, NULL, 'certified_history')",
                [(self.network, self.netuid, hotkey) for hotkey in history.consumed_hotkeys],
            )
        if history.is_windowed:
            db.executemany(
                "INSERT OR IGNORE INTO window_consumed_hotkeys(network, netuid, window_id, hotkey, "
                "consumed_by, source) VALUES (?, ?, ?, ?, NULL, 'certified_history')",
                [
                    (self.network, self.netuid, history.window_id, hotkey)
                    for hotkey in history.consumed_hotkeys
                ],
            )

    def admit(
        self,
        commitment: FinalizedCommitment,
        *,
        history: AuthenticatedAdmissionHistory | None,
        verify_history: Callable[[AuthenticatedAdmissionHistory], bool] | None,
        effective_hash: str | None = None,
        effective_aliases: tuple[str, ...] = (),
    ) -> Admission:
        """Consume a hotkey's one competitive right and append it FIFO.

        No coldkey argument exists: a newly registered hotkey is eligible even
        when an operator controls another consumed hotkey.  Swapping/reusing a
        consumed hotkey remains blocked because the key itself is the lineage.
        """
        history = self._authenticate(history, verify_history)
        if effective_hash is not None:
            _hash(effective_hash, "effective candidate hash")
        aliases = (
            tuple(dict.fromkeys((effective_hash, *effective_aliases))) if effective_hash else ()
        )
        for alias in aliases:
            _hash(alias, "effective candidate alias")
        if commitment.block > history.through_block:
            raise AdmissionError("commitment is not present in finalized certified history")
        if history.is_windowed and not (
            history.window_open_block < commitment.block <= history.window_close_block
        ):
            raise AdmissionError("commitment is outside the authenticated admission window")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._record_history(db, history)
            if commitment.hotkey not in history.registered_hotkeys:
                raise AdmissionError("candidate hotkey is not registered in certified history")
            scope = history.window_id if history.is_windowed else "legacy-lifetime/v1"
            existing = db.execute(
                "SELECT * FROM admissions WHERE network=? AND netuid=? "
                "AND window_id=? AND hotkey=?",
                (self.network, self.netuid, scope, commitment.hotkey),
            ).fetchone()
            if existing:
                if (
                    all(
                        existing[name] == getattr(commitment, name)
                        for name in (
                            "block",
                            "extrinsic_index",
                            "commitment_hash",
                            "artifact_hash",
                            "recipe_hash",
                            "panel_hash",
                        )
                    )
                    and existing["effective_hash"] == effective_hash
                ):
                    db.commit()
                    return self._row(existing)
                raise AdmissionError("hotkey lineage was already consumed by another candidate")
            lineage = tuple(history.lineage(commitment.hotkey))
            placeholders = ",".join("?" for _ in lineage)
            consumed_table = (
                "window_consumed_hotkeys" if history.is_windowed else "consumed_hotkeys"
            )
            scope_clause = " AND window_id=?" if history.is_windowed else ""
            consumed = db.execute(
                f"SELECT 1 FROM {consumed_table} WHERE network=? AND netuid=?{scope_clause} "
                f"AND hotkey IN ({placeholders})",
                (self.network, self.netuid, *((scope,) if history.is_windowed else ()), *lineage),
            ).fetchone()
            if consumed:
                raise AdmissionError("hotkey lineage was already consumed")
            if history.is_windowed:
                active = db.execute(
                    "SELECT 1 FROM admissions WHERE network=? AND netuid=? AND hotkey IN ("
                    + placeholders
                    + ") AND status NOT IN ("
                    "'REJECTED', 'FAILED', 'INCONCLUSIVE', 'CERTIFIED', 'INCOMPLETE')",
                    (self.network, self.netuid, *lineage),
                ).fetchone()
                if active:
                    raise AdmissionError("hotkey lineage already has a pending admission")
            placeholders = ",".join("?" for _ in aliases)
            if (
                aliases
                and db.execute(
                    "SELECT 1 FROM effective_hash_aliases WHERE network=? AND netuid=? "
                    f"AND effective_hash IN ({placeholders})",
                    (self.network, self.netuid, *aliases),
                ).fetchone()
            ):
                raise AdmissionError("effective candidate identity was already admitted")
            latest = db.execute(
                "SELECT block, extrinsic_index FROM admissions WHERE network=? AND netuid=? "
                "ORDER BY block DESC, extrinsic_index DESC LIMIT 1",
                (self.network, self.netuid),
            ).fetchone()
            if latest and (commitment.block, commitment.extrinsic_index) < (
                latest["block"],
                latest["extrinsic_index"],
            ):
                raise AdmissionError("out-of-order finalized commitment; admit a canonical batch")
            sequence = db.execute(
                "SELECT COALESCE(MAX(queue_sequence), 0) + 1 FROM admissions "
                "WHERE network=? AND netuid=?",
                (self.network, self.netuid),
            ).fetchone()[0]
            cursor = db.execute(
                "INSERT INTO admissions("
                "network, netuid, queue_sequence, hotkey, block, extrinsic_index, commitment_hash, "
                "artifact_hash, window_id, "
                "recipe_hash, panel_hash, effective_hash, certified_state_hash, "
                "history_root, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'QUEUED')",
                (
                    self.network,
                    self.netuid,
                    sequence,
                    commitment.hotkey,
                    commitment.block,
                    commitment.extrinsic_index,
                    commitment.commitment_hash,
                    commitment.artifact_hash,
                    scope,
                    commitment.recipe_hash,
                    commitment.panel_hash,
                    effective_hash,
                    history.certified_state_hash,
                    history.history_root,
                ),
            )
            if history.is_windowed:
                db.execute(
                    "INSERT INTO window_consumed_hotkeys(network, netuid, window_id, hotkey, "
                    "consumed_by, source) VALUES (?, ?, ?, ?, ?, 'local_admission')",
                    (self.network, self.netuid, scope, commitment.hotkey, cursor.lastrowid),
                )
            else:
                db.execute(
                    "INSERT INTO consumed_hotkeys(network, netuid, hotkey, consumed_by, source) "
                    "VALUES (?, ?, ?, ?, 'local_admission')",
                    (self.network, self.netuid, commitment.hotkey, cursor.lastrowid),
                )
            if aliases:
                db.executemany(
                    "INSERT INTO effective_hash_aliases("
                    "network, netuid, effective_hash, admission_id, scheme) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            self.network,
                            self.netuid,
                            alias,
                            cursor.lastrowid,
                            "v2" if alias == effective_hash else "v1_alias",
                        )
                        for alias in aliases
                    ],
                )
            row = db.execute("SELECT * FROM admissions WHERE id=?", (cursor.lastrowid,)).fetchone()
            db.commit()
            return self._row(row)

    def admit_finalized_batch(
        self,
        commitments: Iterable[FinalizedCommitment],
        *,
        history: AuthenticatedAdmissionHistory | None,
        verify_history: Callable[[AuthenticatedAdmissionHistory], bool] | None,
        effective_hashes: Mapping[str, str] | None = None,
        effective_aliases: Mapping[str, tuple[str, ...]] | None = None,
    ) -> tuple[Admission, ...]:
        """Admit a canonical finalized batch; reject caller arrival order ambiguity."""
        items = tuple(commitments)
        if items != tuple(sorted(items, key=lambda item: (item.block, item.extrinsic_index))):
            raise AdmissionError("finalized commitments must be in canonical block/extrinsic order")
        identities = effective_hashes or {}
        aliases = effective_aliases or {}
        if set(identities) - {item.commitment_hash for item in items}:
            raise AdmissionError("effective identity is not part of the finalized batch")
        return tuple(
            self.admit(
                item,
                history=history,
                verify_history=verify_history,
                effective_hash=identities.get(item.commitment_hash),
                effective_aliases=aliases.get(item.commitment_hash, ()),
            )
            for item in items
        )

    def next_queued(self) -> Admission | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM admissions WHERE network=? AND netuid=? AND status='QUEUED' "
                "ORDER BY queue_sequence LIMIT 1",
                (self.network, self.netuid),
            ).fetchone()
        return self._row(row) if row else None

    def claim_next_screening(self) -> Admission | None:
        """Atomically claim FIFO head, preventing two workers from screening it."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM admissions WHERE network=? AND netuid=? AND status='QUEUED' "
                "ORDER BY queue_sequence LIMIT 1",
                (self.network, self.netuid),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            db.execute("UPDATE admissions SET status='SCREENING' WHERE id=?", (row["id"],))
            result = db.execute("SELECT * FROM admissions WHERE id=?", (row["id"],)).fetchone()
            db.commit()
            return self._row(result)

    def start_full_evaluation(self, admission_id: int, *, round_id: str) -> Admission:
        _text(round_id, "round id")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM admissions WHERE id=? AND network=? AND netuid=?",
                (admission_id, self.network, self.netuid),
            ).fetchone()
            if row is None or row["status"] not in {"SCREENING", "INFRASTRUCTURE_FAILED"}:
                raise AdmissionError(
                    "only this validator's screened admission may enter full evaluation"
                )
            prior_round = db.execute(
                "SELECT round_id FROM full_evaluation_rounds "
                "WHERE network=? AND netuid=? AND admission_id=?",
                (self.network, self.netuid, admission_id),
            ).fetchone()
            if prior_round is not None and prior_round["round_id"] != round_id:
                raise AdmissionError("full evaluation resume must retain its frozen round")
            if prior_round is None:
                db.execute(
                    "INSERT INTO full_evaluation_rounds VALUES (?, ?, ?, ?)",
                    (self.network, self.netuid, round_id, admission_id),
                )
            db.execute("UPDATE admissions SET status='FULL_EVALUATION' WHERE id=?", (admission_id,))
            result = db.execute("SELECT * FROM admissions WHERE id=?", (admission_id,)).fetchone()
            db.commit()
            return self._row(result)

    def transition(self, admission_id: int, status: str) -> Admission:
        if status not in _STATUSES:
            raise AdmissionError("unknown admission status")
        if status == "FULL_EVALUATION":
            raise AdmissionError("full evaluation must be claimed through its round gate")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM admissions WHERE id=? AND network=? AND netuid=?",
                (admission_id, self.network, self.netuid),
            ).fetchone()
            if row is None:
                raise AdmissionError("admission does not exist")
            if row["status"] in _TERMINAL:
                raise AdmissionError("terminal admission cannot move")
            allowed = {
                "QUEUED": {"SCREENING", "REJECTED", "INFRASTRUCTURE_FAILED"},
                "SCREENING": {
                    "FULL_EVALUATION",
                    "REJECTED",
                    "INCONCLUSIVE",
                    "INCOMPLETE",
                    "INFRASTRUCTURE_FAILED",
                },
                "FULL_EVALUATION": {
                    "CERTIFIED",
                    "FAILED",
                    "INCONCLUSIVE",
                    "INCOMPLETE",
                    "INFRASTRUCTURE_FAILED",
                },
                "INFRASTRUCTURE_FAILED": {"SCREENING", "FULL_EVALUATION", "INCOMPLETE"},
            }
            if status not in allowed[row["status"]]:
                raise AdmissionError(f"cannot move {row['status']} to {status}")
            if status == "INFRASTRUCTURE_FAILED":
                db.execute(
                    "UPDATE admissions SET status=?, resume_status=? WHERE id=?",
                    (status, row["status"], admission_id),
                )
            else:
                db.execute(
                    "UPDATE admissions SET status=?, resume_status=NULL WHERE id=?",
                    (status, admission_id),
                )
            result = db.execute("SELECT * FROM admissions WHERE id=?", (admission_id,)).fetchone()
            db.commit()
            return self._row(result)

    def return_screening_to_queue(self, admission_id: int) -> Admission:
        """Release an unstarted screening claim when reservation fails before work."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM admissions WHERE id=? AND network=? AND netuid=?",
                (admission_id, self.network, self.netuid),
            ).fetchone()
            if row is None or row["status"] != "SCREENING":
                raise AdmissionError("only an unstarted screening claim can return to queue")
            db.execute("UPDATE admissions SET status='QUEUED' WHERE id=?", (admission_id,))
            result = db.execute("SELECT * FROM admissions WHERE id=?", (admission_id,)).fetchone()
            db.commit()
            return self._row(result)

    def get(self, admission_id: int) -> Admission:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM admissions WHERE id=? AND network=? AND netuid=?",
                (admission_id, self.network, self.netuid),
            ).fetchone()
        if row is None:
            raise AdmissionError("admission does not exist")
        return self._row(row)

    def resumable_stage(self, admission_id: int) -> str:
        """Return the frozen stage preceding an infrastructure-only failure."""
        with self._connect() as db:
            row = db.execute(
                "SELECT status, resume_status FROM admissions "
                "WHERE id=? AND network=? AND netuid=?",
                (admission_id, self.network, self.netuid),
            ).fetchone()
        if row is None:
            raise AdmissionError("admission does not exist")
        if row["status"] != "INFRASTRUCTURE_FAILED" or row["resume_status"] not in {
            "SCREENING",
            "FULL_EVALUATION",
        }:
            raise AdmissionError("infrastructure failure has no resumable evaluation stage")
        return row["resume_status"]

    @staticmethod
    def _row(row: sqlite3.Row) -> Admission:
        return Admission(**{field: row[field] for field in Admission.__dataclass_fields__})

    def list(self) -> Iterable[Admission]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM admissions WHERE network=? AND netuid=? ORDER BY queue_sequence",
                (self.network, self.netuid),
            ).fetchall()
        return tuple(self._row(row) for row in rows)
