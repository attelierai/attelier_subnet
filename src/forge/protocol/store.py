"""Transactional SQLite persistence for protocol artifacts and certified state."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class StoreConflictError(RuntimeError):
    pass


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # autocommit mode: transactions are explicit via transaction()
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                round_id TEXT PRIMARY KEY,
                challenge_id TEXT NOT NULL,
                plan_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'FINALIZED', 'REJECTED')),
                payload_json TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_pending_round_per_challenge
                ON rounds(challenge_id) WHERE status = 'PENDING';

            CREATE TABLE IF NOT EXISTS submissions (
                round_id TEXT NOT NULL REFERENCES rounds(round_id),
                hotkey TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                effective_fingerprint TEXT NOT NULL,
                commitment_block INTEGER NOT NULL,
                extrinsic_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(round_id, artifact_id),
                UNIQUE(round_id, hotkey),
                UNIQUE(round_id, effective_fingerprint)
            );

            CREATE TABLE IF NOT EXISTS reports (
                round_id TEXT NOT NULL REFERENCES rounds(round_id),
                validator_hotkey TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(round_id, validator_hotkey)
            );

            CREATE TABLE IF NOT EXISTS certificates (
                certificate_hash TEXT PRIMARY KEY,
                -- no FK to rounds: a syncing node may never have seen the round locally
                round_id TEXT NOT NULL UNIQUE,
                challenge_id TEXT NOT NULL,
                parent_state_hash TEXT NOT NULL,
                resulting_state_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS finalized_states (
                state_hash TEXT PRIMARY KEY,
                challenge_id TEXT NOT NULL,
                state_version INTEGER NOT NULL,
                previous_state_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(challenge_id, state_version)
            );

            CREATE TABLE IF NOT EXISTS challenge_heads (
                challenge_id TEXT PRIMARY KEY,
                state_hash TEXT NOT NULL REFERENCES finalized_states(state_hash)
            );

            -- one attested transition per validator per round, across restarts
            CREATE TABLE IF NOT EXISTS attestations (
                round_id TEXT NOT NULL,
                validator_hotkey TEXT NOT NULL,
                transition_hash TEXT NOT NULL,
                PRIMARY KEY(round_id, validator_hotkey)
            );

            CREATE TABLE IF NOT EXISTS crowns (
                crown_id TEXT PRIMARY KEY,
                certificate_hash TEXT NOT NULL UNIQUE REFERENCES certificates(certificate_hash),
                challenge_id TEXT NOT NULL,
                hotkey TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                chain_time INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _idempotent_row(row: sqlite3.Row | None, expected: dict[str, Any], label: str) -> bool:
        if row is None:
            return False
        for key, value in expected.items():
            if row[key] != value:
                raise StoreConflictError(f"{label} already exists with different {key}")
        return True

    def create_round(
        self,
        round_id: str,
        challenge_id: str,
        plan_hash: str,
        payload: dict[str, Any],
    ) -> bool:
        encoded = _json(payload)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
            ).fetchone()
            if self._idempotent_row(
                row,
                {"challenge_id": challenge_id, "plan_hash": plan_hash, "payload_json": encoded},
                "round",
            ):
                return False
            try:
                connection.execute(
                    "INSERT INTO rounds VALUES (?, ?, ?, 'PENDING', ?)",
                    (round_id, challenge_id, plan_hash, encoded),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreConflictError(
                    "a challenge may have only one pending comparison"
                ) from exc
        return True

    def record_submission(
        self,
        *,
        round_id: str,
        hotkey: str,
        artifact_id: str,
        effective_fingerprint: str,
        commitment_block: int,
        extrinsic_index: int,
        payload: dict[str, Any],
    ) -> bool:
        encoded = _json(payload)
        expected = {
            "hotkey": hotkey,
            "effective_fingerprint": effective_fingerprint,
            "commitment_block": commitment_block,
            "extrinsic_index": extrinsic_index,
            "payload_json": encoded,
        }
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM submissions WHERE round_id = ? AND artifact_id = ?",
                (round_id, artifact_id),
            ).fetchone()
            if self._idempotent_row(row, expected, "submission"):
                return False
            try:
                connection.execute(
                    """INSERT INTO submissions
                    (round_id, hotkey, artifact_id, effective_fingerprint,
                     commitment_block, extrinsic_index, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        round_id,
                        hotkey,
                        artifact_id,
                        effective_fingerprint,
                        commitment_block,
                        extrinsic_index,
                        encoded,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreConflictError(
                    "submission violates the hotkey or fingerprint uniqueness rule"
                ) from exc
        return True

    def record_report(
        self,
        round_id: str,
        validator_hotkey: str,
        report_hash: str,
        payload: dict[str, Any],
    ) -> bool:
        encoded = _json(payload)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM reports WHERE round_id = ? AND validator_hotkey = ?",
                (round_id, validator_hotkey),
            ).fetchone()
            if self._idempotent_row(
                row,
                {"report_hash": report_hash, "payload_json": encoded},
                "validator report",
            ):
                return False
            connection.execute(
                "INSERT INTO reports VALUES (?, ?, ?, ?)",
                (round_id, validator_hotkey, report_hash, encoded),
            )
        return True

    def initialize_state(
        self,
        *,
        state_hash: str,
        challenge_id: str,
        state_version: int,
        previous_state_hash: str,
        payload: dict[str, Any],
    ) -> bool:
        encoded = _json(payload)
        with self.transaction() as connection:
            head = connection.execute(
                "SELECT state_hash FROM challenge_heads WHERE challenge_id = ?", (challenge_id,)
            ).fetchone()
            if head is not None:
                # Idempotent while this genesis is the root of the stored history.
                root = connection.execute(
                    "SELECT state_hash FROM finalized_states"
                    " WHERE challenge_id = ? AND state_version = 0",
                    (challenge_id,),
                ).fetchone()
                if root is None or root["state_hash"] != state_hash:
                    raise StoreConflictError("challenge already has a different genesis")
                return False
            connection.execute(
                "INSERT INTO finalized_states VALUES (?, ?, ?, ?, ?)",
                (state_hash, challenge_id, state_version, previous_state_hash, encoded),
            )
            connection.execute(
                "INSERT INTO challenge_heads VALUES (?, ?)", (challenge_id, state_hash)
            )
        return True

    def apply_certificate(
        self,
        *,
        certificate_hash: str,
        round_id: str,
        challenge_id: str,
        parent_state_hash: str,
        resulting_state_hash: str,
        state_version: int,
        certificate_payload: dict[str, Any],
        state_payload: dict[str, Any],
        crown_payload: dict[str, Any] | None,
        attestations: dict[str, str] | None = None,
    ) -> bool:
        certificate_json = _json(certificate_payload)
        state_json = _json(state_payload)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM certificates WHERE certificate_hash = ?", (certificate_hash,)
            ).fetchone()
            if self._idempotent_row(
                existing,
                {
                    "round_id": round_id,
                    "challenge_id": challenge_id,
                    "parent_state_hash": parent_state_hash,
                    "resulting_state_hash": resulting_state_hash,
                },
                "certificate",
            ):
                return False
            head = connection.execute(
                "SELECT state_hash FROM challenge_heads WHERE challenge_id = ?", (challenge_id,)
            ).fetchone()
            if head is None or head["state_hash"] != parent_state_hash:
                raise StoreConflictError("certificate does not extend the current finalized head")
            connection.execute(
                "INSERT INTO certificates VALUES (?, ?, ?, ?, ?, ?)",
                (
                    certificate_hash,
                    round_id,
                    challenge_id,
                    parent_state_hash,
                    resulting_state_hash,
                    certificate_json,
                ),
            )
            connection.execute(
                "INSERT INTO finalized_states VALUES (?, ?, ?, ?, ?)",
                (
                    resulting_state_hash,
                    challenge_id,
                    state_version,
                    parent_state_hash,
                    state_json,
                ),
            )
            connection.execute(
                "UPDATE challenge_heads SET state_hash = ? WHERE challenge_id = ?",
                (resulting_state_hash, challenge_id),
            )
            connection.execute(
                "UPDATE rounds SET status = 'FINALIZED' WHERE round_id = ?", (round_id,)
            )
            for hotkey, transition_hash in (attestations or {}).items():
                self._record_attestation(connection, round_id, hotkey, transition_hash)
            if crown_payload is not None:
                connection.execute(
                    "INSERT INTO crowns VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        crown_payload["crown_id"],
                        certificate_hash,
                        challenge_id,
                        crown_payload["hotkey"],
                        crown_payload["fingerprint"],
                        crown_payload["chain_time"],
                        _json(crown_payload),
                    ),
                )
        return True

    @staticmethod
    def _record_attestation(
        connection: sqlite3.Connection, round_id: str, hotkey: str, transition_hash: str
    ) -> None:
        row = connection.execute(
            "SELECT transition_hash FROM attestations WHERE round_id = ? AND validator_hotkey = ?",
            (round_id, hotkey),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO attestations VALUES (?, ?, ?)", (round_id, hotkey, transition_hash)
            )
        elif row["transition_hash"] != transition_hash:
            raise StoreConflictError(f"validator {hotkey} attested conflicting transitions")

    def attested_transition(self, round_id: str, hotkey: str) -> str | None:
        row = self.connection.execute(
            "SELECT transition_hash FROM attestations WHERE round_id = ? AND validator_hotkey = ?",
            (round_id, hotkey),
        ).fetchone()
        return None if row is None else str(row["transition_hash"])

    def latest_state_payload(self, challenge_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT s.payload_json FROM finalized_states AS s
            JOIN challenge_heads AS h ON h.state_hash = s.state_hash
            WHERE h.challenge_id = ?""",
            (challenge_id,),
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def state_payload(self, state_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM finalized_states WHERE state_hash = ?", (state_hash,)
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def certificate_payload(self, certificate_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM certificates WHERE certificate_hash = ?", (certificate_hash,)
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def round_payload(self, round_id: str) -> dict[str, Any] | None:
        """Return the exact round plan kept by the durable protocol store."""

        row = self.connection.execute(
            "SELECT payload_json FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def crowns(self, challenge_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload_json FROM crowns WHERE challenge_id = ? ORDER BY chain_time, crown_id",
            (challenge_id,),
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def count(self, table: str) -> int:
        allowed = {
            "rounds",
            "submissions",
            "reports",
            "certificates",
            "attestations",
            "finalized_states",
            "crowns",
        }
        if table not in allowed:
            raise ValueError("unknown table")
        row = self.connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"])
