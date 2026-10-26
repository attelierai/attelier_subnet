"""Narrow migration of the former one-challenger-per-round claim table."""

import sqlite3


def migrate_admission_windows(db: sqlite3.Connection) -> None:
    """Preserve legacy admission rows while replacing lifetime hotkey uniqueness.

    Old rows remain in a synthetic immutable scope so replay keeps its original
    lifetime semantics.  New windowed histories use the rebuilt table.
    """

    columns = {row["name"] for row in db.execute("PRAGMA table_info(admissions)").fetchall()}
    if "window_id" not in columns:
        db.execute("BEGIN IMMEDIATE")
        db.execute("ALTER TABLE admissions RENAME TO legacy_admissions")
        db.execute(
            "CREATE TABLE admissions ("
            "id INTEGER PRIMARY KEY, network TEXT NOT NULL, netuid INTEGER NOT NULL, "
            "queue_sequence INTEGER NOT NULL, hotkey TEXT NOT NULL, block INTEGER NOT NULL, "
            "extrinsic_index INTEGER NOT NULL, commitment_hash TEXT NOT NULL, "
            "artifact_hash TEXT NOT NULL, recipe_hash TEXT NOT NULL, panel_hash TEXT NOT NULL, "
            "effective_hash TEXT, certified_state_hash TEXT NOT NULL, history_root TEXT NOT NULL, "
            "status TEXT NOT NULL, resume_status TEXT, window_id TEXT NOT NULL, "
            "UNIQUE(network, netuid, queue_sequence), "
            "UNIQUE(network, netuid, window_id, hotkey), "
            "UNIQUE(network, netuid, effective_hash))"
        )
        db.execute(
            "INSERT INTO admissions(id, network, netuid, queue_sequence, hotkey, block, "
            "extrinsic_index, commitment_hash, artifact_hash, recipe_hash, panel_hash, "
            "effective_hash, certified_state_hash, history_root, status, resume_status, window_id) "
            "SELECT id, network, netuid, queue_sequence, hotkey, block, extrinsic_index, "
            "commitment_hash, artifact_hash, recipe_hash, panel_hash, effective_hash, "
            "certified_state_hash, history_root, status, resume_status, 'legacy-lifetime/v1' "
            "FROM legacy_admissions"
        )
        db.execute("DROP TABLE legacy_admissions")
        db.commit()
    db.execute(
        "CREATE TABLE IF NOT EXISTS window_consumed_hotkeys ("
        "network TEXT NOT NULL, netuid INTEGER NOT NULL, window_id TEXT NOT NULL, "
        "hotkey TEXT NOT NULL, consumed_by INTEGER, source TEXT NOT NULL, "
        "PRIMARY KEY(network, netuid, window_id, hotkey))"
    )
    db.execute(
        "INSERT OR IGNORE INTO window_consumed_hotkeys(network, netuid, window_id, hotkey, "
        "consumed_by, source) SELECT network, netuid, 'legacy-lifetime/v1', hotkey, "
        "consumed_by, source FROM consumed_hotkeys"
    )
    db.commit()


def migrate_full_evaluation_claims(db: sqlite3.Connection) -> None:
    columns = db.execute("PRAGMA table_info(full_evaluation_rounds)").fetchall()
    primary = [row["name"] for row in sorted(columns, key=lambda row: row["pk"]) if row["pk"]]
    if primary == ["network", "netuid", "admission_id"]:
        return
    if primary != ["network", "netuid", "round_id"] or {row["name"] for row in columns} != {
        "network",
        "netuid",
        "round_id",
        "admission_id",
    }:
        raise ValueError("unknown full-evaluation schema; preserve database for inspection")
    db.execute("BEGIN IMMEDIATE")
    db.execute("ALTER TABLE full_evaluation_rounds RENAME TO legacy_full_evaluation_rounds")
    db.execute(
        "CREATE TABLE full_evaluation_rounds ("
        "network TEXT NOT NULL, netuid INTEGER NOT NULL, round_id TEXT NOT NULL, "
        "admission_id INTEGER NOT NULL, PRIMARY KEY(network, netuid, admission_id))"
    )
    db.execute("INSERT INTO full_evaluation_rounds SELECT * FROM legacy_full_evaluation_rounds")
    db.execute("DROP TABLE legacy_full_evaluation_rounds")
    db.commit()
