"""Read-only, prompt-free operational export. Does not claim real calibration."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def export_local_snapshot(
    database: Path,
    *,
    network: str,
    netuid: int,
    budget_database: Path | None = None,
    candidate_id: int | None = None,
) -> dict:
    if not database.is_file():
        raise ValueError("missing ledger cannot be presented as an empty verified queue")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT id, hotkey, artifact_hash, status FROM admissions "
            "WHERE network=? AND netuid=? ORDER BY queue_sequence",
            (network, netuid),
        ).fetchall()
    by_id = {row["id"]: row for row in rows}
    if candidate_id is not None and candidate_id not in by_id:
        raise ValueError("budget candidate is not in the selected queue scope")
    spent = reserved = None
    if candidate_id is not None and budget_database is not None:
        if not budget_database.is_file():
            raise ValueError("missing budget ledger")
        with sqlite3.connect(budget_database.resolve().as_uri() + "?mode=ro", uri=True) as db:
            totals = dict(
                db.execute(
                    "SELECT status, SUM(amount_usd) FROM budget_reservations "
                    "WHERE candidate_id=? GROUP BY status",
                    (candidate_id,),
                )
            )
        spent = totals.get("CAPTURED", 0.0)
        reserved = sum(totals.get(status, 0.0) for status in ("RESERVED", "UNCERTAIN"))
    return {
        "schema": "forge/mvp-dashboard/v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "evidence_kind": "local_integration",
        "activation": {
            "enabled": False,
            "blockers": [
                "Independent Wan calibration and cost qualification are pending.",
                "Complete testnet certificate and burn replacement are not qualified.",
            ],
        },
        "queue": [
            {
                "candidate_id": row["artifact_hash"],
                "hotkey": row["hotkey"],
                "status": row["status"].lower(),
                "reason": None,
            }
            for row in rows
        ],
        "evaluation": {
            "status": by_id[candidate_id]["status"].lower() if candidate_id else "not_started",
            "screen_prompts": 16,
            "confirmation_prompts": 128,
            "timing_prompts": 32,
            "quality": None,
            "speedup": None,
        },
        "budget": {"limit_usd": 10, "spent_usd": spent, "reserved_usd": reserved},
        "winner": None,
    }
