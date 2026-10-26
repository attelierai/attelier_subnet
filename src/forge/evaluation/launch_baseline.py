"""Predeclared six-prompt screening followed by exactly one confirmation.

Screening selects a measurement to run; it never crowns a miner. The original
Wan50 anchor remains the fallback, and the imported baseline has no reward owner.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from forge.evaluation.launch_policy import AXES, freeze_policy
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_service import write_immutable_json


def _hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("candidate identity must be a lowercase SHA-256")
    return value


def select(screenings: list[dict[str, Any]], *, anchor_hash: str) -> dict[str, Any]:
    """Rank at most three alternatives from six matched prompt measurements."""
    _hash(anchor_hash)
    if len(screenings) > 3:
        raise ValueError("baseline campaign permits at most three alternatives")
    candidates, identities, panel = [], set(), None
    margins = freeze_policy()["quality_margins"]
    for receipt in screenings:
        identity = _hash(receipt["candidate_hash"])
        if identity == anchor_hash or identity in identities:
            raise ValueError("screening candidate duplicates anchor or another alternative")
        identities.add(identity)
        if receipt.get("anchor_hash") != anchor_hash:
            raise ValueError("screening changed the original Wan50 anchor")
        rows = receipt.get("rows")
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError("every alternative needs exactly six screening prompts")
        prompt_ids = tuple(sorted(row["prompt_id"] for row in rows))
        if len(set(prompt_ids)) != 6:
            raise ValueError("screening must use six independent prompts")
        if panel is not None and panel != prompt_ids:
            raise ValueError("all screening alternatives must use the same frozen prompts")
        panel = prompt_ids
        reasons, logs, deltas = [], [], {axis: [] for axis in AXES}
        for row in rows:
            if row.get("status") != "OK":
                reasons.append("incomplete_generation")
                continue
            if not isinstance(row.get("vetoes"), list) or any(
                not isinstance(veto, str) for veto in row["vetoes"]
            ):
                raise ValueError("explicit evaluated vetoes are required")
            reasons.extend(row["vetoes"])
            reference, candidate = row.get("reference_seconds"), row.get("candidate_seconds")
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in (reference, candidate)
            ):
                raise ValueError("screen timing is nonfinite or invalid")
            logs.append(math.log(reference / candidate))
            for side in ("reference_scores", "candidate_scores"):
                scores = row.get(side)
                if not isinstance(scores, dict) or set(scores) != set(AXES):
                    raise ValueError("screening requires separate VQ, MQ and TA")
                if any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in scores.values()
                ):
                    raise ValueError("screen scores must be finite")
            for axis in AXES:
                deltas[axis].append(row["candidate_scores"][axis] - row["reference_scores"][axis])
        means = {
            axis: statistics.fmean(values) if len(values) == 6 else None
            for axis, values in deltas.items()
        }
        reasons.extend(
            f"{axis}_mean_outside_margin"
            for axis, mean in means.items()
            if mean is None or mean < -margins[axis]
        )
        candidates.append(
            {
                "candidate_hash": identity,
                "eligible_for_confirmation": not reasons,
                "speed_geomean": math.exp(statistics.fmean(logs)) if len(logs) == 6 else None,
                "quality_mean_deltas": means,
                "reasons": sorted(set(reasons)),
            }
        )
    eligible = sorted(
        (row for row in candidates if row["eligible_for_confirmation"]),
        key=lambda row: (-row["speed_geomean"], row["candidate_hash"]),
    )
    canonical_screenings = sorted(
        (
            {
                **receipt,
                "rows": sorted(receipt["rows"], key=lambda row: row["prompt_id"]),
            }
            for receipt in screenings
        ),
        key=lambda receipt: receipt["candidate_hash"],
    )
    body = {
        "schema": "forge/launch-baseline-selection/v1",
        "anchor_hash": anchor_hash,
        "screening_prompts": list(panel or ()),
        "candidates": sorted(candidates, key=lambda row: row["candidate_hash"]),
        "selected_candidate_hash": eligible[0]["candidate_hash"] if eligible else None,
        "screening_hash": hash_canonical(
            "forge/launch-baseline-screening/v1", canonical_screenings
        ),
        "status": "AWAITING_SINGLE_CONFIRMATION" if eligible else "KEEP_WAN50",
        "reward_hotkey": None,
    }
    return {**body, "selection_id": hash_canonical("forge/launch-baseline-selection/v1", body)}


class ParentLaunchBaselineStore:
    """One immutable screening choice and one raw V2 confirmation replay."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS parent_launch_baseline (id INTEGER PRIMARY KEY "
                "CHECK(id=1), screenings_json TEXT NOT NULL, selection_json TEXT NOT NULL, "
                "confirmation_plan_json TEXT, confirmation_rows_json TEXT, "
                "qualification_package_json TEXT, result_json TEXT)"
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(parent_launch_baseline)")}
            if "screenings_json" not in columns:
                db.execute("ALTER TABLE parent_launch_baseline ADD COLUMN screenings_json TEXT")
            if "qualification_package_json" not in columns:
                db.execute(
                    "ALTER TABLE parent_launch_baseline ADD COLUMN qualification_package_json TEXT"
                )

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _encoded(value: Any) -> str:
        """Persist ordinary JSON so SQLite reloads numeric evidence as numbers.

        Canonical encodings deliberately tag floats for hashing. They are not a
        storage representation: decoding such a value through ``json.loads``
        turns a measured float into a mapping and changes V2 replay semantics.
        Hashes below retain the canonical representation at their boundary.
        """
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def select_once(self, screenings: list[dict[str, Any]], *, anchor_hash: str) -> dict[str, Any]:
        result = select(screenings, anchor_hash=anchor_hash)
        encoded = self._encoded(result)
        screenings_encoded = self._encoded(screenings)
        with self._db() as db:
            row = db.execute(
                "SELECT screenings_json, selection_json FROM parent_launch_baseline WHERE id=1"
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO parent_launch_baseline "
                    "(id, screenings_json, selection_json, confirmation_plan_json, "
                    "confirmation_rows_json, qualification_package_json, result_json) "
                    "VALUES (1, ?, ?, NULL, NULL, NULL, NULL)",
                    (screenings_encoded, encoded),
                )
            elif row["selection_json"] != encoded or row["screenings_json"] != screenings_encoded:
                raise ValueError("baseline selection is immutable after the first screening")
        return result

    def evidence(self) -> dict[str, Any]:
        """Return the raw durable record used to reconstruct a bootstrap receipt."""
        with self._db() as db:
            row = db.execute("SELECT * FROM parent_launch_baseline WHERE id=1").fetchone()
        if row is None or row["screenings_json"] is None:
            raise ValueError("baseline screening evidence is unavailable")
        return {
            "screenings": json.loads(row["screenings_json"]),
            "selection": json.loads(row["selection_json"]),
            "confirmation_plan": (
                json.loads(row["confirmation_plan_json"])
                if row["confirmation_plan_json"] is not None
                else None
            ),
            "confirmation_rows": (
                json.loads(row["confirmation_rows_json"])
                if row["confirmation_rows_json"] is not None
                else None
            ),
            "qualification_package": (
                json.loads(row["qualification_package_json"])
                if row["qualification_package_json"] is not None
                else None
            ),
            "result": json.loads(row["result_json"]) if row["result_json"] is not None else None,
        }

    def confirm_once(
        self,
        plan: dict[str, Any],
        rows: list[dict[str, Any]] | dict[str, Any],
        qualification_package: dict[str, Any],
    ) -> dict[str, Any]:
        from forge.evaluation.evaluator_qualification import replay_qualification_package
        from forge.evaluation.launch_policy import evaluate_v2

        plan_encoded, rows_encoded = self._encoded(plan), self._encoded(rows)
        package_encoded = self._encoded(qualification_package)
        with self._db() as db:
            row = db.execute("SELECT * FROM parent_launch_baseline WHERE id=1").fetchone()
            if row is None:
                raise ValueError("screening selection must be persisted before confirmation")
            selection = json.loads(row["selection_json"])
            if selection["status"] != "AWAITING_SINGLE_CONFIRMATION":
                raise ValueError("Wan50 fallback was already selected")
            if plan.get("models", {}).get("candidate") != selection["selected_candidate_hash"]:
                raise ValueError("confirmation plan differs from the selected baseline candidate")
            if set(selection["screening_prompts"]) & {
                p.get("prompt_id") for p in plan.get("prompts", [])
            }:
                raise ValueError("confirmation panel reuses a screening prompt")
            if replay_qualification_package(qualification_package) != plan.get(
                "evaluator_qualification"
            ):
                raise ValueError("replayed qualification differs from frozen confirmation plan")
            if row["confirmation_plan_json"] is not None:
                if (
                    row["confirmation_plan_json"] != plan_encoded
                    or row["confirmation_rows_json"] != rows_encoded
                    or row["qualification_package_json"] != package_encoded
                ):
                    raise ValueError("confirmation evidence is immutable after the first attempt")
                return json.loads(row["result_json"])
            result = evaluate_v2(plan, rows)
            imported = result["status"] == "PASS"
            receipt = {
                "schema": "forge/parent-launch-baseline-confirmation/v1",
                "selection_id": selection["selection_id"],
                "screening_hash": selection["screening_hash"],
                "confirmation_plan_hash": hash_canonical(
                    "forge/parent-launch-baseline-plan/v1", plan
                ),
                "confirmation_rows_hash": hash_canonical(
                    "forge/parent-launch-baseline-rows/v1", rows
                ),
                "evaluation": result,
                "status": "IMPORTED_BASELINE" if imported else "KEEP_WAN50",
                "baseline_hash": selection["selected_candidate_hash"]
                if imported
                else selection["anchor_hash"],
                "reward_hotkey": None,
            }
            db.execute(
                "UPDATE parent_launch_baseline SET confirmation_plan_json=?, "
                "confirmation_rows_json=?, qualification_package_json=?, result_json=? WHERE id=1",
                (plan_encoded, rows_encoded, package_encoded, self._encoded(receipt)),
            )
            return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--screening", type=Path, action="append", required=True)
    choose.add_argument("--anchor-hash", required=True)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--plan", type=Path, required=True)
    confirm.add_argument("--rows", type=Path, required=True)
    confirm.add_argument("--qualification-package", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    store = ParentLaunchBaselineStore(args.db)
    if args.command == "select":
        values = [json.loads(path.read_text()) for path in args.screening]
        result = store.select_once(values, anchor_hash=args.anchor_hash)
        result["source_sha256"] = [sha256_file(path) for path in args.screening]
    else:
        result = store.confirm_once(
            json.loads(args.plan.read_text()),
            json.loads(args.rows.read_text()),
            json.loads(args.qualification_package.read_text()),
        )
    if args.out:
        write_immutable_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
