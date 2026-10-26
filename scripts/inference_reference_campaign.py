"""Run the bounded private H3 reference-verification campaign.

This is deliberately an operator-side *driver*, not a worker, wallet, RunPod,
or reward tool.  It owns only a native private campaign directory, sends the
pre-created challenges through :class:`forge.inference.service.Validator`, and
leaves closure and publication to the parent guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import time
from pathlib import Path
from typing import Any

from forge.inference.chain import atomic_json
from forge.inference.service import Validator

BATCH = "reference-20261023"
WINDOW_SECONDS = 120 * 60
POLL_SECONDS = 2
HOLDOUT_OFFSETS = (20 * 60, 36 * 60, 52 * 60, 68 * 60, 84 * 60, 100 * 60)
ATTACK_MODES = ("fast_profile", "replay", "wrong_video") * 2
PILOT_FIELDS = {
    "modality",
    "prompt",
    "seed",
    "duration",
    "resolution",
    "aspectRatio",
    "audio",
    "profile",
}


class CampaignError(ValueError):
    """A safe, operator-actionable campaign precondition failure."""


def _construction_failure(root: Path, error: Exception) -> None:
    """Persist only a class/code when setup fails before Validator exists."""
    try:
        _private_directory(root)
        atomic_json(
            root / "campaign-status.json",
            {
                "schema": "forge/inference-reference-campaign-status/v1",
                "batch": BATCH,
                "state": "INCONCLUSIVE",
                "phase": "construction_failed",
                "reason": f"driver_{type(error).__name__}",
                "readyForClosure": True,
                "publication": {
                    "publishable": False,
                    "controlledScore": None,
                    "controlledWeight": None,
                },
            },
        )
    except Exception:
        # A malformed/untrusted path must not turn failure handling into output
        # that leaks its details.  stderr below still contains only the class.
        pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _private_file(path: Path) -> None:
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise CampaignError("private_file_required")


def _private_directory(path: Path) -> None:
    if not path.is_absolute():
        raise CampaignError("absolute_campaign_root_required")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise CampaignError("private_campaign_root_required")


def _read_json(path: Path, *, private: bool = True) -> dict[str, Any]:
    if private:
        _private_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError("invalid_private_json") from error
    if not isinstance(value, dict):
        raise CampaignError("json_object_required")
    return value


def _parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PILOT_FIELDS:
        raise CampaignError("invalid_challenge_parameters")
    if (
        value.get("modality") != "text-to-video"
        or not isinstance(value.get("prompt"), str)
        or not value["prompt"].strip()
        or type(value.get("seed")) is not int
        or value.get("duration") != 5
        or value.get("resolution") != "480p"
        or value.get("aspectRatio") != "16:9"
        or value.get("audio") is not True
        or value.get("profile") != "base"
    ):
        raise CampaignError("invalid_challenge_parameters")
    return value


def _challenges(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    value = _read_json(path)
    calibration, holdouts = value.get("calibration"), value.get("holdouts")
    if not isinstance(calibration, list) or not isinstance(holdouts, list):
        raise CampaignError("challenge_sets_required")
    if len(calibration) != 2 or len(holdouts) != 6:
        raise CampaignError("challenge_count_required")
    values = [_parameters(item) for item in calibration + holdouts]
    # Seeds are selected before the campaign starts and are never generated or
    # emitted by this process.  Reusing one would make a purported holdout weak.
    if len({item["seed"] for item in values}) != len(values):
        raise CampaignError("challenge_seeds_must_be_unique")
    ceiling = value.get("ceilingMicros")
    if type(ceiling) is not int or ceiling <= 0:
        raise CampaignError("challenge_ceiling_required")
    return values[:2], values[2:], ceiling


def _config(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("network") != "test" or value.get("mechanism") != 1:
        raise CampaignError("testnet_mechanism_required")
    if value.get("referenceVerification") is not True:
        raise CampaignError("reference_verification_required")
    if value.get("calibrationBatch") != BATCH + "-calibration":
        raise CampaignError("calibration_batch_mismatch")
    if value.get("calibrationPerMiner") != 2 or value.get("holdoutsPerMiner") != 6:
        raise CampaignError("campaign_shape_mismatch")
    miners = value.get("miners")
    if (
        not isinstance(miners, list)
        or len(miners) != 2
        or not all(isinstance(item, dict) for item in miners)
    ):
        raise CampaignError("two_miners_required")
    controlled = [item for item in miners if item.get("uid") == 8]
    if len(controlled) != 1 or not isinstance(controlled[0].get("id"), str):
        raise CampaignError("controlled_uid8_required")
    if any(not isinstance(item.get("id"), str) for item in miners):
        raise CampaignError("miner_id_required")
    budget = value.get("challengeBudgetMicros")
    if type(budget) is not int or budget <= 0:
        raise CampaignError("challenge_budget_required")
    return value


def _proxy_config(path: Path, controlled: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    value = _read_json(path)
    required = {
        "schema",
        "listen",
        "upstream",
        "tokenEnv",
        "dataDirectory",
        "signingKeyFile",
        "modelManifestHash",
        "workflowHash",
        "schedule",
    }
    if (
        set(value) != required
        or value.get("schema") != "forge/inference-adversarial-worker/v1"
        or not isinstance(value.get("schedule"), dict)
        or (not resume and value["schedule"])
        or value.get("upstream", {}).get("workerId") != controlled.get("workerId")
    ):
        raise CampaignError("adversarial_proxy_configuration_invalid")
    return value


def _write_proxy_schedule(
    path: Path, proxy: dict[str, Any], job_id: str, mode: str, source: Path
) -> None:
    if mode not in ATTACK_MODES:
        raise CampaignError("invalid_controlled_mode")
    updated = dict(proxy)
    schedule = dict(proxy["schedule"])
    entry: dict[str, str] = {"mode": mode}
    if mode in {"replay", "wrong_video"}:
        if not source.is_absolute() or not source.is_file():
            raise CampaignError("controlled_source_unavailable")
        entry["source"] = str(source)
    schedule[job_id] = entry
    updated["schedule"] = schedule
    atomic_json(path, updated)
    # atomic_json creates mode 0600.  Check it here rather than trusting a
    # platform's rename defaults because the proxy reloads this file per POST.
    _private_file(path)
    proxy.clear()
    proxy.update(updated)


def _rows_by_id(validator: Validator, ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = {row["id"]: row for row in validator.rows()}
    if set(ids) - set(rows):
        raise CampaignError("durable_attempt_missing")
    return {job_id: rows[job_id] for job_id in ids}


def _count(rows: list[dict[str, Any]], *, batch: str, miner: str, status: str) -> int:
    return sum(
        row["batch"] == batch and row["miner"] == miner and row["status"] == status for row in rows
    )


class Campaign:
    def __init__(self, root: Path):
        _private_directory(root)
        self.root = root
        self.config_path = root / "validator-config.json"
        self.challenges_path = root / "challenges.json"
        self.proxy_path = root / "adversarial-config.json"
        self.status_path = root / "campaign-status.json"
        self.record_path = root / "campaign.json"
        self.config = _config(self.config_path)
        self.calibration, self.holdouts, self.ceiling = _challenges(self.challenges_path)
        attempts = len(self.config["miners"]) * (len(self.calibration) + len(self.holdouts))
        if self.ceiling * attempts > self.config["challengeBudgetMicros"]:
            raise CampaignError("challenge_ceiling_exceeds_budget")
        self.controlled = next(item for item in self.config["miners"] if item["uid"] == 8)
        self.honest = next(
            item for item in self.config["miners"] if item["id"] != self.controlled["id"]
        )
        prior = _read_json(self.record_path) if self.record_path.exists() else None
        if prior is not None and (
            prior.get("schema") != "forge/inference-reference-campaign/v1"
            or prior.get("batch") != BATCH
            or prior.get("challengeSetSha256") != _sha256(_read_json(self.challenges_path))
            or prior.get("validatorConfigSha256") != _sha256(self.config)
            or not isinstance(prior.get("startedAt"), (int, float))
            or not isinstance(prior.get("deadline"), (int, float))
        ):
            raise CampaignError("campaign_resume_mismatch")
        self.proxy = _proxy_config(self.proxy_path, self.controlled, resume=prior is not None)
        self.validator = Validator(
            root / "validator-data", self.config, self.config["challengeBudgetMicros"]
        )
        self.started = float(prior["startedAt"]) if prior else time.time()
        self.deadline = float(prior["deadline"]) if prior else self.started + WINDOW_SECONDS
        if abs((self.deadline - self.started) - WINDOW_SECONDS) > 1:
            raise CampaignError("campaign_window_mismatch")
        self.phase = str(prior.get("phase", "calibration")) if prior else "calibration"
        self.outcome = "RUNNING"
        self.reason: str | None = None
        self.frozen_seconds: float | None = prior.get("frozenSeconds") if prior else None
        source = prior.get("controlledSource") if prior else None
        self.controlled_source: Path | None = Path(source) if isinstance(source, str) else None
        if self.controlled_source is not None and not self.controlled_source.is_file():
            raise CampaignError("campaign_source_missing")
        self.availability: dict[str, bool | None] = {
            miner["id"]: None for miner in self.config["miners"]
        }
        self._record()
        self._status()

    def close(self) -> None:
        self.validator.close()

    def _record(self) -> None:
        # This durable index intentionally hashes challenge material rather than
        # duplicating prompts/seeds into a dashboard-visible projection.
        atomic_json(
            self.record_path,
            {
                "schema": "forge/inference-reference-campaign/v1",
                "batch": BATCH,
                "startedAt": self.started,
                "deadline": self.deadline,
                "windowSeconds": WINDOW_SECONDS,
                "challengeSetSha256": _sha256(_read_json(self.challenges_path)),
                "validatorConfigSha256": _sha256(self.config),
                "status": self.outcome,
                "reason": self.reason,
                "phase": self.phase,
                "frozenSeconds": self.frozen_seconds,
                "controlledSource": str(self.controlled_source) if self.controlled_source else None,
            },
        )

    def _status(self) -> None:
        rows = self.validator.rows()
        # No row is emitted: rows contain private prompt/seed data.  The parent
        # watcher can inspect the native SQLite database independently.
        value = {
            "schema": "forge/inference-reference-campaign-status/v1",
            "batch": BATCH,
            "state": self.outcome,
            "phase": self.phase,
            "reason": self.reason,
            "startedAt": self.started,
            "deadline": self.deadline,
            "remainingSeconds": max(0, int(self.deadline - time.time())),
            "calibration": {
                "honestVerified": _count(
                    rows, batch=BATCH + "-calibration", miner=self.honest["id"], status="verified"
                ),
                "controlledVerified": _count(
                    rows,
                    batch=BATCH + "-calibration",
                    miner=self.controlled["id"],
                    status="verified",
                ),
                "frozenSeconds": self.frozen_seconds,
            },
            "holdout": {
                "honestVerified": _count(
                    rows, batch=BATCH, miner=self.honest["id"], status="verified"
                ),
                "controlledInvalid": _count(
                    rows, batch=BATCH, miner=self.controlled["id"], status="invalid"
                ),
                "completedPairs": len(
                    {
                        row["params"]
                        for row in rows
                        if row["batch"] == BATCH and row["completed"] is not None
                    }
                ),
            },
            "availability": self.availability,
            "readyForClosure": self.outcome != "RUNNING",
            "publication": {
                "publishable": self.outcome == "PASS",
                "controlledScore": 0 if self.outcome == "PASS" else None,
                "controlledWeight": 0 if self.outcome == "PASS" else None,
            },
        }
        atomic_json(self.status_path, value)

    def _tick(self) -> None:
        self.validator.step()
        self._status()

    def _monitor_availability(self) -> None:
        """Observe health without creating a quote or a generation."""
        for miner_id, worker in self.validator.workers.items():
            try:
                response = worker.call("/health")
                self.availability[miner_id] = (
                    isinstance(response, dict) and response.get("available") is True
                )
            except Exception:
                self.availability[miner_id] = False

    def _collect_pair(self, ids: list[str]) -> bool:
        """Collect both outputs before a reference replay can occupy the GPU."""
        while time.time() < self.deadline:
            self._tick()
            rows = _rows_by_id(self.validator, ids)
            if all(
                row["status"]
                in {"auditing", "invalid", "failed", "cancelled", "timed_out", "inconclusive"}
                for row in rows.values()
            ):
                return True
            time.sleep(POLL_SECONDS)
        self._tick()
        return False

    def _audit_pair(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        # No step() here: the control GPU is reserved to the full reference
        # replay after both miner media are safely captured.
        self.validator.audit_pending()
        self._status()
        return _rows_by_id(self.validator, ids)

    def _calibrate(self) -> bool:
        calibration_batch = BATCH + "-calibration"
        if self.frozen_seconds is not None:
            rows = self.validator.rows()
            controlled = [
                r
                for r in rows
                if r["batch"] == calibration_batch
                and r["miner"] == self.controlled["id"]
                and r["status"] == "verified"
            ]
            if len(controlled) != 2:
                self.reason = "calibration_resume_mismatch"
                return False
            self.controlled_source = (
                self.root / "validator-data" / f"{controlled[-1]['id']}.mp4"
            ).resolve()
            if not self.controlled_source.is_file():
                self.reason = "calibration_source_unavailable"
                return False
            self.phase = "holdouts"
            return True
        for parameters in self.calibration:
            if time.time() >= self.deadline:
                self.reason = "window_expired_during_calibration"
                return False
            ids = self.validator.enqueue(calibration_batch, parameters, ceiling=self.ceiling)
            self._status()
            existing = _rows_by_id(self.validator, ids)
            if all(row["status"] == "verified" for row in existing.values()):
                audited = existing
            else:
                if not self._collect_pair(ids):
                    self.reason = "calibration_collection_incomplete"
                    return False
                audited = self._audit_pair(ids)
            if any(row["status"] != "verified" for row in audited.values()):
                self.reason = "calibration_not_verified"
                return False
            controlled_id = next(
                job_id for job_id, row in audited.items() if row["miner"] == self.controlled["id"]
            )
            source = (self.root / "validator-data" / f"{controlled_id}.mp4").resolve()
            if self.controlled_source is None and source.is_file():
                self.controlled_source = source
        if self.controlled_source is None:
            self.reason = "calibration_source_unavailable"
            return False
        try:
            self.frozen_seconds = self.validator.freeze_calibration(BATCH)
        except ValueError:
            self.reason = "calibration_freeze_failed"
            return False
        self.phase = "holdouts"
        self._status()
        return True

    def _wait_until(self, target: float) -> bool:
        while time.time() < target:
            if time.time() >= self.deadline:
                return False
            self._status()
            time.sleep(min(POLL_SECONDS, max(0.0, target - time.time())))
        return time.time() < self.deadline

    def _holdouts(self) -> bool:
        assert self.controlled_source is not None
        for parameters, offset, mode in zip(
            self.holdouts, HOLDOUT_OFFSETS, ATTACK_MODES, strict=True
        ):
            target = self.started + offset
            # A delayed calibration may not compress the schedule or overlap
            # work.  The predetermined slot is missed, so evidence is inconclusive.
            if time.time() > target:
                self.reason = "holdout_schedule_missed"
                return False
            if not self._wait_until(target):
                self.reason = "holdout_schedule_missed"
                return False
            ids = self.validator.enqueue(BATCH, parameters, ceiling=self.ceiling)
            rows = _rows_by_id(self.validator, ids)
            controlled_job = next(
                job_id for job_id, row in rows.items() if row["miner"] == self.controlled["id"]
            )
            # This write happens before the next step(), hence before either
            # worker receives its execution POST.  The proxy reloads per POST.
            current = self.proxy["schedule"].get(controlled_job)
            if current is None:
                _write_proxy_schedule(
                    self.proxy_path, self.proxy, controlled_job, mode, self.controlled_source
                )
            elif current.get("mode") != mode:
                self.reason = "controlled_schedule_mismatch"
                return False
            self._status()
            if all(row["status"] in {"verified", "invalid"} for row in rows.values()):
                audited = rows
            else:
                if not self._collect_pair(ids):
                    self.reason = "holdout_collection_incomplete"
                    return False
                audited = self._audit_pair(ids)
            honest = next(row for row in audited.values() if row["miner"] == self.honest["id"])
            controlled = next(
                row for row in audited.values() if row["miner"] == self.controlled["id"]
            )
            if honest["status"] != "verified" or controlled["status"] != "invalid":
                self.reason = "holdout_verification_unexpected"
                return False
        return True

    def _final_outcome(self) -> str:
        rows = self.validator.rows()
        calibration_ok = all(
            _count(rows, batch=BATCH + "-calibration", miner=miner["id"], status="verified") == 2
            for miner in (self.honest, self.controlled)
        )
        honest_ok = _count(rows, batch=BATCH, miner=self.honest["id"], status="verified") == 6
        controlled_rows = [
            r for r in rows if r["batch"] == BATCH and r["miner"] == self.controlled["id"]
        ]
        expected_reason = {
            _canonical(parameters).decode(): (
                "verification_failed" if mode == "wrong_video" else "confirmed_substitution"
            )
            for parameters, mode in zip(self.holdouts, ATTACK_MODES, strict=True)
        }
        controlled_invalid = len(controlled_rows) == 6 and all(
            r["status"] == "invalid" and r["reason"] == expected_reason.get(r["params"])
            for r in controlled_rows
        )
        controlled_verified = _count(
            rows, batch=BATCH, miner=self.controlled["id"], status="verified"
        )
        honest_invalid = _count(rows, batch=BATCH, miner=self.honest["id"], status="invalid")
        if calibration_ok and honest_ok and controlled_invalid:
            return "PASS"
        if controlled_verified or honest_invalid:
            return "FAIL"
        return "INCONCLUSIVE"

    def _request_closure(self) -> None:
        # Parent orchestration must opt in explicitly; this driver never touches
        # a pod/provider and writes only its own campaign directory.
        closure = self.config.get("closureOrchestrator")
        if closure != {"enabled": True}:
            return
        atomic_json(
            self.root / "finish-requested.json",
            {
                "schema": "forge/inference-finish-request/v1",
                "batch": BATCH,
                "state": self.outcome,
                "readyForClosure": True,
            },
        )

    def run(self) -> str:
        try:
            if not self._calibrate():
                self.outcome = "INCONCLUSIVE"
            elif not self._holdouts():
                self.outcome = self._final_outcome()
            else:
                # Continue availability-only observation to the original 120 min
                # endpoint.  No extra quote, POST, or generation is created.
                self.phase = "availability_monitoring"
                while time.time() < self.deadline:
                    self._monitor_availability()
                    self._status()
                    time.sleep(min(POLL_SECONDS, max(0.0, self.deadline - time.time())))
                self.outcome = self._final_outcome()
            self.phase = "finished"
            self._record()
            self._status()
            self._request_closure()
            return self.outcome
        except Exception as error:
            self.phase = "finished"
            self.outcome = "INCONCLUSIVE"
            self.reason = f"driver_{type(error).__name__}"
            self._record()
            self._status()
            self._request_closure()
            return self.outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private bounded H3 reference campaign")
    parser.add_argument(
        "--root", type=Path, required=True, help="absolute private campaign directory"
    )
    args = parser.parse_args()
    campaign: Campaign | None = None
    try:
        campaign = Campaign(args.root)
        outcome = campaign.run()
    except Exception as error:
        # Do not print exception messages: config and worker errors can contain
        # a URL or private prompt.  A durable campaign status, if construction
        # succeeded, contains only this sanitized class name.
        if campaign is None:
            _construction_failure(args.root, error)
        print(type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if campaign is not None:
            campaign.close()
    print(outcome)
    raise SystemExit(0 if outcome == "PASS" else 1)


if __name__ == "__main__":
    main()
