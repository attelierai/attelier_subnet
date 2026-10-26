"""Durable private-testnet challenges against registered H3 workers.

Unknown acceptance is retained and reconciled with the same worker identity.
The operator controls this pilot's worker list; this is not permissionless
endpoint discovery or a proof that a miner owns the declared GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from forge.inference.media import inspect_video
from forge.inference.reference import SCHEMA, replay, verify_receipt
from forge.inference.reference import sha256 as reference_hash
from forge.inference.scoring import (
    TRACE_TRUSTED_RENDER,
    trace_trusted_render_score,
)
from forge.inference.scoring import (
    score as availability_score,
)
from forge.inference.trace_protocol import (
    REQUEST_SCHEMA as TRACE_REQUEST_SCHEMA,
)
from forge.inference.trace_protocol import (
    TraceProtocolError,
    effective_seed,
    load_profile_envelope,
    qualified_lane,
    validate_receipt_metadata,
)
from forge.inference.trace_protocol import (
    file_sha256 as trace_file_sha256,
)
from forge.inference.trace_protocol import (
    read_json as trace_read_json,
)
from forge.inference.trace_protocol import (
    sha256 as trace_hash,
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RegistrationUnavailable(ValueError):
    """The finalized registration projection cannot be used for this attempt."""


class MinerIdentityInvalid(ValueError):
    """A configured live miner cannot prove the identity it declared."""


class WorkerCredentialUnavailable(ValueError):
    """The validator cannot authenticate to its configured worker."""


class WorkerBusy(ValueError):
    """The worker rejected this job before accepting it."""


def _trace_mode(config: dict[str, Any]) -> bool:
    return config.get("verificationMode") == TRACE_TRUSTED_RENDER


class Worker:
    def __init__(self, miner: dict[str, Any]):
        self.miner = miner
        parsed = urllib.parse.urlparse(miner["url"])
        if (
            (
                parsed.scheme != "https"
                and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"))
            )
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Worker requires HTTPS or a local tunnel")
        self.token_env = miner["tokenEnv"]

    def _token(self) -> str:
        """Read a credential only for a network request, never for local scoring."""
        token = os.environ.get(self.token_env, "")
        if len(token) < 32:
            raise WorkerCredentialUnavailable("Worker credential unavailable")
        return token

    def call(
        self,
        route: str,
        body: Any = None,
        destination: Path | None = None,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> Any:
        request = urllib.request.Request(
            self.miner["url"].rstrip("/") + route,
            data=None if body is None else canonical(body).encode(),
            headers={
                "Authorization": "Bearer " + self._token(),
                "Content-Type": "application/json",
            },
        )

        # Never follow a redirect carrying the worker credential.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            # Error responses remain authenticated worker protocol messages too.
            # Do not classify a response from another worker as a capacity result.
            if error.headers.get("x-worker-id") != self.miner["workerId"]:
                raise ValueError("worker_identity_changed") from error
            payload = error.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError("worker_response_too_large") from error
            try:
                code = json.loads(payload).get("error", {}).get("code")
            except (AttributeError, TypeError, json.JSONDecodeError):
                code = None
            if error.code == 409 and code == "gpu_busy":
                raise WorkerBusy("gpu_busy") from error
            if error.code == 401 and code == "unauthorized":
                raise WorkerCredentialUnavailable("worker unauthorized") from error
            # Other HTTP failures retain the existing transient/unreachable
            # classification. Their protocol has no safe service-score meaning.
            raise
        with response:
            if response.headers.get("x-worker-id") != self.miner["workerId"]:
                raise ValueError("worker_identity_changed")
            if destination:
                total = 0
                with destination.open("xb") as output:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("result_too_large")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                return response.headers.get("x-content-sha256")
            payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError("worker_response_too_large")
            return json.loads(payload)


class Validator:
    def __init__(self, directory: Path, config: dict[str, Any], budget_micros: int):
        if config.get("network") != "test" or config.get("mechanism") != 1:
            raise ValueError("Only testnet mechanism 1 is enabled")
        if budget_micros <= 0:
            raise ValueError("Explicit challenge cost limit required")
        self.directory, self.config, self.budget = directory, config, budget_micros
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(directory / "inference.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS attempts(
            id TEXT PRIMARY KEY, batch TEXT NOT NULL, miner TEXT NOT NULL,
            worker TEXT NOT NULL, params TEXT NOT NULL, quote TEXT, ceiling INTEGER NOT NULL,
            started REAL NOT NULL, deadline REAL NOT NULL, completed REAL,
            status TEXT NOT NULL, reason TEXT, result TEXT, judgment TEXT,
            nonce TEXT, sent REAL, generation REAL, verification TEXT, reference TEXT,
            UNIQUE(batch,miner,params));
          CREATE TABLE IF NOT EXISTS reference_windows(
            batch TEXT PRIMARY KEY, frozen_seconds REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS trace_windows(
            batch TEXT PRIMARY KEY, started REAL NOT NULL, expires REAL NOT NULL,
            min_completed INTEGER NOT NULL, min_verified INTEGER NOT NULL);
        """)
        # Existing pilot receipts remain readable; new audit rows carry these fields.
        existing = {r[1] for r in self.db.execute("PRAGMA table_info(attempts)")}
        for column, kind in (
            ("nonce", "TEXT"),
            ("sent", "REAL"),
            ("generation", "REAL"),
            ("verification", "TEXT"),
            ("reference", "TEXT"),
            ("sent_boot", "TEXT"),
            ("trace_request", "TEXT"),
            ("trace_profile", "TEXT"),
            ("trace_receipt", "TEXT"),
            ("trace_result", "TEXT"),
            ("reaudit_deadline", "REAL"),
            ("reaudit_attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("trace_custody_sealed", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in existing:
                self.db.execute(f"ALTER TABLE attempts ADD COLUMN {column} {kind}")
        trace_columns = {r[1] for r in self.db.execute("PRAGMA table_info(trace_windows)")}
        if "reference_seconds" not in trace_columns:
            self.db.execute("ALTER TABLE trace_windows ADD COLUMN reference_seconds REAL")
        self.workers = {m["id"]: Worker(m) for m in config["miners"]}
        if _trace_mode(config):
            self._validate_trace_config()

    def close(self) -> None:
        self.db.close()

    def _trace_config(self) -> dict[str, Any]:
        value = self.config.get("traceVerification")
        if not isinstance(value, dict):
            raise ValueError("traceVerification configuration is required")
        return value

    def _validate_trace_config(self) -> None:
        trace = self._trace_config()
        identity = trace.get("profile")
        if not (
            isinstance(identity, dict)
            and set(identity) == {"id", "hash"}
            and all(isinstance(identity.get(key), str) and identity[key] for key in identity)
        ):
            raise ValueError("traceVerification.profile identity is required")
        root = Path(trace.get("profileRoot", ""))
        command = trace.get("command")
        if (
            not root.is_dir()
            or not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError("traceVerification runtime configuration is invalid")
        origin, output_root = trace.get("comfyOrigin"), trace.get("comfyOutputRoot")
        if (
            not isinstance(origin, str)
            or not origin.startswith("http://127.0.0.1:")
            or not Path(output_root or "").is_dir()
            or not Path(trace.get("runtimeAttestation", "")).is_file()
        ):
            raise ValueError("traceVerification must use an operator-local Comfy runtime")
        for field, minimum in (
            ("timeoutSeconds", 1),
            ("reauditSeconds", 1),
            ("windowSeconds", 1),
            ("minCompleted", 1),
            ("minVerified", 1),
        ):
            if not isinstance(trace.get(field), int) or trace[field] < minimum:
                raise ValueError(f"traceVerification.{field} is required")
        reference = trace.get("referenceSeconds")
        if reference is not None and (
            isinstance(reference, bool)
            or not isinstance(reference, (float, int))
            or not math.isfinite(reference)
            or reference <= 0
        ):
            raise ValueError("traceVerification.referenceSeconds must be a positive measurement")

    def _trace_profile(self) -> tuple[dict[str, Any], dict[str, Any]]:
        trace = self._trace_config()
        identity = trace["profile"]
        root = Path(trace["profileRoot"])
        envelope = load_profile_envelope(root, identity)
        profile = envelope["profile"]
        qualification = envelope.get("qualification")
        if not isinstance(qualification, dict) or type(qualification.get("expiresAt")) is not int:
            raise ValueError("trace_profile_qualification_invalid")
        return profile, qualification

    def _check_trace_attempt_lane(self, row: sqlite3.Row, miner: dict[str, Any]) -> None:
        profile, qualification = self._trace_profile()
        lane = qualified_lane(profile, qualification, miner["hotkey"], miner["workerId"])
        snapshot = json.loads(row["trace_profile"] or "{}")
        if snapshot.get("lane") != lane:
            raise TraceProtocolError("trace_attempt_lane_changed", "inconclusive")

    def _trace_window(self, batch: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM trace_windows WHERE batch=?", (batch,)).fetchone()
        if row:
            return row
        trace = self._trace_config()
        started = time.time()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO trace_windows("
                "batch,started,expires,min_completed,min_verified,reference_seconds)"
                " VALUES(?,?,?,?,?,?)",
                (
                    batch,
                    started,
                    started + trace["windowSeconds"],
                    trace["minCompleted"],
                    trace["minVerified"],
                    trace.get("referenceSeconds"),
                ),
            )
        return self.db.execute("SELECT * FROM trace_windows WHERE batch=?", (batch,)).fetchone()

    def _trace_lane_suspended(self, batch: str, miner_id: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM attempts WHERE batch=? AND miner=? AND reason IN (?,?) LIMIT 1",
            (batch, miner_id, "trace_reaudit_deadline_expired", "trace_reaudit_exhausted"),
        ).fetchone() is not None

    def _registration(self) -> None:
        from datetime import datetime

        try:
            value = json.loads(Path(self.config["registrationFile"]).read_text())
            age = time.time() - datetime.fromisoformat(value["generatedAt"]).timestamp()
            if (
                value["network"] != "test"
                or value["netuid"] != self.config["netuid"]
                or value["mechanismCount"] < 2
                or not -10 <= age <= 120
            ):
                raise RegistrationUnavailable("stale_or_wrong_registration")
            if any(value["uids"].get(str(m["uid"])) != m["hotkey"] for m in self.config["miners"]):
                raise RegistrationUnavailable("registration_changed")
        except RegistrationUnavailable:
            raise
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RegistrationUnavailable("registration_unavailable") from exc

    def _verify_miner(self, miner: dict[str, Any]) -> None:
        if self.config.get("source") == "live":
            from forge.inference.identity import verify

            try:
                verify(miner, self.config["netuid"])
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise MinerIdentityInvalid("miner_identity_invalid") from exc

    def _registered(self) -> None:
        """Global admission gate used before creating or publishing a batch."""
        self._registration()
        for miner in self.config["miners"]:
            self._verify_miner(miner)

    def _admit(self, miner: dict[str, Any]) -> None:
        """Freshly authorize this miner immediately before an execution POST."""
        self._registration()
        self._verify_miner(miner)

    def enqueue(
        self, batch: str, parameters: dict[str, Any], *, ceiling: int, timeout: int = 1200
    ) -> list[str]:
        self._registered()
        if (
            parameters.get("modality") != "text-to-video"
            or parameters.get("duration") != 5
            or parameters.get("resolution") != "480p"
            or parameters.get("aspectRatio") != "16:9"
            or parameters.get("profile") != "base"
            or parameters.get("audio") is not True
        ):
            raise ValueError("Only the shared pilot contract is enabled")
        if not 0 < ceiling <= self.budget or not 30 <= timeout <= 3600:
            raise ValueError("Invalid challenge limits")
        # Reserve every attempted job, including failed and ambiguous acceptance.
        # This intentionally never assumes a provider refunded a failed job.
        ids = []
        trace_profile = trace_qualification = None
        if _trace_mode(self.config):
            self._trace_window(batch)
            trace_profile, trace_qualification = self._trace_profile()
        with self.db:
            for miner in self.config["miners"]:
                existing = self.db.execute(
                    "SELECT id FROM attempts WHERE batch=? AND miner=? AND params=?",
                    (batch, miner["id"], canonical(parameters)),
                ).fetchone()
                if existing:
                    ids.append(existing["id"])
                    continue
                if _trace_mode(self.config) and self._trace_lane_suspended(batch, miner["id"]):
                    raise ValueError("trace_lane_admission_suspended")
                spent = self.db.execute("SELECT coalesce(sum(ceiling),0) FROM attempts").fetchone()[
                    0
                ]
                if spent + ceiling > self.budget:
                    raise ValueError("Challenge budget exhausted")
                job_id = str(uuid.uuid4())
                nonce = secrets.token_hex(32)
                trace_request = None
                trace_profile_snapshot = None
                if trace_profile is not None and trace_qualification is not None:
                    lane = qualified_lane(
                        trace_profile, trace_qualification, miner["hotkey"], miner["workerId"]
                    )
                    identity = self._trace_config()["profile"]
                    effective = effective_seed(job_id, nonce, parameters["seed"], identity["hash"])
                    trace_request = {
                        "schema": TRACE_REQUEST_SCHEMA,
                        "jobId": job_id,
                        "workerId": miner["workerId"],
                        "nonce": nonce,
                        "requestedSeed": parameters["seed"],
                        "profile": identity,
                        "parameters": {**parameters, "seed": effective},
                    }
                    trace_profile_snapshot = {
                        "id": identity["id"],
                        "hash": identity["hash"],
                        "rendererHash": trace_profile["rendererHash"],
                        "qualificationExpiresAt": trace_qualification["expiresAt"],
                        "lane": lane,
                    }
                self.db.execute(
                    "INSERT INTO attempts(id,batch,miner,worker,params,ceiling,"
                    "started,deadline,status,nonce,trace_request,trace_profile)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        batch,
                        miner["id"],
                        miner["workerId"],
                        canonical(parameters),
                        ceiling,
                        time.time(),
                        time.time() + timeout,
                        "queued",
                        nonce,
                        canonical(trace_request) if trace_request else None,
                        canonical(trace_profile_snapshot) if trace_profile_snapshot else None,
                    ),
                )
                ids.append(job_id)
        return ids

    def _update(self, job_id: str, **values: Any) -> None:
        with self.db:
            self.db.execute(
                "UPDATE attempts SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
                (*values.values(), job_id),
            )

    @staticmethod
    def _boot_id() -> str:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def _seal_trace_custody(self, row: sqlite3.Row, elapsed: float) -> None:
        """Add only the first effective receipt download to worker time.

        The validator queue before this method and all retries are deliberately
        excluded. A reboot makes monotonic elapsed time incomparable, so the
        availability result remains usable but its speed component becomes unknown.
        """
        if row["trace_custody_sealed"]:
            return
        values: dict[str, Any] = {"trace_custody_sealed": 1}
        if row["sent_boot"] == self._boot_id() and row["generation"] is not None:
            values["generation"] = max(0.0, float(row["generation"]) + elapsed)
        else:
            values["generation"] = None
        self._update(row["id"], **values)

    def _trace_evidence(self, row: sqlite3.Row, worker: Worker) -> tuple[Path, dict[str, Any]]:
        """Download a receipt and its indexed chunks into one private job directory."""
        self._check_trace_attempt_lane(row, worker.miner)
        try:
            receipt = worker.call(f"/jobs/{row['id']}/evidence")
            request = json.loads(row["trace_request"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceProtocolError("trace_receipt_unavailable", "inconclusive") from exc
        profile, _ = self._trace_profile()
        key = worker.miner.get("verificationPublicKey")
        if not isinstance(key, str) or not key:
            raise TraceProtocolError("worker_public_key_missing", "inconclusive")
        metadata = validate_receipt_metadata(request, receipt, profile, key)
        chunks = metadata["chunks"]
        parent = self.directory / "trace-evidence"
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if parent.is_symlink() or not parent.is_dir():
            raise TraceProtocolError("trace_evidence_root", "inconclusive")
        root = parent / row["id"]
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or not root.is_dir():
            raise TraceProtocolError("trace_evidence_root", "inconclusive")
        receipt_path = root / "receipt.json"
        encoded = canonical(receipt).encode()
        if receipt_path.exists():
            if receipt_path.read_bytes() != encoded:
                raise TraceProtocolError("trace_receipt_changed", "inconclusive")
        else:
            with receipt_path.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        for descriptor in chunks:
            index, name, digest, size = (
                descriptor["index"],
                descriptor["name"],
                descriptor["sha256"],
                descriptor["bytes"],
            )
            relative = Path(name)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory = root
            for part in relative.parts[:-1]:
                directory /= part
                if directory.is_symlink() or not directory.is_dir():
                    raise TraceProtocolError("trace_chunk_symlink", "inconclusive")
            if target.exists():
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or trace_file_sha256(target) != digest
                ):
                    raise TraceProtocolError("trace_chunk_changed", "inconclusive")
                continue
            temporary = target.with_suffix(target.suffix + ".partial")
            if temporary.exists() or temporary.is_symlink():
                if temporary.is_symlink():
                    raise TraceProtocolError("trace_chunk_symlink", "inconclusive")
                temporary.unlink()
            header = worker.call(
                f"/jobs/{row['id']}/evidence/{index}",
                destination=temporary,
                max_bytes=size,
            )
            actual = trace_file_sha256(temporary)
            if header != actual or actual != digest:
                temporary.unlink(missing_ok=True)
                raise TraceProtocolError("trace_chunk_transport_hash", "inconclusive")
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        return root, receipt

    def _trace_runtime_result(
        self, row: sqlite3.Row, evidence_root: Path, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke the local, configured trusted runtime; never a remote adapter by default."""
        trace = self._trace_config()
        request = json.loads(row["trace_request"])
        request_path = evidence_root / "request.json"
        request_bytes = canonical(request).encode()
        if request_path.exists():
            if request_path.read_bytes() != request_bytes:
                raise TraceProtocolError("trusted_request_changed", "inconclusive")
        else:
            request_path.write_bytes(request_bytes)
            os.chmod(request_path, 0o600)
        key_path = evidence_root / "worker-public-key.txt"
        key = self.workers[row["miner"]].miner.get("verificationPublicKey")
        if not isinstance(key, str) or not key:
            raise TraceProtocolError("worker_public_key_missing", "inconclusive")
        if key_path.exists():
            if key_path.read_text() != key + "\n":
                raise TraceProtocolError("worker_public_key_changed", "inconclusive")
        else:
            key_path.write_text(key + "\n")
            os.chmod(key_path, 0o600)
        attempt = int(row["reaudit_attempts"] or 0)
        output = self.directory / "trace-runtime" / row["id"] / f"attempt-{attempt}"
        if output.exists():
            raise TraceProtocolError("runtime_output_exists", "inconclusive")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = [
            *trace["command"],
            "--request",
            str(request_path),
            "--hotkey",
            json.loads(row["trace_profile"])["lane"]["hotkey"],
            "--receipt",
            str(evidence_root / "receipt.json"),
            "--profile-root",
            str(Path(trace["profileRoot"]).resolve()),
            "--evidence-root",
            str(evidence_root),
            "--worker-public-key-file",
            str(key_path),
            "--output-dir",
            str(output),
            "--runtime-attestation",
            str(Path(trace["runtimeAttestation"]).resolve()),
            "--comfy-origin",
            trace["comfyOrigin"],
            "--comfy-output-root",
            str(Path(trace["comfyOutputRoot"]).resolve()),
            "--timeout-seconds",
            str(trace["timeoutSeconds"]),
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=trace["timeoutSeconds"],
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TraceProtocolError("trusted_runtime_unavailable", "inconclusive") from exc
        if completed.returncode:
            raise TraceProtocolError("trusted_runtime_failed", "inconclusive")
        result_path = output / "report.json"
        result = trace_read_json(result_path)
        self._validate_trace_result(row, receipt, result, output)
        return result

    def _validate_trace_result(
        self, row: sqlite3.Row, receipt: dict[str, Any], result: dict[str, Any], output: Path
    ) -> None:
        """Bind the runtime verdict to this stored request, receipt and local renderer bytes."""
        request = json.loads(row["trace_request"])
        snapshot = json.loads(row["trace_profile"])
        if result.get("schema") != "forge-h3-verified-result/v1":
            raise TraceProtocolError("runtime_result_schema", "inconclusive")
        if result.get("jobId") != row["id"] or result.get("requestHash") != trace_hash(request):
            raise TraceProtocolError("runtime_result_request_binding", "inconclusive")
        if result.get("receiptHash") != trace_hash(receipt):
            raise TraceProtocolError("runtime_result_receipt_binding", "inconclusive")
        if result.get("profile") != {"id": snapshot["id"], "hash": snapshot["hash"]}:
            raise TraceProtocolError("runtime_result_profile_binding", "inconclusive")
        if result.get("traceRoot") != receipt.get("trace", {}).get("root"):
            raise TraceProtocolError("runtime_result_trace_binding", "inconclusive")
        if result.get("rendererHash") != snapshot["rendererHash"]:
            raise TraceProtocolError("runtime_result_renderer_binding", "inconclusive")
        if result.get("lane") != snapshot.get("lane"):
            raise TraceProtocolError("runtime_result_lane_binding", "inconclusive")
        if result.get("verdict") not in {"verified", "invalid", "inconclusive"} or not isinstance(
            result.get("reason"), str
        ):
            raise TraceProtocolError("runtime_result_verdict", "inconclusive")
        if result["verdict"] != "verified":
            if "result" in result:
                raise TraceProtocolError("runtime_result_unexpected_media", "inconclusive")
            # A failed audit cannot have passed every check or reached rendering.
            # Request/receipt/profile bindings above still identify this attempt.
            return
        checks = result.get("checks")
        predictions = checks.get("predictions") if isinstance(checks, dict) else None
        if not isinstance(checks, dict) or any(
            checks.get(key) != "pass" for key in ("conditioning", "recurrence", "terminalBinding")
        ):
            raise TraceProtocolError("runtime_result_checks", "inconclusive")
        if not isinstance(predictions, dict) or any(
            type(predictions.get(key)) is not int or predictions[key] < 0
            for key in ("checked", "matched", "total")
        ):
            raise TraceProtocolError("runtime_result_checks", "inconclusive")
        if result["verdict"] == "verified" and predictions != {
            "checked": 20,
            "matched": 20,
            "total": 20,
        }:
            raise TraceProtocolError("runtime_result_not_full20", "inconclusive")
        if not isinstance(result.get("timingsMs"), dict) or any(
            type(value) is not int or value < 0 for value in result["timingsMs"].values()
        ):
            raise TraceProtocolError("runtime_result_timings", "inconclusive")
        final_latent = result.get("finalLatent")
        if not (
            isinstance(final_latent, dict)
            and set(final_latent) == {"video", "audio"}
            and all(
                isinstance(value, str)
                and len(value) == 64
                and not set(value) - set("0123456789abcdef")
                for value in final_latent.values()
            )
        ):
            raise TraceProtocolError("runtime_result_final_latent", "inconclusive")
        try:
            expected_latent = {
                item["component"]: receipt["trace"]["chunks"][item["chunkIndex"]]["sha256"]
                for item in receipt["finalLatent"]
            }
        except (KeyError, TypeError, IndexError) as exc:
            raise TraceProtocolError("runtime_result_final_latent_binding", "inconclusive") from exc
        if final_latent != expected_latent:
            raise TraceProtocolError("runtime_result_final_latent_binding", "inconclusive")
        rendered = result.get("result")
        if not isinstance(rendered, dict) or set(rendered) != {"path", "sha256", "bytes"}:
            raise TraceProtocolError("runtime_result_media", "inconclusive")
        path = Path(rendered["path"]).resolve()
        try:
            path.relative_to(output.resolve())
        except ValueError as exc:
            raise TraceProtocolError("runtime_result_media_path", "inconclusive") from exc
        if (
            not path.is_file()
            or type(rendered["bytes"]) is not int
            or rendered["bytes"] <= 0
            or not isinstance(rendered["sha256"], str)
            or len(rendered["sha256"]) != 64
            or set(rendered["sha256"]) - set("0123456789abcdef")
            or path.stat().st_size != rendered["bytes"]
            or trace_file_sha256(path) != rendered["sha256"]
        ):
            raise TraceProtocolError("runtime_result_media_custody", "inconclusive")

    def step(self) -> list[dict[str, Any]]:
        for row in self.db.execute(
            "SELECT * FROM attempts WHERE "
            "(completed IS NULL AND status NOT IN ('auditing','audit_pending'))"
            " OR status='timed_out'"
            " ORDER BY started,id"
        ).fetchall():
            worker = self.workers.get(row["miner"])
            if worker is None or worker.miner["workerId"] != row["worker"]:
                self._update(
                    row["id"],
                    status="inconclusive",
                    reason="worker_configuration_changed",
                    **(
                        {"completed": row["completed"] or time.time()}
                        if time.time() >= row["deadline"]
                        else {}
                    ),
                )
                continue
            params = json.loads(row["params"])
            try:
                job = worker.call(f"/jobs/{row['id']}")
                if time.time() >= row["deadline"]:
                    if job is not None and job["status"] not in (
                        "succeeded",
                        "failed",
                        "cancelled",
                    ):
                        worker.call(f"/jobs/{row['id']}/cancel", {})
                    if job is None and row["status"] == "queued" and row["reason"] != "gpu_busy":
                        self._update(
                            row["id"],
                            status="inconclusive",
                            reason="challenge_not_submitted",
                            completed=row["completed"] or time.time(),
                        )
                    else:
                        self._update(
                            row["id"],
                            status="timed_out",
                            reason="challenge_deadline",
                            completed=row["completed"] or time.time(),
                        )
                    continue
                if job is None:
                    quote = json.loads(row["quote"]) if row["quote"] else None
                    if not quote or quote["expiresAt"] <= time.time() * 1000:
                        quote = worker.call(
                            "/challenge-quotes"
                            if self.config.get("referenceVerification") or _trace_mode(self.config)
                            else "/quotes",
                            {
                                k: params[k]
                                for k in (
                                    "modality",
                                    "duration",
                                    "resolution",
                                    "aspectRatio",
                                    "audio",
                                    "profile",
                                )
                            }
                            if self.config.get("referenceVerification") or _trace_mode(self.config)
                            else params,
                        )
                        amount = quote.get("amountMicros")
                        if (
                            type(amount) is not int
                            or not 0 <= amount <= row["ceiling"]
                            or not math.isfinite(quote["expiresAt"])
                            or quote["expiresAt"] <= time.time() * 1000
                        ):
                            raise ValueError("quote_exceeds_challenge_limit")
                        self._update(row["id"], quote=canonical(quote))
                    # A recovered `submitting` row may have lost its first POST.
                    # It receives the same fresh admission as an initial request;
                    # GET and cancellation above intentionally remain available
                    # for recovery even if the current registry later goes stale.
                    self._admit(worker.miner)
                    if _trace_mode(self.config):
                        self._check_trace_attempt_lane(row, worker.miner)
                    # Persist send intent before any POST; restart GETs this same ID.
                    self._update(
                        row["id"],
                        status="submitting",
                        sent=row["sent"] if row["sent"] is not None else time.monotonic(),
                        sent_boot=row["sent_boot"]
                        or Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    )
                    # The nonce is intentionally absent from quote/GET; it first
                    # becomes visible in this authenticated execution POST.
                    worker.call(
                        f"/jobs/{row['id']}",
                        {
                            "parameters": params,
                            "quote": quote,
                            **(
                                {
                                    "verification": {
                                        "schema": SCHEMA,
                                        "nonce": row["nonce"],
                                        "parametersHash": reference_hash(params),
                                    }
                                }
                                if self.config.get("referenceVerification")
                                else {}
                            ),
                            **(
                                {"traceRequest": json.loads(row["trace_request"])}
                                if _trace_mode(self.config) and row["trace_request"]
                                else {}
                            ),
                        },
                    )
                    self._update(row["id"], status="running", reason=None)
                elif job["status"] == "succeeded":
                    if _trace_mode(self.config):
                        if job.get("metrics", {}).get("evidenceReady") is not True:
                            raise ValueError("trace_evidence_not_ready")
                        # Monotonic timestamps survive ordinary restarts but not a boot.
                        # Keep availability evidence; make speed unknown on clock discontinuity.
                        generation = (
                            max(0.0, time.monotonic() - (row["sent"] or time.monotonic()))
                            if row["sent_boot"] == self._boot_id()
                            else None
                        )
                        self._update(
                            row["id"],
                            status="auditing",
                            reason=None,
                            generation=generation,
                            reaudit_attempts=0,
                            trace_custody_sealed=0,
                            reaudit_deadline=time.time() + self._trace_config()["reauditSeconds"],
                        )
                        continue
                    path = self.directory / f"{row['id']}.mp4"
                    if not path.exists():
                        temp = path.with_suffix(".partial")
                        if temp.exists():
                            temp.unlink()  # This attempt's bounded incomplete transfer only.
                        header = worker.call(f"/jobs/{row['id']}/result", destination=temp)
                        with temp.open("rb") as stream:
                            digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        if not header or header != digest:
                            raise ValueError("transport_hash_mismatch")
                        temp.replace(path)
                    generation = max(0.0, time.monotonic() - (row["sent"] or time.monotonic()))
                    if self.config.get("referenceVerification"):
                        if (
                            row["sent"] is None
                            or row["sent_boot"]
                            != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                        ):
                            self._update(
                                row["id"],
                                status="inconclusive",
                                reason="measurement_clock_changed",
                                completed=time.time(),
                            )
                            continue
                        with path.open("rb") as stream:
                            evidence = {
                                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                "bytes": path.stat().st_size,
                            }
                    else:
                        evidence = inspect_video(path, params)
                    for prior in self.db.execute(
                        "SELECT params,result FROM attempts WHERE result IS NOT NULL AND id!=?",
                        (row["id"],),
                    ):
                        if (
                            json.loads(prior["result"])["sha256"] == evidence["sha256"]
                            and prior["params"] != row["params"]
                        ):
                            raise ValueError("replayed_result")
                    if self.config.get("referenceVerification"):
                        receipt = job.get("verificationReceipt")
                        if not isinstance(receipt, dict):
                            raise ValueError("missing_verification_receipt")
                        expected = {
                            "jobId": row["id"],
                            "workerId": row["worker"],
                            "nonce": row["nonce"],
                            "parametersHash": reference_hash(params),
                            "resultSha256": evidence["sha256"],
                        }
                        miner = worker.miner
                        if receipt.get("modelManifestHash") != miner.get(
                            "modelManifestHash"
                        ) or receipt.get("workflowHash") != miner.get("workflowHash"):
                            raise ValueError("receipt_declared_manifest_changed")
                        verify_receipt(receipt, expected, miner["verificationPublicKey"])
                        # Collection finishes before any reference work, so a
                        # replay cannot delay observing another miner's deadline.
                        self._update(
                            row["id"],
                            status="auditing",
                            reason=None,
                            result=canonical(evidence),
                            verification=canonical(receipt),
                            generation=generation,
                            completed=time.time(),
                        )
                    else:
                        self._update(
                            row["id"],
                            status="review",
                            result=canonical(evidence),
                            reason="semantic_review_required",
                            completed=time.time(),
                        )
                elif job["status"] in ("failed", "cancelled"):
                    self._update(
                        row["id"],
                        status=job["status"],
                        reason="worker_terminal_failure",
                        completed=time.time(),
                    )
                else:
                    self._update(row["id"], status="running", reason=None)
            except (urllib.error.URLError, TimeoutError, OSError):
                # Unreachable is uncertainty; preserve the intent and don't resend elsewhere.
                self._update(
                    row["id"],
                    reason="worker_unreachable",
                    **(
                        {"status": "timed_out", "completed": row["completed"] or time.time()}
                        if time.time() >= row["deadline"]
                        else {}
                    ),
                )
            except WorkerBusy:
                # The worker rejected this exact ID before accepting it. Retain
                # the quote and ID for a later same-worker retry; no service
                # result or miner penalty has occurred.
                if _trace_mode(self.config):
                    self._update(
                        row["id"],
                        status="busy_capacity",
                        reason="gpu_busy",
                        completed=time.time(),
                    )
                else:
                    self._update(row["id"], status="queued", reason="gpu_busy")
            except (RegistrationUnavailable, WorkerCredentialUnavailable) as exc:
                # A stale/missing finalized projection must block publication but
                # cannot be charged to a miner. Keep a pre-deadline intent durable
                # so a fresh projection can resume it; after the deadline retain a
                # terminal inconclusive record instead of a service failure.
                self._update(
                    row["id"],
                    reason=(
                        "registration_unavailable"
                        if isinstance(exc, RegistrationUnavailable)
                        else "worker_credential_unavailable"
                    ),
                    **(
                        {"status": "inconclusive", "completed": row["completed"] or time.time()}
                        if time.time() >= row["deadline"]
                        else {}
                    ),
                )
            except (ValueError, KeyError, subprocess.SubprocessError):
                if _trace_mode(self.config):
                    # Transport/adapter parsing is validator uncertainty. Only the
                    # signed runtime verdict may establish miner invalidity.
                    self._update(
                        row["id"],
                        status="inconclusive",
                        reason="trace_service_validation_unavailable",
                        completed=time.time(),
                    )
                else:
                    self._update(
                        row["id"],
                        status="invalid",
                        reason="verification_failed",
                        completed=time.time(),
                    )
        return self.rows()

    def audit_pending(self) -> list[dict[str, Any]]:
        """Run reference replays only after the driver has collected all miners."""
        if _trace_mode(self.config):
            for row in self.db.execute(
                "SELECT * FROM attempts WHERE status IN ('auditing','audit_pending')"
                " ORDER BY started,id"
            ).fetchall():
                if row["status"] == "audit_pending" and time.time() >= row["reaudit_deadline"]:
                    self._update(
                        row["id"],
                        status="inconclusive",
                        reason="trace_reaudit_deadline_expired",
                        completed=time.time(),
                    )
                    continue
                if row["status"] == "audit_pending" and int(row["reaudit_attempts"] or 0) >= 2:
                    self._update(
                        row["id"],
                        status="inconclusive",
                        reason="trace_reaudit_exhausted",
                        completed=time.time(),
                    )
                    continue
                try:
                    worker = self.workers[row["miner"]]
                    custody_started = time.monotonic()
                    evidence_root, receipt = self._trace_evidence(row, worker)
                    # The initial effective transfer is part of worker delivery;
                    # later retries and validator runtime queueing are not.
                    if int(row["reaudit_attempts"] or 0) == 0:
                        self._seal_trace_custody(row, time.monotonic() - custody_started)
                    result = self._trace_runtime_result(row, evidence_root, receipt)
                    verdict = result["verdict"]
                    if verdict == "inconclusive":
                        raise TraceProtocolError(result["reason"], "inconclusive")
                    if verdict == "verified":
                        status, reason = "verified", None
                    elif verdict == "invalid":
                        status, reason = "invalid", result["reason"]
                    else:
                        status, reason = "inconclusive", result["reason"]
                    self._update(
                        row["id"],
                        status=status,
                        reason=reason,
                        result=canonical(result.get("result")) if result.get("result") else None,
                        verification=canonical(receipt),
                        trace_receipt=canonical(receipt),
                        trace_result=canonical(result),
                        completed=time.time(),
                    )
                except (TraceProtocolError, OSError, ValueError, urllib.error.URLError) as exc:
                    reason = (
                        exc.reason
                        if isinstance(exc, TraceProtocolError)
                        else "trace_custody_unavailable"
                    )
                    attempts = int(row["reaudit_attempts"] or 0) + 1
                    if attempts >= 2:
                        self._update(
                            row["id"],
                            status="inconclusive",
                            reason="trace_reaudit_exhausted",
                            reaudit_attempts=attempts,
                            completed=time.time(),
                        )
                    else:
                        self._update(
                            row["id"],
                            status="audit_pending",
                            reason=reason,
                            reaudit_attempts=attempts,
                        )
            return self.rows()
        if not self.config.get("referenceVerification"):
            return self.rows()
        for row in self.db.execute(
            "SELECT * FROM attempts WHERE status='auditing' ORDER BY started,id"
        ).fetchall():
            params, receipt, evidence = (
                json.loads(row["params"]),
                json.loads(row["verification"]),
                json.loads(row["result"]),
            )
            try:
                inspected = inspect_video(self.directory / f"{row['id']}.mp4", params)
                if inspected["sha256"] != evidence["sha256"]:
                    self._update(
                        row["id"], status="inconclusive", reason="validator_custody_changed"
                    )
                    continue
                evidence = inspected
                self._update(row["id"], result=canonical(evidence))
            except (OSError, ValueError, subprocess.SubprocessError):
                self._update(row["id"], status="invalid", reason="media_verification_failed")
                continue
            try:
                ref = replay(
                    self.directory,
                    self.config["reference"],
                    {
                        "jobId": row["id"],
                        "nonce": row["nonce"],
                        "parameters": params,
                        "modelManifestHash": receipt["modelManifestHash"],
                        "workflowHash": receipt["workflowHash"],
                    },
                )
            except (OSError, subprocess.SubprocessError, TimeoutError, ValueError):
                self._update(row["id"], status="inconclusive", reason="reference_unavailable")
                continue
            match = (
                evidence["canonicalVideoSha256"] == ref["canonicalVideoSha256"]
                and evidence["canonicalAudioSha256"] == ref["canonicalAudioSha256"]
            )
            calibration = row["batch"] == self.config.get("calibrationBatch") or row[
                "batch"
            ].endswith("-calibration")
            self._update(
                row["id"],
                status="verified" if match else "inconclusive" if calibration else "invalid",
                reason=None
                if match
                else "calibration_mismatch"
                if calibration
                else "confirmed_substitution",
                reference=canonical(ref),
            )
        return self.rows()

    def freeze_calibration(self, batch: str) -> float:
        """Freeze median externally measured honest service time exactly once."""
        calibration = self.config.get("calibrationBatch", batch + "-calibration")
        values = []
        for miner in self.config["miners"]:
            rows = [
                r for r in self.rows() if r["batch"] == calibration and r["miner"] == miner["id"]
            ]
            if len(rows) < int(self.config.get("calibrationPerMiner", 2)) or any(
                r["status"] != "verified" or not r["reference"] or not r["generation"] for r in rows
            ):
                raise ValueError("calibration_not_complete")
            values.extend(r["generation"] for r in rows)
        frozen = statistics.median(values)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO reference_windows(batch,frozen_seconds) VALUES(?,?)",
                (batch, frozen),
            )
        return self.db.execute(
            "SELECT frozen_seconds FROM reference_windows WHERE batch=?", (batch,)
        ).fetchone()[0]

    def review(self, job_id: str, judgment: dict[str, Any]) -> None:
        row = self.db.execute("SELECT * FROM attempts WHERE id=?", (job_id,)).fetchone()
        if not row or row["status"] not in ("review", "verified", "rejected"):
            raise ValueError("No independently decoded result")
        if (
            judgment.get("sha256") != json.loads(row["result"])["sha256"]
            or type(judgment.get("pass")) is not bool
            or not judgment.get("judge")
            or judgment.get("controlsPassed") is not True
        ):
            raise ValueError(
                "A hash-bound semantic review with passed negative controls is required"
            )
        if row["judgment"] and row["judgment"] != canonical(judgment):
            raise ValueError("Review is immutable; create a new challenge")
        self._update(
            job_id,
            judgment=canonical(judgment),
            status="verified" if judgment["pass"] else "rejected",
            reason=None,
        )

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute("SELECT * FROM attempts ORDER BY started,id")]

    def reference_summary(self, batch: str) -> dict[str, Any]:
        """Small durable projection for the dashboard and publication gate."""
        if not self.config.get("referenceVerification"):
            return {"enabled": False, "status": "legacy", "method": None}
        calibration_batch = self.config.get("calibrationBatch", batch + "-calibration")
        required = int(self.config.get("calibrationPerMiner", 2))
        holdouts = int(self.config.get("holdoutsPerMiner", 6))
        values: dict[str, Any] = {
            "enabled": True,
            "status": "pending",
            "method": "full-reference-replay",
            "profile": self.config.get("reference", {}).get("profile", "blackwell-nvfp4"),
            "calibration": {"required": required, "completed": 0, "passed": 0},
            "holdout": {"requiredPerMiner": holdouts},
            "limitation": (
                "Exact decoded replay is sampled output evidence, not remote GPU ownership proof."
            ),
            "miners": {},
        }
        for miner in self.config["miners"]:
            calibration = [
                r
                for r in self.rows()
                if r["batch"] == calibration_batch and r["miner"] == miner["id"]
            ]
            window = [r for r in self.rows() if r["batch"] == batch and r["miner"] == miner["id"]]
            passed = len(calibration) >= required and all(
                r["status"] == "verified" for r in calibration
            )
            values["miners"][miner["id"]] = {
                "calibration": {
                    "required": required,
                    "completed": sum(
                        r["completed"] is not None and r["status"] != "auditing"
                        for r in calibration
                    ),
                    "passed": passed,
                    "passedCount": sum(r["status"] == "verified" for r in calibration),
                },
                "holdout": {
                    "required": holdouts,
                    "completed": sum(
                        r["completed"] is not None and r["status"] != "auditing" for r in window
                    ),
                },
            }
        if values["miners"]:
            values["calibration"]["completed"] = min(
                v["calibration"]["completed"] for v in values["miners"].values()
            )
            values["calibration"]["passed"] = min(
                v["calibration"]["passedCount"] for v in values["miners"].values()
            )
        if any(r["batch"] == calibration_batch for r in self.rows()):
            values["status"] = "calibrating"
        if all(
            v["calibration"]["passed"] and v["holdout"]["completed"] >= holdouts
            for v in values["miners"].values()
        ):
            values["status"] = "verified"
        if any(
            r["batch"] in (batch, calibration_batch) and r["status"] == "inconclusive"
            for r in self.rows()
        ):
            values["status"] = "inconclusive"
        return values

    def _trace_scores(self, batch: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Project trace verdicts into isolated lanes without making pending global."""
        window = self._trace_window(batch)
        policy = {
            "minCompleted": window["min_completed"],
            "minVerified": window["min_verified"],
            "windowStartUnix": window["started"],
            "windowEndUnix": window["expires"],
        }
        scores: dict[str, Any] = {}
        for miner in self.config["miners"]:
            attempts = [row for row in rows if row["miner"] == miner["id"]]
            snapshots = [
                json.loads(row["trace_profile"]) for row in attempts if row["trace_profile"]
            ]
            if not snapshots:
                scores[miner["id"]] = {
                    "uid": miner["uid"],
                    "hotkey": miner["hotkey"],
                    "attempts": 0,
                    "succeeded": 0,
                    "score": None,
                    "weight": 0.0,
                    "verificationStatus": "missing_trace_request",
                    "excluded": False,
                }
                continue
            snapshot = snapshots[0]
            if any(item != snapshot for item in snapshots):
                scores[miner["id"]] = {
                    "uid": miner["uid"],
                    "hotkey": miner["hotkey"],
                    "attempts": len(attempts),
                    "succeeded": 0,
                    "score": None,
                    "weight": 0.0,
                    "verificationStatus": "trace_profile_changed",
                    "excluded": False,
                }
                continue
            lane = {
                "hotkey": miner["hotkey"],
                "workerId": miner["workerId"],
                "profileId": snapshot["id"],
                "profileHash": snapshot["hash"],
                "rendererHash": snapshot["rendererHash"],
            }
            if snapshot.get("lane") != lane:
                scores[miner["id"]] = {
                    "uid": miner["uid"],
                    "hotkey": miner["hotkey"],
                    "attempts": len(attempts),
                    "succeeded": 0,
                    "score": None,
                    "weight": 0.0,
                    "verificationStatus": "trace_attempt_lane_changed",
                    "excluded": False,
                }
                continue
            ledger = []
            for row in attempts:
                status = row["status"]
                event: dict[str, Any] = {"lane": lane, "completedAtUnix": row["completed"]}
                if status == "verified":
                    report = json.loads(row["trace_result"] or "{}")
                    event.update(
                        status="verified",
                        full20Verified=report.get("checks", {}).get("predictions")
                        == {
                            "checked": 20,
                            "matched": 20,
                            "total": 20,
                        },
                        resultTrusted=report.get("verdict") == "verified",
                        workerSeconds=row["generation"],
                        referenceSeconds=window["reference_seconds"],
                        gatewayTtvSeconds=(row["completed"] - row["started"])
                        if row["completed"] is not None
                        else None,
                    )
                elif status in ("invalid", "failed", "cancelled", "timed_out", "busy_capacity"):
                    event.update(
                        status="timeout" if status == "timed_out" else "invalid",
                        attribution="miner",
                    )
                elif status in ("auditing", "audit_pending"):
                    event.update(status="pending", reauditDeadlineUnix=row["reaudit_deadline"])
                else:
                    event.update(
                        status="inconclusive",
                        admissionSuspended=(
                            row["reason"]
                            in {"trace_reaudit_deadline_expired", "trace_reaudit_exhausted"}
                        ),
                    )
                ledger.append(event)
            detail = trace_trusted_render_score(
                ledger,
                policy=policy,
                qualification={
                    "lane": lane,
                    "profileQualified": True,
                    "expiresAtUnix": snapshot["qualificationExpiresAt"],
                },
                now_unix=time.time(),
            )
            scores[miner["id"]] = {
                "uid": miner["uid"],
                "hotkey": miner["hotkey"],
                "attempts": len(attempts),
                "succeeded": sum(row["status"] == "verified" for row in attempts),
                **detail,
                "verificationStatus": detail["admission"],
                "excluded": False,
            }
        eligible = [item for item in scores.values() if item.get("rewardEligible")]
        total = sum(item["score"] or 0.0 for item in eligible)
        for item in scores.values():
            item["weight"] = (item["score"] / total) if item in eligible and total else 0.0
        complete = bool(eligible and total > 0)
        return {
            "schema": "forge/inference-scores/v3",
            "network": "test",
            "netuid": self.config["netuid"],
            "source": self.config.get("source", "fixture"),
            "mechanism": 1,
            "batch": batch,
            "complete": complete,
            "publishable": complete,
            "miners": scores,
            "evidenceHash": hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "trace": {"mode": TRACE_TRUSTED_RENDER, "window": policy},
            "limitations": (
                "Trusted trace/runtime evidence is profile-bound and does not prove "
                "remote GPU ownership."
            ),
        }

    def scores(self, batch: str) -> dict[str, Any]:
        self._registered()
        rows = [r for r in self.rows() if r["batch"] == batch]
        if _trace_mode(self.config):
            return self._trace_scores(batch, rows)
        reference_summary = self.reference_summary(batch)
        frozen = self.db.execute(
            "SELECT frozen_seconds FROM reference_windows WHERE batch=?", (batch,)
        ).fetchone()
        scores = {}
        sets = []
        for miner in self.config["miners"]:
            attempts = [r for r in rows if r["miner"] == miner["id"]]
            sets.append({r["params"] for r in attempts})
            successes = [r for r in attempts if r["status"] == "verified"]
            pending = any(
                r["completed"] is None or r["status"] in ("review", "auditing", "inconclusive")
                for r in attempts
            )
            if self.config.get("referenceVerification"):
                score_details = availability_score(
                    [
                        {
                            **r,
                            "generationSeconds": r["generation"],
                            "reference": (
                                {"generationSeconds": frozen[0]}
                                if frozen
                                else (json.loads(r["reference"]) if r["reference"] else None)
                            ),
                        }
                        for r in attempts
                    ]
                )
                score = (
                    None
                    if pending or len(attempts) < int(self.config.get("holdoutsPerMiner", 6))
                    else score_details["score"]
                )
            else:
                score_details = {}
                score = (
                    None
                    if pending or len(attempts) < 2
                    else sum(
                        min(1.0, 60 / max(1, r["completed"] - r["started"])) for r in successes
                    )
                    / len(attempts)
                )
            scores[miner["id"]] = {
                "uid": miner["uid"],
                "hotkey": miner["hotkey"],
                "attempts": len(attempts),
                "succeeded": len(successes),
                **score_details,
                "score": score,
                "verificationStatus": (
                    "confirmed_substitution"
                    if any(r["reason"] == "confirmed_substitution" for r in attempts)
                    else "verified"
                    if attempts and all(r["status"] == "verified" for r in attempts)
                    else "incomplete"
                )
                if self.config.get("referenceVerification")
                else "legacy",
                "excluded": any(
                    r["status"] == "inconclusive" or r["reason"] == "confirmed_substitution"
                    for r in attempts
                ),
            }
        calibration_ready = not self.config.get("referenceVerification") or all(
            item["calibration"]["passed"] for item in reference_summary["miners"].values()
        )
        holdouts_ready = not self.config.get("referenceVerification") or all(
            item["holdout"]["completed"] >= reference_summary["holdout"]["requiredPerMiner"]
            for item in reference_summary["miners"].values()
        )
        complete = (
            bool(sets and sets[0])
            and all(s == sets[0] for s in sets)
            and all(m["score"] is not None for m in scores.values())
            and calibration_ready
            and (not self.config.get("referenceVerification") or frozen is not None)
            and holdouts_ready
        )
        total = sum(m["score"] or 0 for m in scores.values())
        for m in scores.values():
            m["weight"] = m["score"] / total if complete and total else None
        return {
            "schema": "forge/inference-scores/v2"
            if self.config.get("referenceVerification")
            else "forge/inference-scores/v1",
            "network": "test",
            "netuid": self.config["netuid"],
            "source": self.config.get("source", "fixture"),
            "mechanism": 1,
            "batch": batch,
            "complete": complete,
            "publishable": complete and total > 0,
            "miners": scores,
            "evidenceHash": hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "reference": reference_summary,
            "limitations": "Reference replay binds this audit output to a controlled H3 result; "
            "it does not cryptographically prove remote GPU ownership or every execution.",
        }
