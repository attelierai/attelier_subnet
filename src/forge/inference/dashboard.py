"""Public projection; deliberately excludes worker origins, tokens and prompts."""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.inference.service import Validator
from forge.inference.trace_protocol import TraceProtocolError, load_profile


def iso(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, UTC).isoformat() if value is not None else None


def project(
    validator: Validator,
    batch: str,
    chain: dict[str, Any],
    budget: dict[str, Any] | None,
    publication: dict[str, Any] | None = None,
    observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [r for r in validator.rows() if r["batch"] == batch]
    trace_mode = validator.config.get("verificationMode") == "trace_trusted_render"
    issues = []
    try:
        scores = validator.scores(batch)["miners"]
    except ValueError:
        scores = {}
        issues.append("Registration is stale or changed; admission and weights are blocked.")
    miners = []
    observations = observations or {}
    for descriptor in validator.config["miners"]:
        attempts = [r for r in rows if r["miner"] == descriptor["id"]]
        successful = [r for r in attempts if r["status"] == "verified"]
        failed = [
            r
            for r in attempts
            if r["status"] in ("invalid", "rejected", "failed", "cancelled", "timed_out")
        ]
        score = scores.get(descriptor["id"], {})
        observation = observations.get(descriptor["id"], {})
        measured_state = observation.get("state")
        if measured_state not in (
            "offline",
            "starting",
            "ready",
            "busy",
            "draining",
            "stopped",
            "failed",
        ):
            measured_state = "offline"
        latencies = [
            r.get("generation")
            if not trace_mode and r.get("generation") is not None
            else r["completed"] - r["started"]
            for r in successful
            if r["completed"] is not None
        ]
        observed_weights = (publication or {}).get("observedWeights", {})
        miners.append(
            {
                "id": descriptor["id"],
                "uid": descriptor["uid"],
                "hotkey": descriptor["hotkey"],
                "hardware": descriptor.get("hardware", "Unknown"),
                "model": "MiniMax H3 Base FL2VA NVFP4",
                "state": measured_state,
                "workerId": descriptor["workerId"],
                "attempts": len(attempts),
                "succeeded": len(successful),
                "failed": len(failed),
                "successRate": len(successful) / len(attempts) if attempts else None,
                "medianLatencySeconds": statistics.median(latencies) if latencies else None,
                "score": score.get("score"),
                "weight": score.get("weight"),
                "availabilityRate": score.get("availabilityRate") if attempts else None,
                "speedScore": score.get("speedScore") if attempts else None,
                "verificationStatus": score.get("verificationStatus", "unknown"),
                "excluded": score.get("excluded", False),
                "publishedWeight": (
                    observed_weights.get(str(descriptor["uid"]), 0.0) if observed_weights else None
                ),
                "lastSeen": iso(observation.get("observedAt")),
                "evidenceStatus": "Reviewed" if successful else "Awaiting verified results",
            }
        )
    jobs = [
        {
            "id": r["id"],
            "minerId": r["miner"],
            "kind": "challenge",
            "status": r["status"],
            "startedAt": iso(r["started"]),
            "completedAt": iso(r["completed"]),
            "latencySeconds": r["completed"] - r["started"] if r["completed"] else None,
            "generationSeconds": r.get("generation"),
            "receiptStatus": "verified" if r.get("verification") else "missing",
            "referenceStatus": (
                "verified"
                if r["status"] == "verified" and r.get("reference")
                else "failed"
                if r["reason"] == "confirmed_substitution"
                else "inconclusive"
                if r["status"] == "inconclusive"
                else "pending"
            ),
            "canonicalMatch": (
                True
                if r["status"] == "verified" and r.get("reference")
                else False
                if r["reason"] == "confirmed_substitution"
                else None
            ),
            "valid": True
            if r["status"] == "verified"
            else False
            if r["status"] in ("invalid", "rejected", "failed", "cancelled", "timed_out")
            else None,
            "resultSha256": json.loads(r["result"])["sha256"] if r["result"] else None,
            "reason": r["reason"],
        }
        for r in rows
    ]
    publication = publication or {}
    current_publication = publication if publication.get("scores", {}).get("batch") == batch else {}
    emission_split = {"optimization": None, "inference": None}
    if chain.get("mechanismCount") == 2 and "emissionSplit" in chain:
        raw = chain["emissionSplit"]
        if raw is None:
            emission_split = {"optimization": 0.5, "inference": 0.5}
        elif isinstance(raw, list) and len(raw) == 2 and sum(raw) > 0:
            emission_split = {"optimization": raw[0] / sum(raw), "inference": raw[1] / sum(raw)}
    data = {
        "schema": "forge/inference-dashboard/v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "source": validator.config.get("source", "fixture"),
        "network": "test",
        "netuid": validator.config["netuid"],
        "mechanism": 1,
        "status": "active" if any(m["state"] in ("ready", "busy") for m in miners) else "stopped",
        "emissionSplit": emission_split,
        "chain": {
            "block": chain.get("block"),
            "hash": chain.get("blockHash"),
            "runtime": chain.get("runtime"),
            "mechanismCount": chain.get("mechanismCount"),
            "weightsStatus": current_publication.get("status", "not_published"),
            "weightsBlock": current_publication.get("weightsBlock"),
        },
        "budget": budget,
        "miners": miners,
        "jobs": jobs,
        "issues": issues,
        "checks": [
            {
                "id": "hardware",
                "label": "Hardware identity",
                "status": "pending",
                "detail": "Hardware is operator-declared, not cryptographically attested.",
            },
            {
                "id": "quality",
                "label": "Service validation",
                "status": "pending",
                "detail": "Decoded media and semantic review are separate requirements.",
            },
        ],
    }
    if validator.config.get("referenceVerification"):
        data["verification"] = validator.reference_summary(batch)
        data["campaign"] = {
            key: validator.config.get("campaign", {}).get(key)
            for key in ("startedAt", "deadline", "status")
        }
        data["campaign"]["status"] = data["campaign"]["status"] or "pending"
        data["checks"].append(
            {
                "id": "reference",
                "label": "Full H3 reference replay",
                "status": "pass" if data["verification"]["status"] == "verified" else "blocked",
                "detail": data["verification"]["limitation"],
            }
        )
    if trace_mode:
        _trace_projection(data, validator, rows, scores)
    return data


def _trace_projection(data, validator, rows, scores):
    """Expose measured protocol facts, never private evidence or runtime paths."""
    config = validator.config["traceVerification"]
    identity = config["profile"]
    profile, qualified = None, False
    try:
        profile = load_profile(Path(config["profileRoot"]), identity, allow_unqualified=True)
        load_profile(Path(config["profileRoot"]), identity)
        qualified = True
    except (TraceProtocolError, OSError, ValueError):
        pass
    reports = {row["id"]: json.loads(row.get("trace_result") or "{}") for row in rows}
    renderer_active = qualified and any(
        row["status"] == "verified"
        and reports[row["id"]].get("rendererHash") == profile["rendererHash"]
        for row in rows
    )
    data["verification"] = {
        "method": "trace-trusted-render",
        "profile": {**identity, "status": "qualified" if qualified else "unqualified"},
        "renderer": {
            "status": "active" if renderer_active else "qualifying",
            "hash": profile["rendererHash"] if profile else None,
        },
        "rewardEligible": bool(
            renderer_active and any(s.get("rewardEligible") for s in scores.values())
        ),
        "comparison": "bit_exact",
        "coverage": "full20",
        "limitation": (
            "Verification is restricted to the qualified numerical profile. It does not prove "
            "GPU ownership or exclusive execution; ambiguous jobs earn no new credit."
        ),
    }
    observed = data["emissionSplit"]
    data["chain"]["observedSplit"] = (
        {**observed, "block": data["chain"]["block"]}
        if data["chain"]["block"] is not None and observed["inference"] is not None
        else None
    )
    for miner in data["miners"]:
        attempts = [row for row in rows if row["miner"] == miner["id"]]
        score = scores.get(miner["id"], {})
        verdicts = [row["status"] for row in attempts]
        latest = verdicts[-1] if verdicts else "pending"
        miner.update(
            verifiedJobs=verdicts.count("verified"),
            invalidJobs=verdicts.count("invalid"),
            inconclusiveJobs=verdicts.count("inconclusive"),
            verificationStatus=latest
            if latest in {"verified", "invalid", "inconclusive"}
            else "pending",
            admission={
                "state": "unqualified"
                if not qualified
                else "suspended"
                if score.get("admission")
                in {"suspended_reaudit_expired", "window_expired", "qualification_expired"}
                else "eligible",
                "reason": _reason(score.get("reason")),
            },
        )
    by_id = {row["id"]: row for row in rows}
    for job in data["jobs"]:
        row, report = by_id[job["id"]], reports[job["id"]]
        checks = report.get("checks", {})
        predictions = checks.get("predictions", {})
        timings = report.get("timingsMs", {})
        job.update(
            reason=_reason(row["reason"]),
            receiptStatus="verified" if row["status"] == "verified" else "pending",
            audit={
                "conditioning": _audit(checks.get("conditioning")),
                "recurrence": _audit(checks.get("recurrence")),
                "terminalBinding": _audit(checks.get("terminalBinding")),
                "predictions": {
                    "checked": predictions.get("checked", 0),
                    "matched": predictions.get("matched", 0),
                    "total": 20,
                },
            }
            if checks
            else None,
            delivery={"by": "trusted_renderer", "sha256": job["resultSha256"]}
            if row["status"] == "verified" and job["resultSha256"]
            else None,
            ttv={
                "total": job["latencySeconds"],
                "queue": None,
                "miner": row.get("generation"),
                "audit": (timings["preprocess"] + timings["replay"]) / 1000
                if "preprocess" in timings and "replay" in timings
                else None,
                "render": timings["render"] / 1000 if "render" in timings else None,
                "storage": None,
                "measuredBy": "gateway",
            },
        )
    data["checks"].append(
        {
            "id": "trace",
            "label": "Trace verification and trusted rendering",
            "status": "pass" if data["verification"]["rewardEligible"] else "blocked",
            "detail": data["verification"]["limitation"],
        }
    )


def _audit(value):
    return value if value in {"pass", "fail", "inconclusive"} else "pending"


def _reason(value):
    # Reasons are protocol codes; arbitrary worker/error strings stay private.
    return (
        value
        if isinstance(value, str)
        and value
        and all(c.isascii() and (c.isalnum() or c == "_") for c in value)
        else None
    )
