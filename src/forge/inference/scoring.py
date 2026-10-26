"""Availability-first scoring, including the isolated trusted-trace mode.

``score`` is deliberately retained for the already-published reference-mode
windows.  ``trace_trusted_render_score`` is a new, opt-in policy evaluator;
service admission decides how to persist or publish its result.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

TRACE_TRUSTED_RENDER = "trace_trusted_render"
_MINER_FAULT_STATUSES = {"invalid", "timeout", "busy_capacity"}
_PLATFORM_STATUSES = {"inconclusive", "pending"}


def score(attempts: list[dict[str, Any]]) -> dict[str, float]:
    """A * (.8 + .2 * mean(Tref / (Tref + Tminer))).

    Any confirmed substitution zeros the whole miner window.  Availability is the
    fraction with a verified, canonical reference match; infra-inconclusive rows
    are excluded by the caller before publication rather than charged as fraud.
    """
    if not attempts:
        return {"availabilityRate": 0.0, "speedScore": 0.0, "score": 0.0}
    if any(a.get("reason") == "confirmed_substitution" for a in attempts):
        return {"availabilityRate": 0.0, "speedScore": 0.0, "score": 0.0}
    valid = [a for a in attempts if a.get("status") == "verified"]
    availability = len(valid) / len(attempts)
    ratios = []
    for attempt in valid:
        result = attempt.get("reference") or {}
        tref, miner = result.get("generationSeconds"), attempt.get("generationSeconds")
        if (
            isinstance(tref, (int, float))
            and isinstance(miner, (int, float))
            and tref > 0
            and miner >= 0
        ):
            ratios.append(tref / (tref + miner))
    speed = sum(ratios) / len(ratios) if ratios else 0.0
    return {
        "availabilityRate": availability,
        "speedScore": speed,
        "score": availability * (0.8 + 0.2 * speed),
    }


def trace_trusted_render_score(
    attempts: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    qualification: dict[str, Any],
    now_unix: float,
) -> dict[str, Any]:
    """Evaluate one qualified lane under the v1 trusted-trace policy.

    This function neither trusts a miner-declared profile nor changes the
    historical scorer.  Its input is the validator's already-attributed audit
    ledger: custody/runtime uncertainty remains inconclusive, while an
    *confirmed* miner fault affects only this lane's availability denominator.
    """
    _finite(now_unix, "now_unix")
    min_completed = _positive_int(policy.get("minCompleted"), "policy.minCompleted")
    min_verified = _positive_int(policy.get("minVerified"), "policy.minVerified")
    start = _finite(policy.get("windowStartUnix"), "policy.windowStartUnix")
    end = _finite(policy.get("windowEndUnix"), "policy.windowEndUnix")
    if end <= start:
        raise ValueError("policy windowEndUnix must follow windowStartUnix")
    lane = _qualified_lane(qualification)
    qualification_end = _finite(qualification.get("expiresAtUnix"), "qualification.expiresAtUnix")
    result: dict[str, Any] = {
        "verificationMode": TRACE_TRUSTED_RENDER,
        "lane": lane,
        "policy": {
            "minCompleted": min_completed,
            "minVerified": min_verified,
            "windowStartUnix": start,
            "windowEndUnix": end,
        },
        "coverage": {
            "completed": 0,
            "verified": 0,
            "confirmedMinerFault": 0,
            "platformInconclusive": 0,
            "pending": 0,
            "malformedOrWrongLane": 0,
        },
        "availabilityRate": None,
        "speedScore": None,
        "score": None,
        "rewardEligible": False,
        "admission": "not_qualified",
        "reason": None,
    }
    qualified = qualification.get("profileQualified") is True
    qualification_active = qualified and now_unix < qualification_end

    verified_times: list[float] = []
    speed_ratios: list[float] = []
    reaudits_expired = False
    admission_suspended = False
    for attempt in attempts:
        status = _trace_attempt_status(attempt, lane, start, end)
        admission_suspended |= attempt.get("admissionSuspended") is True and status != "malformed"
        coverage = result["coverage"]
        if status == "verified":
            coverage["completed"] += 1
            coverage["verified"] += 1
            worker = _optional_nonnegative(attempt.get("workerSeconds"))
            reference = _optional_positive(attempt.get("referenceSeconds"))
            if worker is not None and reference is not None:
                speed_ratios.append(reference / (reference + worker))
            ttv = _optional_nonnegative(attempt.get("gatewayTtvSeconds"))
            if ttv is not None:
                verified_times.append(ttv)
        elif status == "miner_fault":
            coverage["completed"] += 1
            coverage["confirmedMinerFault"] += 1
        elif status == "pending":
            coverage["pending"] += 1
            deadline = _optional_nonnegative(attempt.get("reauditDeadlineUnix"))
            reaudits_expired |= deadline is not None and now_unix >= deadline
        elif status == "platform":
            coverage["platformInconclusive"] += 1
        else:
            coverage["malformedOrWrongLane"] += 1
            coverage["platformInconclusive"] += 1

    coverage = result["coverage"]
    if coverage["completed"] < min_completed:
        result["reason"] = "minimum_completed_coverage_not_met"
        return result

    denominator = coverage["verified"] + coverage["confirmedMinerFault"]
    # minVerified above makes zero impossible, but retain this invariant if policy evolves.
    availability = coverage["verified"] / denominator if denominator else 0.0
    speed = sum(speed_ratios) / len(speed_ratios) if speed_ratios else 0.0
    result.update(
        availabilityRate=availability,
        speedScore=speed,
        score=availability * (0.8 + 0.2 * speed),
        verifiedGatewayTtvMedianSeconds=median(verified_times) if verified_times else None,
    )
    # An unresolved re-audit suspends only fresh admission.  It cannot erase
    # already measured availability/speed facts for the lane.
    reward_eligible = (
        qualified
        and qualification_active
        and now_unix < end
        and coverage["verified"] >= min_verified
    )
    result["rewardEligible"] = reward_eligible
    if reaudits_expired or admission_suspended:
        result.update(admission="suspended_reaudit_expired", reason="reaudit_deadline_expired")
    elif now_unix >= end:
        result.update(admission="window_expired", reason="window_expired")
    elif not qualified:
        result.update(admission="not_qualified", reason="profile_not_externally_qualified")
    elif not qualification_active:
        result.update(admission="qualification_expired", reason="profile_qualification_expired")
    elif coverage["verified"] < min_verified:
        # This is a measured zero/low availability diagnostic, not a missing
        # evidence error.  In particular four confirmed miner faults score 0.
        result.update(
            admission="minimum_verified_not_met", reason="minimum_verified_coverage_not_met"
        )
    else:
        result.update(admission="eligible", reason=None)
    return result


def _trace_attempt_status(
    attempt: dict[str, Any], lane: dict[str, Any], start: float, end: float
) -> str:
    """Return a scoring class; malformed evidence is never converted to fault."""
    if not isinstance(attempt, dict) or attempt.get("lane") != lane:
        return "malformed"
    completed = _optional_nonnegative(attempt.get("completedAtUnix"))
    status = attempt.get("status")
    if status != "pending" and (completed is None or not start <= completed < end):
        return "malformed"
    if status == "verified":
        if attempt.get("full20Verified") is True and attempt.get("resultTrusted") is True:
            return "verified"
        return "malformed"
    if status in _MINER_FAULT_STATUSES:
        return "miner_fault" if attempt.get("attribution") == "miner" else "malformed"
    if status == "pending":
        return "pending"
    if status in _PLATFORM_STATUSES:
        return "platform"
    return "malformed"


def _qualified_lane(qualification: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(qualification, dict):
        raise ValueError("qualification must be an object")
    lane = qualification.get("lane")
    if not isinstance(lane, dict):
        raise ValueError("qualification.lane must be an object")
    required = ("hotkey", "workerId", "profileId", "profileHash", "rendererHash")
    if any(not isinstance(lane.get(key), str) or not lane[key] for key in required):
        raise ValueError("qualification.lane is incomplete")
    return {key: lane[key] for key in required}


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = _finite(value, "metric")
    except ValueError:
        return None
    return numeric if numeric >= 0 else None


def _optional_positive(value: Any) -> float | None:
    numeric = _optional_nonnegative(value)
    return numeric if numeric is not None and numeric > 0 else None
