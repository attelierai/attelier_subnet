#!/usr/bin/env python3
"""Price a frozen development pilot without renting resources or authorizing spend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from forge.evaluation.vision_judge import PILOT_RESERVATION_USD
from forge.protocol.hashing import sha256_file

SECONDS = (
    "startup_seconds",
    "metrics_seconds",
    "encoding_seconds",
    "api_and_transfer_idle_seconds",
    "shutdown_seconds",
)
DOLLARS = ("storage_usd", "transfer_usd", "contingency_usd")


def preflight(ledger: dict, costs: dict, *, prior_compute_usd: float = 0.0) -> dict:
    """Unknown costs remain blockers; allowances are estimates, never measured bills."""
    if (
        ledger.get("wan50_gpu_renders") != 32
        or ledger.get("blinded_quality_pairs") != 96
        or ledger.get("blinded_luna_calls") != 192
    ):
        raise ValueError("this preflight is only for the frozen 32-render/96-pair pilot")
    if (
        type(prior_compute_usd) not in (int, float)
        or not math.isfinite(prior_compute_usd)
        or prior_compute_usd < 0
    ):
        raise ValueError("prior compute exposure must be finite and nonnegative")
    expected = {
        *SECONDS,
        *DOLLARS,
        "cap_usd",
        "warmup_generations",
        "wan50_seconds",
        "gpu_usd_hour",
        "luna_usd_call",
        "basis",
        "unresolved",
    }
    if set(costs) != expected:
        raise ValueError("cost fields must match the complete pilot envelope")
    cap = costs["cap_usd"]
    if type(cap) not in (float, int) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("explicit campaign planning cap must be finite and positive")
    if type(costs["warmup_generations"]) is not int or costs["warmup_generations"] < 1:
        raise ValueError("budget at least one warmup generation")
    if not isinstance(costs["basis"], dict) or not costs["basis"]:
        raise ValueError("document measured versus estimated input sources")
    if not isinstance(costs["unresolved"], list) or any(
        not isinstance(item, str) or not item for item in costs["unresolved"]
    ):
        raise ValueError("unresolved inputs must be an explicit list")
    numeric = (*SECONDS, *DOLLARS, "wan50_seconds", "gpu_usd_hour", "luna_usd_call")
    missing = [key for key in numeric if costs[key] is None]
    for key in numeric:
        value = costs[key]
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (key in {"wan50_seconds", "gpu_usd_hour", "luna_usd_call"} and value == 0)
        ):
            raise ValueError(f"{key} must be finite, nonnegative and priced")
    result = {
        "schema": "forge/evaluator-pilot-preflight/v1",
        "ledger": ledger,
        "cap_usd": cap,
        "prior_compute_usd": prior_compute_usd,
        "warning_usd": 0.8 * cap,
        "dispatch_authorized": False,
        "promotion_eligible": False,
        "completion_guaranteed": False,
        "luna_reservation_usd_per_call": PILOT_RESERVATION_USD,
        "luna_all_calls_at_reservation_usd": 192 * PILOT_RESERVATION_USD,
        "missing_costs": missing,
        "unresolved": costs["unresolved"],
        "basis": costs["basis"],
        "status": "INCOMPLETE",
    }
    if missing:
        return result
    generation_seconds = (32 + costs["warmup_generations"]) * costs["wan50_seconds"]
    gpu_seconds = generation_seconds + sum(costs[key] for key in SECONDS)
    gpu_usd = gpu_seconds * costs["gpu_usd_hour"] / 3600
    luna_usd = 192 * costs["luna_usd_call"]
    total = gpu_usd + luna_usd + sum(costs[key] for key in DOLLARS)
    result.update(
        {
            "generation_seconds_including_warmup": generation_seconds,
            "gpu_seconds_including_operations": gpu_seconds,
            "gpu_usd": gpu_usd,
            "luna_usd": luna_usd,
            "other_usd": sum(costs[key] for key in DOLLARS),
            "all_in_estimate_usd": total,
            "campaign_estimate_usd": total + prior_compute_usd,
            "remaining_usd": cap - total - prior_compute_usd,
            "status": "OVER_CAP"
            if total + prior_compute_usd > cap
            else ("INCOMPLETE" if costs["unresolved"] else "ESTIMATE_WITHIN_CAP"),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--guard-envelope", type=Path)
    parser.add_argument("--prior-campaign-result", type=Path)
    args = parser.parse_args()
    from forge.evaluation.pilot_worklist import pilot_worklist

    pilot = json.loads(args.pilot.read_text())
    costs = json.loads(args.costs.read_text())
    work = pilot_worklist(pilot, warmup_generations=costs["warmup_generations"])
    prior_compute = 0.0
    if args.prior_campaign_result:
        prior = json.loads(args.prior_campaign_result.read_text())
        attempts = prior.get("attempts", [])
        if (
            prior.get("schema") != "forge/evaluator-pilot-campaign-result/v1"
            or prior.get("all_owned_pods_absent") is not True
            or not isinstance(attempts, list)
            or not attempts
            or any(not isinstance(row, dict) for row in attempts)
            or any(
                row.get("closed") is not True
                or row.get("absence_verified") is not True
                or not isinstance(row.get("pod_id"), str)
                or not row["pod_id"]
                for row in attempts
            )
            or len({row["pod_id"] for row in attempts}) != len(attempts)
        ):
            raise ValueError("all unique previous attempts require verified closure")
        exposures = [row.get("compute_estimate_usd") for row in attempts]
        declared = prior.get("compute_and_running_disk_estimate_usd")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in [*exposures, declared]
        ):
            raise ValueError("previous cost exposure must be finite and nonnegative")
        prior_compute = sum(exposures)
        if not math.isclose(prior_compute, declared, rel_tol=0, abs_tol=1e-9):
            raise ValueError("previous cost total differs from its attempts")
    result = preflight(work["ledger"], costs, prior_compute_usd=prior_compute)
    if args.prior_campaign_result:
        result["prior_campaign_result_sha256"] = sha256_file(args.prior_campaign_result)
        result["prior_ancillary_accounting"] = (
            "retained inside the full ancillary envelope; not assumed free"
        )
    if args.guard_envelope:
        from forge.evaluation.pilot_guard import guard_envelope

        config = json.loads(args.guard_envelope.read_text())
        config["contingency_usd"] += prior_compute
        guard = guard_envelope(config)
        result["guard_config"] = config
        result["guard_envelope"] = guard
        result["guard_envelope_sha256"] = sha256_file(args.guard_envelope)
        if "gpu_seconds_including_operations" in result:
            work_seconds = result["gpu_seconds_including_operations"] - costs["shutdown_seconds"]
            result["guard_work_margin_seconds"] = guard["work_seconds"] - work_seconds
            problems = []
            if config["cap_usd"] != costs["cap_usd"]:
                problems.append("guard and campaign planning caps differ")
            if work_seconds > guard["work_seconds"]:
                problems.append("estimated work exceeds guard working window")
            if result["luna_usd"] > guard["api_limit_usd"]:
                problems.append("estimated Luna cost exceeds dedicated API envelope")
            if costs["storage_usd"] + costs["transfer_usd"] > guard["ancillary_limit_usd"]:
                problems.append("estimated ancillary costs exceed dedicated envelope")
            if costs["gpu_usd_hour"] > guard["compute_usd_hour"]:
                problems.append("GPU price exceeds compute and disk ceiling")
            result["guard_problems"] = problems
            if problems and result["status"] != "OVER_CAP":
                result["status"] = "INCOMPLETE"
    result["worklist_id"] = work["worklist_id"]
    result["worklist"] = work
    result.update(
        {"pilot_sha256": sha256_file(args.pilot), "costs_sha256": sha256_file(args.costs)}
    )
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "dispatch_authorized")}))
    return 0 if result["status"] == "ESTIMATE_WITHIN_CAP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
