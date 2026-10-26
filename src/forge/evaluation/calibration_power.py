"""Preflight arithmetic for a calibration decision, not a success forecast."""

from __future__ import annotations

from scipy.stats import beta

from forge.evaluation.calibration_campaign import (
    BASE_GROUP_BY_KIND,
    HELD_OUT_SPLITS,
    validate_calibration_campaign,
)
from forge.evaluation.calibration_sources import SCHEMA_V4
from forge.evaluation.mvp_policy import CONTROL_KINDS, LEGITIMATE_CONTROLS


def exact_binomial_bounds(successes: int, count: int, alpha: float) -> tuple[float, float]:
    """Return two-sided Clopper-Pearson bounds used by the calibration gate."""

    if not 0 < alpha < 1 or not 0 <= successes <= count or count <= 0:
        raise ValueError("invalid binomial interval inputs")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, count - successes + 1))
    upper = (
        1.0
        if successes == count
        else float(beta.ppf(1 - alpha / 2, successes + 1, count - successes))
    )
    return lower, upper


def minimum_observations(
    *, sensitivity_lower: float, false_alarm_upper: float, alpha: float
) -> dict[str, int]:
    """Find optimistic lower bounds; real campaigns may require more observations.

    The false-alarm result assumes no false alarms. The sensitivity result
    assumes every observable degradation is detected.  They establish why 32
    was chosen, but never establish that a 32-row campaign will pass.
    """

    if not (0 < sensitivity_lower <= 1 and 0 < false_alarm_upper < 1 and 0 < alpha < 1):
        raise ValueError("invalid calibration targets")
    sensitivity_n = next(
        n for n in range(1, 10_001) if exact_binomial_bounds(n, n, alpha)[0] >= sensitivity_lower
    )
    false_alarm_n = next(
        n for n in range(1, 10_001) if exact_binomial_bounds(0, n, alpha)[1] <= false_alarm_upper
    )
    return {
        "all_hits_minimum": sensitivity_n,
        "zero_false_alarms_minimum": false_alarm_n,
        "required_per_kind": max(sensitivity_n, false_alarm_n),
    }


def campaign_preflight(campaign: dict[str, object], *, alpha: float = 0.05) -> dict[str, object]:
    """Return fixed work counts and statistical limits before any rendering."""

    summary = validate_calibration_campaign(campaign)
    targets = minimum_observations(sensitivity_lower=0.8, false_alarm_upper=0.15, alpha=alpha)
    held_out_rows = 2 * len(CONTROL_KINDS) * 32
    return {
        **summary,
        "execution_ledger": campaign_execution_ledger(campaign),
        "held_out_rows": held_out_rows,
        "blinded_luna_calls": held_out_rows * 2,
        "development_rows": len(campaign["slots"]) - held_out_rows,
        "minimum_observations": targets,
        "pass_tolerance_at_32": {
            "legitimate_controls": "zero false alarms per kind",
            "degradation_controls": "at least 31 detected of 32 per kind",
        },
        "not_a_cost_estimate": True,
    }


def campaign_execution_ledger(campaign: dict[str, object]) -> dict[str, int]:
    """Count media under the declared local-control materialization strategy.

    ``self`` reuses one render, a same-model new run and an independent seed
    each need a second render, static controls derive image defects locally, and
    motion controls derive freeze/shuffle locally while ``wrong_video`` reuses
    another frozen motion base.  This is exact for that strategy, rather than
    calling the 260 distinct families an (incorrect) render count.
    """

    validate_calibration_campaign(campaign)
    if campaign.get("schema") == SCHEMA_V4:
        slots = campaign["slots"]
        # Each partition has 64 references.  Only the 32 new-run and 32
        # independent-seed controls need another Wan invocation; every other
        # control either reuses media or derives a local variant.
        partitions = {slot["split"] for slot in slots}
        references = 64 * len(partitions)
        second_runs = sum(
            1 for slot in slots if slot["kind"] in {"same_model_new_run", "independent_seed"}
        )
        local_derivations = sum(
            1
            for slot in slots
            if slot["kind"]
            in {
                "freeze",
                "blur_mild",
                "blur_severe",
                "shuffle",
                "detail_loss",
                "evaluator_attack_mild",
                "evaluator_attack_severe",
            }
        )
        return {
            "wan50_gpu_renders": references + second_runs,
            "local_derived_videos": local_derivations,
            "unique_media_before_deduplicating_wrong_video": references
            + second_runs
            + local_derivations,
            "blinded_quality_pairs": len(slots),
            "blinded_luna_calls": len(slots) * 2,
        }
    slots = campaign["slots"]
    group_indices: dict[tuple[str, str], set[int]] = {}
    for slot in slots:
        group = BASE_GROUP_BY_KIND[slot["kind"]]
        group_indices.setdefault((slot["split"], group), set()).add(slot["index"])
    counts = {key: len(value) for key, value in group_indices.items()}
    repeat = sum(counts[(split, "repeat")] for split in ("development", *HELD_OUT_SPLITS))
    independent = sum(
        counts[(split, "independent_seed")] for split in ("development", *HELD_OUT_SPLITS)
    )
    static = sum(counts[(split, "static")] for split in ("development", *HELD_OUT_SPLITS))
    motion = sum(counts[(split, "motion")] for split in ("development", *HELD_OUT_SPLITS))
    # See the docstring: repeat and independent each have two generated sides.
    wan50_renders = repeat * 2 + independent * 2 + static + motion
    local_derivations = sum(
        1
        for slot in slots
        if slot["kind"]
        in {
            "freeze",
            "blur_mild",
            "blur_severe",
            "shuffle",
            "detail_loss",
            "evaluator_attack_mild",
            "evaluator_attack_severe",
        }
    )
    return {
        "wan50_gpu_renders": wan50_renders,
        "local_derived_videos": local_derivations,
        "unique_media_before_deduplicating_wrong_video": wan50_renders + local_derivations,
        "blinded_quality_pairs": len(slots),
        "blinded_luna_calls": len(slots) * 2,
    }


def projected_campaign_cost(
    campaign: dict[str, object],
    *,
    wan50_seconds: float,
    gpu_usd_hour: float,
    judge_usd_per_pair: float,
) -> dict[str, float | int]:
    """Price only measured generation and settled-style judge calls.

    Metrics, model loading, storage and transfer are intentionally omitted and
    named by the caller as unknown operational costs.
    """

    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        for value in (wan50_seconds, gpu_usd_hour, judge_usd_per_pair)
    ):
        raise ValueError("campaign projection inputs must be positive")
    ledger = campaign_execution_ledger(campaign)
    generation_usd = ledger["wan50_gpu_renders"] * wan50_seconds / 3600 * gpu_usd_hour
    judge_usd = ledger["blinded_quality_pairs"] * judge_usd_per_pair
    return {
        **ledger,
        "generation_usd": generation_usd,
        "judge_usd": judge_usd,
        "known_usd": generation_usd + judge_usd,
    }


def projected_development_cost(
    campaign: dict[str, object],
    *,
    wan50_seconds: float,
    gpu_usd_hour: float,
    judge_usd_per_pair: float,
) -> dict[str, float | int]:
    """Price the development-only early stop before any holdout is revealed."""

    validate_calibration_campaign(campaign)
    legitimate_samples = campaign["development_legitimate_samples"]
    degradation_samples = campaign["development_degradation_samples"]
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        for value in (wan50_seconds, gpu_usd_hour, judge_usd_per_pair)
    ):
        raise ValueError("development projection inputs must be positive")
    if campaign.get("schema") == SCHEMA_V4:
        # development freezes 64 references, plus 32 new-runs and 32 new
        # seeds.  Seven transformed controls have eight development rows.
        pairs = legitimate_samples * len(LEGITIMATE_CONTROLS) + degradation_samples * (
            len(CONTROL_KINDS) - len(LEGITIMATE_CONTROLS)
        )
        ledger = {
            "wan50_gpu_renders": 64 + legitimate_samples * 2,
            "local_derived_videos": degradation_samples * 7,
            "blinded_quality_pairs": pairs,
            "blinded_luna_calls": pairs * 2,
        }
    else:
        ledger = {
            "wan50_gpu_renders": legitimate_samples * 5 + degradation_samples,
            "local_derived_videos": degradation_samples * 7,
            "blinded_quality_pairs": legitimate_samples * len(LEGITIMATE_CONTROLS)
            + degradation_samples * (len(CONTROL_KINDS) - len(LEGITIMATE_CONTROLS)),
            "blinded_luna_calls": (
                legitimate_samples * len(LEGITIMATE_CONTROLS)
                + degradation_samples * (len(CONTROL_KINDS) - len(LEGITIMATE_CONTROLS))
            )
            * 2,
        }
    generation_usd = ledger["wan50_gpu_renders"] * wan50_seconds / 3600 * gpu_usd_hour
    judge_usd = ledger["blinded_quality_pairs"] * judge_usd_per_pair
    return {
        **ledger,
        "generation_usd": generation_usd,
        "judge_usd": judge_usd,
        "known_usd": generation_usd + judge_usd,
    }
