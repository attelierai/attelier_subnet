"""Fit development thresholds and verify independent calibration evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from scipy.stats import beta

from forge.evaluation.calibration_campaign import bind_calibration_rows
from forge.evaluation.mvp_evidence import _as_records, symmetric_axes
from forge.evaluation.mvp_policy import (
    CONTROL_AXES,
    CONTROL_KINDS,
    LEGITIMATE_CONTROLS,
    QUALITY_AXES,
    Decision,
    EvaluationPolicy,
    _sha,
    freeze_policy,
)


def fit_policy_from_development(
    development_rows: Sequence[Mapping[str, Any]],
    *,
    evaluator_provenance: Mapping[str, str],
    min_speedup_ci: float = 1.1,
    calibration_sensitivity_lower: float = 0.8,
    calibration_false_alarm_upper: float = 0.15,
    calibration_alpha: float = 0.05,
) -> tuple[EvaluationPolicy, dict[str, object]]:
    """Fit a conservative margin before the independent calibration is revealed."""

    rows = _as_records(development_rows, "development calibration rows")
    legitimate = {axis: [] for axis in QUALITY_AXES}
    degradations: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "development" or row.get("kind") not in CONTROL_KINDS:
            raise ValueError("development fitting rows have an invalid split or control")
        if row.get("observable") is not True:
            raise ValueError("development fitting requires observable controls")
        scores = symmetric_axes(row.get("orders"))
        if row["kind"] in LEGITIMATE_CONTROLS:
            for axis, value in scores.items():
                legitimate[axis].append(abs(value))
        else:
            degradations[row["kind"]].append(scores)
    if (
        {row["kind"] for row in rows} != set(CONTROL_KINDS)
        or any(not values for values in legitimate.values())
        or set(degradations) != set(CONTROL_AXES)
    ):
        raise ValueError("development fitting lacks legitimate or degraded coverage")
    margin = max(max(values) for values in legitimate.values())
    for kind, rows_for_kind in degradations.items():
        if not any(
            any(row[axis] < -margin for axis in CONTROL_AXES[kind]) for row in rows_for_kind
        ):
            raise ValueError(f"development fitting cannot separate {kind}")
    derivation = {
        "schema": "forge/mvp-threshold-derivation/v1",
        "method": "maximum-legitimate-absolute-delta",
        "development_rows_hash": _sha(rows),
        "quality_margin": margin,
        "evaluator_provenance": dict(evaluator_provenance),
    }
    derivation["derivation_hash"] = _sha(derivation)
    policy = freeze_policy(
        quality_margin=margin,
        min_speedup_ci=min_speedup_ci,
        calibration_sensitivity_lower=calibration_sensitivity_lower,
        calibration_false_alarm_upper=calibration_false_alarm_upper,
        calibration_alpha=calibration_alpha,
        state="frozen",
        threshold_derivation_hash=derivation["derivation_hash"],
        evaluator_provenance=evaluator_provenance,
    )
    return policy, derivation


def _wilson_lower(successes: int, count: int, alpha: float) -> float:
    if count == 0:
        return 0.0
    # Exact beta interval avoids optimistic normal approximations at small n.
    return 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, count - successes + 1))


def _wilson_upper(successes: int, count: int, alpha: float) -> float:
    if count == 0:
        return 1.0
    return (
        1.0
        if successes == count
        else float(beta.ppf(1 - alpha / 2, successes + 1, count - successes))
    )


def _control_split_summary(
    rows: Sequence[Mapping[str, Any]], policy: EvaluationPolicy
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Measure a frozen control split without changing a fitted threshold.

    The calibration and independent splits serve different purposes.  The former
    checks whether the development-derived margin transfers to fresh families;
    the latter is a final holdout.  Keeping this calculation shared prevents a
    weaker calibration gate than the final check.
    """

    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    if set(by_kind) != set(CONTROL_KINDS) or any(len(items) < 32 for items in by_kind.values()):
        return {}, ["CONTROL_COVERAGE"]
    # Confidence intervals need independent sampling units.  A renamed run or
    # seed does not turn a source/concept family into a new observation.
    if any(
        len({row[field] for row in items}) != len(items)
        for items in by_kind.values()
        for field in ("source_id", "concept_id", "source_family", "concept_family")
    ):
        return {}, ["DEPENDENT_CONTROLS"]

    summary: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for kind in CONTROL_KINDS:
        rows_for_kind = by_kind[kind]
        if not all(row["observable"] for row in rows_for_kind):
            failures.append(f"{kind.upper()}_CONTROL_UNINFORMATIVE")
            continue
        scores = [symmetric_axes(row["orders"]) for row in rows_for_kind]
        count = len(rows_for_kind)
        if kind in LEGITIMATE_CONTROLS:
            positives = sum(
                any(abs(score[axis]) > policy.quality_margin for axis in QUALITY_AXES)
                for score in scores
            )
            upper = _wilson_upper(positives, count, policy.calibration_alpha)
            summary[kind] = {"false_alarm_upper": upper, "count": count}
            if upper > policy.calibration_false_alarm_upper:
                failures.append(f"{kind.upper()}_FALSE_ALARM")
        else:
            protected = CONTROL_AXES[kind]
            positives = sum(
                any(score[axis] < -policy.quality_margin for axis in protected) for score in scores
            )
            lower = _wilson_lower(positives, count, policy.calibration_alpha)
            summary[kind] = {"sensitivity_lower": lower, "count": count}
            if lower < policy.calibration_sensitivity_lower:
                failures.append(f"{kind.upper()}_SENSITIVITY")
    return summary, failures


def validate_calibration(
    calibration: Mapping[str, Any] | None, policy: EvaluationPolicy
) -> dict[str, object]:
    """Check calibration and independent receipts without fitting new thresholds.

    A row has ``split`` (development/calibration/independent), source and
    concept identifiers, a control kind, and both blinded score orders. Both
    post-development splits must meet the frozen rule. Sources and concepts may
    not overlap across splits; that prevents row-level leakage.
    """

    try:
        policy.validate()
    except (TypeError, ValueError):
        return {"status": Decision.INCOMPLETE, "reasons": ["POLICY_INVALID"]}
    if not isinstance(calibration, Mapping):
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_MISSING"]}
    if calibration.get("policy_id") != policy.policy_id:
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_POLICY_UNBOUND"]}
    if policy.state != "frozen":
        return {"status": Decision.INCOMPLETE, "reasons": ["DEVELOPMENT_POLICY_CANNOT_REWARD"]}
    try:
        rows = _as_records(calibration.get("rows"), "calibration rows")
    except ValueError:
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_EVIDENCE_MISSING"]}
    if calibration.get("evaluator_provenance") != dict(policy.evaluator_provenance or {}):
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_PROVENANCE_UNBOUND"]}
    expected_check_hash = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": rows,
        }
    )
    if calibration.get("independent_check_hash") != expected_check_hash:
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_CHECK_HASH_INVALID"]}
    allowed_splits = {"development", "calibration", "independent"}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    concepts: dict[str, set[str]] = defaultdict(set)
    source_families: dict[str, set[str]] = defaultdict(set)
    concept_families: dict[str, set[str]] = defaultdict(set)
    source_kind: set[tuple[str, str, str]] = set()
    try:
        for row in rows:
            split, kind = row["split"], row["kind"]
            source, concept, source_family, concept_family, observable, orders = (
                row["source_id"],
                row["concept_id"],
                row["source_family"],
                row["concept_family"],
                row["observable"],
                row["orders"],
            )
            if split not in allowed_splits or kind not in CONTROL_KINDS:
                raise ValueError
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(concept, str)
                or not concept
                or not isinstance(source_family, str)
                or not source_family
                or not isinstance(concept_family, str)
                or not concept_family
            ):
                raise ValueError
            if type(observable) is not bool:
                raise ValueError
            symmetric_axes(orders)
            identity = (split, kind, source)
            if identity in source_kind:
                raise ValueError
            source_kind.add(identity)
            grouped[split].append(row)
            sources[split].add(source)
            concepts[split].add(concept)
            source_families[split].add(source_family)
            concept_families[split].add(concept_family)
    except (KeyError, ValueError, TypeError):
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_ROW_INVALID"]}
    if not grouped["development"] or not grouped["calibration"] or not grouped["independent"]:
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_SPLIT_MISSING"]}
    if any(not row["observable"] for rows in grouped.values() for row in rows):
        return {"status": Decision.INCONCLUSIVE, "reasons": ["CALIBRATION_CONTROL_UNINFORMATIVE"]}
    if any(
        sources[left] & sources[right]
        or concepts[left] & concepts[right]
        or source_families[left] & source_families[right]
        or concept_families[left] & concept_families[right]
        for left, right in (
            ("development", "calibration"),
            ("development", "independent"),
            ("calibration", "independent"),
        )
    ):
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_SOURCE_CONCEPT_LEAKAGE"]}
    derivation = calibration.get("threshold_derivation")
    if not isinstance(derivation, Mapping):
        return {"status": Decision.INCOMPLETE, "reasons": ["THRESHOLD_DERIVATION_MISSING"]}
    derivation_body = {key: value for key, value in derivation.items() if key != "derivation_hash"}
    development_rows = grouped["development"]
    if (
        derivation.get("derivation_hash") != _sha(derivation_body)
        or derivation.get("derivation_hash") != policy.threshold_derivation_hash
        or derivation.get("development_rows_hash") != _sha(development_rows)
        or derivation.get("quality_margin") != policy.quality_margin
        or derivation.get("evaluator_provenance") != dict(policy.evaluator_provenance or {})
    ):
        return {"status": Decision.INCOMPLETE, "reasons": ["THRESHOLD_DERIVATION_UNBOUND"]}
    # Hash consistency alone does not prove that the declared fitting procedure
    # produced this threshold. Replay development only, never tune on holdouts.
    try:
        replayed_policy, replayed_derivation = fit_policy_from_development(
            development_rows,
            evaluator_provenance=policy.evaluator_provenance or {},
            min_speedup_ci=policy.min_speedup_ci,
            calibration_sensitivity_lower=policy.calibration_sensitivity_lower,
            calibration_false_alarm_upper=policy.calibration_false_alarm_upper,
            calibration_alpha=policy.calibration_alpha,
        )
    except (TypeError, ValueError):
        return {"status": Decision.INCOMPLETE, "reasons": ["THRESHOLD_DERIVATION_NOT_REPRODUCIBLE"]}
    if replayed_policy != policy or replayed_derivation != derivation:
        return {"status": Decision.INCOMPLETE, "reasons": ["THRESHOLD_DERIVATION_NOT_REPRODUCIBLE"]}
    calibration_summary, calibration_failures = _control_split_summary(
        grouped["calibration"], policy
    )
    if calibration_failures:
        malformed = {"CONTROL_COVERAGE", "DEPENDENT_CONTROLS"}
        if any(reason in malformed for reason in calibration_failures):
            return {
                "status": Decision.INCOMPLETE,
                "reasons": [
                    "CALIBRATION_DEPENDENT_CONTROLS"
                    if reason == "DEPENDENT_CONTROLS"
                    else "CALIBRATION_INDEPENDENT_COVERAGE"
                    for reason in calibration_failures
                ],
                "summary": {"calibration": calibration_summary},
            }
        return {
            "status": Decision.INCONCLUSIVE,
            "reasons": [f"CALIBRATION_SPLIT_{reason}" for reason in calibration_failures],
            "summary": {"calibration": calibration_summary},
        }
    independent_summary, independent_failures = _control_split_summary(
        grouped["independent"], policy
    )
    if independent_failures:
        malformed = {"CONTROL_COVERAGE", "DEPENDENT_CONTROLS"}
        if any(reason in malformed for reason in independent_failures):
            return {
                "status": Decision.INCOMPLETE,
                "reasons": [
                    "CALIBRATION_DEPENDENT_CONTROLS"
                    if reason == "DEPENDENT_CONTROLS"
                    else "CALIBRATION_INDEPENDENT_COVERAGE"
                    for reason in independent_failures
                ],
                "summary": {"calibration": calibration_summary, "independent": independent_summary},
            }
        return {
            "status": Decision.INCONCLUSIVE,
            "reasons": [f"INDEPENDENT_SPLIT_{reason}" for reason in independent_failures],
            "summary": {"calibration": calibration_summary, "independent": independent_summary},
        }
    return {
        "status": "PASS",
        "reasons": [],
        "summary": {"calibration": calibration_summary, "independent": independent_summary},
    }


def validate_qualified_calibration(
    calibration: Mapping[str, Any] | None,
    policy: EvaluationPolicy,
    *,
    required_calibration_hash: str | None = None,
) -> dict[str, object]:
    """Validate a promotion-grade calibration receipt and its frozen campaign.

    ``validate_calibration`` remains useful for fitting and local diagnostics.
    A distributed promotion also needs proof that all scored controls were the
    slots frozen before their media existed.  This extra binding is deliberately
    separate so old diagnostic receipts cannot accidentally become eligible.
    """

    result = validate_calibration(calibration, policy)
    if result["status"] != "PASS":
        return result
    if not isinstance(calibration, Mapping):  # defensive; validated above
        return {"status": Decision.INCOMPLETE, "reasons": ["CALIBRATION_MISSING"]}
    try:
        campaign = calibration["campaign"]
        if campaign.get("schema") != "forge/mvp-calibration-campaign/v4":
            return {
                "status": Decision.INCOMPLETE,
                "reasons": ["CALIBRATION_CAMPAIGN_V4_REQUIRED"],
            }
        binding = bind_calibration_rows(campaign, calibration["rows"])
        binding_body = {
            "schema": "forge/mvp-qualified-calibration-binding/v1",
            "policy_id": policy.policy_id,
            "campaign_id": binding["campaign_id"],
            "independent_check_hash": calibration["independent_check_hash"],
            "evidence_hash": binding["evidence_hash"],
        }
        calibration_hash = _sha(binding_body)
    except (KeyError, TypeError, ValueError):
        return {
            "status": Decision.INCOMPLETE,
            "reasons": ["CALIBRATION_CAMPAIGN_UNBOUND"],
        }
    if required_calibration_hash is not None and calibration_hash != required_calibration_hash:
        return {
            "status": Decision.INCOMPLETE,
            "reasons": ["CALIBRATION_HASH_DOES_NOT_MATCH_PLAN"],
        }
    return {
        **result,
        "binding": {**binding_body, "calibration_hash": calibration_hash},
    }
