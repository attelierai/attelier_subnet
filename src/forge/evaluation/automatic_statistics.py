"""Historical diagnostic statistics retained for receipt replay."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from forge.evaluation.automatic_evidence import _BET_SCALES, _EvidenceError


def _mixture_logwealth(values: list[float], null_mean: float) -> float:
    """Fixed nonnegative e-value for ``E[X] <= null_mean``, with X in [-1, 1]."""

    if not -1 < null_mean < 1:
        raise _EvidenceError("mixture null mean is outside (-1, 1)")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or np.any(array < -1) or np.any(array > 1):
        raise _EvidenceError("mixture e-value received an invalid bounded quality delta")
    logs = []
    for scale in _BET_SCALES:
        factors = 1 - scale + scale * (1 + array) / (1 + null_mean)
        if np.any(factors <= 0) or not np.isfinite(factors).all():
            raise _EvidenceError("mixture e-value received an invalid bounded quality delta")
        logs.append(float(np.log(factors).sum()))
    return float(logsumexp(logs) - math.log(len(logs)))


def _mixture_lower(values: list[float], alpha: float) -> float:
    threshold = math.log(1 / alpha)
    left, right = math.nextafter(-1.0, 1.0), math.nextafter(1.0, -1.0)
    if _mixture_logwealth(values, left) < threshold:
        return -1.0
    if _mixture_logwealth(values, right) >= threshold:
        return 1.0
    for _ in range(80):
        middle = (left + right) / 2
        if _mixture_logwealth(values, middle) >= threshold:
            left = middle
        else:
            right = middle
    return left


def _quality_bound(values: list[float], alpha: float) -> dict[str, float | int | str]:
    if len(values) < 2:
        raise _EvidenceError("at least two independent prompts are required")
    lower = _mixture_lower(values, alpha)
    upper = -_mixture_lower([-value for value in values], alpha)
    return {
        "n_prompts": len(values),
        "mean": sum(values) / len(values),
        "lower": lower,
        "upper": upper,
        "method": "fixed-mixture-evalue",
    }


def _timing_bound(values: list[float], alpha: float) -> dict[str, float | int]:
    count = len(values)
    if count < 2:
        raise _EvidenceError("at least two timing prompts are required")
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    standard_error = math.sqrt(variance / count)
    critical = float(student_t.ppf(1 - alpha, df=count - 1))
    return {
        "n_prompts": count,
        "mean_log_ratio": mean,
        "lower_log_ratio": mean - critical * standard_error,
        "upper_log_ratio": mean + critical * standard_error,
    }


def _zero_variance_floor(alpha: float, margin: float) -> int:
    """Minimum tied prompts whose fixed mixture rejects a mean of ``-margin``."""

    threshold = math.log(1 / alpha)
    low, high = 1, 2
    while _mixture_logwealth([0.0] * high, -margin) < threshold:
        low, high = high, high * 2
    while low < high:
        middle = (low + high) // 2
        if _mixture_logwealth([0.0] * middle, -margin) >= threshold:
            high = middle
        else:
            low = middle + 1
    return low
