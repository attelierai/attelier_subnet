"""Provenance checks for validator-run VBench metric evidence.

VBench is an input to the automatic evaluator, not a claim that a local
installation ran it.  A runner must pin its implementation and every model
weight it used before its values can be admitted to a receipt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
METRICS = (
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
)
QUALITY_AXIS_BY_VBENCH_METRIC = {
    "subject_consistency": "subject_consistency",
    "background_consistency": "background_consistency",
    "imaging_quality": "image_quality",
    "motion_smoothness": "temporal_smoothness",
}


class VBenchProvenanceError(ValueError):
    """Raised when metric values cannot be reproduced from a pinned runner."""


def normalize_native_score(value: float) -> float:
    """Bound upstream float32 roundoff, not quality differences.

    Native cosine scores can exceed one by one float32 ULP, including for
    identical frames. The producer keeps the unmodified raw evidence; consumers
    still require strictly bounded normalized receipts.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -1e-6 <= value <= 1 + 1e-6
    ):
        raise VBenchProvenanceError("native VBench score exceeds numerical roundoff bounds")
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class VBenchProvenance:
    implementation_commit: str
    weight_sha256: Mapping[str, str]
    runner_image_digest: str

    def validate(self) -> None:
        if self.implementation_commit != VBENCH_COMMIT:
            raise VBenchProvenanceError("VBench implementation commit is not pinned")
        if (
            not isinstance(self.runner_image_digest, str)
            or not self.runner_image_digest.startswith("sha256:")
            or len(self.runner_image_digest) != 71
            or any(c not in "0123456789abcdef" for c in self.runner_image_digest[7:])
        ):
            raise VBenchProvenanceError(
                "VBench runner image digest must be sha256 plus 64 lowercase hex digits"
            )
        if set(self.weight_sha256) != set(METRICS):
            raise VBenchProvenanceError("VBench provenance must name every metric weight")
        for name, digest in self.weight_sha256.items():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise VBenchProvenanceError(f"VBench weight hash for {name} is invalid")


def validate_metric_evidence(evidence: Mapping[str, object]) -> dict[str, float]:
    """Validate one VBench row and return its bounded metric values.

    The validator owns execution; accepting arbitrary miner output here would
    break that boundary.
    """

    raw_provenance = evidence.get("provenance")
    raw_scores = evidence.get("scores")
    if not isinstance(raw_provenance, Mapping) or not isinstance(raw_scores, Mapping):
        raise VBenchProvenanceError("VBench evidence needs provenance and scores")
    provenance = VBenchProvenance(
        implementation_commit=raw_provenance.get("implementation_commit", ""),
        weight_sha256=raw_provenance.get("weight_sha256", {}),
        runner_image_digest=raw_provenance.get("runner_image_digest", ""),
    )
    provenance.validate()
    if set(raw_scores) != set(METRICS):
        raise VBenchProvenanceError("VBench scores must include the pinned metric set")
    scores: dict[str, float] = {}
    for metric, value in raw_scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise VBenchProvenanceError(f"VBench score for {metric} is outside [0, 1]")
        scores[metric] = float(value)
    return scores


def quality_axis_scores(evidence: Mapping[str, object]) -> dict[str, float]:
    """Map pinned VBench names to Forge axes.

    ``dynamic_degree`` is intentionally omitted: more motion is not uniformly
    better, so its use must remain conditional on the prompt-aware Luna judge.
    """

    scores = validate_metric_evidence(evidence)
    return {axis: scores[metric] for metric, axis in QUALITY_AXIS_BY_VBENCH_METRIC.items()}
