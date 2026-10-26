"""Bounded, fail-closed evaluation primitives for the Wan MVP.

This package deliberately does not invoke a GPU, VBench or a vision API.  It
validates the evidence those trusted validator-owned runners produced.
"""

from forge.evaluation.calibration_campaign import freeze_calibration_campaign
from forge.evaluation.mvp import evaluate_confirmation, evaluate_mvp, evaluate_screening
from forge.evaluation.mvp_calibration import (
    fit_policy_from_development,
    validate_calibration,
    validate_qualified_calibration,
)
from forge.evaluation.mvp_evidence import freeze_evaluation_plan, validate_confirmation_panel
from forge.evaluation.mvp_policy import Decision, EvaluationPolicy, freeze_policy

__all__ = [
    "Decision",
    "EvaluationPolicy",
    "evaluate_mvp",
    "evaluate_confirmation",
    "evaluate_screening",
    "freeze_evaluation_plan",
    "freeze_policy",
    "fit_policy_from_development",
    "freeze_calibration_campaign",
    "validate_calibration",
    "validate_qualified_calibration",
    "validate_confirmation_panel",
]
