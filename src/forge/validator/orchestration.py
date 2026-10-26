"""Small orchestration layer that preserves frozen admission inputs on resume."""

from __future__ import annotations

from dataclasses import dataclass

from forge.validator.admission import Admission, AdmissionError, AdmissionLedger
from forge.validator.budget import BudgetError, BudgetLedger, Reservation


@dataclass(frozen=True)
class EvaluationWork:
    admission: Admission
    stage: str
    reservation: Reservation


class EvaluationCoordinator:
    def __init__(self, admissions: AdmissionLedger, budget: BudgetLedger) -> None:
        self.admissions, self.budget = admissions, budget

    def begin_next_screening(
        self, *, operation_key: str, reserve_usd: float
    ) -> EvaluationWork | None:
        admission = self.admissions.claim_next_screening()
        if admission is None:
            return None
        try:
            reservation = self.budget.reserve(admission.id, operation_key, reserve_usd)
        except BudgetError:
            self.admissions.return_screening_to_queue(admission.id)
            raise
        return EvaluationWork(admission, "SCREENING", reservation)

    def begin_full_evaluation(
        self, admission_id: int, *, round_id: str, operation_key: str, reserve_usd: float
    ) -> EvaluationWork:
        admission = self.admissions.get(admission_id)
        if admission.status != "SCREENING":
            raise AdmissionError("only a screened candidate may enter full evaluation")
        reservation = self.budget.reserve(admission.id, operation_key, reserve_usd)
        try:
            admission = self.admissions.start_full_evaluation(admission.id, round_id=round_id)
        except AdmissionError:
            self.budget.release(reservation.id)
            raise
        return EvaluationWork(admission, "FULL_EVALUATION", reservation)

    def infrastructure_failed(self, admission_id: int) -> Admission:
        return self.admissions.transition(admission_id, "INFRASTRUCTURE_FAILED")

    def resume(
        self,
        admission_id: int,
        *,
        artifact_hash: str,
        recipe_hash: str,
        panel_hash: str,
        operation_key: str,
        reserve_usd: float,
        round_id: str | None = None,
    ) -> EvaluationWork:
        admission = self.admissions.get(admission_id)
        if admission.status != "INFRASTRUCTURE_FAILED":
            raise AdmissionError("only infrastructure failures may resume")
        if (artifact_hash, recipe_hash, panel_hash) != (
            admission.artifact_hash,
            admission.recipe_hash,
            admission.panel_hash,
        ):
            raise AdmissionError("resume must use the frozen artifact, recipe, and panel")
        stage = self.admissions.resumable_stage(admission_id)
        if stage == "FULL_EVALUATION" and round_id is None:
            raise AdmissionError("full evaluation resume requires its frozen round id")
        # Infrastructure failure does not reveal a new chance: it reopens the
        # same stage with the original frozen admission fields.
        reservation = self.budget.reserve(admission_id, operation_key, reserve_usd)
        try:
            if stage == "FULL_EVALUATION":
                admission = self.admissions.start_full_evaluation(admission_id, round_id=round_id)
            else:
                admission = self.admissions.transition(admission_id, stage)
        except AdmissionError:
            self.budget.release(reservation.id)
            raise
        return EvaluationWork(admission, stage, reservation)
