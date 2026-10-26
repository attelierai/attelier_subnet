"""In-memory king-of-the-hill admission and round state machine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum


class SubmissionState(StrEnum):
    SUBMITTED = "SUBMITTED"
    ADMITTED = "ADMITTED"
    SCREENED = "SCREENED"
    DUELED = "DUELED"
    FINALIST_LOCKED = "FINALIST_LOCKED"
    CONFIRMED = "CONFIRMED"
    CROWNED = "CROWNED"
    REJECTED = "REJECTED"
    INFRA_ERROR = "INFRA_ERROR"


@dataclass(frozen=True, order=True)
class CommitmentPosition:
    block: int
    extrinsic_index: int

    def __post_init__(self) -> None:
        if self.block < 0 or self.extrinsic_index < 0:
            raise ValueError("commitment position cannot be negative")


@dataclass(frozen=True)
class Submission:
    hotkey: str
    artifact_id: str
    effective_fingerprint: str
    commitment: CommitmentPosition
    state: SubmissionState = SubmissionState.SUBMITTED


_TRANSITIONS = {
    SubmissionState.SUBMITTED: {SubmissionState.ADMITTED, SubmissionState.REJECTED},
    SubmissionState.ADMITTED: {
        SubmissionState.SCREENED,
        SubmissionState.REJECTED,
        SubmissionState.INFRA_ERROR,
    },
    SubmissionState.SCREENED: {
        SubmissionState.DUELED,
        SubmissionState.REJECTED,
        SubmissionState.INFRA_ERROR,
    },
    SubmissionState.DUELED: {
        SubmissionState.FINALIST_LOCKED,
        SubmissionState.REJECTED,
        SubmissionState.INFRA_ERROR,
    },
    SubmissionState.FINALIST_LOCKED: {
        SubmissionState.CONFIRMED,
        SubmissionState.REJECTED,
        SubmissionState.INFRA_ERROR,
    },
    SubmissionState.CONFIRMED: {SubmissionState.CROWNED, SubmissionState.REJECTED},
    SubmissionState.INFRA_ERROR: {
        SubmissionState.ADMITTED,
        SubmissionState.SCREENED,
        SubmissionState.DUELED,
        SubmissionState.FINALIST_LOCKED,
    },
    SubmissionState.CROWNED: set(),
    SubmissionState.REJECTED: set(),
}


@dataclass
class KothRound:
    challenge_id: str
    round_id: str
    king_fingerprint: str
    quota: int
    submissions: dict[str, Submission] = field(default_factory=dict)
    finalist_artifact_id: str | None = None
    confirmation_block_hash: str | None = None
    certified: bool = False

    def submit(self, submission: Submission) -> Submission:
        existing = self.submissions.get(submission.artifact_id)
        if existing is not None:
            if replace(existing, state=submission.state) == submission:
                return existing
            raise ValueError("artifact id already used by a different submission")
        if len(self.submissions) >= self.quota:
            raise ValueError("round quota is full")
        if submission.hotkey in {item.hotkey for item in self.submissions.values()}:
            raise ValueError("one submission per hotkey per round")
        if submission.effective_fingerprint == self.king_fingerprint:
            raise ValueError("the effective tensors duplicate the current king")
        if submission.effective_fingerprint in {
            item.effective_fingerprint for item in self.submissions.values()
        }:
            raise ValueError("duplicate effective tensor fingerprint")
        self.submissions[submission.artifact_id] = submission
        return submission

    def transition(self, artifact_id: str, new_state: SubmissionState) -> Submission:
        current = self.submissions[artifact_id]
        if new_state == current.state:
            return current
        if new_state not in _TRANSITIONS[current.state]:
            raise ValueError(f"invalid transition {current.state.value} -> {new_state.value}")
        updated = replace(current, state=new_state)
        self.submissions[artifact_id] = updated
        return updated

    def select_finalist(self, duel_lower_bounds: dict[str, float]) -> Submission | None:
        """Pick the single finalist among dueled, promote-eligible candidates.

        Highest duel lower bound wins; exact ties go to the earliest commitment,
        then the artifact id. Returns None when nobody is eligible.
        """

        candidates = [
            self.submissions[artifact_id]
            for artifact_id in duel_lower_bounds
            if self.submissions[artifact_id].state == SubmissionState.DUELED
        ]
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda item: (
                duel_lower_bounds[item.artifact_id],
                -item.commitment.block,
                -item.commitment.extrinsic_index,
            ),
        )
        tied = [
            item
            for item in candidates
            if duel_lower_bounds[item.artifact_id] == duel_lower_bounds[best.artifact_id]
            and item.commitment == best.commitment
        ]
        return min(tied, key=lambda item: item.artifact_id)

    def lock_finalist(self, artifact_id: str) -> Submission:
        if self.confirmation_block_hash is not None:
            raise ValueError("the finalist must be locked before confirmation entropy is drawn")
        if self.finalist_artifact_id is not None and self.finalist_artifact_id != artifact_id:
            raise ValueError("the finalist is already locked")
        item = self.submissions[artifact_id]
        if item.state != SubmissionState.DUELED:
            raise ValueError("only a dueled submission can be locked as finalist")
        self.finalist_artifact_id = artifact_id
        return self.transition(artifact_id, SubmissionState.FINALIST_LOCKED)

    def draw_confirmation(self, block_hash: str) -> None:
        if self.finalist_artifact_id is None:
            raise ValueError("lock the finalist before drawing the confirmation block")
        if not block_hash:
            raise ValueError("confirmation block hash is required")
        if self.confirmation_block_hash not in {None, block_hash}:
            raise ValueError("confirmation entropy is immutable")
        self.confirmation_block_hash = block_hash

    def confirm(self, artifact_id: str) -> Submission:
        if artifact_id != self.finalist_artifact_id:
            raise ValueError("confirmation cannot select or replace the locked finalist")
        if self.confirmation_block_hash is None:
            raise ValueError("confirmation entropy has not been drawn")
        return self.transition(artifact_id, SubmissionState.CONFIRMED)

    def crown(self, artifact_id: str, *, certificate_verified: bool) -> Submission:
        if not certificate_verified:
            raise ValueError("a verified pre-consensus certificate is required")
        if self.certified and self.submissions[artifact_id].state == SubmissionState.CROWNED:
            return self.submissions[artifact_id]
        item = self.transition(artifact_id, SubmissionState.CROWNED)
        self.certified = True
        return item
