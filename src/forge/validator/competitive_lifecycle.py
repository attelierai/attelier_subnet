"""Small durable boundary from a qualified competitive round to weight intent.

This module deliberately has no wallet, extrinsic, GPU, or dashboard client.
It binds the pieces that are authority-bearing: the certified parent state, a
committee-signed admission history, an MVP certificate, and the current UID
snapshot used to construct a one-hot *intent*.  Calling it never publishes
weights.  Diagnostic evidence cannot cross this boundary because the supplied
``MVPCertificateVerifier`` independently replays only qualified execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from forge.chain.reward_publication import WeightIntent, testnet_weight_intent
from forge.evaluation.capacity import CapacityInputs, plan_capacity
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.mvp_evidence import validate_confirmation_panel
from forge.evaluation.mvp_policy import _sha
from forge.protocol.admission_checkpoint import SignedAdmissionHistory, verify_signed_history
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundCertificate, RoundPlan
from forge.protocol.mvp_certification import MVPCertificateVerifier
from forge.protocol.ports import Verifier
from forge.protocol.preconsensus import apply_certificate
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox
from forge.validator.competitive_bridge import DistributedMVPCertificateVerifier


class LifecycleError(ValueError):
    """A round does not extend the authenticated competitive history."""


def _state_from_store(store: Store, state: FinalizedCompetitionState) -> None:
    persisted = store.latest_state_payload(state.challenge_id)
    if persisted is None:
        if state.state_version != 0:
            raise LifecycleError("a non-genesis state is absent from the durable store")
        store.initialize_state(
            state_hash=state.state_hash(),
            challenge_id=state.challenge_id,
            state_version=state.state_version,
            previous_state_hash=state.previous_state_hash,
            payload=state.to_dict(),
        )
        return
    actual = FinalizedCompetitionState.from_dict(persisted)
    if actual.state_hash() != state.state_hash():
        raise LifecycleError("supplied state is not the durable certified head")


def _verify_history(
    plan: RoundPlan,
    signed_history: SignedAdmissionHistory,
    verifier: Verifier,
) -> None:
    history = signed_history.history
    if (history.network, history.netuid) != (plan.network, plan.netuid):
        raise LifecycleError("admission history belongs to another subnet")
    if history.through_block < plan.snapshot_block:
        raise LifecycleError("admission history predates the frozen round snapshot")
    if history.history_root != plan.prompt_seed_inputs.get("admission_history_root"):
        raise LifecycleError("round plan does not bind this admission history")
    if not verify_signed_history(
        history,
        signed_history,
        trusted_committee=plan.committee,
        expected_certified_state_hash=plan.parent_state_hash,
        verifier=verifier,
    ):
        raise LifecycleError("admission history lacks frozen committee authority")


def _verify_capacity_windows(
    plan: RoundPlan,
    distributed_plan: DistributedEvaluationPlan,
    capacity_inputs: CapacityInputs,
) -> None:
    """Recompute, then bind capacity to the frozen competitive plan.

    A capacity receipt is deliberately not accepted here: its status can be
    edited independently of its raw measurements.  The caller supplies the
    immutable distributed plan and raw inputs, which are replayed before the
    round claims its sole durable slot.
    """
    context = distributed_plan.chain_context
    evaluation_plan_id = _sha(
        {
            "schema": "forge/mvp-evaluation-plan/v1",
            "stage": distributed_plan.stage,
            "policy_id": distributed_plan.policy_id,
            "prompts": [dict(prompt) for prompt in distributed_plan.prompts],
            "candidate_manifest_hash": distributed_plan.candidate_manifest_hash,
            "incumbent_manifest_hash": distributed_plan.king_manifest_hash,
            "wan50_manifest_hash": distributed_plan.wan50_manifest_hash,
        }
    )
    if (
        distributed_plan.stage != "confirmation"
        or distributed_plan.round_id != plan.round_id
        or len(distributed_plan.timing_prompt_ids) != 32
        or context.get("mvp_plan_id") != evaluation_plan_id
        or plan.prompt_seed_inputs.get("evaluation_plan_id") != evaluation_plan_id
        or dict(distributed_plan.committee) != dict(plan.committee)
        or context.get("network") != plan.network
        or context.get("netuid") != plan.netuid
        or context.get("close_block") != plan.snapshot_block
        or context.get("round_plan_hash") != plan.plan_hash()
        or distributed_plan.candidate_manifest_hash != plan.challenger_fingerprint
        or distributed_plan.king_manifest_hash != plan.king_fingerprint
    ):
        raise LifecycleError("capacity plan is not bound to this competitive round")
    validate_confirmation_panel(distributed_plan.prompts)
    capacity = plan_capacity(distributed_plan, capacity_inputs)
    if capacity["status"] != "ADMISSIBLE_PROJECTION":
        raise LifecycleError("capacity inputs do not admit the complete frozen panel")
    windows = capacity["windows"]
    required_commit = windows.get("required_commit_window_blocks")
    required_reveal = windows.get("required_reveal_window_blocks")
    if type(required_commit) is not int or type(required_reveal) is not int:
        raise LifecycleError("capacity inputs require measured chain block duration")
    if plan.commit_deadline_block - plan.snapshot_block < required_commit:
        raise LifecycleError("competitive commit window cannot complete the frozen capacity plan")
    if plan.reveal_deadline_block - plan.commit_deadline_block < required_reveal:
        raise LifecycleError("competitive reveal window cannot deliver the frozen capacity plan")


@dataclass(frozen=True)
class WeightDelivery:
    """The only output for the chain publisher; it contains no signing authority."""

    state: FinalizedCompetitionState
    certificate_hash: str
    intent: WeightIntent


class CompetitiveLifecycle:
    """Persist round boundaries and prepare, but never submit, WTA weight intents."""

    def __init__(self, store: Store, signature_verifier: Verifier) -> None:
        self.store = store
        self.signature_verifier = signature_verifier

    def open_round(
        self,
        plan: RoundPlan,
        state: FinalizedCompetitionState,
        signed_history: SignedAdmissionHistory,
        *,
        distributed_plan: DistributedEvaluationPlan,
        capacity_inputs: CapacityInputs,
    ) -> bool:
        """Persist the sole active round after authenticating its admission root.

        SQLite's pending-round uniqueness makes this restart-safe and prevents
        silent overlap.  A caller may retry the exact same open safely.
        """
        if plan.parent_state_hash != state.state_hash() or plan.challenge_id != state.challenge_id:
            raise LifecycleError("round does not extend the supplied certified state")
        _state_from_store(self.store, state)
        _verify_history(plan, signed_history, self.signature_verifier)
        _verify_capacity_windows(plan, distributed_plan, capacity_inputs)
        return self.store.create_round(
            plan.round_id, plan.challenge_id, plan.plan_hash(), plan.to_dict()
        )

    def expire_round(
        self,
        mailbox: RoundMailbox,
        plan: RoundPlan,
        *,
        finalized_block: int,
        finalized_hash: str,
    ) -> dict[str, Any]:
        """Close a missing-message round without manufacturing a KEEP decision."""
        return mailbox.expire(plan, finalized_block=finalized_block, finalized_hash=finalized_hash)

    def certify_and_prepare_weight_intent(
        self,
        *,
        plan: RoundPlan,
        state: FinalizedCompetitionState,
        certificate: RoundCertificate,
        mvp_verifier: MVPCertificateVerifier,
        runtime_hash: str,
        registered_uids: Mapping[int, str],
        public_custody: Mapping[str, str],
        verify_public_custody: Callable[[str, str], bool] | None,
        burn_hotkey: str | None,
        verify_burn_qualification: Callable[[str], str | None] | None,
    ) -> WeightDelivery:
        """Apply an independently replayed certificate and prepare a testnet intent.

        ``testnet_weight_intent`` is intentionally pure.  The caller must hand
        its result to a separate, explicitly authorized publisher if desired.
        """
        _state_from_store(self.store, state)
        if certificate.transition.value == "PROMOTE" and not isinstance(
            mvp_verifier, DistributedMVPCertificateVerifier
        ):
            raise LifecycleError(
                "promotion requires a replaying distributed MVP execution verifier"
            )
        # Validate the entire delivery boundary first.  A malformed custody or
        # runtime receipt must not persist a crown and then fail while forming
        # its weight intent.
        next_state = apply_certificate(state, plan, certificate, mvp_verifier)
        intent = testnet_weight_intent(
            next_state,
            network=plan.network,
            netuid=plan.netuid,
            certificate_hash=certificate.transition_hash(),
            runtime_hash=runtime_hash,
            registered_uids=registered_uids,
            public_custody=public_custody,
            verify_public_custody=verify_public_custody,
            burn_hotkey=burn_hotkey,
            verify_burn_qualification=verify_burn_qualification,
        )
        persisted = apply_certificate(state, plan, certificate, mvp_verifier, store=self.store)
        if persisted.state_hash() != next_state.state_hash():
            raise LifecycleError("persisted certificate state differs from verified transition")
        return WeightDelivery(persisted, certificate.transition_hash(), intent)
