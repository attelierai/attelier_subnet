"""Concrete authority checks for qualified testnet publication.

These checks consume durable protocol objects; callers cannot replace them with
an evidence-origin string or a permissive callback in a live factory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from forge.chain import Sr25519Verifier
from forge.chain.launch_publication import BootstrapAuthority, QualifiedPublicationContext
from forge.chain.reward_publication import WeightIntent
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundCertificate, RoundPlan
from forge.protocol.hashing import hash_canonical
from forge.protocol.preconsensus import apply_certificate
from forge.protocol.store import Store
from forge.validator.launch_bridge import (
    LaunchCertificateVerifier,
    LaunchExecutionReplayer,
    LocalLaunchReceiptStore,
)


@dataclass(frozen=True)
class QualifiedLaunchAuthority:
    store: Store
    plan: RoundPlan
    parent_state: FinalizedCompetitionState
    certificate: RoundCertificate | None
    activation_plan: Mapping[str, Any]
    receipt_root: str | None = None

    def verify(self, context: QualifiedPublicationContext, intent: WeightIntent) -> bool:
        activation = dict(self.activation_plan)
        if activation.get("synthetic") is True or activation.get("fixture") is True:
            return False
        activation_body = {
            key: value for key, value in activation.items() if key != "activation_plan_hash"
        }
        activation_hash = hash_canonical("forge/launch-activation-plan/v1", activation_body)
        if (
            activation.get("activation_plan_hash") != activation_hash
            or activation_hash != context.activation_plan_hash
        ):
            return False
        if (context.network, context.netuid, context.contract_hash, context.plan_hash) != (
            self.plan.network,
            self.plan.netuid,
            self.plan.contract_hash,
            self.plan.plan_hash(),
        ):
            return False
        if activation.get("genesis_hash") != context.genesis_hash:
            return False
        if (
            activation.get("runtime_hash") != context.runtime_hash
            or activation.get("mechanism") != 0
        ):
            return False
        if (
            activation.get("network") != context.network
            or activation.get("netuid") != context.netuid
            or activation.get("contract_hash") != context.contract_hash
            or activation.get("round_plan_hash") != context.plan_hash
            or activation.get("certified_state_hash") != context.certified_state_hash
            or activation.get("certificate_hash") != context.certificate_hash
        ):
            return False
        # This is provenance, never authority by itself: replayed raw controls
        # and the frozen V2 policy below are still mandatory.
        if activation.get("evidence_origin") not in {"external_gpu_qualified_v2"}:
            return False
        policy_plan = activation.get("policy_plan")
        qualification_package = activation.get("qualification_package")
        if (
            not isinstance(policy_plan, Mapping)
            or policy_plan.get("schema") != "forge/launch-policy/v2"
            or policy_plan.get("models", {}).get("candidate") != self.plan.challenger_fingerprint
            or policy_plan.get("models", {}).get("king") != self.plan.king_fingerprint
            or policy_plan.get("plan_id") != self.plan.prompt_seed_inputs.get("launch_plan_id")
        ):
            return False
        try:
            from forge.evaluation.evaluator_qualification import replay_qualification_package
            from forge.evaluation.launch_policy import validate_plan

            validate_plan(
                dict(policy_plan),
                expected_round_id=self.plan.round_id,
                expected_entropy_hash=self.plan.prompt_seed_inputs.get("entropy_hash"),
            )
            if replay_qualification_package(qualification_package) != policy_plan.get(
                "evaluator_qualification"
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        if self.certificate is None:
            return (
                context.certificate_hash is None
                and context.certified_state_hash == self.parent_state.state_hash()
                and self.store.latest_state_payload(self.parent_state.challenge_id)
                == self.parent_state.to_dict()
            )
        if context.certificate_hash != self.certificate.transition_hash():
            return False
        if (
            self.store.certificate_payload(self.certificate.transition_hash())
            != self.certificate.to_dict()
        ):
            return False
        if self.receipt_root is None:
            return False
        try:
            verifier = LaunchCertificateVerifier(
                Sr25519Verifier(),
                replayer=LaunchExecutionReplayer(LocalLaunchReceiptStore(self.receipt_root)),
                store=self.store,
            )
            verifier.verify(self.certificate, self.plan, self.parent_state)
            certified = apply_certificate(self.parent_state, self.plan, self.certificate, verifier)
        except (KeyError, ValueError):
            return False
        if intent.status == "WINNER":
            latest = certified.crown_history[-1] if certified.crown_history else None
            if (
                latest is None
                or latest.baseline
                or latest.fingerprint != certified.current_king_fingerprint
                or len(intent.uid_weights) != 1
                or set(intent.uid_weights) != set(intent.expected_hotkeys)
                or next(iter(intent.uid_weights.values())) != 1.0
                or next(iter(intent.expected_hotkeys.values())) != latest.hotkey
            ):
                return False
        return (
            certified.state_hash() == context.certified_state_hash == intent.certified_state_hash
            and self.store.latest_state_payload(self.parent_state.challenge_id)
            == certified.to_dict()
        )


class BootstrapSignatureAuthority:
    """Verify bootstrap authorization against a public sr25519 trust anchor."""

    def __init__(self, trusted_authority_hotkey: str) -> None:
        if not trusted_authority_hotkey:
            raise ValueError("bootstrap trust anchor is required")
        self.trusted_authority_hotkey = trusted_authority_hotkey
        self.verifier = Sr25519Verifier()

    def verify(self, authority: BootstrapAuthority) -> bool:
        if authority.authority_id != self.trusted_authority_hotkey:
            return False
        try:
            return self.verifier.verify(
                authority.authority_id,
                authority.signed_payload(),
                bytes.fromhex(authority.signature),
            )
        except ValueError:
            return False
