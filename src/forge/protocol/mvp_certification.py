"""MVP evidence bridge to existing commit/reveal certificates.

The independent execution callback is a validator-owned boundary, not an
importable miner recipe. Signatures alone never authorize PROMOTE. Historical
certificate replay remains in preconsensus; only this stricter path may feed
MVP reward activation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from forge.protocol.consensus_types import (
    CertificateError,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    LocalVerdict,
    RoundCertificate,
    RoundPlan,
)
from forge.protocol.hashing import hash_canonical
from forge.protocol.ports import Verifier
from forge.protocol.preconsensus import CertificateVerifier
from forge.protocol.store import Store

BUNDLE_DOMAIN = "forge/mvp-execution-bundle/v1"


def local_report(
    plan: RoundPlan,
    validator_hotkey: str,
    bundle: Mapping[str, Any],
    *,
    independently_execute: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    verify_public_copy: Callable[[str, Mapping[str, Any]], bool] | None,
) -> LocalEvaluationReport:
    """Execute/replay locally before assigning a competitive local verdict.

    Missing trusted execution/custody interfaces always yields INCOMPLETE.
    Their installation must follow calibration and real runtime qualification.
    """
    verdict = LocalVerdict.INCOMPLETE
    evidence_hash = hash_canonical(BUNDLE_DOMAIN, bundle)
    reasons = []
    if validator_hotkey not in plan.committee:
        raise CertificateError("local validator is outside the frozen committee")
    bindings = bundle.get("bindings", {})
    required = {
        "round_plan_hash": plan.plan_hash(),
        "candidate_manifest_hash": plan.challenger_fingerprint,
        "incumbent_manifest_hash": plan.king_fingerprint,
        "parent_state_hash": plan.parent_state_hash,
    }
    if any(bindings.get(key) != value for key, value in required.items()):
        reasons.append("execution evidence does not match the frozen round")
    if bundle.get("evidence_kind") != "validator_wan_execution":
        reasons.append("simulated, historical or diagnostic evidence cannot certify")
    # The plan itself binds these authenticated histories before panel reveal.
    for key in ("admission_history_root", "prompt_history_root", "evaluation_plan_id"):
        value = plan.prompt_seed_inputs.get(key)
        if not isinstance(value, str) or len(value) != 64 or bindings.get(key) != value:
            reasons.append(f"missing certified binding: {key}")
    if independently_execute is None:
        reasons.append("independent validator execution is unavailable")
    gate = None
    custody_hash = None
    if not reasons:
        gate = dict(independently_execute(bundle))
        if gate.get("status") == "PROMOTE_ELIGIBLE":
            custody = bundle.get("public_custody")
            if (
                not isinstance(custody, Mapping)
                or custody.get("manifest_hash") != plan.challenger_fingerprint
                or verify_public_copy is None
                or not verify_public_copy(plan.challenger_fingerprint, custody)
            ):
                reasons.append("durable anonymous public copy is not verified")
            else:
                custody_hash = hash_canonical("forge/mvp-public-custody/v1", custody)
                verdict = LocalVerdict.PROMOTE
        elif gate.get("status") in {"FAIL", "INCONCLUSIVE"}:
            verdict = LocalVerdict.KEEP
        else:
            reasons.append("independent evaluation is incomplete")
    return LocalEvaluationReport(
        network=plan.network,
        netuid=plan.netuid,
        round_id=plan.round_id,
        validator_hotkey=validator_hotkey,
        plan_hash=plan.plan_hash(),
        parent_state_hash=plan.parent_state_hash,
        contract_hash=plan.contract_hash,
        king_fingerprint=plan.king_fingerprint,
        challenger_fingerprint=plan.challenger_fingerprint,
        verdict=verdict,
        metrics={
            "mvp_evidence_hash": evidence_hash,
            "gate": gate,
            "reasons": reasons,
            "public_custody_hash": custody_hash,
        },
    )


class MVPCertificateVerifier(CertificateVerifier):
    def __init__(
        self,
        signature_verifier: Verifier,
        *,
        evidence: Mapping[str, Mapping[str, Any]],
        independently_execute: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        verify_public_copy: Callable[[str, Mapping[str, Any]], bool] | None = None,
        store: Store | None = None,
    ):
        super().__init__(signature_verifier, store)
        self.evidence = evidence
        self.independently_execute = independently_execute
        self.verify_public_copy = verify_public_copy

    def replay_bundle(self, bundle: Mapping[str, Any], plan: RoundPlan) -> Mapping[str, Any] | None:
        """Replay one evidence bundle for this frozen plan.

        Subclasses with a concrete execution transport override this rather
        than installing a mutable callback for each certificate verification.
        A missing executor deliberately remains incomplete.
        """

        if self.independently_execute is None:
            return {"status": "INCOMPLETE"}
        return self.independently_execute(bundle)

    def verify(
        self, certificate: RoundCertificate, plan: RoundPlan, state: FinalizedCompetitionState
    ) -> None:
        # First check pure protocol signatures/quorum without persisting attestations.
        CertificateVerifier(self.signature_verifier).verify(certificate, plan, state)
        checked = {}
        for reveal in certificate.reveals:
            report = reveal.report
            if report.validator_hotkey not in certificate.signatures:
                continue
            digest = report.metrics.get("mvp_evidence_hash")
            bundle = self.evidence.get(digest)
            if bundle is None or hash_canonical(BUNDLE_DOMAIN, bundle) != digest:
                raise CertificateError("signed MVP evidence is absent or altered")
            if digest not in checked:
                checked[digest] = local_report(
                    plan,
                    report.validator_hotkey,
                    bundle,
                    independently_execute=lambda value: self.replay_bundle(value, plan),
                    verify_public_copy=self.verify_public_copy,
                )
            replay = checked[digest]
            if replay.verdict == LocalVerdict.INCOMPLETE or replay.verdict != report.verdict:
                raise CertificateError(
                    "independent MVP execution does not reproduce signed verdict"
                )
            if replay.metrics["public_custody_hash"] != report.metrics.get("public_custody_hash"):
                raise CertificateError("signed public custody differs from verified bundle")
            # Independent stochastic model execution need not reproduce identical floats;
            # it must independently reproduce the competitive decision. The bundle hash
            # authenticates the original signed observations separately.
            if report.metrics.get("gate", {}).get("status") != replay.metrics["gate"]["status"]:
                raise CertificateError("signed MVP gate differs from independent decision")
        super().verify(certificate, plan, state)
