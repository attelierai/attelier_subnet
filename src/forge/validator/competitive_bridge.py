"""Replay the signed distributed confirmation panel before certification.

This is deliberately a narrow authority boundary.  A distributed aggregate is
useful telemetry, but it is not an evaluation result a validator may vote on
until this module has reconstructed it from every signed partial report and
has run the frozen confirmation gate again.  In particular, a JSON field named
``PROMOTE_ELIGIBLE`` is never trusted.

The bridge does not render, download weights, sign a chain transaction, or
publish weights.  It produces the execution bundle consumed by the existing
MVP certificate verifier and only for a full, calibrated confirmation panel.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from forge.evaluation.calibration_sources import _normalized_prompt
from forge.evaluation.distributed import replay_qualified_distributed, verify_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.mvp_calibration import validate_qualified_calibration
from forge.evaluation.mvp_evidence import _quality_rows, validate_confirmation_panel
from forge.evaluation.mvp_policy import QUALITY_AXES, EvaluationPolicy
from forge.protocol.consensus_types import (
    CertificateError,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    RoundCertificate,
    RoundPlan,
)
from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.mvp_certification import MVPCertificateVerifier, local_report
from forge.protocol.ports import Verifier

BUNDLE_SCHEMA = "forge/competitive-distributed-execution/v2"
PANEL_VALIDATION_SCHEMA = "forge/independent-panel-validation/v1"
WORKER_TOLERANCE_SCHEMA = "forge/cross-worker-tolerance/v1"


class DistributedBridgeError(ValueError):
    """The signed distributed evidence cannot be a competitive vote."""


def _hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _worker_bindings(value: object, committee: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """Validate the controlled runtime/hardware population used for an audit.

    A GPU name is not a runtime identity.  The immutable runtime ID and the
    dependency lock are deliberately carried together, so a tolerance measured
    before a container or metric update cannot be reused afterwards.
    """

    if not isinstance(value, Mapping) or len(committee) < 2 or set(value) != set(committee):
        raise DistributedBridgeError("worker bindings do not cover the frozen committee")
    result: dict[str, dict[str, Any]] = {}
    for hotkey in committee:
        binding = value[hotkey]
        if not isinstance(binding, Mapping) or set(binding) != {"runtime_id", "hardware"}:
            raise DistributedBridgeError("worker binding fields are not exact")
        hardware = binding["hardware"]
        if (
            not _hash(binding["runtime_id"])
            or not isinstance(hardware, Mapping)
            or set(hardware) != {"gpu", "precision", "dependency_lock_hash"}
            or not isinstance(hardware["gpu"], str)
            or not hardware["gpu"]
            or not isinstance(hardware["precision"], str)
            or not hardware["precision"]
            or not _hash(hardware["dependency_lock_hash"])
        ):
            raise DistributedBridgeError("worker runtime or hardware binding is invalid")
        result[hotkey] = {"runtime_id": binding["runtime_id"], "hardware": dict(hardware)}
    if (
        len({binding["runtime_id"] for binding in result.values()}) != 1
        or len(
            {
                hash_canonical("forge/worker-hardware/v1", binding["hardware"])
                for binding in result.values()
            }
        )
        != 1
    ):
        raise DistributedBridgeError(
            "initial worker qualification requires identical runtime and hardware"
        )
    return result


def _qualification_evidence(
    *,
    panel_validation: Mapping[str, Any],
    worker_tolerances: Mapping[str, Any],
    policy: EvaluationPolicy,
    calibration: Mapping[str, Any],
    calibration_hash: str,
    distributed_plan: DistributedEvaluationPlan,
) -> dict[str, Any]:
    """Require the two receipts calibration alone cannot establish.

    Calibration says the evaluator distinguishes its source controls.  It does
    not prove the 128-prompt decision unit rejects panel-level degradations, or
    that two controlled workers agree within the tolerances put in a plan.
    These receipts are supplied as immutable evidence and are replayed by every
    certifying validator; no current receipt is synthesized here.
    """

    expected_panel = {
        "schema",
        "policy_id",
        "calibration_hash",
        "evaluator_provenance",
        "worker_bindings",
        "panels",
    }
    expected_tolerances = {
        "schema",
        "policy_id",
        "calibration_hash",
        "evaluator_provenance",
        "development",
        "independent",
        "worker_bindings",
        "quality_tolerance",
        "timing_log_ratio_tolerance",
        "tolerance_derivation_hash",
    }
    if not isinstance(panel_validation, Mapping) or set(panel_validation) != expected_panel:
        raise DistributedBridgeError("independent panel validation receipt fields are not exact")
    if not isinstance(worker_tolerances, Mapping) or set(worker_tolerances) != expected_tolerances:
        raise DistributedBridgeError("cross-worker tolerance receipt fields are not exact")
    if (
        panel_validation["schema"] != PANEL_VALIDATION_SCHEMA
        or worker_tolerances["schema"] != WORKER_TOLERANCE_SCHEMA
        or panel_validation["policy_id"] != policy.policy_id
        or worker_tolerances["policy_id"] != policy.policy_id
        or panel_validation["calibration_hash"] != calibration_hash
        or worker_tolerances["calibration_hash"] != calibration_hash
        or panel_validation["evaluator_provenance"] != dict(policy.evaluator_provenance or {})
        or worker_tolerances["evaluator_provenance"] != dict(policy.evaluator_provenance or {})
    ):
        raise DistributedBridgeError(
            "qualification evidence is not bound to policy calibration and evaluator"
        )
    if not _hash(worker_tolerances["tolerance_derivation_hash"]):
        raise DistributedBridgeError("qualification evidence lacks independent data bindings")

    def tolerance_rows(value: object, label: str) -> list[Mapping[str, Any]]:
        if not isinstance(value, list) or len(value) != 8:
            raise DistributedBridgeError(f"cross-worker {label} needs eight families")
        if (
            any(
                not isinstance(row, Mapping)
                or set(row) != {"family_id", "workers"}
                or not isinstance(row["family_id"], str)
                or not row["family_id"]
                or not isinstance(row["workers"], Mapping)
                or set(row["workers"]) != set(distributed_plan.committee)
                or any(
                    not isinstance(worker, Mapping)
                    or set(worker) != {"quality_axes", "runs"}
                    or not isinstance(worker["quality_axes"], Mapping)
                    or set(worker["quality_axes"]) != set(QUALITY_AXES)
                    or any(
                        type(axis) not in {int, float} or not math.isfinite(axis)
                        for axis in worker["quality_axes"].values()
                    )
                    or not isinstance(worker["runs"], list)
                    or len(worker["runs"]) != 4
                    or [run.get("model") for run in worker["runs"] if isinstance(run, Mapping)]
                    != ["reference", "candidate", "candidate", "reference"]
                    or any(
                        not isinstance(run, Mapping)
                        or type(run.get("seconds")) not in {int, float}
                        or not math.isfinite(run["seconds"])
                        or run["seconds"] <= 0
                        for run in worker["runs"]
                    )
                    for worker in row["workers"].values()
                )
                for row in value
            )
            or len({row["family_id"] for row in value}) != 8
        ):
            raise DistributedBridgeError(f"cross-worker {label} rows are invalid")
        return list(value)

    development = tolerance_rows(worker_tolerances["development"], "development")
    independent = tolerance_rows(worker_tolerances["independent"], "independent")
    if {row["family_id"] for row in development} & {row["family_id"] for row in independent}:
        raise DistributedBridgeError("cross-worker tolerance check reuses development families")

    def differences(row: Mapping[str, Any]) -> tuple[float, float]:
        workers = list(row["workers"].values())
        quality = max(
            max(float(worker["quality_axes"][axis]) for worker in workers)
            - min(float(worker["quality_axes"][axis]) for worker in workers)
            for axis in QUALITY_AXES
        )
        ratios = [
            math.log((worker["runs"][0]["seconds"] + worker["runs"][3]["seconds"]) / 2)
            - math.log((worker["runs"][1]["seconds"] + worker["runs"][2]["seconds"]) / 2)
            for worker in workers
        ]
        return quality, max(ratios) - min(ratios)

    derived_quality = max(differences(row)[0] for row in development)
    derived_timing = max(differences(row)[1] for row in development)
    if worker_tolerances["tolerance_derivation_hash"] != hash_canonical(
        "forge/cross-worker-tolerance-development/v1", development
    ):
        raise DistributedBridgeError("cross-worker tolerance derivation hash is invalid")
    panel_workers = _worker_bindings(
        panel_validation["worker_bindings"], distributed_plan.committee
    )
    tolerance_workers = _worker_bindings(
        worker_tolerances["worker_bindings"], distributed_plan.committee
    )
    if panel_workers != tolerance_workers:
        raise DistributedBridgeError("panel validation and worker tolerance runtimes differ")
    if (
        distributed_plan.quality_tolerance is None
        or distributed_plan.timing_log_ratio_tolerance is None
        or type(worker_tolerances["quality_tolerance"]) not in {int, float}
        or type(worker_tolerances["timing_log_ratio_tolerance"]) not in {int, float}
        or worker_tolerances["quality_tolerance"] != derived_quality
        or worker_tolerances["timing_log_ratio_tolerance"] != derived_timing
        or worker_tolerances["quality_tolerance"] != distributed_plan.quality_tolerance
        or worker_tolerances["timing_log_ratio_tolerance"]
        != distributed_plan.timing_log_ratio_tolerance
    ):
        raise DistributedBridgeError("distributed audit tolerances are not qualified")
    if any(
        differences(row)[0] > derived_quality or differences(row)[1] > derived_timing
        for row in independent
    ):
        raise DistributedBridgeError("cross-worker tolerances failed their independent families")
    panels = panel_validation["panels"]
    if not isinstance(panels, Mapping) or set(panels) != {"same_model", "blur_mild", "wrong_video"}:
        raise DistributedBridgeError("independent validation needs three raw 128-prompt panels")
    excluded = [*calibration["campaign"]["source_bank"], *distributed_plan.prompts]
    excluded_families = {
        source.get(field)
        for source in excluded
        for field in ("source_family", "concept_family")
        if source.get(field)
    }
    excluded_texts = {
        _normalized_prompt(source.get("prompt_text", source.get("text", ""))) for source in excluded
    }
    decisions: dict[str, object] = {}
    panel_slots: set[tuple[str, str, str, str]] | None = None
    panel_families: set[str] | None = None
    for name, panel in panels.items():
        if not isinstance(panel, Mapping) or set(panel) != {
            "prompts",
            "quality_vs_king",
            "source_family_ids",
        }:
            raise DistributedBridgeError("independent panel fields are not exact")
        prompts, families = panel["prompts"], panel["source_family_ids"]
        if (
            not isinstance(prompts, Sequence)
            or isinstance(prompts, (str, bytes))
            or len(prompts) != 128
            or not isinstance(families, list)
            or len(families) != 128
            or len(set(families)) != 128
            or not all(isinstance(family, str) and family for family in families)
            or any(
                not isinstance(prompt, Mapping) or prompt.get("source_family") != family
                for prompt, family in zip(prompts, families, strict=True)
            )
        ):
            raise DistributedBridgeError("independent panel families are invalid")
        try:
            validate_confirmation_panel(prompts)
            if (
                any(
                    not isinstance(prompt.get("text"), str)
                    or not prompt["text"].strip()
                    or not isinstance(prompt.get("concept_family"), str)
                    or not prompt["concept_family"]
                    or type(prompt.get("seed")) is not int
                    or prompt["source_family"] in excluded_families
                    or prompt["concept_family"] in excluded_families
                    or _normalized_prompt(prompt["text"]) in excluded_texts
                    for prompt in prompts
                )
                or len({prompt["concept_family"] for prompt in prompts}) != 128
            ):
                raise ValueError("panel reuses exposed families or lacks lineage")
            slots = {
                (
                    str(prompt["prompt_id"]),
                    str(prompt["subject_class"]),
                    str(prompt["camera_type"]),
                    str(prompt["style_family"]),
                    prompt["seed"],
                    prompt["source_family"],
                    prompt["concept_family"],
                    hash_canonical("forge/mvp-prompt/v1", prompt["text"]),
                )
                for prompt in prompts
            }
            rows = panel["quality_vs_king"]
            if (
                not isinstance(rows, Sequence)
                or {
                    (
                        str(row.get("prompt_id")),
                        str(row.get("subject_class")),
                        str(row.get("camera_type")),
                        str(row.get("style_family")),
                        row.get("seed"),
                        row.get("source_family"),
                        row.get("concept_family"),
                        row.get("prompt_hash"),
                    )
                    for row in rows
                    if isinstance(row, Mapping)
                }
                != slots
            ):
                raise ValueError
            if panel_slots is None:
                panel_slots, panel_families = slots, set(families)
            elif slots != panel_slots or set(families) != panel_families:
                raise ValueError
            means, cells, lower, cell_lower = _quality_rows(panel["quality_vs_king"], 128)
        except (TypeError, ValueError):
            raise DistributedBridgeError("independent panel raw quality rows are invalid") from None
        values = [*means.values(), *(value for axes in cells.values() for value in axes.values())]
        lower_values = [
            *lower.values(),
            *(value for axes in cell_lower.values() for value in axes.values()),
        ]
        if set(means) != set(QUALITY_AXES) or set(lower) != set(QUALITY_AXES):
            raise DistributedBridgeError("independent panel quality axes are incomplete")
        decisions[name] = {
            "quality_clear": min(lower_values) >= -policy.quality_margin,
            "quality_rejected": min(values) < -policy.quality_margin,
        }
    if (
        not decisions["same_model"]["quality_clear"]
        or not decisions["blur_mild"]["quality_rejected"]
        or not decisions["wrong_video"]["quality_rejected"]
    ):
        raise DistributedBridgeError("independent panel replay did not prove required decisions")
    return {
        "panel_validation_hash": hash_canonical(PANEL_VALIDATION_SCHEMA, dict(panel_validation)),
        "worker_tolerance_hash": hash_canonical(WORKER_TOLERANCE_SCHEMA, dict(worker_tolerances)),
        "worker_bindings": panel_workers,
    }


def _bind_report_runtimes(
    reports: Sequence[Mapping[str, Any]],
    plan: DistributedEvaluationPlan,
    verifier: Verifier,
    workers: Mapping[str, Mapping[str, Any]],
) -> None:
    """Every signed contribution must name the runtime calibrated for its hotkey."""

    for report in reports:
        body = verify_partial_report(plan, report, verifier)
        worker = workers[body["validator_hotkey"]]
        for field in ("quality_vs_king", "quality_vs_wan50", "timing"):
            for row in body[field]:
                if row.get("runtime_id") != worker["runtime_id"]:
                    raise DistributedBridgeError(
                        "signed evidence runtime differs from tolerance receipt"
                    )
                if field == "timing" and row.get("hardware") != worker["hardware"]:
                    raise DistributedBridgeError(
                        "signed timing hardware differs from tolerance receipt"
                    )


def _exact_mapping(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DistributedBridgeError(f"{label} fields are not exact")
    return value


def _round_bindings(
    round_plan: RoundPlan, distributed_plan: DistributedEvaluationPlan, mvp_plan: Mapping[str, Any]
) -> dict[str, str]:
    inputs = round_plan.prompt_seed_inputs
    required_inputs = ("admission_history_root", "prompt_history_root", "evaluation_plan_id")
    if any(
        not isinstance(inputs.get(key), str) or len(inputs[key]) != 64 for key in required_inputs
    ):
        raise DistributedBridgeError("round plan lacks certified prompt and admission bindings")
    if inputs["evaluation_plan_id"] != mvp_plan.get("plan_id"):
        raise DistributedBridgeError("round plan does not bind the frozen MVP evaluation plan")
    if dict(round_plan.committee) != dict(distributed_plan.committee):
        raise DistributedBridgeError(
            "distributed committee differs from the frozen round committee"
        )
    if (
        distributed_plan.candidate_manifest_hash != round_plan.challenger_fingerprint
        or distributed_plan.king_manifest_hash != round_plan.king_fingerprint
        or distributed_plan.policy_id == ""
    ):
        raise DistributedBridgeError("distributed model or policy binding differs from the round")
    context = distributed_plan.chain_context
    if (
        context.get("network") != round_plan.network
        or context.get("netuid") != round_plan.netuid
        or context.get("close_block") != round_plan.snapshot_block
        or context.get("round_plan_hash") != round_plan.plan_hash()
        or context.get("mvp_plan_id") != mvp_plan.get("plan_id")
    ):
        raise DistributedBridgeError("distributed plan chain context differs from frozen round")
    return {
        "round_plan_hash": round_plan.plan_hash(),
        "candidate_manifest_hash": round_plan.challenger_fingerprint,
        "incumbent_manifest_hash": round_plan.king_fingerprint,
        "parent_state_hash": round_plan.parent_state_hash,
        **{key: inputs[key] for key in required_inputs},
    }


def build_distributed_execution_bundle(
    *,
    round_plan: RoundPlan,
    distributed_plan: DistributedEvaluationPlan,
    reports: Sequence[Mapping[str, Any]],
    mvp_plan: Mapping[str, Any],
    policy: EvaluationPolicy,
    calibration: Mapping[str, Any],
    panel_validation: Mapping[str, Any],
    worker_tolerances: Mapping[str, Any],
    public_custody: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an immutable candidate bundle; replay happens before every vote.

    This function is intentionally cheap and local.  It only captures the
    exact signed reports and frozen inputs.  ``DistributedExecutionReplayer``
    later reconstructs the result on every certifying validator.
    """

    bindings = _round_bindings(round_plan, distributed_plan, mvp_plan)
    if distributed_plan.stage != "confirmation":
        raise DistributedBridgeError("only confirmation evidence can enter certification")
    if distributed_plan.policy_id != policy.policy_id:
        raise DistributedBridgeError("distributed plan policy differs from the frozen policy")
    if (
        mvp_plan.get("plan_id") != round_plan.prompt_seed_inputs["evaluation_plan_id"]
        or mvp_plan.get("policy_id") != policy.policy_id
        or mvp_plan.get("stage") != "confirmation"
        or canonical_json_bytes(mvp_plan.get("prompts"))
        != canonical_json_bytes([dict(prompt) for prompt in distributed_plan.prompts])
    ):
        raise DistributedBridgeError("distributed plan differs from the frozen MVP evaluation plan")
    calibration_binding = validate_qualified_calibration(calibration, policy)
    if (
        calibration_binding["status"] != "PASS"
        or distributed_plan.audit_calibration_hash
        != calibration_binding["binding"]["calibration_hash"]
    ):
        raise DistributedBridgeError("distributed plan does not bind the calibration receipt")
    qualification = _qualification_evidence(
        panel_validation=panel_validation,
        worker_tolerances=worker_tolerances,
        policy=policy,
        calibration=calibration,
        calibration_hash=calibration_binding["binding"]["calibration_hash"],
        distributed_plan=distributed_plan,
    )
    custody = dict(public_custody)
    if custody.get("manifest_hash") != round_plan.challenger_fingerprint:
        raise DistributedBridgeError("public custody does not name the frozen challenger")
    return {
        "schema": BUNDLE_SCHEMA,
        "evidence_kind": "validator_wan_execution",
        "round_plan": round_plan.to_dict(),
        "distributed_plan": distributed_plan.to_dict(),
        "mvp_plan": dict(mvp_plan),
        "reports": [dict(report) for report in reports],
        "policy": policy.to_dict(),
        "calibration": dict(calibration),
        "panel_validation": dict(panel_validation),
        "worker_tolerances": dict(worker_tolerances),
        "public_custody": custody,
        "bindings": {**bindings, "qualification": qualification},
    }


class DistributedExecutionReplayer:
    """Deterministic, validator-owned replay for the competitive bridge."""

    def __init__(self, signature_verifier: Verifier) -> None:
        self.signature_verifier = signature_verifier

    def replay(self, bundle: Mapping[str, Any], round_plan: RoundPlan) -> dict[str, Any]:
        raw = _exact_mapping(
            bundle,
            {
                "schema",
                "evidence_kind",
                "round_plan",
                "distributed_plan",
                "mvp_plan",
                "reports",
                "policy",
                "calibration",
                "panel_validation",
                "worker_tolerances",
                "public_custody",
                "bindings",
            },
            "distributed execution bundle",
        )
        if raw["schema"] != BUNDLE_SCHEMA:
            raise DistributedBridgeError("unknown distributed execution bundle")
        if raw["evidence_kind"] != "validator_wan_execution":
            raise DistributedBridgeError(
                "distributed execution is not validator-owned Wan evidence"
            )
        try:
            bundled_round = RoundPlan.from_dict(dict(raw["round_plan"]))
            distributed_plan = DistributedEvaluationPlan.from_dict(dict(raw["distributed_plan"]))
            policy = EvaluationPolicy.from_dict(raw["policy"])
        except (TypeError, ValueError) as error:
            raise DistributedBridgeError("distributed execution bundle is malformed") from error
        if bundled_round.plan_hash() != round_plan.plan_hash():
            raise DistributedBridgeError("execution bundle belongs to another round")
        mvp_plan = raw["mvp_plan"]
        if not isinstance(mvp_plan, Mapping):
            raise DistributedBridgeError("frozen MVP evaluation plan is missing")
        bindings = _round_bindings(round_plan, distributed_plan, mvp_plan)
        calibration = raw["calibration"]
        if not isinstance(calibration, Mapping):
            raise DistributedBridgeError("calibration receipt is missing")
        calibration_binding = validate_qualified_calibration(calibration, policy)
        if (
            calibration_binding["status"] != "PASS"
            or distributed_plan.audit_calibration_hash
            != calibration_binding["binding"]["calibration_hash"]
        ):
            raise DistributedBridgeError("calibration receipt differs from the frozen plan")
        panel_validation = raw["panel_validation"]
        worker_tolerances = raw["worker_tolerances"]
        if not isinstance(panel_validation, Mapping) or not isinstance(worker_tolerances, Mapping):
            raise DistributedBridgeError("qualification receipts are missing")
        qualification = _qualification_evidence(
            panel_validation=panel_validation,
            worker_tolerances=worker_tolerances,
            policy=policy,
            calibration=calibration,
            calibration_hash=calibration_binding["binding"]["calibration_hash"],
            distributed_plan=distributed_plan,
        )
        if dict(raw["bindings"]) != {**bindings, "qualification": qualification}:
            raise DistributedBridgeError(
                "execution bundle bindings differ from qualification evidence"
            )
        custody = raw["public_custody"]
        if (
            not isinstance(custody, Mapping)
            or custody.get("manifest_hash") != round_plan.challenger_fingerprint
        ):
            raise DistributedBridgeError("public custody differs from frozen challenger")
        reports = raw["reports"]
        if (
            not isinstance(reports, list)
            or not reports
            or not all(isinstance(report, Mapping) for report in reports)
        ):
            raise DistributedBridgeError("signed distributed reports are required")
        _bind_report_runtimes(
            reports, distributed_plan, self.signature_verifier, qualification["worker_bindings"]
        )
        # This function validates signatures, assignment coverage, duplicate and
        # contested evidence, calibration, and the exact 128/32 gate itself.
        result = replay_qualified_distributed(
            distributed_plan,
            reports,
            self.signature_verifier,
            policy,
            calibration,
            mvp_plan=mvp_plan,
        )
        gate = result["decision"]
        if not isinstance(gate, Mapping):
            raise DistributedBridgeError("qualified replay returned no gate")
        binding = result.get("eligibility_binding")
        if gate.get("status") == "PROMOTE_ELIGIBLE" and not isinstance(binding, Mapping):
            raise DistributedBridgeError("eligible replay lacks its immutable binding")
        if not isinstance(binding, Mapping):
            # An incomplete, failing, or inconclusive replay is a valid local
            # outcome.  It must become a non-promotion vote, not an exception
            # that a caller can accidentally retry under different evidence.
            return {
                "status": gate.get("status"),
                "reasons": list(gate.get("reasons", [])),
                "execution_bundle_hash": hash_canonical(BUNDLE_SCHEMA, raw),
                "replayed": True,
            }
        # LocalEvaluationReport is hash-serialized.  The evaluator's display
        # payload can contain tuple category keys, which are deliberately not
        # a protocol encoding.  The full decision remains reproducible from
        # ``distributed_replay_hash``; consensus needs only status/reasons.
        return {
            "status": gate.get("status"),
            "reasons": list(gate.get("reasons", [])),
            "distributed_replay_hash": hash_canonical(
                # The evaluator's display payload contains tuple category keys;
                # the binding already commits to the complete canonical raw
                # aggregate and frozen calibration/policy inputs used to derive
                # it.  Hash that protocol-safe binding rather than UI-shaped
                # summaries.
                "forge/qualified-distributed-replay/v1",
                {
                    "eligibility_binding": binding,
                    "qualification": qualification,
                    "decision_status": gate.get("status"),
                },
            ),
            "execution_bundle_hash": hash_canonical(BUNDLE_SCHEMA, raw),
            "replayed": True,
        }

    def local_report(
        self,
        round_plan: RoundPlan,
        validator_hotkey: str,
        bundle: Mapping[str, Any],
        *,
        verify_public_copy,
    ) -> LocalEvaluationReport:
        return local_report(
            round_plan,
            validator_hotkey,
            bundle,
            independently_execute=lambda value: self.replay(value, round_plan),
            verify_public_copy=verify_public_copy,
        )


class DistributedMVPCertificateVerifier(MVPCertificateVerifier):
    """Certificate verifier that refuses callback-only promotion evidence."""

    def __init__(
        self,
        signature_verifier: Verifier,
        *,
        evidence: Mapping[str, Mapping[str, Any]],
        replayer: DistributedExecutionReplayer,
        verify_public_copy,
    ) -> None:
        # The base class still verifies the certificate signatures and all
        # report/bundle hash bindings.  Its executor is our deterministic
        # replayer, never a supplied result callback.
        super().__init__(
            signature_verifier,
            evidence=evidence,
            independently_execute=None,
            verify_public_copy=verify_public_copy,
        )
        self.replayer = replayer

    def verify(
        self, certificate: RoundCertificate, plan: RoundPlan, state: FinalizedCompetitionState
    ) -> None:
        try:
            return super().verify(certificate, plan, state)
        except DistributedBridgeError as error:
            raise CertificateError(str(error)) from error

    def replay_bundle(self, bundle: Mapping[str, Any], plan: RoundPlan) -> Mapping[str, Any]:
        """Ignore any embedded result and recompute the frozen global gate."""

        return self.replayer.replay(bundle, plan)


def execution_evidence_map(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the exact map consumed by ``MVPCertificateVerifier``."""

    return {hash_canonical("forge/mvp-execution-bundle/v1", bundle): dict(bundle)}
