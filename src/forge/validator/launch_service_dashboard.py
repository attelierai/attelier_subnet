"""Public, additive operational projection of the durable Launch service state.

Unlike ``launch_certified_dashboard``, this module does not certify a crown or
activate rewards.  It makes pending, expired and incomplete paths visible while
retaining the stricter certified exporter for a fully replayed transition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.validator.launch_config import LaunchConfig, require_hash
from forge.validator.launch_service import LaunchServiceStore, atomic_json

SCHEMA = "forge/launch-operations/v1"
_EVALUATION = {"PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE"}
_CONSENSUS = {
    "NO_CERTIFICATE",
    "PENDING",
    "AWAITING_CONSENSUS",
    "EXPIRED_NO_CERTIFICATE",
    "CERTIFIED",
    "CERTIFIED_NOT_YET_WEIGHTS",
}
_PUBLICATION = {"NOT_SUBMITTED", "PREPARED", "SUBMISSION_UNKNOWN", "SUBMITTED", "FINALIZED"}
_EFFECTIVE = {"NOT_VERIFIED", "PENDING", "FINALIZED", "MISMATCH"}


def _public_hash(value: object, name: str) -> str:
    return require_hash(value, name)


def _mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _bootstrap(value: Mapping[str, Any], config: LaunchConfig) -> dict[str, Any]:
    result = _mapping(value, "bootstrap reference")
    if result.get("schema") == "forge/launch-qualified-bootstrap-reference/v1":
        from forge.evaluation.launch_reference import replay_bootstrap_reference

        fingerprint = result.get("bootstrap_reference_hash")
        anchor = replay_bootstrap_reference(
            result, contract_hash=config.contract_hash, genesis_reference_hash=fingerprint
        )
        manifest = result["wan50_manifest" if fingerprint == anchor else "imported_manifest"]
        recipe = manifest["generation_config"]
        reference = manifest["reference"]
        public = {
            "fingerprint": fingerprint,
            "label": f"{reference['repo_id'].split('/')[-1]} · {recipe['steps']} steps",
        }
        if reference["backend"] == "hf":
            public["source"] = (
                f"https://huggingface.co/{reference['repo_id']}/tree/{reference['revision']}"
            )
        return public
    fingerprint = result.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("bootstrap reference fingerprint is required")
    public = {"fingerprint": fingerprint}
    if isinstance(result.get("label"), str):
        public["label"] = result["label"]
    if isinstance(result.get("source"), str):
        public["source"] = result["source"]
    if isinstance(result.get("manifest_hash"), str):
        public["manifest_hash"] = _public_hash(result["manifest_hash"], "bootstrap manifest")
    return public


def _evaluation(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    if not rounds:
        return {"status": "NO_EVALUATION", "round_id": None, "reason": None}
    current = rounds[-1]
    evidence = current.get("evaluation")
    if isinstance(evidence, Mapping):
        status = evidence.get("status")
        if status in _EVALUATION:
            return {
                "status": status,
                "round_id": current["round_id"],
                "reason": evidence.get("reason")
                if isinstance(evidence.get("reason"), str)
                else None,
            }
    if current["status"] in {"INCOMPLETE", "EXPIRED"}:
        return {
            "status": "INCOMPLETE",
            "round_id": current["round_id"],
            "reason": "round expired before complete evaluation"
            if current["status"] == "EXPIRED"
            else None,
        }
    return {"status": "PENDING", "round_id": current["round_id"], "reason": None}


def _certificate(value: Mapping[str, Any] | None) -> dict[str, Any]:
    status = _mapping(value, "consensus status").get("status", "NO_CERTIFICATE")
    if status in {
        "COMMITTED_WAITING_REVEAL", "REVEALED_WAITING_ACTIVATION",
        "COMMIT_RECONCILIATION_PENDING", "REVEAL_RECONCILIATION_PENDING",
        "INCOMPLETE_NO_VOTE", "INSUFFICIENT_COMMIT_WINDOW", "WAITING_ACTIVATION",
    }:
        status = "PENDING"
    if status not in _CONSENSUS:
        raise ValueError("unknown consensus status")
    result: dict[str, Any] = {"status": status, "certificate_hash": None}
    source = _mapping(value, "consensus status")
    if status in {"CERTIFIED", "CERTIFIED_NOT_YET_WEIGHTS"}:
        result["certificate_hash"] = _public_hash(
            source.get("certificate_hash"), "certificate hash"
        )
    if isinstance(source.get("finalized_block"), int) and source["finalized_block"] >= 0:
        result["finalized_block"] = source["finalized_block"]
    return result


def _publication(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(value, "publication")
    status = source.get("submission_status", "NOT_SUBMITTED")
    if status not in _PUBLICATION:
        raise ValueError("unknown publication status")
    result: dict[str, Any] = {
        "submission_status": status,
        "intent_hash": None,
        "submitted_extrinsic_hash": None,
        "effective_status": "NOT_VERIFIED",
        "finalized_block": None,
    }
    for name in ("intent_hash", "submitted_extrinsic_hash"):
        if source.get(name) is not None:
            result[name] = _public_hash(source[name], name)
    if source.get("effective_status") is not None:
        if source["effective_status"] not in _EFFECTIVE:
            raise ValueError("unknown effective weight status")
        result["effective_status"] = source["effective_status"]
    if isinstance(source.get("finalized_block"), int) and source["finalized_block"] >= 0:
        result["finalized_block"] = source["finalized_block"]
    return result


def _durable_consensus(rounds: list[Any]) -> dict[str, Any] | None:
    if not rounds:
        return None
    current = rounds[-1]
    if not isinstance(current, Mapping):
        return None
    certification = current.get("certification")
    if isinstance(certification, Mapping):
        return {"status": "CERTIFIED", "certificate_hash": certification.get("certificate_hash")}
    if current.get("status") == "EXPIRED":
        return {"status": "EXPIRED_NO_CERTIFICATE"}
    if current.get("status") == "AWAITING_CONSENSUS":
        return {"status": "AWAITING_CONSENSUS"}
    return None


def _durable_publication(rounds: list[Any]) -> dict[str, Any] | None:
    if not rounds or not isinstance(rounds[-1], Mapping):
        return None
    source = rounds[-1].get("publication")
    if not isinstance(source, Mapping):
        return None
    status = source.get("status")
    if status not in {"SUBMITTED", "FINALIZED"}:
        return None
    return {
        "submission_status": status,
        "intent_hash": source.get("intent_hash"),
        "submitted_extrinsic_hash": source.get("submitted_extrinsic_hash"),
        "effective_status": "FINALIZED" if status == "FINALIZED" else "PENDING",
        "finalized_block": source.get("finalized_block"),
    }


def project_launch_operations(
    service_snapshot: Mapping[str, Any],
    config: LaunchConfig,
    *,
    bootstrap_reference: Mapping[str, Any],
    consensus_status: Mapping[str, Any] | None = None,
    publication: Mapping[str, Any] | None = None,
    capacity: Mapping[str, Any] | None = None,
    worker_health: Mapping[str, Any] | None = None,
    observed_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Project persisted facts without treating missing operational evidence as success."""
    source = _mapping(service_snapshot, "service snapshot")
    if source.get("schema") != "forge/launch-service/v1":
        raise ValueError("service snapshot has the wrong schema")
    if (source.get("network"), source.get("netuid"), source.get("config_hash")) != (
        config.network,
        config.netuid,
        config.config_hash(),
    ):
        raise ValueError("service snapshot is not bound to the launch configuration")
    rounds = source.get("rounds")
    budget = source.get("budget")
    if not isinstance(rounds, list) or not isinstance(budget, Mapping):
        raise ValueError("service snapshot lacks durable rounds or budget")
    queue = []
    for round_ in rounds:
        if not isinstance(round_, Mapping) or not isinstance(round_.get("round_id"), str):
            raise ValueError("service snapshot has an invalid round")
        if round_.get("status") not in {"COMPLETE", "EXPIRED", "INCOMPLETE"}:
            queue.append({"round_id": round_["round_id"], "status": round_.get("status")})
    if observed_cost_usd is not None and (
        type(observed_cost_usd) not in (int, float)
        or not math.isfinite(observed_cost_usd)
        or observed_cost_usd < 0
    ):
        raise ValueError("observed cost must be nonnegative when recorded")
    return {
        "schema": SCHEMA,
        "observed_at": datetime.now(UTC).isoformat(),
        "identity": {
            "network": config.network,
            "netuid": config.netuid,
            "genesis_hash": config.genesis_hash,
            "contract_hash": config.contract_hash,
            "config_hash": config.config_hash(),
        },
        "bootstrap_reference": _bootstrap(bootstrap_reference, config),
        "king": None,
        "evaluation": _evaluation(rounds),
        "certificate": _certificate(consensus_status or _durable_consensus(rounds)),
        "publication": _publication(publication or _durable_publication(rounds)),
        "queue": {"active": queue, "active_count": len(queue)},
        "capacity": _mapping(capacity, "capacity") or {"status": "NOT_RECORDED"},
        "worker_health": _mapping(worker_health, "worker health") or {"status": "NOT_RECORDED"},
        "cost": {
            "observed_usd": observed_cost_usd,
            "estimated_exposure_usd": budget.get("reserved_or_captured_usd"),
            "campaign_limit_usd": budget.get("limit_usd"),
            "closure_reserve_usd": budget.get("closure_reserve_usd"),
        },
    }


def export_launch_operations(
    destination: Path,
    store: LaunchServiceStore,
    *,
    bootstrap_reference: Mapping[str, Any],
    consensus_status: Mapping[str, Any] | None = None,
    publication: Mapping[str, Any] | None = None,
    capacity: Mapping[str, Any] | None = None,
    worker_health: Mapping[str, Any] | None = None,
    observed_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Write a runtime status file atomically; callers choose a non-build destination."""
    result = project_launch_operations(
        store.snapshot(),
        store.config,
        bootstrap_reference=bootstrap_reference,
        consensus_status=consensus_status,
        publication=publication,
        capacity=capacity,
        worker_health=worker_health,
        observed_cost_usd=observed_cost_usd,
    )
    atomic_json(Path(destination), result)
    return result
