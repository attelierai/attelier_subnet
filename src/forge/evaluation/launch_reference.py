"""Durable, replayable V2 bootstrap-reference evidence.

The quality anchor always remains the original Wan50 pipeline. A separately
confirmed baseline can only replace the bootstrap/current reference and never
names a reward owner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.evaluation.evaluator_qualification import replay_qualification_package
from forge.evaluation.launch_baseline import ParentLaunchBaselineStore, select
from forge.evaluation.launch_policy import evaluate_v2
from forge.models.artifact_manifest import ModelManifest
from forge.models.generation import GenerationRecipe, effective_generation_recipe
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.launch_service import write_immutable_json

SCHEMA = "forge/launch-qualified-bootstrap-reference/v1"


def _validate_wan50(manifest: ModelManifest, contract: ChallengeContract) -> str:
    if (
        manifest.role != "pipeline"
        or manifest.base_reference is not None
        or (manifest.reference.repo_id, manifest.reference.revision)
        != (contract.base_model_id, contract.base_model_revision)
    ):
        raise ValueError("Wan50 manifest differs from the contract's original pipeline")
    recipe = effective_generation_recipe(
        GenerationRecipe.from_dict(dict(manifest.generation_config)), is_original_reference=True
    )
    if recipe != {
        "steps": contract.reference_steps,
        "guidance_scale": 5.0,
        "scheduler": "original_unipc",
    }:
        raise ValueError("Wan50 manifest must use the original 50-step UniPC recipe")
    return manifest.fingerprint()


def _validate_imported(manifest: ModelManifest, contract: ChallengeContract) -> str:
    base = (contract.base_model_id, contract.base_model_revision)
    if manifest.role in {"full_transformer", "lora"}:
        if (
            manifest.base_reference is None
            or (
                manifest.base_reference.repo_id,
                manifest.base_reference.revision,
            )
            != base
        ):
            raise ValueError("imported candidate differs from the contract base")
    elif manifest.role == "pipeline":
        # A Wan44/Wan40 baseline reuses the exact original public pipeline files
        # but changes the execution recipe explicitly; omission could silently
        # recover original UniPC and is therefore not an imported baseline.
        if (
            manifest.base_reference is not None
            or (
                manifest.reference.repo_id,
                manifest.reference.revision,
            )
            != base
            or "scheduler" not in manifest.generation_config
        ):
            raise ValueError("imported pipeline baseline must explicitly bind its changed recipe")
        recipe = effective_generation_recipe(
            GenerationRecipe.from_dict(dict(manifest.generation_config)),
            is_original_reference=False,
        )
        if recipe == {
            "steps": contract.reference_steps,
            "guidance_scale": 5.0,
            "scheduler": "original_unipc",
        }:
            raise ValueError("imported pipeline baseline duplicates original Wan50")
    else:
        raise ValueError("imported baseline has an unsupported manifest role")
    return manifest.fingerprint()


def _confirmation(
    evidence: Mapping[str, Any], *, anchor_hash: str, imported_hash: str
) -> tuple[dict[str, Any] | None, str]:
    selection = evidence["selection"]
    screenings = evidence["screenings"]
    recomputed = select(screenings, anchor_hash=anchor_hash)
    if selection != recomputed or selection["anchor_hash"] != anchor_hash:
        raise ValueError("baseline selection does not replay from the original Wan50 screenings")
    if selection["reward_hotkey"] is not None:
        raise ValueError("baseline selection may not assign a reward hotkey")
    if selection["status"] == "KEEP_WAN50":
        return None, anchor_hash

    plan, rows, package, stored = (
        evidence["confirmation_plan"],
        evidence["confirmation_rows"],
        evidence["qualification_package"],
        evidence["result"],
    )
    if any(value is None for value in (plan, rows, package, stored)):
        raise ValueError("selected baseline remains pending its one confirmation attempt")
    if not all(isinstance(value, Mapping) for value in (plan, package, stored)):
        raise ValueError("durable baseline confirmation is invalid")
    models = plan.get("models")
    if not isinstance(models, Mapping) or dict(models) != {
        "candidate": selection["selected_candidate_hash"],
        "king": anchor_hash,
        "anchor": anchor_hash,
    }:
        raise ValueError("confirmation must compare the selected candidate against original Wan50")
    if models["candidate"] != imported_hash:
        raise ValueError("imported manifest differs from the confirmed candidate")
    prompts = plan.get("prompts")
    if not isinstance(prompts, list) or set(selection["screening_prompts"]) & {
        prompt.get("prompt_id") for prompt in prompts if isinstance(prompt, Mapping)
    }:
        raise ValueError("confirmation panel must be disjoint from screening prompts")
    qualification = replay_qualification_package(package)
    if qualification != plan.get("evaluator_qualification"):
        raise ValueError("confirmation plan differs from replayed qualification")
    evaluation = evaluate_v2(dict(plan), rows)
    if (
        stored.get("schema") != "forge/parent-launch-baseline-confirmation/v1"
        or stored.get("selection_id") != selection["selection_id"]
        or stored.get("screening_hash") != selection["screening_hash"]
        or stored.get("confirmation_plan_hash")
        != hash_canonical("forge/parent-launch-baseline-plan/v1", dict(plan))
        or stored.get("confirmation_rows_hash")
        != hash_canonical("forge/parent-launch-baseline-rows/v1", rows)
        or stored.get("evaluation") != evaluation
        or stored.get("reward_hotkey") is not None
    ):
        raise ValueError("stored baseline confirmation hashes or status do not replay")
    confirmation = {
        "plan": dict(plan),
        "rows": rows,
        "qualification": dict(package),
        "result": dict(stored),
    }
    if evaluation["status"] != "PASS":
        if stored.get("status") != "KEEP_WAN50" or stored.get("baseline_hash") != anchor_hash:
            raise ValueError("nonpassing confirmation must retain Wan50")
        return confirmation, anchor_hash
    if (
        stored.get("status") != "IMPORTED_BASELINE"
        or stored.get("baseline_hash") != imported_hash
        or selection["selected_candidate_hash"] != imported_hash
    ):
        raise ValueError("stored imported baseline result is inconsistent")
    return confirmation, imported_hash


def _body(
    evidence: Mapping[str, Any],
    *,
    wan50_manifest: ModelManifest,
    imported_manifest: ModelManifest,
    contract: ChallengeContract,
) -> dict[str, Any]:
    quality_anchor_hash = _validate_wan50(wan50_manifest, contract)
    imported_manifest_hash = _validate_imported(imported_manifest, contract)
    confirmation, bootstrap_reference_hash = _confirmation(
        evidence, anchor_hash=quality_anchor_hash, imported_hash=imported_manifest_hash
    )
    return {
        "schema": SCHEMA,
        # Receipts cross a filesystem boundary.  Keep the signed body in plain
        # JSON types so loading it cannot turn contract tuples into lists and
        # make an otherwise valid self-contained receipt fail replay.
        "contract": json.loads(json.dumps(contract.to_dict())),
        "contract_hash": contract.contract_hash(),
        "wan50_manifest": wan50_manifest.to_dict(),
        "imported_manifest": imported_manifest.to_dict(),
        "quality_anchor_hash": quality_anchor_hash,
        "bootstrap_reference_hash": bootstrap_reference_hash,
        "imported_manifest_hash": imported_manifest_hash,
        "screenings": evidence["screenings"],
        "selection": evidence["selection"],
        "confirmation": confirmation,
        "reward_hotkey": None,
    }


def qualified_bootstrap_reference(
    store: ParentLaunchBaselineStore,
    *,
    wan50_manifest: ModelManifest,
    imported_manifest: ModelManifest,
    contract: ChallengeContract,
) -> dict[str, Any]:
    body = _body(
        store.evidence(),
        wan50_manifest=wan50_manifest,
        imported_manifest=imported_manifest,
        contract=contract,
    )
    return {**body, "receipt_sha256": hash_canonical(SCHEMA, body)}


def replay_bootstrap_reference(
    receipt: Mapping[str, Any], *, contract_hash: str, genesis_reference_hash: str
) -> str:
    """Recreate a receipt without trusting its stated target or a current king."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA:
        raise ValueError("bootstrap receipt schema is invalid")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hash_canonical(SCHEMA, body):
        raise ValueError("bootstrap receipt hash differs")
    if receipt.get("reward_hotkey") is not None or receipt.get("contract_hash") != contract_hash:
        raise ValueError("bootstrap receipt authority binding differs")
    try:
        contract = ChallengeContract.from_dict(dict(receipt["contract"]))
        wan50 = ModelManifest.from_dict(receipt["wan50_manifest"])
        imported = ModelManifest.from_dict(receipt["imported_manifest"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("bootstrap receipt manifests or contract are invalid") from error
    if contract.contract_hash() != contract_hash:
        raise ValueError("bootstrap receipt contract does not replay")
    evidence = {
        "screenings": receipt.get("screenings"),
        "selection": receipt.get("selection"),
        "confirmation_plan": (receipt.get("confirmation") or {}).get("plan"),
        "confirmation_rows": (receipt.get("confirmation") or {}).get("rows"),
        "qualification_package": (receipt.get("confirmation") or {}).get("qualification"),
        "result": (receipt.get("confirmation") or {}).get("result"),
    }
    rebuilt = _body(evidence, wan50_manifest=wan50, imported_manifest=imported, contract=contract)
    if rebuilt != body or rebuilt["bootstrap_reference_hash"] != genesis_reference_hash:
        raise ValueError(
            "bootstrap receipt does not reproduce the immutable genesis baseline reference"
        )
    return str(rebuilt["quality_anchor_hash"])


def export_qualified_bootstrap_reference(
    destination: Path,
    store: ParentLaunchBaselineStore,
    *,
    wan50_manifest: ModelManifest,
    imported_manifest: ModelManifest,
    contract: ChallengeContract,
) -> dict[str, Any]:
    receipt = qualified_bootstrap_reference(
        store,
        wan50_manifest=wan50_manifest,
        imported_manifest=imported_manifest,
        contract=contract,
    )
    write_immutable_json(destination, receipt)
    return receipt
