"""Offline, restart-safe initialization of a fresh Launch V2 campaign."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.evaluation.evaluator_qualification import replay_qualification_package
from forge.evaluation.launch_reference import replay_bootstrap_reference
from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.validator.launch_config import LaunchConfig
from forge.validator.launch_service import LaunchServiceStore, write_immutable_json


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _copy_immutable(source: Path, destination: Path) -> None:
    """Copy reviewed input bytes once, or prove an interrupted retry is identical."""
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"immutable campaign evidence changed: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(destination.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_unprogressed(
    root: Path, state: FinalizedCompetitionState, service: LaunchServiceStore
) -> None:
    if (
        state.state_version != 0
        or state.last_round_id is not None
        or state.crown_history
    ):
        raise ValueError("campaign state has already progressed beyond genesis")
    if service.snapshot()["rounds"]:
        raise ValueError("campaign has an existing round and cannot be initialized again")
    if any(path.is_dir() for path in root.glob("round-*")):
        raise ValueError("campaign has round evidence and cannot be initialized again")


def initialize(
    *,
    config: LaunchConfig,
    contract: ChallengeContract,
    root: Path,
    bootstrap_reference: Path,
    qualification_package: Path,
    activation_block: int,
) -> dict[str, Any]:
    """Materialize only the immutable offline genesis inputs for a new campaign.

    This command deliberately does not create worker-ready capacity evidence,
    contact a chain, render media, sign, or publish anything.
    """
    config.verify_contract(contract)
    if type(activation_block) is not int or activation_block <= 0:
        raise ValueError("activation block must be a positive finalized block")
    root = Path(root)
    bootstrap_reference = Path(bootstrap_reference)
    qualification_package = Path(qualification_package)
    if not bootstrap_reference.is_file() or not qualification_package.is_file():
        raise ValueError("bootstrap reference and qualification package must be reviewed files")

    bootstrap = _read_object(bootstrap_reference)
    qualification_package_value = _read_object(qualification_package)
    qualification = replay_qualification_package(qualification_package_value)
    quality_anchor = replay_bootstrap_reference(
        bootstrap,
        contract_hash=config.contract_hash,
        genesis_reference_hash=str(bootstrap.get("bootstrap_reference_hash", "")),
    )
    bootstrap_reference_hash = bootstrap.get("bootstrap_reference_hash")
    if not isinstance(bootstrap_reference_hash, str) or len(bootstrap_reference_hash) != 64:
        raise ValueError("bootstrap reference identity is invalid")
    if bootstrap.get("quality_anchor_hash") != quality_anchor:
        raise ValueError("bootstrap quality anchor differs from its replay")

    confirmation = bootstrap.get("confirmation")
    if confirmation is not None:
        if not isinstance(confirmation, Mapping) or not isinstance(
            confirmation.get("qualification"), Mapping
        ):
            raise ValueError("bootstrap confirmation qualification is invalid")
        confirmed = replay_qualification_package(confirmation["qualification"])
        if qualification != confirmed:
            raise ValueError("qualification receipt differs from baseline confirmation")

    state = FinalizedCompetitionState.genesis(
        config.campaign_id, bootstrap_reference_hash, activation_block
    )
    root.mkdir(parents=True, exist_ok=True)
    service = LaunchServiceStore(root / "budget.sqlite", config)
    existing_state = root / "state.json"
    if existing_state.exists():
        existing = FinalizedCompetitionState.from_dict(_read_object(existing_state))
        _assert_unprogressed(root, existing, service)
        if existing.to_dict() != state.to_dict():
            raise ValueError("genesis state differs from immutable initialization inputs")
    elif service.snapshot()["rounds"] or any(path.is_dir() for path in root.glob("round-*")):
        raise ValueError("campaign has progress without a genesis state")

    _copy_immutable(bootstrap_reference, root / "bootstrap-reference.json")
    _copy_immutable(qualification_package, root / "qualification-package.json")
    write_immutable_json(root / "state.json", state.to_dict())
    (root / "manifests").mkdir(exist_ok=True)
    write_immutable_json(
        root / "manifests" / "king.json", {"fingerprint": bootstrap_reference_hash}
    )
    write_immutable_json(root / "manifests" / "anchor.json", {"fingerprint": quality_anchor})
    return {
        "schema": "forge/launch-initialization/v1",
        "status": "OFFLINE_GENESIS_INITIALIZED",
        "network": config.network,
        "netuid": config.netuid,
        "config_hash": config.config_hash(),
        "contract_hash": config.contract_hash,
        "activation_block": activation_block,
        "state_hash": state.state_hash(),
        "king_fingerprint": bootstrap_reference_hash,
        "quality_anchor_fingerprint": quality_anchor,
        "qualification_receipt_id": qualification["receipt_id"],
        "authority": "NO_RENDER_NO_SIGNATURE_NO_PUBLICATION",
    }
