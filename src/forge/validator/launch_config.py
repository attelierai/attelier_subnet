"""Frozen Wan pilot configuration and non-grindable round selection.

Configuration is an input to a campaign, not permission to rent compute or sign.
The first supported deployment is testnet; mainnet is deliberately unavailable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical


def require_hash(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256")
    value = value.removeprefix("0x")
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class LaunchConfig:
    network: str
    netuid: int
    genesis_hash: str
    contract_hash: str
    campaign_id: str
    admission_window_blocks: int = 300
    entropy_offset_blocks: int = 10
    max_active_candidates: int = 1
    campaign_limit_usd: float = 50.0
    closure_reserve_usd: float = 5.0
    policy_schema: str = "forge/launch-policy/v2"

    def __post_init__(self) -> None:
        from math import isfinite

        if self.network != "test":
            raise ValueError("this pilot is explicitly testnet only")
        if type(self.netuid) is not int or self.netuid <= 0:
            raise ValueError("a positive testnet netuid is required")
        for name in ("genesis_hash", "contract_hash"):
            object.__setattr__(self, name, require_hash(getattr(self, name), name))
        if (
            not isinstance(self.campaign_id, str)
            or not self.campaign_id
            or len(self.campaign_id) > 64
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.campaign_id)
        ):
            raise ValueError("campaign id must be a short lowercase identifier")
        if (
            type(self.admission_window_blocks) is not int
            or self.admission_window_blocks < 1
            or type(self.entropy_offset_blocks) is not int
            or self.entropy_offset_blocks != 10
        ):
            raise ValueError("pilot requires positive admission windows and E = C + 10")
        if type(self.max_active_candidates) is not int or self.max_active_candidates != 1:
            raise ValueError("pilot permits exactly one active candidate")
        for name in ("campaign_limit_usd", "closure_reserve_usd"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.closure_reserve_usd >= self.campaign_limit_usd:
            raise ValueError("closure reserve must leave capacity for work")
        if self.policy_schema != "forge/launch-policy/v2":
            raise ValueError("new campaigns require the versioned 32-prompt policy")

    @classmethod
    def read(cls, path: Path) -> LaunchConfig:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.pop("schema", None) != "forge/launch-config/v1":
            raise ValueError("invalid launch configuration schema")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "forge/launch-config/v1", **asdict(self)}

    def config_hash(self) -> str:
        return hash_canonical("forge/launch-config/v1", self.to_dict())

    def verify_contract(self, contract: ChallengeContract) -> None:
        if contract.contract_hash() != self.contract_hash:
            raise ValueError("contract differs from frozen launch configuration")
        if (
            contract.base_model_id != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            or (contract.width, contract.height, contract.frames, contract.fps)
            != (832, 480, 81, 16)
        ):
            raise ValueError("pilot supports only Wan2.1 T2V 1.3B at 832x480/81/16")


def closed_selection(
    config: LaunchConfig,
    *,
    close_block: int,
    close_hash: str,
    finalized_block: int,
    parent_state_hash: str,
    candidate_manifest_hash: str,
    candidate_hotkey: str,
    commitment_block: int,
) -> dict[str, Any]:
    """Persist this declaration before observing E; no operator round-id input."""
    if (
        type(close_block) is not int
        or close_block < 0
        or close_block % config.admission_window_blocks
        or type(finalized_block) is not int
        or not close_block <= finalized_block < close_block + config.entropy_offset_blocks
        or type(commitment_block) is not int
        or not (
            max(0, close_block - config.admission_window_blocks) < commitment_block <= close_block
        )
    ):
        raise ValueError("selection must close its admission window before future entropy")
    if not isinstance(candidate_hotkey, str) or not candidate_hotkey:
        raise ValueError("candidate hotkey is required")
    payload = {
        "schema": "forge/launch-closed-selection/v1",
        "config_hash": config.config_hash(),
        "network": config.network,
        "netuid": config.netuid,
        "genesis_hash": config.genesis_hash,
        "contract_hash": config.contract_hash,
        "close_block": close_block,
        "close_hash": require_hash(close_hash, "close hash"),
        "entropy_block": close_block + config.entropy_offset_blocks,
        "parent_state_hash": require_hash(parent_state_hash, "parent state hash"),
        "candidate_manifest_hash": require_hash(candidate_manifest_hash, "candidate manifest"),
        "candidate_hotkey": candidate_hotkey,
        "commitment_block": commitment_block,
    }
    # Compact enough for the existing 128-byte on-chain report commitment.
    identity = hash_canonical("forge/launch-round-identity/v1", payload)
    return {**payload, "round_id": identity[:32], "selection_hash": identity}


def verify_selection(
    config: LaunchConfig, selection: dict[str, Any], *, entropy_block: int, entropy_hash: str
) -> str:
    """Validate an immutable selection before deriving its public prompt panel."""
    expected = closed_selection(
        config,
        close_block=selection["close_block"],
        close_hash=selection["close_hash"],
        finalized_block=selection["close_block"],
        parent_state_hash=selection["parent_state_hash"],
        candidate_manifest_hash=selection["candidate_manifest_hash"],
        candidate_hotkey=selection["candidate_hotkey"],
        commitment_block=selection["commitment_block"],
    )
    if expected != selection or entropy_block != expected["entropy_block"]:
        raise ValueError("selection or predetermined entropy height changed")
    return require_hash(entropy_hash, "entropy hash")
