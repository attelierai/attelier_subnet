"""Pure winner-takes-all derivation from the certified crown history."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Crown:
    crown_id: str
    hotkey: str
    fingerprint: str
    chain_time: int
    baseline: bool = False


@dataclass(frozen=True)
class RewardRules:
    """Compatibility marker for callers deriving certified winner weights.

    Winner-takes-all intentionally has no tunable decay, expiry, or historical
    allocation policy. Crown history remains in the certified state for audit.
    """


@dataclass(frozen=True)
class HotkeyRewardPlan:
    weights: dict[str, float]
    unallocated_share: float
    crown_ids: tuple[str, ...]


@dataclass(frozen=True)
class UIDRewardPlan:
    weights: dict[int, float]
    unallocated_share: float


def derive_hotkey_weights(
    crown_history: tuple[Crown, ...] | list[Crown],
    rules: RewardRules,
    eligible_hotkeys: set[str] | frozenset[str],
    reference_chain_time: int,
) -> HotkeyRewardPlan:
    """Pay the current certified winner only.

    ``crown_history`` is append-only certified state, so its last entry is the
    current king's promotion. Older crowns are evidence only: they never regain
    weight if the current winner deregisters.
    """

    if not crown_history:
        return HotkeyRewardPlan({}, 1.0, ())
    current = crown_history[-1]
    if current.baseline or current.hotkey not in eligible_hotkeys:
        return HotkeyRewardPlan({}, 1.0, ())
    return HotkeyRewardPlan({current.hotkey: 1.0}, 0.0, (current.crown_id,))


def map_hotkey_weights_to_uids(
    plan: HotkeyRewardPlan, uid_snapshot: dict[int, str]
) -> UIDRewardPlan:
    if len(set(uid_snapshot.values())) != len(uid_snapshot):
        raise ValueError("a hotkey cannot occupy multiple UIDs in one snapshot")
    hotkey_to_uid = {hotkey: uid for uid, hotkey in uid_snapshot.items()}
    weights: dict[int, float] = {}
    unallocated = plan.unallocated_share
    for hotkey, share in plan.weights.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is None:
            unallocated += share
        else:
            weights[uid] = share
    return UIDRewardPlan(weights, unallocated)
