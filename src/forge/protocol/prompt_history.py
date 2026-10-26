"""Auditable cross-round prompt reuse checks rooted in external certified state.

The supplied checkpoint is an input from an independent state/certificate system.
This module verifies its internal contents and a revealed-plan chain from that root;
it never treats a plan-local bundle or SQLite file as certification of completeness.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping

from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import canonical_json_bytes, hash_canonical
from forge.protocol.prompt_pack import PACK_SOURCE, build_pack, seal_pack, verified_plan_prompts

CHECKPOINT_SCHEMA = "forge/prompt-history-checkpoint/v1"
BUNDLE_SCHEMA = "forge/prompt-history-bundle/v1"
LINK_SCHEMA = "forge/prompt-history-link/v1"
TEXT_DOMAIN = "forge/prompt-history-text/v1"
SOURCE_GROUP_DOMAIN = "forge/prompt-history-source-group/v1"
IDENTITY_ROOT_DOMAIN = "forge/prompt-history-identities/v1"
LINK_DOMAIN = "forge/prompt-history-link/v1"
MAX_HISTORY_LINKS = 64
MAX_HISTORY_BUNDLE_BYTES = 20_000_000


class PromptHistoryError(ValueError):
    """The supplied history is malformed, inconsistent, or reuses an identity."""


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromptHistoryError(f"{label} must be a lowercase SHA-256")
    return value


def _block(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PromptHistoryError(f"{label} must be a nonnegative integer")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PromptHistoryError(f"{label} is required")
    return value


def normalize_identity(value: object, label: str) -> str:
    """Normalize the exact textual identity, not semantic similarity."""

    if not isinstance(value, str):
        raise PromptHistoryError(f"{label} must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    if not normalized:
        raise PromptHistoryError(f"{label} cannot be empty")
    return normalized


def entry_identity(entry: Mapping[str, object]) -> dict[str, str]:
    text = normalize_identity(entry.get("text"), "prompt text")
    source_group = normalize_identity(entry.get("source_group"), "source_group")
    return {
        "text_id": hash_canonical(TEXT_DOMAIN, text),
        "source_group_id": hash_canonical(SOURCE_GROUP_DOMAIN, source_group),
    }


def _identity_ids(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise PromptHistoryError(f"{label} must be a list")
    ids = [_sha256(item, label) for item in value]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise PromptHistoryError(f"{label} must be sorted and unique")
    return ids


def identity_root(network: str, netuid: int, text_ids: set[str], group_ids: set[str]) -> str:
    return hash_canonical(
        IDENTITY_ROOT_DOMAIN,
        {
            "network": _text(network, "network"),
            "netuid": _block(netuid, "netuid"),
            "text_ids": sorted(text_ids),
            "source_group_ids": sorted(group_ids),
        },
    )


def genesis_checkpoint(
    network: str, netuid: int, *, certified_state_hash: str
) -> dict[str, object]:
    """Create an empty checkpoint fixture; external systems must certify it."""

    network, netuid = _text(network, "network"), _block(netuid, "netuid")
    identities = identity_root(network, netuid, set(), set())
    return {
        "schema": CHECKPOINT_SCHEMA,
        "network": network,
        "netuid": netuid,
        "root": hash_canonical(
            LINK_DOMAIN, {"network": network, "netuid": netuid, "identities": identities}
        ),
        "identity_root": identities,
        "through_close_block": 0,
        "certified_state_hash": _sha256(certified_state_hash, "checkpoint state hash"),
        "text_ids": [],
        "source_group_ids": [],
    }


def _checkpoint(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema",
        "network",
        "netuid",
        "root",
        "identity_root",
        "through_close_block",
        "certified_state_hash",
        "text_ids",
        "source_group_ids",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PromptHistoryError("prompt history checkpoint fields are invalid")
    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise PromptHistoryError("prompt history checkpoint schema is invalid")
    network, netuid = (
        _text(value.get("network"), "checkpoint network"),
        _block(value.get("netuid"), "checkpoint netuid"),
    )
    texts = _identity_ids(value.get("text_ids"), "checkpoint text identities")
    groups = _identity_ids(value.get("source_group_ids"), "checkpoint source_group identities")
    expected_identity_root = identity_root(network, netuid, set(texts), set(groups))
    if value.get("identity_root") != expected_identity_root:
        raise PromptHistoryError("prompt history checkpoint identity root is invalid")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "network": network,
        "netuid": netuid,
        "root": _sha256(value.get("root"), "checkpoint root"),
        "identity_root": expected_identity_root,
        "through_close_block": _block(value.get("through_close_block"), "checkpoint close block"),
        "certified_state_hash": _sha256(value.get("certified_state_hash"), "checkpoint state hash"),
        "text_ids": texts,
        "source_group_ids": groups,
    }


def _plan_entries(plan: Mapping[str, object]) -> tuple[str, int, int, str, list[dict[str, str]]]:
    if plan.get("prompt_source") != PACK_SOURCE:
        raise PromptHistoryError("history plans must use sealed prompt packs")
    try:
        contract = ChallengeContract.from_dict(plan["contract"])
        if plan.get("contract_hash") != contract.contract_hash():
            raise PromptHistoryError("history plan contract hash differs")
        if plan.get("prompts") != verified_plan_prompts(plan, contract):
            raise PromptHistoryError("history plan prompts differ from its sealed pack")
        raw_pack = plan.get("prompt_pack")
        if not isinstance(raw_pack, Mapping):
            raise PromptHistoryError("history plan prompt pack is invalid")
        pack = build_pack(
            raw_pack.get("context", {}), raw_pack.get("salt"), raw_pack.get("entries")
        )
        pack_hash = seal_pack(pack)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, PromptHistoryError):
            raise
        raise PromptHistoryError("history plan sealed pack is invalid") from error
    return (
        _text(plan.get("network"), "history plan network"),
        _block(plan.get("netuid"), "history plan netuid"),
        _block(plan.get("close_block"), "history plan close block"),
        pack_hash,
        [entry_identity(entry) for entry in pack["entries"]],
    )


def _link_payload(
    plan: Mapping[str, object],
    previous_root: str,
    identity_root_value: str,
    identities: list[dict[str, str]],
) -> dict[str, object]:
    network, netuid, close, pack_hash, _ = _plan_entries(plan)
    return {
        "schema": LINK_SCHEMA,
        "network": network,
        "netuid": netuid,
        "round_id": _text(plan.get("round_id"), "history plan round_id"),
        "close_block": close,
        "plan_hash": hash_canonical("forge/evaluation-plan/v1", plan),
        "pack_hash": pack_hash,
        "previous_root": _sha256(previous_root, "previous history root"),
        "identity_root": _sha256(identity_root_value, "history identity root"),
        "entry_identities": identities,
    }


def history_link(plan: Mapping[str, object], checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Append a revealed plan to a checkpoint/state returned by this module."""

    checkpoint = _checkpoint(checkpoint)
    network, netuid, close, _, identities = _plan_entries(plan)
    if (network, netuid) != (checkpoint["network"], checkpoint["netuid"]):
        raise PromptHistoryError("history plan belongs to another subnet")
    if close <= checkpoint["through_close_block"]:
        raise PromptHistoryError("history plan does not follow its checkpoint")
    texts = set(checkpoint["text_ids"])
    groups = set(checkpoint["source_group_ids"])
    current_texts = {item["text_id"] for item in identities}
    current_groups = {item["source_group_id"] for item in identities}
    if texts & current_texts or groups & current_groups:
        raise PromptHistoryError("history plan reuses a revealed prompt identity")
    next_identity_root = identity_root(
        network, netuid, texts | current_texts, groups | current_groups
    )
    payload = _link_payload(plan, checkpoint["root"], next_identity_root, identities)
    return {**payload, "root": hash_canonical(LINK_DOMAIN, payload), "plan": dict(plan)}


def _link(
    value: Mapping[str, object],
    previous_root: str,
    known_texts: set[str],
    known_groups: set[str],
    checkpoint: Mapping[str, object],
) -> tuple[dict[str, object], set[str], set[str]]:
    expected = {
        "schema",
        "network",
        "netuid",
        "round_id",
        "close_block",
        "plan_hash",
        "pack_hash",
        "previous_root",
        "identity_root",
        "entry_identities",
        "root",
        "plan",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PromptHistoryError("prompt history link fields are invalid")
    if value.get("schema") != LINK_SCHEMA or value.get("previous_root") != previous_root:
        raise PromptHistoryError("prompt history chain is discontinuous")
    plan = value.get("plan")
    if not isinstance(plan, Mapping):
        raise PromptHistoryError("prompt history link plan is invalid")
    network, netuid, close, _, identities = _plan_entries(plan)
    if (network, netuid) != (checkpoint["network"], checkpoint["netuid"]):
        raise PromptHistoryError("history plan network differs from checkpoint")
    if close <= checkpoint["through_close_block"]:
        raise PromptHistoryError("history link predates its checkpoint")
    texts = {item["text_id"] for item in identities}
    groups = {item["source_group_id"] for item in identities}
    if (
        len(texts) != len(identities)
        or len(groups) != len(identities)
        or texts & known_texts
        or groups & known_groups
    ):
        raise PromptHistoryError("prompt history reuses a prior prompt identity")
    next_identity_root = identity_root(network, netuid, known_texts | texts, known_groups | groups)
    payload = _link_payload(plan, previous_root, next_identity_root, identities)
    if any(value.get(key) != item for key, item in payload.items()):
        raise PromptHistoryError("prompt history link does not match its revealed plan")
    if value.get("root") != hash_canonical(LINK_DOMAIN, payload):
        raise PromptHistoryError("prompt history link root is invalid")
    return dict(payload, root=value["root"], plan=dict(plan)), texts, groups


def verify_prompt_history(
    plan: Mapping[str, object],
    checkpoint: Mapping[str, object] | None,
    *,
    expected_certified_state_hash: str | None = None,
) -> dict[str, object]:
    """Verify current pack against an externally rooted history segment.

    A caller must authenticate checkpoint completeness independently. Missing
    checkpoints fail closed, because plan-local history cannot establish omission.
    """

    if checkpoint is None:
        raise PromptHistoryError("independently certified prompt history checkpoint is required")
    checkpoint = _checkpoint(checkpoint)
    if expected_certified_state_hash is not None and checkpoint["certified_state_hash"] != _sha256(
        expected_certified_state_hash, "expected certified state hash"
    ):
        raise PromptHistoryError("prompt history checkpoint is not bound to the incumbent state")
    network, netuid, current_close, _, current_identities = _plan_entries(plan)
    if (network, netuid) != (checkpoint["network"], checkpoint["netuid"]):
        raise PromptHistoryError("prompt history checkpoint belongs to another subnet")
    if current_close <= checkpoint["through_close_block"]:
        raise PromptHistoryError("current plan does not follow the certified history checkpoint")
    bundle = plan.get("prompt_history")
    if not isinstance(bundle, Mapping) or set(bundle) != {"schema", "checkpoint_root", "links"}:
        raise PromptHistoryError("prompt history bundle is required")
    if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("checkpoint_root") != checkpoint["root"]:
        raise PromptHistoryError("prompt history bundle does not start at the certified checkpoint")
    links = bundle.get("links")
    if not isinstance(links, list) or len(links) > MAX_HISTORY_LINKS:
        raise PromptHistoryError("prompt history links are invalid or exceed the segment limit")
    if len(canonical_json_bytes(bundle)) > MAX_HISTORY_BUNDLE_BYTES:
        raise PromptHistoryError("prompt history bundle exceeds the segment size limit")
    root, last_close = checkpoint["root"], checkpoint["through_close_block"]
    current_round_id = _text(plan.get("round_id"), "current plan round_id")
    seen_rounds, texts, groups = (
        set(),
        set(checkpoint["text_ids"]),
        set(checkpoint["source_group_ids"]),
    )
    for raw_link in links:
        link, link_texts, link_groups = _link(raw_link, root, texts, groups, checkpoint)
        if link["close_block"] <= last_close or link["close_block"] >= current_close:
            raise PromptHistoryError(
                "prompt history closes are not strictly before the current plan"
            )
        reveal_block = _block(link["plan"].get("pack_reveal_block"), "history pack reveal block")
        if reveal_block >= current_close:
            raise PromptHistoryError("history pack was not revealed before the current close")
        if link["round_id"] == current_round_id or link["round_id"] in seen_rounds:
            raise PromptHistoryError("prompt history round is duplicated")
        seen_rounds.add(link["round_id"])
        texts.update(link_texts)
        groups.update(link_groups)
        root, last_close = link["root"], link["close_block"]
    if plan.get("prompt_history_root") != root:
        raise PromptHistoryError("prompt history root does not match its verified chain")
    current_texts = {item["text_id"] for item in current_identities}
    current_groups = {item["source_group_id"] for item in current_identities}
    if current_texts & texts or current_groups & groups:
        raise PromptHistoryError("current sealed pack reuses a revealed prompt identity")
    return {
        "checkpoint_root": checkpoint["root"],
        "checkpoint_state_hash": checkpoint["certified_state_hash"],
        "history_root": root,
        "history_identity_root": identity_root(network, netuid, texts, groups),
        "history_links": len(links),
    }
