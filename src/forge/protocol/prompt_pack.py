"""Renewable, sealed prompt packs for candidate-independent round evaluation."""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Mapping

from forge.evaluation.benchmark import generate_prompts
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import canonical_json_bytes, hash_canonical

PACK_SCHEMA = "forge/sealed-prompt-pack/v1"
PACK_SOURCE = "forge/sealed-pack/v1"
LEGACY_SOURCE = "forge/procedural-grammar/v1"
STAGES = ("screen", "duel", "confirmation")
SUBJECT_CLASSES = ("rigid_object", "human", "animal", "scenery")
CAMERA_MODES = ("static", "moving")
STYLE_FAMILIES = ("realistic", "stylized")
MAX_PACK_BYTES = 10_000_000
MAX_ENTRIES = 20_000
MAX_TEXT_CHARS = 2_048
MAX_RUBRIC_ITEMS = 8
MAX_RUBRIC_CHARS = 512
MAX_PROPERTIES = 32
_SALT_RE = re.compile(r"[0-9a-f]{64}\Z")


def pack_context(
    network: str, netuid: int, round_id: str, contract_hash: str, close_block: int
) -> dict[str, str | int]:
    """Create the immutable round binding included in a sealed pack."""
    if not all(isinstance(item, str) and item for item in (network, round_id, contract_hash)):
        raise ValueError("pack context string identifiers are required")
    if type(netuid) is not int or netuid < 0 or type(close_block) is not int or close_block < 0:
        raise ValueError("pack netuid and close block must be non-negative integers")
    return {
        "network": network,
        "netuid": netuid,
        "round_id": round_id,
        "contract_hash": contract_hash,
        "close_block": close_block,
    }


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("prompt text must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    if not normalized:
        raise ValueError("prompt text cannot be empty")
    return normalized


def _entry(entry: object) -> dict[str, object]:
    if not isinstance(entry, Mapping) or set(entry) != {
        "stage",
        "text",
        "properties",
        "rubric",
        "source_group",
    }:
        raise ValueError("pack entries must be objects")
    stage = entry.get("stage")
    if stage not in STAGES:
        raise ValueError("pack entry has an unknown stage")
    text = entry.get("text")
    if len(_normalized_text(text)) > MAX_TEXT_CHARS:
        raise ValueError("pack entry text exceeds the character limit")
    properties = entry.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("pack entry properties must be an object")
    if len(properties) > MAX_PROPERTIES:
        raise ValueError("pack entry properties exceed the limit")
    properties = dict(properties)
    if properties.get("subject_class") not in SUBJECT_CLASSES:
        raise ValueError("pack entry subject_class is invalid")
    if properties.get("camera_mode") not in CAMERA_MODES:
        raise ValueError("pack entry camera_mode is invalid")
    if properties.get("style_family") not in STYLE_FAMILIES:
        raise ValueError("pack entry style_family is invalid")
    if any(
        not isinstance(key, str) or type(value) not in {str, int}
        for key, value in properties.items()
    ):
        raise ValueError("pack properties must use string keys and string or integer values")
    rubric = entry.get("rubric")
    if (
        not isinstance(rubric, list)
        or not rubric
        or len(rubric) > MAX_RUBRIC_ITEMS
        or any(
            not isinstance(item, str) or not item or len(item) > MAX_RUBRIC_CHARS for item in rubric
        )
    ):
        raise ValueError("pack entry rubric must be a non-empty list of strings")
    source_group = entry.get("source_group")
    if not isinstance(source_group, str) or not _normalized_text(source_group):
        raise ValueError("pack entry source_group is required")
    return {
        "stage": stage,
        "text": text,
        "properties": properties,
        "rubric": list(rubric),
        "source_group": source_group,
    }


def build_pack(
    context: Mapping[str, object], salt: str, entries: list[Mapping[str, object]]
) -> dict[str, object]:
    """Validate and construct a private pack before publishing only its commitment."""
    expected = pack_context(
        context.get("network"),
        context.get("netuid"),
        context.get("round_id"),
        context.get("contract_hash"),
        context.get("close_block"),
    )
    if set(context) != set(expected) or dict(context) != expected:
        raise ValueError("pack context has unknown or inconsistent fields")
    if not isinstance(salt, str) or not _SALT_RE.fullmatch(salt):
        raise ValueError("pack salt must be exactly 64 lowercase hexadecimal characters")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_ENTRIES:
        raise ValueError("pack requires entries")
    normalized_entries = [_entry(item) for item in entries]
    texts = [_normalized_text(item["text"]) for item in normalized_entries]
    groups = [_normalized_text(item["source_group"]) for item in normalized_entries]
    if len(texts) != len(set(texts)):
        raise ValueError("pack cannot contain duplicate normalized prompt text")
    if len(groups) != len(set(groups)):
        raise ValueError("each pack entry needs an independent source_group")
    pack: dict[str, object] = {
        "schema": PACK_SCHEMA,
        "context": expected,
        "salt": salt,
        "entries": normalized_entries,
    }
    if len(canonical_json_bytes(pack)) > MAX_PACK_BYTES:
        raise ValueError("pack exceeds the 10 MB limit")
    return pack


def seal_pack(pack: Mapping[str, object]) -> str:
    """Return the public commitment hash; never publish the input before close."""
    rebuilt = build_pack(pack.get("context", {}), pack.get("salt"), pack.get("entries"))
    if pack.get("schema") != PACK_SCHEMA or dict(pack) != rebuilt:
        raise ValueError("pack has unknown or inconsistent fields")
    return hash_canonical(PACK_SCHEMA, rebuilt)


def _plan_string(plan: Mapping[str, object], name: str, values: tuple[str, ...] = ()) -> str:
    value = plan.get(name)
    if not isinstance(value, str) or not value or values and value not in values:
        raise ValueError(f"plan {name} is required")
    return value


def _stage(plan: Mapping[str, object]) -> str:
    stage = _plan_string(plan, "stage")
    if stage not in STAGES:
        raise ValueError("plan stage is required")
    return stage


def _case_dict(
    entry: Mapping[str, object], pack_hash: str, root: str, index: int, seed_count: int
) -> dict[str, object]:
    properties = dict(entry["properties"] if isinstance(entry["properties"], Mapping) else {})
    properties.update({"prompt_source": PACK_SOURCE, "source_group": entry["source_group"]})
    seed_rng = random.Random(int(hash_canonical("forge/prompt-pack/seeds/v1", root, index), 16))
    return {
        "prompt_id": hash_canonical("forge/prompt-pack/prompt/v1", pack_hash, root, index, entry),
        "text": entry["text"],
        "properties": properties,
        "rubric": list(entry["rubric"]),
        "video_seeds": [seed_rng.getrandbits(63) for _ in range(seed_count)],
    }


def _select_pack_entries(
    entries: list[dict[str, object]], stage: str, count: int, entropy: str, pack_hash: str
) -> list[dict[str, object]]:
    root = hash_canonical("forge/prompt-pack/select/v1", entropy, stage, pack_hash)
    class_order = list(SUBJECT_CLASSES)
    random.Random(int(hash_canonical("forge/prompt-pack/classes/v1", root), 16)).shuffle(
        class_order
    )
    targets = {subject_class: count // len(SUBJECT_CLASSES) for subject_class in SUBJECT_CLASSES}
    for subject_class in class_order[: count % len(SUBJECT_CLASSES)]:
        targets[subject_class] += 1
    selected: list[dict[str, object]] = []
    for subject_class in SUBJECT_CLASSES:
        pool = [
            entry
            for entry in entries
            if isinstance(entry["properties"], Mapping)
            and entry["properties"].get("subject_class") == subject_class
        ]
        random.Random(
            int(hash_canonical("forge/prompt-pack/pool/v1", root, subject_class), 16)
        ).shuffle(pool)
        if len(pool) < targets[subject_class]:
            raise ValueError(f"sealed pack lacks {subject_class} entries for {stage}")
        selected.extend(pool[: targets[subject_class]])
    random.Random(int(hash_canonical("forge/prompt-pack/order/v1", root), 16)).shuffle(selected)
    return selected


def _legacy_prompts(
    plan: Mapping[str, object], contract: ChallengeContract
) -> list[dict[str, object]]:
    cases = generate_prompts(
        contract,
        _plan_string(plan, "round_id"),
        _stage(plan),
        _plan_string(plan, "entropy"),
    )
    return [
        {
            "prompt_id": case.prompt_id,
            "text": case.text,
            "properties": dict(case.properties),
            "rubric": list(case.rubric),
            "video_seeds": list(case.video_seeds),
        }
        for case in cases
    ]


def verified_plan_prompts(
    plan: Mapping[str, object], contract: ChallengeContract
) -> list[dict[str, object]]:
    """Resolve a replayable legacy plan or verify and select a sealed public pack."""
    source = plan.get("prompt_source")
    if source in (None, LEGACY_SOURCE):
        if plan.get("contract_hash") != contract.contract_hash():
            raise ValueError("plan contract hash does not match contract")
        return _legacy_prompts(plan, contract)
    if source != PACK_SOURCE:
        raise ValueError("plan prompt_source is unsupported")
    close = plan.get("close_block")
    context = pack_context(
        _plan_string(plan, "network"),
        plan.get("netuid") if type(plan.get("netuid")) is int else -1,
        _plan_string(plan, "round_id"),
        _plan_string(plan, "contract_hash"),
        close if type(close) is int else -1,
    )
    if context["contract_hash"] != contract.contract_hash():
        raise ValueError("plan contract hash does not match contract")
    pack = plan.get("prompt_pack")
    if not isinstance(pack, Mapping):
        raise ValueError("sealed plan requires a revealed prompt_pack")
    pack_hash = seal_pack(pack)
    if pack["context"] != context:
        raise ValueError("prompt pack context is not bound to the plan")
    commitment = plan.get("pack_commitment")
    if not isinstance(commitment, Mapping) or set(commitment) != {"hash", "block", "hotkey"}:
        raise ValueError("sealed plan requires an exact pack_commitment")
    if (
        commitment.get("hash") != pack_hash
        or not isinstance(commitment.get("hotkey"), str)
        or not commitment["hotkey"]
    ):
        raise ValueError("prompt pack commitment does not match its reveal")
    _, entropy_block, commit_block = (
        close,
        plan.get("entropy_block"),
        commitment.get("block"),
    )
    if any(type(item) is not int for item in (close, entropy_block, commit_block)) or not (
        0 <= commit_block <= close < entropy_block
    ):
        raise ValueError("prompt pack must commit by close before entropy")
    reveal_block = plan.get("pack_reveal_block")
    if type(reveal_block) is not int or reveal_block < entropy_block:
        raise ValueError("prompt pack reveal must be observed at or after entropy")
    stage = _stage(plan)
    if stage == "confirmation":
        lock = plan.get("finalist_lock_block")
        if type(lock) is not int or not close <= lock < entropy_block:
            raise ValueError("confirmation finalist must lock after close and before entropy")
    count, seed_count = contract.stage_sizes()[stage]
    entropy = _plan_string(plan, "entropy")
    root = hash_canonical("forge/prompt-pack/select/v1", entropy, stage, pack_hash)
    entries = [item for item in pack["entries"] if item["stage"] == stage]
    return [
        _case_dict(entry, pack_hash, root, index, seed_count)
        for index, entry in enumerate(
            _select_pack_entries(entries, stage, count, entropy, pack_hash)
        )
    ]


def verified_production_plan_prompts(
    plan: Mapping[str, object],
    contract: ChallengeContract,
    *,
    prompt_history_checkpoint: Mapping[str, object] | None,
    expected_certified_state_hash: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Resolve a sealed production panel and verify its externally rooted history.

    The checkpoint deliberately arrives as an argument rather than from ``plan``:
    callers must obtain it from independently certified state.
    """

    if plan.get("prompt_source") != PACK_SOURCE:
        raise ValueError("production plans must use a sealed prompt pack")
    prompts = verified_plan_prompts(plan, contract)
    from forge.protocol.prompt_history import verify_prompt_history

    history = verify_prompt_history(
        plan,
        prompt_history_checkpoint,
        expected_certified_state_hash=expected_certified_state_hash,
    )
    return prompts, history
