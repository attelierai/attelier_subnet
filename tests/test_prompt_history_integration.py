import pytest

from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.protocol.prompt_history import (
    PromptHistoryError,
    genesis_checkpoint,
    history_link,
    verify_prompt_history,
)
from forge.protocol.prompt_pack import (
    PACK_SOURCE,
    build_pack,
    pack_context,
    seal_pack,
    verified_plan_prompts,
)


def _entries(prefix: str) -> list[dict]:
    return [
        {
            "stage": "screen",
            "text": f"{prefix} {subject} completes a distinct visible action.",
            "properties": {
                "subject_class": subject,
                "camera_mode": "static",
                "style_family": "realistic",
            },
            "rubric": [f"The {subject} remains visible."],
            "source_group": f"{prefix}-{subject}",
        }
        for subject in ("rigid_object", "human", "animal", "scenery")
    ]


def _plan(
    contract: ChallengeContract, round_id: str, close: int, salt: str, entries: list[dict]
) -> dict:
    pack = build_pack(
        pack_context("test", 9, round_id, contract.contract_hash(), close), salt, entries
    )
    plan = {
        "schema": "forge/panel-plan/v1",
        "network": "test",
        "netuid": 9,
        "round_id": round_id,
        "close_block": close,
        "entropy_block": close + 1,
        "entropy": f"{close + 1:064x}",
        "stage": "screen",
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash(),
        "prompt_source": PACK_SOURCE,
        "prompt_pack": pack,
        "pack_commitment": {"hash": seal_pack(pack), "block": close, "hotkey": "validator"},
        "pack_reveal_block": close + 1,
    }
    plan["prompts"] = verified_plan_prompts(plan, contract)
    return plan


def test_history_segment_rejects_revealed_text_reuse_and_truncated_chain():
    contract = ChallengeContract(screen_prompts=4, screen_seeds=1)
    checkpoint = genesis_checkpoint("test", 9, certified_state_hash="a" * 64)
    prior = _plan(contract, "r10", 10, "1" * 64, _entries("first"))
    prior_link = history_link(prior, checkpoint)
    reused = _plan(contract, "r20", 20, "2" * 64, _entries("first"))
    reused.update(
        prompt_history={
            "schema": "forge/prompt-history-bundle/v1",
            "checkpoint_root": checkpoint["root"],
            "links": [prior_link],
        },
        prompt_history_root=prior_link["root"],
    )

    with pytest.raises(PromptHistoryError, match="reuses a revealed prompt identity"):
        verify_prompt_history(reused, checkpoint)

    source_group_reused_entries = _entries("second")
    for prior_entry, entry in zip(_entries("first"), source_group_reused_entries, strict=True):
        entry["source_group"] = prior_entry["source_group"]
    source_group_reused = _plan(
        contract, "r20-source-group", 20, "4" * 64, source_group_reused_entries
    )
    source_group_reused.update(
        prompt_history={
            "schema": "forge/prompt-history-bundle/v1",
            "checkpoint_root": checkpoint["root"],
            "links": [prior_link],
        },
        prompt_history_root=prior_link["root"],
    )
    with pytest.raises(PromptHistoryError, match="reuses a revealed prompt identity"):
        verify_prompt_history(source_group_reused, checkpoint)

    fresh = _plan(contract, "r20", 20, "3" * 64, _entries("second"))
    fresh.update(
        prompt_history={
            "schema": "forge/prompt-history-bundle/v1",
            "checkpoint_root": checkpoint["root"],
            "links": [prior_link],
        },
        prompt_history_root=prior_link["root"],
    )
    verified = verify_prompt_history(fresh, checkpoint, expected_certified_state_hash="a" * 64)
    assert verified["history_root"] == prior_link["root"]

    # A checkpoint for a later certified state cannot be silently rolled back
    # to the older root or supplied with a truncated link segment.
    later_checkpoint = {
        **checkpoint,
        "root": prior_link["root"],
        "identity_root": hash_canonical(
            "forge/prompt-history-identities/v1",
            {
                "network": "test",
                "netuid": 9,
                "text_ids": sorted(item["text_id"] for item in prior_link["entry_identities"]),
                "source_group_ids": sorted(
                    item["source_group_id"] for item in prior_link["entry_identities"]
                ),
            },
        ),
        "through_close_block": 10,
        "text_ids": sorted(item["text_id"] for item in prior_link["entry_identities"]),
        "source_group_ids": sorted(
            item["source_group_id"] for item in prior_link["entry_identities"]
        ),
    }
    fresh["prompt_history"]["checkpoint_root"] = checkpoint["root"]
    with pytest.raises(PromptHistoryError, match="does not start at the certified checkpoint"):
        verify_prompt_history(fresh, later_checkpoint)


def test_history_rejects_round_replay_and_pack_revealed_after_current_close():
    contract = ChallengeContract(screen_prompts=4, screen_seeds=1)
    checkpoint = genesis_checkpoint("test", 9, certified_state_hash="a" * 64)
    prior = _plan(contract, "r10", 10, "1" * 64, _entries("first"))
    link = history_link(prior, checkpoint)
    replayed_round = _plan(contract, "r10", 20, "2" * 64, _entries("second"))
    replayed_round.update(
        prompt_history={
            "schema": "forge/prompt-history-bundle/v1",
            "checkpoint_root": checkpoint["root"],
            "links": [link],
        },
        prompt_history_root=link["root"],
    )
    with pytest.raises(PromptHistoryError, match="round is duplicated"):
        verify_prompt_history(replayed_round, checkpoint)

    late = _plan(contract, "r11", 11, "3" * 64, _entries("late"))
    late["pack_reveal_block"] = 25
    late["prompts"] = verified_plan_prompts(late, contract)
    late_link = history_link(late, checkpoint)
    current = _plan(contract, "r20", 20, "4" * 64, _entries("third"))
    current.update(
        prompt_history={
            "schema": "forge/prompt-history-bundle/v1",
            "checkpoint_root": checkpoint["root"],
            "links": [late_link],
        },
        prompt_history_root=late_link["root"],
    )
    with pytest.raises(PromptHistoryError, match="not revealed before the current close"):
        verify_prompt_history(current, checkpoint)
