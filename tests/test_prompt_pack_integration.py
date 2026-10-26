from collections import Counter
from copy import deepcopy

import pytest

from forge.evaluation.benchmark import generate_prompts
from forge.protocol.contract import ChallengeContract
from forge.protocol.prompt_pack import (
    LEGACY_SOURCE,
    PACK_SOURCE,
    build_pack,
    pack_context,
    seal_pack,
    verified_plan_prompts,
)


def _contract() -> ChallengeContract:
    return ChallengeContract(screen_prompts=8, duel_prompts=8, confirmation_prompts=8)


def _entries() -> list[dict]:
    entries = []
    for stage in ("screen", "duel", "confirmation"):
        for subject_class in ("rigid_object", "human", "animal", "scenery"):
            for number in range(2):
                concept = f"{stage} {subject_class} independent concept {number}"
                entries.append(
                    {
                        "stage": stage,
                        "text": f"A {concept} moves through a distinct scene.",
                        "properties": {
                            "subject_class": subject_class,
                            "camera_mode": ("static", "moving")[number],
                            "style_family": ("realistic", "stylized")[number],
                        },
                        "rubric": ["The specified independent concept remains visible."],
                        "source_group": concept,
                    }
                )
    return entries


def _plan(pack: dict, contract: ChallengeContract, **changes: object) -> dict:
    commitment = {"hash": seal_pack(pack), "block": 100, "hotkey": "validator-hotkey"}
    plan = {
        "network": "test",
        "netuid": 9,
        "round_id": "r101",
        "contract_hash": contract.contract_hash(),
        "stage": "confirmation",
        "prompt_source": PACK_SOURCE,
        "prompt_pack": pack,
        "pack_commitment": commitment,
        "close_block": 100,
        "entropy_block": 101,
        "entropy": "0xfinalized",
        "pack_reveal_block": 101,
        "finalist_lock_block": 100,
    }
    plan.update(changes)
    return plan


def test_sealed_pack_binds_round_selects_balanced_prompts_and_replays_exactly():
    contract = _contract()
    context = pack_context("test", 9, "r101", contract.contract_hash(), 100)
    pack = build_pack(context, "a" * 64, _entries())
    plan = _plan(pack, contract)

    prompts = verified_plan_prompts(plan, contract)

    assert prompts == verified_plan_prompts(plan, contract)
    assert len(prompts) == 8
    assert len({prompt["text"] for prompt in prompts}) == 8
    assert len({prompt["prompt_id"] for prompt in prompts}) == 8
    assert all(
        isinstance(prompt["rubric"], list) and isinstance(prompt["video_seeds"], list)
        for prompt in prompts
    )
    assert all(len(prompt["video_seeds"]) == contract.confirmation_seeds for prompt in prompts)
    assert Counter(prompt["properties"]["subject_class"] for prompt in prompts) == {
        subject_class: 2 for subject_class in ("rigid_object", "human", "animal", "scenery")
    }


def test_sealed_pack_rejects_tampering_wrong_pack_late_commit_or_late_finalist_lock():
    contract = _contract()
    pack = build_pack(
        pack_context("test", 9, "r101", contract.contract_hash(), 100), "b" * 64, _entries()
    )
    plan = _plan(pack, contract)

    tampered = deepcopy(pack)
    tampered["entries"][0]["text"] = "A substituted concept moves through a distinct scene."
    with pytest.raises(ValueError, match="does not match"):
        verified_plan_prompts(
            _plan(tampered, contract, pack_commitment=plan["pack_commitment"]), contract
        )
    other_pack = build_pack(
        pack_context("test", 9, "r101", contract.contract_hash(), 100), "c" * 64, _entries()
    )
    with pytest.raises(ValueError, match="does not match"):
        verified_plan_prompts(
            _plan(other_pack, contract, pack_commitment=plan["pack_commitment"]), contract
        )
    with pytest.raises(ValueError, match="commit by close"):
        verified_plan_prompts(
            _plan(pack, contract, pack_commitment={**plan["pack_commitment"], "block": 102}),
            contract,
        )
    with pytest.raises(ValueError, match="context"):
        verified_plan_prompts(_plan(pack, contract, close_block=99), contract)
    with pytest.raises(ValueError, match="reveal"):
        verified_plan_prompts(_plan(pack, contract, pack_reveal_block=100), contract)
    with pytest.raises(ValueError, match="finalist"):
        verified_plan_prompts(_plan(pack, contract, finalist_lock_block=101), contract)


def test_legacy_replay_stays_available_but_a_finite_grammar_oracle_cannot_cover_a_novel_pack():
    contract = _contract()
    legacy_plan = {
        "network": "test",
        "netuid": 9,
        "round_id": "r101",
        "contract_hash": contract.contract_hash(),
        "stage": "screen",
        "prompt_source": LEGACY_SOURCE,
        "entropy": "0xfinalized",
    }
    legacy = verified_plan_prompts(legacy_plan, contract)
    assert legacy == [
        {
            "prompt_id": case.prompt_id,
            "text": case.text,
            "properties": dict(case.properties),
            "rubric": list(case.rubric),
            "video_seeds": list(case.video_seeds),
        }
        for case in generate_prompts(contract, "r101", "screen", "0xfinalized")
    ]

    novel_entries = []
    subjects = {
        "rigid_object": ("ceramic kite", "glass lantern", "woven basket", "brass compass"),
        "human": ("violinist", "potter", "gardener", "astronomer"),
        "animal": ("otter", "penguin", "gecko", "hummingbird"),
        "scenery": ("coral reef", "sand dune", "glacier", "canyon"),
    }
    for subject_class, names in subjects.items():
        for number, name in enumerate(names):
            novel_entries.append(
                {
                    "stage": "screen",
                    "text": f"A {name} appears in its own natural setting.",
                    "properties": {
                        "subject_class": subject_class,
                        "camera_mode": ("static", "moving")[number % 2],
                        "style_family": ("realistic", "stylized")[number % 2],
                    },
                    "rubric": [f"The {name} remains identifiable."],
                    "source_group": f"novel concept {name}",
                }
            )
    novel_contract = ChallengeContract(screen_prompts=16, duel_prompts=8, confirmation_prompts=8)
    pack = build_pack(
        pack_context("test", 9, "r101", novel_contract.contract_hash(), 100),
        "d" * 64,
        novel_entries,
    )
    novel = verified_plan_prompts(_plan(pack, novel_contract, stage="screen"), novel_contract)
    legacy_subject_vocabulary = (
        "red ceramic sphere",
        "blue wooden cube",
        "silver toy top",
        "yellow paper plane",
        "person in a green jacket",
        "dancer in a blue outfit",
        "runner in a red shirt",
        "runners in red shirt",
        "hiker with a yellow backpack",
        "hikers with yellow backpack",
        "skateboarder",
        "cyclist",
        "golden retriever",
        "black horse",
        "white rabbit",
        "orange butterfly",
        "mallard duck",
        "deer",
        "waterfall",
        "bank of clouds",
        "field of tall grass",
        "ocean waves",
        "autumn leaves",
        "river",
    )

    def finite_vocabulary_oracle(text):
        return float(any(subject in text.casefold() for subject in legacy_subject_vocabulary))

    public_v3 = generate_prompts(
        ChallengeContract(grammar_version="forge/bench/grammar/v3"), "r101", "screen", "0xfinalized"
    )
    assert all(finite_vocabulary_oracle(case.text) == 1.0 for case in public_v3)
    assert all(finite_vocabulary_oracle(prompt["text"]) == 0.0 for prompt in novel)
