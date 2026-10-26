from collections import Counter
from dataclasses import replace

import pytest

from forge.evaluation.benchmark import GRAMMAR_V2, GRAMMAR_V3, generate_prompts
from forge.protocol.contract import ChallengeContract


def test_v2_round_panels_are_reproducible_disjoint_unique_and_stratified():
    contract = ChallengeContract(grammar_version=GRAMMAR_V2)
    panels = {
        stage: generate_prompts(contract, "round-42", stage, "0xfinalized")
        for stage in ("screen", "duel", "confirmation")
    }
    assert panels["duel"] == generate_prompts(contract, "round-42", "duel", "0xfinalized")

    all_ids = [case.prompt_id for panel in panels.values() for case in panel]
    assert len(all_ids) == len(set(all_ids))
    for stage, cases in panels.items():
        assert len(cases) == contract.stage_sizes()[stage][0]
        assert len({case.text for case in cases}) == len(cases)
        assert len({case.prompt_id for case in cases}) == len(cases)
        assert set(Counter(case.properties["subject_class"] for case in cases).values()) == {
            len(cases) // 4
        }
        for property_name in ("camera_mode", "style_family", "speed_band"):
            counts = Counter(case.properties[property_name] for case in cases)
            assert max(counts.values()) - min(counts.values()) <= 1
        applicable = [
            case.properties["count_group"]
            for case in cases
            if case.properties["subject_class"] != "scenery"
        ]
        counts = Counter(applicable)
        assert abs(counts["single"] - counts["multiple"]) <= 1
        for case in cases:
            assert case.properties["evaluation_slices"]
            assert case.text.endswith(".")
            assert "  " not in case.text
            if case.properties["camera_mode"] == "static":
                assert "with a static camera" in case.text
            else:
                assert "while the camera" in case.text
            if case.properties["subject_class"] == "scenery":
                assert case.properties["direction"] == "none"
                assert case.properties["count_group"] == "not_applicable"
                assert case.text.startswith(
                    (
                        "A waterfall",
                        "A bank of clouds",
                        "A field of tall grass",
                        "Ocean waves",
                        "Autumn leaves",
                        "A river",
                    )
                )
            if case.properties["subject"] == "mallard duck":
                assert case.properties["environment"] in {
                    "across a calm pond",
                    "across a quiet lake",
                    "along a slow river",
                }
            if case.properties["subject"] == "silver toy top":
                assert case.properties["direction"] == "none"
                assert "in place" in case.text
                assert case.properties["environment"] in {
                    "on a polished wooden tabletop",
                    "on a clean studio floor",
                }


def test_v2_rejects_unknown_grammar_version():
    contract = replace(ChallengeContract(), grammar_version="forge/bench/grammar/future")
    with pytest.raises(ValueError, match="unsupported benchmark grammar"):
        generate_prompts(contract, "round-42", "screen", "0xfinalized")


def test_v3_confirmation_has_4096_unique_balanced_semantic_prompts():
    contract = ChallengeContract(
        grammar_version=GRAMMAR_V3,
        confirmation_prompts=4_096,
        confirmation_seeds=2,
    )
    cases = generate_prompts(contract, "confirmation-4096", "confirmation", "0xentropy")

    assert len(cases) == 4_096
    assert len({case.text for case in cases}) == len(cases)
    assert len({case.prompt_id for case in cases}) == len(cases)
    assert all(len(case.video_seeds) == 2 and len(set(case.video_seeds)) == 2 for case in cases)
    for property_name in (
        "subject_class",
        "camera_mode",
        "style_family",
        "speed_band",
        "lighting",
        "framing",
    ):
        counts = Counter(case.properties[property_name] for case in cases)
        assert max(counts.values()) - min(counts.values()) <= 1
    for case in cases:
        assert case.properties["grammar_version"] == GRAMMAR_V3
        assert case.properties["lighting"] in case.text
        assert case.properties["framing"] in case.text
        assert "sunlit" not in case.text
        assert "stormy sky" not in case.text
