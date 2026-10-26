"""Frozen calibration-source validation and deterministic allocation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from forge.protocol.hashing import hash_canonical

SCHEMA_V2 = "forge/mvp-calibration-campaign/v2"
SCHEMA_V3 = "forge/mvp-calibration-campaign/v3"
SCHEMA_V4 = "forge/mvp-calibration-campaign/v4"
SOURCE_BANK_SCHEMA_V2 = "forge/mvp-calibration-source-bank/v1"
SOURCE_BANK_SCHEMA_V3 = "forge/mvp-calibration-source-bank/v2"
SPLITS = ("development", "calibration", "independent")
HELD_OUT_SPLITS = frozenset(("calibration", "independent"))

BASE_GROUP_BY_KIND = {
    "self": "repeat",
    "same_model_new_run": "repeat",
    "independent_seed": "independent_seed",
    "static_scene": "static",
    "freeze": "motion",
    "blur_mild": "static",
    "blur_severe": "static",
    "shuffle": "motion",
    "wrong_video": "motion",
    "detail_loss": "static",
    "evaluator_attack_mild": "static",
    "evaluator_attack_severe": "static",
}

SUBJECT_CLASSES = ("human", "animal", "rigid_object", "scenery")
CAMERA_TYPES = ("static", "moving")
STYLE_FAMILIES = ("realistic", "stylized")
CELL_FIELDS = ("subject_class", "camera_type", "style_family")
CELL_VOCABULARIES = (SUBJECT_CLASSES, CAMERA_TYPES, STYLE_FAMILIES)
V2_SOURCE_FIELD_ORDER = (
    "base_group",
    "source_id",
    "concept_id",
    "source_family",
    "concept_family",
    "prompt_text",
)
V3_SOURCE_FIELD_ORDER = (*V2_SOURCE_FIELD_ORDER, *CELL_FIELDS)
V2_SOURCE_FIELDS = frozenset(V2_SOURCE_FIELD_ORDER)
V3_SOURCE_FIELDS = frozenset(V3_SOURCE_FIELD_ORDER)
V4_SOURCE_FIELD_ORDER = (
    "source_mode",
    "source_id",
    "concept_id",
    "source_family",
    "concept_family",
    "prompt_text",
    *CELL_FIELDS,
)
V4_SOURCE_FIELDS = frozenset(V4_SOURCE_FIELD_ORDER)
SOURCE_MODES = ("static", "motion")


def _normalized_prompt(text: str) -> str:
    text = re.sub(r"\b(?:take|version|variation|sample)\s*\d+\b", "", text.casefold())
    return " ".join(text.split())


def _validate_source_bank(
    source_bank: Sequence[Mapping[str, Any]], *, schema: str | None = None
) -> tuple[list[dict[str, str]], str]:
    """Validate supplied source/concept metadata before split assignment."""

    if not isinstance(source_bank, Sequence) or isinstance(source_bank, (str, bytes)):
        raise ValueError("a frozen calibration source bank is required")
    if not source_bank:
        raise ValueError("a frozen calibration source bank is required")
    source_field_sets = {
        frozenset(source) if isinstance(source, Mapping) else frozenset() for source in source_bank
    }
    if len(source_field_sets) != 1:
        raise ValueError("calibration source bank versions cannot be mixed")
    fields = source_field_sets.pop()
    inferred_schema = (
        SCHEMA_V4
        if fields == V4_SOURCE_FIELDS
        else SCHEMA_V3
        if fields == V3_SOURCE_FIELDS
        else SCHEMA_V2
        if fields == V2_SOURCE_FIELDS
        else None
    )
    if inferred_schema is None or (schema is not None and schema != inferred_schema):
        raise ValueError("calibration source-bank fields are not exact")
    field_order = (
        V4_SOURCE_FIELD_ORDER
        if inferred_schema == SCHEMA_V4
        else V3_SOURCE_FIELD_ORDER
        if inferred_schema == SCHEMA_V3
        else V2_SOURCE_FIELD_ORDER
    )
    validated: list[dict[str, str]] = []
    seen = {
        field: set() for field in ("source_id", "concept_id", "source_family", "concept_family")
    }
    prompts: set[str] = set()
    for source in source_bank:
        if not isinstance(source, Mapping) or frozenset(source) != fields:
            raise ValueError("calibration source-bank fields are not exact")
        if (
            (
                inferred_schema != SCHEMA_V4
                and source.get("base_group") not in set(BASE_GROUP_BY_KIND.values())
            )
            or (inferred_schema == SCHEMA_V4 and source.get("source_mode") not in SOURCE_MODES)
            or any(
                not isinstance(source.get(field), str) or not source[field].strip()
                for field in fields - {"base_group"}
            )
        ):
            raise ValueError("calibration source-bank entry is invalid")
        if inferred_schema in {SCHEMA_V3, SCHEMA_V4} and any(
            source[field] not in vocabulary
            for field, vocabulary in zip(CELL_FIELDS, CELL_VOCABULARIES, strict=True)
        ):
            raise ValueError("calibration source-bank category is invalid")
        if (
            inferred_schema == SCHEMA_V4
            and source["source_mode"] == "static"
            and source["camera_type"] != "static"
        ):
            raise ValueError("a static calibration source cannot claim a moving camera")
        normalized = _normalized_prompt(source["prompt_text"])
        if normalized in prompts:
            raise ValueError("calibration source bank repeats a semantic prompt")
        prompts.add(normalized)
        for field in seen:
            if source[field] in seen[field]:
                raise ValueError("calibration source bank repeats an asserted family identity")
            seen[field].add(source[field])
        validated.append({field: str(source[field]) for field in field_order})
    return validated, inferred_schema


def _source_bank_domain(schema: str) -> str:
    return (
        "forge/mvp-calibration-source-bank/v4"
        if schema == SCHEMA_V4
        else SOURCE_BANK_SCHEMA_V3
        if schema == SCHEMA_V3
        else SOURCE_BANK_SCHEMA_V2
    )


def _cell(source: Mapping[str, str]) -> tuple[str, str, str]:
    return tuple(source[field] for field in CELL_FIELDS)  # type: ignore[return-value]


def allocate_sources(
    *,
    sources: Sequence[dict[str, str]],
    schema: str,
    entropy: str,
    required_by_split_group: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Select disjoint split sources, stratifying v3 held-outs by category."""

    available: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in sources:
        available[source["base_group"]].append(source)
    allocations: dict[tuple[str, str], list[dict[str, str]]] = {}
    for group in set(BASE_GROUP_BY_KIND.values()):
        ordered = sorted(
            available[group],
            key=lambda item: hash_canonical("forge/mvp-calibration-source-order/v1", entropy, item),
        )
        if schema == SCHEMA_V2:
            cursor = 0
            for split in SPLITS:
                count = required_by_split_group[(split, group)]
                selected = ordered[cursor : cursor + count]
                if len(selected) != count:
                    raise ValueError(
                        f"calibration source bank lacks {count} real {group} families for {split}"
                    )
                allocations[(split, group)] = selected
                cursor += count
            continue

        remaining = set(item["source_id"] for item in ordered)
        by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for item in ordered:
            by_cell[_cell(item)].append(item)
        for split in ("calibration", "independent"):
            selected: list[dict[str, str]] = []
            for subject in SUBJECT_CLASSES:
                for camera in CAMERA_TYPES:
                    for style in STYLE_FAMILIES:
                        cell = (subject, camera, style)
                        choices = [item for item in by_cell[cell] if item["source_id"] in remaining]
                        if len(choices) < 2:
                            raise ValueError(
                                "calibration source bank lacks two real "
                                f"{group} families for {split} {cell}"
                            )
                        selected.extend(choices[:2])
                        remaining.difference_update(item["source_id"] for item in choices[:2])
            allocations[(split, group)] = selected
        development_count = required_by_split_group[("development", group)]
        selected = [item for item in ordered if item["source_id"] in remaining][:development_count]
        if len(selected) != development_count:
            raise ValueError(
                "calibration source bank lacks "
                f"{development_count} real {group} families for development"
            )
        allocations[("development", group)] = selected
    return allocations
