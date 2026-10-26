"""Freeze and verify calibration work before any validator renders it.

Calibration is evidence about the evaluator, not an opportunity to tune it
after looking at a candidate.  This module describes the work independently
of GPU execution.  The runner later adds video hashes, VBench rows and blinded
Luna orders to the exact slots frozen here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from forge.evaluation.calibration_sources import (
    BASE_GROUP_BY_KIND,
    CAMERA_TYPES,
    CELL_FIELDS,
    CELL_VOCABULARIES,
    HELD_OUT_SPLITS,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V4,
    SPLITS,
    STYLE_FAMILIES,
    SUBJECT_CLASSES,
    _cell,
    _normalized_prompt,
    _source_bank_domain,
    _validate_source_bank,
    allocate_sources,
)
from forge.evaluation.mvp_policy import CONTROL_KINDS, LEGITIMATE_CONTROLS, _sha
from forge.protocol.hashing import hash_canonical

# Frozen transform recipe, duplicated here deliberately: the campaign is the
# execution contract and must not inherit mutable runner defaults.
V4_CONTROL_SPECS: dict[str, dict[str, object]] = {
    "self": {"operation": "identical_reference"},
    "same_model_new_run": {"operation": "new_generation_same_seed"},
    "independent_seed": {"operation": "new_generation_independent_seed"},
    "static_scene": {"operation": "static_scene_reference"},
    "freeze": {"frame_index": 0},
    "blur_mild": {"gaussian_radius_px": 1.0},
    "blur_severe": {"gaussian_radius_px": 4.0},
    "shuffle": {"permutation": "seeded_nonidentity"},
    "wrong_video": {"replacement": "different_frozen_prompt"},
    "detail_loss": {"downscale_factor": 4, "upscale": "nearest"},
    "evaluator_attack_mild": {"temporal_luma_offset": 3, "period_frames": 2},
    "evaluator_attack_severe": {"temporal_luma_offset": 16, "period_frames": 2},
}


def _count_for(
    split: str,
    kind: str,
    development_legitimate_samples: int,
    development_degradation_samples: int,
) -> int:
    if split != "development":
        return 32
    return (
        development_legitimate_samples
        if kind in LEGITIMATE_CONTROLS
        else development_degradation_samples
    )


def freeze_calibration_campaign(
    *,
    campaign_label: str,
    entropy: str,
    source_bank: Sequence[Mapping[str, Any]],
    development_legitimate_samples: int = 32,
    development_degradation_samples: int = 8,
    development_pilot: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Return a deterministic campaign with frozen prompts and controls.

    ``entropy`` must come from finalized chain state.  Candidates are committed
    before it is revealed; it is not a secret shared by a central coordinator.
    """

    if not isinstance(campaign_label, str) or not campaign_label:
        raise ValueError("calibration campaign label is required")
    if not isinstance(entropy, str) or len(entropy) < 16:
        raise ValueError("finalized entropy is required for calibration campaign")
    if (
        type(development_legitimate_samples) is not int
        or development_legitimate_samples < 32
        or type(development_degradation_samples) is not int
        or development_degradation_samples < 8
    ):
        raise ValueError("development sampling needs 32 legitimate and 8 degradation rows")
    sources, schema = _validate_source_bank(source_bank)
    required_by_split_group = {
        (split, group): max(
            _count_for(split, kind, development_legitimate_samples, development_degradation_samples)
            for kind, candidate_group in BASE_GROUP_BY_KIND.items()
            if candidate_group == group
        )
        for split in SPLITS
        for group in set(BASE_GROUP_BY_KIND.values())
    }
    if schema == SCHEMA_V4:
        return _freeze_v4_campaign(
            campaign_label=campaign_label,
            entropy=entropy,
            sources=sources,
            development_legitimate_samples=development_legitimate_samples,
            development_degradation_samples=development_degradation_samples,
            development_pilot=development_pilot,
        )
    allocations = allocate_sources(
        sources=sources,
        schema=schema,
        entropy=entropy,
        required_by_split_group=required_by_split_group,
    )
    slots: list[dict[str, object]] = []
    for split in SPLITS:
        for kind in CONTROL_KINDS:
            group = BASE_GROUP_BY_KIND[kind]
            for index in range(
                _count_for(
                    split,
                    kind,
                    development_legitimate_samples,
                    development_degradation_samples,
                )
            ):
                source = allocations[(split, group)][index]
                base_id = hash_canonical(
                    "forge/mvp-calibration-base/v1", campaign_label, split, group, source
                )
                text = source["prompt_text"]
                seed_hash = hash_canonical(
                    "forge/mvp-calibration-seed/v1", {"entropy": entropy, "base_id": base_id}
                )
                body = {
                    "split": split,
                    "kind": kind,
                    "index": index,
                    "base_id": base_id,
                    "source_id": source["source_id"],
                    "concept_id": source["concept_id"],
                    "source_family": source["source_family"],
                    "concept_family": source["concept_family"],
                    "prompt_text": text,
                    "prompt_hash": hash_canonical("forge/mvp-prompt/v1", text),
                    "seed": int(seed_hash[:15], 16),
                    "control_spec": f"forge/mvp-control/{kind}/v1",
                }
                if schema == SCHEMA_V3:
                    body.update({field: source[field] for field in CELL_FIELDS})
                slots.append({**body, "slot_id": _sha(body)})
    body = {
        "schema": schema,
        "campaign_label": campaign_label,
        "entropy": entropy,
        "development_legitimate_samples": development_legitimate_samples,
        "development_degradation_samples": development_degradation_samples,
        "source_bank": sources,
        "source_bank_hash": hash_canonical(_source_bank_domain(schema), sources),
        "slots": slots,
    }
    return {**body, "campaign_id": _sha(body)}


def _freeze_v4_campaign(
    *,
    campaign_label: str,
    entropy: str,
    sources: list[dict[str, str]],
    development_legitimate_samples: int,
    development_degradation_samples: int,
    development_pilot: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Freeze v4's reusable 64-family partition design.

    Every partition has 32 deliberately static and 32 requested-motion source
    families.  A family may support several *different* controls, but appears
    once inside each control.  This makes render reuse explicit while retaining
    32 independent observations for every control kind.
    """
    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in sources:
        by_mode[source["source_mode"]].append(source)
    pilot_sources: set[str] = set()
    pilot_bases: dict[str, tuple[str, int]] = {}
    pilot_id: str | None = None
    if development_pilot is not None:
        ledger = pilot_execution_ledger(development_pilot)
        pilot_id = str(ledger["pilot_id"])
        pilot_sources = {str(source["source_id"]) for source in development_pilot["source_bank"]}
        for slot in development_pilot["slots"]:
            previous = pilot_bases.setdefault(
                str(slot["source_id"]), (str(slot["base_id"]), int(slot["seed"]))
            )
            if previous != (slot["base_id"], slot["seed"]):
                raise ValueError("development pilot source has inconsistent render identity")
        if not pilot_sources <= {source["source_id"] for source in sources}:
            raise ValueError("development pilot sources must be present in the v4 source bank")
    selected: dict[tuple[str, str], list[dict[str, str]]] = {}
    for mode in ("static", "motion"):
        cells = (
            [(subject, "static", style) for subject in SUBJECT_CLASSES for style in STYLE_FAMILIES]
            if mode == "static"
            else [
                (subject, camera, style)
                for subject in SUBJECT_CLASSES
                for camera in CAMERA_TYPES
                for style in STYLE_FAMILIES
            ]
        )
        per_cell = 4 if mode == "static" else 2
        chosen_by_split = {split: [] for split in SPLITS}
        for cell in cells:
            ordered = sorted(
                [item for item in by_mode[mode] if _cell(item) == cell],
                key=lambda item: hash_canonical(
                    "forge/mvp-calibration-source-order/v1", entropy, item
                ),
            )
            required = per_cell * len(SPLITS)
            if len(ordered) < required:
                raise ValueError(
                    "calibration source bank lacks "
                    f"{required} real {mode} families for category {cell}"
                )
            pilot_in_cell = [source for source in ordered if source["source_id"] in pilot_sources]
            if len(pilot_in_cell) > per_cell:
                raise ValueError("development pilot overfills a v4 category cell")
            development = (
                pilot_in_cell
                + [source for source in ordered if source not in pilot_in_cell][
                    : per_cell - len(pilot_in_cell)
                ]
            )
            remaining = [source for source in ordered if source not in development]
            chosen_by_split["development"].extend(development)
            for number, split in enumerate(("calibration", "independent")):
                chosen_by_split[split].extend(
                    remaining[number * per_cell : (number + 1) * per_cell]
                )
        for split in SPLITS:
            selected[(split, mode)] = sorted(
                chosen_by_split[split],
                key=lambda item: (
                    0 if split == "development" and item["source_id"] in pilot_sources else 1,
                    item["source_id"]
                    if split == "development" and item["source_id"] in pilot_sources
                    else hash_canonical("forge/mvp-calibration-source-order/v1", entropy, item),
                ),
            )

    static_kinds = {
        "static_scene",
        "blur_mild",
        "blur_severe",
        "detail_loss",
        "evaluator_attack_mild",
        "evaluator_attack_severe",
    }
    motion_kinds = {"freeze", "shuffle", "wrong_video"}
    slots: list[dict[str, object]] = []
    for split in SPLITS:
        for kind in CONTROL_KINDS:
            count = _count_for(
                split, kind, development_legitimate_samples, development_degradation_samples
            )
            # Development is intentionally smaller for degradation recipes. It
            # still takes a deterministic prefix from the 64 frozen families.
            if kind in static_kinds:
                candidates = selected[(split, "static")]
            elif kind in motion_kinds:
                candidates = selected[(split, "motion")]
            elif kind == "self":
                # The identity control covers the requested-motion half; the
                # static-scene control covers the static half. Together they
                # materialize every one of the 64 partition references.
                candidates = selected[(split, "motion")]
            else:
                # New-run and new-seed controls retain both source types.
                candidates = selected[(split, "static")][:16] + selected[(split, "motion")][:16]
            for index, source in enumerate(candidates[:count]):
                base_id = hash_canonical(
                    "forge/mvp-calibration-base/v2", campaign_label, split, source
                )
                text = source["prompt_text"]
                seed_hash = hash_canonical(
                    "forge/mvp-calibration-seed/v1", {"entropy": entropy, "base_id": base_id}
                )
                if split == "development" and source["source_id"] in pilot_bases:
                    base_id, seed = pilot_bases[source["source_id"]]
                else:
                    seed = int(seed_hash[:15], 16)
                body: dict[str, object] = {
                    "split": split,
                    "kind": kind,
                    "index": index,
                    "base_id": base_id,
                    "source_id": source["source_id"],
                    "concept_id": source["concept_id"],
                    "source_family": source["source_family"],
                    "concept_family": source["concept_family"],
                    "prompt_text": text,
                    "prompt_hash": hash_canonical("forge/mvp-prompt/v1", text),
                    "seed": seed,
                    "control_spec": V4_CONTROL_SPECS[kind],
                    "source_mode": source["source_mode"],
                    **{field: source[field] for field in CELL_FIELDS},
                }
                if kind == "wrong_video":
                    # A wrong-video comparison needs two distinct lineages.
                    # Pair all 32 static targets to all 32 motion donors, so no
                    # lineage can be reused within this control.
                    donor = selected[(split, "motion")][index]
                    body.update(
                        {
                            "donor_source_id": donor["source_id"],
                            "donor_concept_id": donor["concept_id"],
                            "donor_source_family": donor["source_family"],
                            "donor_concept_family": donor["concept_family"],
                            "donor_prompt_hash": hash_canonical(
                                "forge/mvp-prompt/v1", donor["prompt_text"]
                            ),
                        }
                    )
                    # The target is deliberately static only for this control;
                    # the donor is a requested-motion lineage.  Do not label it
                    # as a moving-camera target merely to fill a category cell.
                    source = selected[(split, "static")][index]
                    base_id = hash_canonical(
                        "forge/mvp-calibration-base/v2", campaign_label, split, source
                    )
                    seed_hash = hash_canonical(
                        "forge/mvp-calibration-seed/v1", {"entropy": entropy, "base_id": base_id}
                    )
                    if split == "development" and source["source_id"] in pilot_bases:
                        base_id, seed = pilot_bases[source["source_id"]]
                    else:
                        seed = int(seed_hash[:15], 16)
                    body.update(
                        {
                            "base_id": base_id,
                            "seed": seed,
                            "source_id": source["source_id"],
                            "concept_id": source["concept_id"],
                            "source_family": source["source_family"],
                            "concept_family": source["concept_family"],
                            "prompt_text": source["prompt_text"],
                            "prompt_hash": hash_canonical(
                                "forge/mvp-prompt/v1", source["prompt_text"]
                            ),
                            "source_mode": source["source_mode"],
                            **{field: source[field] for field in CELL_FIELDS},
                        }
                    )
                slots.append({**body, "slot_id": _sha(body)})
    body = {
        "schema": SCHEMA_V4,
        "campaign_label": campaign_label,
        "entropy": entropy,
        "development_legitimate_samples": development_legitimate_samples,
        "development_degradation_samples": development_degradation_samples,
        "source_bank": sources,
        "source_bank_hash": hash_canonical(_source_bank_domain(SCHEMA_V4), sources),
        "development_pilot": dict(development_pilot) if development_pilot is not None else None,
        "development_pilot_id": pilot_id,
        "slots": slots,
    }
    return {**body, "campaign_id": _sha(body)}


def validate_calibration_campaign(campaign: Mapping[str, Any]) -> dict[str, object]:
    """Validate coverage and independence without looking at scored evidence."""

    if not isinstance(campaign, Mapping):
        raise ValueError("calibration campaign is required")
    body = {key: value for key, value in campaign.items() if key != "campaign_id"}
    schema = body.get("schema")
    required_body_keys = {
        "schema",
        "campaign_label",
        "entropy",
        "development_legitimate_samples",
        "development_degradation_samples",
        "source_bank",
        "source_bank_hash",
        "slots",
    }
    if schema == SCHEMA_V4:
        required_body_keys.update({"development_pilot", "development_pilot_id"})
    if (
        set(body) != required_body_keys
        or schema not in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4}
        or campaign.get("campaign_id") != _sha(body)
        or not isinstance(body["campaign_label"], str)
        or not body["campaign_label"]
        or not isinstance(body["entropy"], str)
        or len(body["entropy"]) < 16
        or type(body["development_legitimate_samples"]) is not int
        or body["development_legitimate_samples"] < 32
        or type(body["development_degradation_samples"]) is not int
        or body["development_degradation_samples"] < 8
        or not isinstance(body["source_bank_hash"], str)
        or len(body["source_bank_hash"]) != 64
        or not isinstance(body["slots"], list)
    ):
        raise ValueError("calibration campaign is not canonically frozen")
    try:
        source_bank, source_schema = _validate_source_bank(body["source_bank"], schema=schema)
    except ValueError as error:
        raise ValueError("calibration campaign source bank is invalid") from error
    if source_schema != schema or body["source_bank_hash"] != hash_canonical(
        _source_bank_domain(schema), source_bank
    ):
        raise ValueError("calibration campaign source bank hash is invalid")
    if schema == SCHEMA_V4:
        pilot = body["development_pilot"]
        if pilot is not None:
            try:
                pilot_ledger = pilot_execution_ledger(pilot)
            except (TypeError, ValueError):
                raise ValueError("calibration v4 development pilot is invalid") from None
            if body["development_pilot_id"] != pilot_ledger["pilot_id"]:
                raise ValueError("calibration v4 development pilot id is invalid")
        elif body["development_pilot_id"] is not None:
            raise ValueError("calibration v4 development pilot id is invalid")
        expected = _freeze_v4_campaign(
            campaign_label=body["campaign_label"],
            entropy=body["entropy"],
            sources=source_bank,
            development_legitimate_samples=body["development_legitimate_samples"],
            development_degradation_samples=body["development_degradation_samples"],
            development_pilot=pilot,
        )
        if dict(campaign) != expected:
            raise ValueError("calibration v4 campaign differs from its frozen allocation")
        base_ids = {slot["base_id"] for slot in body["slots"]}
        return {
            "campaign_id": campaign["campaign_id"],
            "slot_count": len(body["slots"]),
            "distinct_base_families": len(base_ids),
        }
    sources_by_identity = {
        (
            source["source_id"],
            source["concept_id"],
            source["source_family"],
            source["concept_family"],
            source["prompt_text"],
            *(tuple(source[field] for field in CELL_FIELDS) if schema == SCHEMA_V3 else ()),
        ): source
        for source in source_bank
    }
    slots = body["slots"]
    development_legitimate_samples = body["development_legitimate_samples"]
    development_degradation_samples = body["development_degradation_samples"]
    expected = sum(
        _count_for(
            split,
            kind,
            development_legitimate_samples,
            development_degradation_samples,
        )
        for split in SPLITS
        for kind in CONTROL_KINDS
    )
    if len(slots) != expected:
        raise ValueError("calibration campaign has wrong slot count")
    seen_slots: set[str] = set()
    seen_by_kind: dict[tuple[str, str], set[str]] = defaultdict(set)
    bases_by_split: dict[str, set[str]] = defaultdict(set)
    source_ids_by_split: dict[str, set[str]] = defaultdict(set)
    concept_ids_by_split: dict[str, set[str]] = defaultdict(set)
    source_families_by_split: dict[str, set[str]] = defaultdict(set)
    concept_families_by_split: dict[str, set[str]] = defaultdict(set)
    prompts_by_split: dict[str, set[str]] = defaultdict(set)
    expected_keys = {
        "split",
        "kind",
        "index",
        "base_id",
        "source_id",
        "concept_id",
        "source_family",
        "concept_family",
        "prompt_text",
        "prompt_hash",
        "seed",
        "control_spec",
        "slot_id",
    }
    if schema == SCHEMA_V3:
        expected_keys.update(CELL_FIELDS)
    heldout_cells_by_kind: dict[tuple[str, str], dict[tuple[str, str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for slot in slots:
        if not isinstance(slot, Mapping) or set(slot) != expected_keys:
            raise ValueError("calibration campaign slot fields are not exact")
        slot_body = {key: value for key, value in slot.items() if key != "slot_id"}
        split, kind = slot.get("split"), slot.get("kind")
        if (
            split not in SPLITS
            or kind not in CONTROL_KINDS
            or type(slot.get("index")) is not int
            or slot["index"] < 0
            or not isinstance(slot.get("base_id"), str)
            or not slot["base_id"]
            or any(
                not isinstance(slot.get(field), str) or not slot[field]
                for field in (
                    "source_id",
                    "concept_id",
                    "source_family",
                    "concept_family",
                    "prompt_text",
                    "prompt_hash",
                    "control_spec",
                    "slot_id",
                )
            )
            or type(slot.get("seed")) is not int
            or slot["seed"] < 0
            or not isinstance(slot.get("source_family"), str)
            or slot.get("slot_id") != _sha(slot_body)
            or slot["prompt_hash"] != hash_canonical("forge/mvp-prompt/v1", slot["prompt_text"])
            or slot["control_spec"] != f"forge/mvp-control/{kind}/v1"
        ):
            raise ValueError("calibration campaign slot is invalid")
        if slot["slot_id"] in seen_slots:
            raise ValueError("calibration campaign slot repeats")
        identity_fields = (
            "source_id",
            "concept_id",
            "source_family",
            "concept_family",
            "prompt_text",
            *(CELL_FIELDS if schema == SCHEMA_V3 else ()),
        )
        source = sources_by_identity.get(tuple(slot[field] for field in identity_fields))
        if source is None or source["base_group"] != BASE_GROUP_BY_KIND[kind]:
            raise ValueError("calibration campaign slot is not from its frozen source bank")
        seen_slots.add(slot["slot_id"])
        key = (split, kind)
        if slot["source_family"] in seen_by_kind[key]:
            raise ValueError("calibration campaign control families are dependent")
        seen_by_kind[key].add(slot["source_family"])
        if schema == SCHEMA_V3:
            if any(
                slot[field] not in vocabulary
                for field, vocabulary in zip(CELL_FIELDS, CELL_VOCABULARIES, strict=True)
            ):
                raise ValueError("calibration campaign slot category is invalid")
            if split in HELD_OUT_SPLITS:
                heldout_cells_by_kind[key][_cell(slot)] += 1
        bases_by_split[split].add(slot["base_id"])
        source_ids_by_split[split].add(slot["source_id"])
        concept_ids_by_split[split].add(slot["concept_id"])
        source_families_by_split[split].add(slot["source_family"])
        concept_families_by_split[split].add(slot["concept_family"])
        prompts_by_split[split].add(_normalized_prompt(slot["prompt_text"]))
    for split in SPLITS:
        for kind in CONTROL_KINDS:
            if len(seen_by_kind[(split, kind)]) != _count_for(
                split,
                kind,
                development_legitimate_samples,
                development_degradation_samples,
            ):
                raise ValueError("calibration campaign coverage is incomplete")
            if schema == SCHEMA_V3 and split in HELD_OUT_SPLITS:
                expected_cells = {
                    (subject, camera, style)
                    for subject in SUBJECT_CLASSES
                    for camera in CAMERA_TYPES
                    for style in STYLE_FAMILIES
                }
                if set(heldout_cells_by_kind[(split, kind)]) != expected_cells or set(
                    heldout_cells_by_kind[(split, kind)].values()
                ) != {2}:
                    raise ValueError(
                        "calibration campaign held-out category coverage is incomplete"
                    )
    split_pairs = (
        ("development", "calibration"),
        ("development", "independent"),
        ("calibration", "independent"),
    )
    if any(
        bases_by_split[left] & bases_by_split[right]
        or source_ids_by_split[left] & source_ids_by_split[right]
        or concept_ids_by_split[left] & concept_ids_by_split[right]
        or source_families_by_split[left] & source_families_by_split[right]
        or concept_families_by_split[left] & concept_families_by_split[right]
        or prompts_by_split[left] & prompts_by_split[right]
        for left, right in split_pairs
    ):
        raise ValueError(
            "calibration campaign leaks source, concept, or prompt families across splits"
        )
    return {
        "campaign_id": campaign["campaign_id"],
        "slot_count": len(slots),
        # A family is not a render count: self/new-run and independent-seed
        # comparisons can require two generations of the same base family.
        "distinct_base_families": sum(len(values) for values in bases_by_split.values()),
    }


def bind_calibration_rows(
    campaign: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, object]:
    """Ensure scored rows are exactly the previously frozen calibration slots.

    This intentionally accepts additional runner evidence (orders, video hashes,
    metric receipts) while preventing prompt, split or control substitution.
    """

    campaign_summary = validate_calibration_campaign(campaign)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("calibration rows are required")
    slots = {slot["slot_id"]: slot for slot in campaign["slots"]}
    if len(rows) != len(slots):
        raise ValueError("calibration evidence does not cover campaign slots")
    seen: set[str] = set()
    bound_fields = (
        "split",
        "kind",
        "index",
        "base_id",
        "source_id",
        "concept_id",
        "source_family",
        "concept_family",
        "prompt_text",
        "prompt_hash",
        "seed",
        "control_spec",
        *(CELL_FIELDS if campaign.get("schema") in {SCHEMA_V3, SCHEMA_V4} else ()),
        *(("source_mode",) if campaign.get("schema") == SCHEMA_V4 else ()),
    )
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("slot_id"), str):
            raise ValueError("calibration row lacks frozen slot id")
        slot = slots.get(row["slot_id"])
        if slot is None or row["slot_id"] in seen:
            raise ValueError("calibration row is outside frozen campaign")
        if any(row.get(field) != slot[field] for field in bound_fields) or (
            campaign.get("schema") == SCHEMA_V4
            and slot["kind"] == "wrong_video"
            and any(
                row.get(field) != slot[field]
                for field in (
                    "donor_source_id",
                    "donor_concept_id",
                    "donor_source_family",
                    "donor_concept_family",
                    "donor_prompt_hash",
                )
            )
        ):
            raise ValueError("calibration row differs from frozen campaign slot")
        seen.add(row["slot_id"])
    return {**campaign_summary, "evidence_hash": _sha(list(rows))}


def freeze_evaluator_pilot(
    *, pilot_label: str, entropy: str, source_bank: Sequence[Mapping[str, Any]]
) -> dict[str, object]:
    """Freeze the non-qualifying 16-family evaluator pilot.

    It exists only to decide whether the evaluator separates declared controls
    from normal variation.  It has no development/calibration/holdout splits
    and is intentionally rejected by qualification code.
    """
    sources, schema = _validate_source_bank(source_bank, schema=SCHEMA_V4)
    if (
        not pilot_label
        or not isinstance(pilot_label, str)
        or not isinstance(entropy, str)
        or len(entropy) < 16
    ):
        raise ValueError("pilot label and finalized entropy are required")
    modes = {
        mode: sorted([s for s in sources if s["source_mode"] == mode], key=lambda s: s["source_id"])
        for mode in ("static", "motion")
    }
    if any(len(modes[mode]) != 8 for mode in modes) or len(sources) != 16:
        raise ValueError("pilot requires exactly eight static and eight motion source families")
    slots: list[dict[str, object]] = []
    static_kinds = {
        "static_scene",
        "blur_mild",
        "blur_severe",
        "detail_loss",
        "evaluator_attack_mild",
        "evaluator_attack_severe",
    }
    motion_kinds = {"freeze", "shuffle"}
    cross = modes["static"][:4] + modes["motion"][:4]
    for kind in CONTROL_KINDS:
        candidates = (
            modes["static"]
            if kind in static_kinds
            else modes["motion"]
            if kind in motion_kinds or kind == "self"
            else cross
        )
        if kind == "wrong_video":
            candidates = modes["static"]
        for index, source in enumerate(candidates):
            base_id = hash_canonical("forge/evaluator-pilot-base/v1", pilot_label, source)
            body: dict[str, object] = {
                "kind": kind,
                "index": index,
                "base_id": base_id,
                "source_id": source["source_id"],
                "concept_id": source["concept_id"],
                "source_family": source["source_family"],
                "concept_family": source["concept_family"],
                "prompt_text": source["prompt_text"],
                "prompt_hash": hash_canonical("forge/mvp-prompt/v1", source["prompt_text"]),
                "seed": int(
                    hash_canonical(
                        "forge/evaluator-pilot-seed/v1", {"entropy": entropy, "base_id": base_id}
                    )[:15],
                    16,
                ),
                "control_spec": V4_CONTROL_SPECS[kind],
                "source_mode": source["source_mode"],
                **{field: source[field] for field in CELL_FIELDS},
            }
            if kind == "wrong_video":
                donor = modes["motion"][index]
                body.update(
                    {
                        "donor_source_id": donor["source_id"],
                        "donor_concept_id": donor["concept_id"],
                        "donor_source_family": donor["source_family"],
                        "donor_concept_family": donor["concept_family"],
                        "donor_prompt_hash": hash_canonical(
                            "forge/mvp-prompt/v1", donor["prompt_text"]
                        ),
                    }
                )
            slots.append({**body, "slot_id": _sha(body)})
    body = {
        "schema": "forge/evaluator-calibration-pilot/v1",
        "pilot_label": pilot_label,
        "entropy": entropy,
        "source_bank": sources,
        "source_bank_hash": hash_canonical(_source_bank_domain(SCHEMA_V4), sources),
        "review_status": "unreviewed",
        "qualification": "prohibited",
        "slots": slots,
    }
    return {**body, "pilot_id": _sha(body)}


def pilot_execution_ledger(pilot: Mapping[str, Any]) -> dict[str, object]:
    """Return exact pilot work; the pilot remains non-qualifying evidence."""
    if (
        not isinstance(pilot, Mapping)
        or pilot.get("schema") != "forge/evaluator-calibration-pilot/v1"
    ):
        raise ValueError("evaluator pilot is required")
    body = {key: value for key, value in pilot.items() if key != "pilot_id"}
    if pilot.get("pilot_id") != _sha(body) or len(body.get("slots", [])) != 96:
        raise ValueError("evaluator pilot is not canonically frozen")
    expected = freeze_evaluator_pilot(
        pilot_label=body["pilot_label"], entropy=body["entropy"], source_bank=body["source_bank"]
    )
    if dict(pilot) != expected:
        raise ValueError("evaluator pilot differs from its frozen allocation")
    slots = body["slots"]
    base_jobs = sorted({str(slot["base_id"]) for slot in slots})
    rerun_jobs = sorted(
        str(slot["slot_id"]) for slot in slots if slot["kind"] == "same_model_new_run"
    )
    seed_jobs = sorted(str(slot["slot_id"]) for slot in slots if slot["kind"] == "independent_seed")
    if len(base_jobs) != 16 or len(rerun_jobs) != 8 or len(seed_jobs) != 8:
        raise ValueError("evaluator pilot has an invalid generation allocation")
    return {
        "pilot_id": str(pilot["pilot_id"]),
        "wan50_gpu_renders": 32,
        "blinded_quality_pairs": 96,
        "blinded_luna_calls": 192,
        "local_derived_videos": 56,
        "generation_jobs": {
            "reference": base_jobs,
            "same_model_new_run": rerun_jobs,
            "independent_seed": seed_jobs,
        },
    }
