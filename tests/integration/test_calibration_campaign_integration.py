"""A frozen calibration campaign cannot be relabelled after judging."""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import product

import pytest

from forge.cli import main
from forge.evaluation.calibration_campaign import (
    bind_calibration_rows,
    freeze_calibration_campaign,
    freeze_evaluator_pilot,
    validate_calibration_campaign,
)
from forge.evaluation.calibration_power import (
    campaign_execution_ledger,
    campaign_preflight,
    minimum_observations,
    projected_campaign_cost,
    projected_development_cost,
)
from forge.evaluation.mvp_calibration import (
    fit_policy_from_development,
    validate_qualified_calibration,
)
from forge.evaluation.mvp_policy import (
    CONTROL_AXES,
    CONTROL_KINDS,
    LEGITIMATE_CONTROLS,
    QUALITY_AXES,
    _sha,
)
from forge.evaluation.panel_power import confirmation_power_preflight, probe_quality_information

PROVENANCE = {
    "source_implementation_hash": "forge-fusion@pinned",
    "vbench_model_hashes": "vbench-weights@pinned",
    "luna_evaluator_id": "luna@pinned",
}


def _source_bank() -> list[dict[str, str]]:
    """Synthetic fixture metadata only; it never represents a reviewed source bank."""

    counts = {"repeat": 96, "independent_seed": 96, "static": 96, "motion": 72}
    cells = list(
        product(
            ("human", "animal", "rigid_object", "scenery"),
            ("static", "moving"),
            ("realistic", "stylized"),
        )
    )
    return [
        {
            "base_group": group,
            "source_id": f"fixture-source-{group}-{index}",
            "concept_id": f"fixture-concept-{group}-{index}",
            "source_family": f"fixture-source-family-{group}-{index}",
            "concept_family": f"fixture-concept-family-{group}-{index}",
            "prompt_text": f"Fixture calibration scene {group} with distinct concept {index}.",
            "subject_class": cells[index % len(cells)][0],
            "camera_type": cells[index % len(cells)][1],
            "style_family": cells[index % len(cells)][2],
        }
        for group, count in counts.items()
        for index in range(count)
    ]


def _legacy_source_bank() -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in source.items()
            if key not in {"subject_class", "camera_type", "style_family"}
        }
        for source in _source_bank()
    ]


def _v4_source_bank() -> list[dict[str, str]]:
    """Qualification fixture: 64 reusable families per partition."""
    rows = []
    for mode in ("static", "motion"):
        mode_cells = (
            [
                (subject, "static", style)
                for subject in ("human", "animal", "rigid_object", "scenery")
                for style in ("realistic", "stylized")
            ]
            if mode == "static"
            else list(
                product(
                    ("human", "animal", "rigid_object", "scenery"),
                    ("static", "moving"),
                    ("realistic", "stylized"),
                )
            )
        )
        repeats = 12 if mode == "static" else 6
        for cell_number, cell in enumerate(mode_cells):
            for repeat in range(repeats):
                index = cell_number * repeats + repeat
                rows.append(
                    {
                        "source_mode": mode,
                        "source_id": f"v4-{mode}-source-{index}",
                        "concept_id": f"v4-{mode}-concept-{index}",
                        "source_family": f"v4-{mode}-source-family-{index}",
                        "concept_family": f"v4-{mode}-concept-family-{index}",
                        "prompt_text": f"V4 unique {mode} semantic source {index}.",
                        "subject_class": cell[0],
                        "camera_type": cell[1],
                        "style_family": cell[2],
                    }
                )
    return rows


def _orders(kind: str) -> list[dict[str, object]]:
    axes = {axis: 0.0 for axis in QUALITY_AXES}
    if kind not in LEGITIMATE_CONTROLS:
        for axis in CONTROL_AXES[kind]:
            axes[axis] = -0.5
    return [
        {"reference_first": True, "axes": dict(axes)},
        {"reference_first": False, "axes": dict(axes)},
    ]


def _rows(campaign: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            **slot,
            "observable": True,
            "orders": _orders(str(slot["kind"])),
        }
        for slot in campaign["slots"]
    ]


def _receipt(*, v4: bool = True) -> tuple[object, dict[str, object]]:
    campaign = freeze_calibration_campaign(
        campaign_label="calibration-r1",
        entropy="f" * 64,
        source_bank=_v4_source_bank() if v4 else _source_bank(),
    )
    rows = _rows(campaign)
    development = [row for row in rows if row["split"] == "development"]
    policy, derivation = fit_policy_from_development(development, evaluator_provenance=PROVENANCE)
    receipt = {
        "policy_id": policy.policy_id,
        "campaign": campaign,
        "rows": rows,
        "evaluator_provenance": PROVENANCE,
        "threshold_derivation": derivation,
    }
    receipt["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": rows,
        }
    )
    return policy, receipt


def test_campaign_has_exact_holdouts_and_reusable_base_renders_are_explicit():
    campaign = freeze_calibration_campaign(
        campaign_label="calibration-r1", entropy="f" * 64, source_bank=_source_bank()
    )
    summary = validate_calibration_campaign(campaign)

    assert campaign["schema"] == "forge/mvp-calibration-campaign/v3"
    assert summary["slot_count"] == 960
    assert summary["distinct_base_families"] == 360
    base_counts = {
        split: len({slot["base_id"] for slot in campaign["slots"] if slot["split"] == split})
        for split in ("calibration", "independent")
    }
    assert base_counts == {"calibration": 128, "independent": 128}
    for split in ("calibration", "independent"):
        for kind in CONTROL_KINDS:
            cells = [
                (slot["subject_class"], slot["camera_type"], slot["style_family"])
                for slot in campaign["slots"]
                if slot["split"] == split and slot["kind"] == kind
            ]
            assert len(cells) == 32
            assert {cells.count(cell) for cell in set(cells)} == {2}
            assert len(set(cells)) == 16
    preflight = campaign_preflight(campaign)
    assert preflight["held_out_rows"] == 768
    assert preflight["blinded_luna_calls"] == 1_536
    assert campaign_execution_ledger(campaign) == {
        "wan50_gpu_renders": 552,
        "local_derived_videos": 504,
        "unique_media_before_deduplicating_wrong_video": 1_056,
        "blinded_quality_pairs": 960,
        "blinded_luna_calls": 1_920,
    }
    cost = projected_campaign_cost(
        campaign,
        wan50_seconds=176.47976928891148,
        gpu_usd_hour=0.99,
        judge_usd_per_pair=0.1579168 / 18,
    )
    assert cost["known_usd"] == pytest.approx(35.218, abs=0.01)
    development_cost = projected_development_cost(
        campaign,
        wan50_seconds=176.47976928891148,
        gpu_usd_hour=0.99,
        judge_usd_per_pair=0.1579168 / 18,
    )
    assert development_cost["wan50_gpu_renders"] == 168
    assert development_cost["blinded_luna_calls"] == 384
    assert development_cost["known_usd"] == pytest.approx(9.845, abs=0.01)
    assert preflight["minimum_observations"] == {
        "all_hits_minimum": 17,
        "zero_false_alarms_minimum": 23,
        "required_per_kind": 23,
    }
    assert (
        minimum_observations(sensitivity_lower=0.8, false_alarm_upper=0.15, alpha=0.05)[
            "required_per_kind"
        ]
        < 32
    )


def test_v4_binds_pilot_families_to_development_only():
    pilot_bank = []
    for mode in ("static", "motion"):
        for index in range(8):
            pilot_bank.append(
                {
                    "source_mode": mode,
                    "source_id": f"pilot-{mode}-{index}",
                    "concept_id": f"pilot-concept-{mode}-{index}",
                    "source_family": f"pilot-family-{mode}-{index}",
                    "concept_family": f"pilot-concept-family-{mode}-{index}",
                    "prompt_text": f"Pilot {mode} independently reviewed family {index}.",
                    "subject_class": ("human", "animal", "rigid_object", "scenery")[index % 4],
                    "camera_type": "static"
                    if mode == "static"
                    else ("static", "moving")[index % 2],
                    "style_family": ("realistic", "stylized")[index % 2],
                }
            )
    pilot = freeze_evaluator_pilot(pilot_label="pilot", entropy="e" * 64, source_bank=pilot_bank)
    campaign = freeze_calibration_campaign(
        campaign_label="full",
        entropy="f" * 64,
        source_bank=_v4_source_bank() + pilot_bank,
        development_pilot=pilot,
    )
    assert campaign["development_pilot_id"] == pilot["pilot_id"]
    pilot_ids = {row["source_id"] for row in pilot_bank}
    assert pilot_ids <= {
        row["source_id"] for row in campaign["slots"] if row["split"] == "development"
    }
    assert not pilot_ids & {
        row["source_id"] for row in campaign["slots"] if row["split"] != "development"
    }
    development_controls = {
        (
            row["kind"],
            row["source_id"],
            row.get("donor_source_id"),
            row["base_id"],
            row["seed"],
            tuple(sorted(row["control_spec"].items())),
        )
        for row in campaign["slots"]
        if row["split"] == "development"
    }
    for row in pilot["slots"]:
        assert (
            row["kind"],
            row["source_id"],
            row.get("donor_source_id"),
            row["base_id"],
            row["seed"],
            tuple(sorted(row["control_spec"].items())),
        ) in development_controls


def test_campaign_rejects_renamed_families_with_the_same_prompt_or_a_numbered_take():
    bank = _source_bank()
    bank[-1] = {
        **bank[-1],
        "source_id": "renamed-source",
        "concept_id": "renamed-concept",
        "source_family": "renamed-source-family",
        "concept_family": "renamed-concept-family",
        "prompt_text": bank[0]["prompt_text"],
    }
    with pytest.raises(ValueError, match="repeats a semantic prompt"):
        freeze_calibration_campaign(campaign_label="duplicate", entropy="f" * 64, source_bank=bank)

    bank = _source_bank()
    bank[0]["prompt_text"] = "A cyclist pedals beside a canal, take 1."
    bank[-1]["prompt_text"] = "A cyclist pedals beside a canal, take 2."
    with pytest.raises(ValueError, match="repeats a semantic prompt"):
        freeze_calibration_campaign(campaign_label="takes", entropy="f" * 64, source_bank=bank)


def test_v3_rejects_missing_held_out_category_and_label_mutation_after_scoring():
    bank = _source_bank()
    for source in bank:
        if source["base_group"] == "repeat" and source["subject_class"] == "scenery":
            source["subject_class"] = "human"
    with pytest.raises(ValueError, match="lacks two real repeat families"):
        freeze_calibration_campaign(
            campaign_label="missing-cell", entropy="f" * 64, source_bank=bank
        )

    campaign = freeze_calibration_campaign(
        campaign_label="labels-bound", entropy="f" * 64, source_bank=_source_bank()
    )
    rows = _rows(campaign)
    rows[0]["style_family"] = "realistic" if rows[0]["style_family"] == "stylized" else "stylized"
    with pytest.raises(ValueError, match="differs from frozen"):
        bind_calibration_rows(campaign, rows)


def test_historical_v2_campaign_still_validates_and_binds_rows(tmp_path):
    campaign = freeze_calibration_campaign(
        campaign_label="legacy-v2", entropy="f" * 64, source_bank=_legacy_source_bank()
    )
    assert campaign["schema"] == "forge/mvp-calibration-campaign/v2"
    assert validate_calibration_campaign(campaign)["slot_count"] == 960
    binding = bind_calibration_rows(campaign, _rows(campaign))
    assert binding["campaign_id"] == campaign["campaign_id"]
    legacy_bank = tmp_path / "legacy-bank.json"
    legacy_bank.write_text(json.dumps(_legacy_source_bank()))
    output = tmp_path / "new-campaign.json"
    with pytest.raises(ValueError, match="preregistered category labels"):
        main(
            [
                "mvp-calibration-plan",
                "--campaign-label",
                "new",
                "--entropy",
                "f" * 64,
                "--source-bank",
                str(legacy_bank),
                "--out",
                str(output),
            ]
        )
    assert not output.exists()


def test_qualified_calibration_requires_frozen_slot_binding_and_plan_hash():
    policy, receipt = _receipt()

    qualified = validate_qualified_calibration(receipt, policy)

    assert qualified["status"] == "PASS"
    calibration_hash = qualified["binding"]["calibration_hash"]
    assert (
        validate_qualified_calibration(receipt, policy, required_calibration_hash=calibration_hash)[
            "status"
        ]
        == "PASS"
    )

    relabelled = deepcopy(receipt)
    relabelled["rows"][0]["kind"] = "blur_mild"
    relabelled["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": relabelled["rows"],
        }
    )
    with pytest.raises(ValueError, match="differs from frozen"):
        bind_calibration_rows(relabelled["campaign"], relabelled["rows"])
    assert validate_qualified_calibration(relabelled, policy)["status"] == "INCOMPLETE"


def test_cli_freezes_a_new_campaign_without_rendering(tmp_path):
    destination = tmp_path / "calibration.json"
    source_bank = tmp_path / "source-bank.json"
    source_bank.write_text(json.dumps(_source_bank()))

    assert (
        main(
            [
                "mvp-calibration-plan",
                "--campaign-label",
                "calibration-r1",
                "--entropy",
                "f" * 64,
                "--source-bank",
                str(source_bank),
                "--out",
                str(destination),
            ]
        )
        == 0
    )
    assert destination.exists()
    with pytest.raises(FileExistsError):
        main(
            [
                "mvp-calibration-plan",
                "--campaign-label",
                "calibration-r1",
                "--entropy",
                "f" * 64,
                "--source-bank",
                str(source_bank),
                "--out",
                str(destination),
            ]
        )


def test_panel_power_refuses_to_infer_128_prompt_precision_from_the_small_probe():
    preflight = confirmation_power_preflight(quality_margin=0.125)

    assert preflight["quality_bounds_examined"] == 272
    assert preflight["intersection_union_false_promotion_alpha"] == 0.05
    assert preflight["all_bounds_at_boundary_independent_inconclusive_probability"] > 0.999
    probe = {
        "controls": [
            {"kind": "blur_mild", "axes": {"image_quality": 0.0}, "order_disagreement": 0.0},
            {"kind": "freeze", "axes": {"motion_quality": -0.5}, "order_disagreement": 0.25},
        ]
    }
    information = probe_quality_information(probe)
    assert information["panel_power_established"] is False
    assert information["zero_signal_control_kinds"] == ["blur_mild"]


def test_cli_writes_an_informational_power_report_without_changing_policy(tmp_path):
    policy, _ = _receipt()
    policy_path = tmp_path / "policy.json"
    probe_path = tmp_path / "probe.json"
    output = tmp_path / "power.json"
    policy_path.write_text(json.dumps(policy.to_dict()))
    probe_path.write_text(json.dumps({"controls": []}))

    assert (
        main(
            [
                "mvp-panel-power",
                "--policy",
                str(policy_path),
                "--probe",
                str(probe_path),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["promotion_enabled"] is False
    assert report["confirmation"]["quality_bounds_examined"] == 272
    assert "not measured panel power" in report["confirmation"]["boundary_probability_assumptions"]


def test_cli_replays_frozen_strata_without_treating_resamples_as_independent_panels(tmp_path):
    policy, receipt = _receipt(v4=False)
    policy_path, evidence_path = tmp_path / "policy.json", tmp_path / "evidence.json"
    policy_path.write_text(json.dumps(policy.to_dict()))
    evidence_path.write_text(json.dumps(receipt))
    output = tmp_path / "replay.json"
    command = [
        "mvp-panel-power",
        "--policy",
        str(policy_path),
        "--calibration",
        str(evidence_path),
        "--draws",
        "2",
        "--out",
        str(output),
    ]
    assert main(command) == 0
    diagnostic = json.loads(output.read_text())["conditional_replay"]
    assert diagnostic["promotion_enabled"] is False
    assert diagnostic["panel_power_established"] is False
    assert len(diagnostic["controls"]) == 24
    for control in diagnostic["controls"]:
        assert control["independent_source_families"] == 32
        assert control["sources_per_cell"] == 2
        expected = 1.0 if control["kind"] in LEGITIMATE_CONTROLS else 0.0
        assert control["conditional_quality_clear_fraction"] == expected
    # Positive deltas elsewhere hide the bad cell in the global mean, but not the gate.
    for row in receipt["rows"]:
        if row["kind"] != "self" or row["split"] == "development":
            continue
        bad_cell = row["subject_class"] == "human" and (
            row["camera_type"],
            row["style_family"],
        ) == ("static", "realistic")
        for order in row["orders"]:
            order["axes"]["image_quality"] = -0.5 if bad_cell else 0.1
    receipt["independent_check_hash"] = _sha(
        {
            "schema": "forge/mvp-independent-calibration/v1",
            "policy_id": policy.policy_id,
            "threshold_derivation_hash": policy.threshold_derivation_hash,
            "rows": receipt["rows"],
        }
    )
    evidence_path.write_text(json.dumps(receipt))
    command[-1] = str(tmp_path / "category-defect.json")
    assert main(command) == 0
    report = json.loads((tmp_path / "category-defect.json").read_text())["conditional_replay"]
    assert all(x["outcomes"]["FAIL"] == 2 for x in report["controls"] if x["kind"] == "self")
    assert report["source_calibration"]["status"] != "PASS"
    receipt["rows"][0]["style_family"] = "post-hoc-label"
    evidence_path.write_text(json.dumps(receipt))
    command[-1] = str(tmp_path / "relabelled.json")
    with pytest.raises(ValueError, match="differs from frozen"):
        main(command)
    assert not (tmp_path / "relabelled.json").exists()
