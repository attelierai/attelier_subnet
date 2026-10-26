"""Public diagnostic snapshot data; never authorizes rewards."""

import json
import math
from datetime import UTC, datetime
from statistics import mean

from forge.evaluation.vision_calibration import (
    QUALITY_AGGREGATION_V1,
    QUALITY_AGGREGATION_V2,
    QUALITY_AGGREGATION_V3,
    pair_values,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import sha256_file


def temporal_probe_row(result_path, plan_path):
    """Recompute a constructed diagnostic, keeping it distinct from Wan calibration."""
    probe = json.loads(result_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert probe["schema"] == plan["schema"] == "forge/constructed-temporal-probe/v1"
    assert probe["plan_sha256"] == sha256_file(plan_path)
    assert probe["evaluator_id"] == plan["evaluator_id"]
    assert probe["promotion_eligible"] is False and "completed_unix" in probe
    assert len(plan["scenes"]) == 8 and len(probe["pairs"]) == len(plan["pairs"]) == 24
    axes = {
        "self": tuple(ChallengeContract().axes),
        "static_freeze": ("motion_adherence", "temporal_stability", "motion_quality"),
        "dynamic_freeze": ("motion_adherence", "motion_quality"),
        "shuffle": ("temporal_stability", "motion_quality"),
        "reverse": ("motion_adherence",),
    }
    rule = plan["decision_rule"]
    summary = {}
    for actual, expected in zip(probe["pairs"], plan["pairs"], strict=True):
        assert all(actual[key] == value for key, value in expected.items())
        if actual["judgment"]["status"] == "OK":
            pair_values(actual["judgment"], QUALITY_AGGREGATION_V3)
    for kind, protected in axes.items():
        rows = [row["judgment"] for row in probe["pairs"] if row["kind"] == kind]
        assert len(rows) == (8 if kind == "self" else 4)
        identical = kind in {"self", "static_freeze"}
        hits = sum(
            (
                max(abs(row["axes"][axis]) for axis in protected) > rule["identical_tolerance"]
                if identical
                else min(row["axes"][axis] for axis in protected) <= rule["negative_delta_max"]
            )
            for row in rows
            if row["status"] == "OK"
        )
        summary[kind] = {
            "pairs": len(rows),
            "invalid": sum(row["status"] != "OK" for row in rows),
            "detections": hits,
            "order_disagreement_over_diagnostic_threshold": sum(
                row["order_disagreement"] > rule["order_disagreement_diagnostic_threshold"]
                for row in rows
            ),
        }
    assert summary == probe["summary"]
    invalid = sum(row["invalid"] for row in summary.values())
    passed = all(
        row["detections"] == 0
        if kind in {"self", "static_freeze"}
        else row["detections"] >= rule["minimum_detections_per_dynamic_family"]
        for kind, row in summary.items()
    )
    expected_status = "INCONCLUSIVE" if invalid else "PASS_DIAGNOSTIC_ONLY" if passed else "FAIL"
    assert probe["status"] == expected_status
    counts = "; ".join(
        f"{label} {summary[kind]['detections']}/{summary[kind]['pairs']}"
        for kind, label in (
            ("shuffle", "shuffled motion detected"),
            ("reverse", "reversed direction detected"),
            ("dynamic_freeze", "incorrect freezing detected"),
            ("self", "identical-copy false alarms"),
            ("static_freeze", "legitimately still false alarms"),
        )
    )
    return {
        "label": "Constructed motion · v3 diagnostic",
        "phase": "current",
        "status": {
            "PASS_DIAGNOSTIC_ONLY": "diagnostic_pass",
            "FAIL": "diagnostic_fail",
            "INCONCLUSIVE": "diagnostic_inconclusive",
        }[expected_status],
        "evaluator_id": probe["evaluator_id"],
        "note": (
            f"Eight constructed scenes from one procedural family; 24 pairs, 48 calls. {counts}. "
            f"Invalid pairs: {invalid}. "
            "Six image sheets include secret adjacent-frame transitions. "
            "Two blinded orders are averaged; disagreement is diagnostic. "
            "This is not a Wan calibration or evidence that natural-video quality is preserved. "
            "Quality promotion remains blocked."
        ),
    }


def temporal_replay_row(result_path, plan_path, prior_controls_path):
    replay, plan = (json.loads(path.read_text()) for path in (result_path, plan_path))
    assert replay["schema"] == "forge/wan-temporal-replay/v1"
    assert plan["schema"] == "forge/wan-temporal-replay-plan/v1"
    assert replay["status"] == "COMPLETE_DEVELOPMENT_REPLAY"
    assert replay["plan_sha256"] == sha256_file(plan_path)
    assert replay["evaluator_id"] == plan["evaluator_id"]
    assert plan["prior_controls_sha256"] == sha256_file(prior_controls_path)
    historical = json.loads(prior_controls_path.read_text())["controls"]["shuffle"]
    assert {row["prompt_id"] for row in historical} == {row["prompt_id"] for row in replay["pairs"]}
    previous_means = [pair_values(row, QUALITY_AGGREGATION_V1)[0] for row in historical]
    previous_flags = sum(
        min(axes["temporal_stability"], axes["motion_quality"]) <= -0.25 for axes in previous_means
    )
    assert replay["promotion_eligible"] is replay["independent_calibration"] is False
    assert len(replay["pairs"]) == 32
    assert [row["index"] for row in replay["pairs"]] == list(range(32))
    assert len({row["prompt_id"] for row in replay["pairs"]}) == 32
    informative = [
        row
        for row in replay["pairs"]
        if row["reference_pixels_sha256"] != row["control_pixels_sha256"]
    ]
    valid = [row for row in informative if row["judgment"]["status"] == "OK"]
    values = [pair_values(row["judgment"], QUALITY_AGGREGATION_V3) for row in valid]
    summary = {
        "pairs": 32,
        "informative": len(informative),
        "invalid": len(informative) - len(valid),
        "detections": sum(
            min(axes["temporal_stability"], axes["motion_quality"]) <= -0.25 for axes, _ in values
        ),
        "order_disagreement_over_025": sum(disagreement > 0.25 for _, disagreement in values),
    }
    assert summary == replay["summary"]
    return {
        "label": "Wan shuffled frames · v3 development replay",
        "phase": "current",
        "status": "development_replay",
        "evaluator_id": replay["evaluator_id"],
        "note": (
            f"The updated evaluator flagged {summary['detections']}/32 historical shuffled "
            f"pairs; {summary['invalid']} invalid pairs. Both blinded orders use the mean. "
            f"Historical responses rescored with that mean flagged {previous_flags}/32. "
            "All 32 old source videos were included without selection by outcome. "
            "This material informed v3 design, so the replay is not independent calibration. "
            "Shuffling a nearly static source may not visibly harm quality. "
            "No calibration pass, model promotion or reward follows from this result."
        ),
    }


def exploratory_experiment(plan, evaluator, report_path, quality_path, golden_path):
    """Describe measured public research weights without implying timed pairing or admission."""
    report = json.loads(report_path.read_text())
    quality = json.loads(quality_path.read_text())
    labels = {
        "fastwan_dense_adaptation": "FastWan dense adaptation · 3 steps",
        "qad_dense_adaptation": "QAD dense adaptation · 3 steps",
    }
    assert report["adaptation_flag"] in labels
    assert report["status"] == "completed_dense_adaptation_experiment"
    assert report["public_candidate"] is False
    assert quality["schema"] == "forge/auxiliary-quality-diagnostic/v1"
    assert quality["promotion_eligible"] is False
    assert quality["candidate_report_sha256"] == sha256_file(report_path)
    assert quality["reference_report_sha256"] == sha256_file(golden_path)
    assert quality["evaluator_id"] == evaluator
    assert quality["geometry"] == plan["geometry"]
    assert quality["status"] in ("COMPLETE_UNCALIBRATED_DIAGNOSTIC", "INCOMPLETE")
    assert "completed_unix" in quality
    slots = {(p["prompt_id"], seed) for p in plan["prompts"] for seed in p["video_seeds"]}
    rows = report["observations"]
    assert len(rows) == len(slots)
    assert {(r["prompt_id"], r["seed"]) for r in rows} == slots
    judgments = {(r["prompt_id"], r["seed"]): r for r in quality["pairs"]}
    assert len(quality["pairs"]) == len(judgments) == len(slots)
    times = []
    for row in rows:
        assert row["transformer_forwards"] == 3
        assert (
            row["video_sha256"]
            == judgments[(row["prompt_id"], row["seed"])]["candidate_video_sha256"]
        )
        seconds = row["generation_elapsed_seconds"]
        assert type(seconds) in (float, int) and math.isfinite(seconds) and seconds > 0
        times.append(seconds)
    deltas = None
    note = (
        "Public research weights run with dense BF16 attention and the original Wan VAE. "
        "This adaptation does not establish equivalence to the authors' native runtime. "
        "No paired speedup or quality preservation has been established. "
    )
    if quality["status"] == "INCOMPLETE":
        note += "Automatic quality judgments are incomplete."
    else:
        axes = quality["axis_means"]
        assert axes and all(type(v) in (float, int) and math.isfinite(v) for v in axes.values())
        worst = min(axes, key=axes.get)
        deltas = [{"axis": axis, "golden": axes[axis]} for axis in plan["contract"]["axes"]]
        note += (
            f"Uncalibrated mean ordinal delta against original Wan: {mean(axes.values()):+.3f}; "
            f"lowest axis, {worst.replace('_', ' ')}: {axes[worst]:+.3f}. "
            "These are not percentages or quality guarantees."
        )
    return {
        "observed_at": datetime.fromtimestamp(quality["completed_unix"], UTC).isoformat(),
        "status": "unpaired_diagnostic",
        "measurement_design": "unpaired_diagnostic",
        "prompts": len(plan["prompts"]),
        "seeds_per_prompt": len(plan["prompts"][0]["video_seeds"]),
        "measured_calls": len(rows),
        "reference_label": "Original Wan quality reference",
        "candidate_label": labels[report["adaptation_flag"]],
        "public_artifact": True,
        "admission_status": "not_admitted",
        "reference_seconds": None,
        "candidate_seconds": mean(times),
        "speedup": None,
        "ci95": None,
        "quality_status": "not_certified",
        "quality_note": note,
        **({"quality_deltas": deltas} if deltas else {}),
    }


def native_training_snapshot(path):
    """Expose measured private training progress without presenting loss as quality."""
    record = json.loads(path.read_text())
    assert record["schema"] == "attelier/native64-progress/v1"
    assert record["private_experiment"] is True and record["promotion_eligible"] is False
    assert record["geometry"] == {"frames": 81, "height": 480, "width": 832}
    assert record["planned_updates"] == record["training_prompts"] == 64
    assert record["heldout_prompts"] == 8
    assert record["teacher_steps"] == 16 and record["student_steps"] == 8
    count = record["completed_updates"]
    assert type(count) is int and 0 <= count <= 64 and len(record["losses"]) == count
    assert all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in record["losses"])
    state = record["execution_state"]
    assert state in ("PREPARING", "RUNNING_TRAINING", "RUNNING_HELDOUT", "COMPLETE", "FAILED")
    measurements = record["measurements"]
    assert isinstance(measurements, list) and len(measurements) <= count
    seen, seconds, memory = set(), [], []
    for row in measurements:
        step = row["optimizer_step"]
        assert type(step) is int and 1 <= step <= count and step not in seen
        seen.add(step)
        elapsed, peak = row["update_elapsed_seconds"], row["peak_cuda_allocated_bytes"]
        assert type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed > 0
        assert type(peak) is int and peak > 0
        seconds.append(elapsed)
        memory.append(peak)
    if state == "COMPLETE":
        assert count == 64
    return {
        "label": "Native 480p LoRA distillation",
        "status": {"FAILED": "failed", "COMPLETE": "completed"}.get(state, "running"),
        "completed_updates": count,
        "planned_updates": 64,
        "training_prompts": 64,
        "heldout_prompts": 8,
        "teacher_steps": 16,
        "student_steps": 8,
        "last_loss": record["losses"][-1] if count else None,
        "mean_update_seconds": mean(seconds) if seconds else None,
        "peak_vram_gib": max(memory) / 1024**3 if memory else None,
        "observed_at": datetime.fromtimestamp(record["observed_unix"], UTC).isoformat(),
        "quality_note": (
            "Private research experiment, not an admitted public candidate. Loss measures a "
            "training objective; it does not establish video quality or speed improvement. "
            "Eight reserved prompts compare the trained adapter "
            "with the unchanged eight-step base. "
            + (
                "Memory and update-time telemetry excludes the first fit-probe update."
                if record["historical_measurements_missing"]
                else ""
            )
        ).strip(),
    }


def native_heldout_snapshot(progress_path, report_path, quality_path, evaluator, axes):
    """Recompute a private adapter diagnostic from the exact rendered pair judgments."""
    progress, report, quality = (
        json.loads(path.read_text()) for path in (progress_path, report_path, quality_path)
    )
    assert progress["execution_state"] == "COMPLETE" and progress["completed_updates"] == 64
    assert report["status"] == "completed"
    assert quality["schema"] == "forge/local-transition-heldout-luna-diagnostic/v1"
    assert quality["promotion_eligible"] is False
    assert quality["evaluator_policy"]["evaluator_id"] == evaluator
    assert quality["render_report"]["sha256"] == sha256_file(report_path)
    assert quality["heldout_prompt_binding"] == report["heldout_prompt_binding"]
    assert quality["adapter"] == report["adapter"]
    assert report["adapter"]["sha256"] == progress["final_adapter_file_sha256"]
    assert report["sampling"]["student_steps"] == 8
    assert len(report["observations"]) == len(quality["pairs"]) == 8
    valid = []
    for index, (rendered, judged) in enumerate(
        zip(report["observations"], quality["pairs"], strict=True)
    ):
        assert judged["slot"] == index
        assert judged["heldout_prompt_sha256"] == rendered["heldout_prompt_sha256"]
        assert judged["reference_video_sha256"] == rendered["base"]["video_sha256"]
        assert judged["candidate_video_sha256"] == rendered["adapter"]["video_sha256"]
        try:
            values, _ = pair_values(judged["judgment"], QUALITY_AGGREGATION_V2)
        except ValueError:
            assert judged["minimum_axes"] is None
        else:
            assert values == judged["minimum_axes"] and set(values) == set(axes)
            valid.append(values)
    invalid = 8 - len(valid)
    assert invalid == quality["invalid_pairs"]
    assert quality["status"] == (
        "COMPLETE_UNCALIBRATED_DIAGNOSTIC_WITH_INVALID_PAIRS"
        if invalid
        else "COMPLETE_UNCALIBRATED_DIAGNOSTIC"
    )
    means = {axis: mean(row[axis] for row in valid) for axis in axes} if not invalid else None
    if means:
        assert means == quality["minimum_axis_means"]
        assert mean(means.values()) == quality["mean_minimum_axis_delta"]
    return {
        "status": "incomplete" if invalid else "uncalibrated_diagnostic",
        "reference_label": "Unchanged eight-step base",
        "prompts": 8,
        "invalid_pairs": invalid,
        "mean_ordinal_delta": mean(means.values()) if means else None,
        "quality_deltas": [
            {"axis": axis, "delta": means[axis] if means else None} for axis in axes
        ],
        "observed_at": datetime.fromtimestamp(quality["completed_unix"], UTC).isoformat(),
        "quality_note": (
            "Same initial latent seeds, eight reserved prompts. Each axis uses the minimum "
            "of two blinded judge orders. Automatic ordinal judgments are uncalibrated; "
            "this is not a comparison with the original 50-step model, "
            "a percentage, a speedup measurement, or a quality certificate."
        ),
    }


def same_model_null_snapshot(heldout_path, render_path, quality_path, evaluator, axes):
    """Expose order aggregation bias without treating individual random draws as equal quality."""
    first, second, quality = (
        json.loads(path.read_text()) for path in (heldout_path, render_path, quality_path)
    )
    assert quality["schema"] == "attelier/same-model-null-quality/v1"
    assert quality["promotion_eligible"] is False and quality["evaluator_id"] == evaluator
    assert second["status"] == "COMPLETED_PRIVATE_NULL_DIAGNOSTIC"
    assert quality["reference_report_sha256"] == sha256_file(heldout_path)
    assert quality["independent_report_sha256"] == sha256_file(render_path)
    assert second["reference_report"]["sha256"] == sha256_file(heldout_path)
    assert quality["heldout_prompt_binding"] == first["heldout_prompt_binding"]
    assert second["heldout_prompt_binding"] == first["heldout_prompt_binding"]
    assert len(first["observations"]) == len(second["observations"]) == len(quality["pairs"]) == 8
    pairs = []
    for index, (old, new, judged) in enumerate(
        zip(first["observations"], second["observations"], quality["pairs"], strict=True)
    ):
        assert judged["slot"] == index == new["slot"]
        assert old["heldout_prompt_sha256"] == new["heldout_prompt_sha256"]
        expected = [old["base"]["video_sha256"], new["video_sha256"]]
        if not judged["assignment"]["second_realization_is_candidate"]:
            expected.reverse()
        assert [judged["reference_sha256"], judged["candidate_sha256"]] == expected
        try:
            average, _ = pair_values(judged["judgment"], QUALITY_AGGREGATION_V1)
            minimum, _ = pair_values(judged["judgment"], QUALITY_AGGREGATION_V2)
        except ValueError:
            assert judged["mean_axes"] is None and judged["minimum_axes"] is None
        else:
            assert average == judged["mean_axes"] and minimum == judged["minimum_axes"]
            assert set(average) == set(minimum) == set(axes)
            pairs.append((average, minimum))
    invalid = 8 - len(pairs)
    assert invalid == quality["invalid_pairs"]
    assert quality["status"] == (
        "INCOMPLETE_DIAGNOSTIC" if invalid else "COMPLETE_UNCALIBRATED_DIAGNOSTIC"
    )
    averages = [mean(a.values()) for a, _ in pairs]
    minima = [mean(b.values()) for _, b in pairs]
    return {
        "status": "incomplete" if invalid else "uncalibrated_diagnostic",
        "prompts": 8,
        "invalid_pairs": invalid,
        "mean_of_orders": mean(averages) if not invalid else None,
        "minimum_of_orders": mean(minima) if not invalid else None,
        "mean_minus_minimum": mean(averages) - mean(minima) if not invalid else None,
        "observed_at": datetime.fromtimestamp(quality["completed_unix"], UTC).isoformat(),
        "quality_note": (
            "The same unchanged model generated two independent seeds per prompt, with "
            "preregistered random reference/candidate assignment. Individual videos need not "
            "have equal quality. This measures order-aggregation behavior on eight pairs; "
            "it is not a calibrated population false-rejection rate."
        ),
    }


def teacher_reference_experiment(progress_path, heldout_path, render_path, quality_path, evaluator):
    """Report adapter8 versus frozen teacher16 without claiming a paired speed measurement."""
    progress, heldout, teacher, quality = (
        json.loads(path.read_text())
        for path in (progress_path, heldout_path, render_path, quality_path)
    )
    assert progress["execution_state"] == "COMPLETE" and progress["completed_updates"] == 64
    assert teacher["status"] == "COMPLETED_PRIVATE_TEACHER_REFERENCE"
    assert teacher["schema"] == "attelier/private-teacher-reference/v1"
    assert quality["schema"] == "attelier/native64-teacher-quality/v1"
    assert quality["promotion_eligible"] is False and quality["evaluator_id"] == evaluator
    assert quality["training_evidence_sha256"] == sha256_file(progress_path)
    assert quality["reference_report_sha256"] == sha256_file(heldout_path)
    assert quality["teacher_report_sha256"] == sha256_file(render_path)
    assert teacher["reference_report"]["sha256"] == sha256_file(heldout_path)
    assert teacher["adapter_identity"] == quality["adapter_identity"] == heldout["adapter"]
    assert heldout["adapter"]["sha256"] == progress["final_adapter_file_sha256"]
    assert quality["heldout_prompt_binding"] == teacher["heldout_prompt_binding"]
    assert quality["heldout_prompt_binding"] == heldout["heldout_prompt_binding"]
    assert teacher["sampling"]["steps"] == 16 and heldout["sampling"]["student_steps"] == 8
    assert teacher["dependencies"] == heldout["dependencies"]
    assert (
        len(teacher["observations"]) == len(heldout["observations"]) == len(quality["pairs"]) == 8
    )
    minima, averages, seconds, teacher_seconds = [], [], [], []
    for index, (candidate, reference, judged) in enumerate(
        zip(heldout["observations"], teacher["observations"], quality["pairs"], strict=True)
    ):
        assert judged["slot"] == reference["slot"] == index
        assert candidate["latent_seed"] == reference["latent_seed"] == judged["latent_seed"]
        assert reference["actual_transformer_forwards"] == 32
        assert candidate["adapter"]["actual_transformer_forwards"] == 16
        assert reference["video_sha256"] == judged["reference_video_sha256"]
        assert candidate["adapter"]["video_sha256"] == judged["candidate_video_sha256"]
        elapsed = candidate["adapter"]["generation_elapsed_seconds"]
        assert type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0
        seconds.append(elapsed)
        reference_elapsed = reference["generation_elapsed_seconds"]
        assert (
            type(reference_elapsed) in (int, float)
            and math.isfinite(reference_elapsed)
            and reference_elapsed > 0
        )
        teacher_seconds.append(reference_elapsed)
        try:
            average, _ = pair_values(judged["judgment"], QUALITY_AGGREGATION_V1)
            minimum, _ = pair_values(judged["judgment"], QUALITY_AGGREGATION_V2)
        except ValueError:
            assert judged["minimum_axes"] is None and judged["mean_axes"] is None
        else:
            assert minimum == judged["minimum_axes"] and average == judged["mean_axes"]
            minima.append(minimum)
            averages.append(average)
    invalid = 8 - len(minima)
    assert invalid == quality["invalid_pairs"]
    assert quality["status"] == (
        "INCOMPLETE_UNCALIBRATED_DIAGNOSTIC" if invalid else "COMPLETE_UNCALIBRATED_DIAGNOSTIC"
    )
    note = (
        "Private native 480p LoRA after 64 training updates, measured on eight reserved prompts. "
    )
    if invalid:
        note += "Teacher comparison has invalid judgments; no aggregate quality delta is reported. "
    else:
        minimum_delta = mean(mean(row.values()) for row in minima)
        average_delta = mean(mean(row.values()) for row in averages)
        note += (
            f"Uncalibrated ordinal delta against the 16-step teacher: {minimum_delta:+.3f} "
            f"using the minimum of two judge orders; {average_delta:+.3f} using their mean. "
        )
        axis_means = {axis: mean(row[axis] for row in minima) for axis in minima[0]}
        lowest_axis = min(axis_means, key=axis_means.get)
        note += (
            f"Lowest axis mean: {lowest_axis.replace('_', ' ')} {axis_means[lowest_axis]:+.3f}. "
        )
    note += (
        "These are not percentages or quality guarantees. "
        f"The separately measured teacher16 reference averaged {mean(teacher_seconds):.2f}s "
        "on the same GPU. Timings reuse the adapter renders "
        "and exclude cached text encoding; they are not ABBA measurements. "
        "No original 50-step quality preservation or paired speedup has been established."
    )
    return {
        "observed_at": datetime.fromtimestamp(quality["completed_unix"], UTC).isoformat(),
        "status": "unpaired_diagnostic",
        "measurement_design": "unpaired_diagnostic",
        "prompts": 8,
        "seeds_per_prompt": 1,
        "measured_calls": 8,
        "reference_label": "Frozen Wan teacher · 16 Uniform Euler steps",
        "candidate_label": "Native trained LoRA · 8 steps",
        "public_artifact": False,
        "admission_status": "not_admitted",
        "reference_seconds": None,
        "candidate_seconds": mean(seconds),
        "speedup": None,
        "ci95": None,
        "quality_status": "not_certified",
        "quality_note": note,
    }
