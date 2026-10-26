"""Public diagnostic snapshot data; never authorizes rewards."""

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from forge.evaluation.vision_calibration import (
    calibrate_controls,
    pair_values,
)
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.validator.dashboard_history import (
    exploratory_experiment,
    native_heldout_snapshot,
    native_training_snapshot,
    same_model_null_snapshot,
    teacher_reference_experiment,
    temporal_probe_row,
    temporal_replay_row,
)
from forge.validator.dashboard_media import public_examples

BASE_MODEL_NAME = "Wan2.1 T2V 1.3B"


def _named_reference(label):
    """Older snapshots called the base "Wan base"; the page names the model."""
    match = re.fullmatch(r"Wan base · (\d+) steps", label) if isinstance(label, str) else None
    return f"{BASE_MODEL_NAME} · {match.group(1)} steps" if match else label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "contract", "timing", "quality", "calibration", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "training-evidence",
        "training-render-report",
        "training-quality",
        "null-render-report",
        "null-quality",
        "teacher-render-report",
        "teacher-quality",
        "testnet-model",
        "testnet-pack",
        "testnet-plan",
        "testnet-evaluation",
        "history",
        "media-selection",
        "golden-panel",
        "incumbent-panel",
        "candidate-panel",
        "media-out",
        "temporal-probe",
        "temporal-probe-plan",
        "temporal-replay",
        "temporal-replay-plan",
        "validators",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--redteam", type=Path, action="append", default=[])
    parser.add_argument("--exploratory-report", type=Path, action="append", default=[])
    parser.add_argument("--exploratory-quality", type=Path, action="append", default=[])
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text())

    plan, timing, quality, controls = map(
        read, (args.plan, args.timing, args.quality, args.calibration)
    )
    contract = ChallengeContract.from_dict(read(args.contract))
    pilot_contract = ChallengeContract.from_dict(plan["contract"])
    assert plan["contract_hash"] == pilot_contract.contract_hash(), "invalid pilot contract"
    assert timing["contract_hash"] == quality["contract_hash"] == plan["contract_hash"]
    assert controls["contract_hash"] == plan["contract_hash"]
    assert controls == calibrate_controls(controls), "control receipt does not recompute"
    assert "completed_unix" in quality, "quality receipt is incomplete"
    assert "completed_unix" in controls, "control campaign is incomplete"
    admission = timing["candidate_admission"]
    assert admission in {"private_experiment", "public_snapshot"}
    for key in ("base_model_id", "base_model_revision", "width", "height", "frames", "fps"):
        assert getattr(contract, key) == getattr(pilot_contract, key), "different challenge"
    slots = {(p["prompt_id"], s) for p in plan["prompts"] for s in p["video_seeds"]}
    grouped, durations = defaultdict(list), defaultdict(list)
    measurements = timing["measurements"]
    assert len(measurements) == len(slots) * 4
    seen = set()
    for start in range(0, len(measurements), 4):
        rows = measurements[start : start + 4]
        slot = (rows[0]["prompt_id"], rows[0]["seed"])
        assert slot in slots and slot not in seen
        seen.add(slot)
        assert [r["side"] for r in rows] == ["reference", "candidate", "candidate", "reference"]
        assert [r["run_order"] for r in rows] == [0, 1, 2, 3]
        for row in rows:
            assert (row["prompt_id"], row["seed"]) == slot and row["status"] == "OK"
            value = row["wall_time_seconds"]
            assert type(value) in (float, int) and math.isfinite(value) and value > 0
            durations[row["side"]].append(value)
        grouped[slot[0]].append(
            (
                math.log(rows[0]["wall_time_seconds"] / rows[1]["wall_time_seconds"])
                + math.log(rows[3]["wall_time_seconds"] / rows[2]["wall_time_seconds"])
            )
            / 2
        )
    logs = [mean(values) for values in grouped.values()]
    assert len(logs) >= 2
    recipes = {
        side: {row["steps"] for row in measurements if row["side"] == side}
        for side in ("reference", "candidate")
    }
    assert all(len(steps) == 1 for steps in recipes.values()), "mixed generation recipes"
    center = mean(logs)
    radius = float(t.ppf(0.975, len(logs) - 1)) * stdev(logs) / math.sqrt(len(logs))
    labels = {
        "self": "Identical copies",
        "freeze": "Frozen motion",
        "blur": "Blur",
        "shuffle": "Shuffled frames",
        "wrong_video": "Wrong scene",
    }
    snapshot = {
        "schema": "forge/dashboard-development/v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "phase": "development",
        "winner": None,
        "challenge": {
            "name": "Faster video, preserved quality",
            "base_model": BASE_MODEL_NAME,
            "base_url": f"https://huggingface.co/{contract.base_model_id}/tree/{contract.base_model_revision}",
            "revision": contract.base_model_revision,
            "width": contract.width,
            "height": contract.height,
            "frames": contract.frames,
            "fps": contract.fps,
            "storage_backends": ["Hugging Face", "Hippius"],
            "reward_rule": "100% to the certified winner",
            "min_speedup": contract.min_speedup,
            "quality_margin": contract.axis_noninferiority_margin,
        },
        "benchmark": {
            "status": "development_diagnostic",
            "prompts": len(plan["prompts"]),
            "seeds_per_prompt": len(plan["prompts"][0]["video_seeds"]),
            "measured_calls": len(measurements),
            "reference_label": f"{BASE_MODEL_NAME} · {next(iter(recipes['reference']))} steps",
            "candidate_label": (
                "Experimental LoRA" if admission == "private_experiment" else "Public candidate"
            )
            + f" · {next(iter(recipes['candidate']))} step(s)",
            "public_artifact": admission == "public_snapshot",
            "reference_seconds": mean(durations["reference"]),
            "candidate_seconds": mean(durations["candidate"]),
            "speedup": math.exp(center),
            "ci95": [math.exp(center - radius), math.exp(center + radius)],
            "quality_status": "not_certified",
            "quality_note": (
                "Quality preservation has not been demonstrated. "
                f"Automatic calibration: {controls['status'].lower()}. "
                + (
                    "This private experiment is not an admitted model."
                    if admission == "private_experiment"
                    else "Public availability alone does not certify quality."
                )
            ),
        },
        "controls": [
            {
                "kind": k,
                "label": labels[k],
                "total": v["n_prompts"],
                "detected": v["detections"],
                "unstable": v["invalid"],
            }
            for k, v in controls["summary"].items()
        ],
        "controls_policy": {
            "version": controls["control_rule"]["version"],
            "aggregation": controls["control_rule"].get(
                "quality_aggregation", "mean_of_blinded_orders"
            ),
            "status": controls["status"],
            "minimum_prompts": controls["control_rule"]["min_prompts_per_kind"],
        },
        "protocol": {
            "screen_prompts": contract.screen_prompts,
            "duel_prompts": contract.duel_prompts,
            "confirmation_prompts": contract.confirmation_prompts,
            "seeds_per_prompt": contract.confirmation_seeds,
            "axes": list(contract.axes),
            "prompt_policy": (
                "Private prompt packs committed before model submissions close; "
                "selection and seeds use later finalized chain entropy. "
                "Pack entries separate calibration, screening and confirmation. "
                "The history verifier rejects normalized text and source-group reuse "
                "against a supplied checkpoint. Authenticated checkpoint transport "
                "from certified state is still missing; production blocks without it. "
                "Semantic separation and prompt secrecy remain additional requirements."
            ),
        },
    }
    snapshot["experiments"] = []
    if args.history:
        previous = read(args.history)
        assert previous["schema"] == snapshot["schema"] and previous["winner"] is None
        assert previous["challenge"]["revision"] == contract.base_model_revision
        snapshot["experiments"].extend(
            {**run, "reference_label": _named_reference(run.get("reference_label"))}
            for run in previous.get("experiments")
            or [{"observed_at": previous["updated_at"], **previous["benchmark"]}]
        )
    aggregation = controls["control_rule"].get("quality_aggregation", "mean_of_blinded_orders")
    quality_diagnostics = []
    quality_axes = {}
    for comparison in quality["comparisons"]:
        try:
            axes = [pair_values(row, aggregation)[0] for row in comparison["observations"]]
            assert len(axes) == len(slots)
        except (ValueError, AssertionError):
            quality_axes[comparison["reference_role"]] = None
            quality_diagnostics.append(
                f"{comparison['reference_role']}: incomplete automatic judgments"
            )
        else:
            quality_axes[comparison["reference_role"]] = {
                axis: mean(row[axis] for row in axes) for axis in contract.axes
            }
            delta = mean(value for row in axes for value in row.values())
            quality_diagnostics.append(f"{comparison['reference_role']}: {delta:+.3f}")
    snapshot["benchmark"]["quality_note"] += (
        " Uncalibrated mean ordinal deltas (candidate minus reference): "
        + "; ".join(quality_diagnostics)
        + ". These are not percentages or quality guarantees."
    )
    snapshot["benchmark"]["quality_deltas"] = [
        {
            "axis": axis,
            **{
                role: values[axis] if values is not None else None
                for role, values in quality_axes.items()
            },
        }
        for axis in contract.axes
    ]
    snapshot["experiments"].append(
        {
            "observed_at": datetime.fromtimestamp(quality["completed_unix"], UTC).isoformat(),
            **snapshot["benchmark"],
        }
    )
    assert len(args.exploratory_report) == len(args.exploratory_quality)
    if args.exploratory_report:
        assert args.golden_panel, "exploratory quality must identify the golden reference panel"
        for report_path, quality_path in zip(
            args.exploratory_report, args.exploratory_quality, strict=True
        ):
            snapshot["experiments"].append(
                exploratory_experiment(
                    plan, quality["evaluator_id"], report_path, quality_path, args.golden_panel
                )
            )
    if args.training_evidence:
        snapshot["training_run"] = native_training_snapshot(args.training_evidence)
    assert bool(args.training_render_report) == bool(args.training_quality)
    if args.training_quality:
        assert args.training_evidence
        snapshot["training_run"]["heldout_evaluation"] = native_heldout_snapshot(
            args.training_evidence,
            args.training_render_report,
            args.training_quality,
            quality["evaluator_id"],
            contract.axes,
        )
    assert bool(args.null_render_report) == bool(args.null_quality)
    if args.null_quality:
        assert args.training_quality
        snapshot["training_run"]["same_model_control"] = same_model_null_snapshot(
            args.training_render_report,
            args.null_render_report,
            args.null_quality,
            quality["evaluator_id"],
            contract.axes,
        )
    assert bool(args.teacher_render_report) == bool(args.teacher_quality)
    if args.teacher_quality:
        assert args.training_evidence and args.training_render_report
        snapshot["experiments"].append(
            teacher_reference_experiment(
                args.training_evidence,
                args.training_render_report,
                args.teacher_render_report,
                args.teacher_quality,
                quality["evaluator_id"],
            )
        )
    if args.redteam:
        snapshot["robustness"] = []
        for source in args.redteam:
            attack = read(source)
            if attack["schema"] == "forge/frozen-judge-probe/v1":
                assert attack["state"] == "COMPLETE"
                labels = {
                    "checker1": "One-pixel corruption · v2 evaluator",
                    "checker2": "Two-pixel corruption · v2 evaluator",
                    "image-instructions": "Instructions inside video frames",
                }
                kind = attack["attack"]
                assert kind in labels
                detected = (
                    attack["judgment"]["status"] == "OK"
                    and min(attack["minimum_axes"].values()) <= -0.25
                )
                assert detected == attack["detected"]
                snapshot["robustness"].append(
                    {
                        "label": labels[kind],
                        "status": "defect_detected" if detected else "judge_blind_spot_reproduced",
                        "phase": "current",
                        "corrupted_frames": 81 if kind == "image-instructions" else 78,
                        "total_frames": 81,
                        "detail_frames": len(attack["detail_frame_indices"]),
                        "detail_crops": len(attack["detail_crop_regions"]),
                        "evaluator_id": attack["evaluator_id"],
                        "raster_guard": attack["raster"]["status"],
                        "note": (
                            "Synthetic media probe with nine secret detail frames and three "
                            "matched enlarged crops. Both blinded orders were judged. "
                            + (
                                "The separate raster guard also blocked this corruption. "
                                if attack["raster"]["status"] == "ANOMALY"
                                else "The raster guard found no high-frequency anomaly. "
                            )
                            + "One detected case does not establish sensitivity or model safety."
                        ),
                    }
                )
                continue
            assert attack["schema"] == "forge/research-attack/v1"
            assert attack["kind"] == "native_frame_selection_overfit"
            assert type(attack["detected_visual_or_temporal_defect"]) is bool
            fixed = "detail_frame_indices" not in attack
            snapshot["robustness"].append(
                {
                    "label": "Fixed detail frames" if fixed else "Secret detail frames",
                    "phase": "historical",
                    "status": (
                        "defect_detected"
                        if attack["detected_visual_or_temporal_defect"]
                        else "judge_blind_spot_reproduced"
                    ),
                    "corrupted_frames": attack["corrupted_frames"],
                    "total_frames": pilot_contract.frames,
                    "detail_frames": 3 if fixed else len(attack["detail_frame_indices"]),
                    "evaluator_id": attack["evaluator_id"],
                    "note": (
                        "Synthetic one-pixel checkerboard corruption; two blinded judgments. "
                        "This tests the evaluator, not an adversarial model checkpoint."
                    ),
                }
            )
    shuffle = controls["summary"].get("shuffle")
    if shuffle and shuffle["n_prompts"] >= 32 and shuffle["rate_upper"] < 0.8:
        snapshot.setdefault("robustness", []).append(
            {
                "label": "Temporal order · completed control panel",
                "status": "judge_blind_spot_reproduced",
                "phase": "current",
                "evaluator_id": controls["evaluator_id"],
                "note": (
                    f"The frozen evaluator detected {shuffle['detections']} of "
                    f"{shuffle['n_prompts']} shuffled videos. The sensitivity upper "
                    "confidence bound is below the required 0.8. Quality promotion "
                    "remains blocked; spatial probe detections do not repair this failure."
                ),
            }
        )
    assert bool(args.temporal_probe) == bool(args.temporal_probe_plan)
    if args.temporal_probe:
        row = temporal_probe_row(args.temporal_probe, args.temporal_probe_plan)
        assert controls["evaluator_id"] != row["evaluator_id"], "controls must be historical"
        for old in snapshot.setdefault("robustness", []):
            assert old["evaluator_id"] != row["evaluator_id"], "probe phase would be mislabeled"
            old["phase"] = "historical"
        snapshot["robustness"].append(row)
        snapshot["controls_policy"].update(historical=True, evaluator_id=controls["evaluator_id"])
        snapshot["benchmark"]["quality_note"] = (
            "Quality preservation has not been demonstrated. The historical evaluator failed "
            "calibration; the updated v3 evaluator has limited development diagnostics "
            "and no independent Wan calibration. "
            "Public availability alone does not certify quality."
        )
    assert bool(args.temporal_replay) == bool(args.temporal_replay_plan)
    if args.temporal_replay:
        assert args.temporal_probe, "Wan replay accompanies the current constructed diagnostic"
        snapshot["robustness"].append(
            temporal_replay_row(args.temporal_replay, args.temporal_replay_plan, args.calibration)
        )
    chain_args = (args.testnet_model, args.testnet_pack, args.testnet_plan)
    if any(chain_args):
        assert all(chain_args), "model, pack and plan receipts are required together"
        model, pack, chain_plan = map(read, chain_args)
        assert model["status"] == pack["status"] == "FINALIZED"
        assert model["network"] == pack["network"] == chain_plan["network"] == "test"
        assert model["netuid"] == pack["netuid"] == chain_plan["netuid"]
        assert model["commitment_block"] <= chain_plan["close_block"]
        assert pack["commitment_block"] <= chain_plan["close_block"] < chain_plan["entropy_block"]
        assert any(
            item["manifest_hash"] == model["manifest_hash"] for item in chain_plan["admitted"]
        )
        snapshot["testnet"] = {
            "network": "test",
            "netuid": model["netuid"],
            "status": "finalized_commitments",
            "model_block": model["commitment_block"],
            "model_extrinsic": model["extrinsic_id"],
            "pack_block": pack["commitment_block"],
            "pack_extrinsic": pack["extrinsic_id"],
            "close_block": chain_plan["close_block"],
            "entropy_block": chain_plan["entropy_block"],
            "manifest_hash": model["manifest_hash"],
            "candidate_url": (
                "https://huggingface.co/"
                + model["manifest"]["reference"]["repo_id"]
                + "/tree/"
                + model["manifest"]["reference"]["revision"]
            ),
            "evaluation_record": None,
            "note": "Real testnet commitments. No certified winner or reward activation.",
        }
    if args.testnet_evaluation:
        from forge.chain import Sr25519Verifier, encode_evaluation_record_commitment
        from forge.protocol.evaluation_record import SignedEvaluationRecord, verify_record

        assert all(chain_args), "evaluation anchor requires model, pack and plan receipts"
        anchor = read(args.testnet_evaluation)
        assert anchor["status"] == "FINALIZED_EVALUATION_EVIDENCE_ONLY_NOT_A_CERTIFICATE_OR_REWARD"
        signed = SignedEvaluationRecord.from_dict(anchor["signed_record"])
        record = signed.record
        assert verify_record(signed, Sr25519Verifier())
        assert record.record_hash() == anchor["record_hash"]
        assert anchor["commitment"] == encode_evaluation_record_commitment(record.record_hash())
        assert record.network == "test" and record.netuid == snapshot["testnet"]["netuid"]
        assert record.plan_hash == hash_canonical("forge/evaluation-plan/v1", plan)
        for label, value in (("quality", quality), ("timing", timing), ("calibration", controls)):
            assert getattr(record, label + "_hash") == hash_canonical(
                f"forge/evaluation-{label}/v1", value
            )
        snapshot["testnet"]["status"] = "finalized_evaluation_evidence"
        snapshot["testnet"]["evaluation_record"] = {
            "status": "Evidence only; no rewards",
            "block": anchor["commitment_block"],
            "record_hash": record.record_hash(),
            "gate_status": record.gate_status,
            "validator_hotkey": signed.validator_hotkey,
        }
    if args.validators:
        assert "testnet" in snapshot, "validators accompany the testnet commitments"
        observed = read(args.validators)
        assert observed["schema"] == "forge/validator-snapshot/v1"
        assert observed["network"] == "test" and observed["netuid"] == snapshot["testnet"]["netuid"]
        signer = (snapshot["testnet"]["evaluation_record"] or {}).get("validator_hotkey")
        snapshot["validators"] = {
            "block": observed["block"],
            "observed_at": datetime.fromtimestamp(observed["observed_unix"], UTC).isoformat(),
            "neurons": observed["neurons"],
            "entries": [
                {
                    "uid": row["uid"],
                    "hotkey": row["hotkey"],
                    "stake": row["stake_rao"] / 1e9,
                    "last_update_block": row["last_update_block"],
                    "active": row["active"],
                    "anchored_evaluation": row["hotkey"] == signer,
                }
                for row in observed["validators"]
            ],
        }
    media_args = (
        args.media_selection,
        args.golden_panel,
        args.incumbent_panel,
        args.candidate_panel,
        args.media_out,
    )
    if any((args.media_selection, args.media_out, args.incumbent_panel, args.candidate_panel)):
        assert all(media_args), "selection, all three panels and media output are required together"
        snapshot["media_examples"] = public_examples(
            plan,
            quality,
            args.media_selection,
            {
                "golden": args.golden_panel,
                "incumbent": args.incumbent_panel,
                "candidate": args.candidate_panel,
            },
            args.media_out,
        )
    elif args.golden_panel:
        assert args.exploratory_report, "a standalone golden panel requires exploratory reports"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    with temporary.open("x") as handle:
        json.dump(snapshot, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(args.out)
    print(json.dumps({"output": str(args.out), "phase": "development", "winner": None}))
