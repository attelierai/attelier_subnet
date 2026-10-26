"""Local MVP preparation and evidence replay; no spending or chain writes."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from forge.evaluation import (
    EvaluationPolicy,
    evaluate_mvp,
    fit_policy_from_development,
    freeze_evaluation_plan,
    freeze_policy,
    validate_calibration,
)
from forge.evaluation.calibration_campaign import freeze_calibration_campaign
from forge.evaluation.calibration_power import campaign_preflight
from forge.evaluation.panel_power import confirmation_power_preflight, probe_quality_information
from forge.evaluation.panel_replay import conditional_panel_replay
from forge.protocol.hashing import hash_canonical
from forge.validator.admission import AdmissionLedger


def _read(path: Path):
    return json.loads(path.read_text())


def _write_new(path: Path, payload: object) -> None:
    # Frozen plans/receipts cannot be silently replaced on rerun.
    import os

    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as stream:
        stream.write(encoded)


def add_commands(commands):
    policy = commands.add_parser(
        "mvp-policy", help="freeze an experimental bounded policy (not calibrated)"
    )
    policy.add_argument("--out", type=Path, required=True)
    fit = commands.add_parser(
        "mvp-fit-policy", help="fit and freeze thresholds using development only"
    )
    fit.add_argument("--development", type=Path, required=True)
    fit.add_argument("--provenance", type=Path, required=True)
    fit.add_argument("--out-dir", type=Path, required=True)
    calibrate = commands.add_parser(
        "mvp-calibrate", help="verify independent controls without retuning"
    )
    calibrate.add_argument("--policy", type=Path, required=True)
    calibrate.add_argument("--evidence", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    calibration_plan = commands.add_parser(
        "mvp-calibration-plan",
        help="freeze calibration controls and their work count; no GPU or API calls",
    )
    calibration_plan.add_argument("--campaign-label", required=True)
    calibration_plan.add_argument("--entropy", required=True)
    calibration_plan.add_argument("--source-bank", type=Path, required=True)
    calibration_plan.add_argument("--development-legitimate-samples", type=int, default=32)
    calibration_plan.add_argument("--development-degradation-samples", type=int, default=8)
    calibration_plan.add_argument("--out", type=Path, required=True)
    panel_power = commands.add_parser(
        "mvp-panel-power",
        help="report confirmation coverage and probe information without changing the policy",
    )
    panel_power.add_argument("--policy", type=Path, required=True)
    power_source = panel_power.add_mutually_exclusive_group(required=True)
    power_source.add_argument("--probe", type=Path)
    power_source.add_argument("--calibration", type=Path)
    panel_power.add_argument("--draws", type=int, default=64)
    panel_power.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser(
        "mvp-freeze", help="freeze separate screening and sealed confirmation packages"
    )
    for name in ("policy", "screening", "confirmation"):
        freeze.add_argument("--" + name, type=Path, required=True)
    for name in ("candidate-manifest-hash", "incumbent-manifest-hash", "wan50-manifest-hash"):
        freeze.add_argument("--" + name, required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)
    replay = commands.add_parser(
        "mvp-evaluate", help="replay local MVP evidence; never issues a certificate"
    )
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    dashboard = commands.add_parser(
        "mvp-dashboard", help="export local operational snapshot without future prompts"
    )
    dashboard.add_argument("--db", type=Path, required=True)
    dashboard.add_argument("--budget-db", type=Path)
    dashboard.add_argument("--candidate-id", type=int)
    dashboard.add_argument("--network", default="test")
    dashboard.add_argument("--netuid", type=int, required=True)
    dashboard.add_argument("--out", type=Path, required=True)
    queue = commands.add_parser("mvp-queue", help="read persisted MVP FIFO status")
    queue.add_argument("--db", type=Path, required=True)
    queue.add_argument("--network", default="test")
    queue.add_argument("--netuid", type=int, required=True)


def _disjoint(screen, confirmation):
    for field in ("prompt_id", "source_id", "concept_id"):
        left, right = ({row.get(field) for row in panel} for panel in (screen, confirmation))
        if None in left or None in right or "" in left or "" in right or left & right:
            raise ValueError(f"screening/confirmation require disjoint nonempty {field}")

    def normalize(text):
        return " ".join(text.casefold().split())

    texts = [{normalize(row["text"]) for row in panel} for panel in (screen, confirmation)]
    if "" in texts[0] or "" in texts[1] or texts[0] & texts[1]:
        raise ValueError("screening and confirmation repeat normalized prompt text")
    if any(type(row.get("seed")) is not int or row["seed"] < 0 for row in [*screen, *confirmation]):
        raise ValueError("each prompt requires one frozen nonnegative seed")


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command == "mvp-policy":
        _write_new(args.out, freeze_policy().to_dict())
        return 0
    if args.command == "mvp-fit-policy":
        policy, derivation = fit_policy_from_development(
            _read(args.development), evaluator_provenance=_read(args.provenance)
        )
        args.out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        _write_new(args.out_dir / "policy.json", policy.to_dict())
        _write_new(args.out_dir / "derivation.json", derivation)
        return 0
    if args.command == "mvp-calibrate":
        result = validate_calibration(
            _read(args.evidence), EvaluationPolicy.from_dict(_read(args.policy))
        )
        _write_new(args.out, result)
        print(json.dumps(result))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "mvp-calibration-plan":
        campaign = freeze_calibration_campaign(
            campaign_label=args.campaign_label,
            entropy=args.entropy,
            source_bank=_read(args.source_bank),
            development_legitimate_samples=args.development_legitimate_samples,
            development_degradation_samples=args.development_degradation_samples,
        )
        if campaign["schema"] != "forge/mvp-calibration-campaign/v3":
            raise ValueError("new calibration plans require preregistered category labels")
        _write_new(args.out, {**campaign, "preflight": campaign_preflight(campaign)})
        return 0
    if args.command == "mvp-panel-power":
        policy = EvaluationPolicy.from_dict(_read(args.policy))
        _write_new(
            args.out,
            {
                "schema": "forge/mvp-panel-power/v1",
                "policy_id": policy.policy_id,
                "confirmation": confirmation_power_preflight(quality_margin=policy.quality_margin),
                **(
                    {"probe": probe_quality_information(_read(args.probe))}
                    if args.probe
                    else {
                        "conditional_replay": conditional_panel_replay(
                            _read(args.calibration), policy, draws=args.draws
                        )
                    }
                ),
                "promotion_enabled": False,
            },
        )
        return 0
    if args.command == "mvp-freeze":
        policy = EvaluationPolicy.from_dict(_read(args.policy))
        screen, confirmation = _read(args.screening), _read(args.confirmation)
        _disjoint(screen, confirmation)
        bindings = {
            key: getattr(args, key)
            for key in ("candidate_manifest_hash", "incumbent_manifest_hash", "wan50_manifest_hash")
        }
        plans = {
            stage: freeze_evaluation_plan(stage=stage, prompts=panel, policy=policy, **bindings)
            for stage, panel in (("screening", screen), ("confirmation", confirmation))
        }
        args.out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        packages = {
            stage: {
                "schema": "forge/mvp-sealed-stage/v1",
                "salt": secrets.token_hex(32),
                "plan": plan,
            }
            for stage, plan in plans.items()
        }
        for stage, package in packages.items():
            _write_new(args.out_dir / f"{stage}.private.json", package)
        _write_new(
            args.out_dir / "commitment.json",
            {
                "schema": "forge/mvp-stage-commitments/v1",
                **bindings,
                "policy_id": policy.policy_id,
                "stages": {
                    stage: hash_canonical("forge/mvp-sealed-stage/v1", package)
                    for stage, package in packages.items()
                },
                "screening_prompts": 16,
                "confirmation_prompts": 128,
                "timing_prompts": 32,
                "publication_enabled": False,
            },
        )
        return 0
    if args.command == "mvp-evaluate":
        bundle = _read(args.bundle)
        gate = evaluate_mvp(
            plan=bundle["plan"],
            quality_rows=bundle["quality_rows"],
            timing_rows=bundle["timing_rows"],
            policy=EvaluationPolicy.from_dict(bundle["policy"]),
            calibration=bundle.get("calibration"),
            wan50_quality_rows=bundle.get("wan50_quality_rows"),
        )
        _write_new(
            args.out,
            {
                "schema": "forge/mvp-local-replay/v1",
                "bundle_hash": hash_canonical("forge/mvp-execution-bundle/v1", bundle),
                "gate": gate,
                "certified": False,
                "publication_enabled": False,
                "reason": "Local replay is not independent execution or certification.",
            },
        )
        print(json.dumps({"status": gate["status"], "certified": False}))
        return 0 if gate["status"] not in {"INCOMPLETE"} else 2
    if args.command == "mvp-dashboard":
        from forge.validator.dashboard import export_local_snapshot

        _write_new(
            args.out,
            export_local_snapshot(
                args.db,
                network=args.network,
                netuid=args.netuid,
                budget_database=args.budget_db,
                candidate_id=args.candidate_id,
            ),
        )
        return 0
    if args.command == "mvp-queue":
        from dataclasses import asdict

        if not args.db.is_file():
            raise ValueError("admission database is missing; empty state is not verified history")
        ledger = AdmissionLedger(args.db, network=args.network, netuid=args.netuid)
        print(
            json.dumps(
                {"schema": "forge/mvp-queue/v1", "entries": [asdict(row) for row in ledger.list()]},
                indent=2,
            )
        )
        return 0
    return None
