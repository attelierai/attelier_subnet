"""Dispatch local admission, budget, calibration and evaluation commands."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from forge.evaluation.benchmark import generate_prompts
from forge.models.artifact_manifest import ArtifactError, ArtifactReference, ModelManifest
from forge.models.artifacts import fetch_verified, snapshot_manifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.protocol.prompt_pack import verified_plan_prompts


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _plan_contract(plan: dict, *, regenerate_prompts: bool) -> ChallengeContract:
    """Check a frozen plan before it controls GPU or promotion evidence."""
    contract = ChallengeContract.from_dict(plan["contract"])
    if plan.get("contract_hash") != contract.contract_hash():
        raise ValueError("panel plan contract hash mismatch")
    if regenerate_prompts:
        try:
            prescribed = verified_plan_prompts(plan, contract)
        except KeyError as error:
            raise ValueError("frozen plan lacks prompt-generation inputs") from error
        if json.loads(json.dumps(prescribed)) != plan.get("prompts"):
            raise ValueError("prompt panel differs from the frozen generation plan")
    return contract


def _cache_path(cache_dir: Path, manifest: ModelManifest) -> Path:
    return cache_dir / hash_canonical("forge/snapshot/v1", manifest.reference.to_dict())


PREFLIGHT_DISK_FLOOR_BYTES = 2 * 1024**3


def _preflight_disk_requirement(
    cache_dir: Path, manifests: tuple[ModelManifest, ...]
) -> tuple[int, int]:
    """Return free bytes required before fetching a closed public batch.

    The final cache holds each missing declared file once per immutable
    reference.  A Hugging Face fetch also writes one temporary file in that
    same filesystem before replacing its final path, so reserve the largest
    missing file once in addition to the final byte total.  This is a
    conservative capacity preflight, not an atomic reservation: other writers
    or provider quotas can still change the outcome after this check.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise ArtifactError("preflight cache path must be a directory")
    pending: dict[Path, int] = {}
    for manifest in manifests:
        root = _cache_path(cache_dir, manifest)
        for item in manifest.files:
            target = root / item.path
            if target.exists():
                continue
            previous = pending.get(target)
            pending[target] = max(item.bytes, previous or 0)
    pending_bytes = sum(pending.values())
    temporary_headroom = max(pending.values(), default=0)
    required = pending_bytes + temporary_headroom + PREFLIGHT_DISK_FLOOR_BYTES
    stats = os.statvfs(cache_dir)
    available = stats.f_bavail * stats.f_frsize
    return required, available


def _require_preflight_disk_capacity(cache_dir: Path, manifests: tuple[ModelManifest, ...]) -> None:
    required, available = _preflight_disk_requirement(cache_dir, manifests)
    if available < required:
        raise ArtifactError(
            "insufficient free space for closed preflight batch "
            f"(required={required} available={available})"
        )


def _require_admitted(plan: dict, manifests: tuple[ModelManifest, ...]) -> None:
    if "admitted" not in plan:
        return
    admitted = {item.get("manifest_hash") for item in plan["admitted"] if isinstance(item, dict)}
    if any(manifest.fingerprint() not in admitted for manifest in manifests):
        raise ValueError("public duel model was not admitted in the closed chain snapshot")


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command == "model-manifest":
        from forge.models.generation import GenerationRecipe

        reference = ArtifactReference(args.backend, args.repo, args.revision)
        if reference.backend != "hf":
            raise ValueError(
                "Hippius candidates supply forge-manifest.json; use resolve-submission"
            )
        recipe = GenerationRecipe.from_dict(
            {
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "scheduler": args.scheduler,
            }
        )
        base_ref = (
            ModelManifest.from_dict(_read(args.base_manifest)).reference
            if args.base_manifest
            else None
        )
        manifest = snapshot_manifest(
            reference, role=args.role, generation_config=recipe.to_dict(), base_reference=base_ref
        )
        _write(args.out, manifest.to_dict())
        print(json.dumps({"manifest_hash": manifest.fingerprint(), "files": len(manifest.files)}))
        return 0
    if args.command == "prepare-submission":
        from forge.validator.submission import encode_artifact_commitment, prepare_declaration

        manifest = ModelManifest.from_dict(_read(args.manifest))
        _write(args.out, prepare_declaration(manifest))
        print(
            json.dumps(
                {
                    "commitment_example": encode_artifact_commitment(manifest.reference),
                    "note": "upload declaration first; commit the resulting NEW immutable revision",
                }
            )
        )
        return 0
    if args.command == "public-round-plan":
        from forge.chain import BittensorChain
        from forge.validator.round_planner import build_public_round_plan

        contract = ChallengeContract.from_dict(_read(args.contract))
        chain = BittensorChain(args.network, args.netuid)
        try:
            plan = build_public_round_plan(
                chain,
                contract,
                args.close_block,
                args.entropy_block,
                prompt_pack=_read(args.prompt_pack) if args.prompt_pack else None,
                pack_hotkey=args.pack_hotkey,
                stage=args.stage,
                finalist_lock_block=args.finalist_lock_block,
                incumbent=_read(args.incumbent_state) if args.incumbent_state else None,
                candidate_hotkey=args.candidate_hotkey,
                finalist_lock=_read(args.finalist_lock) if args.finalist_lock else None,
                finalist_lock_hotkey=args.finalist_lock_hotkey,
                calibration_plan_hash=(
                    hash_canonical("forge/evaluation-plan/v1", _read(args.calibration_plan))
                    if args.calibration_plan
                    else None
                ),
            )
            _write(args.out, plan)
            print(
                json.dumps(
                    {
                        "status": plan["status"],
                        "admitted": len(plan["admitted"]),
                        "prompts": len(plan["prompts"]),
                    }
                )
            )
        finally:
            chain.close()
        return 0
    if args.command == "fetch-model":
        manifest = ModelManifest.from_dict(_read(args.manifest))
        from forge.models.artifacts import inspect_public_artifact

        inspect_public_artifact(manifest.reference)
        if manifest.base_reference:
            inspect_public_artifact(manifest.base_reference)
        path = fetch_verified(manifest, args.destination)
        print(
            json.dumps({"verified_directory": str(path), "manifest_hash": manifest.fingerprint()})
        )
        return 0
    if args.command == "preflight-submission":
        from forge.validator.runtime_preflight import (
            effective_candidate_hash,
            validate_public_candidate,
            verify_local,
        )

        if args.out.exists():
            raise ValueError("submission preflight output must be new")
        candidate = ModelManifest.from_dict(_read(args.manifest))
        base = ModelManifest.from_dict(_read(args.base_manifest))
        contract = ChallengeContract.from_dict(_read(args.contract))
        _require_preflight_disk_capacity(args.cache_dir, (base, candidate))
        base_path = _cache_path(args.cache_dir, base)
        candidate_path = _cache_path(args.cache_dir, candidate)
        fetch_verified(base, base_path)
        if candidate_path != base_path:
            fetch_verified(candidate, candidate_path)
        verify_local(base, base_path)
        weights = validate_public_candidate(candidate, candidate_path, base, base_path, contract)
        receipt = {
            "schema": "forge/submission-preflight/v1",
            "status": "PUBLIC_ARTIFACT_PREFLIGHT_PASSED",
            "admission_granted": False,
            "quality_evaluated": False,
            "duplicate_history_checked": False,
            "manifest_hash": candidate.fingerprint(),
            "base_manifest_hash": base.fingerprint(),
            "contract_hash": contract.contract_hash(),
            "weight_content_hash": weights.content_fingerprint,
            "effective_candidate_hash": effective_candidate_hash(
                candidate, weights, base, contract
            ),
        }
        _write(args.out, receipt)
        print(json.dumps(receipt))
        return 0
    if args.command == "preflight-round":
        from forge.validator.runtime_preflight import preflight_candidate_batch

        plan = _read(args.plan)
        contract = _plan_contract(plan, regenerate_prompts=True)
        base = ModelManifest.from_dict(_read(args.base_manifest))
        admitted = plan.get("admitted")
        if not isinstance(admitted, list) or len(admitted) > contract.max_candidates:
            raise ValueError(
                "round preflight requires closed public admissions within the candidate cap"
            )
        manifests = []
        admitted_hashes = set()
        for admission in admitted:
            if not isinstance(admission, dict):
                raise ValueError("round admission is invalid")
            block, hotkey = admission.get("block"), admission.get("hotkey")
            if type(block) is not int or block < 0 or not isinstance(hotkey, str) or not hotkey:
                raise ValueError("round admission identity is invalid")
            manifest = ModelManifest.from_dict(admission.get("manifest"))
            manifest_hash = manifest.fingerprint()
            if admission.get("manifest_hash") != manifest_hash or manifest_hash in admitted_hashes:
                raise ValueError("round admission manifest hash is invalid")
            admitted_hashes.add(manifest_hash)
            manifests.append(manifest)
        _require_preflight_disk_capacity(args.cache_dir, (base, *manifests))
        base_path = _cache_path(args.cache_dir, base)
        fetch_verified(base, base_path)
        candidates = []
        for manifest in manifests:
            candidate_path = _cache_path(args.cache_dir, manifest)
            try:
                if candidate_path != base_path:
                    fetch_verified(manifest, candidate_path)
            except ArtifactError:
                candidates.append((manifest, None))
            else:
                candidates.append((manifest, candidate_path))
        receipt = preflight_candidate_batch(plan, contract, base, base_path, candidates)
        _write(args.out, receipt)
        print(
            json.dumps(
                {
                    "candidates": len(receipt["candidates"]),
                    "unique": sum(
                        item["status"].endswith("PASSED") for item in receipt["candidates"]
                    ),
                    "output": str(args.out),
                }
            )
        )
        return 0
    if args.command == "resolve-submission":
        from forge.validator.submission import decode_artifact_commitment, resolve_submission

        reference = decode_artifact_commitment(args.commitment)
        if reference is None:
            raise ValueError("not a valid public model commitment")
        manifest = resolve_submission(reference)
        _write(args.out, manifest.to_dict())
        print(
            json.dumps(
                {"reference": reference.immutable_id, "manifest_hash": manifest.fingerprint()}
            )
        )
        return 0
    if args.command == "panel-plan":
        contract = ChallengeContract.from_dict(_read(args.contract))
        prompts = generate_prompts(contract, args.round_id, args.stage, args.block_hash)
        _write(
            args.out,
            {
                "schema": "forge/panel-plan/v1",
                "purpose": "evaluation",
                "contract": contract.to_dict(),
                "contract_hash": contract.contract_hash(),
                "round_id": args.round_id,
                "campaign_id": args.campaign_id or args.round_id,
                "stage": args.stage,
                "entropy": args.block_hash,
                "prompts": [asdict(prompt) for prompt in prompts],
            },
        )
        return 0
    if args.command == "render-panel":
        from forge.models.runtime import run_panel
        from forge.validator.runtime_preflight import validate_preflight_receipt

        plan = _read(args.plan)
        contract = _plan_contract(plan, regenerate_prompts=True)
        base = ModelManifest.from_dict(_read(args.base_manifest))
        candidate = ModelManifest.from_dict(_read(args.manifest))
        if (
            "admitted" in plan
            and not args.reference
            and not any(
                item["manifest_hash"] == candidate.fingerprint() for item in plan["admitted"]
            )
        ):
            raise ValueError("candidate was not admitted in the closed chain snapshot")
        if not args.reference and not args.experimental_adapter:
            if args.preflight is None:
                raise ValueError("public candidate rendering requires a runtime preflight receipt")
            receipt = validate_preflight_receipt(
                _read(args.preflight),
                plan,
                contract,
                candidate_manifest_hash=candidate.fingerprint(),
            )
            if receipt["base_manifest_hash"] != base.fingerprint():
                raise ValueError("runtime preflight base differs from render base")
        base_path = _cache_path(args.cache_dir, base)
        candidate_path = _cache_path(args.cache_dir, candidate)
        fetch_verified(base, base_path)
        if candidate_path != base_path:
            fetch_verified(candidate, candidate_path)
        report = run_panel(
            base=base,
            base_dir=base_path,
            candidate=candidate,
            candidate_dir=candidate_path,
            prompts=plan["prompts"],
            contract=contract,
            out=args.out,
            experimental_adapter=args.experimental_adapter,
            reference=args.reference,
            scheduler_kind=args.scheduler,
        )
        return 0 if all(item["status"] == "OK" for item in report["observations"]) else 2
    if args.command == "render-paired":
        from forge.models.paired_runtime import run_paired_panel
        from forge.validator.runtime_preflight import validate_preflight_receipt

        plan = _read(args.plan)
        stage = plan.get("stage")
        if stage not in {"screen", "screening", "duel", "confirmation"}:
            raise ValueError("frozen paired timing plan must declare a known stage")
        contract = _plan_contract(plan, regenerate_prompts=True)
        base = ModelManifest.from_dict(_read(args.base_manifest))
        incumbent = ModelManifest.from_dict(_read(args.incumbent_manifest))
        candidate = ModelManifest.from_dict(_read(args.manifest))
        if not args.experimental_adapter:
            _require_admitted(plan, (candidate,))
            if args.preflight is None:
                raise ValueError("public candidate timing requires a runtime preflight receipt")
            receipt = validate_preflight_receipt(
                _read(args.preflight),
                plan,
                contract,
                candidate_manifest_hash=candidate.fingerprint(),
            )
            if receipt["base_manifest_hash"] != base.fingerprint():
                raise ValueError("runtime preflight base differs from timing base")
        base_path = _cache_path(args.cache_dir, base)
        incumbent_path = _cache_path(args.cache_dir, incumbent)
        candidate_path = _cache_path(args.cache_dir, candidate)
        fetch_verified(base, base_path)
        if incumbent_path != base_path:
            fetch_verified(incumbent, incumbent_path)
        if candidate_path != base_path and candidate_path != incumbent_path:
            fetch_verified(candidate, candidate_path)
        reports = run_paired_panel(
            base=base,
            base_dir=base_path,
            reference_model=incumbent,
            reference_dir=incumbent_path,
            candidate=candidate,
            candidate_dir=candidate_path,
            prompts=plan["prompts"],
            contract=contract,
            out=args.out,
            experimental_adapter=args.experimental_adapter,
            timing_design=(
                "screening_ab" if stage in {"screen", "screening"} else "confirmation_abba"
            ),
            frozen_plan_id=(
                plan.get("plan_id") or hash_canonical("forge/frozen-paired-plan/v1", plan)
                if stage in {"screen", "screening"}
                else None
            ),
        )
        observations = [
            *reports["reference"]["observations"],
            *reports["candidate"]["observations"],
        ]
        return 0 if all(item["status"] == "OK" for item in observations) else 2
    if args.command == "vision-policy":
        from forge.evaluation.automatic_quality import default_policy

        policy = default_policy()
        _write(args.out, policy)
        print(json.dumps(policy, indent=2))
        return 0
    if args.command == "judge-panels":
        from forge.evaluation.automatic_quality import score_reports
        from forge.evaluation.vision_judge import ApiBudget

        evidence = score_reports(
            args.plan,
            args.golden,
            args.incumbent,
            args.challenger,
            args.out,
            ApiBudget(args.ledger, args.api_budget),
            args.env_file,
            workers=args.workers,
        )
        print(json.dumps({"status": evidence["status"], "output": str(args.out)}))
        return 0
    if args.command == "judge-controls":
        from forge.evaluation.vision_calibration import run_controls
        from forge.evaluation.vision_judge import ApiBudget

        calibration = run_controls(
            _read(args.plan),
            args.golden,
            args.out,
            budget=ApiBudget(args.ledger, args.api_budget),
            env_file=args.env_file,
            max_prompts=args.max_prompts,
            workers=args.workers,
        )
        print(
            json.dumps({"calibration_id": calibration["calibration_id"], "output": str(args.out)})
        )
        return 0
    if args.command == "compare-automatic":
        from forge.evaluation.automatic_gate import evaluate_automatic

        plan = _read(args.plan)
        contract = _plan_contract(plan, regenerate_prompts=not args.development)
        geometry = {
            "width": contract.width,
            "height": contract.height,
            "frames": contract.frames,
            "fps": contract.fps,
        }
        if "geometry" in plan and plan["geometry"] != geometry:
            raise ValueError("plan geometry differs from its contract")
        plan["geometry"] = geometry
        result = evaluate_automatic(
            plan,
            _read(args.policy),
            _read(args.quality),
            _read(args.timing),
            _read(args.calibration),
            preflight=_read(args.preflight) if args.preflight else None,
            development=args.development,
        )
        _write(args.out, result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PROMOTE_ELIGIBLE" else 2
    if args.command == "compare-panels":
        from forge.evaluation.speed_gate import compare_evidence

        plan, policy = _read(args.plan), _read(args.policy)
        contract = ChallengeContract.from_dict(plan["contract"])
        if plan["contract_hash"] != contract.contract_hash():
            raise ValueError("panel plan contract hash mismatch")
        reports = {
            "golden": _read(args.golden),
            "incumbent": _read(args.incumbent),
            "challenger": _read(args.challenger),
        }
        results = {}
        for role in ("golden", "incumbent"):
            results[role] = compare_evidence(
                reports[role],
                reports["challenger"],
                required_contract=contract,
                expected_prompts=plan["prompts"],
                expected_runtime_id=policy["runtime_id"],
                expected_geometry={
                    "height": contract.height,
                    "width": contract.width,
                    "frames": contract.frames,
                    "fps": contract.fps,
                },
                expected_evaluator_id=policy["evaluator_id"],
                min_speedup=contract.min_speedup,
                quality_margin=contract.axis_noninferiority_margin,
                alpha=contract.alpha / contract.max_candidates / 2,
            )
        eligible = all(item["status"] == "PROMOTE_ELIGIBLE" for item in results.values())
        result = {
            "status": "PROMOTE_ELIGIBLE" if eligible else "NOT_ELIGIBLE",
            "comparisons": results,
            "note": "evidence gate only; requires validator consensus before a crown",
        }
        _write(args.out, result)
        print(json.dumps(result, indent=2))
        return 0 if eligible else 2
    return None
