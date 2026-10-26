"""Validator-owned executor for one validator's assigned competitive MVP slots.

The worker never executes miner supplied code.  It consumes the frozen MVP and
allocation plans, reserves its local budget before CUDA work, renders only its
assigned slots, then signs raw VBench/Luna-backed partial evidence.  The caller
still has to exchange reports and replay the global gate before certification.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from forge.evaluation.distributed import signed_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan, assignments_for
from forge.evaluation.fusion import (
    evaluator_implementation_hash,
    fuse_bootstrap_alias_row,
    fuse_quality_row,
)
from forge.evaluation.mvp_evidence import _sha
from forge.evaluation.vbench import VBENCH_COMMIT, VBenchProvenance, validate_metric_evidence
from forge.evaluation.vbench_runner import run_vbench_isolated
from forge.evaluation.vision_judge import ApiBudget, judge_pair
from forge.evaluation.vision_media import (
    load_video,
    make_sheets,
    select_detail_crops,
    select_detail_frames,
    select_transition_frames,
)
from forge.models.artifact_manifest import ModelManifest
from forge.models.generation import GenerationRecipe, paired_reference_scheduler
from forge.models.paired_runtime import run_paired_panel
from forge.models.runtime import run_panel
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.protocol.ports import Signer
from forge.validator.budget import BudgetLedger
from forge.validator.mvp_runtime import MVPWork, timing_rows_from_paired_receipt


def runtime_identity(job: AssignedRuntime) -> str:
    """Stable evaluator/runtime identity; it intentionally excludes the candidate."""

    return _sha(
        {
            "runner_image_digest": job.vbench.runner_image_digest,
            "dependency_lock_hash": job.dependency_lock_hash,
            "vbench_commit": VBENCH_COMMIT,
            "vbench_weights": job.vbench.weight_sha256,
            "dino_revision": job.vbench.dino_revision,
            "policy_provenance": dict(job.policy_provenance),
        }
    )


@dataclass(frozen=True)
class VBenchConfig:
    checkout: Path
    cache: Path
    interpreter: Path
    weight_sha256: dict[str, str]
    dino_revision: str
    runner_image_digest: str


@dataclass(frozen=True)
class AssignedRuntime:
    """All validator-owned inputs for a single signed partial report."""

    plan: DistributedEvaluationPlan
    mvp_plan: Mapping[str, Any]
    work: MVPWork
    policy_provenance: Mapping[str, str]
    contract: ChallengeContract
    base: ModelManifest
    base_dir: Path
    candidate: ModelManifest
    candidate_dir: Path
    incumbent: ModelManifest
    incumbent_dir: Path
    wan50: ModelManifest
    wan50_dir: Path
    dependency_lock_hash: str
    vbench: VBenchConfig
    luna_budget: ApiBudget
    luna_env_file: Path | None
    budget: BudgetLedger
    reservation_id: int
    candidate_id: int
    operation_key: str
    minimum_reserved_usd: float


def _new_dir(path: Path) -> None:
    if path.exists():
        raise ValueError("assigned runtime output must be a new directory")
    path.mkdir(parents=True)


def _checkpoint(path: Path, binding_hash: str, value: Mapping[str, Any]) -> None:
    """Write immutable evidence for one expensive slot after its bytes exist."""
    if path.exists():
        raise ValueError("checkpoint already exists; do not overwrite evidence")
    _write_json_durable(
        path,
        {
            "schema": "forge/assigned-runtime-checkpoint/v1",
            "binding_hash": binding_hash,
            "value": value,
        },
    )


def _read_checkpoint(path: Path, binding_hash: str) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema", "binding_hash", "value"}
        or raw.get("schema") != "forge/assigned-runtime-checkpoint/v1"
        or raw.get("binding_hash") != binding_hash
        or not isinstance(raw.get("value"), Mapping)
    ):
        raise ValueError("runtime checkpoint is incomplete or bound to different work")
    return raw["value"]


def _job_binding(job: AssignedRuntime, signer: Signer) -> str:
    return hash_canonical(
        "forge/assigned-runtime-binding/v1",
        {
            "plan": job.plan.plan_hash(),
            "mvp_plan": job.mvp_plan,
            "work": job.work.receipt(),
            "contract": job.contract.contract_hash(),
            "validator": signer.validator_hotkey,
            "operation_key": job.operation_key,
            "models": {
                "base": job.base.fingerprint(),
                "candidate": job.candidate.fingerprint(),
                "incumbent": job.incumbent.fingerprint(),
                "wan50": job.wan50.fingerprint(),
                "base_dir": str(job.base_dir.resolve()),
                "candidate_dir": str(job.candidate_dir.resolve()),
                "incumbent_dir": str(job.incumbent_dir.resolve()),
                "wan50_dir": str(job.wan50_dir.resolve()),
            },
            "evaluator": {
                "policy_provenance": dict(job.policy_provenance),
                "dependency_lock_hash": job.dependency_lock_hash,
                "vbench": {
                    "checkout": str(job.vbench.checkout.resolve()),
                    "cache": str(job.vbench.cache.resolve()),
                    "interpreter": str(job.vbench.interpreter.resolve()),
                    "weight_sha256": job.vbench.weight_sha256,
                    "dino_revision": job.vbench.dino_revision,
                    "runner_image_digest": job.vbench.runner_image_digest,
                },
            },
        },
    )


@contextmanager
def _exclusive_gpu() -> Any:
    """Serialize one validator host's CUDA work, including VBench scoring.

    Competitive timings are invalid if a second worker shares the device.  The
    lock key follows CUDA_VISIBLE_DEVICES, so separate validators on separate
    hosts never contend while two local worker processes using the same visible
    GPU fail before rendering anything.
    """
    import fcntl

    visible = hash_canonical(
        "forge/validator-gpu-lock/v1", os.environ.get("CUDA_VISIBLE_DEVICES", "default")
    )
    directory = Path(os.environ.get("FORGE_GPU_LOCK_DIR", tempfile.gettempdir()))
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"forge-validator-gpu-{visible}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("validator GPU is already executing another assignment") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _require_budget(job: AssignedRuntime) -> None:
    """Re-read the durable reservation immediately before costly work."""

    job.budget.require_active(
        job.reservation_id,
        candidate_id=job.candidate_id,
        operation_key=job.operation_key,
        minimum_usd=job.minimum_reserved_usd,
    )


def _assert_no_unsettled_luna_call(budget: ApiBudget) -> None:
    """A prior timeout may have charged; a new assignment must not retry it."""

    if not budget.path.exists():
        return
    raw = json.loads(budget.path.read_text())
    calls = raw.get("calls") if isinstance(raw, Mapping) else None
    if not isinstance(calls, list):
        raise ValueError("Luna budget ledger is malformed; reconcile before dispatch")
    if any(isinstance(call, Mapping) and call.get("state") == "RESERVED" for call in calls):
        raise ValueError("Luna charge is unsettled; reconcile before another evaluation")


def _write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    """Persist slot bindings before a fusion/restart can consume their receipts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _observation_paths(report: Mapping[str, Any], root: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for row in report.get("observations", []):
        if not isinstance(row, Mapping) or row.get("status") != "OK":
            raise RuntimeError("render failed; retain terminal evidence and do not sign a report")
        identity = (row.get("prompt_id"), row.get("seed"))
        if not isinstance(identity[0], str) or type(identity[1]) is not int or identity in result:
            raise ValueError("render receipt lacks a unique prompt slot")
        path = root / str(row.get("video"))
        if not path.is_file() or sha256_file(path) != row.get("video_sha256"):
            raise ValueError("rendered video bytes differ from the runtime receipt")
        result[identity] = path
    return result


def _journal_verifier(
    journal: Path, bindings: Mapping[str, Mapping[str, Any]]
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool]:
    """Bind a Luna response to its persisted provider journal and frozen slot."""
    rows: dict[str, Mapping[str, Any]] = {}
    if journal.is_file():
        for line in journal.read_text().splitlines():
            item = json.loads(line)
            if isinstance(item, Mapping) and isinstance(item.get("response_id"), str):
                rows[item["response_id"]] = item

    def verify(call: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        response_id = call.get("response_id")
        binding = bindings.get(response_id) if isinstance(response_id, str) else None
        entry = rows.get(response_id) if isinstance(response_id, str) else None
        if not isinstance(binding, Mapping) or not isinstance(entry, Mapping):
            return False
        # The provider journal authenticates the returned response and input sheets;
        # our immutable sidecar binds those sheets to this exact frozen comparison.
        return (
            binding.get("context") == dict(context)
            and binding.get("request_image_hash") == call.get("request_image_hash")
            and entry.get("response_id") == response_id
            and entry.get("model") == call.get("model")
            and entry.get("request_image_hash") == call.get("request_image_hash")
            and isinstance(entry.get("sanitized_answer"), Mapping)
            and entry["sanitized_answer"].get("axes")
            == (call.get("answer") if isinstance(call.get("answer"), Mapping) else {}).get("axes")
            and entry["sanitized_answer"].get("uncertain_axes") == []
            and entry.get("settlement_error") is None
        )

    return verify


def _sheets(video: Path, geometry: Mapping[str, Any]) -> list[bytes]:
    media = load_video(video, geometry)
    detail = select_detail_frames(len(media.frames))
    return make_sheets(
        media.frames,
        geometry["fps"],
        detail_indices=detail,
        detail_crops=select_detail_crops(media.frames, detail),
        transition_indices=select_transition_frames(len(media.frames)),
    )


def _assert_bindings(
    job: AssignedRuntime, signer: Signer
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan = job.plan
    if plan.stage != "confirmation" or signer.validator_hotkey not in plan.committee:
        raise ValueError("only a committee validator can execute confirmation slots")
    mvp_body = {key: value for key, value in job.mvp_plan.items() if key != "plan_id"}
    if (
        job.mvp_plan.get("schema") != "forge/mvp-evaluation-plan/v1"
        or job.mvp_plan.get("plan_id") != _sha(mvp_body)
        or plan.chain_context.get("mvp_plan_id") != job.mvp_plan.get("plan_id")
    ):
        raise ValueError("distributed runtime requires a canonical chain-bound MVP plan")
    if job.work.stage != "confirmation" or job.work.plan_id != job.mvp_plan.get("plan_id"):
        raise ValueError("runtime work is not the frozen confirmation plan")
    frozen_prompts = job.mvp_plan.get("prompts")
    if not isinstance(frozen_prompts, list) or frozen_prompts != [
        dict(row) for row in plan.prompts
    ]:
        raise ValueError("distributed plan and MVP prompt bytes differ")
    if (
        job.mvp_plan.get("stage") != "confirmation"
        or job.mvp_plan.get("policy_id") != plan.policy_id
        or job.mvp_plan.get("candidate_manifest_hash") != job.candidate.fingerprint()
        or job.mvp_plan.get("incumbent_manifest_hash") != job.incumbent.fingerprint()
        or job.mvp_plan.get("wan50_manifest_hash") != job.wan50.fingerprint()
    ):
        raise ValueError("frozen MVP plan differs from validator-owned models or policy")
    if (
        job.work.candidate_manifest_hash != job.candidate.fingerprint()
        or job.work.incumbent_manifest_hash != job.incumbent.fingerprint()
        or job.work.wan50_manifest_hash != job.wan50.fingerprint()
        or job.work.contract_hash != job.contract.contract_hash()
    ):
        raise ValueError("runtime work differs from validator-owned model or contract bindings")
    if (
        plan.policy_id != job.work.policy_id
        or plan.candidate_manifest_hash != job.candidate.fingerprint()
    ):
        raise ValueError("distributed plan differs from validator-owned candidate work")
    if (
        plan.king_manifest_hash != job.incumbent.fingerprint()
        or plan.wan50_manifest_hash != job.wan50.fingerprint()
    ):
        raise ValueError("distributed references differ from validator-owned work")
    expected_runtime = {
        row["prompt_id"]: {
            "prompt_id": row["prompt_id"],
            "text": row["text"],
            "video_seeds": [row["seed"]],
        }
        for row in plan.prompts
    }
    quality_work = {row["prompt_id"]: row for row in job.work.quality_prompts}
    if (
        len(job.work.quality_prompts) != len(expected_runtime)
        or len(quality_work) != len(job.work.quality_prompts)
        or quality_work != expected_runtime
    ):
        raise ValueError("distributed plan and frozen MVP quality panel differ")
    expected_timing = {
        prompt_id: expected_runtime[prompt_id] for prompt_id in plan.timing_prompt_ids
    }
    timing_work = {row["prompt_id"]: row for row in job.work.timing_prompts}
    if (
        len(job.work.timing_prompts) != len(expected_timing)
        or len(timing_work) != len(job.work.timing_prompts)
        or timing_work != expected_timing
    ):
        raise ValueError("distributed plan and frozen MVP timing panel differ")
    if job.policy_provenance.get("source_implementation_hash") != evaluator_implementation_hash():
        raise ValueError("frozen evaluator provenance differs from the executing implementation")
    expected_operation = f"distributed-mvp:{plan.plan_hash()}:{signer.validator_hotkey}"
    if job.operation_key != expected_operation:
        raise ValueError("runtime reservation key is not bound to this validator assignment")
    _require_budget(job)
    _assert_no_unsettled_luna_call(job.luna_budget)
    # Use the chain-authenticated prompt rows so protected categories and the
    # original seed survive into fusion; runtime only wants video_seeds added.
    by_id = {row["prompt_id"]: {**dict(row), "video_seeds": [row["seed"]]} for row in plan.prompts}
    timing_by_id = {row["prompt_id"]: by_id[row["prompt_id"]] for row in job.work.timing_prompts}
    assigned = assignments_for(plan, signer.validator_hotkey)
    # A validator can receive an audit replica only once.  Defend against a
    # malformed allocation repeating a local slot: one signed report has one
    # observation per field and frozen prompt.
    quality = list(
        {
            item.prompt_id: by_id[item.prompt_id] for item in assigned if item.kind == "quality"
        }.values()
    )
    timing = list(
        {
            item.prompt_id: timing_by_id[item.prompt_id]
            for item in assigned
            if item.kind == "timing"
        }.values()
    )
    if not quality and not timing:
        raise ValueError("validator has no assigned work")
    return quality, timing


def execute_assigned_confirmation(
    job: AssignedRuntime,
    signer: Signer,
    *,
    out: Path,
    panel_runner: Callable[..., dict[str, Any]] = run_panel,
    paired_runner: Callable[..., dict[str, Any]] = run_paired_panel,
    vbench_runner: Callable[..., dict[str, Any]] = run_vbench_isolated,
    luna_judge: Callable[..., dict[str, Any]] = judge_pair,
) -> dict[str, Any]:
    """Run assigned slots and persist a signed report; no network/chain publication.

    Dependency injection exists solely to exercise the real file/signature boundary
    locally.  Production uses the pinned VBench and Luna functions above.
    """
    quality_prompts, timing_prompts = _assert_bindings(job, signer)
    VBenchProvenance(
        VBENCH_COMMIT, job.vbench.weight_sha256, job.vbench.runner_image_digest
    ).validate()
    with _exclusive_gpu():
        return _execute_locked(
            job,
            signer,
            out=out,
            quality_prompts=quality_prompts,
            timing_prompts=timing_prompts,
            panel_runner=panel_runner,
            paired_runner=paired_runner,
            vbench_runner=vbench_runner,
            luna_judge=luna_judge,
        )


def _execute_locked(
    job: AssignedRuntime,
    signer: Signer,
    *,
    out: Path,
    quality_prompts: list[dict[str, Any]],
    timing_prompts: list[dict[str, Any]],
    panel_runner: Callable[..., dict[str, Any]],
    paired_runner: Callable[..., dict[str, Any]],
    vbench_runner: Callable[..., dict[str, Any]],
    luna_judge: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Execute after admission and exclusive-GPU acquisition."""
    binding_hash = _job_binding(job, signer)
    binding_path = out / "job-binding.json"
    if out.exists():
        existing = _read_checkpoint(binding_path, binding_hash)
        if existing is None:
            raise ValueError("existing runtime output lacks an authenticated job binding")
    else:
        _new_dir(out)
        _checkpoint(binding_path, binding_hash, {"plan_hash": job.plan.plan_hash()})

    def completed_role(label: str, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        checkpoint = out / "checkpoints" / f"{label}.json"
        saved = _read_checkpoint(checkpoint, binding_hash)
        if saved is not None:
            report = saved.get("report")
            if not isinstance(report, Mapping):
                raise ValueError("completed render checkpoint lacks its full receipt")
            # Re-hash every referenced byte before reusing it.
            _observation_paths(report, out / label)
            return dict(report)
        if (out / label).exists():
            raise ValueError("partial render evidence is INCOMPLETE; do not regenerate it")
        _require_budget(job)
        report = run()
        _observation_paths(report, out / label)
        _checkpoint(checkpoint, binding_hash, {"report": report})
        return report

    geometry = {
        "width": job.contract.width,
        "height": job.contract.height,
        "frames": job.contract.frames,
        "fps": job.contract.fps,
    }
    candidate_report = wan_report = incumbent_report = None
    if quality_prompts:
        candidate_report = completed_role(
            "candidate",
            lambda: panel_runner(
                base=job.base,
                base_dir=job.base_dir,
                candidate=job.candidate,
                candidate_dir=job.candidate_dir,
                prompts=quality_prompts,
                contract=job.contract,
                out=out / "candidate",
            ),
        )
        wan_report = completed_role(
            "wan50",
            lambda: panel_runner(
                base=job.base,
                base_dir=job.base_dir,
                candidate=job.wan50,
                candidate_dir=job.wan50_dir,
                prompts=quality_prompts,
                contract=job.contract,
                out=out / "wan50",
                reference=True,
                scheduler_kind="original_unipc",
            ),
        )
        if job.incumbent.fingerprint() != job.wan50.fingerprint():
            incumbent_report = completed_role(
                "incumbent",
                lambda: panel_runner(
                    base=job.base,
                    base_dir=job.base_dir,
                    candidate=job.incumbent,
                    candidate_dir=job.incumbent_dir,
                    prompts=quality_prompts,
                    contract=job.contract,
                    out=out / "incumbent",
                    reference=(
                        job.incumbent.role == "pipeline"
                        and job.incumbent.generation_config["steps"] == job.contract.reference_steps
                    ),
                    scheduler_kind=paired_reference_scheduler(
                        job.incumbent,
                        job.base,
                        GenerationRecipe.from_dict(dict(job.incumbent.generation_config)),
                        job.contract,
                    ),
                ),
            )
        else:
            incumbent_report = wan_report
    timing_rows: list[dict[str, Any]] = []
    if timing_prompts:
        checkpoint = out / "checkpoints" / "timing.json"
        saved = _read_checkpoint(checkpoint, binding_hash)
        scoped = replace(job.work, timing_prompts=tuple(timing_prompts))
        if saved is not None:
            paired = saved.get("paired")
            if not isinstance(paired, Mapping):
                raise ValueError("completed timing checkpoint lacks its full receipt")
        else:
            if (out / "timing").exists():
                raise ValueError("partial timing evidence is INCOMPLETE; do not regenerate it")
            _require_budget(job)
            paired = paired_runner(
                base=job.base,
                base_dir=job.base_dir,
                reference_model=job.incumbent,
                reference_dir=job.incumbent_dir,
                candidate=job.candidate,
                candidate_dir=job.candidate_dir,
                prompts=timing_prompts,
                contract=job.contract,
                out=out / "timing",
            )
            _checkpoint(checkpoint, binding_hash, {"paired": paired})
        timing_rows = timing_rows_from_paired_receipt(
            scoped, paired["paired_timing"], dependency_lock_hash=job.dependency_lock_hash
        )

    bindings: dict[str, Mapping[str, Any]] = {}
    bindings_path = out / "luna-bindings.json"
    used: set[str] = set()
    quality_king: list[dict[str, Any]] = []
    quality_wan: list[dict[str, Any]] = []
    if quality_prompts:
        candidate_videos = _observation_paths(candidate_report, out / "candidate")
        wan_videos = _observation_paths(wan_report, out / "wan50")
        incumbent_videos = _observation_paths(
            incumbent_report, out / ("wan50" if incumbent_report is wan_report else "incumbent")
        )
        metric_cache: dict[tuple[Path, str, str, int], dict[str, Any]] = {}

        def metric(
            video: Path, prompt: Mapping[str, Any], manifest: ModelManifest
        ) -> dict[str, Any]:
            key = (video, prompt["text"], manifest.fingerprint(), prompt["seed"])
            if key not in metric_cache:
                metric_id = hash_canonical(
                    "forge/assigned-runtime-metric/v1", str(video), manifest.fingerprint()
                )
                checkpoint = out / "checkpoints" / "metrics" / f"{metric_id}.json"
                metric_out = out / "vbench" / metric_id
                saved = _read_checkpoint(checkpoint, binding_hash)
                if saved is not None:
                    receipt = saved.get("receipt")
                    raw_path = metric_out / "metrics_eval_results.json"
                    if (
                        not isinstance(receipt, Mapping)
                        or receipt.get("video_sha256") != sha256_file(video)
                        or receipt.get("plan_id") != job.work.plan_id
                        or receipt.get("prompt_id") != prompt["prompt_id"]
                        or receipt.get("seed") != prompt["seed"]
                        or receipt.get("manifest_hash") != manifest.fingerprint()
                        or receipt.get("prompt_hash")
                        != hash_canonical("forge/mvp-prompt/v1", prompt["text"])
                        or not raw_path.is_file()
                        or receipt.get("raw_sha256") != sha256_file(raw_path)
                    ):
                        raise ValueError(
                            "metric checkpoint does not bind complete raw VBench evidence"
                        )
                    validate_metric_evidence(receipt)
                    metric_cache[key] = dict(receipt)
                    return metric_cache[key]
                if metric_out.exists():
                    raise ValueError("partial VBench evidence is INCOMPLETE; do not regenerate it")
                _require_budget(job)
                metric_cache[key] = vbench_runner(
                    video=video,
                    prompt=prompt["text"],
                    output=metric_out,
                    checkout=job.vbench.checkout,
                    cache=job.vbench.cache,
                    interpreter=job.vbench.interpreter,
                    weight_sha256=job.vbench.weight_sha256,
                    dino_revision=job.vbench.dino_revision,
                    runner_image_digest=job.vbench.runner_image_digest,
                    plan_id=job.work.plan_id,
                    prompt_id=prompt["prompt_id"],
                    seed=prompt["seed"],
                    manifest_hash=manifest.fingerprint(),
                )
                _checkpoint(checkpoint, binding_hash, {"receipt": metric_cache[key]})
            return metric_cache[key]

        frozen_by_id = {row["prompt_id"]: row for row in job.mvp_plan["prompts"]}

        def fused(
            runtime_prompt: Mapping[str, Any],
            frozen_prompt: Mapping[str, Any],
            role: str,
            reference_video: Path,
            reference: ModelManifest,
            bootstrap: bool = False,
        ) -> dict[str, Any]:
            candidate_video = candidate_videos[
                (runtime_prompt["prompt_id"], runtime_prompt["video_seeds"][0])
            ]
            _require_budget(job)
            _assert_no_unsettled_luna_call(job.luna_budget)
            result = luna_judge(
                frozen_prompt["text"],
                _sheets(reference_video, geometry),
                _sheets(candidate_video, geometry),
                budget=job.luna_budget,
                env_file=job.luna_env_file,
                pair_id=(
                    f"{job.operation_key}:{role}:{frozen_prompt['prompt_id']}:"
                    f"{frozen_prompt['seed']}"
                ),
            )
            context_base = {
                "plan_id": job.work.plan_id,
                "prompt_id": frozen_prompt["prompt_id"],
                "seed": frozen_prompt["seed"],
                "prompt_hash": hash_canonical("forge/mvp-prompt/v1", frozen_prompt["text"]),
                "reference_role": role,
                "reference_manifest_hash": reference.fingerprint(),
                "candidate_manifest_hash": job.candidate.fingerprint(),
                "reference_video_sha256": sha256_file(reference_video),
                "candidate_video_sha256": sha256_file(candidate_video),
            }
            for call in result.get("calls", []):
                bindings[str(call["response_id"])] = {
                    "context": {**context_base, "response_id": call["response_id"]},
                    "request_image_hash": call.get("request_image_hash"),
                }
            _write_json_durable(bindings_path, bindings)
            kwargs = dict(
                plan=job.mvp_plan,
                prompt=frozen_prompt,
                reference_role=role,
                reference_video_sha256=sha256_file(reference_video),
                candidate_video_sha256=sha256_file(candidate_video),
                luna=result,
                reference_metrics=metric(reference_video, frozen_prompt, reference),
                candidate_metrics=metric(candidate_video, frozen_prompt, job.candidate),
                evaluator_provenance=job.policy_provenance,
                verify_journal=_journal_verifier(
                    job.luna_budget.path.with_suffix(".responses.jsonl"), bindings
                ),
                used_response_ids=used,
            )
            return fuse_bootstrap_alias_row(**kwargs) if bootstrap else fuse_quality_row(**kwargs)

        for prompt in quality_prompts:
            frozen_prompt = frozen_by_id[prompt["prompt_id"]]
            wan = fused(
                prompt,
                frozen_prompt,
                "wan50",
                wan_videos[(prompt["prompt_id"], prompt["video_seeds"][0])],
                job.wan50,
                job.incumbent.fingerprint() == job.wan50.fingerprint(),
            )
            quality_wan.append(wan)
            quality_king.append(
                wan
                if job.incumbent.fingerprint() == job.wan50.fingerprint()
                else fused(
                    prompt,
                    frozen_prompt,
                    "incumbent",
                    incumbent_videos[(prompt["prompt_id"], prompt["video_seeds"][0])],
                    job.incumbent,
                )
            )
    _write_json_durable(bindings_path, bindings)
    identity = runtime_identity(job)
    for row in [*quality_king, *quality_wan, *timing_rows]:
        row["runtime_id"] = identity
    report = signed_partial_report(
        job.plan,
        signer,
        quality_vs_king=quality_king,
        quality_vs_wan50=quality_wan,
        timing=timing_rows,
    )
    (out / "partial-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return {
        "schema": "forge/assigned-mvp-execution/v1",
        "plan_hash": job.plan.plan_hash(),
        "validator_hotkey": signer.validator_hotkey,
        "assignment_count": len(quality_prompts) + len(timing_prompts),
        "partial_report": report,
        "output_hash": hash_canonical("forge/assigned-mvp-execution/v1", report, bindings),
    }
