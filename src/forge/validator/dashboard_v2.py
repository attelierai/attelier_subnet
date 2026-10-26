"""Public `forge/dashboard/v2` snapshot from distributed-evaluation round artifacts.

Inputs are explicit files: frozen plans, model publications, signed partial reports,
resolved aggregates (`distributed-aggregate` or `distributed-batch-resolve` output),
anchor receipts and worker render receipts. The snapshot never carries coldkeys,
peer URLs, filesystem paths, salts or signatures. Absolute seconds are shown only
per video with the GPU that produced them; speed is always a same-GPU ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.evaluation.distributed_plan import DistributedEvaluationPlan, allocate_work
from forge.evaluation.mvp_policy import QUALITY_AXES
from forge.models.artifact_manifest import ModelManifest
from forge.models.distributed_worker import RECEIPT_SCHEMA
from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.protocol.distributed_batch import DiagnosticBatch
from forge.protocol.distributed_transport import report_digest
from forge.protocol.hashing import sha256_file

SCHEMA = "forge/dashboard/v2"
CONSENSUS_RULE = "stake_majority_v2"
BASE_MODEL_NAME = "Wan2.1 T2V 1.3B"
RAO_PER_TAO = 1_000_000_000
DIAGNOSTIC_REASON = "Diagnostic round: no calibration, no rewards"
# Distributed/MVP decisions mapped onto the dashboard vocabulary.
DECISIONS = {
    "DIAGNOSTIC": "DIAGNOSTIC",
    "PROMOTE_ELIGIBLE": "PROMOTE",
    "FAIL": "FAIL",
    "SCREEN_FAIL": "FAIL",
    "INCONCLUSIVE": "KEEP",
    "INCOMPLETE": "INCOMPLETE",
    "SCREEN_PASS": "INCOMPLETE",  # screening passed; confirmation still pending
}
CHALLENGER_STATUS = {
    "DIAGNOSTIC": "diagnostic",
    "PROMOTE": "promoted",
    "KEEP": "kept",
    "FAIL": "failed",
    "INCOMPLETE": "incomplete",
}


def decode_floats(value: Any) -> Any:
    """Undo `forge.protocol.hashing` tagged floats (`{"__forge_float__": repr}`)."""
    if isinstance(value, dict):
        if set(value) == {"__forge_float__"}:
            return float(value["__forge_float__"])
        return {key: decode_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_floats(item) for item in value]
    return value


def _read(path: Path, *, decode: bool = True) -> dict:
    value = json.loads(Path(path).read_text())
    value = decode_floats(value) if decode else value
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {Path(path).name}")
    return value


def _tao(rao: int) -> float:
    return rao / RAO_PER_TAO


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _job(validator: str, plan: DistributedEvaluationPlan, prompt, manifest_hash: str) -> tuple:
    return (
        validator,
        plan.round_id,
        prompt["prompt_id"],
        prompt["seed"],
        prompt["text"],
        manifest_hash,
    )


def _load_receipts(sources: list[Path]) -> dict[tuple, tuple[Path, dict]]:
    """Completed render receipts keyed by the exact job they bind."""
    files: list[Path] = []
    for source in sources:
        files.extend(sorted(source.glob("*.json")) if source.is_dir() else [source])
    receipts: dict[tuple, tuple[Path, dict]] = {}
    for path in files:
        r = _read(path)
        if r.get("schema") != RECEIPT_SCHEMA or r.get("status") != "OK":
            continue
        job = r["job"]
        key = (
            r["validator_hotkey"],
            r["round_id"],
            job["prompt_id"],
            job["seed"],
            job["prompt"],
            job["model_manifest_hash"],
        )
        if key in receipts and receipts[key][1]["video_sha256"] != r["video_sha256"]:
            raise ValueError("conflicting render receipts for one job")
        receipts.setdefault(key, (path, r))
    return receipts


def _poster(video: Path, geometry: dict) -> bytes:
    import imageio_ffmpeg
    from PIL import Image

    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        if tuple(metadata["size"]) != (geometry["width"], geometry["height"]):
            raise ValueError("video geometry differs from its render receipt")
        frame = next(reader)
    finally:
        reader.close()
    encoded = io.BytesIO()
    Image.frombytes("RGB", metadata["size"], frame).save(encoded, format="JPEG", quality=90)
    return encoded.getvalue()


def _place(target: Path, digest: str, write) -> None:
    """Content-addressed copy; an existing file must already hold exactly these bytes."""
    if target.is_symlink():
        raise ValueError("refusing a symlinked media target")
    if target.exists():
        if sha256_file(target) != digest:
            raise ValueError(f"refusing to overwrite media with different bytes: {target.name}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        write(handle)
    if sha256_file(temporary) != digest:
        temporary.unlink()
        raise ValueError("copied media does not match its digest")
    temporary.replace(target)


class _Media:
    def __init__(self, media_dirs: list[Path], media_out: Path):
        self.media_dirs, self.media_out = media_dirs, media_out
        media_out.mkdir(parents=True, exist_ok=True)

    def publish(self, receipt_path: Path, receipt: dict) -> dict:
        digest, name = receipt["video_sha256"], Path(receipt["video"]).name
        candidates = [receipt_path.parent / name]
        for folder in self.media_dirs:
            candidates += [folder / name, folder / f"{digest}.mp4"]
        source = next(
            (
                p
                for p in candidates
                if p.is_file() and not p.is_symlink() and sha256_file(p) == digest
            ),
            None,
        )
        if source is None:
            raise ValueError(f"no local video matches receipt hash {digest}")

        def copy(handle):
            with source.open("rb") as src:
                shutil.copyfileobj(src, handle)

        _place(self.media_out / f"{digest}.mp4", digest, copy)
        poster = _poster(source, receipt["geometry"])
        poster_digest = hashlib.sha256(poster).hexdigest()
        _place(self.media_out / f"{poster_digest}.jpg", poster_digest, lambda h: h.write(poster))
        return {
            "sha256": digest,
            "src": f"/media/{digest}.mp4",
            "poster": f"/media/{poster_digest}.jpg",
            "seconds": round(float(receipt["full_pipeline_seconds"]), 2),
            "gpu": receipt["runtime"]["gpu_name"],
        }


def _aggregates(resolved: list[dict]) -> dict[str, tuple[dict, dict]]:
    """plan_hash -> (aggregate, enclosing document) for single or batch results."""
    found: dict[str, tuple[dict, dict]] = {}
    for document in resolved:
        items = (
            document["aggregates"].values()
            if document.get("schema") == "forge/diagnostic-batch-result/v1"
            else [document]
        )
        for aggregate in items:
            if aggregate.get("schema") != "forge/distributed-evaluation-aggregate/v1":
                raise ValueError("resolved input is not a distributed aggregate")
            if aggregate["plan_hash"] in found:
                raise ValueError("two resolved aggregates for one plan")
            found[aggregate["plan_hash"]] = (aggregate, document)
    return found


def _axes(row: dict | None) -> dict[str, float] | None:
    if row is None:
        return None
    orders = row["orders"]
    return {axis: sum(o["axes"][axis] for o in orders) / len(orders) for axis in QUALITY_AXES}


def _label(manifest: ModelManifest) -> str:
    """The base model keeps its own name; other repositories are named by their repo."""
    steps = manifest.generation_config["steps"]
    base = manifest.base_reference or manifest.reference
    same = manifest.reference.repo_id == base.repo_id
    name = BASE_MODEL_NAME if same else manifest.reference.repo_id.split("/")[-1]
    return f"{name} · {steps} step{'' if steps == 1 else 's'}"


def _artifact_url(backend: str, repo: str, revision: str) -> str | None:
    """Public pinned link. Manifest validation already restricts repo/revision characters.
    Hippius has no agreed public gateway in this repository, so it gets no link."""
    return f"https://huggingface.co/{repo}/tree/{revision}" if backend == "hf" else None


def _model(manifest: ModelManifest) -> dict:
    ref = manifest.reference
    return {
        "backend": ref.backend,
        "repo": ref.repo_id,
        "revision": ref.revision,
        "url": _artifact_url(ref.backend, ref.repo_id, ref.revision),
        "steps": manifest.generation_config["steps"],
        "guidance": manifest.generation_config.get("guidance_scale"),
        "manifest_hash": manifest.fingerprint(),
        "license": manifest.license,
        "role": manifest.role,
        "files": [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.bytes,
                "url": (
                    f"https://huggingface.co/{ref.repo_id}/resolve/{ref.revision}/{item.path}"
                    if ref.backend == "hf" else None
                ),
            }
            for item in manifest.files
        ],
        "base": (
            {
                "repo": manifest.base_reference.repo_id,
                "revision": manifest.base_reference.revision,
                "url": _artifact_url(
                    manifest.base_reference.backend, manifest.base_reference.repo_id,
                    manifest.base_reference.revision,
                ),
            }
            if manifest.base_reference is not None else None
        ),
    }


def build_snapshot(
    *,
    plans: list[dict],
    publications: list[dict],
    close: dict,
    validators: dict,
    contract: dict,
    resolved: list[dict],
    reports: list[dict],
    anchors: list[dict],
    receipt_sources: list[Path],
    media_dirs: list[Path],
    media_out: Path,
    uids: dict | None = None,
    king_state: dict | None = None,
    batch: dict | None = None,
    now: datetime | None = None,
) -> dict:
    terms = ChallengeContract.from_dict(contract)
    frozen = [DistributedEvaluationPlan.from_dict(raw) for raw in plans]
    if not frozen:
        raise ValueError("at least one plan is required")
    hotkey_uid = {hotkey: int(uid) for uid, hotkey in (uids or {}).get("uids", {}).items()}
    hotkey_uid.update({row["hotkey"]: row["uid"] for row in validators["validators"]})
    models: dict[str, tuple[str, ModelManifest]] = {}
    for publication in publications:
        manifest = ModelManifest.from_dict(publication["manifest"])
        if manifest.fingerprint() != publication["manifest_hash"]:
            raise ValueError("model publication hash differs from its manifest")
        models[publication["manifest_hash"]] = (publication["hotkey"], manifest)
    aggregates = _aggregates(resolved)
    batch_hash_all = commit_deadline = None
    if batch is not None:
        frozen_batch = DiagnosticBatch.from_dict(batch)
        if {p.plan_hash() for p in frozen_batch.plans} != {p.plan_hash() for p in frozen}:
            raise ValueError("batch plan binds different challengers")
        batch_hash_all = frozen_batch.batch_hash()
        commit_deadline = frozen_batch.commit_deadline_block
    digests = {(r["plan_hash"], r["validator_hotkey"]): report_digest(r) for r in reports}
    receipts = _load_receipts(receipt_sources)
    media = _Media(media_dirs, media_out)
    wan50 = {plan.wan50_manifest_hash for plan in frozen}
    if len(wan50) != 1:
        raise ValueError("plans disagree on the Wan2.1 50-step reference")
    wan50_hash = wan50.pop()

    rounds: dict[str, dict] = {}
    challengers, evaluations = [], []
    for plan in frozen:
        context = plan.chain_context
        if (
            close.get("close_block") != context.get("close_block")
            or close.get("entropy_block") != context.get("entropy_block")
            or dict(close.get("committee", {})) != dict(plan.committee)
        ):
            raise ValueError("plan differs from the recorded submission close")
        plan_hash = plan.plan_hash()
        if plan_hash not in aggregates:
            raise ValueError(f"no resolved aggregate for plan {plan_hash}")
        aggregate, document = aggregates[plan_hash]
        if plan.candidate_manifest_hash not in models:
            raise ValueError("no model publication for the plan candidate")
        hotkey, manifest = models[plan.candidate_manifest_hash]
        if context.get("candidate_hotkey", hotkey) != hotkey:
            raise ValueError("model publication hotkey differs from the plan candidate")
        uid = hotkey_uid.get(hotkey)
        challenger_id = (
            f"uid{uid}-{plan.candidate_manifest_hash[:8]}"
            if uid is not None
            else (plan.candidate_manifest_hash[:12])
        )
        king_is_wan50 = plan.king_manifest_hash == plan.wan50_manifest_hash
        # Synthetic diagnostic panels derive from finalized entropy; sealed packs only
        # once the reveal is recorded in the frozen plan context.
        reveal_text = plan.stage == "diagnostic" or "pack_reveal_block" in context
        work = allocate_work(plan)
        contributors = set(aggregate["contributors"])
        failures = {f["hotkey"]: f["reason"] for f in document.get("failures", [])}
        equivocated = {
            item.split(":", 1)[1]
            for item in aggregate["invalid"]
            if item.startswith("EQUIVOCATION:")
        }

        statuses = {
            validator: "valid"
            if validator in contributors
            else "invalid"
            if validator in equivocated
            or failures.get(validator, "MISSING_ANCHOR_OR_PEER") != "MISSING_ANCHOR_OR_PEER"
            else "missing"
            for validator in plan.committee
        }
        batch_hash = document.get("batch_hash", batch_hash_all)
        anchor_blocks = {
            a["hotkey"]: a["commitment_block"]
            for a in anchors
            if a.get("status") == "FINALIZED"
            and (batch_hash is None or a.get("batch_hash") == batch_hash)
        }
        reports_out = []
        for validator in sorted(plan.committee):
            mine = [item for item in work if item.validator_hotkey == validator]
            ids = {item.prompt_id for item in mine}
            audits = {item.prompt_id for item in mine if item.audit}
            status = statuses[validator]
            reports_out.append(
                {
                    "validator": validator,
                    "prompts": [p["prompt_id"] for p in plan.prompts if p["prompt_id"] in ids],
                    "audits": [p["prompt_id"] for p in plan.prompts if p["prompt_id"] in audits],
                    "report_sha256": digests.get((plan_hash, validator))
                    if status == "valid"
                    else None,
                    "anchor_block": anchor_blocks.get(validator) if status == "valid" else None,
                    "status": status,
                }
            )

        rows = {
            field: {row["prompt_id"]: row for row in aggregate[field]}
            for field in ("quality_vs_king", "quality_vs_wan50", "timing")
        }
        prompts_out = []
        for prompt in plan.prompts:
            prompt_id = prompt["prompt_id"]
            assigned = [item for item in work if item.prompt_id == prompt_id]
            king_row = rows["quality_vs_king"].get(prompt_id)
            wan50_row = rows["quality_vs_wan50"].get(prompt_id)
            measured_by = [
                c["hotkey"]
                for c in (wan50_row or king_row or {}).get("contributions", [])
                if c["hotkey"] in contributors
            ]
            # Speed: the aggregate ratio when timing exists; else per-GPU render ratios
            # (king seconds / challenger seconds on one machine), stake-weighted in log space.
            speed = None
            if prompt_id in rows["timing"]:
                runs = rows["timing"][prompt_id]["runs"]
                speed = runs[0]["seconds"] / runs[1]["seconds"]
            else:
                logs = []
                for validator in measured_by:
                    ref = receipts.get(_job(validator, plan, prompt, plan.king_manifest_hash))
                    cand = receipts.get(_job(validator, plan, prompt, plan.candidate_manifest_hash))
                    if ref and cand:
                        ratio = ref[1]["full_pipeline_seconds"] / cand[1]["full_pipeline_seconds"]
                        logs.append((plan.committee[validator], math.log(ratio)))
                if logs:
                    speed = math.exp(sum(w * v for w, v in logs) / sum(w for w, _ in logs))
            axes = _axes(wan50_row if king_is_wan50 else king_row)
            videos = None
            roles = [("wan50", plan.wan50_manifest_hash)]
            if not king_is_wan50:
                roles.append(("king", plan.king_manifest_hash))
            roles.append(("challenger", plan.candidate_manifest_hash))
            for validator in measured_by:
                found = {role: receipts.get(_job(validator, plan, prompt, h)) for role, h in roles}
                if all(found.values()):
                    expected = wan50_row if king_is_wan50 else king_row
                    if expected and (
                        expected.get(
                            "candidate_video_sha256", found["challenger"][1]["video_sha256"]
                        )
                        != found["challenger"][1]["video_sha256"]
                    ):
                        continue
                    videos = {role: media.publish(*found[role]) for role, _ in roles}
                    break
            quality_king = _axes(king_row)
            quality_wan50 = _axes(wan50_row)
            prompts_out.append(
                {
                    "id": prompt_id,
                    "text": prompt["text"] if reveal_text else None,
                    "category": {
                        "subject": prompt["subject_class"],
                        "camera": prompt["camera_type"],
                        "style": prompt["style_family"],
                    },
                    "validators": sorted({item.validator_hotkey for item in assigned}),
                    "audit": any(item.audit for item in assigned),
                    "speed_ratio": _round(speed),
                    "quality_vs_king": None
                    if king_is_wan50 or quality_king is None
                    else _round(_mean(list(quality_king.values()))),
                    "quality_vs_wan50": None
                    if quality_wan50 is None
                    else _round(_mean(list(quality_wan50.values()))),
                    "axes": None if axes is None else {k: _round(v) for k, v in axes.items()},
                    "videos": videos,
                }
            )

        decision = aggregate.get("decision")
        if plan.stage == "diagnostic":
            status = "DIAGNOSTIC"
            reasons = [DIAGNOSTIC_REASON]
            if aggregate["status"] != "DIAGNOSTIC":
                reasons.append(f"Aggregate {aggregate['status'].lower()}")
        elif decision is not None:
            status = DECISIONS.get(str(decision["status"]), "INCOMPLETE")
            reasons = [str(reason) for reason in decision.get("reasons", [])]
        else:
            status, reasons = (
                "INCOMPLETE",
                [f"Aggregate {aggregate['status'].lower()}, no decision"],
            )
        reasons += [
            f"Report {r['status']}: {r['validator']}" for r in reports_out if r["status"] != "valid"
        ]
        speeds = [p["speed_ratio"] for p in prompts_out if p["speed_ratio"] is not None]
        kings = [p["quality_vs_king"] for p in prompts_out if p["quality_vs_king"] is not None]
        wans = [p["quality_vs_wan50"] for p in prompts_out if p["quality_vs_wan50"] is not None]
        total_stake = sum(plan.committee.values())
        favorable = sum(plan.committee[h] for h in contributors) if status == "PROMOTE" else 0
        evaluations.append(
            {
                "challenger_id": challenger_id,
                "plan_hash": plan_hash,
                "verdict": {
                    "status": status,
                    "reasons": reasons,
                    # Geometric mean of per-prompt ratios; range is the per-prompt spread.
                    "speed_ratio": _round(math.exp(_mean([math.log(s) for s in speeds])))
                    if speeds
                    else None,
                    "speed_range": [min(speeds), max(speeds)] if speeds else None,
                    "quality_vs_king": None if king_is_wan50 else _round(_mean(kings)),
                    "quality_vs_wan50": _round(_mean(wans)),
                },
                "consensus": {
                    "rule": CONSENSUS_RULE,
                    "favorable_stake": _tao(favorable),
                    "total_stake": _tao(total_stake),
                    "reports": reports_out,
                },
                "prompts": prompts_out,
            }
        )
        challengers.append(
            {
                "id": challenger_id,
                "round_id": plan.round_id,
                "label": _label(manifest),
                "hotkey": hotkey,
                "uid": uid,
                "model": _model(manifest),
                "manifest_hash": plan.candidate_manifest_hash,
                "commit_block": context.get("candidate_commit_block"),
                "status": CHALLENGER_STATUS[status],
            }
        )
        complete = aggregate["status"] in {"COMPLETE", "DIAGNOSTIC"} and all(
            r["status"] == "valid" for r in reports_out
        )
        entry = rounds.setdefault(
            plan.round_id,
            {
                "round_id": plan.round_id,
                "stage": "final",
                "diagnostic": plan.stage == "diagnostic",
                "close_block": context.get("close_block"),
                "entropy_block": context.get("entropy_block"),
                "commit_close_block": document.get("commit_deadline_block", commit_deadline),
                "committee": [
                    {"hotkey": h, "stake": _tao(s)} for h, s in sorted(plan.committee.items())
                ],
                "challengers": [],
            },
        )
        entry["challengers"].append(challenger_id)
        if not complete:
            entry["stage"] = "consensus"

    plan_kings = {plan.king_manifest_hash for plan in frozen}
    king, crowns = _king(king_state, wan50_hash, models, hotkey_uid, plan_kings)
    return {
        "schema": SCHEMA,
        "updated_at": (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "network": {
            "name": frozen[0].chain_context.get("network"),
            "netuid": frozen[0].chain_context.get("netuid"),
        },
        "base_model": {
            "name": BASE_MODEL_NAME,
            "repo": terms.base_model_id,
            "revision": terms.base_model_revision,
            "url": _artifact_url("hf", terms.base_model_id, terms.base_model_revision),
            "steps": terms.reference_steps,
        },
        "contract": {
            "min_speedup": terms.min_speedup,
            "quality_margin": terms.axis_noninferiority_margin,
            "axes": list(QUALITY_AXES),
            "consensus_rule": CONSENSUS_RULE,
            "reference_steps": terms.reference_steps,
        },
        "king": king,
        "crowns": crowns,
        "validators": [
            {
                "uid": row["uid"],
                "hotkey": row["hotkey"],
                "stake": _tao(row["stake_rao"]),
                "active": row["active"],
            }
            for row in validators["validators"]
        ],
        "rounds": list(rounds.values()),
        "challengers": challengers,
        "evaluations": evaluations,
    }


def _king(state_raw, wan50_hash, models, hotkey_uid, plan_kings):
    """Project explicit finalized state; an evaluation plan is not a crown."""
    if state_raw is None:
        if plan_kings == {wan50_hash}:
            return None, []
        raise ValueError("non-baseline king requires explicit finalized king state")
    state = FinalizedCompetitionState.from_dict(state_raw)

    def known(fingerprint):
        found = models.get(fingerprint)
        return (_label(found[1]), _model(found[1])) if found else (None, None)

    crowns = [
        {
            "crown_id": c.crown_id,
            "hotkey": c.hotkey,
            "label": known(c.fingerprint)[0],
            "block": None,
            "speed_ratio": None,
            "quality": None,
        }
        for c in reversed(state.crown_history)
        if not c.baseline
    ]
    if state.current_king_fingerprint == wan50_hash:
        return None, crowns
    crown = next(
        (
            c
            for c in reversed(state.crown_history)
            if c.fingerprint == state.current_king_fingerprint
        ),
        None,
    )
    if crown is None:
        raise ValueError("certified king has no crown record")
    label, model = known(crown.fingerprint)
    return {
        "hotkey": crown.hotkey,
        "uid": hotkey_uid.get(crown.hotkey),
        "label": label,
        "model": model,
        "manifest_hash": crown.fingerprint,
        "since_block": None,
        "crown_id": crown.crown_id,
        "speed_ratio_vs_wan50": None,
        "quality_vs_wan50": None,
    }, crowns


def add_commands(commands: argparse._SubParsersAction) -> None:
    command = commands.add_parser(
        "dashboard-export-v2", help="public forge/dashboard/v2 snapshot from distributed rounds"
    )
    many = {"type": Path, "action": "append", "required": True}
    command.add_argument("--plan", **many)
    command.add_argument("--model-publication", **many)
    command.add_argument("--resolved", **many)
    command.add_argument("--worker-receipts", **many)
    command.add_argument("--report", type=Path, action="append", default=[])
    command.add_argument("--anchor", type=Path, action="append", default=[])
    command.add_argument("--media-dir", type=Path, action="append", default=[])
    command.add_argument("--raw-evidence", type=Path, action="append", default=[])
    command.add_argument("--close", type=Path, required=True)
    command.add_argument("--validators", type=Path, required=True)
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--uids", type=Path)
    command.add_argument("--king-state", type=Path)
    command.add_argument("--batch", type=Path, help="diagnostic batch plan (commit deadline)")
    command.add_argument("--media-out", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command != "dashboard-export-v2":
        return None
    snapshot = build_snapshot(
        plans=[_read(p) for p in args.plan],
        publications=[_read(p) for p in args.model_publication],
        close=_read(args.close),
        validators=_read(args.validators),
        contract=_read(args.contract),
        resolved=[_read(p) for p in args.resolved],
        reports=[_read(p, decode=False) for p in args.report],  # digest of exact bytes
        anchors=[_read(p) for p in args.anchor],
        receipt_sources=args.worker_receipts,
        media_dirs=args.media_dir,
        media_out=args.media_out,
        uids=_read(args.uids) if args.uids else None,
        king_state=_read(args.king_state) if args.king_state else None,
        batch=_read(args.batch) if args.batch else None,
    )
    if args.raw_evidence:
        from forge.validator.dashboard_native_metrics import attach_native_metrics

        snapshot = attach_native_metrics(
            snapshot,
            [_read(p) for p in args.plan],
            [_read(p, decode=False) for p in args.report],
            args.raw_evidence,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"schema": SCHEMA, "evaluations": len(snapshot["evaluations"])}))
    return 0
