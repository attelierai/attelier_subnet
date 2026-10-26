"""V1 bridge contract through filesystem receipts, SQLite and sr25519.

The raw records below are an explicitly ``synthetic: true`` external-GPU
fixture held in a trusted validator-root.  They exercise receipt authority,
signing and restart only; they never assert real GPU execution or model quality.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.chain.launch_authority import BootstrapSignatureAuthority, QualifiedLaunchAuthority
from forge.chain.launch_publication import BootstrapAuthority, QualifiedPublicationContext
from forge.chain.reward_publication import WeightIntent, WeightSubmissionReceipt
from forge.evaluation import launch_policy
from forge.evaluation.evaluator_qualification import finalize_qualification, prepare_controls
from forge.evaluation.synthetic_prompts import launch_panel, launch_panel_v2
from forge.evaluation.vision_media import load_video
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.consensus_types import (
    CertificateError,
    ConsensusPolicy,
    FinalizedCompetitionState,
    RoundPlan,
)
from forge.protocol.hashing import hash_canonical
from forge.protocol.preconsensus import build_certificate, commit_report, decide_round
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox
from forge.validator.admission import build_authenticated_history
from forge.validator.launch_bridge import (
    LaunchBridgeError,
    LaunchCertificateVerifier,
    LaunchExecutionReplayer,
    LaunchV1Bridge,
    LaunchV1Capacity,
    LocalLaunchReceiptStore,
    launch_local_report,
)
from forge.validator.launch_certified_dashboard import (
    CertifiedDashboardError,
    export_certified_snapshot,
)
from forge.validator.launch_voting import LaunchV1VotingWorkflow, LaunchVotingError


def sha(number):
    return f"{number:064x}"


def _external_gpu_contract_package(
    tmp_path,
    *,
    candidate,
    king,
    anchor=None,
    runtime=None,
    entropy="e" * 64,
    round_id="authority-contract-v2",
):
    """Simulated external-GPU response for authorization-contract coverage only.

    The MP4/control boundary and the full 96-row replay are real. Scores are
    deliberately provider responses made in this test, never campaign evidence
    and never exported outside ``tmp_path``.
    """
    tmp_path.mkdir()
    sources = []
    for split in ("development", "confirmation"):
        for index, family in enumerate(("human", "animal", "object", "scenery")):
            path = tmp_path / f"{split}-{family}.mp4"
            frames = np.full(
                (4, 32, 32, 3),
                24 + index * 20 + (80 if split == "confirmation" else 0),
                dtype=np.uint8,
            )
            for frame_index, frame in enumerate(frames):
                frame[:, (frame_index * 5) % 24 : (frame_index * 5) % 24 + 8] = 240
            writer = imageio_ffmpeg.write_frames(
                str(path),
                (32, 32),
                fps=8,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
            )
            writer.send(None)
            for frame in frames:
                writer.send(frame.tobytes())
            writer.close()
            sources.append(
                {
                    "source_id": f"{split}-{family}",
                    "split": split,
                    "family": family,
                    "prompt": f"{family} contract source",
                    "path": str(path),
                }
            )
    control_root = tmp_path / "external-gpu-contract-controls"
    control_plan = prepare_controls(
        sources, {"width": 32, "height": 32, "frames": 4, "fps": 8}, control_root
    )
    spec, evaluator_runtime = {"provider": "simulated-contract"}, {"device": "simulated-boundary"}
    raw = {
        "schema": control_plan["schema"],
        "control_plan_id": control_plan["control_plan_id"],
        "evaluator": {
            "spec": spec,
            "runtime": evaluator_runtime,
            "spec_hash": hash_canonical("forge/launch-evaluator-spec/v1", spec),
            "evaluator_id": hash_canonical(
                "forge/videoreward-study-runtime/v1", spec, evaluator_runtime
            ),
        },
        "rows": [],
    }
    axis = {
        "freeze": "MQ",
        "black_frame": "VQ",
        "magenta_frame": "VQ",
        "segment_corruption": "VQ",
        "flicker": "MQ",
        "blur": "VQ",
        "semantic_mismatch": "TA",
    }
    for control in control_plan["controls"]:
        row = {"control_id": control["control_id"]}
        if control["kind"] == "missing":
            row["status"] = "MISSING_REJECTED"
        elif control["kind"] == "hash_tamper":
            row["status"] = "HASH_MISMATCH_REJECTED"
        else:
            decoded = load_video(
                control_root / control["path"], control_plan["manifest"]["geometry"]
            )
            scores = {"VQ": 0.0, "MQ": 0.0, "TA": 0.0}
            if control["kind"] in axis:
                scores[axis[control["kind"]]] = -1.0
            row.update(
                status="OK",
                result={
                    "video_sha256": decoded.video_sha256,
                    "scores": scores,
                    "diagnostics": decoded.diagnostics,
                },
            )
        raw["rows"].append(row)
    raw["raw_receipt_id"] = hash_canonical("forge/evaluator-controls-score/v1", raw)
    result = finalize_qualification(control_plan, raw)
    assert result["qualification"] == "QUALIFIED", result

    prompts = launch_panel_v2(entropy, round_id=round_id)
    timing = sorted(
        (item["prompt_id"] for item in prompts),
        key=lambda value: hash_canonical("forge/launch-v2-timing/v1", entropy, round_id, value),
    )[:8]
    diversity = []
    for family in sorted({item["family"] for item in prompts}):
        choices = [item["prompt_id"] for item in prompts if item["family"] == family]
        diversity.append(
            min(
                choices,
                key=lambda value: hash_canonical(
                    "forge/launch-v2-diversity/v1", entropy, round_id, family, value
                ),
            )
        )
    policy = launch_policy.freeze_v2(
        {
            "models": {"candidate": candidate, "king": king, "anchor": anchor or sha(998)},
            "evaluator": {
                "evaluator_id": result["evaluator_id"],
                "spec_hash": result["spec_hash"],
            },
            "runtime": runtime
            or {"runtime_id": sha(990), "hardware_id": sha(991), "dependency_lock_hash": sha(992)},
            "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
            "prompts": prompts,
            "timing_prompt_ids": timing,
            "diversity_prompt_ids": diversity,
            "evaluator_qualification": result,
        }
    )
    return policy, {"plan": control_plan, "raw": raw, "result": result}


def test_publication_authority_rejects_synthetic_fixture_and_bootstrap_signature_is_bound(tmp_path):
    """Real sr25519 signing and SQLite state cannot launder this synthetic fixture."""
    signer = signers()[0]
    state = FinalizedCompetitionState.genesis("authority", sha(1))
    with Store(tmp_path / "authority.sqlite") as store:
        store.initialize_state(
            state_hash=state.state_hash(),
            challenge_id=state.challenge_id,
            state_version=0,
            previous_state_hash=state.previous_state_hash,
            payload=state.to_dict(),
        )
        plan = round_plan(state, {signer.validator_hotkey: 1}, policy()[0], sha(2), "authority")
        context = QualifiedPublicationContext(
            "bittensor-sdk/testnet-qualified-v1",
            "test",
            578,
            sha(3),
            sha(4),
            0,
            plan.contract_hash,
            plan.plan_hash(),
            state.state_hash(),
            None,
            sha(5),
        )
        intent = WeightIntent(
            "test",
            578,
            state.state_hash(),
            sha(6),
            sha(4),
            sha(7),
            sha(8),
            "BURN_REQUIRED",
            {0: 1.0},
            {0: "fallback"},
            "fixture",
        )
        authority = QualifiedLaunchAuthority(
            store,
            plan,
            state,
            None,
            {
                "synthetic": True,
                "genesis_hash": sha(3),
                "runtime_hash": sha(4),
                "mechanism": 0,
                "activation_plan_hash": sha(5),
            },
        )
        assert not authority.verify(context, intent)
    unsigned = BootstrapAuthority(
        signer.validator_hotkey, "00", sha(3), sha(5), sha(8), 0, "fallback"
    )
    signed = BootstrapAuthority(
        signer.validator_hotkey,
        signer.sign(unsigned.signed_payload()).hex(),
        sha(3),
        sha(5),
        sha(8),
        0,
        "fallback",
    )
    verifier = BootstrapSignatureAuthority(signer.validator_hotkey)
    assert verifier.verify(signed)
    assert not verifier.verify(
        BootstrapAuthority(
            signer.validator_hotkey, signed.signature, sha(9), sha(5), sha(8), 0, "fallback"
        )
    )


def test_publication_authority_replays_external_contract_package_and_signed_certificate(tmp_path):
    """A simulated provider response cannot bypass replay or the signed certificate.

    This is authorization-contract coverage only.  It creates no campaign
    artifact and says nothing about scientific qualification or a live chain.
    """
    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity = (
        open_context(tmp_path)
    )
    receipts = LocalLaunchReceiptStore(root)
    replayer = LaunchExecutionReplayer(receipts)
    reports = [
        launch_local_report(
            plan,
            signer.validator_hotkey,
            raw_evidence(
                root,
                frozen,
                manifests,
                hardware,
                runtime_doc,
                plan,
                signer.validator_hotkey,
                candidate_score=10.0,
            ),
            replayer,
        )
        for signer in members
    ]
    commits, reveals = zip(
        *(
            commit_report(plan, report, 150, salt=f"authority-{index}")
            for index, report in enumerate(reports)
        ),
        strict=True,
    )
    certificate = build_certificate(
        plan, decide_round(plan, commits, reveals), 401, members, commits=commits, reveals=reveals
    )
    with Store(tmp_path / "authority-positive.sqlite") as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
        bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        certified = bridge.apply(
            plan,
            state,
            certificate,
            LaunchCertificateVerifier(Sr25519Verifier(), replayer=replayer, store=store),
        )
        policy, package = _external_gpu_contract_package(
            tmp_path / "provider-contract",
            candidate=plan.challenger_fingerprint,
            king=plan.king_fingerprint,
        )
        body = {
            "network": "test",
            "netuid": 578,
            "genesis_hash": sha(301),
            "runtime_hash": sha(302),
            "mechanism": 0,
            "contract_hash": plan.contract_hash,
            "round_plan_hash": plan.plan_hash(),
            "certified_state_hash": certified.state_hash(),
            "certificate_hash": certificate.transition_hash(),
            "evidence_origin": "external_gpu_qualified_v2",
            "policy_plan": policy,
            "qualification_package": package,
        }
        activation = {
            **body,
            "activation_plan_hash": hash_canonical("forge/launch-activation-plan/v1", body),
        }
        context = QualifiedPublicationContext(
            "bittensor-sdk/testnet-qualified-v1",
            "test",
            578,
            sha(301),
            sha(302),
            0,
            plan.contract_hash,
            plan.plan_hash(),
            certified.state_hash(),
            certificate.transition_hash(),
            activation["activation_plan_hash"],
        )
        intent = WeightIntent(
            "test",
            578,
            certified.state_hash(),
            certificate.transition_hash(),
            sha(302),
            sha(303),
            sha(304),
            "WINNER",
            {1: 1.0},
            {1: "winner"},
            "contract-test",
        )
        authority = QualifiedLaunchAuthority(store, plan, state, certificate, activation, str(root))
        # A V1 certificate cannot be laundered through a separately-created
        # V2 package; a real positive path must certify that exact V2 plan.
        assert not authority.verify(context, intent)

        for field, value in (
            ("genesis_hash", sha(305)),
            ("contract_hash", sha(306)),
            ("round_plan_hash", sha(307)),
            ("certified_state_hash", sha(308)),
        ):
            altered_body = {**body, field: value}
            altered = {
                **altered_body,
                "activation_plan_hash": hash_canonical(
                    "forge/launch-activation-plan/v1", altered_body
                ),
            }
            assert not QualifiedLaunchAuthority(
                store, plan, state, certificate, altered, str(root)
            ).verify(context, intent)
        assert not QualifiedLaunchAuthority(
            store,
            plan,
            state,
            certificate,
            {**activation, "activation_plan_hash": sha(309)},
            str(root),
        ).verify(context, intent)
        changed_package = json.loads(json.dumps(package))
        changed_package["raw"]["rows"][0]["status"] = "CHANGED"
        altered_package = {
            **body,
            "qualification_package": changed_package,
        }
        altered_package = {
            **altered_package,
            "activation_plan_hash": hash_canonical(
                "forge/launch-activation-plan/v1", altered_package
            ),
        }
        assert not QualifiedLaunchAuthority(
            store, plan, state, certificate, altered_package, str(root)
        ).verify(
            QualifiedPublicationContext(
                **{
                    **context.__dict__,
                    "activation_plan_hash": altered_package["activation_plan_hash"],
                }
            ),
            intent,
        )


def put(root, kind, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path = root / kind / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return digest


def signers():
    from bittensor.keyfiles import Keypair

    return [
        Sr25519Signer(SimpleNamespace(hotkey=Keypair.create_from_uri(uri)))
        for uri in ("//Alice", "//Bob", "//Charlie")
    ]


class _ControlledSigner:
    """External-signer boundary used to exercise durable retry behaviour."""

    def __init__(self, signer, *, fail=False):
        self.signer, self.fail, self.calls = signer, fail, 0
        self.validator_hotkey = signer.validator_hotkey

    def sign(self, payload):
        self.calls += 1
        if self.fail:
            raise RuntimeError("external signer unavailable")
        return self.signer.sign(payload)


def manifest(number):
    base = ArtifactReference("hf", "fixture/Wan21T2V13B", "a" * 40)
    return ModelManifest(
        ArtifactReference("hf", f"fixture/wan-miner-{number}", f"{number:040x}"),
        "lora",
        (ManifestFile("adapter_model.safetensors", sha(number + 30), 128),),
        "mit",
        {"steps": 16, "guidance_scale": 5.0, "scheduler": "original_unipc"},
        base,
    )


def policy(
    candidate_number=3, king_number=4, anchor_number=4, entropy_hash=None, round_id="launch-r1"
):
    entropy_hash = sha(700) if entropy_hash is None else entropy_hash
    manifests = {
        "candidate": manifest(candidate_number),
        "king": manifest(king_number),
        "anchor": manifest(anchor_number),
    }
    models = {name: item.fingerprint() for name, item in manifests.items()}
    hardware = {"gpu_name": "synthetic-external-gpu-fixture", "vram_capacity_bytes": 1}
    dependencies = {"torch": "fixture"}
    runtime_doc = {"dependencies": dependencies, "runner": "external-fixture"}
    prompts = launch_panel(entropy_hash, round_id=round_id)
    timing_prompt_ids = [
        prompt_id
        for prompt_id in sorted(
            (prompt["prompt_id"] for prompt in prompts),
            key=lambda prompt_id: hash_canonical(
                "forge/launch-v1-timing/v1", entropy_hash, round_id, prompt_id
            ),
        )[:8]
    ]
    return (
        launch_policy.freeze(
            {
                "models": models,
                "evaluator": {"evaluator_id": sha(6), "spec_hash": sha(7)},
                "runtime": {
                    "runtime_id": hash_canonical("forge/runtime/v1", runtime_doc),
                    "hardware_id": hash_canonical("forge/launch-hardware/v1", hardware),
                    "dependency_lock_hash": hash_canonical(
                        "forge/launch-dependencies/v1", dependencies
                    ),
                },
                "geometry": {"width": 832, "height": 480, "frames": 81, "fps": 16},
                "prompts": prompts,
                "timing_prompt_ids": timing_prompt_ids,
                "assignments": [{"prompt_id": prompt["prompt_id"], **models} for prompt in prompts],
            }
        ),
        manifests,
        hardware,
        runtime_doc,
    )


def round_plan(state, committee, frozen, admission_root, round_id="launch-r1", entropy_hash=None):
    entropy_hash = sha(700) if entropy_hash is None else entropy_hash
    return RoundPlan(
        "test",
        578,
        round_id,
        state.state_hash(),
        state.challenge_id,
        state.current_king_fingerprint,
        frozen["models"]["candidate"],
        "miner",
        sha(9),
        {
            "admission_history_root": admission_root,
            "prompt_history_root": sha(10),
            "launch_plan_id": frozen["plan_id"],
            "anchor_fingerprint": frozen["models"]["anchor"],
            "entropy_block": 101,
            "entropy_hash": entropy_hash,
        },
        committee,
        100,
        200,
        300,
        400,
        ConsensusPolicy.STAKE_MAJORITY_V2,
    )


def raw_evidence(
    root, frozen, manifests, hardware, runtime_doc, plan, validator, candidate_score=10.0
):
    """Create two exact 8-prompt shards per model from an external-GPU fixture."""
    (root / "provenance.json").write_text(
        json.dumps({"synthetic": True, "purpose": "protocol integration fixture"})
    )
    panels, panel_docs = [], {}
    emitted = set()
    for name, item in manifests.items():
        if item.fingerprint() in emitted:
            continue
        emitted.add(item.fingerprint())
        existing = next(
            (
                prior
                for prior, docs in panel_docs.items()
                if manifests[prior].fingerprint() == item.fingerprint()
            ),
            None,
        )
        if existing is not None:
            panel_docs[name] = panel_docs[existing]
            continue
        panel_docs[name] = []
        for shard in (frozen["prompts"][:8], frozen["prompts"][8:]):
            doc = {
                "schema": "forge/panel/v1",
                "contract_hash": plan.contract_hash,
                "admission": "public_snapshot",
                "manifest": item.to_dict(),
                "manifest_hash": item.fingerprint(),
                "geometry": frozen["geometry"],
                "runtime_id": frozen["runtime"]["runtime_id"],
                "runtime": runtime_doc,
                "hardware": hardware,
                "prompts": [
                    {
                        "prompt_id": prompt["prompt_id"],
                        "text": prompt["text"],
                        "video_seeds": [prompt["seed"]],
                    }
                    for prompt in shard
                ],
                "observations": [
                    {
                        "prompt_id": prompt["prompt_id"],
                        "seed": prompt["seed"],
                        "video_sha256": hashlib.sha256(
                            f"{name}:{prompt['prompt_id']}".encode()
                        ).hexdigest(),
                        "status": "OK",
                    }
                    for prompt in shard
                ],
            }
            digest = put(root, "panels", doc)
            panels.append(digest)
            panel_docs[name].append((digest, doc))
    scores = []
    scored_manifests = set()
    for name, shard_docs in panel_docs.items():
        if manifests[name].fingerprint() in scored_manifests:
            continue
        scored_manifests.add(manifests[name].fingerprint())
        score_value = candidate_score if name == "candidate" else 0.0
        for panel_digest, panel in shard_docs:
            for observation in panel["observations"]:
                scores.append(
                    put(
                        root,
                        "scores",
                        {
                            "schema": "forge/launch-score/v1",
                            "plan_id": frozen["plan_id"],
                            "prompt_id": observation["prompt_id"],
                            "prompt_text": next(
                                prompt["text"]
                                for prompt in frozen["prompts"]
                                if prompt["prompt_id"] == observation["prompt_id"]
                            ),
                            "seed": observation["seed"],
                            "manifest_hash": manifests[name].fingerprint(),
                            "video_sha256": observation["video_sha256"],
                            "evaluator_id": frozen["evaluator"]["evaluator_id"],
                            "spec_hash": frozen["evaluator"]["spec_hash"],
                            "geometry": frozen["geometry"],
                            "result": {
                                "scores": {"VQ": score_value, "MQ": score_value, "TA": score_value},
                                "diagnostics": {
                                    "frozen_fraction": 0.0,
                                    "luma_acceleration_median": 0.1,
                                    "centered_transition_median": 0.1,
                                    "temporal_delta": 0.02,
                                },
                                "elapsed_seconds": 1.0,
                                "peak_cuda_bytes": 1,
                                "video_sha256": observation["video_sha256"],
                            },
                            "source_report_sha256": panel_digest,
                        },
                    )
                )

    def timing_receipt(shard, schema, design, order):
        links = {"reference": {}, "candidate": {}}
        measurements = []
        for prompt in shard:
            key = f"{prompt['prompt_id']}:{prompt['seed']}"
            for side, name in (("reference", "king"), ("candidate", "candidate")):
                observation = next(
                    item
                    for _, doc in panel_docs[name]
                    for item in doc["observations"]
                    if item["prompt_id"] == prompt["prompt_id"]
                )
                links[side][key] = observation["video_sha256"]
            for run_order, (side, seconds) in enumerate(order):
                measurements.append(
                    {
                        "status": "OK",
                        "prompt_id": prompt["prompt_id"],
                        "seed": prompt["seed"],
                        "pair_id": key,
                        "run_order": run_order,
                        "side": side,
                        "full_pipeline_seconds": seconds,
                        "gpu_synchronized": True,
                    }
                )
        return put(
            root,
            "timing",
            {
                "schema": schema,
                "frozen_plan_id": frozen["plan_id"],
                "contract_hash": plan.contract_hash,
                "reference_manifest_hash": frozen["models"]["king"],
                "candidate_manifest_hash": frozen["models"]["candidate"],
                "runtime_id": frozen["runtime"]["runtime_id"],
                "runtime": runtime_doc,
                "hardware": hardware,
                "geometry": frozen["geometry"],
                "design": {"name": design},
                "report_video_sha256": links,
                "measurements": measurements,
            },
        )

    timing_prompts = [
        prompt for prompt in frozen["prompts"] if prompt["prompt_id"] in frozen["timing_prompt_ids"]
    ]
    screening_prompts = [
        prompt
        for prompt in frozen["prompts"]
        if prompt["prompt_id"] not in frozen["timing_prompt_ids"]
    ]
    timings = [
        timing_receipt(
            timing_prompts,
            "forge/paired-timing/v1",
            "confirmation_abba",
            (("reference", 12.0), ("candidate", 9.0), ("candidate", 9.0), ("reference", 12.0)),
        ),
        timing_receipt(
            screening_prompts,
            "forge/screen-timing/v1",
            "screening_ab",
            (("reference", 12.0), ("candidate", 9.0)),
        ),
    ]
    return put(
        root,
        "bundles",
        {
            "schema": "forge/launch-execution-bundle/v1",
            "validator_hotkey": validator,
            "round_plan_hash": plan.plan_hash(),
            "parent_state_hash": plan.parent_state_hash,
            "admission_history_root": plan.prompt_seed_inputs["admission_history_root"],
            "prompt_history_root": plan.prompt_seed_inputs["prompt_history_root"],
            "launch_plan_id": frozen["plan_id"],
            "models": frozen["models"],
            "launch_plan": frozen,
            "panel_receipts": panels,
            "timing_receipts": timings,
            "score_receipts": scores,
        },
    )


def _raw_evidence_v2(
    root, frozen, manifests, hardware, runtime_doc, plan, validator, package, candidate_score=1.0
):
    """Complete V2 validator-custody fixture; provider values are simulated."""
    panels, scores = [], []
    diagnostics = {
        "dark_frame_fraction": 0.0,
        "magenta_frame_fraction": 0.0,
        "bright_frame_fraction": 0.0,
        "frozen_fraction": 0.0,
        "luma_acceleration_p95": 0.1,
        "high_frequency_proxy": 1.0,
        "temporal_delta": 0.1,
    }
    emitted = set()
    for name, item in manifests.items():
        if item.fingerprint() in emitted:
            continue
        emitted.add(item.fingerprint())
        doc = {
            "schema": "forge/panel/v1",
            "contract_hash": plan.contract_hash,
            "admission": "public_snapshot",
            "manifest": item.to_dict(),
            "manifest_hash": item.fingerprint(),
            "geometry": frozen["geometry"],
            "runtime_id": frozen["runtime"]["runtime_id"],
            "runtime": runtime_doc,
            "hardware": hardware,
            "prompts": [
                {"prompt_id": p["prompt_id"], "text": p["text"], "video_seeds": [p["seed"]]}
                for p in frozen["prompts"]
            ],
            "observations": [
                {
                    "prompt_id": p["prompt_id"],
                    "seed": p["seed"],
                    "video_sha256": hashlib.sha256(
                        f"v2:{name}:{p['prompt_id']}".encode()
                    ).hexdigest(),
                    "status": "OK",
                }
                for p in frozen["prompts"]
            ],
        }
        digest = put(root, "panels", doc)
        panels.append(digest)
        for obs in doc["observations"]:
            scores.append(
                put(
                    root,
                    "scores",
                    {
                        "schema": "forge/launch-score/v1",
                        "plan_id": frozen["plan_id"],
                        "prompt_id": obs["prompt_id"],
                        "prompt_text": next(
                            p["text"]
                            for p in frozen["prompts"]
                            if p["prompt_id"] == obs["prompt_id"]
                        ),
                        "seed": obs["seed"],
                        "manifest_hash": item.fingerprint(),
                        "video_sha256": obs["video_sha256"],
                        "evaluator_id": frozen["evaluator"]["evaluator_id"],
                        "spec_hash": frozen["evaluator"]["spec_hash"],
                        "geometry": frozen["geometry"],
                        "result": {
                            "scores": {
                                "VQ": candidate_score if name == "candidate" else 0.0,
                                "MQ": candidate_score if name == "candidate" else 0.0,
                                "TA": candidate_score if name == "candidate" else 0.0,
                            },
                            "diagnostics": diagnostics,
                            "elapsed_seconds": 1.0,
                            "peak_cuda_bytes": 1,
                            "video_sha256": obs["video_sha256"],
                        },
                        "source_report_sha256": digest,
                    },
                )
            )
    timing = []
    for subset, schema, design, order in (
        (
            frozen["timing_prompt_ids"],
            "forge/paired-timing/v1",
            "confirmation_abba",
            (("reference", 12.0), ("candidate", 9.0), ("candidate", 9.0), ("reference", 12.0)),
        ),
        (
            [
                p["prompt_id"]
                for p in frozen["prompts"]
                if p["prompt_id"] not in frozen["timing_prompt_ids"]
            ],
            "forge/screen-timing/v1",
            "screening_ab",
            (("reference", 12.0), ("candidate", 9.0)),
        ),
    ):
        links = {"reference": {}, "candidate": {}}
        measurements = []
        for pid in subset:
            p = next(x for x in frozen["prompts"] if x["prompt_id"] == pid)
            key = f"{pid}:{p['seed']}"
            for side, name in (("reference", "king"), ("candidate", "candidate")):
                panel = next(
                    json.loads((root / "panels" / f"{d}.json").read_text())
                    for d in panels
                    if json.loads((root / "panels" / f"{d}.json").read_text())["manifest_hash"]
                    == manifests[name].fingerprint()
                )
                links[side][key] = next(
                    o["video_sha256"] for o in panel["observations"] if o["prompt_id"] == pid
                )
            measurements += [
                {
                    "status": "OK",
                    "prompt_id": pid,
                    "seed": p["seed"],
                    "pair_id": key,
                    "run_order": i,
                    "side": side,
                    "full_pipeline_seconds": secs,
                    "gpu_synchronized": True,
                }
                for i, (side, secs) in enumerate(order)
            ]
        timing.append(
            put(
                root,
                "timing",
                {
                    "schema": schema,
                    "frozen_plan_id": frozen["plan_id"],
                    "contract_hash": plan.contract_hash,
                    "reference_manifest_hash": frozen["models"]["king"],
                    "candidate_manifest_hash": frozen["models"]["candidate"],
                    "runtime_id": frozen["runtime"]["runtime_id"],
                    "runtime": runtime_doc,
                    "hardware": hardware,
                    "geometry": frozen["geometry"],
                    "design": {"name": design},
                    "report_video_sha256": links,
                    "measurements": measurements,
                },
            )
        )
    candidate_panel = {
        "schema": "forge/panel/v1",
        "contract_hash": plan.contract_hash,
        "admission": "public_snapshot",
        "manifest": manifests["candidate"].to_dict(),
        "manifest_hash": manifests["candidate"].fingerprint(),
        "geometry": frozen["geometry"],
        "runtime_id": frozen["runtime"]["runtime_id"],
        "runtime": runtime_doc,
        "hardware": hardware,
        "prompts": [],
        "observations": [],
    }
    rows = []
    for pid in frozen["diversity_prompt_ids"]:
        p = next(x for x in frozen["prompts"] if x["prompt_id"] == pid)
        seeds = launch_policy.diversity_seeds(frozen, pid)
        candidate_panel["prompts"].append(
            {"prompt_id": pid, "text": p["text"], "video_seeds": seeds}
        )
        videos = []
        for seed in seeds:
            video = hashlib.sha256(f"v2-div:{pid}:{seed}".encode()).hexdigest()
            videos.append(video)
            candidate_panel["observations"].append(
                {"prompt_id": pid, "seed": seed, "video_sha256": video, "status": "OK"}
            )
        rows.append(
            {
                "type": "diversity",
                "plan_id": frozen["plan_id"],
                "prompt_id": pid,
                "seeds": seeds,
                "source_video_sha256": videos,
                "decoded_video_sha256": [
                    hashlib.sha256(f"decoded:{v}".encode()).hexdigest() for v in videos
                ],
            }
        )
    source = put(root, "panels", candidate_panel)
    diversity = put(
        root,
        "diversity",
        {
            "schema": "forge/launch-diversity/v2",
            "plan_id": frozen["plan_id"],
            "source_report_sha256": source,
            "rows": rows,
            "receipt_id": hash_canonical(
                "forge/launch-diversity/v2",
                {
                    "schema": "forge/launch-diversity/v2",
                    "plan_id": frozen["plan_id"],
                    "source_report_sha256": source,
                    "rows": rows,
                },
            ),
        },
    )
    qualification = put(root, "qualification", package)
    return put(
        root,
        "bundles",
        {
            "schema": "forge/launch-execution-bundle/v2",
            "validator_hotkey": validator,
            "round_plan_hash": plan.plan_hash(),
            "parent_state_hash": plan.parent_state_hash,
            "admission_history_root": plan.prompt_seed_inputs["admission_history_root"],
            "prompt_history_root": plan.prompt_seed_inputs["prompt_history_root"],
            "launch_plan_id": frozen["plan_id"],
            "models": frozen["models"],
            "launch_plan": frozen,
            "panel_receipts": panels,
            "timing_receipts": timing,
            "score_receipts": scores,
            "diversity_receipts": [diversity],
            "qualification_receipt": qualification,
        },
    )


def test_v2_certificate_replays_same_qualified_plan_for_publication_authority(tmp_path):
    base, manifests, hardware, runtime_doc = policy()
    state = FinalizedCompetitionState.genesis("wan", manifests["king"].fingerprint())
    members = signers()
    committee = {item.validator_hotkey: 1 for item in members}
    frozen, package = _external_gpu_contract_package(
        tmp_path / "external-provider",
        candidate=manifests["candidate"].fingerprint(),
        king=manifests["king"].fingerprint(),
        anchor=state.current_king_fingerprint,
        runtime={
            key: base["runtime"][key]
            for key in ("runtime_id", "hardware_id", "dependency_lock_hash")
        },
        entropy=sha(700),
        round_id="launch-v2-authority",
    )
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=100,
        certified_state_hash=state.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    plan = round_plan(
        state, committee, frozen, history.history_root, "launch-v2-authority", sha(700)
    )
    root = tmp_path / "trusted-v2"
    bundle = _raw_evidence_v2(
        root, frozen, manifests, hardware, runtime_doc, plan, members[0].validator_hotkey, package
    )
    capacity = put(
        root,
        "capacity",
        {
            "schema": "forge/launch-capacity-receipt/v1",
            "round_plan_hash": plan.plan_hash(),
            "launch_plan_id": frozen["plan_id"],
            "measured_parallelism": 1,
            "worst_case_execution_seconds": 1.0,
            "commit_delivery_seconds": 1.0,
            "reveal_delivery_seconds": 1.0,
            "chain_block_seconds": 1.0,
        },
    )
    with Store(tmp_path / "v2.sqlite") as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), LocalLaunchReceiptStore(root))
        bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        replay = LaunchExecutionReplayer(LocalLaunchReceiptStore(root))
        assert (
            replay.replay(bundle, plan, validator_hotkey=members[0].validator_hotkey)["status"]
            == "PASS"
        ), replay.replay(bundle, plan, validator_hotkey=members[0].validator_hotkey)
        reports = [
            launch_local_report(
                plan,
                signer.validator_hotkey,
                _raw_evidence_v2(
                    root,
                    frozen,
                    manifests,
                    hardware,
                    runtime_doc,
                    plan,
                    signer.validator_hotkey,
                    package,
                ),
                replay,
            )
            for signer in members
        ]
        assert {report.verdict.value for report in reports} == {"PROMOTE"}, reports
        commits, reveals = zip(
            *(commit_report(plan, report, 150, salt=f"v2-{i}") for i, report in enumerate(reports)),
            strict=True,
        )
        certificate = build_certificate(
            plan,
            decide_round(plan, commits, reveals),
            401,
            members,
            commits=commits,
            reveals=reveals,
        )
        certified = bridge.apply(
            plan,
            state,
            certificate,
            LaunchCertificateVerifier(Sr25519Verifier(), replayer=replay, store=store),
        )
        assert certified.current_king_fingerprint == manifests["candidate"].fingerprint()
        body = {
            "network": "test",
            "netuid": 578,
            "genesis_hash": sha(801),
            "runtime_hash": sha(802),
            "mechanism": 0,
            "contract_hash": plan.contract_hash,
            "round_plan_hash": plan.plan_hash(),
            "certified_state_hash": certified.state_hash(),
            "certificate_hash": certificate.transition_hash(),
            "evidence_origin": "external_gpu_qualified_v2",
            "policy_plan": frozen,
            "qualification_package": package,
        }
        activation = {
            **body,
            "activation_plan_hash": hash_canonical("forge/launch-activation-plan/v1", body),
        }
        context = QualifiedPublicationContext(
            "bittensor-sdk/testnet-qualified-v1",
            "test",
            578,
            sha(801),
            sha(802),
            0,
            plan.contract_hash,
            plan.plan_hash(),
            certified.state_hash(),
            certificate.transition_hash(),
            activation["activation_plan_hash"],
        )
        intent = WeightIntent(
            "test",
            578,
            certified.state_hash(),
            certificate.transition_hash(),
            sha(802),
            sha(803),
            sha(804),
            "WINNER",
            {1: 1.0},
            {1: "miner"},
            "v2-contract",
        )
        assert QualifiedLaunchAuthority(
            store, plan, state, certificate, activation, str(root)
        ).verify(context, intent)


def open_context(tmp_path):
    root = tmp_path / "trusted-local-receipts"
    frozen, manifests, hardware, runtime_doc = policy()
    members = signers()
    committee = {item.validator_hotkey: 1 for item in members}
    state = FinalizedCompetitionState.genesis("wan", frozen["models"]["king"])
    history = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=100,
        certified_state_hash=state.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    )
    plan = round_plan(state, committee, frozen, history.history_root)
    capacity = put(
        root,
        "capacity",
        {
            "schema": "forge/launch-capacity-receipt/v1",
            "round_plan_hash": plan.plan_hash(),
            "launch_plan_id": frozen["plan_id"],
            "measured_parallelism": 1,
            "worst_case_execution_seconds": 1.0,
            "commit_delivery_seconds": 1.0,
            "reveal_delivery_seconds": 1.0,
            "chain_block_seconds": 1.0,
        },
    )
    return root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity


def test_launch_operator_prepares_a_future_entropy_bound_round_from_files(tmp_path):
    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, _ = open_context(
        tmp_path
    )
    entropy_hash, round_id = sha(700), "future-r1"
    prompts = launch_panel(entropy_hash, round_id=round_id)
    frozen = launch_policy.freeze(
        {
            "models": frozen["models"],
            "evaluator": frozen["evaluator"],
            "runtime": frozen["runtime"],
            "geometry": frozen["geometry"],
            "prompts": prompts,
            "timing_prompt_ids": sorted(
                (prompt["prompt_id"] for prompt in prompts),
                key=lambda prompt_id: hash_canonical(
                    "forge/launch-v1-timing/v1", entropy_hash, round_id, prompt_id
                ),
            )[:8],
            "assignments": [
                {"prompt_id": prompt["prompt_id"], **frozen["models"]} for prompt in prompts
            ],
        }
    )
    inputs = tmp_path / "operator-input"
    inputs.mkdir()
    launch_plan = inputs / "launch-plan.json"
    state_path = inputs / "state.json"
    committee_path = inputs / "committee.json"
    checkpoint_path = inputs / "checkpoint.json"
    launch_plan.write_text(json.dumps(frozen))
    state_path.write_text(json.dumps(state.to_dict()))
    committee_path.write_text(json.dumps({item.validator_hotkey: 1 for item in members}))
    checkpoint_path.write_text(
        json.dumps(
            {
                "history": history.__dict__,
                "signatures": {
                    item.validator_hotkey: item.sign(
                        __import__(
                            "forge.protocol.admission_checkpoint", fromlist=["signing_payload"]
                        ).signing_payload(history)
                    ).hex()
                    for item in members
                },
            }
        )
    )
    manifest_paths = []
    for name, item in manifests.items():
        if item.fingerprint() in {
            ModelManifest.from_dict(json.loads(path.read_text())).fingerprint()
            for path in manifest_paths
        }:
            continue
        path = inputs / f"{name}.json"
        path.write_text(json.dumps(item.to_dict()))
        manifest_paths.append(path)
    out, prompt_history = inputs / "round-plan.json", inputs / "prompt-history.json"
    command = [
        sys.executable,
        "-m",
        "forge.validator.launch_operator",
        "prepare",
        "--state",
        str(state_path),
        "--launch-plan",
        str(launch_plan),
        "--admission-checkpoint",
        str(checkpoint_path),
        "--committee",
        str(committee_path),
        "--network",
        "test",
        "--netuid",
        "578",
        "--round-id",
        round_id,
        "--candidate-hotkey",
        "miner",
        "--contract-hash",
        plan.contract_hash,
        "--close-block",
        "100",
        "--entropy-block",
        "101",
        "--entropy-hash",
        entropy_hash,
        "--commit-deadline",
        "200",
        "--reveal-deadline",
        "300",
        "--activation-block",
        "400",
        "--prompt-history-out",
        str(prompt_history),
        "--out",
        str(out),
    ]
    for manifest_path in manifest_paths:
        command.extend(["--manifest", str(manifest_path)])
    tampered = command.copy()
    tampered[tampered.index(entropy_hash)] = sha(701)
    rejected = subprocess.run(tampered, capture_output=True, text=True)
    assert rejected.returncode != 0 and "entropy" in rejected.stderr
    result = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    assert result["status"] == "PREPARED_NOT_OPEN" and out.exists() and prompt_history.exists()
    prepared = RoundPlan.from_dict(json.loads(out.read_text()))
    assert prepared.consensus_policy == ConsensusPolicy.STAKE_MAJORITY_V2
    capacity_path = inputs / "capacity.json"
    capacity_path.write_text(
        json.dumps(
            {
                "schema": "forge/launch-capacity-receipt/v1",
                "round_plan_hash": prepared.plan_hash(),
                "launch_plan_id": frozen["plan_id"],
                "measured_parallelism": 1,
                "worst_case_execution_seconds": 1.0,
                "commit_delivery_seconds": 1.0,
                "reveal_delivery_seconds": 1.0,
                "chain_block_seconds": 1.0,
            }
        )
    )
    opened = json.loads(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.validator.launch_operator",
                "open",
                "--db",
                str(tmp_path / "operator.sqlite"),
                "--receipt-root",
                str(root),
                "--state",
                str(state_path),
                "--round-plan",
                str(out),
                "--launch-plan",
                str(launch_plan),
                "--admission-checkpoint",
                str(checkpoint_path),
                "--capacity",
                str(capacity_path),
                "--out",
                str(inputs / "opened.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert opened["status"] == "OPENED"
    reopened = json.loads(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.validator.launch_operator",
                "open",
                "--db",
                str(tmp_path / "operator.sqlite"),
                "--receipt-root",
                str(root),
                "--state",
                str(state_path),
                "--round-plan",
                str(out),
                "--launch-plan",
                str(launch_plan),
                "--admission-checkpoint",
                str(checkpoint_path),
                "--capacity",
                str(capacity_path),
                "--out",
                str(inputs / "opened.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert reopened["status"] == "ALREADY_OPEN"
    # The raw fixture has non-canonical whitespace; import must retain its source bytes.
    raw_bundle = raw_evidence(
        root, frozen, manifests, hardware, runtime_doc, prepared, members[0].validator_hotkey
    )
    raw = json.loads((root / "bundles" / f"{raw_bundle}.json").read_text())
    panel_paths = [root / "panels" / f"{digest}.json" for digest in raw["panel_receipts"]]
    timing_paths = [root / "timing" / f"{digest}.json" for digest in raw["timing_receipts"]]
    score_paths = [root / "scores" / f"{digest}.json" for digest in raw["score_receipts"]]
    bundle_command = [
        sys.executable,
        "-m",
        "forge.validator.launch_operator",
        "bundle",
        "--receipt-root",
        str(root),
        "--round-plan",
        str(out),
        "--launch-plan",
        str(launch_plan),
        "--validator-hotkey",
        members[0].validator_hotkey,
        "--out",
        str(inputs / "bundle.json"),
    ]
    for flag, paths in (
        ("--panel", panel_paths),
        ("--timing", timing_paths),
        ("--score", score_paths),
    ):
        for path in paths:
            bundle_command.extend([flag, str(path)])
    subprocess.run(bundle_command, check=True, capture_output=True, text=True)
    report_result = json.loads(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.validator.launch_operator",
                "report",
                "--receipt-root",
                str(root),
                "--round-plan",
                str(out),
                "--bundle",
                str(inputs / "bundle.json"),
                "--validator-hotkey",
                members[0].validator_hotkey,
                "--out",
                str(inputs / "report.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert report_result["verdict"] == "PROMOTE"
    with Store(tmp_path / "operator.sqlite") as store:
        assert store.count("rounds") == 1


def test_launch_bridge_promotes_after_restart_and_prepares_testnet_intent(tmp_path):
    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity = (
        open_context(tmp_path)
    )
    db = tmp_path / "state.sqlite"
    receipts = LocalLaunchReceiptStore(root)
    with Store(db) as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
        assert bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        assert not bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        replayer = LaunchExecutionReplayer(receipts)
        reports = [
            launch_local_report(
                plan,
                signer.validator_hotkey,
                raw_evidence(
                    root, frozen, manifests, hardware, runtime_doc, plan, signer.validator_hotkey
                ),
                replayer,
            )
            for signer in members
        ]
        assert {report.verdict.value for report in reports} == {"PROMOTE"}
        pairs = [
            commit_report(plan, report, 150, salt=f"salt-{index}")
            for index, report in enumerate(reports)
        ]
        commits, reveals = zip(*pairs, strict=True)
        decision = decide_round(plan, commits, reveals)
        certificate = build_certificate(
            plan, decision, 401, members, commits=commits, reveals=reveals
        )
        next_state = bridge.apply(
            plan,
            state,
            certificate,
            LaunchCertificateVerifier(Sr25519Verifier(), replayer=replayer, store=store),
        )
        assert next_state.current_king_fingerprint == frozen["models"]["candidate"]
        custody = put(
            root,
            "custody",
            {
                "schema": "forge/launch-public-custody/v1",
                "fingerprint": frozen["models"]["candidate"],
                "publicly_downloadable": True,
            },
        )
        burn_hotkey = members[0].validator_hotkey
        burn = put(
            root,
            "burn",
            {
                "schema": "forge/launch-burn-qualification/v1",
                "hotkey": burn_hotkey,
                "runtime_hash": sha(88),
                "network": plan.network,
                "netuid": plan.netuid,
            },
        )
        intent = bridge.prepare_testnet_weight_intent(
            next_state,
            plan=plan,
            certificate=certificate,
            runtime_hash=sha(88),
            registered_uids={0: burn_hotkey, 1: "miner"},
            public_custody={frozen["models"]["candidate"]: custody},
            burn_hotkey=burn_hotkey,
            burn_receipt_hash=burn,
        )
        assert intent.status == "WINNER" and intent.uid_weights == {1: 1.0}
        foreign_plan = replace(plan, network="foreign-test")
        with pytest.raises(LaunchBridgeError, match="stored round plan"):
            bridge.prepare_testnet_weight_intent(
                next_state,
                plan=foreign_plan,
                certificate=certificate,
                runtime_hash=sha(88),
                registered_uids={0: burn_hotkey, 1: "miner"},
                public_custody={frozen["models"]["candidate"]: custody},
                burn_hotkey=burn_hotkey,
                burn_receipt_hash=burn,
            )

        # The second crown keeps the immutable genesis anchor, not the current king.
        frozen2, _, _, _ = policy(
            candidate_number=5,
            king_number=3,
            anchor_number=4,
            entropy_hash=sha(701),
            round_id="launch-r2",
        )
        history2 = build_authenticated_history(
            network="test",
            netuid=578,
            through_block=100,
            certified_state_hash=next_state.state_hash(),
            consumed_hotkeys=(),
            registered_hotkeys=("miner",),
        )
        plan2 = round_plan(
            next_state,
            committee := {item.validator_hotkey: 1 for item in members},
            frozen2,
            history2.history_root,
            "launch-r2",
            sha(701),
        )
        capacity2 = put(
            root,
            "capacity",
            {
                "schema": "forge/launch-capacity-receipt/v1",
                "round_plan_hash": plan2.plan_hash(),
                "launch_plan_id": frozen2["plan_id"],
                "measured_parallelism": 1,
                "worst_case_execution_seconds": 1.0,
                "commit_delivery_seconds": 1.0,
                "reveal_delivery_seconds": 1.0,
                "chain_block_seconds": 1.0,
            },
        )
        assert bridge.open_round(
            plan2,
            next_state,
            create_signed_history(history2, members),
            frozen2,
            LaunchV1Capacity(capacity2),
        )
        retouched, _, _, _ = policy(
            candidate_number=5,
            king_number=3,
            anchor_number=3,
            entropy_hash=sha(701),
            round_id="launch-anchor-retouched",
        )
        retouched_plan = round_plan(
            next_state,
            committee,
            retouched,
            history2.history_root,
            "launch-anchor-retouched",
            sha(701),
        )
        with pytest.raises(LaunchBridgeError, match="certified state"):
            bridge.open_round(
                retouched_plan,
                next_state,
                create_signed_history(history2, members),
                retouched,
                LaunchV1Capacity(sha(123)),
            )
    with Store(db) as restarted:
        assert (
            restarted.latest_state_payload("wan")["current_king_fingerprint"]
            == frozen["models"]["candidate"]
        )


def test_launch_bridge_rejects_rehashed_semantic_tampering_and_incomplete_certificate(tmp_path):
    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity = (
        open_context(tmp_path)
    )
    receipts, replayer = (
        LocalLaunchReceiptStore(root),
        LaunchExecutionReplayer(LocalLaunchReceiptStore(root)),
    )
    bundle = raw_evidence(
        root, frozen, manifests, hardware, runtime_doc, plan, members[0].validator_hotkey
    )
    original = receipts.read("bundles", bundle)
    entropy_tampered_plan = RoundPlan(
        **{
            **plan.__dict__,
            "prompt_seed_inputs": {**plan.prompt_seed_inputs, "entropy_hash": sha(998)},
        }
    )
    assert replayer.replay(bundle, entropy_tampered_plan)["status"] == "INCOMPLETE"
    # Rehashing passes filesystem integrity; bindings still reject changed semantics.
    bad_model = {**original, "models": {**original["models"], "candidate": sha(99)}}
    assert replayer.replay(put(root, "bundles", bad_model), plan)["status"] == "INCOMPLETE"
    bad_score = receipts.read("scores", original["score_receipts"][0])
    bad_score["prompt_text"] = "changed"
    bad_bundle = {
        **original,
        "score_receipts": [put(root, "scores", bad_score), *original["score_receipts"][1:]],
    }
    assert replayer.replay(put(root, "bundles", bad_bundle), plan)["status"] == "INCOMPLETE"
    bad_timing = receipts.read("timing", original["timing_receipts"][0])
    bad_timing["measurements"][0]["seed"] = 999
    bad_bundle = {**original, "timing_receipts": [put(root, "timing", bad_timing)]}
    assert replayer.replay(put(root, "bundles", bad_bundle), plan)["status"] == "INCOMPLETE"
    report = launch_local_report(
        plan, members[0].validator_hotkey, put(root, "bundles", bad_bundle), replayer
    )
    assert report.verdict.value == "INCOMPLETE"
    with Store(tmp_path / "incomplete.sqlite") as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
        bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        # A syntactically valid promotion certificate cannot launder an INCOMPLETE replay.
        valid_reports = []
        for member in members:
            valid_bundle = raw_evidence(
                root, frozen, manifests, hardware, runtime_doc, plan, member.validator_hotkey
            )
            valid_reports.append(
                launch_local_report(plan, member.validator_hotkey, valid_bundle, replayer)
            )
        incomplete_bundle = put(root, "bundles", bad_bundle)
        laundered = [
            replace(
                value,
                metrics={
                    **value.metrics,
                    "launch_bundle_hash": incomplete_bundle,
                    "launch_status": "PROMOTE_ELIGIBLE",
                },
            )
            for value in valid_reports
        ]
        pairs = [
            commit_report(plan, value, 150, salt=f"incomplete-{i}")
            for i, value in enumerate(laundered)
        ]
        commits, reveals = zip(*pairs, strict=True)
        certificate = build_certificate(
            plan,
            decide_round(plan, commits, reveals),
            401,
            members,
            commits=commits,
            reveals=reveals,
        )
        with pytest.raises(CertificateError, match="trusted V1 receipt replay"):
            LaunchCertificateVerifier(Sr25519Verifier(), replayer=replayer, store=store).verify(
                certificate, plan, state
            )


def test_launch_v1_round_expires_without_inventing_a_keep_certificate(tmp_path):
    root, frozen, _, _, _, members, state, history, plan, capacity = open_context(tmp_path)
    with Store(tmp_path / "expiry.sqlite") as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), LocalLaunchReceiptStore(root))
        assert bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        receipt = RoundMailbox(store, Sr25519Verifier()).expire(
            plan, finalized_block=301, finalized_hash="ab" * 32
        )
        assert receipt["status"] == "EXPIRED_NO_CERTIFICATE"
        assert store.count("certificates") == 0


def test_launch_open_rejects_old_parent(tmp_path):
    root, frozen, _, _, _, members, state, history, plan, capacity = open_context(tmp_path)
    old_parent = RoundPlan(**{**plan.__dict__, "parent_state_hash": sha(777)})
    with Store(tmp_path / "old-parent.sqlite") as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), LocalLaunchReceiptStore(root))
        with pytest.raises(LaunchBridgeError):
            bridge.open_round(
                old_parent,
                state,
                create_signed_history(history, members),
                frozen,
                LaunchV1Capacity(capacity),
            )
        entropy_retouched = RoundPlan(
            **{
                **plan.__dict__,
                "round_id": "launch-entropy-retouched",
                "prompt_seed_inputs": {
                    **plan.prompt_seed_inputs,
                    "entropy_hash": sha(998),
                },
            }
        )
        with pytest.raises(LaunchBridgeError, match="entropy panel"):
            bridge.open_round(
                entropy_retouched,
                state,
                create_signed_history(history, members),
                frozen,
                LaunchV1Capacity(capacity),
            )


def test_launch_open_counts_capacity_from_entropy_not_close(tmp_path):
    root, frozen, _, _, _, members, state, history, plan, _ = open_context(tmp_path)
    # Two blocks from close to commit look sufficient, but entropy arrives one
    # block before commit and the two-second execution-plus-delivery work no
    # longer fits the usable window.
    short_window = replace(
        plan,
        commit_deadline_block=103,
        reveal_deadline_block=104,
        activation_block=105,
        prompt_seed_inputs={**plan.prompt_seed_inputs, "entropy_block": 102},
    )
    capacity = put(
        root,
        "capacity",
        {
            "schema": "forge/launch-capacity-receipt/v1",
            "round_plan_hash": short_window.plan_hash(),
            "launch_plan_id": frozen["plan_id"],
            "measured_parallelism": 1,
            "worst_case_execution_seconds": 1.0,
            "commit_delivery_seconds": 1.0,
            "reveal_delivery_seconds": 1.0,
            "chain_block_seconds": 1.0,
        },
    )
    with Store(tmp_path / "capacity-after-entropy.sqlite") as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), LocalLaunchReceiptStore(root))
        with pytest.raises(LaunchBridgeError, match="deadlines do not fit"):
            bridge.open_round(
                short_window,
                state,
                create_signed_history(history, members),
                frozen,
                LaunchV1Capacity(capacity),
            )


def test_launch_voting_replays_signed_mailbox_after_restart_and_fails_closed_on_receipts(tmp_path):
    """Signed transport is accepted only after the local protected receipt replay."""
    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity = (
        open_context(tmp_path)
    )
    receipts = LocalLaunchReceiptStore(root)
    bundles = {
        signer.validator_hotkey: raw_evidence(
            root, frozen, manifests, hardware, runtime_doc, plan, signer.validator_hotkey
        )
        for signer in members
    }
    voter_db, tally_db = tmp_path / "voter.sqlite", tmp_path / "tally.sqlite"
    signed_commits = []
    with Store(voter_db) as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
        assert bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        failed = _ControlledSigner(members[0], fail=True)
        with pytest.raises(RuntimeError, match="external signer unavailable"):
            workflow.commit_envelope(
                plan,
                failed,
                bundle_hash=bundles[failed.validator_hotkey],
                finalized_block=150,
                salt="external-gpu-fixture-salt-0",
            )
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM launch_v1_commit_reservations"
            ).fetchone()[0]
            == 1
        )

    with Store(voter_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        conflicting = _ControlledSigner(members[0])
        with pytest.raises(LaunchVotingError, match="reserved different execution"):
            workflow.commit_envelope(
                plan,
                conflicting,
                bundle_hash=bundles[conflicting.validator_hotkey],
                finalized_block=150,
                salt="different-salt",
            )
        assert conflicting.calls == 0
        signed_commits = [
            workflow.commit_envelope(
                plan,
                members[0],
                bundle_hash=bundles[members[0].validator_hotkey],
                finalized_block=150,
                salt="external-gpu-fixture-salt-0",
            )
        ]
        signed_commits.extend(
            workflow.commit_envelope(
                plan,
                signer,
                bundle_hash=bundles[signer.validator_hotkey],
                finalized_block=150,
                salt=f"external-gpu-fixture-salt-{index}",
            )
            for index, signer in enumerate(members[1:], start=1)
        )

    with Store(tally_db) as store:
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
        assert bridge.open_round(
            plan, state, create_signed_history(history, members), frozen, LaunchV1Capacity(capacity)
        )
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        assert all(
            workflow.receive(plan, envelope, finalized_block=150) for envelope in signed_commits
        )

    with Store(voter_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        signed_reveals = [
            workflow.reveal_envelope(plan, signer, finalized_block=201) for signer in members
        ]

    with Store(tally_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        assert all(
            workflow.receive(plan, envelope, finalized_block=201) for envelope in signed_reveals
        )
        with pytest.raises(LaunchVotingError, match="certificate"):
            workflow.certificate(
                plan,
                activation_chain_time=401,
                signers=(members[0], _ControlledSigner(members[1], fail=True), members[2]),
            )
        assert store.count("attestations") == len(members)

    with Store(tally_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        conflicting = _ControlledSigner(members[0])
        with pytest.raises(LaunchVotingError, match="certificate"):
            workflow.certificate(
                plan, activation_chain_time=402, signers=(conflicting, members[1], members[2])
            )
        assert conflicting.calls == 0
        certificate = workflow.certificate(plan, activation_chain_time=401, signers=members)
        assert certificate.transition.value == "PROMOTE"
    # A missing or rehashed source cannot turn an authenticated envelope into a certificate.
    bundle = receipts.read("bundles", bundles[members[0].validator_hotkey])
    absent = root / "scores" / f"{bundle['score_receipts'][0]}.json"
    absent.rename(absent.with_suffix(".missing"))
    with Store(tally_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        with pytest.raises(LaunchVotingError, match="incomplete|certificate"):
            workflow.certificate(plan, activation_chain_time=401, signers=members)
    absent.with_suffix(".missing").rename(absent)
    timing = root / "timing" / f"{bundle['timing_receipts'][0]}.json"
    timing_original = timing.read_bytes()
    timing.write_bytes(timing_original + b" ")
    with Store(tally_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        with pytest.raises(LaunchVotingError, match="incomplete|certificate"):
            workflow.certificate(plan, activation_chain_time=401, signers=members)
    timing.write_bytes(timing_original)
    with Store(tally_db) as store:
        workflow = LaunchV1VotingWorkflow(
            store, Sr25519Verifier(), LaunchExecutionReplayer(receipts)
        )
        certificate = workflow.certificate(plan, activation_chain_time=401, signers=members)
        bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
        next_state = bridge.apply(
            plan,
            state,
            certificate,
            LaunchCertificateVerifier(
                Sr25519Verifier(), replayer=LaunchExecutionReplayer(receipts), store=store
            ),
        )
        assert next_state.current_king_fingerprint == frozen["models"]["candidate"]


@pytest.mark.parametrize("transition", ["PROMOTE", "KEEP"])
def test_certified_dashboard_requires_replayed_certificate_and_final_weight_readbacks(
    tmp_path, transition
):
    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity = (
        open_context(tmp_path)
    )
    replayer = LaunchExecutionReplayer(LocalLaunchReceiptStore(root))
    reports = [
        launch_local_report(
            plan,
            signer.validator_hotkey,
            raw_evidence(
                root,
                frozen,
                manifests,
                hardware,
                runtime_doc,
                plan,
                signer.validator_hotkey,
                candidate_score=10.0 if transition == "PROMOTE" else -10.0,
            ),
            replayer,
        )
        for signer in members
    ]
    pairs = [
        commit_report(plan, report, 150, salt=f"dashboard-{index}")
        for index, report in enumerate(reports)
    ]
    commits, reveals = zip(*pairs, strict=True)
    certificate = build_certificate(
        plan, decide_round(plan, commits, reveals), 401, members, commits=commits, reveals=reveals
    )
    run = tmp_path / "launch-v1-certified-fixture"
    shutil.copytree(root, run / "trusted-receipts")
    custody = {
        "schema": "forge/launch-public-custody/v1",
        "fingerprint": plan.challenger_fingerprint,
        "publicly_downloadable": True,
    }
    burn = {
        "schema": "forge/launch-burn-qualification/v1",
        "hotkey": members[0].validator_hotkey,
        "runtime_hash": sha(88),
        "network": "test",
        "netuid": 578,
    }
    for name, value, kind in (
        ("public-custody.json", custody, "custody"),
        ("burn-qualification.json", burn, "burn"),
    ):
        path = run / name
        path.write_text(json.dumps(value))
        raw = path.read_bytes()
        (run / "trusted-receipts" / kind).mkdir(parents=True, exist_ok=True)
        (run / "trusted-receipts" / kind / f"{hashlib.sha256(raw).hexdigest()}.json").write_bytes(
            raw
        )
    for label in ("a", "b"):
        db = run / f"validator-{label}.sqlite"
        with Store(db) as store:
            bridge = LaunchV1Bridge(
                store, Sr25519Verifier(), LocalLaunchReceiptStore(run / "trusted-receipts")
            )
            bridge.open_round(
                plan,
                state,
                create_signed_history(history, members),
                frozen,
                LaunchV1Capacity(capacity),
            )
            certified = bridge.apply(
                plan,
                state,
                certificate,
                LaunchCertificateVerifier(
                    Sr25519Verifier(),
                    replayer=LaunchExecutionReplayer(
                        LocalLaunchReceiptStore(run / "trusted-receipts")
                    ),
                    store=store,
                ),
            )

    def write(name, value):
        (run / name).write_text(json.dumps(value))

    write("round-plan.json", plan.to_dict())
    write("state.json", state.to_dict())
    write("certificate.json", certificate.to_dict())
    write("certified-state-a.json", certified.to_dict())
    write("certified-state-b.json", certified.to_dict())
    journals, readback = {}, []
    for index, label in enumerate(("a", "b")):
        custody_hash = hashlib.sha256((run / "public-custody.json").read_bytes()).hexdigest()
        burn_hash = hashlib.sha256((run / "burn-qualification.json").read_bytes()).hexdigest()
        with Store(run / "validator-a.sqlite") as intent_store:
            intent = LaunchV1Bridge(
                intent_store, Sr25519Verifier(), LocalLaunchReceiptStore(run / "trusted-receipts")
            ).prepare_testnet_weight_intent(
                certified,
                plan=plan,
                certificate=certificate,
                runtime_hash=sha(88),
                registered_uids={
                    0: members[0].validator_hotkey,
                    1: "miner",
                    2: members[1].validator_hotkey,
                },
                public_custody={plan.challenger_fingerprint: custody_hash},
                burn_hotkey=members[0].validator_hotkey,
                burn_receipt_hash=burn_hash,
            )
        receipt = WeightSubmissionReceipt(intent.intent_hash(), "test", 578, 401, sha(91 + index))
        journals[label] = {
            "intent": intent.__dict__,
            "intent_hash": intent.intent_hash(),
            "validator_hotkey": members[index].validator_hotkey,
            "validator_uid": index * 2,
            "submission_receipt": receipt.__dict__,
            "extrinsic_id": f"401-{index}",
        }
        write(f"weight-submission-{label}.json", journals[label])
        readback.append(
            {
                "validator": label,
                "status": "FINALIZED_WEIGHT_VECTOR_VERIFIED",
                "weights_u16": [[next(iter(intent.uid_weights)), 65535]],
                "intent_hash": intent.intent_hash(),
                "validator_uid": index * 2,
                "validator_hotkey": members[index].validator_hotkey,
                "finalized_extrinsic_hash": receipt.submitted_extrinsic_hash,
                "extrinsic_id": f"401-{index}",
            }
        )
    write(
        "weight-readback.json",
        {
            "status": "FINALIZED_WEIGHTS_VERIFIED",
            "network": "test",
            "netuid": 578,
            "block": 402,
            "block_hash": sha(92),
            "runtime_hash": sha(88),
            "registered_uids": {
                "0": members[0].validator_hotkey,
                "1": "miner",
                "2": members[1].validator_hotkey,
            },
            "certificate_hash": certificate.transition_hash(),
            "transition": transition,
            "results": readback,
        },
    )
    campaign = run.name
    cid = f"launch-v1-{campaign}-{plan.challenger_fingerprint[:12]}"
    snapshot = {
        "schema": "forge/dashboard/v2",
        "updated_at": "2026-10-15T00:00:00Z",
        "network": {"name": "test", "netuid": 578},
        "base_model": {"name": "synthetic fixture", "steps": 50},
        "contract": {
            "min_speedup": 1.1,
            "quality_margin": 0.0,
            "axes": [],
            "consensus_rule": plan.consensus_policy.value,
        },
        "validators": [],
        "king": None,
        "crowns": [],
        "challengers": [
            {
                "id": cid,
                "round_id": f"operator-experiment-{campaign}",
                "manifest_hash": plan.challenger_fingerprint,
                "model": {"repo": "fixture", "steps": 16},
                "hotkey": "miner",
                "label": "fixture",
                "status": "diagnostic",
            }
        ],
        "evaluations": [
            {
                "challenger_id": cid,
                "plan_hash": frozen["plan_id"],
                "verdict": {"status": "DIAGNOSTIC"},
            }
        ],
        "rounds": [
            {
                "round_id": f"operator-experiment-{campaign}",
                "created_at": "2026-10-15T00:00:00Z",
                "diagnostic": True,
                "challengers": [cid],
            }
        ],
    }
    exported = export_certified_snapshot(snapshot, run)
    if transition == "PROMOTE":
        assert exported["king"]["manifest_hash"] == plan.challenger_fingerprint
        assert exported["king"]["uid"] == 1
    else:
        assert exported["king"] is None and exported["crowns"] == []
        assert intent.status == "BURN_REQUIRED" and intent.uid_weights == {0: 1.0}
    assert exported["evaluations"][0]["verdict"]["status"] == transition
    assert (
        exported["launch_v1_certification"]["activation"]["status"] == "FINALIZED_WEIGHTS_VERIFIED"
    )
    assert exported["network"]["finalized_block"] == 402
    assert {r["anchor_block"] for r in exported["evaluations"][0]["consensus"]["reports"]} == {150}
    # Keep a serialized cross-language boundary artifact for the dashboard's real checker.
    write("exported-snapshot.json", exported)
    valid_readback = json.loads((run / "weight-readback.json").read_text())
    duplicate = json.loads(json.dumps(valid_readback))
    duplicate["results"][1] = {**duplicate["results"][0], "validator": "b"}
    write("weight-submission-b.json", journals["a"])
    write("weight-readback.json", duplicate)
    with pytest.raises(CertifiedDashboardError, match="distinct"):
        export_certified_snapshot(snapshot, run)
    write("weight-submission-b.json", journals["b"])
    write("weight-readback.json", valid_readback)
    # Hash-consistent forged journals cannot redirect a certified winner to another UID.
    forged = replace(intent, uid_weights={9: 1.0}, expected_hotkeys={9: "other-miner"})
    fake = json.loads(json.dumps(valid_readback))
    fake["registered_uids"]["9"] = "other-miner"
    for index, label in enumerate(("a", "b")):
        journal = json.loads(json.dumps(journals[label]))
        journal["intent"] = forged.__dict__
        journal["intent_hash"] = forged.intent_hash()
        journal["submission_receipt"]["intent_hash"] = forged.intent_hash()
        fake["results"][index].update(intent_hash=forged.intent_hash(), weights_u16=[[9, 65535]])
        write(f"weight-submission-{label}.json", journal)
    write("weight-readback.json", fake)
    with pytest.raises(CertifiedDashboardError, match="reconstructed"):
        export_certified_snapshot(snapshot, run)
    for label in ("a", "b"):
        write(f"weight-submission-{label}.json", journals[label])
    write("weight-readback.json", valid_readback)
    readback[0]["weights_u16"] = {"1": 1}
    write(
        "weight-readback.json",
        {
            "status": "FINALIZED_WEIGHTS_VERIFIED",
            "network": "test",
            "netuid": 578,
            "block": 402,
            "block_hash": sha(92),
            "runtime_hash": sha(88),
            "registered_uids": {
                "0": members[0].validator_hotkey,
                "1": "miner",
                "2": members[1].validator_hotkey,
            },
            "certificate_hash": certificate.transition_hash(),
            "transition": transition,
            "results": readback,
        },
    )
    with pytest.raises(CertifiedDashboardError, match="replacement"):
        export_certified_snapshot(snapshot, run)
    readback[0]["weights_u16"] = {"1": 65535}
    write(
        "weight-readback.json",
        {
            "status": "FINALIZED_WEIGHTS_VERIFIED",
            "network": "test",
            "netuid": 578,
            "block": 402,
            "block_hash": sha(92),
            "runtime_hash": sha(88),
            "certificate_hash": sha(999),
            "transition": transition,
            "results": readback,
        },
    )
    with pytest.raises(CertifiedDashboardError, match="readback"):
        export_certified_snapshot(snapshot, run)


def test_launch_consensus_runner_restarts_after_ambiguous_chain_commit(tmp_path):
    """Real SQLite/signatures/evidence; only the unavailable chain is simulated."""
    from forge.chain import decode_commitment
    from forge.validator.launch_consensus import LaunchConsensusRunner

    root, frozen, manifests, hardware, runtime_doc, members, state, history, plan, capacity = (
        open_context(tmp_path)
    )
    run = tmp_path / "operator"
    run.mkdir()
    wallets = {
        f"v{index}": SimpleNamespace(hotkey=signer._keypair) for index, signer in enumerate(members)
    }
    for name, value in (("round-plan.json", plan.to_dict()), ("state.json", state.to_dict())):
        (run / name).write_text(json.dumps(value, indent=2))
    for label, member in zip(wallets, members, strict=True):
        digest = raw_evidence(
            root,
            frozen,
            manifests,
            hardware,
            runtime_doc,
            plan,
            member.validator_hotkey,
        )
        (run / f"bundle-{label}.json").write_text(json.dumps({"bundle_hash": digest}))
    shutil.copytree(root, run / "trusted-receipts")
    for label in wallets:
        with Store(run / f"validator-{label}.sqlite") as store:
            LaunchV1Bridge(
                store, Sr25519Verifier(), LocalLaunchReceiptStore(run / "trusted-receipts")
            ).open_round(
                plan,
                state,
                create_signed_history(history, members),
                frozen,
                LaunchV1Capacity(capacity),
            )

    class ChainBoundary:
        network, netuid = "test", 578
        height, writes = 150, 0
        fail_after_inclusion = True

        def __init__(self):
            self.rows = {}

        def finalized_height(self):
            return self.height

        def finalized_block_hash(self, height):
            return sha(height)

        def chain_time(self, height):
            return height

        def commitment(self, hotkey, height):
            return self.rows.get(hotkey)

        def publish(self, wallet, text):
            self.writes += 1
            hotkey = wallet.hotkey.ss58_address
            self.rows[hotkey] = decode_commitment(hotkey, self.height, text)
            if self.fail_after_inclusion:
                self.fail_after_inclusion = False
                raise TimeoutError("external chain accepted; response lost")
            return f"{self.height}-0000"

    chain = ChainBoundary()
    with pytest.raises(TimeoutError):
        LaunchConsensusRunner(run, chain=chain, wallets=wallets).tick()
    assert chain.writes == 1
    # Fresh runner recovers the finalized commitment, preserving the same salt.
    assert LaunchConsensusRunner(run, chain=chain, wallets=wallets).tick()["status"] == (
        "COMMITTED_WAITING_REVEAL"
    )
    assert chain.writes == len(wallets)
    # Reformat bytes as SQLite historically did: values, not key order, define replay.
    original = json.loads((run / "commit-envelope-v0.json").read_text())
    (run / "commit-envelope-v0.json").write_text(json.dumps(original, sort_keys=False, indent=4))
    LaunchConsensusRunner(run, chain=chain, wallets=wallets).tick()
    assert chain.writes == len(wallets)
    chain.height = plan.commit_deadline_block + 1
    assert LaunchConsensusRunner(run, chain=chain, wallets=wallets).tick()["status"] == (
        "REVEALED_WAITING_ACTIVATION"
    )
    chain.height = plan.activation_block
    result = LaunchConsensusRunner(run, chain=chain, wallets=wallets).tick()
    assert result["status"] == "CERTIFIED_NOT_YET_WEIGHTS"
    assert result["transition"] == "PROMOTE"
    assert LaunchConsensusRunner(run, chain=chain, wallets=wallets).tick() == result
    for label in wallets:
        with Store(run / f"validator-{label}.sqlite") as store:
            assert store.count("certificates") == 1
    assert chain.writes == len(wallets)
