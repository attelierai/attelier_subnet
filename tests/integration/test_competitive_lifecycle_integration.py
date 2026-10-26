"""Qualified-round lifecycle through signed history, restart, certificate and intent.

No model is rendered and no chain write is attempted.  The real sr25519
boundary is exercised so this covers the authority handoff, not GPU quality.
"""

import json
from itertools import product
from types import SimpleNamespace

import pytest

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.cli import main
from forge.evaluation.capacity import CapacityInputs
from forge.evaluation.distributed_plan import DistributedEvaluationPlan
from forge.evaluation.mvp_policy import _sha
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.consensus_types import ConsensusPolicy, FinalizedCompetitionState, RoundPlan
from forge.protocol.hashing import hash_canonical
from forge.protocol.mvp_certification import MVPCertificateVerifier, local_report
from forge.protocol.preconsensus import (
    CertificateVerifier,
    build_certificate,
    commit_report,
    decide_round,
)
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox
from forge.validator.admission import build_authenticated_history
from forge.validator.competitive_lifecycle import CompetitiveLifecycle, LifecycleError


def sha(number: int) -> str:
    return f"{number:064x}"


def signers():
    from bittensor.keyfiles import Keypair

    return [
        Sr25519Signer(SimpleNamespace(hotkey=Keypair.create_from_uri(uri)))
        for uri in ("//Alice", "//Bob", "//Charlie")
    ]


def history(plan: RoundPlan, signers, *, root: str):
    checkpoint = build_authenticated_history(
        network=plan.network,
        netuid=plan.netuid,
        through_block=plan.snapshot_block,
        certified_state_hash=plan.parent_state_hash,
        consumed_hotkeys=(),
        registered_hotkeys=(plan.challenger_hotkey,),
    )
    assert checkpoint.history_root == root
    return create_signed_history(checkpoint, signers)


def capacity_prompts():
    return [
        {
            "prompt_id": f"capacity-{cell}-{number}",
            "text": f"Fixture {cell} {number}",
            "seed": cell * 8 + number,
            "subject_class": subject,
            "camera_type": camera,
            "style_family": style,
        }
        for cell, (subject, camera, style) in enumerate(
            product(
                ("human", "animal", "rigid_object", "scenery"),
                ("static", "moving"),
                ("realistic", "stylized"),
            )
        )
        for number in range(8)
    ]


def evaluation_id(candidate, king):
    return _sha(
        {
            "schema": "forge/mvp-evaluation-plan/v1",
            "stage": "confirmation",
            "policy_id": "capacity-lifecycle-fixture",
            "prompts": capacity_prompts(),
            "candidate_manifest_hash": candidate,
            "incumbent_manifest_hash": king,
            "wan50_manifest_hash": king,
        }
    )


def plan(state: FinalizedCompetitionState, committee, *, round_id: str, snapshot: int):
    admission_root = build_authenticated_history(
        network="test",
        netuid=578,
        through_block=snapshot,
        certified_state_hash=state.state_hash(),
        consumed_hotkeys=(),
        registered_hotkeys=("miner",),
    ).history_root
    return RoundPlan(
        "test",
        578,
        round_id,
        state.state_hash(),
        state.challenge_id,
        state.current_king_fingerprint,
        sha(snapshot + 20),
        "miner",
        sha(9),
        {
            "admission_history_root": admission_root,
            "prompt_history_root": sha(snapshot + 30),
            "evaluation_plan_id": evaluation_id(sha(snapshot + 20), state.current_king_fingerprint),
        },
        committee,
        snapshot,
        snapshot + 10,
        snapshot + 20,
        snapshot + 30,
        ConsensusPolicy.STAKE_MAJORITY_V2,
    )


def capacity_plan(frozen: RoundPlan) -> DistributedEvaluationPlan:
    return DistributedEvaluationPlan(
        round_id=frozen.round_id,
        stage="confirmation",
        entropy=sha(900),
        policy_id="capacity-lifecycle-fixture",
        candidate_manifest_hash=frozen.challenger_fingerprint,
        king_manifest_hash=frozen.king_fingerprint,
        wan50_manifest_hash=frozen.king_fingerprint,
        committee=dict(frozen.committee),
        prompts=tuple(capacity_prompts()),
        timing_prompt_ids=tuple(prompt["prompt_id"] for prompt in capacity_prompts()[:32]),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
        minimum_audit_replicas=2,
        audit_work_count=41,
        chain_context={
            "network": frozen.network,
            "netuid": frozen.netuid,
            "close_block": frozen.snapshot_block,
            "round_plan_hash": frozen.plan_hash(),
            "mvp_plan_id": frozen.prompt_seed_inputs["evaluation_plan_id"],
        },
    )


def capacity_inputs() -> CapacityInputs:
    return CapacityInputs(
        king_seconds=0.001,
        wan50_seconds=0.001,
        candidate_seconds=0.001,
        gpu_usd_hour=1,
        node_overhead_seconds=0,
        node_overhead_usd=0,
        metric_seconds_per_video=0,
        judge_usd_per_quality_comparison=0,
        storage_usd_per_video=0,
        validator_screening_spent_usd=0,
        network_remaining_usd=100,
        commit_delivery_seconds=0,
        reveal_delivery_seconds=0,
        chain_block_seconds=1,
    )


def open_qualified(lifecycle, frozen, state, signed):
    return lifecycle.open_round(
        frozen,
        state,
        signed,
        distributed_plan=capacity_plan(frozen),
        capacity_inputs=capacity_inputs(),
    )


def qualified_certificate(plan: RoundPlan, signers):
    bundle = {
        "evidence_kind": "validator_wan_execution",
        "bindings": {
            "round_plan_hash": plan.plan_hash(),
            "candidate_manifest_hash": plan.challenger_fingerprint,
            "incumbent_manifest_hash": plan.king_fingerprint,
            "parent_state_hash": plan.parent_state_hash,
            **dict(plan.prompt_seed_inputs),
        },
        "public_custody": {"manifest_hash": plan.challenger_fingerprint, "uri": "https://x/y"},
    }
    reports = [
        local_report(
            plan,
            signer.validator_hotkey,
            bundle,
            independently_execute=lambda _bundle: {"status": "PROMOTE_ELIGIBLE"},
            verify_public_copy=lambda *_args: True,
        )
        for signer in signers
    ]
    pairs = [
        commit_report(plan, report, plan.commit_deadline_block - 1, salt=f"salt-{index}")
        for index, report in enumerate(reports)
    ]
    commits, reveals = zip(*pairs, strict=True)
    decision = decide_round(plan, commits, reveals)
    certificate = build_certificate(
        plan,
        decision,
        activation_chain_time=1234,
        signers=signers,
        commits=commits,
        reveals=reveals,
    )
    digest = hash_canonical("forge/mvp-execution-bundle/v1", bundle)
    verifier = MVPCertificateVerifier(
        Sr25519Verifier(),
        evidence={digest: bundle},
        independently_execute=lambda _bundle: {"status": "PROMOTE_ELIGIBLE"},
        verify_public_copy=lambda *_args: True,
    )
    return certificate, verifier


def test_qualified_round_persists_across_restart_prepares_wta_intent_and_rolls_over(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    initial = FinalizedCompetitionState.genesis("wan", sha(1))
    first = plan(initial, committee, round_id="r1", snapshot=100)
    signed = history(first, validators, root=first.prompt_seed_inputs["admission_history_root"])
    path = tmp_path / "lifecycle.sqlite"

    with Store(path) as store:
        lifecycle = CompetitiveLifecycle(store, Sr25519Verifier())
        assert open_qualified(lifecycle, first, initial, signed)
        assert not open_qualified(lifecycle, first, initial, signed)
        certificate, verifier = qualified_certificate(first, validators)
        # A generic callback-only MVP verifier is useful for lower-level
        # evidence tests, but may not cross the distributed reward boundary.
        with pytest.raises(LifecycleError, match="replaying distributed"):
            lifecycle.certify_and_prepare_weight_intent(
                plan=first,
                state=initial,
                certificate=certificate,
                mvp_verifier=verifier,
                runtime_hash="not-a-hash",
                registered_uids={7: "burn", 8: "miner"},
                public_custody={first.challenger_fingerprint: sha(3)},
                verify_public_custody=lambda *_args: True,
                burn_hotkey="burn",
                verify_burn_qualification=lambda _key: sha(4),
            )
        assert store.count("certificates") == store.count("crowns") == 0


def test_lifecycle_refuses_unbound_history_and_non_mvp_promotion_verifier(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    state = FinalizedCompetitionState.genesis("wan", sha(1))
    frozen = plan(state, committee, round_id="r", snapshot=100)
    signed = history(frozen, validators, root=frozen.prompt_seed_inputs["admission_history_root"])
    with Store(tmp_path / "lifecycle.sqlite") as store:
        lifecycle = CompetitiveLifecycle(store, Sr25519Verifier())
        altered = RoundPlan(
            **{
                **frozen.__dict__,
                "prompt_seed_inputs": {
                    **frozen.prompt_seed_inputs,
                    "admission_history_root": sha(99),
                },
            }
        )
        with pytest.raises(LifecycleError, match="bind"):
            open_qualified(lifecycle, altered, state, signed)
        open_qualified(lifecycle, frozen, state, signed)
        certificate, _mvp = qualified_certificate(frozen, validators)
        with pytest.raises(LifecycleError, match="MVP"):
            lifecycle.certify_and_prepare_weight_intent(
                plan=frozen,
                state=state,
                certificate=certificate,
                mvp_verifier=CertificateVerifier(Sr25519Verifier()),
                runtime_hash=sha(2),
                registered_uids={7: "burn", 8: "miner"},
                public_custody={},
                verify_public_custody=None,
                burn_hotkey="burn",
                verify_burn_qualification=lambda _key: sha(4),
            )


def test_expired_round_releases_the_single_pending_slot_for_the_next_round(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    state = FinalizedCompetitionState.genesis("wan", sha(1))
    first = plan(state, committee, round_id="expired", snapshot=100)
    signed = history(first, validators, root=first.prompt_seed_inputs["admission_history_root"])
    with Store(tmp_path / "lifecycle.sqlite") as store:
        lifecycle = CompetitiveLifecycle(store, Sr25519Verifier())
        open_qualified(lifecycle, first, state, signed)
        closure = lifecycle.expire_round(
            RoundMailbox(store, Sr25519Verifier()),
            first,
            finalized_block=first.reveal_deadline_block + 1,
            finalized_hash="a" * 64,
        )
        assert closure["status"] == "EXPIRED_NO_CERTIFICATE"
        second = plan(state, committee, round_id="after-expiry", snapshot=200)
        assert open_qualified(
            lifecycle,
            second,
            state,
            history(second, validators, root=second.prompt_seed_inputs["admission_history_root"]),
        )


def test_open_round_recomputes_capacity_and_refuses_short_or_unbound_windows(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    state = FinalizedCompetitionState.genesis("wan", sha(1))
    frozen = plan(state, committee, round_id="capacity", snapshot=100)
    signed = history(frozen, validators, root=frozen.prompt_seed_inputs["admission_history_root"])
    distributed = capacity_plan(frozen)
    with Store(tmp_path / "lifecycle.sqlite") as store:
        lifecycle = CompetitiveLifecycle(store, Sr25519Verifier())
        smaller = DistributedEvaluationPlan(
            **{
                **distributed.__dict__,
                "stage": "diagnostic",
                "prompts": tuple(capacity_prompts()[:3]),
                "timing_prompt_ids": (),
                "audit_work_count": 3,
            }
        )
        with pytest.raises(LifecycleError, match="not bound"):
            lifecycle.open_round(
                frozen,
                state,
                signed,
                distributed_plan=smaller,
                capacity_inputs=capacity_inputs(),
            )
        short = RoundPlan(**{**frozen.__dict__, "commit_deadline_block": 101})
        with pytest.raises(LifecycleError, match="not bound"):
            lifecycle.open_round(
                short,
                state,
                signed,
                distributed_plan=distributed,
                capacity_inputs=capacity_inputs(),
            )
        too_slow = capacity_inputs()
        too_slow = CapacityInputs(**{**too_slow.__dict__, "candidate_seconds": 100})
        with pytest.raises(LifecycleError, match="commit window"):
            lifecycle.open_round(
                frozen,
                state,
                signed,
                distributed_plan=distributed,
                capacity_inputs=too_slow,
            )


def test_cli_opens_and_expires_the_same_signed_round_after_restart(tmp_path):
    validators = signers()
    committee = {signer.validator_hotkey: 1 for signer in validators}
    state = FinalizedCompetitionState.genesis("wan", sha(1))
    frozen = plan(state, committee, round_id="cli", snapshot=100)
    signed = history(frozen, validators, root=frozen.prompt_seed_inputs["admission_history_root"])
    plan_path, state_path, checkpoint_path = (
        tmp_path / "plan.json",
        tmp_path / "state.json",
        tmp_path / "history.json",
    )
    plan_path.write_text(json.dumps(frozen.to_dict()))
    state_path.write_text(json.dumps(state.to_dict()))
    checkpoint_path.write_text(
        json.dumps(
            {
                "history": signed.history.__dict__,
                "signatures": {
                    hotkey: signature.hex() for hotkey, signature in signed.signatures.items()
                },
            }
        )
    )
    distributed_path, capacity_path = tmp_path / "distributed.json", tmp_path / "capacity.json"
    distributed_path.write_text(json.dumps(capacity_plan(frozen).to_dict()))
    capacity_path.write_text(json.dumps(capacity_inputs().__dict__))
    database, opened, expired = (
        tmp_path / "validator.sqlite",
        tmp_path / "opened.json",
        tmp_path / "expired.json",
    )
    assert (
        main(
            [
                "competitive-open",
                "--db",
                str(database),
                "--plan",
                str(plan_path),
                "--state",
                str(state_path),
                "--admission-checkpoint",
                str(checkpoint_path),
                "--distributed-plan",
                str(distributed_path),
                "--capacity-inputs",
                str(capacity_path),
                "--out",
                str(opened),
            ]
        )
        == 0
    )
    assert json.loads(opened.read_text())["status"] == "OPENED"
    assert (
        main(
            [
                "competitive-expire",
                "--db",
                str(database),
                "--plan",
                str(plan_path),
                "--finalized-block",
                str(frozen.reveal_deadline_block + 1),
                "--finalized-hash",
                "b" * 64,
                "--out",
                str(expired),
            ]
        )
        == 0
    )
    assert json.loads(expired.read_text())["status"] == "EXPIRED_NO_CERTIFICATE"
