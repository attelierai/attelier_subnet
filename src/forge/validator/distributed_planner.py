"""Reconstruct a diagnostic round independently from finalized testnet state."""

from __future__ import annotations

from forge.evaluation.distributed import DistributedEvaluationPlan
from forge.evaluation.synthetic_prompts import GENERATOR_ID, synthetic_panel
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.hashing import sha256_file


def diagnostic_plan(
    chain,
    *,
    close_block: int,
    entropy_block: int,
    candidate: ModelManifest,
    candidate_hotkey: str,
    wan50: ModelManifest,
) -> DistributedEvaluationPlan:
    """Six real prompts, two audited overlaps, no reward eligibility.

    This small integration panel deliberately does not impersonate the full
    16/128/32 challenge. Identical target controls are allowed only here.
    """
    from forge.chain import decode_commitment
    from forge.evaluation import synthetic_prompts
    from forge.validator.submission import V3SubmissionClaim, encode_v3_submission_commitment

    if chain.network != "test" or not 0 <= close_block < entropy_block:
        raise ValueError("diagnostic plan needs testnet and post-close entropy")
    if chain.finalized_height() < entropy_block:
        raise ValueError("entropy block is not finalized")
    committee = chain.committee_snapshot(close_block)
    if len(committee) < 2:
        raise ValueError("diagnostic pilot requires two real permitted validators")
    claim = V3SubmissionClaim(
        candidate.reference, candidate.role, dict(candidate.generation_config)
    )
    committed = chain.commitment(candidate_hotkey, close_block)
    expected = decode_commitment(
        candidate_hotkey,
        committed.block if committed else 0,
        encode_v3_submission_commitment(claim),
    )
    if committed is None or committed != expected:
        raise ValueError("candidate was not committed before the frozen close")
    entropy = chain.finalized_block_hash(entropy_block)
    round_id = f"test-{chain.netuid}-b{close_block}-e{entropy_block}"
    panel = synthetic_panel(entropy, round_id=round_id, phase="diagnostic", per_cell=1)
    prompts = tuple(panel[index] for index in (0, 5, 8, 11, 12, 15))
    return DistributedEvaluationPlan(
        round_id=round_id,
        stage="diagnostic",
        entropy=entropy,
        policy_id="diagnostic-no-calibration-no-rewards-v1",
        candidate_manifest_hash=candidate.fingerprint(),
        king_manifest_hash=wan50.fingerprint(),
        wan50_manifest_hash=wan50.fingerprint(),
        committee=committee,
        prompts=prompts,
        timing_prompt_ids=(),
        quality_tolerance=None,
        timing_log_ratio_tolerance=None,
        audit_work_count=2,
        chain_context={
            "network": chain.network,
            "netuid": chain.netuid,
            "close_block": close_block,
            "close_block_hash": chain.finalized_block_hash(close_block),
            "entropy_block": entropy_block,
            "entropy_block_hash": entropy,
            "generator_id": GENERATOR_ID,
            "generator_sha256": sha256_file(synthetic_prompts.__file__),
            "candidate_hotkey": candidate_hotkey,
            "candidate_commit_block": committed.block,
        },
    )
