"""One hash per validator binds every challenger before a common reveal window."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from forge.evaluation.distributed import aggregate_partial_reports, verify_partial_report
from forge.evaluation.distributed_plan import DistributedEvaluationPlan, allocate_work
from forge.protocol.distributed_transport import (
    fetch_peer_report,
    replay_diagnostic_report,
    report_digest,
    verify_plan_chain_context,
)
from forge.protocol.hashing import canonical_json_bytes, hash_canonical


@dataclass(frozen=True)
class DiagnosticBatch:
    plans: tuple[DistributedEvaluationPlan, ...]
    commit_deadline_block: int
    reveal_deadline_block: int

    def __post_init__(self):
        if not self.plans or any(plan.stage != "diagnostic" for plan in self.plans):
            raise ValueError("this delivery route only accepts diagnostic plans")
        first = self.plans[0]
        if len({plan.candidate_manifest_hash for plan in self.plans}) != len(self.plans):
            raise ValueError("batch candidates must be unique")
        for plan in self.plans:
            if (
                plan.allocation_seed() != first.allocation_seed()
                or plan.policy_id != first.policy_id
                or allocate_work(plan) != allocate_work(first)
                or plan.king_manifest_hash != first.king_manifest_hash
                or plan.wan50_manifest_hash != first.wan50_manifest_hash
                or any(
                    plan.chain_context.get(key) != first.chain_context.get(key)
                    for key in (
                        "network",
                        "netuid",
                        "close_block",
                        "close_block_hash",
                        "entropy_block",
                        "entropy_block_hash",
                    )
                )
            ):
                raise ValueError("batch plans disagree on round, committee, panel or chain")
        entropy = first.chain_context.get("entropy_block")
        if (
            type(entropy) is not int
            or type(self.commit_deadline_block) is not int
            or type(self.reveal_deadline_block) is not int
            or not entropy < self.commit_deadline_block < self.reveal_deadline_block
        ):
            raise ValueError("batch delivery blocks must be strictly increasing")
        object.__setattr__(
            self, "plans", tuple(sorted(self.plans, key=lambda plan: plan.plan_hash()))
        )

    def to_dict(self):
        return {
            "schema": "forge/diagnostic-batch-plan/v1",
            "plans": [plan.to_dict() for plan in self.plans],
            "commit_deadline_block": self.commit_deadline_block,
            "reveal_deadline_block": self.reveal_deadline_block,
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"schema", "plans", "commit_deadline_block", "reveal_deadline_block"} or (
            value["schema"] != "forge/diagnostic-batch-plan/v1"
        ):
            raise ValueError("batch plan fields are not exact")
        return cls(
            tuple(DistributedEvaluationPlan.from_dict(plan) for plan in value["plans"]),
            value["commit_deadline_block"],
            value["reveal_deadline_block"],
        )

    def batch_hash(self):
        return hash_canonical("forge/diagnostic-batch-plan/v1", self.to_dict())

    def round_token(self):
        return "b" + self.batch_hash()[:16]


def sign_batch_index(batch, reports, signer, verifier):
    by_plan = {}
    for report in reports:
        key = report.get("plan_hash")
        if key in by_plan:
            raise ValueError("duplicate candidate report")
        by_plan[key] = report
    if set(by_plan) != {plan.plan_hash() for plan in batch.plans}:
        raise ValueError("every frozen challenger needs a partial report, even if incomplete")
    for plan in batch.plans:
        report = by_plan[plan.plan_hash()]
        verify_partial_report(plan, report, verifier)
        if report["validator_hotkey"] != signer.validator_hotkey:
            raise ValueError("batch contains another validator's report")
    body = {
        "schema": "forge/diagnostic-batch-index/v1",
        "batch_hash": batch.batch_hash(),
        "validator_hotkey": signer.validator_hotkey,
        "reports": {key: report_digest(value) for key, value in sorted(by_plan.items())},
    }
    return {**body, "signature": signer.sign(canonical_json_bytes(body)).hex()}


def verify_batch_index(batch, index, hotkey, verifier):
    if set(index) != {"schema", "batch_hash", "validator_hotkey", "reports", "signature"}:
        raise ValueError("batch index fields are not exact")
    if (
        index["schema"] != "forge/diagnostic-batch-index/v1"
        or index["batch_hash"] != batch.batch_hash()
        or index["validator_hotkey"] != hotkey
        or hotkey not in batch.plans[0].committee
        or not isinstance(index["reports"], Mapping)
        or set(index["reports"]) != {plan.plan_hash() for plan in batch.plans}
    ):
        raise ValueError("batch index differs from frozen committee/candidates")
    for digest in index["reports"].values():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("invalid report address")
    body = {key: value for key, value in index.items() if key != "signature"}
    if not verifier.verify(hotkey, canonical_json_bytes(body), bytes.fromhex(index["signature"])):
        raise ValueError("invalid batch index signature")


def resolve_batch(chain, batch, peers, verifier):
    """Replay a frozen commitment cut; missing peers expire without a promotion."""
    height = chain.finalized_height()
    if height <= batch.commit_deadline_block:
        raise ValueError("report reveal must wait until after the frozen commitment deadline")
    first = batch.plans[0]
    verify_plan_chain_context(chain, first, batch.commit_deadline_block)
    reports = {plan.plan_hash(): [] for plan in batch.plans}
    accepted, failures, response_ids = [], [], set()
    anchors = {
        item.hotkey: item
        for item in chain.commitments(list(first.committee), batch.commit_deadline_block)
    }
    for hotkey in sorted(first.committee):
        anchor, url = anchors.get(hotkey), peers.get(hotkey)
        if anchor is None or url is None:
            failures.append({"hotkey": hotkey, "reason": "MISSING_ANCHOR_OR_PEER"})
            continue
        try:
            if (
                anchor.kind != "rep"
                or anchor.fields[0] != batch.round_token()
                or not first.chain_context["entropy_block"]
                < anchor.block
                <= batch.commit_deadline_block
            ):
                raise ValueError("anchor is outside the frozen batch window")
            index = fetch_peer_report(
                f"{url.rstrip('/')}/{anchor.fields[1]}.json", anchor.fields[1]
            )
            verify_batch_index(batch, index, hotkey, verifier)
            staged, staged_ids = {}, set()
            for plan in batch.plans:
                digest = index["reports"][plan.plan_hash()]
                report = fetch_peer_report(f"{url.rstrip('/')}/{digest}.json", digest)
                verify_partial_report(plan, report, verifier)
                if report["validator_hotkey"] != hotkey:
                    raise ValueError("peer report differs from the committing hotkey")
                identifiers = replay_diagnostic_report(plan, report, url)
                if identifiers & (response_ids | staged_ids):
                    raise ValueError("provider response reused across candidate/validator work")
                staged_ids.update(identifiers)
                staged[plan.plan_hash()] = report
            for key, report in staged.items():
                reports[key].append(report)
            response_ids.update(staged_ids)
            accepted.append(hotkey)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append({"hotkey": hotkey, "reason": str(exc), "error": type(exc).__name__})
    aggregates = {
        plan.plan_hash(): aggregate_partial_reports(plan, reports[plan.plan_hash()], verifier)
        for plan in batch.plans
    }
    status = "COMPLETE_DIAGNOSTIC"
    if any(value["status"] not in {"COMPLETE", "DIAGNOSTIC"} for value in aggregates.values()):
        status = "INCOMPLETE"
    if failures:
        status = "PENDING"
    # A slow peer cannot extend the reveal window while its evidence is fetched.
    height = chain.finalized_height()
    if height > batch.reveal_deadline_block:
        status = "EXPIRED"
    return {
        "schema": "forge/diagnostic-batch-result/v1",
        "batch_hash": batch.batch_hash(),
        "status": status,
        "commit_deadline_block": batch.commit_deadline_block,
        "commit_deadline_hash": chain.finalized_block_hash(batch.commit_deadline_block),
        "observed_finalized_block": height,
        "contributors": accepted,
        "failures": failures,
        "aggregates": aggregates,
        "chain_anchors_verified": len(accepted) == len(first.committee),
        "raw_evidence_replayed": len(accepted) == len(first.committee),
        "reward_eligible": False,
    }
