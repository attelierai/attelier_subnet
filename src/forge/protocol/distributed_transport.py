"""Content-addressed peer reports anchored to a common finalized chain snapshot.

Peers may each serve static files. Neither a peer URL nor a local file is a
trust root: the finalized commitment, assigned work and hotkey signature are.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.request import urlopen

from forge.protocol.hashing import canonical_json_bytes, hash_canonical, sha256_bytes

MAX_REPORT_BYTES = 4 * 1024 * 1024


def report_digest(report: Mapping) -> str:
    return sha256_bytes(canonical_json_bytes(report))


def verify_plan_chain_context(chain, plan, commit_close_block: int) -> None:
    """Authenticate a supplied plan against the chain adapter, not its own JSON."""
    context = plan.chain_context
    if (
        context.get("network") != chain.network
        or context.get("netuid") != chain.netuid
        or type(context.get("close_block")) is not int
        or type(context.get("entropy_block")) is not int
        or not context["close_block"] < context["entropy_block"] < commit_close_block
    ):
        raise ValueError("plan has no matching finalized round context")
    if (
        chain.finalized_block_hash(context["close_block"]) != context.get("close_block_hash")
        or chain.finalized_block_hash(context["entropy_block"]) != plan.entropy
        or context.get("entropy_block_hash") != plan.entropy
        or chain.committee_snapshot(context["close_block"]) != dict(plan.committee)
    ):
        raise ValueError("plan committee or entropy differs from finalized chain")


def _fetch_json(url: str) -> dict:
    if not url.startswith(("https://", "http://")):
        raise ValueError("peer report must use HTTP(S)")
    with urlopen(url, timeout=30) as response:
        payload = response.read(MAX_REPORT_BYTES + 1)
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError("peer report exceeds size limit")
    report = json.loads(payload)
    if not isinstance(report, dict):
        raise ValueError("peer evidence must be a JSON object")
    return report


def fetch_peer_report(url: str, expected_digest: str) -> dict:
    """Fetch bounded public JSON; hashes and signatures make the host untrusted."""
    report = _fetch_json(url)
    if report_digest(report) != expected_digest:
        raise ValueError("peer report does not match its content address")
    return report


def replay_diagnostic_report(plan, report: Mapping, base_url: str) -> set[str]:
    """Recompute deltas from hash-bound raw evidence, without another API call.

    This proves numerical replay, not honest execution or a provider-signed API
    response. Independent overlapping GPU work is the separate audit mechanism.
    """
    from forge.evaluation.diagnostic_fusion import fuse_diagnostic_quality

    if plan.stage != "diagnostic" or plan.king_manifest_hash != plan.wan50_manifest_hash:
        raise ValueError("diagnostic replay requires the exact Wan-50 bootstrap")
    if report["quality_vs_king"] != report["quality_vs_wan50"] or report["timing"]:
        raise ValueError("diagnostic bootstrap aliases must contain identical evidence")
    response_ids = set()
    for row in report["quality_vs_king"]:
        digest = row.get("raw_evidence_hash")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("diagnostic row has no valid evidence address")
        raw = _fetch_json(f"{base_url.rstrip('/')}/{digest}.json")
        if hash_canonical("forge/diagnostic-quality/v1", raw) != digest:
            raise ValueError("raw diagnostic evidence differs from its commitment")
        if set(raw) != {"pair", "reference_metrics", "candidate_metrics"}:
            raise ValueError("raw diagnostic evidence fields are not exact")
        if any(
            raw["pair"][key]["validator_hotkey"] != report["validator_hotkey"]
            for key in ("reference_receipt", "candidate_receipt")
        ):
            raise ValueError("render receipts belong to another validator")
        recomputed = fuse_diagnostic_quality(plan, prompt_id=row["prompt_id"], **raw)
        if canonical_json_bytes(recomputed) != canonical_json_bytes(row):
            raise ValueError("reported deltas differ from raw evidence replay")
        for call in raw["pair"]["judgment"]["calls"]:
            if call["response_id"] in response_ids:
                raise ValueError("provider response reused across distinct prompt work")
            response_ids.add(call["response_id"])
    return response_ids
