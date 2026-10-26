"""Import bounded receipts fetched by the operator from its own remote worker.

This is not a miner upload endpoint. The operator authenticates the remote
machine and retains the videos there; the index binds every imported JSON byte.
Replaying the full execution graph remains mandatory before a result is emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forge.evaluation import launch_policy
from forge.protocol.consensus_types import RoundPlan
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.validator.launch_bridge import LaunchExecutionReplayer, LocalLaunchReceiptStore
from forge.validator.launch_operator import _store_receipt, bundle
from forge.validator.launch_service import write_immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("receipt must contain a JSON object")
    return value


_RECEIPT_KINDS = ("panels", "timing", "scores", "diversity", "diversity_panels", "qualification")


def _campaign_entries(source: Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    """Read one bounded, hash-complete campaign export before local custody."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("operator receipt directory must be a real directory")
    index = _read(source / "receipt-index.json")
    if index.get("schema") != "forge/launch-campaign-receipts/v2":
        raise ValueError("receipt campaign schema is invalid")
    entries: dict[str, list[Path]] = {}
    seen: set[str] = set()
    for name in _RECEIPT_KINDS:
        values = index.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"receipt index has no {name}")
        entries[name] = []
        for relative in values:
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or any(part in {".", ".."} for part in Path(relative).parts)
                or relative in seen
            ):
                raise ValueError("receipt index contains an unsafe or duplicated path")
            path = source / relative
            if (
                any(
                    parent.is_symlink()
                    for parent in (path, *path.parents)
                    if parent != source.parent
                )
                or not path.is_file()
                or not path.resolve().is_relative_to(source.resolve())
                or path.stat().st_size > 2_000_000
                or sha256_file(path) != index.get("sha256", {}).get(relative)
            ):
                raise ValueError("operator receipt is absent, oversized or changed")
            _read(path)
            seen.add(relative)
            entries[name].append(path)
    if seen != set(index.get("sha256", {})) or len(entries["qualification"]) != 1:
        raise ValueError("receipt index hash coverage or qualification is invalid")
    return index, entries


def replay_baseline_campaign(
    source: Path,
    receipt_root: Path,
    *,
    plan: dict[str, Any],
    contract_hash: str,
    selection_id: str,
) -> dict[str, Any]:
    """Import and replay a selected baseline's V2 campaign without a RoundPlan.

    Baseline confirmation has its own immutable selection authority.  It must
    not manufacture consensus histories, validator hotkeys or a certificate to
    reuse the raw receipt graph.
    """
    source, receipt_root = Path(source), Path(receipt_root)
    if not isinstance(selection_id, str) or not selection_id:
        raise ValueError("baseline selection id is invalid")
    launch_policy._check_plan_v2(plan)
    if plan["models"]["king"] != plan["models"]["anchor"]:
        raise ValueError("baseline confirmation requires king and anchor to match")
    entropy = hash_canonical("forge/baseline-confirmation-entropy/v1", selection_id)
    launch_policy.validate_plan(
        plan, expected_entropy_hash=entropy, expected_round_id=selection_id[:32]
    )
    index, entries = _campaign_entries(source)
    source_plan, campaign = _read(source / "launch-plan.json"), _read(source / "campaign.json")
    campaign_body = {key: value for key, value in campaign.items() if key != "campaign_hash"}
    if (
        source_plan != plan
        or index.get("plan_id") != plan["plan_id"]
        or campaign.get("schema") != "forge/launch-campaign/v2"
        or campaign.get("campaign_hash")
        != hash_canonical("forge/launch-campaign/v2", campaign_body)
        or campaign.get("campaign_hash") != index.get("campaign_hash")
        or campaign.get("plan_id") != plan["plan_id"]
        or campaign.get("contract_hash") != contract_hash
    ):
        raise ValueError("baseline campaign does not bind the selected plan and contract")
    imported = {
        kind: [_store_receipt(receipt_root, kind if kind != "diversity_panels" else "panels", path)
               for path in paths]
        for kind, paths in entries.items()
    }
    replayer = LaunchExecutionReplayer(LocalLaunchReceiptStore(receipt_root))
    return replayer.replay_receipts(
        policy_plan=plan,
        contract_hash=contract_hash,
        panel_receipts=imported["panels"],
        timing_receipts=imported["timing"],
        score_receipts=imported["scores"],
        diversity_receipts=imported["diversity"],
        qualification_receipt=imported["qualification"][0],
    )


def import_campaign(
    round_dir: Path,
    source: Path,
    *,
    validators: dict[str, str],
    request_hash: str,
) -> dict[str, Any]:
    round_dir, source = Path(round_dir), Path(source)
    index, entries = _campaign_entries(source)
    plan = RoundPlan.from_dict(_read(round_dir / "round-plan.json"))
    policy = _read(round_dir / "launch-plan.json")
    request = _read(round_dir / "worker-request.json")
    if (
        index.get("schema") != "forge/launch-campaign-receipts/v2"
        or index.get("plan_id") != policy["plan_id"]
        or plan.prompt_seed_inputs.get("launch_plan_id") != policy["plan_id"]
        or request.get("request_hash") != request_hash
        or len(validators) != len(set(validators.values()))
        or set(validators.values()) != set(plan.committee)
    ):
        raise ValueError("receipt campaign, request or committee binding differs")
    import re

    if any(not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", label) for label in validators):
        raise ValueError("validator label is unsafe")
    receipt_root = round_dir / "trusted-receipts"
    replay = LaunchExecutionReplayer(LocalLaunchReceiptStore(receipt_root))
    bundles, results = {}, {}
    for label, hotkey in validators.items():
        result = bundle(
            argparse.Namespace(
                round_plan=round_dir / "round-plan.json",
                launch_plan=round_dir / "launch-plan.json",
                receipt_root=receipt_root,
                validator_hotkey=hotkey,
                panel=entries["panels"],
                timing=entries["timing"],
                score=entries["scores"],
                diversity=entries["diversity"],
                diversity_panel=entries["diversity_panels"],
                qualification=entries["qualification"][0],
                out=round_dir / f"bundle-{label}.json",
            )
        )
        digest = result["bundle_hash"]
        checked = replay.replay(digest, plan, validator_hotkey=hotkey)
        if checked["status"] == "INCOMPLETE":
            raise ValueError(f"imported execution is incomplete: {checked.get('reasons')}")
        bundles[hotkey] = digest
        results[hotkey] = checked["result_hash"]
    if len(set(results.values())) != 1:
        raise ValueError("validators reconstructed different policy decisions")
    result = {"request_hash": request_hash, "trusted_bundle_hashes": bundles}
    write_immutable_json(round_dir / "worker-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validators", type=Path, required=True)
    parser.add_argument("--request-hash", required=True)
    args = parser.parse_args()
    result = import_campaign(
        args.round_dir,
        args.source,
        validators=_read(args.validators),
        request_hash=args.request_hash,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
