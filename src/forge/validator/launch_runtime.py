"""One durable tick for the bounded testnet Launch V2 operator.

The runtime only starts an explicitly supplied worker process after an immutable
request and budget reservation exist.  Shadow mode never starts a process,
signs, or writes to chain.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundCertificate, RoundPlan
from forge.protocol.contract import ChallengeContract
from forge.protocol.store import Store
from forge.validator.admission import build_authenticated_history
from forge.validator.launch_bridge import (
    LaunchExecutionReplayer,
    LaunchV1Bridge,
    LaunchV1Capacity,
    LocalLaunchReceiptStore,
)
from forge.validator.launch_config import LaunchConfig
from forge.validator.launch_consensus import MINIMUM_COMMIT_WINDOW_BLOCKS
from forge.validator.launch_round import freeze_closed_round, select_closed_round
from forge.validator.launch_service import LaunchServiceStore, atomic_json, write_immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _immutable(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(path, value)


def _active(service: LaunchServiceStore) -> dict[str, Any] | None:
    rounds = service.snapshot()["rounds"]
    return next(
        (
            row
            for row in reversed(rounds)
            if row["status"] not in {"COMPLETE", "EXPIRED", "INCOMPLETE"}
        ),
        None,
    )


def _worker_window_fits(
    plan: RoundPlan, capacity: Mapping[str, Any], finalized_height: int
) -> bool:
    """Leave the same pre-commit margin enforced by consensus.

    ``worst_case_execution_seconds`` covers the entire remote route: render,
    score, receipt import and replay.  Consensus still needs a strictly earlier
    block than its own no-new-commit cutoff to sign and submit the report.
    """
    execution_blocks = math.ceil(
        capacity["worst_case_execution_seconds"] / capacity["chain_block_seconds"]
    )
    return (
        finalized_height + execution_blocks
        < plan.commit_deadline_block - MINIMUM_COMMIT_WINDOW_BLOCKS
    )


def _close(height: int, config: LaunchConfig) -> int:
    return height - height % config.admission_window_blocks


def _window_history(
    chain: Any,
    config: LaunchConfig,
    parent: FinalizedCompetitionState,
    close: int,
    service: LaunchServiceStore,
    *,
    exclude_round_id: str | None = None,
):
    # The closed registry is authoritative; service lineage is retained only as
    # consumed operational history, never as a replacement for chain state.
    prior = tuple(
        sorted(
            {
                r["selection"]["candidate_hotkey"]
                for r in service.snapshot()["rounds"]
                if r["round_id"] != exclude_round_id
                and r["selection"]["close_block"] == close
            }
        )
    )
    registered = tuple(sorted(set(chain.uid_snapshot(close).values())))
    return build_authenticated_history(
        network=config.network,
        netuid=config.netuid,
        through_block=close,
        certified_state_hash=parent.state_hash(),
        consumed_hotkeys=prior,
        registered_hotkeys=registered,
        window_id=f"launch-{close}",
        window_open_block=max(0, close - config.admission_window_blocks),
        window_close_block=close,
    )


def _copy_sqlite(source: Path, target: Path, *, initialize: bool = False) -> None:
    """Take one consistent SQLite backup; never reuse a live WAL directory."""
    if initialize and target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as origin, sqlite3.connect(target) as destination:
        origin.backup(destination)


def _open_bridges(
    round_dir: Path,
    plan: RoundPlan,
    parent: FinalizedCompetitionState,
    policy: dict[str, Any],
    history: Any,
    wallets: Mapping[str, Any],
) -> None:
    """Open each local validator store using only locally held testnet wallets."""
    signers = [Sr25519Signer(wallet) for wallet in wallets.values()]
    if {signer.validator_hotkey for signer in signers} != set(plan.committee):
        raise ValueError("local wallets must cover the frozen committee exactly")
    receipts_root = round_dir / "trusted-receipts"
    capacity_path = round_dir / "capacity.json"
    encoded = capacity_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    target = receipts_root / "capacity" / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != encoded:
        raise ValueError("trusted capacity receipt hash collision")
    if not target.exists():
        target.write_bytes(encoded)
    signed_history = create_signed_history(history, signers)
    for label in wallets:
        round_store = round_dir / f"validator-{label}.sqlite"
        canonical_store = round_dir.parent / f"canonical-validator-{label}.sqlite"
        if canonical_store.exists():
            _copy_sqlite(canonical_store, round_store, initialize=True)
        with Store(round_store) as store:
            existing = store.latest_state_payload(parent.challenge_id)
            if (
                existing is not None
                and FinalizedCompetitionState.from_dict(existing).state_hash()
                != parent.state_hash()
            ):
                raise ValueError("canonical validator history does not end at the certified parent")
            if existing is None:
                store.initialize_state(
                    state_hash=parent.state_hash(),
                    challenge_id=parent.challenge_id,
                    state_version=parent.state_version,
                    previous_state_hash=parent.previous_state_hash,
                    payload=parent.to_dict(),
                )
            LaunchV1Bridge(
                store, Sr25519Verifier(), LocalLaunchReceiptStore(receipts_root)
            ).open_round(plan, parent, signed_history, policy, LaunchV1Capacity(digest))


def tick(
    root: Path,
    *,
    config: LaunchConfig,
    contract: ChallengeContract,
    chain: Any,
    wallets: Mapping[str, Any] | None = None,
    worker_command: Sequence[str] | None = None,
    reserve_usd: float = 0.0,
    shadow: bool = True,
    publication_step: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    export_destination: Path | None = None,
) -> dict[str, Any]:
    """Advance at most one durable stage; no implicit worker or signer actions."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    service = LaunchServiceStore(root / "budget.sqlite", config)
    parent = FinalizedCompetitionState.from_dict(_read(root / "state.json"))
    height = chain.finalized_height()
    current = _active(service)
    if current is None:
        close = _close(height, config)
        if (
            not close <= height < close + config.entropy_offset_blocks
            or any(row["selection"]["close_block"] == close for row in service.snapshot()["rounds"])
        ):
            return _export(
                root, config, service,
                {"status": "WAITING_CLOSE_WINDOW", "finalized_block": height},
                export_destination,
            )
        round_dir = root / f"round-{close}"
        try:
            selection = select_closed_round(
                chain=chain,
                contract=contract,
                config=config,
                parent=parent,
                service=service,
                admission_history=_window_history(chain, config, parent, close, service),
                close_block=close,
                finalized_block=height,
                round_dir=round_dir,
            )
        except ValueError as error:
            if str(error) == "no unselected admitted candidate committed in the closed window":
                return _export(
                    root,
                    config,
                    service,
                    {"status": "WAITING_CANDIDATE", "finalized_block": height},
                    export_destination,
                )
            raise
        result = {"status": "SELECTED_WAITING_ENTROPY", "round_id": selection["round_id"]}
    else:
        selection, result = (
            current["selection"],
            {"status": current["status"], "round_id": current["round_id"]},
        )
        round_dir = root / f"round-{selection['close_block']}"
    if height < selection["entropy_block"]:
        return _export(root, config, service, result, export_destination)
    if current is None or current["status"] == "WAITING_ENTROPY":
        history = _window_history(
            chain,
            config,
            parent,
            selection["close_block"],
            service,
            exclude_round_id=selection["round_id"],
        )
        manifests = {
            name: _read(root / "manifests" / f"{name}.json")["fingerprint"]
            for name in ("king", "anchor")
        }
        manifests["king"] = parent.current_king_fingerprint
        manifests["candidate"] = selection["candidate_manifest_hash"]
        policy, plan = freeze_closed_round(
            config=config,
            parent=parent,
            service=service,
            selection=selection,
            entropy_hash=chain.finalized_block_hash(selection["entropy_block"]),
            finalized_block=height,
            basic_manifests=manifests,
            worker_ready=_read(root / "worker-ready.json"),
            qualification_package=_read(root / "qualification-package.json"),
            admission_history=history,
            committee=_read(round_dir / "committee.json")["committee"],
            round_dir=round_dir,
            bootstrap_receipt_path=root / "bootstrap-reference.json",
        )
        result = {
            "status": "READY",
            "round_id": plan.round_id,
            "plan_hash": plan.plan_hash(),
            "launch_plan_id": policy["plan_id"],
        }
    current = _active(service)
    if current and current["status"] == "REQUESTED":
        deadline = RoundPlan.from_dict(_read(round_dir / "round-plan.json")).reveal_deadline_block
        if height > deadline:
            service.expire(
                current["round_id"],
                finalized_block=height,
                reason="worker evidence missed frozen reveal deadline",
            )
            return _export(
                root,
                config,
                service,
                {"status": "EXPIRED", "round_id": current["round_id"]},
                export_destination,
            )
        result_path = round_dir / "worker-result.json"
        if result_path.exists():
            if service.effective_admission(current["round_id"]) is None:
                raise ValueError("worker result has no accepted global effective admission")
            raw = _read(result_path)
            if (
                set(raw) != {"request_hash", "trusted_bundle_hashes"}
                or raw["request_hash"] != current["request_hash"]
            ):
                raise ValueError("worker result is not bound to the durable request")
            bundles = raw["trusted_bundle_hashes"]
            plan = RoundPlan.from_dict(_read(round_dir / "round-plan.json"))
            if not isinstance(bundles, Mapping) or set(bundles) != set(plan.committee):
                raise ValueError(
                    "worker result must name each frozen validator bundle exactly once"
                )
            replay = LaunchExecutionReplayer(
                LocalLaunchReceiptStore(round_dir / "trusted-receipts")
            )
            rows = [
                replay.replay(digest, plan, validator_hotkey=hotkey)
                for hotkey, digest in bundles.items()
            ]
            if wallets:
                labels = {
                    Sr25519Signer(wallet).validator_hotkey: label
                    for label, wallet in wallets.items()
                }
                if set(labels) != set(plan.committee):
                    raise ValueError("local wallets differ from frozen committee")
                for hotkey, digest in bundles.items():
                    _immutable(
                        round_dir / f"bundle-{labels[hotkey]}.json",
                        {"bundle_hash": digest, "validator_hotkey": hotkey},
                    )
            if any(
                row.get("status") not in {"PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE"}
                for row in rows
            ):
                raise ValueError("trusted bundle replay returned an invalid result")
            statuses = {row["status"] for row in rows}
            outcome = statuses.pop() if len(statuses) == 1 else "INCONCLUSIVE"
            service.record_evaluation(
                current["round_id"],
                request_hash=current["request_hash"],
                result={"status": outcome, "replays": rows},
            )
            result = {
                "status": "INCOMPLETE" if outcome == "INCOMPLETE" else "EVALUATION_REPLAYED",
                "round_id": current["round_id"],
                "outcome": outcome,
            }
        elif (round_dir / "dispatch-intent.json").exists():
            result = {
                "status": "WORKER_DISPATCH_UNKNOWN_NO_RELAUNCH",
                "round_id": current["round_id"],
            }
    current = _active(service)
    if current and current["status"] == "AWAITING_CONSENSUS" and not shadow:
        if not wallets:
            raise ValueError("live consensus requires explicit local validator wallets")
        from forge.validator.launch_consensus import LaunchConsensusRunner

        consensus = LaunchConsensusRunner(round_dir, chain=chain, wallets=dict(wallets)).tick()
        result = {"status": consensus["status"], "round_id": current["round_id"]}
        if consensus["status"] == "EXPIRED_NO_CERTIFICATE":
            service.expire(
                current["round_id"], finalized_block=height,
                reason="frozen round expired without a quorum certificate",
            )
            result["status"] = "EXPIRED"
        if consensus["status"] == "CERTIFIED_NOT_YET_WEIGHTS":
            certificate = _read(round_dir / "certificate.json")
            certified = _read(round_dir / f"certified-state-{next(iter(wallets))}.json")
            certificate_hash = RoundCertificate.from_dict(certificate).transition_hash()
            certified_hash = FinalizedCompetitionState.from_dict(certified).state_hash()
            for label in wallets:
                canonical = root / f"canonical-validator-{label}.sqlite"
                if canonical.exists():
                    with Store(canonical) as store:
                        latest = store.latest_state_payload(parent.challenge_id)
                    if latest is None or FinalizedCompetitionState.from_dict(
                        latest
                    ).state_hash() not in {
                        certificate["parent_state_hash"], certified_hash
                    }:
                        raise ValueError("canonical validator history diverged from this round")
                _copy_sqlite(
                    round_dir / f"validator-{label}.sqlite",
                    canonical,
                )
            atomic_json(root / "state.json", certified)
            atomic_json(
                root / "manifests" / "king.json",
                {"fingerprint": certified["current_king_fingerprint"]},
            )
            # Advance the service last. A crash during any SQLite/file write
            # replays this same certificate before publication can begin.
            service.record_certification(
                current["round_id"],
                certificate_hash=certificate_hash,
                certified_state_hash=certified_hash,
                plan_hash=current["plan_hash"],
            )
            result = {"status": "CERTIFIED_AWAITING_PUBLICATION", "round_id": current["round_id"]}
    current = _active(service)
    if current and current["status"] == "CERTIFIED":
        if publication_step is None:
            result = {"status": "CERTIFIED_AWAITING_PUBLICATION", "round_id": current["round_id"]}
        elif shadow:
            result = {"status": "SHADOW_CERTIFIED_NO_PUBLICATION", "round_id": current["round_id"]}
        else:
            certificate, certified = (
                _read(round_dir / "certificate.json"),
                _read(root / "state.json"),
            )
            receipt = publication_step(
                {
                    "round_dir": str(round_dir.resolve()),
                    "certificate": certificate,
                    "state": certified,
                }
            )
            if receipt is None:
                result = {
                    "status": "PUBLICATION_RECONCILIATION_PENDING",
                    "round_id": current["round_id"],
                }
            else:
                service.record_publication(current["round_id"], receipt=receipt)
                result = {
                    "status": "COMPLETE" if receipt["status"] == "FINALIZED" else receipt["status"],
                    "round_id": current["round_id"],
                }
    current = _active(service)
    if current and current["status"] == "READY" and not shadow:
        if not wallets:
            raise ValueError("live launch requires explicit local validator wallets")
        plan = RoundPlan.from_dict(_read(round_dir / "round-plan.json"))
        capacity = _read(round_dir / "capacity.json")
        fresh_height = chain.finalized_height()
        if not _worker_window_fits(plan, capacity, fresh_height):
            service.expire(
                current["round_id"], finalized_block=fresh_height,
                reason="frozen window lacks the required pre-commit worker margin",
            )
            return _export(
                root, config, service,
                {"status": "EXPIRED", "round_id": current["round_id"]},
                export_destination,
            )
        policy = _read(round_dir / "launch-plan.json")
        history = _window_history(
            chain,
            config,
            parent,
            selection["close_block"],
            service,
            exclude_round_id=selection["round_id"],
        )
        _open_bridges(round_dir, plan, parent, policy, history, wallets)
    if current and current["status"] == "READY" and shadow:
        return _export(
            root,
            config,
            service,
            {"status": "SHADOW_READY_NO_SIGN_NO_GPU", "round_id": current["round_id"]},
            export_destination,
        )
    if current and current["status"] == "READY":
        request = {
            "schema": "forge/launch-worker-request/v2",
            "round_id": current["round_id"],
            "plan_hash": current["plan_hash"],
            "round_dir": str(round_dir),
            "result_path": str(round_dir / "worker-result.json"),
        }
        service.request_work(
            current["round_id"], reserve_usd=reserve_usd, request=request
        )
    current = _active(service)
    if current and current["status"] == "REQUESTED" and not shadow:
        prepared = service.work_request(current["round_id"])
        _immutable(round_dir / "worker-request.json", prepared)
        if not worker_command:
            result = {
                "status": "REQUESTED_AWAITING_OPERATOR_WORKER",
                "round_id": current["round_id"],
            }
        else:
            intent = round_dir / "dispatch-intent.json"
            if not intent.exists():
                capacity = _read(round_dir / "capacity.json")
                plan = RoundPlan.from_dict(_read(round_dir / "round-plan.json"))
                fresh_height = chain.finalized_height()
                if not _worker_window_fits(plan, capacity, fresh_height):
                    service.expire(
                        current["round_id"], finalized_block=fresh_height,
                        reason="reserved request lacks the required pre-commit worker margin",
                    )
                    return _export(
                        root, config, service,
                        {"status": "EXPIRED", "round_id": current["round_id"]},
                        export_destination,
                    )
                _immutable(
                    intent,
                    {
                        "schema": "forge/launch-dispatch-intent/v1",
                        "request_hash": prepared["request_hash"],
                        "status": "UNKNOWN_AFTER_START",
                    },
                )
                subprocess.Popen(
                    [
                        *worker_command,
                        "--request",
                        str(round_dir / "worker-request.json"),
                        "--result",
                        str(round_dir / "worker-result.json"),
                    ]
                )
                result = {"status": "WORKER_DISPATCHED_UNKNOWN", "round_id": current["round_id"]}
            else:
                result = {
                    "status": "WORKER_DISPATCH_UNKNOWN_NO_RELAUNCH",
                    "round_id": current["round_id"],
                }
    return _export(root, config, service, result, export_destination)


def _export(
    root: Path,
    config: LaunchConfig,
    service: LaunchServiceStore,
    result: dict[str, Any],
    destination: Path | None,
) -> dict[str, Any]:
    atomic_json(root / "runtime-status.json", result)
    if destination is not None:
        from forge.validator.launch_service_dashboard import project_launch_operations

        display = root / "bootstrap-display.json"
        if display.exists():
            bootstrap = _read(display)
        elif (root / "bootstrap-reference.json").exists():
            bootstrap = {
                "fingerprint": _read(root / "bootstrap-reference.json")["bootstrap_reference_hash"]
            }
        else:
            bootstrap = _read(root / "manifests" / "anchor.json")
        snapshot = service.snapshot()
        latest = snapshot["rounds"][-1] if snapshot["rounds"] else None
        consensus_path = (
            root / f"round-{latest['selection']['close_block']}" / "consensus-status.json"
            if latest else None
        )
        consensus = _read(consensus_path) if consensus_path and consensus_path.exists() else None
        atomic_json(
            destination,
            project_launch_operations(
                snapshot,
                config,
                bootstrap_reference=bootstrap,
                consensus_status=consensus,
            ),
        )
    return result
