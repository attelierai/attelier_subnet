"""Project a *verified* Launch V1 testnet round into a dashboard snapshot.

Diagnostic dashboard evidence remains diagnostic until this exporter has replayed
both validator stores, the certificate and receipt graph, and each finalized
weight-vector readback.  It performs no chain or filesystem publication.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Verifier
from forge.chain.reward_publication import (
    WeightIntent,
    WeightSubmissionReceipt,
    verify_finalized_replacement,
)
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundCertificate, RoundPlan
from forge.protocol.hashing import sha256_file
from forge.protocol.preconsensus import apply_certificate
from forge.protocol.store import Store
from forge.validator.launch_bridge import (
    LaunchCertificateVerifier,
    LaunchExecutionReplayer,
    LaunchV1Bridge,
    LocalLaunchReceiptStore,
)


class CertifiedDashboardError(ValueError):
    """A dashboard cannot claim a V1 transition without all durable evidence."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CertifiedDashboardError(f"invalid required run artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise CertifiedDashboardError(f"required run artifact is not an object: {path.name}")
    return value


def _same(first: Path, second: Path, label: str) -> dict[str, Any]:
    value = _read(first)
    if value != _read(second):
        raise CertifiedDashboardError(f"{label} copies differ")
    return value


def _intent(value: Mapping[str, Any]) -> WeightIntent:
    required = set(WeightIntent.__dataclass_fields__)
    if set(value) != required:
        raise CertifiedDashboardError("weight journal intent fields are invalid")
    try:
        return WeightIntent(
            **{
                **dict(value),
                "uid_weights": {int(uid): weight for uid, weight in value["uid_weights"].items()},
                "expected_hotkeys": {
                    int(uid): hotkey for uid, hotkey in value["expected_hotkeys"].items()
                },
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CertifiedDashboardError("weight journal intent is invalid") from error


def _receipt(value: Mapping[str, Any]) -> WeightSubmissionReceipt:
    try:
        return WeightSubmissionReceipt(**dict(value))
    except (TypeError, ValueError) as error:
        raise CertifiedDashboardError("weight journal submission receipt is invalid") from error


def _run_evidence(
    run_dir: Path,
) -> tuple[
    RoundPlan,
    FinalizedCompetitionState,
    RoundCertificate,
    FinalizedCompetitionState,
    dict[str, Any],
]:
    plan = RoundPlan.from_dict(_read(run_dir / "round-plan.json"))
    parent = FinalizedCompetitionState.from_dict(_read(run_dir / "state.json"))
    certificate = RoundCertificate.from_dict(_read(run_dir / "certificate.json"))
    certified = FinalizedCompetitionState.from_dict(
        _same(
            run_dir / "certified-state-a.json",
            run_dir / "certified-state-b.json",
            "certified state",
        )
    )
    readback = _read(run_dir / "weight-readback.json")
    if plan.network != "test" or plan.netuid != 578:
        raise CertifiedDashboardError("Launch V1 dashboard projection is testnet 578 only")
    if readback.get("status") != "FINALIZED_WEIGHTS_VERIFIED":
        raise CertifiedDashboardError("weight activation is not finalized")
    if (
        readback.get("network") != plan.network
        or readback.get("netuid") != plan.netuid
        or readback.get("certificate_hash") != certificate.transition_hash()
        or readback.get("transition") != certificate.transition.value
        or not isinstance(readback.get("block"), int)
        or not isinstance(readback.get("runtime_hash"), str)
    ):
        raise CertifiedDashboardError("weight readback is not bound to the certified transition")
    return plan, parent, certificate, certified, readback


def _verify_validator(
    *,
    run_dir: Path,
    label: str,
    plan: RoundPlan,
    parent: FinalizedCompetitionState,
    certificate: RoundCertificate,
    certified: FinalizedCompetitionState,
    receipts: LocalLaunchReceiptStore,
    readback: Mapping[str, Any],
) -> None:
    journal = _read(run_dir / f"weight-submission-{label}.json")
    if not {
        "intent",
        "intent_hash",
        "validator_hotkey",
        "validator_uid",
        "submission_receipt",
        "extrinsic_id",
    } <= set(journal):
        raise CertifiedDashboardError(f"weight journal {label} is incomplete")
    intent, receipt = _intent(journal["intent"]), _receipt(journal["submission_receipt"])
    if (
        journal["intent_hash"] != intent.intent_hash()
        or receipt.intent_hash != intent.intent_hash()
        or journal["validator_hotkey"] not in plan.committee
        or type(journal["validator_uid"]) is not int
        or intent.network != plan.network
        or intent.netuid != plan.netuid
        or intent.certificate_hash != certificate.transition_hash()
        or intent.certified_state_hash != certified.state_hash()
        or intent.runtime_hash != readback["runtime_hash"]
    ):
        raise CertifiedDashboardError(f"weight journal {label} does not bind this certified round")
    results = readback.get("results")
    if not isinstance(results, list):
        raise CertifiedDashboardError("weight readback results are invalid")
    actual_uids = readback.get("registered_uids")
    if not isinstance(actual_uids, Mapping):
        raise CertifiedDashboardError("readback lacks finalized UID snapshot")
    try:
        registered = {int(uid): hotkey for uid, hotkey in actual_uids.items()}
    except (TypeError, ValueError) as error:
        raise CertifiedDashboardError("readback UID snapshot is invalid") from error
    if registered.get(journal["validator_uid"]) != journal["validator_hotkey"]:
        raise CertifiedDashboardError(f"weight journal {label} validator UID changed")
    result = next((item for item in results if item.get("validator") == label), None)
    if (
        not isinstance(result, dict)
        or result.get("status") != "FINALIZED_WEIGHT_VECTOR_VERIFIED"
        or result.get("intent_hash") != intent.intent_hash()
        or result.get("extrinsic_id") != journal["extrinsic_id"]
        or result.get("validator_uid") != journal["validator_uid"]
        or result.get("validator_hotkey") != journal["validator_hotkey"]
        or result.get("finalized_extrinsic_hash") != receipt.submitted_extrinsic_hash
        or not isinstance(result.get("weights_u16"), (Mapping, list))
    ):
        raise CertifiedDashboardError(f"weight readback for {label} is absent or unverified")
    try:
        raw_weights = result["weights_u16"]
        weights = (
            {int(uid): value for uid, value in raw_weights.items()}
            if isinstance(raw_weights, Mapping)
            else {int(uid): value for uid, value in raw_weights}
        )
        verify_finalized_replacement(
            intent,
            receipt,
            network=plan.network,
            netuid=plan.netuid,
            finalized_block=readback["block"],
            finalized_extrinsic_hash=receipt.submitted_extrinsic_hash,
            finalized_runtime_hash=readback["runtime_hash"],
            weights_u16=weights,
            registered_uids=registered,
        )
    except (TypeError, ValueError) as error:
        raise CertifiedDashboardError(
            f"weight readback {label} failed replacement verification"
        ) from error
    with Store(run_dir / f"validator-{label}.sqlite") as store:
        if store.certificate_payload(certificate.transition_hash()) != certificate.to_dict():
            raise CertifiedDashboardError(f"validator {label} lacks the stored certificate")
        if store.state_payload(parent.state_hash()) != parent.to_dict():
            raise CertifiedDashboardError(f"validator {label} lacks the certified parent state")
        if store.latest_state_payload(parent.challenge_id) != certified.to_dict():
            raise CertifiedDashboardError(
                f"validator {label} durable head differs from certified state"
            )
        verifier = LaunchCertificateVerifier(
            Sr25519Verifier(), replayer=LaunchExecutionReplayer(receipts), store=store
        )
        verifier.verify(certificate, plan, parent)
        expected = apply_certificate(parent, plan, certificate, verifier)
        if expected != certified:
            raise CertifiedDashboardError(
                f"validator {label} certified state does not follow certificate"
            )
        custody_path, burn_path = (
            run_dir / "public-custody.json",
            run_dir / "burn-qualification.json",
        )
        custody, burn = _read(custody_path), _read(burn_path)
        custody_hash, burn_hash = sha256_file(custody_path), sha256_file(burn_path)
        if (
            receipts.read("custody", custody_hash) != custody
            or receipts.read("burn", burn_hash) != burn
        ):
            raise CertifiedDashboardError("trusted custody or burn receipt bytes differ")
        reconstructed = LaunchV1Bridge(
            store, Sr25519Verifier(), receipts
        ).prepare_testnet_weight_intent(
            certified,
            plan=plan,
            certificate=certificate,
            runtime_hash=readback["runtime_hash"],
            registered_uids=registered,
            public_custody={plan.challenger_fingerprint: custody_hash},
            burn_hotkey=burn.get("hotkey"),
            burn_receipt_hash=burn_hash,
        )
        if reconstructed != intent:
            raise CertifiedDashboardError(f"weight journal {label} intent was not reconstructed")


def export_certified_snapshot(snapshot: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Return a copied dashboard snapshot only after complete V1 certification replay."""

    if snapshot.get("schema") != "forge/dashboard/v2":
        raise CertifiedDashboardError("diagnostic snapshot has the wrong schema")
    plan, parent, certificate, certified, readback = _run_evidence(run_dir)
    receipts = LocalLaunchReceiptStore(run_dir / "trusted-receipts")
    journal_hotkeys = []
    for label in ("a", "b"):
        journal_hotkeys.append(
            _read(run_dir / f"weight-submission-{label}.json")["validator_hotkey"]
        )
        _verify_validator(
            run_dir=run_dir,
            label=label,
            plan=plan,
            parent=parent,
            certificate=certificate,
            certified=certified,
            receipts=receipts,
            readback=readback,
        )
    if len(set(journal_hotkeys)) != 2 or any(
        hotkey not in plan.committee or hotkey not in certificate.signatures
        for hotkey in journal_hotkeys
    ):
        raise CertifiedDashboardError(
            "weight journals are not two distinct certified committee validators"
        )
    return _project_certified(snapshot, run_dir, plan, certificate, certified, readback, 2)


def export_native_certified_snapshot(
    snapshot: Mapping[str, Any], run_dir: Path, *, chain: Any,
    validator_hotkeys: Mapping[str, str],
) -> dict[str, Any]:
    """Replay V2 authority and recheck native attribution without opening a wallet."""
    from forge.chain.launch_authority import QualifiedLaunchAuthority
    from forge.chain.launch_native_publication import TestnetNativePublicationReader
    from forge.chain.launch_publication import QualifiedPublicationContext

    if snapshot.get("schema") != "forge/dashboard/v2":
        raise CertifiedDashboardError("diagnostic snapshot has the wrong schema")
    plan = RoundPlan.from_dict(_read(run_dir / "round-plan.json"))
    parent = FinalizedCompetitionState.from_dict(_read(run_dir / "state.json"))
    certificate = RoundCertificate.from_dict(_read(run_dir / "certificate.json"))
    activation = _read(run_dir / "activation-plan.json")
    public_custody = _read(run_dir / "public-custody-set.json")
    if (plan.network, plan.netuid) != ("test", 578):
        raise CertifiedDashboardError("native dashboard projection is testnet 578 only")
    if (
        not validator_hotkeys or len(set(validator_hotkeys.values())) != len(validator_hotkeys)
        or any(key not in certificate.signatures for key in validator_hotkeys.values())
    ):
        raise CertifiedDashboardError("native validators must be distinct certificate signers")
    receipts = LocalLaunchReceiptStore(run_dir / "trusted-receipts")
    states, finals, contexts = [], [], []
    for label, hotkey in validator_hotkeys.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", label):
            raise CertifiedDashboardError("invalid native validator label")
        intent = _intent(_read(run_dir / f"weight-intent-{label}.json"))
        try:
            with sqlite3.connect(
                f"file:{run_dir / f'publication-{label}.sqlite'}?mode=ro", uri=True,
            ) as db:
                db.row_factory = sqlite3.Row
                row = db.execute(
                    "SELECT * FROM launch_publication_intents WHERE intent_hash=?",
                    (intent.intent_hash(),),
                ).fetchone()
        except sqlite3.Error as error:
            raise CertifiedDashboardError("native publication journal unavailable") from error
        if row is None or row["status"] != "FINALIZED" or not row["receipt_json"]:
            raise CertifiedDashboardError("native publication is not finalized")
        context = QualifiedPublicationContext(**json.loads(row["context_json"]))
        submission = _receipt(json.loads(row["receipt_json"]))
        if (
            row["context_hash"] != context.context_hash()
            or _intent(json.loads(row["intent_json"])) != intent
        ):
            raise CertifiedDashboardError("native publication journal identity differs")
        certified = FinalizedCompetitionState.from_dict(
            _read(run_dir / f"certified-state-{label}.json")
        )
        persisted = _read(run_dir / f"weights-readback-{label}.json")
        registered = {int(uid): key for uid, key in persisted["registered_uids"].items()}
        uid = next((uid for uid, key in registered.items() if key == hotkey), None)
        witness = run_dir / f"native-witness-{label}.sqlite"
        if uid is None or hotkey not in plan.committee or not witness.is_file():
            raise CertifiedDashboardError("native signer identity or witness is absent")
        with Store(run_dir / f"validator-{label}.sqlite") as store:
            authority = QualifiedLaunchAuthority(
                store, plan, parent, certificate, activation, str(receipts.root),
            )
            if (
                not authority.verify(context, intent)
                or certified.state_hash() != context.certified_state_hash
            ):
                raise CertifiedDashboardError("native publication authority failed replay")
            burn = (
                receipts.read("burn", intent.burn_qualification_hash)
                if intent.burn_qualification_hash else {}
            )
            bridge = LaunchV1Bridge(store, Sr25519Verifier(), receipts)
            expected = bridge.prepare_testnet_weight_intent(
                certified, plan=plan, certificate=certificate, runtime_hash=context.runtime_hash,
                registered_uids=registered,
                public_custody=public_custody,
                burn_hotkey=burn.get("hotkey"), burn_receipt_hash=intent.burn_qualification_hash,
            )
            if expected != intent:
                raise CertifiedDashboardError("native weight intent failed reconstruction")
        final = TestnetNativePublicationReader(
            chain, validator_hotkey=hotkey, validator_uid=uid, witness_path=witness,
        ).finalized_weights(context, intent, submission)
        if final is None or (
            final.genesis_hash != context.genesis_hash or final.mechanism != context.mechanism
            or final.last_update_block < submission.submitted_block
            or final.last_step_block <= submission.submitted_block
            or final.registered_uids.get(uid) != hotkey
        ):
            raise CertifiedDashboardError("native effective weights are not attributable")
        verify_finalized_replacement(
            intent, submission, network=plan.network, netuid=plan.netuid,
            finalized_block=final.finalized_block, finalized_extrinsic_hash=final.extrinsic_hash,
            finalized_runtime_hash=final.runtime_hash, weights_u16=final.weights_u16,
            registered_uids=final.registered_uids,
        )
        states.append(certified)
        finals.append(final)
        contexts.append(context.context_hash())
    if len({state.state_hash() for state in states}) != 1 or len(set(contexts)) != 1:
        raise CertifiedDashboardError("native validators disagree on certified authority")
    if any(final.registered_uids != finals[0].registered_uids for final in finals):
        raise CertifiedDashboardError("native UID mapping changed during export")
    return _project_certified(
        snapshot, run_dir, plan, certificate, states[0],
        {"block": max(final.finalized_block for final in finals),
         "runtime_hash": finals[0].runtime_hash, "registered_uids": finals[0].registered_uids},
        len(validator_hotkeys),
    )


def _project_certified(
    snapshot: Mapping[str, Any], run_dir: Path, plan: RoundPlan,
    certificate: RoundCertificate, certified: FinalizedCompetitionState,
    readback: Mapping[str, Any], validator_count: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(snapshot))
    registered = {int(uid): hotkey for uid, hotkey in readback["registered_uids"].items()}
    result.setdefault("network", {}).update(
        name=plan.network, netuid=plan.netuid, finalized_block=readback["block"]
    )
    result["validators"] = [
        {
            "hotkey": hotkey,
            "uid": next(uid for uid, value in registered.items() if value == hotkey),
            "stake": stake / 1_000_000_000,
            "active": True,
            "stake_snapshot_block": plan.snapshot_block,
        }
        for hotkey, stake in sorted(plan.committee.items())
        if hotkey in registered.values()
    ]
    campaign_id = run_dir.name
    diagnostic_round = f"operator-experiment-{campaign_id}"
    challenger_id = f"launch-v1-{campaign_id}-{plan.challenger_fingerprint[:12]}"
    challenger = next(
        (row for row in result.get("challengers", []) if row.get("id") == challenger_id), None
    )
    evaluation = next(
        (row for row in result.get("evaluations", []) if row.get("challenger_id") == challenger_id),
        None,
    )
    diagnostic = next(
        (row for row in result.get("rounds", []) if row.get("round_id") == diagnostic_round), None
    )
    if (
        challenger is None
        or evaluation is None
        or challenger.get("round_id") != diagnostic_round
        or not isinstance(diagnostic, dict)
        or diagnostic.get("diagnostic") is not True
        or evaluation.get("plan_hash") != plan.prompt_seed_inputs.get("launch_plan_id")
    ):
        raise CertifiedDashboardError("diagnostic snapshot lacks the exact replayed challenger")
    if challenger.get("manifest_hash") != plan.challenger_fingerprint:
        raise CertifiedDashboardError("diagnostic challenger fingerprint differs from certificate")
    diagnostic.update(
        {
            "round_id": plan.round_id,
            "origin": "testnet_chain",
            "stage": "final",
            "diagnostic": False,
            "close_block": plan.snapshot_block,
            "entropy_block": plan.prompt_seed_inputs.get("entropy_block"),
            "commit_close_block": plan.commit_deadline_block,
            "shared_execution": True,
            "independent_operators": False,
            "committee": [
                {"hotkey": hotkey, "stake": stake / 1_000_000_000}
                for hotkey, stake in sorted(plan.committee.items())
            ],
        }
    )
    challenger["round_id"] = plan.round_id
    challenger["verdict"] = certificate.transition.value
    evaluation["execution"] = {
        "shared": True,
        "independent_operators": False,
        "validator_count": validator_count,
    }
    evaluation["consensus"] = {
        "rule": plan.consensus_policy.value,
        "favorable_stake": sum(
            plan.committee[reveal.report.validator_hotkey]
            for reveal in certificate.reveals
            if reveal.report.verdict.value == "PROMOTE"
        ) / 1_000_000_000,
        "total_stake": sum(plan.committee.values()) / 1_000_000_000,
        "reports": [
            {
                "validator": reveal.report.validator_hotkey,
                "status": "valid",
                "report_sha256": reveal.report.report_hash(),
                "anchor_block": next(
                    commit.commit_block
                    for commit in certificate.commits
                    if commit.validator_hotkey == reveal.report.validator_hotkey
                ),
                "prompts": [prompt["id"] for prompt in evaluation.get("prompts", [])],
                "audits": [],
            }
            for reveal in certificate.reveals
        ],
    }
    for prompt in evaluation.get("prompts", []):
        prompt["validators"] = sorted(
            {reveal.report.validator_hotkey for reveal in certificate.reveals}
        )
    # The public dashboard model projection is already normalized in `challenger`;
    # retain it rather than fetching mutable artifact metadata during export.
    transition = certificate.transition.value
    activation = {
        "status": "FINALIZED_WEIGHTS_VERIFIED",
        "finalized_block": readback["block"],
        "runtime_hash": readback["runtime_hash"],
        "validator_count": validator_count,
    }
    result["launch_v1_certification"] = {
        "round_id": plan.round_id,
        "status": "CERTIFIED",
        "transition": transition,
        "certificate_hash": certificate.transition_hash(),
        "certified_state_hash": certified.state_hash(),
        "activation": activation,
    }
    evaluation["certification"] = result["launch_v1_certification"]
    if transition == "KEEP":
        challenger["status"] = "kept"
        evaluation["verdict"] = {**evaluation.get("verdict", {}), "status": "KEEP"}
        return result
    crown = next(
        (
            item
            for item in certified.crown_history
            if item.crown_id == certificate.transition_hash()
        ),
        None,
    )
    if crown is None or crown.fingerprint != plan.challenger_fingerprint:
        raise CertifiedDashboardError("promotion certificate has no matching certified crown")
    challenger["status"] = "promoted"
    challenger["hotkey"] = crown.hotkey
    evaluation["verdict"] = {**evaluation.get("verdict", {}), "status": "PROMOTE"}
    result["king"] = {
        "hotkey": crown.hotkey,
        "uid": next((uid for uid, hotkey in registered.items() if hotkey == crown.hotkey), None),
        "label": challenger.get("label"),
        "model": challenger["model"],
        "manifest_hash": crown.fingerprint,
        "since_block": readback["block"],
        "crown_id": crown.crown_id,
        "speed_ratio_vs_wan50": None,
        "quality_vs_wan50": None,
    }
    crowns = result.setdefault("crowns", [])
    if not any(item.get("crown_id") == crown.crown_id for item in crowns):
        crowns.insert(
            0,
            {
                "crown_id": crown.crown_id,
                "hotkey": crown.hotkey,
                "label": challenger.get("label"),
                "block": readback["block"],
                "speed_ratio": None,
                "quality": None,
            },
        )
    return result
