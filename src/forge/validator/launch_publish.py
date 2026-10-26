"""Connect certified V2 rounds to the durable native testnet publisher.

No wallet material crosses this boundary. Callers pass already selected local
wallet handles; each validator has its own submission journal and native
ciphertext witness. A round completes only after every configured validator's
new vector is proven effective.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge.chain import Sr25519Verifier
from forge.chain.launch_authority import QualifiedLaunchAuthority
from forge.chain.launch_native_publication import TestnetNativePublicationReader
from forge.chain.launch_publication import (
    BittensorSdkTestnetSigner,
    LaunchPublicationStore,
    LaunchPublisher,
    QualifiedPublicationContext,
)
from forge.models.artifact_manifest import ModelManifest
from forge.models.artifacts import snapshot_manifest
from forge.protocol.consensus_types import FinalizedCompetitionState, RoundCertificate, RoundPlan
from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical
from forge.protocol.store import Store
from forge.validator.launch_bridge import LaunchV1Bridge, LocalLaunchReceiptStore
from forge.validator.launch_config import LaunchConfig
from forge.validator.launch_service import atomic_json, write_immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("publication artifact must contain a JSON object")
    return value


def _receipt(root: Path, kind: str, value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path = root / kind / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("publication receipt hash collision")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    return digest


def _public_custody(
    root: Path, receipts: LocalLaunchReceiptStore, state: FinalizedCompetitionState
) -> dict[str, str]:
    latest = state.crown_history[-1] if state.crown_history else None
    if latest is None or latest.baseline:
        return {}
    # The manifest comes from a replayed validator report. Re-resolve its public
    # registry revision and every declared file identity before naming a reward.
    manifest = None
    for path in sorted((receipts.root / "panels").glob("*.json")):
        panel = _read(path)
        if panel.get("manifest_hash") == state.current_king_fingerprint:
            manifest = ModelManifest.from_dict(panel["manifest"])
            break
    if manifest is None or manifest.fingerprint() != state.current_king_fingerprint:
        raise ValueError("certified winner has no custodied public manifest")
    resolved = snapshot_manifest(
        manifest.reference,
        role=manifest.role,
        generation_config=manifest.generation_config,
        base_reference=manifest.base_reference,
    )
    if resolved.fingerprint() != manifest.fingerprint():
        raise ValueError("winner is no longer anonymously available with its pinned file hashes")
    return {
        manifest.fingerprint(): _receipt(
            root,
            "custody",
            {
                "schema": "forge/launch-public-custody/v1",
                "fingerprint": manifest.fingerprint(),
                "publicly_downloadable": True,
            },
        )
    }


def publication_callback(
    root: Path,
    *,
    configuration: dict[str, Any],
    config: LaunchConfig,
    contract: ChallengeContract,
    chain: Any,
    wallets: dict[str, Any],
    shadow: bool,
):
    """Construct the concrete certified path; bootstrap has separate authority."""
    if set(configuration) != {"evidence_origin", "validator_labels"}:
        raise ValueError("publication configuration must name provenance and validator labels")
    if configuration["evidence_origin"] != "external_gpu_qualified_v2":
        raise ValueError("live publication requires actual qualified external GPU evidence")
    labels = configuration["validator_labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) for label in labels)
    ):
        raise ValueError("publication validator labels are invalid")
    if not shadow and set(labels) != set(wallets):
        raise ValueError("publication must cover exactly the selected local validators")
    root = Path(root).resolve()

    def publish(evidence: dict[str, Any]) -> dict[str, Any] | None:
        if shadow:
            return None
        from forge.chain.launch_fallback import qualify_fallback
        from forge.validator.launch_cli import preflight

        round_dir = Path(evidence["round_dir"]).resolve()
        if round_dir.parent != root:
            raise ValueError("publication round is outside the configured campaign")
        plan = RoundPlan.from_dict(_read(round_dir / "round-plan.json"))
        parent = FinalizedCompetitionState.from_dict(_read(round_dir / "state.json"))
        certificate = RoundCertificate.from_dict(evidence["certificate"])
        certified = FinalizedCompetitionState.from_dict(evidence["state"])
        fresh = preflight(config, contract, chain)
        fallback = qualify_fallback(
            chain,
            genesis_hash=config.genesis_hash,
            runtime_hash=fresh["runtime_hash"],
            finalized_block=fresh["finalized_block"],
        )
        # Freeze authority before the first submission. Fresh native reads must
        # continue agreeing with that target and runtime during reconciliation.
        fallback_file = round_dir / "fallback-proof.json"
        if fallback_file.exists():
            original = _read(fallback_file)
            for key in ("target_uid", "target_hotkey", "runtime_hash", "genesis_hash"):
                if original[key] != fallback[key]:
                    raise ValueError("qualified fallback changed during publication")
        else:
            write_immutable_json(fallback_file, fallback)
        receipt_root = round_dir / "trusted-receipts"
        receipts = LocalLaunchReceiptStore(receipt_root)
        burn_hash = _receipt(
            receipt_root,
            "burn",
            {
                "schema": "forge/launch-burn-qualification/v1",
                "hotkey": fallback["target_hotkey"],
                "runtime_hash": fresh["runtime_hash"],
                "network": config.network,
                "netuid": config.netuid,
            },
        )
        public = _public_custody(receipt_root, receipts, certified)
        write_immutable_json(round_dir / "public-custody-set.json", public)
        activation_body = {
            "network": config.network,
            "netuid": config.netuid,
            "genesis_hash": config.genesis_hash,
            "runtime_hash": fresh["runtime_hash"],
            "mechanism": 0,
            "contract_hash": config.contract_hash,
            "round_plan_hash": plan.plan_hash(),
            "certified_state_hash": certified.state_hash(),
            "certificate_hash": certificate.transition_hash(),
            "evidence_origin": configuration["evidence_origin"],
            "policy_plan": _read(round_dir / "launch-plan.json"),
            "qualification_package": _read(round_dir / "qualification-package.json"),
        }
        activation = {
            **activation_body,
            "activation_plan_hash": hash_canonical(
                "forge/launch-activation-plan/v1", activation_body
            ),
        }
        write_immutable_json(round_dir / "activation-plan.json", activation)
        context = QualifiedPublicationContext(
            mode="bittensor-sdk/testnet-qualified-v1",
            network=config.network,
            netuid=config.netuid,
            genesis_hash=config.genesis_hash,
            runtime_hash=fresh["runtime_hash"],
            mechanism=0,
            contract_hash=config.contract_hash,
            plan_hash=plan.plan_hash(),
            certified_state_hash=certified.state_hash(),
            certificate_hash=certificate.transition_hash(),
            activation_plan_hash=activation["activation_plan_hash"],
        )
        statuses, observed, intents = {}, {}, []
        for label in labels:
            wallet = wallets[label]
            address = wallet.hotkey.ss58_address
            uids = {int(uid): hotkey for uid, hotkey in fresh["registered_uids"].items()}
            uid = next((key for key, value in uids.items() if value == address), None)
            if uid is None or address not in plan.committee:
                raise ValueError(
                    "publication signer is no longer registered in the frozen committee"
                )
            with Store(round_dir / f"validator-{label}.sqlite") as store:
                authority = QualifiedLaunchAuthority(
                    store, plan, parent, certificate, activation, str(receipt_root)
                )
                intent = LaunchV1Bridge(
                    store, Sr25519Verifier(), receipts
                ).prepare_testnet_weight_intent(
                    certified,
                    plan=plan,
                    certificate=certificate,
                    runtime_hash=fresh["runtime_hash"],
                    registered_uids=uids,
                    public_custody=public,
                    burn_hotkey=fallback["target_hotkey"],
                    burn_receipt_hash=burn_hash,
                )
                write_immutable_json(round_dir / f"weight-intent-{label}.json", asdict(intent))
                native = TestnetNativePublicationReader(
                    chain,
                    validator_hotkey=address,
                    validator_uid=uid,
                    witness_path=round_dir / f"native-witness-{label}.sqlite",
                )
                journal = LaunchPublicationStore(round_dir / f"publication-{label}.sqlite")
                publisher = LaunchPublisher(
                    journal,
                    BittensorSdkTestnetSigner.with_native_reader(chain, wallet, native),
                    authority.verify,
                    lambda _: False,
                    lambda *_: False,
                    lambda ctx, proposed: (
                        proposed.burn_qualification_hash == burn_hash
                        and proposed.uid_weights == {fallback["target_uid"]: 1.0}
                        and proposed.expected_hotkeys
                        == {fallback["target_uid"]: fallback["target_hotkey"]}
                        and ctx.runtime_hash == fallback["runtime_hash"]
                    ),
                )
                status = publisher.publish_or_reconcile(context, intent)
                statuses[label] = status
                intents.append(intent.intent_hash())
                _, submission = journal.load(intent.intent_hash())
                if submission is not None:
                    observed[label] = {"submission": asdict(submission)}
                    if status == "FINALIZED":
                        readback_file = round_dir / f"weights-readback-{label}.json"
                        if readback_file.exists():
                            readback = _read(readback_file)
                        else:
                            final = native.finalized_weights(context, intent, submission)
                            if final is None:
                                raise ValueError("finalized native readback became unavailable")
                            write_immutable_json(readback_file, asdict(final))
                            # Hash the persisted JSON identically across restart.
                            readback = _read(readback_file)
                        observed[label]["readback"] = readback
        atomic_json(
            round_dir / "publication-status.json",
            {
                "schema": "forge/launch-native-publication-status/v1",
                "validators": statuses,
                "evidence": observed,
            },
        )
        if len(set(intents)) != 1:
            raise ValueError("validators prepared different certified weight intents")
        if len(observed) != len(labels):
            return None
        result = {
            "status": "FINALIZED" if set(statuses.values()) == {"FINALIZED"} else "SUBMITTED",
            "certificate_hash": certificate.transition_hash(),
            "intent_hash": intents[0],
            "submitted_extrinsic_hash": observed[labels[0]]["submission"][
                "submitted_extrinsic_hash"
            ],
        }
        if result["status"] == "FINALIZED":
            result.update(
                finalized_block=max(
                    value["readback"]["finalized_block"] for value in observed.values()
                ),
                readback_hash=hash_canonical("forge/launch-all-validators-readback/v1", observed),
            )
        return result

    return publish
