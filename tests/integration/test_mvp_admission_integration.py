"""End-to-end public admission bridge using real sr25519 checkpoint signatures."""

from types import SimpleNamespace

import pytest
from bittensor.keyfiles import Keypair

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.admission_checkpoint import create_signed_history
from forge.protocol.hashing import hash_canonical
from forge.validator.admission import (
    AdmissionError,
    AdmissionLedger,
    FinalizedCommitment,
    build_authenticated_history,
)
from forge.validator.mvp_admission import (
    PublicAdmissionError,
    PublicCandidate,
    admit_public_candidates,
)
from forge.validator.runtime_preflight import PASS


def digest(value: int) -> str:
    return f"{value:064x}"


def manifest(number: int) -> ModelManifest:
    return ModelManifest(
        ArtifactReference("hf", f"public/model{number}", "a" * 40),
        "lora",
        (ManifestFile("adapter.safetensors", digest(number), 1),),
        "apache-2.0",
        {"steps": 8, "guidance_scale": 5.0},
        ArtifactReference("hf", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "b" * 40),
    )


def candidate(
    hotkey: str, index: int, model: ModelManifest, effective: str | None = None
) -> PublicCandidate:
    effective, panel = effective or digest(30 + index), digest(40)
    recipe = hash_canonical(
        "forge/mvp-admission-recipe/v1",
        {
            "manifest_hash": model.fingerprint(),
            "generation_config": dict(model.generation_config),
            "effective_weights_hash": effective,
        },
    )
    return PublicCandidate(
        FinalizedCommitment(
            hotkey, 99, index, digest(10 + index), model.fingerprint(), recipe, panel
        ),
        model,
        effective,
        panel,
    )


def test_public_batch_binds_signed_history_manifest_preflight_recipe_panel_and_fifo(tmp_path):
    keys = [Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob", "//Charlie")]
    signers = [Sr25519Signer(SimpleNamespace(hotkey=key)) for key in keys]
    committee = {signer.validator_hotkey: 1 for signer in signers}
    history = build_authenticated_history(
        network="test",
        netuid=9,
        through_block=100,
        certified_state_hash=digest(1),
        consumed_hotkeys=(),
        registered_hotkeys=("miner-a", "miner-b", "miner-c"),
    )
    signed = create_signed_history(history, signers)
    first, second = candidate("miner-a", 1, manifest(1)), candidate("miner-b", 2, manifest(2))
    plan_hash = digest(99)
    receipt = {
        "plan_hash": plan_hash,
        "candidates": [
            {
                "manifest_hash": first.manifest.fingerprint(),
                "effective_candidate_hash": first.effective_weights_hash,
                "status": PASS,
            },
            {
                "manifest_hash": second.manifest.fingerprint(),
                "effective_candidate_hash": second.effective_weights_hash,
                "status": PASS,
            },
        ],
    }
    ledger = AdmissionLedger(tmp_path / "validator.sqlite", network="test", netuid=9)
    admitted = admit_public_candidates(
        ledger,
        (first, second),
        preflight_receipt=receipt,
        expected_preflight_plan_hash=plan_hash,
        signed_history=signed,
        trusted_committee=committee,
        expected_certified_state_hash=digest(1),
        signature_verifier=Sr25519Verifier(),
    )
    assert [item.hotkey for item in admitted] == ["miner-a", "miner-b"]
    duplicate = candidate("miner-c", 3, manifest(3), first.effective_weights_hash)
    duplicate_receipt = {
        "plan_hash": plan_hash,
        "candidates": [
            {
                "manifest_hash": first.manifest.fingerprint(),
                "effective_candidate_hash": first.effective_weights_hash,
                "status": PASS,
            },
            {
                "manifest_hash": duplicate.manifest.fingerprint(),
                "effective_candidate_hash": duplicate.effective_weights_hash,
                "status": PASS,
            },
        ],
    }
    with pytest.raises(PublicAdmissionError, match="duplicate"):
        admit_public_candidates(
            AdmissionLedger(tmp_path / "other.sqlite", network="test", netuid=9),
            (first, duplicate),
            preflight_receipt=duplicate_receipt,
            expected_preflight_plan_hash=plan_hash,
            signed_history=signed,
            trusted_committee=committee,
            expected_certified_state_hash=digest(1),
            signature_verifier=Sr25519Verifier(),
        )
    later_receipt = {
        "plan_hash": plan_hash,
        "candidates": [
            {
                "manifest_hash": duplicate.manifest.fingerprint(),
                "effective_candidate_hash": duplicate.effective_weights_hash,
                "status": PASS,
            }
        ],
    }
    with pytest.raises(AdmissionError, match="effective candidate identity"):
        admit_public_candidates(
            AdmissionLedger(tmp_path / "validator.sqlite", network="test", netuid=9),
            (duplicate,),
            preflight_receipt=later_receipt,
            expected_preflight_plan_hash=plan_hash,
            signed_history=signed,
            trusted_committee=committee,
            expected_certified_state_hash=digest(1),
            signature_verifier=Sr25519Verifier(),
        )
