import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("bittensor")

from bittensor.keyfiles import Keypair

from forge.chain import OnChainCommitment, Sr25519Signer, Sr25519Verifier
from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.protocol.contract import ChallengeContract
from forge.protocol.evaluation_anchor import (
    prepare_evaluation_anchor,
    verify_finalized_evaluation_anchor,
)
from forge.protocol.finalist_lock import build_finalist_lock, finalist_lock_hash


def _sha(number: int) -> str:
    return f"{number:064x}"


class _Wallet:
    def __init__(self, keypair):
        self.hotkey = keypair


@dataclass
class ReplayTestnetChain:
    """Explicit finalized replay port for an anchor, with no chain SDK mock."""

    hotkey: str
    network: str = "test"
    netuid: int = 578
    current_head: int = 140
    finalized: int = 139
    observed: OnChainCommitment | None = None
    closed: dict[str, OnChainCommitment] = field(default_factory=dict)
    history: dict[tuple[str, int], OnChainCommitment] = field(default_factory=dict)

    def head(self) -> int:
        return self.current_head

    def finalized_height(self) -> int:
        return self.finalized

    def finalized_block_hash(self, height: int) -> str:
        assert height <= self.finalized
        return f"0x{height:064x}"

    def uid_snapshot(self, height: int) -> dict[int, str]:
        assert height in {100, self.finalized}
        return {0: self.hotkey, 1: "miner-public"}

    def committee_snapshot(self, height: int) -> dict[str, int]:
        assert height in {100, self.finalized}
        return {self.hotkey: 1_000}

    def commitments(self, hotkeys: list[str], height: int) -> list[OnChainCommitment]:
        assert height == 100
        return [self.closed[hotkey] for hotkey in hotkeys if hotkey in self.closed]

    def commitment(self, hotkey: str, height: int) -> OnChainCommitment | None:
        if height == 100:
            return self.closed.get(hotkey)
        if (hotkey, height) in self.history:
            return self.history[(hotkey, height)]
        if hotkey == self.hotkey and height == self.finalized:
            return self.observed
        return None


@contextmanager
def public_registry(manifest: ModelManifest):
    reference = manifest.reference

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            expected_path = f"/api/models/{reference.repo_id}/revision/{reference.revision}"
            if path != expected_path:
                self.send_error(404)
                return
            body = json.dumps(
                {
                    "private": False,
                    "gated": False,
                    "disabled": False,
                    "sha": reference.revision,
                    "cardData": {"license": manifest.license},
                    "siblings": [
                        {
                            "rfilename": item.path,
                            "size": item.bytes,
                            "lfs": {"sha256": item.sha256, "size": item.bytes},
                        }
                        for item in manifest.files
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    candidate, incumbent = _sha(1), _sha(2)
    plan = {
        "network": "test",
        "netuid": 578,
        "round_id": "r-negative-evidence",
        "close_block": 100,
        "admitted": [{"manifest_hash": candidate, "hotkey": "miner-public", "block": 99}],
        "candidate": {
            "manifest_hash": candidate,
            "hotkey": "miner-public",
            "admission_block": 99,
        },
        "incumbent": {
            "manifest_hash": incumbent,
            "certified_state_hash": _sha(3),
            "certified_activation_block": 90,
        },
    }
    policy = {"evaluator_id": "luna-vision/v1"}
    quality = {
        "evaluator_id": "luna-vision/v1",
        "comparisons": [
            {
                "reference_role": "golden",
                "reference_manifest_hash": _sha(4),
                "candidate_manifest_hash": candidate,
                "reference_admission": "public_snapshot",
                "candidate_admission": "public_snapshot",
            },
            {
                "reference_role": "incumbent",
                "reference_manifest_hash": incumbent,
                "candidate_manifest_hash": candidate,
                "reference_admission": "public_snapshot",
                "candidate_admission": "public_snapshot",
            },
        ],
    }
    timing = {
        "reference_manifest_hash": incumbent,
        "candidate_manifest_hash": candidate,
        "reference_admission": "public_snapshot",
        "candidate_admission": "public_snapshot",
    }
    return plan, policy, quality, timing, {"schema": "incomplete-calibration"}


def test_negative_record_is_signed_preflighted_and_read_back_without_reward():
    signer = Sr25519Signer(_Wallet(Keypair.create_from_uri("//Alice")))
    chain = ReplayTestnetChain(signer.validator_hotkey)
    intent = prepare_evaluation_anchor(
        chain,
        signer,
        Sr25519Verifier(),
        *_inputs(),
        evaluator_journal_sha256=_sha(9),
        development=True,
    )

    receipt = intent.receipt(status="DRY_RUN")
    assert intent.record.gate_status == "INCOMPLETE"
    assert receipt["commitment"] == f"forge:v3:eval:{intent.record.record_hash()}"
    assert receipt["statement"] == "SIGNED_EVALUATION_EVIDENCE_ONLY_NOT_A_CERTIFICATE_OR_REWARD"
    assert receipt["signed_record"]["validator_hotkey"] == signer.validator_hotkey
    chain.observed = OnChainCommitment(
        signer.validator_hotkey, 139, "evaluation", (intent.record.record_hash(),)
    )
    observed, finalized, block_hash = verify_finalized_evaluation_anchor(chain, intent)
    assert (observed.block, finalized, block_hash) == (139, 139, chain.finalized_block_hash(139))


def test_active_sealed_pack_cannot_be_replaced_by_an_evaluation_anchor():
    signer = Sr25519Signer(_Wallet(Keypair.create_from_uri("//Alice")))
    chain = ReplayTestnetChain(signer.validator_hotkey, current_head=100)
    plan, policy, quality, timing, calibration = _inputs()
    plan["pack_commitment"] = {"hash": _sha(8), "block": 99, "hotkey": signer.validator_hotkey}
    with pytest.raises(ValueError, match="active sealed"):
        prepare_evaluation_anchor(
            chain,
            signer,
            Sr25519Verifier(),
            plan,
            policy,
            quality,
            timing,
            calibration,
            evaluator_journal_sha256=_sha(9),
            development=True,
        )


def _chain_bound_plan(chain, signer):
    reference = ArtifactReference("hf", "public/candidate", "a" * 40)
    base = ArtifactReference(
        "hf", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"
    )
    manifest = ModelManifest(
        reference,
        "full_transformer",
        (ManifestFile("transformer/model.safetensors", _sha(7), 1),),
        "apache-2.0",
        {"steps": 12, "guidance_scale": 5.0},
        base,
    )
    candidate = manifest.fingerprint()
    pack_hash = _sha(8)
    chain.closed = {
        "miner-public": OnChainCommitment(
            "miner-public",
            99,
            "model_v3",
            ("hf", "public/candidate", "a" * 40, "full_transformer", "12", "5"),
        ),
        signer.validator_hotkey: OnChainCommitment(
            signer.validator_hotkey, 98, "pack", (pack_hash,)
        ),
    }
    plan, policy, quality, timing, calibration = _inputs()
    contract = ChallengeContract(max_steps=16)
    plan.update(
        contract=contract.to_dict(),
        contract_hash=contract.contract_hash(),
        admission_algorithm="forge/closed-admission/v1",
        close_block_hash=chain.finalized_block_hash(100),
        entropy_block=101,
        entropy=chain.finalized_block_hash(101),
        committee_rao={signer.validator_hotkey: 1_000},
        prompt_source="forge/sealed-pack/v1",
        pack_commitment={"hash": pack_hash, "block": 98, "hotkey": signer.validator_hotkey},
        admitted=[
            {
                "manifest_hash": candidate,
                "hotkey": "miner-public",
                "block": 99,
                "manifest": manifest.to_dict(),
                "commitment_kind": "model_v3",
                "admission_status": "METADATA_VALIDATED_RUNTIME_PREFLIGHT_PENDING",
            }
        ],
        candidate={"manifest_hash": candidate, "hotkey": "miner-public", "admission_block": 99},
    )
    for row in quality["comparisons"]:
        row["candidate_manifest_hash"] = candidate
    timing["candidate_manifest_hash"] = candidate
    return plan, policy, quality, timing, calibration, manifest


def test_chain_bound_anchor_reconstructs_canonical_admissions_before_signing():
    signer = Sr25519Signer(_Wallet(Keypair.create_from_uri("//Alice")))
    chain = ReplayTestnetChain(signer.validator_hotkey)
    *inputs, manifest = _chain_bound_plan(chain, signer)
    with public_registry(manifest) as registry:
        intent = prepare_evaluation_anchor(
            chain,
            signer,
            Sr25519Verifier(),
            *inputs,
            evaluator_journal_sha256=_sha(9),
            hf_base_url=registry,
        )
        assert intent.record.candidate_manifest_hash == inputs[0]["candidate"]["manifest_hash"]
        broken = _chain_bound_plan(chain, signer)
        *broken_inputs, broken_manifest = broken
        broken_inputs[0]["close_block_hash"] = "forged"
        with pytest.raises(ValueError, match="close block hash"):
            prepare_evaluation_anchor(
                chain,
                signer,
                Sr25519Verifier(),
                *broken_inputs,
                evaluator_journal_sha256=_sha(9),
                hf_base_url=registry,
            )
        broken = _chain_bound_plan(chain, signer)
        *broken_inputs, broken_manifest = broken
        broken_inputs[0]["admitted"] = []
        with pytest.raises(ValueError, match="canonical admissions"):
            prepare_evaluation_anchor(
                chain,
                signer,
                Sr25519Verifier(),
                *broken_inputs,
                evaluator_journal_sha256=_sha(9),
                hf_base_url=registry,
            )
        chain.closed["miner-public"] = OnChainCommitment(
            "miner-public",
            99,
            "model_v3",
            ("hf", "public/other", "b" * 40, "full_transformer", "12", "5"),
        )
        with pytest.raises(ValueError, match="canonical admissions"):
            prepare_evaluation_anchor(
                chain,
                signer,
                Sr25519Verifier(),
                *inputs,
                evaluator_journal_sha256=_sha(9),
                hf_base_url=registry,
            )


def test_confirmation_anchor_rejects_unknown_earlier_lock_before_valid_second_lock():
    signer = Sr25519Signer(_Wallet(Keypair.create_from_uri("//Alice")))
    chain = ReplayTestnetChain(signer.validator_hotkey)
    *inputs, manifest = _chain_bound_plan(chain, signer)
    plan = inputs[0]
    plan.update(stage="confirmation", entropy_block=105, entropy=chain.finalized_block_hash(105))
    lock_fields = {
        "schema": "forge/finalist-lock/v1",
        "network": "test",
        "netuid": 578,
        "round_id": plan["round_id"],
        "close_block": 100,
        "close_block_hash": plan["close_block_hash"],
        "contract_hash": plan["contract_hash"],
        "candidate_manifest_hash": plan["candidate"]["manifest_hash"],
        "candidate_hotkey": "miner-public",
        "candidate_admission_block": 99,
        "incumbent_state_hash": plan["incumbent"]["certified_state_hash"],
    }
    second = build_finalist_lock(lock_fields)
    chain.history[(signer.validator_hotkey, 104)] = OnChainCommitment(
        signer.validator_hotkey, 104, "finalist_lock", (finalist_lock_hash(second),)
    )
    chain.history[(signer.validator_hotkey, 103)] = OnChainCommitment(
        signer.validator_hotkey, 102, "finalist_lock", (_sha(77),)
    )
    plan.update(
        finalist_lock=second,
        finalist_lock_hotkey=signer.validator_hotkey,
        finalist_lock_block=104,
    )
    with (
        public_registry(manifest) as registry,
        pytest.raises(ValueError, match="finalist lock is unknown"),
    ):
        prepare_evaluation_anchor(
            chain,
            signer,
            Sr25519Verifier(),
            *inputs,
            evaluator_journal_sha256=_sha(9),
            hf_base_url=registry,
        )
