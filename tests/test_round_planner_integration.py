import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from forge.chain import OnChainCommitment, encode_prompt_pack_commit
from forge.models.artifact_manifest import ArtifactReference
from forge.protocol.contract import ChallengeContract
from forge.protocol.prompt_pack import build_pack, pack_context, seal_pack, verified_plan_prompts
from forge.validator.round_planner import build_public_round_plan
from forge.validator.submission import V3SubmissionClaim, encode_v3_submission_commitment

V2_REPO = "public/v2-adapter"
V3_REPO = "public/v3-adapter"
V2_REVISION = "a" * 40
V3_REVISION = "b" * 40
WEIGHTS = b"validator derives this from immutable registry metadata"
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()
HONEST_WEIGHTS = b"independently trained public transformer"
HONEST_WEIGHTS_SHA256 = hashlib.sha256(HONEST_WEIGHTS).hexdigest()


@dataclass
class ReplayChain:
    """A recorded finalized-chain port, not a simulated planner dependency."""

    network: str
    netuid: int
    finalized: int
    block_hashes: dict[int, str]
    uids: dict[int, str]
    committee: dict[str, int]
    closed_commitments: list[OnChainCommitment]
    calls: list[tuple[str, int]] = field(default_factory=list)
    historical: dict[tuple[str, int], OnChainCommitment] = field(default_factory=dict)

    def finalized_height(self) -> int:
        self.calls.append(("finalized_height", self.finalized))
        return self.finalized

    def finalized_block_hash(self, height: int) -> str:
        self.calls.append(("block_hash", height))
        if height > self.finalized:
            raise ValueError("unfinalized replay block")
        return self.block_hashes[height]

    def uid_snapshot(self, height: int) -> dict[int, str]:
        self.calls.append(("uids", height))
        return dict(self.uids)

    def committee_snapshot(self, height: int) -> dict[str, int]:
        self.calls.append(("committee", height))
        return dict(self.committee)

    def commitments(self, hotkeys: list[str], height: int) -> list[OnChainCommitment]:
        self.calls.append(("commitments", height))
        assert set(hotkeys) == set(self.uids.values())
        return list(self.closed_commitments)

    def commitment(self, hotkey: str, height: int) -> OnChainCommitment | None:
        return self.historical.get((hotkey, height))


@contextmanager
def public_hf_registry(models=None):
    declaration = {
        "schema": "forge-manifest/v1",
        "role": "lora",
        "files": [
            {
                "path": "adapter/model.safetensors",
                "sha256": WEIGHTS_SHA256,
                "bytes": len(WEIGHTS),
            }
        ],
        "license": "apache-2.0",
        "generation_config": {"steps": 16, "guidance_scale": 5.0},
        "base_reference": {
            "backend": "hf",
            "repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
        },
    }
    models = models or {V2_REPO: V2_REVISION, V3_REPO: V3_REVISION}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            for repo, revision in models.items():
                if self.path.startswith(f"/api/models/{repo}/revision/{revision}"):
                    weight_path = (
                        "transformer/model.safetensors"
                        if repo.startswith("public/full-")
                        else "adapter/model.safetensors"
                    )
                    blob = HONEST_WEIGHTS if repo == "public/full-honest" else WEIGHTS
                    blob_sha256 = hashlib.sha256(blob).hexdigest()
                    siblings = [
                        {
                            "rfilename": weight_path,
                            "size": len(blob),
                            "lfs": {"sha256": blob_sha256, "size": len(blob)},
                        }
                    ]
                    if repo == V2_REPO:
                        siblings.insert(0, {"rfilename": "forge-manifest.json"})
                    body = json.dumps(
                        {
                            "private": False,
                            "gated": False,
                            "disabled": False,
                            "sha": revision,
                            "cardData": {"license": "apache-2.0"},
                            "siblings": siblings,
                        }
                    ).encode()
                    break
            else:
                if self.path == f"/{V2_REPO}/resolve/{V2_REVISION}/forge-manifest.json":
                    body = json.dumps(declaration).encode()
                else:
                    self.send_error(404)
                    return
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


def _entries() -> list[dict]:
    return [
        {
            "stage": "screen",
            "text": f"A distinct {subject} moves through a unique scene.",
            "properties": {
                "subject_class": subject,
                "camera_mode": "static",
                "style_family": "realistic",
            },
            "rubric": [f"The {subject} stays identifiable."],
            "source_group": f"screen-{subject}",
        }
        for subject in ("rigid_object", "human", "animal", "scenery")
    ]


def _contract() -> ChallengeContract:
    return ChallengeContract(screen_prompts=4, screen_seeds=1, max_candidates=4)


def test_planner_replays_closed_v2_v3_and_sealed_pack_admissions():
    contract = _contract()
    pack = build_pack(
        pack_context("test", 578, "sealed-r100", contract.contract_hash(), 100),
        "c" * 64,
        _entries(),
    )
    chain = ReplayChain(
        network="test",
        netuid=578,
        finalized=110,
        block_hashes={100: "close-finalized", 101: "entropy-finalized"},
        uids={0: "miner-v2", 1: "miner-v3", 2: "validator-pack"},
        committee={"validator-pack": 1_000},
        closed_commitments=[
            OnChainCommitment("miner-v2", 99, "model", ("hf", V2_REPO, V2_REVISION)),
            OnChainCommitment(
                "miner-v3",
                100,
                "model_v3",
                ("hf", V3_REPO, V3_REVISION, "lora", "8", "5"),
            ),
            OnChainCommitment("validator-pack", 99, "pack", (seal_pack(pack),)),
        ],
    )

    with public_hf_registry() as registry:
        plan = build_public_round_plan(
            chain,
            contract,
            close_block=100,
            entropy_block=101,
            prompt_pack=pack,
            pack_hotkey="validator-pack",
            finalist_lock_block=100,
            incumbent={
                "manifest_hash": "1" * 64,
                "certified_state_hash": "2" * 64,
                "certified_activation_block": 90,
            },
            candidate_hotkey="miner-v3",
            hf_base_url=registry,
        )

    assert {
        ("block_hash", 100),
        ("block_hash", 101),
        ("uids", 100),
        ("committee", 100),
        ("commitments", 100),
    } <= set(chain.calls)
    assert plan["geometry"] == {"height": 480, "width": 832, "frames": 81, "fps": 16}
    assert plan["campaign_id"] == plan["round_id"] == "sealed-r100"
    assert plan["pack_commitment"] == {
        "hash": seal_pack(pack),
        "block": 99,
        "hotkey": "validator-pack",
    }
    assert plan["pack_reveal_block"] == 110
    assert {item["commitment_kind"] for item in plan["admitted"]} == {"model", "model_v3"}
    assert all(
        item["admission_status"] == "METADATA_VALIDATED_RUNTIME_PREFLIGHT_PENDING"
        for item in plan["admitted"]
    )
    expected_v3_hash = next(
        item["manifest_hash"] for item in plan["admitted"] if item["hotkey"] == "miner-v3"
    )
    assert plan["candidate"] == {
        "manifest_hash": expected_v3_hash,
        "hotkey": "miner-v3",
        "admission_block": 100,
    }
    assert plan["finalist_lock_block"] == 100
    assert verified_plan_prompts(plan, contract) == plan["prompts"]


def test_planner_requires_closed_committee_pack_and_selected_candidate():
    contract = _contract()
    pack = build_pack(
        pack_context("test", 578, "sealed-r100", contract.contract_hash(), 100),
        "d" * 64,
        _entries(),
    )
    chain = ReplayChain(
        network="test",
        netuid=578,
        finalized=101,
        block_hashes={100: "close-finalized", 101: "entropy-finalized"},
        uids={0: "miner"},
        committee={"other-validator": 1_000},
        closed_commitments=[
            OnChainCommitment("miner", 100, "pack", (seal_pack(pack),)),
        ],
    )
    with pytest.raises(ValueError, match="committee"):
        build_public_round_plan(
            chain,
            contract,
            100,
            101,
            prompt_pack=pack,
            pack_hotkey="miner",
        )

    legacy_chain = ReplayChain(
        network="test",
        netuid=578,
        finalized=101,
        block_hashes={100: "close-finalized", 101: "entropy-finalized"},
        uids={0: "miner"},
        committee={},
        closed_commitments=[],
    )
    plan = build_public_round_plan(legacy_chain, contract, 100, 101)
    assert "candidate" not in plan and "finalist_lock_block" not in plan
    with pytest.raises(ValueError, match="candidate hotkey"):
        build_public_round_plan(legacy_chain, contract, 100, 101, candidate_hotkey="miner")


def test_pack_commitment_encoding_matches_the_replayed_chain_fixture():
    digest = "e" * 64
    assert encode_prompt_pack_commit(digest) == f"forge:v3:pack:{digest}"
    claim = V3SubmissionClaim(
        ArtifactReference("hf", V3_REPO, V3_REVISION), "lora", {"steps": 8, "guidance_scale": 5.0}
    )
    assert encode_v3_submission_commitment(claim).endswith(":l:8:5")


def test_early_full_transformer_blob_dedup_leaves_room_before_the_admission_cap():
    contract = ChallengeContract(screen_prompts=4, screen_seeds=1, max_candidates=16)
    fork_revisions = {f"public/full-fork-{index:02}": f"{index:040x}" for index in range(16)}
    honest_repo, honest_revision = "public/full-honest", "f" * 40
    models = {**fork_revisions, honest_repo: honest_revision}
    commitments = [
        OnChainCommitment(
            f"fork-{index:02}",
            98 if index == 0 else 99,
            "model_v3",
            ("hf", repo, revision, "full_transformer", "16", "5"),
        )
        for index, (repo, revision) in enumerate(fork_revisions.items())
    ]
    commitments.append(
        OnChainCommitment(
            "honest",
            100,
            "model_v3",
            ("hf", honest_repo, honest_revision, "full_transformer", "16", "5"),
        )
    )
    chain = ReplayChain(
        network="test",
        netuid=578,
        finalized=101,
        block_hashes={100: "close-finalized", 101: "entropy-finalized"},
        uids={index: item.hotkey for index, item in enumerate(commitments)},
        committee={},
        closed_commitments=commitments,
    )

    with public_hf_registry(models) as registry:
        plan = build_public_round_plan(
            chain, contract, close_block=100, entropy_block=101, hf_base_url=registry
        )

    assert len(plan["admitted"]) == 2
    assert {item["hotkey"] for item in plan["admitted"]} == {"fork-00", "honest"}
    assert len(plan["early_full_transformer_duplicates"]) == 15
    assert plan["admission_counts"] == {
        "metadata_validated": 17,
        "pending_runtime_preflight": 2,
        "early_full_transformer_duplicates": 15,
        "overflow_before_runtime_preflight": 0,
    }
    assert plan["admission_overflow"] == []


def test_early_full_transformer_identity_keeps_same_weights_with_a_distinct_scheduler():
    """A sparse legacy recipe resolves to Euler, so explicit UniPC is distinct."""

    contract = ChallengeContract(screen_prompts=4, screen_seeds=1, max_candidates=4)
    legacy_repo, unipc_repo = "public/full-legacy", "public/full-unipc"
    chain = ReplayChain(
        network="test",
        netuid=578,
        finalized=101,
        block_hashes={100: "close-finalized", 101: "entropy-finalized"},
        uids={0: "legacy", 1: "unipc"},
        committee={},
        closed_commitments=[
            OnChainCommitment(
                "legacy",
                99,
                "model_v3",
                ("hf", legacy_repo, "1" * 40, "full_transformer", "16", "5"),
            ),
            OnChainCommitment(
                "unipc",
                100,
                "model_v4",
                ("hf", unipc_repo, "2" * 40, "full_transformer", "16", "5", "original_unipc"),
            ),
        ],
    )

    with public_hf_registry({legacy_repo: "1" * 40, unipc_repo: "2" * 40}) as registry:
        plan = build_public_round_plan(
            chain, contract, close_block=100, entropy_block=101, hf_base_url=registry
        )

    assert {item["hotkey"] for item in plan["admitted"]} == {"legacy", "unipc"}
    assert plan["early_full_transformer_duplicates"] == []


def test_confirmation_requires_matching_finalized_lock_not_a_local_declaration():
    from forge.protocol.finalist_lock import build_finalist_lock, finalist_lock_hash
    from forge.validator.submission import resolve_v3_submission

    contract = _contract()
    close_hash, entropy = "0x" + "a" * 64, "0x" + "b" * 64
    candidate_commitment = OnChainCommitment(
        "miner", 99, "model_v3", ("hf", V3_REPO, V3_REVISION, "lora", "8", "5")
    )
    chain = ReplayChain(
        "test",
        578,
        106,
        {100: close_hash, 106: entropy},
        {0: "miner", 1: "locker"},
        {"locker": 1},
        [candidate_commitment],
    )
    incumbent = {
        "manifest_hash": "1" * 64,
        "certified_state_hash": "2" * 64,
        "certified_activation_block": 90,
    }
    with public_hf_registry() as registry:
        manifest = resolve_v3_submission(
            V3SubmissionClaim(
                ArtifactReference("hf", V3_REPO, V3_REVISION),
                "lora",
                {"steps": 8, "guidance_scale": 5.0},
            ),
            contract,
            hf_base_url=registry,
        )
        lock = build_finalist_lock(
            {
                "schema": "forge/finalist-lock/v1",
                "network": "test",
                "netuid": 578,
                "round_id": "r100",
                "close_block": 100,
                "close_block_hash": close_hash,
                "contract_hash": contract.contract_hash(),
                "candidate_manifest_hash": manifest.fingerprint(),
                "candidate_hotkey": "miner",
                "candidate_admission_block": 99,
                "incumbent_state_hash": incumbent["certified_state_hash"],
            }
        )
        with pytest.raises(ValueError, match="finalized finalist lock"):
            build_public_round_plan(
                chain,
                contract,
                100,
                106,
                stage="confirmation",
                finalist_lock_block=105,
                incumbent=incumbent,
                candidate_hotkey="miner",
                calibration_plan_hash="c" * 64,
                hf_base_url=registry,
            )
        chain.historical[("locker", 105)] = OnChainCommitment(
            "locker", 105, "finalist_lock", (finalist_lock_hash(lock),)
        )
        plan = build_public_round_plan(
            chain,
            contract,
            100,
            106,
            stage="confirmation",
            finalist_lock_block=105,
            incumbent=incumbent,
            candidate_hotkey="miner",
            finalist_lock=lock,
            finalist_lock_hotkey="locker",
            calibration_plan_hash="c" * 64,
            hf_base_url=registry,
        )
        assert plan["finalist_lock"] == lock
        assert plan["calibration_plan_hash"] == "c" * 64
        swapped = dict(lock, candidate_manifest_hash="3" * 64)
        chain.historical[("locker", 105)] = OnChainCommitment(
            "locker", 105, "finalist_lock", (finalist_lock_hash(swapped),)
        )
        with pytest.raises(ValueError, match="finalized finalist lock"):
            build_public_round_plan(
                chain,
                contract,
                100,
                106,
                stage="confirmation",
                finalist_lock_block=105,
                incumbent=incumbent,
                candidate_hotkey="miner",
                finalist_lock=swapped,
                finalist_lock_hotkey="locker",
                calibration_plan_hash="c" * 64,
                hf_base_url=registry,
            )
        chain.historical[("locker", 105)] = OnChainCommitment(
            "locker", 104, "finalist_lock", (finalist_lock_hash(lock),)
        )
        with pytest.raises(ValueError, match="FIRST_VALID_LOCK_AFTER_C"):
            build_public_round_plan(
                chain,
                contract,
                100,
                106,
                stage="confirmation",
                finalist_lock_block=105,
                incumbent=incumbent,
                candidate_hotkey="miner",
                finalist_lock=lock,
                finalist_lock_hotkey="locker",
                calibration_plan_hash="c" * 64,
                hf_base_url=registry,
            )
