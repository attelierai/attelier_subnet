"""Testnet intent/readback binding; this never publishes a weight transaction."""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from forge.chain.launch_publication import (
    BootstrapAuthority,
    FinalizedWeights,
    JsonHttpPublicationAdapter,
    LaunchPublicationStore,
    LaunchPublisher,
    QualifiedPublicationContext,
)
from forge.chain.reward_publication import (
    WeightSubmissionReceipt,
    record_submission,
    testnet_weight_intent,
    verify_finalized_replacement,
)
from forge.protocol.consensus_types import FinalizedCompetitionState
from forge.protocol.rewards import Crown


def digest(value: int) -> str:
    return f"{value:064x}"


def intent(**changes):
    values = {
        "network": "test",
        "netuid": 9,
        "certificate_hash": digest(1),
        "runtime_hash": digest(2),
        "registered_uids": {3: "burn"},
        "public_custody": {},
        "verify_public_custody": None,
        "burn_hotkey": "burn",
        "verify_burn_qualification": lambda key: digest(3) if key == "burn" else None,
    }
    values.update(changes)
    state = FinalizedCompetitionState.genesis("wan", "base")
    return testnet_weight_intent(state, **values)


def test_burn_requires_verified_receipt_and_intent_binds_runtime_state_and_submission():
    blocked = intent(verify_burn_qualification=None)
    assert blocked.status == "BLOCKED"
    prepared = intent()
    assert prepared.status == "BURN_REQUIRED"
    assert prepared.network == "test" and prepared.netuid == 9
    assert (
        prepared.certified_state_hash
        == FinalizedCompetitionState.genesis("wan", "base").state_hash()
    )
    receipt = record_submission(prepared, submitted_block=100, submitted_extrinsic_hash=digest(4))
    verify_finalized_replacement(
        prepared,
        receipt,
        network="test",
        netuid=9,
        finalized_block=100,
        finalized_extrinsic_hash=digest(4),
        finalized_runtime_hash=digest(2),
        weights_u16={3: 65535},
        registered_uids={3: "burn"},
    )
    with pytest.raises(ValueError, match="another subnet"):
        verify_finalized_replacement(
            prepared,
            receipt,
            network="test",
            netuid=10,
            finalized_block=100,
            finalized_extrinsic_hash=digest(4),
            finalized_runtime_hash=digest(2),
            weights_u16={3: 65535},
            registered_uids={3: "burn"},
        )
    with pytest.raises(ValueError, match="runtime hash"):
        verify_finalized_replacement(
            prepared,
            receipt,
            network="test",
            netuid=9,
            finalized_block=100,
            finalized_extrinsic_hash=digest(4),
            finalized_runtime_hash=digest(99),
            weights_u16={3: 65535},
            registered_uids={3: "burn"},
        )


def test_bare_custody_or_burn_names_do_not_enable_and_tampered_submission_fails():
    assert intent(network="main").status == "BLOCKED"
    prepared = intent()
    receipt = record_submission(prepared, submitted_block=100, submitted_extrinsic_hash=digest(4))
    with pytest.raises(ValueError, match="extrinsic"):
        verify_finalized_replacement(
            prepared,
            receipt,
            network="test",
            netuid=9,
            finalized_block=101,
            finalized_extrinsic_hash=digest(5),
            finalized_runtime_hash=digest(2),
            weights_u16={3: 65535},
            registered_uids={3: "burn"},
        )


def test_winner_needs_verified_public_custody_receipt_not_a_fingerprint():
    winner = "winner-fingerprint"
    state = FinalizedCompetitionState(
        "wan",
        1,
        digest(9),
        winner,
        (Crown("crown", "winner-hotkey", winner, 10),),
        10,
        "r1",
        digest(8),
    )
    values = {
        "network": "test",
        "netuid": 9,
        "certificate_hash": digest(1),
        "runtime_hash": digest(2),
        "registered_uids": {3: "burn", 4: "winner-hotkey"},
        "burn_hotkey": "burn",
        "verify_burn_qualification": lambda _key: digest(3),
        "public_custody": {winner: digest(4)},
    }
    rejected = testnet_weight_intent(state, verify_public_custody=lambda *_args: False, **values)
    assert rejected.status == "BURN_REQUIRED"
    accepted = testnet_weight_intent(state, verify_public_custody=lambda *_args: True, **values)
    assert accepted.status == "WINNER" and accepted.expected_hotkeys == {4: "winner-hotkey"}


def test_durable_http_publisher_reconciles_and_verifies_qualified_bootstrap(tmp_path):
    """The signer is an HTTP boundary; SQLite contains intent before it sees submit."""

    prepared = intent()
    state = FinalizedCompetitionState.genesis("wan", "base")
    context = QualifiedPublicationContext(
        "bittensor-sdk/testnet-qualified-v1",
        "test",
        9,
        digest(10),
        digest(2),
        0,
        digest(11),
        digest(12),
        state.state_hash(),
        None,
        digest(13),
    )
    bootstrap = BootstrapAuthority(
        "activation-authority", "trusted", digest(10), digest(13), digest(14), 3, "burn"
    )
    journal = LaunchPublicationStore(tmp_path / "publication.sqlite")
    events: list[str] = []

    class Signer(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - HTTP API spelling
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            events.append(self.path)
            if self.path == "/submit":
                with sqlite3.connect(journal.path) as db:
                    assert (
                        db.execute("SELECT status FROM launch_publication_intents").fetchone()[0]
                        == "SUBMITTING"
                    )
                payload = {
                    "intent_hash": prepared.intent_hash(),
                    "network": "test",
                    "netuid": 9,
                    "submitted_block": 100,
                    "submitted_extrinsic_hash": digest(15),
                }
            elif self.path == "/finalized":
                payload = {
                    "finalized_block": 101,
                    "extrinsic_hash": digest(15),
                    "genesis_hash": digest(10),
                    "runtime_hash": digest(2),
                    "mechanism": 0,
                    "last_update_block": 100,
                    "last_step_block": 101,
                    "weights_u16": {"3": 65535},
                    "registered_uids": {"3": "burn"},
                }
            else:
                self.send_response(404)
                self.end_headers()
                return
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Signer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        publisher = LaunchPublisher(
            journal,
            JsonHttpPublicationAdapter(f"http://127.0.0.1:{server.server_port}"),
            verify_publication_authority=lambda *_args: True,
            verify_bootstrap_signature=lambda authority: authority.signature == "trusted",
            verify_fallback_qualification=lambda bound, authority: (
                bound.runtime_hash == digest(2) and authority.target_hotkey == "burn"
            ),
            verify_certified_fallback=lambda *_args: True,
        )
        assert publisher.publish_or_reconcile(context, prepared, bootstrap=bootstrap) == "FINALIZED"
        assert events == ["/submit", "/finalized"]
        assert publisher.publish_or_reconcile(context, prepared, bootstrap=bootstrap) == "FINALIZED"
        assert events == ["/submit", "/finalized"]
    finally:
        server.shutdown()
        thread.join()


def test_no_winner_needs_external_bootstrap_signature_and_qualification(tmp_path):
    prepared = intent()
    context = QualifiedPublicationContext(
        "bittensor-sdk/testnet-qualified-v1",
        "test",
        9,
        digest(10),
        digest(2),
        0,
        digest(11),
        digest(12),
        FinalizedCompetitionState.genesis("wan", "base").state_hash(),
        None,
        digest(13),
    )
    bootstrap = BootstrapAuthority(
        "activation-authority", "untrusted", digest(10), digest(13), digest(14), 3, "burn"
    )
    publisher = LaunchPublisher(
        LaunchPublicationStore(tmp_path / "publication.sqlite"),
        adapter=None,  # authority validation happens before any external signer is used
        verify_publication_authority=lambda *_args: True,
        verify_bootstrap_signature=lambda _authority: False,
        verify_fallback_qualification=lambda *_args: True,
        verify_certified_fallback=lambda *_args: True,
    )
    with pytest.raises(ValueError, match="not trusted"):
        publisher.publish_or_reconcile(context, prepared, bootstrap=bootstrap)


def test_timeout_reconciles_before_any_resubmit(tmp_path):
    prepared = intent()
    context = QualifiedPublicationContext(
        "bittensor-sdk/testnet-qualified-v1",
        "test",
        9,
        digest(10),
        digest(2),
        0,
        digest(11),
        digest(12),
        FinalizedCompetitionState.genesis("wan", "base").state_hash(),
        None,
        digest(13),
    )
    bootstrap = BootstrapAuthority(
        "activation-authority", "trusted", digest(10), digest(13), digest(14), 3, "burn"
    )

    class TimeoutThenReconcile:
        submits = 0
        reconciles = 0

        def submit(self, *_args):
            self.submits += 1
            raise TimeoutError("RPC deadline")

        def reconcile(self, *_args):
            self.reconciles += 1
            return WeightSubmissionReceipt(prepared.intent_hash(), "test", 9, 100, digest(15))

        def finalized_weights(self, *_args):
            return None

    adapter = TimeoutThenReconcile()
    publisher = LaunchPublisher(
        LaunchPublicationStore(tmp_path / "publication.sqlite"),
        adapter,
        verify_publication_authority=lambda *_args: True,
        verify_bootstrap_signature=lambda authority: authority.signature == "trusted",
        verify_fallback_qualification=lambda *_args: True,
        verify_certified_fallback=lambda *_args: True,
    )
    assert publisher.publish_or_reconcile(context, prepared, bootstrap=bootstrap) == "SUBMITTED"
    assert (adapter.submits, adapter.reconciles) == (1, 1)
    # The persisted receipt means restarts reconcile; they never send another set_weights call.
    assert publisher.publish_or_reconcile(context, prepared, bootstrap=bootstrap) == "SUBMITTED"
    assert (adapter.submits, adapter.reconciles) == (1, 2)


def test_live_publisher_rejects_synthetic_authority_before_persisting_or_signing(tmp_path):
    prepared = intent()
    context = QualifiedPublicationContext(
        "bittensor-sdk/testnet-qualified-v1",
        "test",
        9,
        digest(10),
        digest(2),
        0,
        digest(11),
        digest(12),
        FinalizedCompetitionState.genesis("wan", "base").state_hash(),
        None,
        digest(13),
    )
    publisher = LaunchPublisher(
        LaunchPublicationStore(tmp_path / "publication.sqlite"),
        None,
        verify_publication_authority=lambda *_args: False,
        verify_bootstrap_signature=lambda *_args: True,
        verify_fallback_qualification=lambda *_args: True,
        verify_certified_fallback=lambda *_args: True,
    )
    with pytest.raises(ValueError, match="authority is not trusted"):
        publisher.publish_or_reconcile(
            context,
            prepared,
            bootstrap=BootstrapAuthority(
                "a", "fixture", digest(10), digest(13), digest(14), 3, "burn"
            ),
        )


def test_certified_keep_burn_publishes_without_bootstrap_and_pending_readback_stays_submitted(
    tmp_path,
):
    prepared = intent()
    context = QualifiedPublicationContext(
        "bittensor-sdk/testnet-qualified-v1",
        "test",
        9,
        digest(10),
        digest(2),
        0,
        digest(11),
        digest(12),
        FinalizedCompetitionState.genesis("wan", "base").state_hash(),
        digest(1),
        digest(13),
    )

    class CertifiedFallbackAdapter:
        def submit(self, _context, _intent):
            return WeightSubmissionReceipt(prepared.intent_hash(), "test", 9, 100, digest(15))

        def reconcile(self, *_args):
            return None

        def finalized_weights(self, _context, _intent, _receipt):
            return FinalizedWeights(
                finalized_block=100,
                extrinsic_hash=digest(15),
                genesis_hash=digest(10),
                runtime_hash=digest(2),
                mechanism=0,
                last_update_block=99,
                last_step_block=100,
                weights_u16={3: 65535},
                registered_uids={3: "burn"},
            )

    publisher = LaunchPublisher(
        LaunchPublicationStore(tmp_path / "publication.sqlite"),
        CertifiedFallbackAdapter(),
        verify_publication_authority=lambda *_args: True,
        verify_bootstrap_signature=lambda *_args: False,
        verify_fallback_qualification=lambda *_args: False,
        verify_certified_fallback=lambda _context, candidate: (
            candidate.burn_qualification_hash == digest(3)
        ),
    )
    assert publisher.publish_or_reconcile(context, prepared) == "SUBMITTED"
