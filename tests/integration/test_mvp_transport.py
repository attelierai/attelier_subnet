"""Real sr25519 messages across separate SQLite stores; no model-quality claims."""

import copy
from types import SimpleNamespace

import pytest

from forge.chain import Sr25519Signer, Sr25519Verifier
from forge.protocol.consensus_types import RoundPlan
from forge.protocol.store import Store
from forge.protocol.transport import RoundMailbox, sign_envelope


def test_round_exchange_restart_tamper_and_deadline(tmp_path):
    from bittensor.keyfiles import Keypair

    key = Keypair.create_from_uri("//Alice")
    signer = Sr25519Signer(SimpleNamespace(hotkey=key))
    plan = RoundPlan(
        "test",
        578,
        "round",
        "a" * 64,
        "wan",
        "b" * 64,
        "c" * 64,
        "miner",
        "d" * 64,
        {},
        {signer.validator_hotkey: 1},
        10,
        20,
        30,
        40,
    )
    verifier = Sr25519Verifier()
    message = sign_envelope(plan, signer, "commit", {"commitment": "e" * 64})
    for name in ("one", "two"):
        with Store(tmp_path / f"{name}.sqlite") as store:
            store.create_round(plan.round_id, plan.challenge_id, plan.plan_hash(), plan.to_dict())
            mailbox = RoundMailbox(store, verifier)
            assert mailbox.receive(plan, message, finalized_block=15)
            assert mailbox.export(plan) == [message]
    with Store(tmp_path / "two.sqlite") as store:
        mailbox = RoundMailbox(store, verifier)
        assert not mailbox.receive(plan, message, finalized_block=16)
        changed = copy.deepcopy(message)
        changed["payload"]["commitment"] = "f" * 64
        with pytest.raises(ValueError, match="signature"):
            mailbox.receive(plan, changed, finalized_block=16)
        conflict = sign_envelope(plan, signer, "commit", {"commitment": "f" * 64})
        with pytest.raises(ValueError, match="conflicting"):
            mailbox.receive(plan, conflict, finalized_block=16)
        reveal = sign_envelope(plan, signer, "reveal", {"salt": "abc"})
        with pytest.raises(ValueError, match="before"):
            mailbox.receive(plan, reveal, finalized_block=20)
        with pytest.raises(ValueError, match="deadline"):
            mailbox.receive(plan, reveal, finalized_block=31)
        receipt = mailbox.expire(plan, finalized_block=31, finalized_hash="1" * 64)
        assert not receipt["winner_changed"]
        assert store.count("certificates") == store.count("crowns") == 0
        store.create_round("next", "wan", "2" * 64, {})
    with Store(tmp_path / "two.sqlite") as store:
        mailbox = RoundMailbox(store, verifier)
        assert mailbox.expire(plan, finalized_block=32, finalized_hash="3" * 64) == receipt
        with pytest.raises(ValueError, match="closed"):
            mailbox.receive(plan, reveal, finalized_block=25)
