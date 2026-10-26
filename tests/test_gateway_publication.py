import json
from types import SimpleNamespace

import pytest

from forge.inference.gateway_publication import (
    _advance_revealed_state,
    _authority_path,
    _authority_revealed_master,
    _reject_published_sequence,
    _validate_gateway_weight_receipt,
    publish_gateway_weights,
)


def test_gateway_cr_receipt_stays_in_canonical_authority_and_binds_master(tmp_path):
    authority = tmp_path / "pair-validator-proof"
    authority.mkdir()
    receipt_path = authority / "gateway-light-7.json"
    state_path = authority / "gateway-light-state.json"
    assert _authority_path(authority, receipt_path, "receipt") == authority.resolve()
    with pytest.raises(ValueError, match="gateway_receipt_authority"):
        _authority_path(authority, tmp_path / "other.json", "receipt")

    master = {
        "digest": "d" * 64,
        "body": {"sequence": 7, "weights": [["miner-a", 32768], ["miner-b", 32767]]},
    }
    receipt = {
        "schema": "forge/inference-publication/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "scores": {
            "gatewayMasterDigest": master["digest"],
            "miners": {
                "41": {"uid": 41, "hotkey": "miner-a", "weight": 32768},
                "42": {"uid": 42, "hotkey": "miner-b", "weight": 32767},
            },
        },
        "requestedWeights": {"41": 32768, "42": 32767},
    }
    _validate_gateway_weight_receipt(receipt, master)
    receipt["scores"]["miners"]["41"]["weight"] = 1
    with pytest.raises(ValueError, match="gateway_weight_receipt_plan"):
        _validate_gateway_weight_receipt(receipt, master)

    _advance_revealed_state(state_path, master)
    assert json.loads(state_path.read_text()) == {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "sequence": 7,
        "masterDigest": "d" * 64,
    }
    conflicting = {**master, "digest": "e" * 64}
    with pytest.raises(ValueError, match="gateway_light_replay_or_conflict"):
        _advance_revealed_state(state_path, conflicting)


def test_gateway_light_publish_rejects_unpermitted_publisher_before_cr_intent(
    tmp_path, monkeypatch
):
    authority = tmp_path / "pair-validator-proof"
    authority.mkdir()
    receipt_path = authority / "gateway-light-8.json"
    state_path = authority / "gateway-light-state.json"
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="light-hotkey"))
    master = {"body": {"weights": [["miner-a", 65535]]}}
    plan = {"uids": [7], "weights": [65535], "nextState": {}}
    state = {"committee": {}, "uids": {"0": "light-hotkey", "7": "miner-a"}}
    monkeypatch.setattr("forge.inference.gateway_publication.live_anchor", lambda *_: ({}, {}, {}))
    monkeypatch.setattr("forge.inference.gateway_publication.light_plan", lambda *_: plan)
    monkeypatch.setattr("forge.inference.gateway_publication._ready", lambda _: state)
    with pytest.raises(ValueError, match="finalized permit/stake"):
        publish_gateway_weights(object(), wallet, master, {}, receipt_path, state_path, authority)
    assert not receipt_path.exists()
    assert not (authority / "publication-intents").exists()


def test_gateway_revealed_state_or_authority_receipt_blocks_republication(tmp_path):
    authority = tmp_path / "pair-validator-proof"
    authority.mkdir()
    state = authority / "gateway-light-state.json"
    state.write_text(json.dumps({"sequence": 8, "masterDigest": "d" * 64}))
    with pytest.raises(ValueError, match="gateway_light_already_published"):
        _reject_published_sequence(state, {"sequence": 8, "masterDigest": "d" * 64})

    prior = authority / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema": "forge/inference-publication/v1",
                "status": "REVEALED",
                "scores": {"gatewayMasterDigest": "d" * 64},
            }
        )
    )
    intents = authority / "publication-intents"
    intents.mkdir()
    pointer = intents / "test-578-m1-light-hotkey.json"
    pointer.write_text(json.dumps({"receiptPath": str(prior)}))
    assert _authority_revealed_master(pointer, "d" * 64)
    assert not _authority_revealed_master(pointer, "e" * 64)
