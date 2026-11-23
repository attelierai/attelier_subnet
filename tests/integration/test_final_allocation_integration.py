"""Signed final allocation crosses crypto, SQLite and M0 publication boundaries."""

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from forge.inference.final_allocation import (
    DOMAIN,
    SCHEMA,
    FinalAllocationError,
    FinalAllocationRunner,
    verify_final_allocation,
    verify_primary_echo,
)
from forge.inference.prelaunch import GENESIS, HOUR_MS, canonical, digest


def _signed(
    tmp_path: Path,
    *,
    sequence: int = 1,
    weight: int = 3535,
    private: Path | None = None,
    primary: str = "primary-validator",
) -> tuple[bytes, str, dict, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    private = private or tmp_path / "gateway.key"
    if not private.exists():
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)], check=True
        )
    public_der = subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout
    public = base64.b64encode(public_der[-32:]).decode()
    window = {"startMs": HOUR_MS, "endMs": 2 * HOUR_MS}
    registry = {
        "finalizedBlock": 123,
        "finalizedBlockHash": "a" * 64,
        "finalizedAtMs": 2 * HOUR_MS,
        "subnetOwnerColdkey": "owner-coldkey",
    }
    weights = [
        {"uid": 0, "hotkey": "owner-hotkey", "coldkey": "owner-coldkey", "weight": 65535 - weight},
        {"uid": 7, "hotkey": "miner-hotkey", "coldkey": "miner-coldkey", "weight": weight},
    ]
    body = {
        "schema": "attelier/final-allocation/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "issuedAtMs": 2 * HOUR_MS,
        "expiresAtMs": 3 * HOUR_MS,
        "window": window,
        "sequence": sequence,
        "policyHash": "b" * 64,
        "allocationHash": digest({"registry": registry, "weights": weights}),
        "registry": registry,
        "primaryValidatorHotkey": primary,
        "weights": weights,
    }
    message = tmp_path / "message"
    message.write_bytes(DOMAIN + canonical(body))
    signature = subprocess.run(
        ["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message)],
        check=True,
        capture_output=True,
    ).stdout
    envelope = {
        "schema": SCHEMA,
        "keyId": hashlib.sha256(base64.b64decode(public)).hexdigest(),
        "body": body,
        "digest": digest(body),
        "signature": base64.b64encode(signature).decode(),
    }
    return canonical(envelope), public, body, private


def test_final_allocation_is_signed_and_durable_for_walletless_shadow(tmp_path):
    raw, public, body, _ = _signed(tmp_path)
    runner = FinalAllocationRunner(tmp_path / "state", public, policy_hash="b" * 64)
    accepted = runner.process(raw, now_ms=2 * HOUR_MS + 1)
    assert accepted == body
    status = tmp_path / "state" / "status.json"
    runner.unavailable("light_feed_fetch")
    assert json.loads(status.read_text())["status"] == "unavailable"
    assert runner.process(raw, now_ms=2 * HOUR_MS + 1) == body
    assert json.loads(status.read_text()) == {
        "schema": "attelier/light-status/v1",
        "status": "ready",
        "sequence": body["sequence"],
        "digest": digest(body),
    }
    from forge.inference.final_allocation import onchain_weights

    assert onchain_weights(accepted) == {0: 65535, 7: 3737}


def test_api_primary_dry_run_requires_no_wallet_or_signer(tmp_path, monkeypatch):
    """Exercise the CLI entrypoint with a gateway-signed API allocation."""
    from forge.inference.primary_follow import PRIMARY_HOTKEY

    raw, public, body, _ = _signed(tmp_path, primary=PRIMARY_HOTKEY)
    key_file = tmp_path / "gateway.pub"
    key_file.write_text(public)
    import forge.inference.final_allocation as allocation
    import forge.inference.light as light

    monkeypatch.setattr(allocation.time, "time", lambda: (2 * HOUR_MS + 1) / 1000)
    monkeypatch.setattr(light, "fetch_feed", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(light, "verify_finalized_registry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attelier-validator",
            "--im",
            "api",
            "--role",
            "primary",
            "--dry-run",
            "--once",
            "--state-dir",
            str(tmp_path / "state"),
            "--final-allocation-url",
            "http://127.0.0.1:1/fixture/latest",
            "--final-allocation-public-key-file",
            str(key_file),
        ],
    )
    light.main()
    output = json.loads((tmp_path / "state" / "live-dry-run.json").read_text())
    assert output["allocationShares"] == [[row["uid"], row["weight"]] for row in body["weights"]]


def test_final_allocation_rechecks_signed_identities_at_the_finalized_registry(tmp_path):
    raw, public, _, _ = _signed(tmp_path)
    observed = []
    runner = FinalAllocationRunner(
        tmp_path / "state",
        public,
        registry_verifier=lambda registry: observed.append(registry),
    )
    runner.process(raw, now_ms=2 * HOUR_MS + 1)
    assert observed[0]["mechanism"] == 0
    assert observed[0]["miners"] == [
        {"uid": 0, "hotkey": "owner-hotkey", "coldkey": "owner-coldkey"},
        {"uid": 7, "hotkey": "miner-hotkey", "coldkey": "miner-coldkey"},
    ]


def test_final_allocation_uses_the_installed_sdk_u16_normalization(tmp_path):
    from bittensor.intents.weights import normalize

    from forge.inference.final_allocation import onchain_weights

    raw, public, body, _ = _signed(tmp_path)
    accepted = verify_final_allocation(raw, public, now_ms=2 * HOUR_MS + 1)
    uids, values = normalize(
        [row["uid"] for row in body["weights"]], [row["weight"] for row in body["weights"]]
    )
    assert onchain_weights(accepted) == dict(zip(uids, values, strict=True))


def test_final_allocation_allows_only_a_zero_burn_share(tmp_path):
    raw, public, body, _ = _signed(tmp_path, weight=65535)
    accepted = verify_final_allocation(raw, public, now_ms=2 * HOUR_MS + 1)
    assert accepted["weights"][0]["weight"] == 0
    from forge.inference.final_allocation import onchain_weights

    assert onchain_weights(accepted) == {0: 0, 7: 65535}


def test_final_allocation_rejects_tamper_replay_and_primary_mismatch(tmp_path):
    raw, public, body, private = _signed(tmp_path)
    runner = FinalAllocationRunner(tmp_path / "state", public)
    runner.process(raw, now_ms=2 * HOUR_MS + 1)
    changed, _, _, _ = _signed(tmp_path / "changed", sequence=1, weight=3000, private=private)
    with pytest.raises(FinalAllocationError, match="final_allocation_replay_or_rollback"):
        runner.process(changed, now_ms=2 * HOUR_MS + 1)
    overlapping, _, _, _ = _signed(tmp_path / "overlap", sequence=2, private=private)
    with pytest.raises(FinalAllocationError, match="final_allocation_window_overlap"):
        runner.process(overlapping, now_ms=2 * HOUR_MS + 1)
    with pytest.raises(FinalAllocationError, match="final_allocation_primary_mismatch"):
        verify_primary_echo(
            body,
            {
                "network": "finney",
                "netuid": 126,
                "mechanism": 0,
                "hotkey": "primary-validator",
                "observedBlock": 130,
                "observedBlockHash": "c" * 64,
                "lastUpdate": 130,
                "weights": [[0, 65535], [7, 3000]],
            },
        )
    verify_primary_echo(
        body,
        {
            "network": "finney",
            "netuid": 126,
            "mechanism": 0,
            "hotkey": "primary-validator",
            "observedBlock": 130,
            "observedBlockHash": "c" * 64,
            "lastUpdate": 130,
            "weights": [[0, 65535], [7, 3737]],
        },
    )
    with pytest.raises(FinalAllocationError, match="final_allocation_primary_observation"):
        verify_primary_echo(
            body,
            {
                "network": "finney",
                "netuid": 126,
                "mechanism": 0,
                "hotkey": "primary-validator",
                "observedBlock": 130,
                "observedBlockHash": "c" * 64,
                "lastUpdate": 122,
                "weights": [[0, 65535], [7, 3737]],
            },
        )
    with pytest.raises(FinalAllocationError, match="final_allocation_hash"):
        verify_final_allocation(raw.replace(b"3535", b"3534"), public, now_ms=2 * HOUR_MS + 1)
    # A hotkey maps to at most one UID on-chain; accepting two rows would make
    # the signed vector ambiguous before the live identity guard can run.
    duplicate_hotkey = json.loads(raw)
    duplicate_hotkey["body"]["weights"][1]["hotkey"] = "owner-hotkey"
    duplicate_hotkey["body"]["allocationHash"] = digest(
        {
            "registry": duplicate_hotkey["body"]["registry"],
            "weights": duplicate_hotkey["body"]["weights"],
        }
    )
    with pytest.raises(FinalAllocationError, match="final_allocation_weights"):
        verify_final_allocation(canonical(duplicate_hotkey), public, now_ms=2 * HOUR_MS + 1)


def test_walletless_cli_waits_for_fresh_feed_without_legacy_continuity(tmp_path, monkeypatch):
    from forge.inference import light

    _, public, _, _ = _signed(tmp_path / "fixture")
    pub = tmp_path / "gateway.pub"
    pub.write_text(public)
    state = tmp_path / "state"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validator",
            "--dry-run",
            "--once",
            "--state-dir",
            str(state),
            "--final-allocation-url",
            "https://example.invalid/v1/subnet/126/final-allocation/latest",
            "--final-allocation-public-key-file",
            str(pub),
        ],
    )

    def missing(*args, **kwargs):
        raise light.LightError("light_feed_fetch")

    monkeypatch.setattr(light, "fetch_feed", missing)
    with pytest.raises(SystemExit) as exit_status:
        light.main()
    assert exit_status.value.code == 1
    assert json.loads((state / "status.json").read_text())["reason"] == "light_feed_fetch"
    assert not (state / "live-dry-run.json").exists()


def test_light_cli_reads_finalized_raw_primary_weights_without_gateway_or_wallet(
    tmp_path, monkeypatch
):
    import time
    from types import SimpleNamespace

    from bittensor import storage

    import forge.chain
    from forge.inference import light
    from forge.inference.primary_follow import BURN_OWNER_COLDKEY, PRIMARY_HOTKEY

    chain_data = {
        "LastUpdate": [0, 798],
        "Weights": [(0, 65535), (7, 3737)],
    }
    owners = {0: ("owner", BURN_OWNER_COLDKEY), 1: (PRIMARY_HOTKEY, "validator"), 7: ("m", "c")}

    class At:
        def query(self, item, args=None):
            if item is storage.Timestamp.Now:
                return int(time.time() * 1000) - 12000
            if item is storage.SubtensorModule.Weights:
                assert args == [126, 1]
                return chain_data["Weights"]
            assert args == [126]
            return {
                storage.SubtensorModule.WeightsSetRateLimit: 100,
                storage.SubtensorModule.MechanismCountCurrent: 1,
                storage.SubtensorModule.ValidatorPermit: [False, True],
                storage.SubtensorModule.ActivityCutoff: 500,
                storage.SubtensorModule.LastUpdate: chain_data["LastUpdate"],
                storage.SubtensorModule.SubnetOwner: BURN_OWNER_COLDKEY,
            }[item]

    raw = SimpleNamespace(get_block_hash=lambda height: GENESIS)
    chain = SimpleNamespace(
        network="finney",
        netuid=126,
        subtensor=SimpleNamespace(
            at=lambda height: At(),
            _call=lambda result: result,
            _client=SimpleNamespace(_substrate=SimpleNamespace(raw=raw)),
        ),
        finalized_height=lambda: 804,
        finalized_block_hash=lambda height: "0x" + "a" * 64,
        owner_snapshot=lambda height: owners,
        close=lambda: None,
    )
    monkeypatch.setattr(forge.chain, "BittensorChain", lambda *args, **kw: chain)
    import bittensor

    def unexpected(*args, **kwargs):
        pytest.fail("a light dry-run must read the chain without a wallet or gateway")

    monkeypatch.setattr(bittensor, "Wallet", unexpected)
    monkeypatch.setattr(light, "fetch_feed", unexpected)
    for name, updated, succeeds in (("fresh", 798, True), ("stale", 100, False)):
        chain_data["LastUpdate"] = [0, updated]
        state = tmp_path / name
        monkeypatch.setattr(
            sys,
            "argv",
            ["validator", "--role", "light", "--dry-run", "--once", "--state-dir", str(state)],
        )
        if succeeds:
            light.main()
            assert json.loads((state / "live-dry-run.json").read_bytes())["relativeWeights"] == [
                [0, 65535],
                [7, 3737],
            ]
        else:
            with pytest.raises(SystemExit):
                light.main()
            assert (
                json.loads((state / "status.json").read_bytes())["reason"]
                == "light_primary_row_stale"
            )
            assert not (state / "live-dry-run.json").exists()
