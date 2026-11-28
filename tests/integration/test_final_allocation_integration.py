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
from forge.inference.prelaunch import GENESIS, HOUR_MS, TESTNET_GENESIS, canonical, digest


@pytest.mark.parametrize("mode", [[], ["--dry-run"], ["--burn-only", "--dry-run"]])
def test_embedded_observer_cannot_enable_reward_publication(tmp_path, mode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge.inference.light",
            *mode,
            "--observe-api-allocation",
            "https://example.invalid/allocation",
            "--observe-api-public-key-file",
            str(tmp_path / "unused.pub"),
            "--state-dir",
            str(tmp_path / "state"),
            "--once",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires" in result.stderr
    assert not (tmp_path / "state").exists()


def test_embedded_observer_recovers_without_blocking_host_or_loading_wallet(tmp_path):
    """Real HTTP, signature, SDK boundary, journal and daemon lifecycle."""
    import http.server
    import os
    import sqlite3
    import threading
    import time

    raw, public, body, _ = _signed(
        tmp_path,
        window_end=int(time.time() * 1000) // HOUR_MS * HOUR_MS,
    )
    (tmp_path / "gateway.pub").write_text(public)
    # Only the external SDK is replaced. Wallet/signing access is a hard failure.
    (tmp_path / "bittensor.py").write_text(f"""
from types import SimpleNamespace
def Wallet(*args, **kwargs):
    raise AssertionError("observer attempted wallet access")
class Raw:
    def get_chain_finalised_head(self): return "finalized"
    def get_block_number(self, value): return 123
    def get_block_hash(self, height): return {GENESIS!r}
class At:
    def query(self, item, args):
        return {{"SubnetOwner": "owner-coldkey", "MechanismCountCurrent": 1}}[item]
    def read(self, name, **kwargs):
        return {{"hotkeys": ["owner-hotkey"] + ["unused"] * 6 + ["miner-hotkey"],
                 "coldkeys": ["owner-coldkey"] + ["unused"] * 6 + ["miner-coldkey"]}}
class Subtensor:
    def __init__(self, **kwargs):
        self.block = 123
        self._client = SimpleNamespace(_substrate=SimpleNamespace(raw=Raw()))
    def _call(self, value): return value
    def at(self, height): return At()
    def read(self, name, **kwargs): return {{"hash": "0x" + "a" * 64}}
    def close(self): pass
storage = SimpleNamespace(SubtensorModule=SimpleNamespace(
    SubnetOwner="SubnetOwner", MechanismCountCurrent="MechanismCountCurrent"))
""")
    blocked, release = threading.Event(), threading.Event()
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if len(requests) == 2:
                blocked.set()
                release.wait(10)
            payload = b"{}" if len(requests) == 1 else raw
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    script = """
import json, time
from pathlib import Path
from forge.inference.allocation_observer import start_allocation_observer
root = Path.cwd()
worker, stop = start_allocation_observer(root / "state", URL, root / "gateway.pub",
    network="wss://fixture.invalid", interval=0.1, allow_http_fixture=True)
deadline = time.monotonic() + 15
try:
    while time.monotonic() < deadline:
        (root / "host-heartbeat").write_text(str(time.monotonic_ns()))
        if (root / "state/live-dry-run.json").exists():
            break
        time.sleep(0.05)
    else:
        raise AssertionError("observer did not recover")
finally:
    stop.set()
    worker.join(2)
assert not worker.is_alive()
""".replace("URL", repr(f"http://127.0.0.1:{server.server_port}/fixture"))
    source = str(Path(__file__).parents[2] / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), source])},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert blocked.wait(8)
        status = json.loads((tmp_path / "state/observer.json").read_text())
        assert status["status"] == "unavailable" and not status["publishable"]
        heartbeat = (tmp_path / "host-heartbeat").read_text()
        time.sleep(0.2)
        assert (tmp_path / "host-heartbeat").read_text() != heartbeat
        release.set()
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stdout + stderr
        output = json.loads((tmp_path / "state/live-dry-run.json").read_text())
        assert output["bodyDigest"] == digest(body)
        assert output["relativeWeights"] == [[0, 65535], [7, 3737]]
        assert json.loads((tmp_path / "state/observer.json").read_text())["status"] == "ready"
        with sqlite3.connect(tmp_path / "state/light.sqlite3") as db:
            assert db.execute("SELECT COUNT(*) FROM final_allocation").fetchone() == (1,)
            assert not db.execute(
                "SELECT name FROM sqlite_master WHERE name='live_intent'"
            ).fetchall()
    finally:
        release.set()
        if process.poll() is None:
            process.kill()
            process.wait()
        server.shutdown()
        server.server_close()
        server_thread.join()


def _signed(
    tmp_path: Path,
    *,
    sequence: int = 1,
    weight: int = 3535,
    private: Path | None = None,
    primary: str = "primary-validator",
    window_end: int = 2 * HOUR_MS,
    network: str = "finney",
    netuid: int = 126,
    genesis: str = GENESIS,
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
    window = {"startMs": window_end - HOUR_MS, "endMs": window_end}
    registry = {
        "finalizedBlock": 123,
        "finalizedBlockHash": "a" * 64,
        "finalizedAtMs": window_end,
        "subnetOwnerColdkey": "owner-coldkey",
    }
    weights = [
        {"uid": 0, "hotkey": "owner-hotkey", "coldkey": "owner-coldkey", "weight": 65535 - weight},
        {"uid": 7, "hotkey": "miner-hotkey", "coldkey": "miner-coldkey", "weight": weight},
    ]
    body = {
        "schema": "attelier/final-allocation/v1",
        "network": network,
        "genesis": genesis,
        "netuid": netuid,
        "mechanism": 0,
        "issuedAtMs": window_end,
        "expiresAtMs": window_end + HOUR_MS,
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


def test_community_validator_role_publishes_directly_with_env_defaults_and_non_primary_hotkey(
    tmp_path, monkeypatch
):
    """Any registered SN126 validator can verify and publish the signed API allocation directly."""
    from forge.inference.final_allocation import onchain_weights
    from forge.inference.primary_follow import PRIMARY_HOTKEY

    raw, public, body, _ = _signed(tmp_path, primary=PRIMARY_HOTKEY)
    key_file = tmp_path / "gateway.pub"
    key_file.write_text(public)
    import forge.chain as forge_chain
    import forge.inference.final_allocation as allocation
    import forge.inference.light as light
    import forge.inference.light_publication as light_pub

    published: list[tuple[dict, dict]] = []

    class FakeHotkey:
        ss58_address = "5CommunityValidatorHotkey11111111111111111111111"

    class FakeWallet:
        def __init__(self, **_kwargs):
            self.hotkey = FakeHotkey()

    class FakeChain:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class FakePublisher:
        def __init__(self, _journal, sdk_pub):
            self.publisher = sdk_pub

        def process(self, payload, artifact, forced_weights=None):
            published.append((payload, forced_weights))
            return {"status": "REVEALED"}

    monkeypatch.setattr(allocation.time, "time", lambda: (2 * HOUR_MS + 1) / 1000)
    monkeypatch.setattr(light, "fetch_feed", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(light, "verify_finalized_registry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(forge_chain, "BittensorChain", FakeChain)
    monkeypatch.setattr(light_pub, "LivePublisher", FakePublisher)
    import bittensor as bt

    monkeypatch.setattr(bt, "Wallet", FakeWallet)
    monkeypatch.setenv("WALLET_NAME", "community")
    monkeypatch.setenv("WALLET_HOTKEY", "val1")
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("VALIDATOR_ROLE", "validator")
    monkeypatch.setenv("VALIDATOR_IM", "api")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attelier-validator",
            "--once",
            "--final-allocation-url",
            "http://127.0.0.1:1/fixture/latest",
            "--final-allocation-public-key-file",
            str(key_file),
        ],
    )
    light.main()
    assert len(published) == 1
    assert published[0][1] == onchain_weights(body)


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


def test_testnet_final_allocation_verifies_explicitly_and_fails_closed_on_finney(tmp_path):
    raw, public, _, _ = _signed(
        tmp_path,
        network="test",
        netuid=578,
        genesis=TESTNET_GENESIS,
    )
    with pytest.raises(FinalAllocationError, match="final_allocation_namespace"):
        verify_final_allocation(raw, public, now_ms=2 * HOUR_MS)

    verified = verify_final_allocation(
        raw,
        public,
        now_ms=2 * HOUR_MS,
        expected_network="test",
        expected_netuid=578,
        expected_genesis=TESTNET_GENESIS,
    )
    assert verified["network"] == "test"
    assert verified["netuid"] == 578
    assert verified["genesis"] == TESTNET_GENESIS
