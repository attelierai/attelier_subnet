import base64
import hashlib
import http.server
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from forge.chain import BittensorChain
from forge.inference.light import (
    DOMAIN,
    LightError,
    LightRunner,
    fetch_feed,
    verify_feed,
)
from forge.inference.light_publication import (
    LiveIntent,
    LiveJournal,
    LivePublisher,
    PublicationError,
    SdkM0Publisher,
    _chain_bytes,
    relative_vector,
)
from forge.inference.light_registry import RegistryError, verify_finalized_registry
from forge.inference.prelaunch import canonical, digest

GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
OWNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MINER_A = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
MINER_B = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"
DAY_MS = 86_400_000


def _inputs():
    gateway = {
        "minSuccessBps": 9500,
        "maxTtvMs": 180000,
        "serviceProfile": {
            "id": "h3-base-5s-480p-20-v1",
            "hash": "a" * 64,
            "durationMs": 5000,
            "resolution": "480p",
        },
    }
    workers = [
        {"minerHotkey": MINER_A, "workerId": "fixture-a", "trustMode": "operator_whitelist"},
        {"minerHotkey": MINER_B, "workerId": "fixture-b", "trustMode": "operator_whitelist"},
    ]
    policy = {
        "schema": "attelier/prelaunch-policy/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "gatewayPolicy": gateway,
        "workers": workers,
        "simulation": {
            "serviceBudgetBps": 1000,
            "targetVideosPerDay": 240,
            "nominalMinerEmissionRaoPerDay": "2952000000000",
            "allocationMode": "fixed",
        },
    }
    registry = {
        "schema": "attelier/prelaunch-registry/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "finalizedBlock": 1,
        "finalizedBlockHash": "b" * 64,
        "subnetOwnerColdkey": OWNER,
        "miners": [
            {"uid": 3, "hotkey": MINER_A, "coldkey": MINER_A},
            {"uid": 4, "hotkey": MINER_B, "coldkey": MINER_B},
        ],
    }
    rows = []
    for key, worker in ((MINER_A, "fixture-a"), (MINER_B, "fixture-b")):
        rows.append(
            {
                "minerHotkey": key,
                "workerId": worker,
                "trustMode": "operator_whitelist",
                "profileId": gateway["serviceProfile"]["id"],
                "profileHash": "a" * 64,
                "attempted": 120,
                "succeeded": 120,
                "failed": 0,
                "cancelled": 0,
                "inconclusive": 0,
                "completedVideoMs": 600000,
                "eligibleSucceeded": 120,
                "eligibleVideoMs": 600000,
                "slowSucceeded": 0,
            }
        )
    ledger = {
        "schema": "attelier/prelaunch-ledger/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "window": {"startMs": DAY_MS, "endMs": 2 * DAY_MS},
        "sequence": 1,
        "closedAtMs": 2 * DAY_MS,
        "aggregates": rows,
        "gatewayPolicyHash": digest(gateway),
        "workerPoolHash": digest(workers),
        "policyHash": digest(policy),
    }
    return ledger, policy, registry


def _keypair(tmp_path):
    private = tmp_path / "fixture-ed25519.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    der = subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout
    return private, base64.b64encode(der[-32:]).decode("ascii")


def _feed(tmp_path, private, *, sequence=1, target=240, expires=None, tamper_signature=False):
    ledger, policy, registry = _inputs()
    ledger["sequence"] = sequence
    policy["simulation"]["targetVideosPerDay"] = target
    ledger["policyHash"] = digest(policy)
    body = {
        "schema": "attelier/light-input/v1",
        "network": "finney",
        "genesis": GENESIS,
        "netuid": 126,
        "mechanism": 0,
        "mode": "shadow",
        "issuedAtMs": ledger["closedAtMs"],
        "expiresAtMs": expires if expires is not None else 3 * DAY_MS,
        "policy": policy,
        "ledger": ledger,
        "registry": registry,
    }
    message = tmp_path / "message"
    message.write_bytes(DOMAIN + canonical(body))
    signature = subprocess.run(
        ["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message)],
        check=True,
        capture_output=True,
    ).stdout
    if tamper_signature:
        signature = b"x" * 64
    raw_key = subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout[-32:]
    envelope = {
        "schema": "attelier/light-feed/v1",
        "keyId": hashlib.sha256(raw_key).hexdigest(),
        "body": body,
        "digest": digest(body),
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    return canonical(envelope)


@contextmanager
def _server(payload):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture"
    finally:
        server.shutdown()
        thread.join()


@contextmanager
def _counting_server(payload, requests):
    """A fixture endpoint which proves CLI argument failures did not fetch."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture"
    finally:
        server.shutdown()
        thread.join()


def test_light_runner_verifies_real_http_signature_registry_sqlite_and_artifact(tmp_path):
    private, public = _keypair(tmp_path)
    payload = _feed(tmp_path, private)
    verified = []
    with _server(payload) as url:
        raw = fetch_feed(url, allow_http_fixture=True)
        artifact = LightRunner(
            tmp_path / "state", public, registry_verifier=lambda registry: verified.append(registry)
        ).process(raw, now_ms=2 * DAY_MS + 1)
    assert verified and artifact["simulationOnly"] is True and artifact["publishable"] is False
    assert artifact["chainCalls"] == []
    assert (tmp_path / "state" / "light.sqlite3").is_file()
    assert (
        json.loads((tmp_path / "state" / "latest-artifact.json").read_text())[
            "unsignedProposedWeights"
        ]["total"]
        == 65535
    )


def test_light_runner_rejects_bad_signature_replay_and_expiry_without_artifact_change(tmp_path):
    private, public = _keypair(tmp_path)
    runner = LightRunner(tmp_path / "state", public, registry_verifier=lambda _: None)
    valid = _feed(tmp_path, private)
    runner.process(valid, now_ms=2 * DAY_MS + 1)
    artifact = (tmp_path / "state" / "latest-artifact.json").read_bytes()
    with pytest.raises(LightError, match="light_simulation_input"):
        runner.process(_feed(tmp_path, private, sequence=0), now_ms=2 * DAY_MS + 1)
    with pytest.raises(LightError, match="light_replay_or_rollback"):
        runner.process(_feed(tmp_path, private, target=480), now_ms=2 * DAY_MS + 1)
    with pytest.raises(LightError, match="light_signature"):
        LightRunner(tmp_path / "bad", public, registry_verifier=lambda _: None).process(
            _feed(tmp_path, private, tamper_signature=True), now_ms=2 * DAY_MS + 1
        )
    with pytest.raises(LightError, match="light_expired_or_not_yet_valid"):
        LightRunner(tmp_path / "expired", public, registry_verifier=lambda _: None).process(
            _feed(tmp_path, private), now_ms=3 * DAY_MS
        )
    assert (tmp_path / "state" / "latest-artifact.json").read_bytes() == artifact


def test_light_runner_accepts_signed_target_ramp_and_recovers_latest_after_restart(tmp_path):
    private, public = _keypair(tmp_path)
    for target in (240, 480, 960):
        state = tmp_path / str(target)
        raw = _feed(tmp_path, private, target=target)
        runner = LightRunner(state, public, registry_verifier=lambda _: None)

        def process(_: int, runner=runner, raw=raw):
            return runner.process(raw, now_ms=2 * DAY_MS + 1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            artifacts = list(executor.map(process, range(2)))
        assert all(item["service"]["targetVideosPerDay"] == target for item in artifacts)
        (state / "latest-artifact.json").unlink()
        recovered = LightRunner(state, public, registry_verifier=lambda _: None)
        assert (state / "latest-artifact.json").is_file()
        assert recovered.process(raw, now_ms=2 * DAY_MS + 1) == artifacts[0]


def test_light_runner_marks_malformed_input_unavailable_without_reusing_status(tmp_path):
    _, public = _keypair(tmp_path)
    runner = LightRunner(tmp_path / "state", public, registry_verifier=lambda _: None)
    with pytest.raises(LightError, match="light_json"):
        runner.process(b"{malformed", now_ms=2 * DAY_MS + 1)
    runner.unavailable("light_json")
    assert json.loads((tmp_path / "state" / "status.json").read_text()) == {
        "schema": "attelier/light-status/v1",
        "status": "unavailable",
        "reason": "light_json",
    }


def test_registry_fixture_checks_finality_owner_mechanism_and_identities(monkeypatch):
    ledger, _, registry = _inputs()
    del ledger

    class Value:
        def __init__(self, value):
            self.value = value

    class At:
        def query(self, key, _args):
            return Value({"owner": OWNER, "mechanism": 1}[key])

    class Raw:
        def get_block_hash(self, _height):
            return "genesis"

    class FakeChain:
        def __init__(self, height=1, genesis=GENESIS, block_hash="0x" + "b" * 64):
            self.height = height
            self.subtensor = types.SimpleNamespace(
                _client=types.SimpleNamespace(_substrate=types.SimpleNamespace(raw=Raw())),
                _call=lambda _: genesis,
                at=lambda _: At(),
            )
            self.block_hash = block_hash

        def finalized_height(self):
            return self.height

        def finalized_block_hash(self, _block):
            return self.block_hash

        def owner_snapshot(self, _block):
            return {3: (MINER_A, MINER_A), 4: (MINER_B, MINER_B)}

        def close(self):
            return None

    storage = types.SimpleNamespace(
        SubtensorModule=types.SimpleNamespace(
            SubnetOwner="owner", MechanismCountCurrent="mechanism"
        )
    )
    monkeypatch.setitem(sys.modules, "bittensor", types.SimpleNamespace(storage=storage))
    verify_finalized_registry(registry, chain_factory=lambda *_: FakeChain())
    with pytest.raises(RegistryError, match="light_registry_not_finalized"):
        verify_finalized_registry(registry, chain_factory=lambda *_: FakeChain(height=0))
    observed = []

    def custom_factory(network, netuid):
        observed.append((network, netuid))
        return FakeChain()

    endpoint = "wss://subtensor.example:443"
    verify_finalized_registry(registry, network=endpoint, chain_factory=custom_factory)
    assert observed == [(endpoint, 126)]
    with pytest.raises(RegistryError, match="light_registry_genesis"):
        verify_finalized_registry(
            registry, network=endpoint, chain_factory=lambda *_: FakeChain(genesis="0xwrong")
        )
    with pytest.raises(RegistryError, match="light_registry_block_hash"):
        verify_finalized_registry(
            registry, network=endpoint, chain_factory=lambda *_: FakeChain(block_hash="0xwrong")
        )


def test_custom_subtensor_transport_keeps_finney_publisher_identity_and_rejects_bad_genesis(
    monkeypatch,
):
    """The SDK sees the custom URL while the publication guard still sees Finney."""

    class Item:
        def __init__(self, name):
            self.name = name

    class Value:
        def __init__(self, value):
            self.value = value

    genesis = [GENESIS]
    endpoint_calls = []

    class Raw:
        def get_chain_finalised_head(self):
            return "finalized"

        def get_block_number(self, _hash):
            return 12

        def get_block_hash(self, height):
            return "genesis" if height == 0 else "0xblock12"

        def query(self, module, item, args, block_hash):
            assert (module, item, args, block_hash) == (
                "SubtensorModule",
                "SubnetEpochConsensus",
                [126],
                "0xblock12",
            )
            return "Yuma"

    hotkeys = (
        ["burn-hotkey", "unused", "unused", "unused", MINER_A] + ["unused"] * 69 + ["validator"]
    )
    coldkeys = (
        [OWNER, "unused", "unused", "unused", "miner-cold"] + ["unused"] * 69 + ["validator-cold"]
    )

    class At:
        def query(self, item, _args=None):
            values = {
                "SubnetOwner": OWNER,
                "MechanismCountCurrent": 1,
                "CommitRevealWeightsEnabled": True,
                "WeightsSetRateLimit": 100,
                "Tempo": 360,
                "CommitRevealWeightsVersion": 4,
                "WeightsVersionKey": 0,
                "MaxWeightsLimit": 65_535,
                "RecycleOrBurn": "Burn",
                "Owner": OWNER,
            }
            return Value(values[item.name])

        def read(self, name, **_kwargs):
            if name == "metagraph":
                return {
                    "hotkeys": hotkeys,
                    "coldkeys": coldkeys,
                    "validator_permit": [False] * 74 + [True],
                }
            raise AssertionError(name)

    class Subtensor:
        def __init__(self, *, network, fallback_endpoints=None, archive_endpoints=None):
            endpoint_calls.append(
                {
                    "network": network,
                    "fallback_endpoints": fallback_endpoints,
                    "archive_endpoints": archive_endpoints,
                }
            )
            self.block = 12
            self._client = types.SimpleNamespace(_substrate=types.SimpleNamespace(raw=Raw()))

        def _call(self, value):
            return genesis[0] if value == "genesis" else value

        def at(self, _height):
            return At()

        def read(self, name, **_kwargs):
            if name == "block_info":
                return {"hash": "0xblock12"}
            raise AssertionError(name)

        def close(self):
            return None

    storage = types.SimpleNamespace(
        SubtensorModule=types.SimpleNamespace(
            **{
                name: Item(name)
                for name in (
                    "SubnetOwner",
                    "MechanismCountCurrent",
                    "CommitRevealWeightsEnabled",
                    "WeightsSetRateLimit",
                    "Tempo",
                    "CommitRevealWeightsVersion",
                    "WeightsVersionKey",
                    "MaxWeightsLimit",
                    "RecycleOrBurn",
                    "Owner",
                )
            }
        )
    )
    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Subtensor=Subtensor, storage=storage)
    )
    endpoint = "ws://subtensor.example.com:9944"
    chain = BittensorChain("finney", 126, endpoint=endpoint)
    assert (chain.network, chain.netuid) == ("finney", 126)
    assert endpoint_calls == [
        {"network": endpoint, "fallback_endpoints": [], "archive_endpoints": []}
    ]
    intent = LiveIntent(
        1,
        "digest",
        {"startMs": 0, "endMs": 3_600_000},
        "registry",
        {0: 95, 4: 5},
        {0: ("", OWNER), 4: (MINER_A, "miner-cold")},
    )
    publisher = SdkM0Publisher(chain, object(), "validator")
    assert publisher.guard(intent)["validatorUid"] == 74
    genesis[0] = "0xwrong"
    with pytest.raises(PublicationError, match="light_live_genesis"):
        publisher.guard(intent)


def test_light_cli_custom_network_and_alias_reach_registry_through_pinned_sdk(tmp_path):
    """Dry-run exercises CLI parsing, real registry verification and the SDK boundary."""

    private, public = _keypair(tmp_path)
    envelope = json.loads(_feed(tmp_path, private))
    body = envelope["body"]
    now = int(time.time() * 1000)
    window_end = now // 3_600_000 * 3_600_000
    body["mode"] = "live"
    body["policy"]["windowMs"] = 3_600_000
    body["policy"]["simulation"]["serviceBudgetBps"] = 500
    body["ledger"]["window"] = {"startMs": window_end - 3_600_000, "endMs": window_end}
    body["ledger"]["closedAtMs"] = window_end
    body["issuedAtMs"] = window_end
    body["expiresAtMs"] = window_end + 3_600_000
    body["ledger"]["policyHash"] = digest(body["policy"])
    message = tmp_path / "cli-live-message"
    message.write_bytes(DOMAIN + canonical(body))
    envelope["digest"] = digest(body)
    envelope["signature"] = base64.b64encode(
        subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message)],
            check=True,
            capture_output=True,
        ).stdout
    ).decode("ascii")

    runtime = tmp_path / "runtime"
    shutil.copytree(Path(__file__).parents[2] / "src" / "forge", runtime / "forge")
    (runtime / "forge" / "inference" / "defaults.json").write_text(
        json.dumps({"lightPublicKey": public}), encoding="ascii"
    )
    transport_log = tmp_path / "transport.jsonl"
    (runtime / "bittensor.py").write_text(
        f"""import json, os
from types import SimpleNamespace

OWNER = {OWNER!r}
GENESIS = {GENESIS!r}
MINER_A = {MINER_A!r}
MINER_B = {MINER_B!r}

class _Raw:
    def get_chain_finalised_head(self): return 'finalized'
    def get_block_number(self, _): return 1
    def get_block_hash(self, _): return 'genesis'

class _At:
    def query(self, item, _): return {{'SubnetOwner': OWNER, 'MechanismCountCurrent': 1}}[item]
    def read(self, name, **_):
        assert name == 'metagraph'
        return {{
            'hotkeys': ['unused', 'unused', 'unused', MINER_A, MINER_B],
            'coldkeys': ['unused', 'unused', 'unused', MINER_A, MINER_B],
        }}

class Subtensor:
    def __init__(self, *, network, fallback_endpoints=None, archive_endpoints=None):
        with open(os.environ['ATELIER_TEST_TRANSPORT_LOG'], 'a', encoding='ascii') as out:
            out.write(json.dumps([network, fallback_endpoints, archive_endpoints]) + '\\n')
        self.block = 1
        self._client = SimpleNamespace(_substrate=SimpleNamespace(raw=_Raw()))
    def _call(self, value): return GENESIS if value == 'genesis' else value
    def at(self, _): return _At()
    def read(self, name, **_):
        assert name == 'block_info'
        return {{'hash': '0x' + 'b' * 64}}
    def close(self): pass

storage = SimpleNamespace(
    SubtensorModule=SimpleNamespace(
        SubnetOwner='SubnetOwner', MechanismCountCurrent='MechanismCountCurrent'
    )
)
""",
        encoding="utf-8",
    )
    endpoint = "wss://subtensor.example:443"
    payload = canonical(envelope)
    with _server(payload) as url:
        for index, (flag, compatibility) in enumerate(
            (("--network", ()), ("--subtensor-endpoint", ("--live",)))
        ):
            environment = {
                **os.environ,
                "PYTHONPATH": str(runtime),
                "PYTEST_CURRENT_TEST": "fixture",
                "ATELIER_TEST_TRANSPORT_LOG": str(transport_log),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "forge.inference.light",
                    "--state-dir",
                    str(tmp_path / f"state-{index}"),
                    "--url",
                    url,
                    "--allow-http-fixture",
                    *compatibility,
                    "--dry-run",
                    "--once",
                    flag,
                    endpoint,
                ],
                cwd=runtime,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr + result.stdout
            dry_run = json.loads((tmp_path / f"state-{index}" / "live-dry-run.json").read_text())
            assert all(
                isinstance(pair, list) and len(pair) == 2 for pair in dry_run["relativeWeights"]
            )
    assert [json.loads(line) for line in transport_log.read_text().splitlines()] == [
        [endpoint, [], []],
        [endpoint, [], []],
    ]


@pytest.mark.parametrize("sdk_error", ["RuntimeError", "ChainError"])
def test_light_cli_default_live_feed_recovers_same_window_after_burn_before_sdk_guard(
    tmp_path, sdk_error
):
    """A fresh authenticated W+1 feed resumes prospectively after fallback burn."""

    private, public = _keypair(tmp_path)
    envelope = json.loads(_feed(tmp_path, private))
    body = envelope["body"]
    now = int(time.time() * 1000)
    window_end = now // 3_600_000 * 3_600_000
    body["mode"] = "live"
    body["policy"]["windowMs"] = 3_600_000
    body["policy"]["simulation"]["serviceBudgetBps"] = 500
    body["ledger"]["window"] = {"startMs": window_end - 3_600_000, "endMs": window_end}
    body["ledger"]["closedAtMs"] = window_end
    body["issuedAtMs"] = window_end
    body["expiresAtMs"] = window_end + 3_600_000
    body["ledger"]["policyHash"] = digest(body["policy"])
    message = tmp_path / "default-live-message"
    message.write_bytes(DOMAIN + canonical(body))
    envelope["digest"] = digest(body)
    envelope["signature"] = base64.b64encode(
        subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message)],
            check=True,
            capture_output=True,
        ).stdout
    ).decode("ascii")

    runtime = tmp_path / "runtime"
    shutil.copytree(Path(__file__).parents[2] / "src" / "forge", runtime / "forge")
    (runtime / "forge" / "inference" / "defaults.json").write_text(
        json.dumps({"lightPublicKey": public}), encoding="ascii"
    )
    trace = tmp_path / "trace.jsonl"
    (runtime / "bittensor.py").write_text(
        f"""import os
from types import SimpleNamespace

OWNER = {OWNER!r}
GENESIS = {GENESIS!r}
MINER_A = {MINER_A!r}
MINER_B = {MINER_B!r}
instances = 0

class ChainError(Exception):
    pass

def log(value):
    with open(os.environ['ATELIER_TEST_TRACE'], 'a', encoding='ascii') as out:
        out.write(value + '\\n')

class Wallet:
    def __init__(self, **_kwargs):
        log('wallet')
        self.hotkey = SimpleNamespace(ss58_address='validator')

class _Raw:
    def __init__(self, role): self.role, self.calls = role, 0
    def get_chain_finalised_head(self):
        if self.role == 'publisher':
            log('publisher-finalized-head')
            self.calls += 1
            if self.calls == 2:
                raise SystemExit(0)  # External boundary proves the polling loop survived.
            raise {sdk_error}('publisher-guard-sentinel')
        return 'finalized'
    def get_block_number(self, _): return 1
    def get_block_hash(self, _): return 'genesis'

class _At:
    def query(self, item, _):
        return {{'SubnetOwner': OWNER, 'MechanismCountCurrent': 1}}[item]
    def read(self, name, **_):
        assert name == 'metagraph'
        return {{
            'hotkeys': ['unused', 'unused', 'unused', MINER_A, MINER_B],
            'coldkeys': ['unused', 'unused', 'unused', MINER_A, MINER_B],
        }}

class Subtensor:
    def __init__(self, **_kwargs):
        global instances
        instances += 1
        self.role = 'publisher' if instances == 1 else 'registry'
        log(self.role)
        self.block = 1
        self._client = SimpleNamespace(_substrate=SimpleNamespace(raw=_Raw(self.role)))
    def _call(self, value): return GENESIS if value == 'genesis' else value
    def at(self, _): return _At()
    def read(self, name, **_):
        assert name == 'block_info'
        return {{'hash': '0x' + 'b' * 64}}
    def close(self): pass

storage = SimpleNamespace(
    SubtensorModule=SimpleNamespace(
        SubnetOwner='SubnetOwner', MechanismCountCurrent='MechanismCountCurrent'
    )
)
""",
        encoding="utf-8",
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "live-feed-missing.json").write_text(
        json.dumps(
            {
                "schema": "attelier/light-feed-missing/v1",
                "missingSinceMs": None,
                "burnedThroughEndMs": window_end,
            }
        ),
        encoding="utf-8",
    )
    with _server(canonical(envelope)) as url:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.inference.light",
                "--state-dir",
                str(state),
                "--url",
                url,
                "--allow-http-fixture",
                *(["--once"] if sdk_error == "RuntimeError" else ["--interval-seconds", "1"]),
                "--wallet.name",
                "writer",
                "--wallet.hotkey",
                "hot",
            ],
            cwd=runtime,
            env={
                **os.environ,
                "PYTHONPATH": str(runtime),
                "PYTEST_CURRENT_TEST": "fixture",
                "ATELIER_TEST_TRACE": str(trace),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    assert result.returncode == (1 if sdk_error == "RuntimeError" else 0)
    assert result.stdout.strip() == "light_publication_unavailable"
    assert trace.read_text().splitlines() == [
        "wallet",
        "publisher",
        "registry",
        "publisher-finalized-head",
    ] + (["publisher-finalized-head"] if sdk_error == "ChainError" else [])
    with sqlite3.connect(state / "light.sqlite3") as database:
        assert database.execute("SELECT status FROM live_intent").fetchall() == [("PREPARED",)]


def test_light_cli_default_writer_requires_explicit_wallet_before_sdk_or_fetch(tmp_path):
    """Default mode reaches the external wallet boundary; dry-run does not."""

    _private, public = _keypair(tmp_path)
    runtime = tmp_path / "runtime"
    shutil.copytree(Path(__file__).parents[2] / "src" / "forge", runtime / "forge")
    (runtime / "forge" / "inference" / "defaults.json").write_text(
        json.dumps({"lightPublicKey": public}), encoding="ascii"
    )
    wallet_log = tmp_path / "wallet.jsonl"
    (runtime / "bittensor.py").write_text(
        """import json, os
from types import SimpleNamespace

class Wallet:
    def __init__(self, *, name, hotkey, path):
        with open(os.environ['ATELIER_TEST_WALLET_LOG'], 'a', encoding='utf-8') as out:
            out.write(json.dumps([name, hotkey, path]) + '\\n')
        if os.environ.get('ATELIER_TEST_WALLET_MODE') != 'ok':
            raise RuntimeError('wallet-sentinel')
        self.hotkey = SimpleNamespace(ss58_address='validator')

class Subtensor:
    def __init__(self, **_kwargs):
        with open(os.environ['ATELIER_TEST_WALLET_LOG'], 'a', encoding='utf-8') as out:
            out.write('subtensor\\n')
""",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "PYTEST_CURRENT_TEST": "fixture",
        "ATELIER_TEST_WALLET_LOG": str(wallet_log),
    }
    requests = []
    with _counting_server(b"{}", requests) as url:
        missing = subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.inference.light",
                "--state-dir",
                str(tmp_path / "missing"),
                "--url",
                url,
                "--allow-http-fixture",
                "--once",
            ],
            cwd=runtime,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert missing.returncode == 2
        assert "--wallet.name and --wallet.hotkey are required" in missing.stderr
        assert requests == [] and not wallet_log.exists()

        commands = [
            ["--wallet.name", "canonical", "--wallet.hotkey", "hot-canonical"],
            [
                "--wallet",
                "alias",
                "--hotkey",
                "hot-alias",
                "--wallet-path",
                str(tmp_path / "custom-wallets"),
                "--live",
            ],
            [
                "--wallet.name",
                "dotted-path",
                "--wallet.hotkey",
                "hot-dotted",
                "--wallet.path",
                str(tmp_path / "dotted-wallets"),
            ],
        ]
        for index, command in enumerate(commands):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "forge.inference.light",
                    "--state-dir",
                    str(tmp_path / f"writer-{index}"),
                    "--url",
                    url,
                    "--allow-http-fixture",
                    "--once",
                    *command,
                ],
                cwd=runtime,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
        assert requests == []

    records = [json.loads(line) for line in wallet_log.read_text().splitlines()]
    assert records == [
        ["canonical", "hot-canonical", str(Path.home() / ".bittensor" / "wallets")],
        ["alias", "hot-alias", str(tmp_path / "custom-wallets")],
        ["dotted-path", "hot-dotted", str(tmp_path / "dotted-wallets")],
    ]


def test_light_cli_writer_rejects_signed_shadow_feed_before_publication(tmp_path):
    """A signed review receipt cannot reach publisher submission in default mode."""

    private, public = _keypair(tmp_path)
    envelope = json.loads(_feed(tmp_path, private))
    body = envelope["body"]
    now = int(time.time() * 1000)
    day_end = now // DAY_MS * DAY_MS
    body["ledger"]["window"] = {"startMs": day_end - DAY_MS, "endMs": day_end}
    body["ledger"]["closedAtMs"] = day_end
    body["issuedAtMs"] = day_end
    body["expiresAtMs"] = day_end + DAY_MS
    body["ledger"]["policyHash"] = digest(body["policy"])
    message = tmp_path / "shadow-message"
    message.write_bytes(DOMAIN + canonical(body))
    envelope["digest"] = digest(body)
    envelope["signature"] = base64.b64encode(
        subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message)],
            check=True,
            capture_output=True,
        ).stdout
    ).decode("ascii")

    runtime = tmp_path / "runtime"
    shutil.copytree(Path(__file__).parents[2] / "src" / "forge", runtime / "forge")
    (runtime / "forge" / "inference" / "defaults.json").write_text(
        json.dumps({"lightPublicKey": public}), encoding="ascii"
    )
    trace = tmp_path / "trace.jsonl"
    (runtime / "bittensor.py").write_text(
        """import os
from types import SimpleNamespace

class Wallet:
    def __init__(self, **_kwargs):
        with open(os.environ['ATELIER_TEST_TRACE'], 'a', encoding='ascii') as out:
            out.write('wallet\\n')
        self.hotkey = SimpleNamespace(ss58_address='validator')

class Subtensor:
    def __init__(self, **_kwargs):
        with open(os.environ['ATELIER_TEST_TRACE'], 'a', encoding='ascii') as out:
            out.write('subtensor\\n')
""",
        encoding="utf-8",
    )
    with _server(canonical(envelope)) as url:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "forge.inference.light",
                "--state-dir",
                str(tmp_path / "state"),
                "--url",
                url,
                "--allow-http-fixture",
                "--once",
                "--wallet.name",
                "writer",
                "--wallet.hotkey",
                "hot",
            ],
            cwd=runtime,
            env={
                **os.environ,
                "PYTHONPATH": str(runtime),
                "PYTEST_CURRENT_TEST": "fixture",
                "ATELIER_TEST_TRACE": str(trace),
            },
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 1
    assert result.stdout.strip() == "light_live_feed_required"
    assert trace.read_text().splitlines() == ["wallet", "subtensor"]


def test_hourly_live_feed_requires_fixed_five_percent_and_preserves_quantized_ratio(tmp_path):
    private, public = _keypair(tmp_path)
    raw = _feed(tmp_path, private)
    envelope = json.loads(raw)
    body = envelope["body"]
    body["mode"] = "live"
    body["policy"]["windowMs"] = 3_600_000
    body["policy"]["simulation"]["serviceBudgetBps"] = 500
    body["ledger"]["window"] = {"startMs": 3_600_000, "endMs": 7_200_000}
    body["ledger"]["closedAtMs"] = 7_200_000
    body["issuedAtMs"] = 7_200_000
    body["expiresAtMs"] = 10_800_000
    body["ledger"]["policyHash"] = digest(body["policy"])
    message = tmp_path / "live-message"
    message.write_bytes(DOMAIN + canonical(body))
    signature = subprocess.run(
        ["openssl", "pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message)],
        check=True,
        capture_output=True,
    ).stdout
    envelope["digest"] = digest(body)
    envelope["signature"] = base64.b64encode(signature).decode("ascii")
    live = canonical(envelope)
    verified = verify_feed(live, public, now_ms=7_200_001, allow_live=True)
    assert verified["mode"] == "live"
    # Default review accepts the canonical live hourly feed but remains the
    # existing unsigned simulation boundary (no wallet or publisher exists).
    reviewed = LightRunner(
        tmp_path / "default-live-review",
        public,
        allow_live=True,
        registry_verifier=lambda _: None,
    ).process(live, now_ms=7_200_001)
    assert reviewed["simulationOnly"] is True and reviewed["chainCalls"] == []
    with pytest.raises(LightError, match="light_mode"):
        verify_feed(live, public, now_ms=7_200_001)
    vector = relative_vector({4: 1})
    assert vector == {0: 95, 4: 5}
    # SDK's max-upscale quantization: 5/95 becomes 3449/65535.
    assert round(vector[4] / vector[0] * 65535) == 3449


def test_live_journal_keeps_committed_windows_moving_and_blocks_unknown(tmp_path):
    class FakePublisher:
        def guard(self, intent):
            return {"sequence": intent.sequence}

        def submit(self, intent, _guard):
            return {"status": "COMMITTED", "sequence": intent.sequence}

        def reconcile(self, intent, receipt):
            if receipt.get("error"):
                return {**receipt, "status": "UNKNOWN_AFTER_SUBMIT"}
            return {**receipt, "status": "REVEAL_PENDING"}

    journal = LiveJournal(tmp_path / "light.sqlite3")
    publisher = LivePublisher(journal, FakePublisher())
    registry = {
        "subnetOwnerColdkey": OWNER,
        "miners": [{"uid": 4, "hotkey": "worker", "coldkey": "worker-cold"}],
    }
    base = {
        "ledger": {"sequence": 1, "window": {"startMs": 0, "endMs": 3_600_000}},
        "registry": registry,
    }
    artifact = {"miners": [{"uid": 4, "hotkey": "worker", "eligibleVideos": 1}]}
    assert publisher.process(base, artifact)["status"] == "COMMITTED"
    next_body = {
        "ledger": {"sequence": 2, "window": {"startMs": 3_600_000, "endMs": 7_200_000}},
        "registry": registry,
    }
    assert publisher.process(next_body, artifact)["status"] == "COMMITTED"
    # A crash after the durable SUBMITTING marker must prevent a third write.
    third = {
        "ledger": {"sequence": 3, "window": {"startMs": 7_200_000, "endMs": 10_800_000}},
        "registry": registry,
    }
    intent = publisher.process
    del intent
    from forge.inference.light_publication import LiveIntent

    pending = LiveIntent(
        3,
        digest(third),
        third["ledger"]["window"],
        digest(registry),
        {0: 95, 4: 5},
        {0: ("", OWNER), 4: ("worker", "worker-cold")},
    )
    journal.prepare(pending)
    journal.update(pending, "UNKNOWN_AFTER_SUBMIT", {"error": "Timeout"})
    with pytest.raises(PublicationError, match="unreconciled"):
        publisher.process(third, artifact)


def test_sdk_m0_persists_exact_call_before_submit_reconciles_and_blocks_uid_reuse(tmp_path):
    """HTTP/SQLite test fixture drives the actual M0 adapter boundary.

    The fake chain is only transport/state: call parsing, durable journal
    ordering and SdkM0Publisher validation execute unmodified production code.
    """

    class Value:
        def __init__(self, value):
            self.value = value

    class Call:
        data = b"exact-commit-call"

    class Plan:
        ok, signer, signer_address, call = True, "hotkey", "validator", Call()

    class Result:
        success, extrinsic_id = True, "11-0000"

    class Signer:
        ss58_address = "validator"
        public_key = b"x" * 32
        crypto_type = 1

        def sign(self, _message):
            return b"fixture-signature"

    commit = b"ciphertext"
    decoded = {
        "call_module": "SubtensorModule",
        "call_function": "commit_timelocked_mechanism_weights",
        "call_args": [
            {"name": "netuid", "value": 126},
            {"name": "mecid", "value": 0},
            {"name": "commit", "value": commit},
            {"name": "reveal_round", "value": 99},
            {"name": "commit_reveal_version", "value": 4},
            {"name": "dests", "value": [0, 4]},
            {"name": "weights", "value": [65535, 3449]},
        ],
        "call_hash": hashlib.sha256(Call.data).hexdigest(),
    }

    class At:
        def __init__(self, height):
            self.height = height

        def query(self, item, args=None):
            name = item.name
            values = {
                "SubnetOwner": OWNER,
                "MechanismCountCurrent": 1,
                "CommitRevealWeightsEnabled": True,
                "WeightsSetRateLimit": 100,
                "Tempo": 360,
                "CommitRevealWeightsVersion": 4,
                "WeightsVersionKey": 0,
                "MaxWeightsLimit": 65535,
                "SubnetMechanism": 0,
                "RecycleOrBurn": "Burn",
                "Owner": OWNER,
            }
            if name == "Events":
                return [{"extrinsic_idx": 0, "module_id": "System", "event_id": "ExtrinsicSuccess"}]
            return Value(values[name])

        def query_map(self, item, args):
            assert item.name == "TimelockedWeightCommits" and args == [126]
            return []

        def read(self, name, **kwargs):
            if name == "metagraph":
                return {"validator_permit": [False] * 74 + [True]}
            if name == "weights":
                return {74: {0: 0.95, 4: 0.05}}
            raise AssertionError(name)

    class Raw:
        def get_block_hash(self, height):
            return height

        def query(self, module, item, args, block_hash):
            assert (module, item, args, block_hash) == (
                "SubtensorModule",
                "SubnetEpochConsensus",
                [126],
                "0xblock12",
            )
            return "Yuma"

    class Subtensor:
        def __init__(self):
            self._client = types.SimpleNamespace(
                _substrate=types.SimpleNamespace(
                    raw=Raw(),
                    decode_scale=lambda *_: decoded,
                    submit=lambda call, signer, **kwargs: Result(),
                )
            )
            self.submitted = 0

        def _call(self, value):
            if value == 0:
                return GENESIS
            return value

        def at(self, height):
            return At(height)

        def plan(self, intent, wallet):
            # This assertion proves SDK SetWeights (with default version_key 0)
            # reached the fixture rather than a hand-rolled call.
            assert intent.netuid == 126 and intent.mechid == 0 and intent.version_key == 0
            if self.submitted == 0:
                from bittensor.result import ChainError

                self.submitted += 1
                raise ChainError("Weights were last set 69 blocks ago; rate limit is 100")
            return Plan()

        def block_info(self, height):
            assert height == 11
            return types.SimpleNamespace(extrinsics=[{"address": "validator", "call": decoded}])

    class Chain:
        network, netuid = "finney", 126

        def __init__(self):
            self.subtensor = Subtensor()

        def finalized_height(self):
            return 12

        def finalized_block_hash(self, height):
            return f"0xblock{height}"

        def uid_snapshot(self, height):
            return {0: "burn-hotkey", 4: MINER_A, 74: "validator"}

        def owner_snapshot(self, height):
            return {
                0: ("burn-hotkey", OWNER),
                4: (MINER_A, "miner-cold"),
                74: ("validator", "validator-cold"),
            }

    chain = Chain()
    journal = LiveJournal(tmp_path / "light.sqlite3")
    adapter = SdkM0Publisher(chain, Signer(), "validator", lock_dir=tmp_path / "locks")
    publisher = LivePublisher(journal, adapter)
    body = {
        "ledger": {"sequence": 7, "window": {"startMs": 0, "endMs": 3_600_000}},
        "registry": {
            "subnetOwnerColdkey": OWNER,
            "miners": [{"uid": 4, "hotkey": MINER_A, "coldkey": "miner-cold"}],
        },
    }
    artifact = {"miners": [{"uid": 4, "hotkey": MINER_A, "eligibleVideos": 1}]}
    from bittensor.result import ChainError

    with pytest.raises(ChainError):
        publisher.process(body, artifact)
    with journal._connect() as db:
        assert db.execute("SELECT status FROM live_intent").fetchall() == [("PREPARED",)]
    assert adapter._submission_lock is None
    committed = publisher.process(body, artifact)
    assert committed["status"] == "COMMITTED"
    with journal._connect() as db:
        persisted = json.loads(
            db.execute("SELECT receipt FROM live_intent WHERE sequence=7").fetchone()[0]
        )
    assert persisted["preparedCallHex"] == "0x" + Call.data.hex()
    assert persisted["ciphertextSha256"] == hashlib.sha256(commit).hexdigest()
    assert publisher.process(body, artifact)["status"] == "REVEALED"

    # Same UID with a new hotkey/coldkey cannot receive the frozen feed's 5%.
    chain.owner_snapshot = lambda height: {
        0: ("burn-hotkey", OWNER),
        4: ("reused", "other-cold"),
        74: ("validator", "validator-cold"),
    }
    changed = {
        **body,
        "ledger": {"sequence": 8, "window": {"startMs": 3_600_000, "endMs": 7_200_000}},
    }
    with pytest.raises(PublicationError, match="target_identity_changed"):
        publisher.process(changed, artifact)


def test_restart_reconciles_ambiguous_submission_without_retransmit_and_pending_reveal_moves_on(
    tmp_path,
):
    class Publisher:
        submits = 0

        def guard(self, _intent):
            return {}

        def submit(self, _intent, _guard):
            self.submits += 1
            return {"status": "COMMITTED"}

        def reconcile(self, _intent, receipt):
            return {**receipt, "status": "REVEAL_PENDING"}

    registry = {
        "subnetOwnerColdkey": OWNER,
        "miners": [{"uid": 4, "hotkey": "worker", "coldkey": "worker-cold"}],
    }
    artifact = {"miners": [{"uid": 4, "hotkey": "worker", "eligibleVideos": 1}]}
    body = {
        "ledger": {"sequence": 1, "window": {"startMs": 0, "endMs": 3_600_000}},
        "registry": registry,
    }
    journal = LiveJournal(tmp_path / "light.sqlite3")
    first = LivePublisher(journal, Publisher())
    assert first.process(body, artifact)["status"] == "COMMITTED"
    # A fresh process sees the durable committed receipt, reconciles it and
    # then may publish the next hourly vector while reveal remains pending.
    restarted_publisher = Publisher()
    restarted = LivePublisher(journal, restarted_publisher)
    assert restarted.process(body, artifact)["status"] == "REVEAL_PENDING"
    next_body = {
        "ledger": {"sequence": 2, "window": {"startMs": 3_600_000, "endMs": 7_200_000}},
        "registry": registry,
    }
    assert restarted.process(next_body, artifact)["status"] == "COMMITTED"
    assert restarted_publisher.submits == 1


def _legacy_transport_row(tmp_path, status, endpoint="wss://rpc.example:443"):
    registry = {
        "subnetOwnerColdkey": OWNER,
        "miners": [{"uid": 4, "hotkey": "worker", "coldkey": "worker-cold"}],
    }
    artifact = {"miners": [{"uid": 4, "hotkey": "worker", "eligibleVideos": 1}]}
    body = {
        "ledger": {"sequence": 1, "window": {"startMs": 0, "endMs": 3_600_000}},
        "registry": registry,
    }
    legacy = LiveIntent(
        1,
        digest(body),
        body["ledger"]["window"],
        digest(registry),
        {0: 95, 4: 5},
        {0: ("", OWNER), 4: ("worker", "worker-cold")},
        endpoint,
    )
    journal = LiveJournal(tmp_path / "light.sqlite3")
    original = canonical(legacy.as_dict()).decode("ascii")
    with journal._connect() as db:
        db.execute(
            "INSERT INTO live_intent(sequence,intent,status,receipt,updated_at) VALUES(?,?,?,?,?)",
            (
                1,
                original,
                status,
                json.dumps({"status": status}) if status != "PREPARED" else None,
                0,
            ),
        )
    return journal, body, registry, artifact, legacy, original


def test_legacy_transport_label_reconciles_without_rewriting_then_new_intent_is_finney(tmp_path):
    """A prior URL-labelled journal remains recoverable but never seeds new identity."""

    class Publisher:
        def __init__(self):
            self.submits = 0
            self.reconciled_networks = []

        def guard(self, _intent):
            return {}

        def submit(self, _intent, _guard):
            self.submits += 1
            return {"status": "COMMITTED"}

        def reconcile(self, intent, receipt):
            self.reconciled_networks.append(intent.network)
            return {**receipt, "status": "REVEALED"}

    journal, first, registry, artifact, legacy, original = _legacy_transport_row(
        tmp_path, "COMMITTED"
    )
    second = {
        "ledger": {"sequence": 2, "window": {"startMs": 3_600_000, "endMs": 7_200_000}},
        "registry": registry,
    }
    with pytest.raises(PublicationError, match="journal_intent"):
        journal.prepare(legacy)

    publisher = Publisher()
    restored = LivePublisher(journal, publisher).process(first, artifact)
    assert restored["status"] == "REVEALED"
    assert restored["intent"]["network"] == "wss://rpc.example:443"
    assert publisher.submits == 0
    assert publisher.reconciled_networks == ["wss://rpc.example:443"]
    with journal._connect() as db:
        assert (
            db.execute("SELECT intent FROM live_intent WHERE sequence=1").fetchone()[0] == original
        )

    fresh = LivePublisher(journal, publisher).process(second, artifact)
    assert fresh["status"] == "COMMITTED"
    assert fresh["intent"]["network"] == "finney"
    assert publisher.submits == 1
    with journal._connect() as db:
        assert (
            json.loads(db.execute("SELECT intent FROM live_intent WHERE sequence=2").fetchone()[0])[
                "network"
            ]
            == "finney"
        )

    invalid = legacy.as_dict()
    for field, value in (("network", "test"), ("netuid", 125), ("mechanism", 1)):
        changed = {**invalid, field: value}
        with pytest.raises(PublicationError, match="journal_intent"):
            LiveIntent.from_dict(changed)


def test_legacy_prepared_transport_intent_is_journaled_before_submit_without_rewrite(tmp_path):
    journal, body, _registry, artifact, _legacy, original = _legacy_transport_row(
        tmp_path, "PREPARED"
    )

    class Publisher:
        def __init__(self):
            self.before_submit = None

        def guard(self, _intent):
            return {"guard": "fixture"}

        def submit(self, intent, _guard):
            with journal._connect() as db:
                self.before_submit = db.execute(
                    "SELECT intent,status FROM live_intent WHERE sequence=?", (intent.sequence,)
                ).fetchone()
            return {"status": "COMMITTED"}

        def reconcile(self, _intent, receipt):
            return receipt

    publisher = Publisher()
    result = LivePublisher(journal, publisher).process(body, artifact)
    assert result["status"] == "COMMITTED"
    assert publisher.before_submit == (original, "SUBMITTING")
    with journal._connect() as db:
        assert db.execute("SELECT intent,status FROM live_intent WHERE sequence=1").fetchone() == (
            original,
            "COMMITTED",
        )


def test_legacy_unknown_transport_intent_reconciles_without_retransmit(tmp_path):
    journal, body, _registry, artifact, _legacy, original = _legacy_transport_row(
        tmp_path, "UNKNOWN_AFTER_SUBMIT", "ws://rpc.example:9944"
    )

    class Publisher:
        submits = 0

        def guard(self, _intent):
            raise AssertionError("an unknown submission must never prepare another call")

        def submit(self, _intent, _guard):
            self.submits += 1
            raise AssertionError("an unknown submission must never retransmit")

        def reconcile(self, intent, receipt):
            assert intent.network == "ws://rpc.example:9944"
            return receipt

    publisher = Publisher()
    with pytest.raises(PublicationError, match="unreconciled_submission"):
        LivePublisher(journal, publisher).process(body, artifact)
    assert publisher.submits == 0
    with journal._connect() as db:
        assert db.execute("SELECT intent,status FROM live_intent WHERE sequence=1").fetchone() == (
            original,
            "UNKNOWN_AFTER_SUBMIT",
        )


def test_ambiguous_restart_stays_blocked_and_outage_burn_blocks_same_closed_window(tmp_path):
    class UnknownPublisher:
        submits = 0

        def guard(self, _intent):
            return {}

        def prepare_submission(self, _intent, guard):
            return {"opaque": True}, {
                "guard": guard,
                "preparedCallHex": "0xcafe",
                "ciphertextSha256": "a" * 64,
            }

        def submit(self, _intent, _guard, _prepared):
            self.submits += 1
            raise TimeoutError("transport lost")

        def reconcile(self, _intent, receipt):
            return {**receipt, "status": "UNKNOWN_AFTER_SUBMIT"}

    registry = {
        "subnetOwnerColdkey": OWNER,
        "miners": [{"uid": 4, "hotkey": "worker", "coldkey": "worker-cold"}],
    }
    body = {
        "ledger": {"sequence": 1, "window": {"startMs": 0, "endMs": 3_600_000}},
        "registry": registry,
    }
    artifact = {"miners": [{"uid": 4, "hotkey": "worker", "eligibleVideos": 1}]}
    journal = LiveJournal(tmp_path / "light.sqlite3")
    writer = UnknownPublisher()
    with pytest.raises(PublicationError, match="unknown_after_submit"):
        LivePublisher(journal, writer).process(body, artifact)
    with journal._connect() as db:
        preserved = json.loads(
            db.execute("SELECT receipt FROM live_intent WHERE sequence=1").fetchone()[0]
        )
    assert preserved["preparedCallHex"] == "0xcafe" and preserved["ciphertextSha256"] == "a" * 64
    with pytest.raises(PublicationError, match="unreconciled_submission"):
        LivePublisher(journal, UnknownPublisher()).process(body, artifact)
    assert writer.submits == 1
    later = {
        "ledger": {"sequence": 2, "window": {"startMs": 3_600_000, "endMs": 7_200_000}},
        "registry": registry,
    }
    with pytest.raises(PublicationError, match="unreconciled_submission"):
        LivePublisher(journal, UnknownPublisher()).process(later, artifact)
    assert _chain_bytes("0x63697068657274657874") == b"ciphertext"
