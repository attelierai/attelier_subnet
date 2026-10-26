"""The timer runner must resume one M1 intent without a second submission."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CYCLE = ROOT / "scripts" / "inference_gateway_cycle.py"


def _cycle_module():
    spec = importlib.util.spec_from_file_location("inference_gateway_cycle", CYCLE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def test_cycle_recovers_in_order_and_never_advances_before_reveal(tmp_path):
    closed, work, authority = tmp_path / "closed", tmp_path / "work", tmp_path / "authority"
    body = {
        "schema": "attelier/gateway-window/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "sequence": 1,
        "previousHash": None,
    }
    _write(closed / "000001.json", body)
    log, fail = tmp_path / "commands.jsonl", tmp_path / "fail-light-once"
    fail.write_text("yes", encoding="utf8")
    fake = tmp_path / "epoch.py"
    fake.write_text(
        """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ['CYCLE_LOG'])
with log.open('a') as out: out.write(json.dumps(args) + '\\n')
def value(flag): return Path(args[args.index(flag) + 1])
command = args[0]
if command == 'gateway-sign':
    body = json.loads(value('--body').read_text())
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    digest = __import__('hashlib').sha256(encoded).hexdigest()
    envelope = {'body': body, 'digest': digest}; print(json.dumps(envelope))
elif command == 'master-live-sign':
    value('--preflight-out').write_text('{}')
    gateway = json.loads(value('--envelope').read_text())
    encoded = json.dumps(
        gateway, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode()
    source = __import__('hashlib').sha256(encoded).hexdigest()
    print(json.dumps({'body': {'sequence': gateway['body']['sequence'],
                      'sourceGatewayEnvelopeHash': source}, 'digest': 'master-1'}))
elif command == 'master-commit':
    value('--receipt').parent.mkdir(parents=True, exist_ok=True)
    value('--receipt').write_text(json.dumps({'status': 'SUBMITTED'})); print('{}')
elif command == 'master-reconcile':
    receipt = value('--receipt')
    receipt.write_text(json.dumps({'status': 'FINALIZED'})); print('{}')
elif command == 'light-publish':
    marker = Path(os.environ['CYCLE_FAIL'])
    if marker.exists(): marker.unlink(); raise SystemExit(9)
    value('--receipt').write_text(json.dumps({'status': 'SUBMITTED'})); print('{}')
elif command == 'light-reconcile':
    receipt = value('--receipt'); receipt.write_text(json.dumps({'status': 'REVEALED'}))
    state = value('--state')
    state.write_text(json.dumps({'sequence': 1, 'masterDigest': 'master-1'})); print('{}')
else: raise SystemExit(command)
""",
        encoding="utf8",
    )
    policy = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "masterPublisherHotkey": "master-ss58",
        "m0PublisherHotkey": "m0",
        "workers": [{"trustMode": "operator_whitelist"}],
    }
    policy_path = tmp_path / "policy.json"
    _write(policy_path, policy)
    config = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "epochCommand": [sys.executable, str(fake)],
        "gatewaySealCommand": [sys.executable, "-c", "pass"],
        "closedBodyDirectory": str(closed),
        "workDirectory": str(work),
        "policy": str(policy_path),
        "gatewayPrivateKey": str(tmp_path / "gateway.key"),
        "master": {
            "walletName": "master",
            "walletHotkey": "master-alias",
            "publisherHotkey": "master-ss58",
            "walletPath": str(tmp_path / "wallets"),
        },
        "light": {
            "walletName": "light",
            "walletHotkey": "light",
            "walletPath": str(tmp_path / "wallets"),
        },
        "publicationAuthorityDirectory": str(authority),
    }
    config_path = tmp_path / "cycle.json"
    _write(config_path, config)
    environment = {**os.environ, "CYCLE_LOG": str(log), "CYCLE_FAIL": str(fail)}
    first = subprocess.run(
        [sys.executable, str(CYCLE), "--config", str(config_path)],
        text=True,
        capture_output=True,
        env=environment,
    )
    assert first.returncode != 0
    assert not (work / "cycle-state.json").exists()
    second = subprocess.run(
        [sys.executable, str(CYCLE), "--config", str(config_path)],
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    assert json.loads(second.stdout) == {
        "masterDigest": "master-1",
        "sequence": 1,
        "status": "REVEALED",
    }
    third = subprocess.run(
        [sys.executable, str(CYCLE), "--config", str(config_path)],
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    assert json.loads(third.stdout) == {"status": "IDLE"}
    commands = [json.loads(line)[0] for line in log.read_text(encoding="utf8").splitlines()]
    assert commands == [
        "gateway-sign",
        "master-live-sign",
        "master-commit",
        "master-reconcile",
        "light-publish",
        "light-publish",
        "light-reconcile",
    ]
    master_live = json.loads(log.read_text(encoding="utf8").splitlines()[1])
    assert master_live[master_live.index("--wallet-hotkey") + 1] == "master-alias"


def test_master_artifact_binds_the_signed_gateway_envelope_not_its_body_digest(tmp_path):
    from forge.inference.gateway_epoch import MASTER_DOMAIN, canonical, digest, master_window

    class Signer:
        validator_hotkey = "master-ss58"

        def sign(self, message):
            assert message.startswith(MASTER_DOMAIN)
            return b"test-signature"

    key = tmp_path / "gateway.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)], check=True)
    public = subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout[-32:]
    signer = Signer()
    policy = {
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "gatewayPublicKey": base64.b64encode(public).decode(),
        "pool": {},
        "gatewayPolicy": {
            "minSuccessBps": 0,
            "maxTtvMs": 1000,
            "serviceProfile": {
                "id": "p",
                "hash": "a" * 64,
                "durationMs": 5000,
                "resolution": "480p",
            },
        },
        "workers": [],
        "masterPublisherHotkey": signer.validator_hotkey,
        "m0PublisherHotkey": "m0-ss58",
    }
    body = {
        "schema": "attelier/gateway-window/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "window": {"startMs": 1, "endMs": 2},
        "sequence": 1,
        "createdAtMs": 2,
        "expiresAtMs": 9_999_999_999_999,
        "policyHash": digest(policy["gatewayPolicy"]),
        "poolHash": digest(policy["pool"]),
        "ledgerRowsHash": "b" * 64,
        "previousHash": None,
        "aggregates": [],
    }
    body_path = tmp_path / "body.json"
    _write(body_path, body)
    signed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "inference_gateway_epoch.py"),
            "gateway-sign",
            "--body",
            str(body_path),
            "--private-key",
            str(key),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    gateway = json.loads(signed.stdout)
    master = master_window(gateway, policy, signer, "c" * 64, now_ms=3, finalized_block=1)
    cycle = _cycle_module()
    paths = {"master": tmp_path / "master.json", "masterReceipt": tmp_path / "master.receipt.json"}
    _write(paths["master"], master)
    _write(paths["masterReceipt"], {"status": "FINALIZED"})
    assert cycle._master({}, paths, gateway) == master
    bad_body = copy.deepcopy(master["body"])
    bad_body["sourceGatewayEnvelopeHash"] = gateway["digest"]
    bad = {
        "schema": master["schema"],
        "body": bad_body,
        "digest": digest(bad_body),
        "signature": signer.sign(MASTER_DOMAIN + canonical(bad_body)).hex(),
    }
    _write(paths["master"], bad)
    with pytest.raises(ValueError, match="gateway_cycle_master_binding"):
        cycle._master({}, paths, gateway)


def test_gateway_sign_command_receives_only_body_and_output_and_returns_real_envelope(tmp_path):
    from forge.inference.gateway_epoch import digest

    key = tmp_path / "gateway.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)], check=True)
    profile = {"id": "p", "hash": "a" * 64, "durationMs": 5000, "resolution": "480p"}
    body = {
        "schema": "attelier/gateway-window/v1",
        "network": "test",
        "netuid": 578,
        "mechanism": 1,
        "window": {"startMs": 1, "endMs": 2},
        "sequence": 1,
        "createdAtMs": 2,
        "expiresAtMs": 9_999_999_999_999,
        "policyHash": digest({"minSuccessBps": 0, "maxTtvMs": 1, "serviceProfile": profile}),
        "poolHash": digest({}),
        "ledgerRowsHash": "b" * 64,
        "previousHash": None,
        "aggregates": [],
    }
    body_path = tmp_path / "closed.json"
    _write(body_path, body)
    wrapper = tmp_path / "remote-sign.py"
    wrapper.write_text(
        """import os, subprocess, sys
from pathlib import Path
args=sys.argv[1:]; body=Path(args[args.index('--body')+1]); out=Path(args[args.index('--out')+1])
result=subprocess.run(
    [sys.executable, os.environ['EPOCH'], 'gateway-sign', '--body', str(body),
     '--private-key', os.environ['KEY']], text=True, capture_output=True, check=True)
out.write_text(result.stdout)
""",
        encoding="utf8",
    )
    cycle = _cycle_module()
    paths = {"gateway": tmp_path / "cold" / "signed.json"}
    config = {
        "gatewaySignCommand": [sys.executable, str(wrapper), "--body", "{body}", "--out", "{out}"]
    }
    old = os.environ.copy()
    os.environ.update(
        EPOCH=str(ROOT / "scripts" / "inference_gateway_epoch.py"),
        KEY=str(key),
        PYTHONPATH=str(ROOT / "src"),
    )
    try:
        envelope = cycle._gateway(config, body_path, body, paths)
    finally:
        os.environ.clear()
        os.environ.update(old)
    assert envelope["body"] == body and envelope["digest"] == cycle._body_digest(body)


def test_light_reveal_pending_is_a_normal_resume_state(tmp_path):
    cycle = _cycle_module()
    fake = tmp_path / "epoch.py"
    fake.write_text(
        """import json, sys
from pathlib import Path
args=sys.argv[1:]
if args[0] != 'light-reconcile': raise SystemExit(args[0])
receipt=Path(args[args.index('--receipt')+1])
receipt.write_text(json.dumps({'status':'REVEAL_PENDING'})); print('{}')
""",
        encoding="utf8",
    )
    authority = tmp_path / "cold-authority"
    receipt = authority / "m1-000.light-receipt.json"
    _write(receipt, {"status": "SUBMITTED"})
    paths = {
        "lightReceipt": receipt,
        "lightState": authority / "gateway-light-state.json",
        "master": tmp_path / "master.json",
    }
    config = {
        "epochCommand": [sys.executable, str(fake)],
        "policy": str(tmp_path / "policy.json"),
        "light": {"walletName": "n", "walletHotkey": "h", "walletPath": str(tmp_path)},
        "publicationAuthorityDirectory": str(authority),
    }
    assert cycle._light(config, paths, {"body": {"sequence": 3}, "digest": "m"}) is False
    assert json.loads(receipt.read_text())["status"] == "REVEAL_PENDING"
