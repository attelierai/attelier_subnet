#!/usr/bin/env python3
"""Exercise the built inference API ledger against the offline M1 verifier.

All SQLite data and keys are created below a temporary directory.  This is an
offline wire check: it has no RPC, wallet, provider, or publication path.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from bittensor.keyfiles import Keypair

from forge.chain import Sr25519Signer
from forge.inference.gateway_epoch import (
    commitment,
    digest,
    light_plan,
    master_window,
    sign_gateway,
)

NODE_FIXTURE = r"""
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
const root = resolve(process.argv[2]);
const { Store } = await import(join(root, "api/dist/store.js"));
const { exportGatewayWindow } = await import(
  join(root, "api/dist/gateway-ledger.js")
);
const dir = mkdtempSync(join(tmpdir(), "attelier-gateway-wire-"));
const now = 2_000_000, start = 1_990_000, end = 2_000_000;
const profileHash = "a".repeat(64), profileId = "h3-base5s";
const client = {
  id: "wire-client", scopes: "[]", models: "[]", limit_micros: 0,
  spent_micros: 0, reserved_micros: 0, enabled: 1
};
const params = { prompt: "synthetic wire fixture", duration: 5 };
const store = new Store(dir);
try {
  store.createClient(client.id, 1_000_000, ["requests:write"], ["attelier/h3"]);
  function add(id, minerHotkey, trafficKind, admittedAtMs, terminalAtMs, actual) {
    const quote = store.quote(client, params, {
      amountMicros: 7, expiresAt: Date.now() + 60_000, token: id
    });
    const job = store.enqueue(client, quote.id, id, 1_000_000);
    store.gatewayDispatch(job.id, {
      network: "test", netuid: 578, mechanism: 1, minerHotkey,
      workerId: `${minerHotkey}-worker`, trustMode: "operator_whitelist",
      trafficKind, profileId, profileHash, videoDurationMs: 5000, admittedAtMs
    });
    store.finish(job, "succeeded", actual, {
      sha256: "b".repeat(64), sizeBytes: 1,
      costAccuracy: actual === null ? "estimated" : "measured"
    });
    store.db.prepare(
      "UPDATE gateway_ledger SET terminal_at=?, dispatched_at=? WHERE job_id=?"
    ).run(terminalAtMs, admittedAtMs, job.id);
  }
  // Two equal, SLA-eligible 5s rows and one deliberately slow organic row.
  add("organic-fast", "miner-a", "organic", now - 900, now - 500, null);
  add("synthetic-fast", "miner-b", "synthetic", now - 900, now - 500, 7);
  add("organic-slow", "miner-a", "organic", now - 3_000, now - 1_500, null);
  const policy = { minSuccessBps: 0, maxTtvMs: 1000,
    serviceProfile: { id: profileId, hash: profileHash, durationMs: 5000, resolution: "480p" } };
  const pool = { workers: 2, fixture: "wire-v1" };
  const body = exportGatewayWindow(store.db, { network: "test", netuid: 578,
    mechanism: 1, startMs: start, endMs: end, sequence: 1, createdAtMs: now,
    expiresAtMs: now + 10_000, policy, pool });
  writeFileSync(join(dir, "body.json"), JSON.stringify(body));
  console.log(JSON.stringify({ body, assertions: {
    aggregateCount: body.aggregates.length,
    organicBilledNull:
      body.aggregates.find(x => x.trafficKind === "organic").billedAmountMicros === null,
    syntheticBilledZero:
      body.aggregates.find(x => x.trafficKind === "synthetic").billedAmountMicros === 0,
    slowExcluded: body.aggregates.find(x => x.trafficKind === "organic").slowSucceeded === 1,
  }}));
} finally { store.db.close(); rmSync(dir, { recursive: true, force: true }); }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    api_root = args.api_root.resolve()
    if not (api_root / "api" / "dist" / "store.js").is_file():
        raise SystemExit("--api-root must contain the current built api/dist/store.js")
    temporary = Path(tempfile.mkdtemp(prefix="gateway-wire-check-"))
    try:
        fixture = temporary / "fixture.mjs"
        fixture.write_text(NODE_FIXTURE, encoding="utf8")
        node = subprocess.run(["node", str(fixture), str(api_root)], capture_output=True, text=True)
        if node.returncode:
            raise RuntimeError(f"Node fixture failed: {node.stderr.strip()}")
        exported = json.loads(node.stdout)
        body = exported["body"]
        if exported["assertions"] != {
            "aggregateCount": 2,
            "organicBilledNull": True,
            "syntheticBilledZero": True,
            "slowExcluded": True,
        }:
            raise AssertionError("unexpected TypeScript ledger aggregation")
        private_key = temporary / "gateway-ed25519.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
            check=True,
            capture_output=True,
        )
        public = subprocess.run(
            ["openssl", "pkey", "-in", str(private_key), "-pubout", "-outform", "DER"],
            check=True,
            capture_output=True,
        ).stdout[-32:]
        alice = Sr25519Signer(SimpleNamespace(hotkey=Keypair.create_from_uri("//Alice")))
        bob = Keypair.create_from_uri("//Bob").ss58_address
        policy = {
            "network": "test",
            "netuid": 578,
            "mechanism": 1,
            "gatewayPublicKey": base64.b64encode(public).decode(),
            "pool": {"workers": 2, "fixture": "wire-v1"},
            "gatewayPolicy": {
                "minSuccessBps": 0,
                "maxTtvMs": 1000,
                "serviceProfile": {
                    "id": "h3-base5s",
                    "hash": "a" * 64,
                    "durationMs": 5000,
                    "resolution": "480p",
                },
            },
            "workers": [
                {
                    "minerHotkey": "miner-a",
                    "workerId": "miner-a-worker",
                    "trustMode": "operator_whitelist",
                },
                {
                    "minerHotkey": "miner-b",
                    "workerId": "miner-b-worker",
                    "trustMode": "operator_whitelist",
                },
            ],
            "masterPublisherHotkey": alice.validator_hotkey,
            "m0PublisherHotkey": bob,
        }
        assert body["policyHash"] == digest(policy["gatewayPolicy"])
        assert body["poolHash"] == digest(policy["pool"])
        gateway = sign_gateway(body, private_key)
        # Deliberately feed an RPC-shaped hash.  The signed artifact is bare hex.
        master = master_window(
            gateway, policy, alice, "0x" + "c" * 64, finalized_block=123, now_ms=2_000_000
        )
        assert master["body"]["finalizedRegistryBlockHash"] == "c" * 64
        anchor = {
            "network": "test",
            "netuid": 578,
            "mechanism": 1,
            "finalizedRegistryBlock": 123,
            "finalizedRegistryBlockHash": "c" * 64,
            "observedBlock": 124,
            "observedBlockHash": "d" * 64,
            "commitmentBlock": 124,
            "publisherHotkey": alice.validator_hotkey,
            "rawCommitment": commitment(master),
            "m0PublisherHotkey": bob,
        }
        metagraph = {
            "network": "test",
            "netuid": 578,
            "mechanism": 1,
            "observedBlock": 124,
            "observedBlockHash": "d" * 64,
            "uids": {"miner-a": 41, "miner-b": 42},
        }
        plan = light_plan(master, policy, anchor, metagraph, now_ms=2_000_000)
        assert plan["uids"] == [41, 42] and plan["weights"] == [32768, 32767]
        receipt = {
            "schema": "attelier/gateway-wire-proof/v1",
            "offline": True,
            "apiRoot": str(api_root),
            "fixture": {"jobs": 3, "profile": "h3-base5s", "minSuccessBps": 0, "maxTtvMs": 1000},
            "ts": exported["assertions"],
            "aggregates": body["aggregates"],
            "master": {
                "finalizedRegistryBlock": 123,
                "finalizedRegistryBlockHash": master["body"]["finalizedRegistryBlockHash"],
                "weights": master["body"]["weights"],
            },
            "plan": {"uids": plan["uids"], "weights": plan["weights"]},
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8"
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
