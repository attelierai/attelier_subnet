# Inference prelaunch simulator

This is the only reward calculation tool permitted during the current prelaunch
rehearsal. It reads closed, local data and emits an unsigned review artifact. It
does not import a wallet, signer, chain client or publication component. It has
no `--execute`, wallet or publication-authority option.

The namespace is Finney SN126, mechanism 0, with the fixed Finney genesis hash.
Every ledger window is exactly one UTC day. Each row must bind a reviewed
`operator_whitelist` worker to the policy and a finalized public registry entry.
Missing, duplicate, unregistered or malformed rows fail; they are never treated
as an empty day.

Run it locally:

```sh
PYTHONPATH=src .venv/bin/python scripts/inference_prelaunch.py \
  --ledger closed-ledger.json --policy configs/inference-prelaunch.json \
  --registry finalized-registry.json --out simulation.json
```

The policy resembles `configs/inference-prelaunch.example.json`; its 95% / 180s
values are illustrative inputs, not an accepted production SLA. The ledger has
schema `attelier/prelaunch-ledger/v1`, namespace fields, a UTC `window`, a
positive sequence, `closedAtMs`, exact `policyHash`, `gatewayPolicyHash` and `workerPoolHash`,
and one aggregate per `(minerHotkey,workerId)`.
An aggregate contains the worker/profile/trust binding and terminal/eligible
counts. The registry has schema `attelier/prelaunch-registry/v1`, final block
metadata, `subnetOwnerColdkey`, and `{uid,hotkey,coldkey}` records.

The simulator applies the configured basis-point service budget linearly up to
the daily target, derives deterministic integer weights totalling 65,535 with a
synthetic burn entry, and emits alpha-rao values as decimal strings. The
top-level `asset:{kind:"alpha",netuid:126,decimals:9}` identifies SN126 alpha:
1,000,000,000 alpha-rao equals one alpha, not one TAO. It separately
shows counterfactual service merit and the payout expected under owner rules.
The latter is derived from the quantized unsigned weight, not from an ideal
fraction, so a positive merit row with zero integer weight has zero expected
payout.
An owner-associated test row can have service merit but its expected
mining payout is zero; that allocation appears as simulated burn. A non-owner
allocation remains conditional on external validator consensus and is not a
prediction of Yuma or a reward promise.

The output always says `simulationOnly:true`, `publishable:false` and
`chainCalls:[]`, and records the closed-ledger timestamp and finalized observed
block/hash used by the registry, plus the frozen gateway-policy and worker-pool
hashes. It cannot be submitted to chain. A later launch requires a
separate explicit decision and a different, reviewed publication workflow.

`policyHash` binds the entire policy, including the reward budget and target,
to the closed window. A different price or emission comparison belongs in a
separate market-simulation artifact; it cannot reprice an existing receipt.

`--out` creates the artifact atomically and only accepts a byte-identical retry.
It refuses to replace a different existing artifact, so an HTTP reader cannot
observe a partial rewrite or silently serve a result for changed inputs.

## Closing the gateway ledger

The approved gateway sealer produces the closed ledger from its durable
`prelaunch_gateway_ledger` SQLite table. It requires an existing database, the
complete policy, the UTC day start, the next sequence and a new output path. It
is separate from the historical signed gateway exporter.

```sh
node scripts/prelaunch-seal.mjs --db /var/lib/attelier-inference/ledger.sqlite \
  --policy /etc/attelier/inference-prelaunch.json --start-ms UTC_DAY_START_MS \
  --sequence 1 --out /var/lib/attelier-inference/closed-ledger.json
```

The example timestamp is illustrative; choose the actual completed UTC day.
Work belongs to its terminal completion day, even if it started earlier. Pending
work is not a success. Successful rows require a locally custodied media hash
and size. Organic and synthetic work use the same criteria; latency includes
queue time from admission to terminal delivery. The sealer enforces contiguous
closed days and immutable, byte-identical retries, including after a restart.
Fixture tests may control time only in isolated databases; never rewrite the
timestamps of a real GPU rehearsal.

## Finalized registry projection

`scripts/inference_prelaunch_registry.py` is a separate read-only producer for
the provider gateway. It requires prelaunch mode, an operator-owned
`attelier/subnet-h3/v1` configuration, and either a live finalized Finney read
or a local finalized snapshot fixture. It never accepts a wallet, signing or
publication option.

The live path lazily opens `BittensorChain("finney", 126)`, reads one finalized
block and its owner snapshot, verifies the exact genesis and that there is one
mechanism, then writes `forge/inference-registry/v1`. The output lists every
finalized UID but only includes workers that are operator-enabled, unexpired at
the observed timestamp and exactly match their configured UID/hotkey. A reviewed
but expired or unregistered enabled worker fails closed. An empty worker map is
valid for onboarding.

For offline checks, `--chain-snapshot` accepts a local
`forge/inference-registry-snapshot/v1` object with `finalized:true`, timestamp,
block/hash, genesis, mechanism count, subnet owner and UID-to-hotkey/coldkey
records. A fixture exercises projection code only; it never proves current chain
state. UID keys are the only numeric JSON object keys accepted for this snapshot;
the registry schema validates them again before projection. `--simulation-registry-out` additionally writes the restricted registry
input consumed by the reward simulator. The registry projection is freshness
data, so a newer finalized output may atomically replace an older one.
