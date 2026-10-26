# Historical testnet M1 gateway epochs

## Current status

This file describes the closed testnet-578 mechanism-1 gateway pilot. Its signed
master, commitment and publication commands are historical and are **not** part
of the current SN126 prelaunch rehearsal. Do not run its cycle for prelaunch:
it deliberately contains `--execute` publication actions.

The current rehearsal is permissioned and uses local 24-hour ledger simulation
only. [`INFERENCE_PRELAUNCH.md`](INFERENCE_PRELAUNCH.md) is the canonical
prelaunch simulator contract. It has no wallet, signer, chain client,
publication-authority option or `--execute` option, and produces only unsigned
artifacts marked non-publishable. This restriction applies to the rewards
pipeline: provider onboarding separately verifies an existing hotkey through a
bounded signed ownership challenge before operator review. That proof is neither
self-approval nor a chain publication. Provider manual/API admission, worker
credentials and review are described in [`INFERENCE.md`](INFERENCE.md).

The remaining sections are retained to explain the former testnet evidence and
its safety boundaries. They do not authorize a Finney configuration, a weight
write, commit/reveal, external-validator rollout, or a reward claim.

The gateway exports one canonical `attelier/gateway-window/v1` body for testnet
`netuid: 578`, `mechanism: 1`. It contains the bounded window, sequence, hashes
of the operator policy/pool/private sorted ledger rows, previous window body hash,
and per-worker aggregates. All JSON keys are ASCII; numbers are safe integers;
floating point is rejected. The gateway signs `attelier/gateway-window/v1\0 ||
canonical(body)` with its Ed25519 key. The public key is pinned in the operator
policy, never supplied by the envelope.

Only explicitly `operator_whitelist` workers can earn M1 weight. `trace_verified`
is audit evidence, not a trust escalation. Organic and synthetic Poisson
bootstrap traffic use the same serving counters, but this testnet reward signal
is distinct from commercial billing. Eligible service is the configured 5-second
480p profile and its per-request SLA. This initial contract has no uptime or
availability reward claim; `minSuccessBps` is measured over attempted requests.

The master is a different hotkey from the M0 publisher. It verifies the gateway
signature, TTL, policy and pool hashes, builds a largest-remainder vector totaling
65,535 (or `[]` if nobody is eligible), binds its finalized registry block hash,
and Sr25519-signs the exact body. The commitment is
`attelier:m1:v1:<sha256(master envelope)>`. Signing and planning commands do not
broadcast. `master-commit --execute` publishes the commitment and
`light-publish --execute` submits M1 weights through the guarded receipt path.

Light validators verify the master signature and exact body, local policy, raw
on-chain commitment/publisher/anchor, freshness and persistent monotonic sequence
state. They then map only the current anchored metagraph hotkeys to UIDs; an
unregistered or remapped hotkey is rejected rather than trusting an old UID.
An empty vector is an explicit no-op, never an owner burn.

Windows are contiguous: a successor starts exactly at its predecessor's end and
references the SHA-256 of that predecessor's canonical body. Terminal time assigns
each completed request to one window; an in-flight request carries into a later
window. Before closing a bounded campaign's final window, settle all admitted work
and choose its end after the last terminal row. A no-op does not erase previous
on-chain weights or guarantee an immediate halt to protocol emissions.

`light-publish` is execution-gated and requires an explicit canonical
`--publication-authority-directory`. Its receipt and optional replay-state file
must reside directly in that established M1 CR authority directory, rather than
a per-window directory, so its durable single-writer intent covers every UID-0
M1 publication. Submission does not consume the gateway sequence. Run
`light-reconcile` with the same master, policy, receipt, state and authority;
only a receipt proven `REVEALED`, whose stored master digest and requested UID
weights match the signed master, advances replay state.

## Trusted-cycle operator

`scripts/inference_gateway_cycle.py` is the narrow persistent runner for this
whitelist pilot. It is not a miner process and it has no alternative chain,
signature, or publication implementation: it invokes the guarded epoch CLI.
Before every bounded attempt it invokes the protected `gatewaySealCommand`
argv from the operator config. That issuer/API-root-owned command atomically
materializes only *closed, terminal* gateway bodies in
`closedBodyDirectory`; a body is eligible only after every request assigned to
its bounded window is terminal. The runner refuses a sequence gap or a
`previousHash` that does not match its last revealed body.

For one window it performs, in this order: gateway signing; M1 master live
preflight/sign; M1 master commitment and reconciliation; light publication and
reconciliation. It writes its next cycle state only after the existing light
receipt is `REVEALED` **and** the existing light replay state names the same
sequence and master digest. A restart reuses the signed artifacts and receipts;
it never submits a second intent. `NOOP` and pending/unknown receipts stop the
cycle without advancing to the next body.

The versioned template is
`configs/inference-gateway-cycle.example.json`; copy it as the protected,
operator-owned `/etc/attelier-inference/gateway-cycle.json`, fill public hotkey
names and protected local paths, and keep the gateway private key and wallets
out of Git. The runner enforces testnet/578/M1, a master distinct from M0, and
that every configured worker is exactly `operator_whitelist`. It does not turn
trace audit material into a gate or an authorization.

The master configuration deliberately separates `walletHotkey` (the local
Bittensor keyfile alias passed to `Wallet`) from `publisherHotkey` (the public
SS58 expected by the pinned policy). The runner compares only the latter to the
policy; the guarded epoch commands still derive and check the wallet's actual
SS58 before signing or broadcasting. Do not put an SS58 in the local alias
field merely to satisfy configuration validation.

`gatewaySealCommand` is an argv array, never shell text. It owns the private
ledger and any issuer API authentication; the runner supplies it neither wallet
arguments nor credentials. Its only output contract is immutable JSON bodies in
the configured closed-body directory. Exactly one signing mode is configured:
`gatewayPrivateKey` keeps `gateway-sign` local, while `gatewaySignCommand` is an
argv template containing exactly one `{body}` and one `{out}`. The latter is for
a root-owned transport wrapper to an isolated signing host; it must write one
signed envelope to `{out}`. The local runner validates that envelope against the
body before master signing and never supplies wallet arguments to the wrapper.

`ops/inference-gateway-cycle.service` and `.timer` run one bounded attempt
every five minutes. They must be installed for one service user only; no manual
invocation may overlap that unit or write its authority directory. Before
enabling the timer, the root operator must test one non-broadcasting invocation
against the protected config, ensure the wallet service credentials are
available to that service user, and verify the authority directory is durable
and empty except for prior M1 receipts/state. The runner itself has no secrets
and no default config path.

External light validators may supply the envelope with `--master-url` instead
of `--master-envelope` to `light-live-plan`, `light-publish`, or
`light-reconcile`. The URL must be HTTPS, is fetched without redirects with a
10-second socket timeout and a 1 MiB cap, and must decode to one JSON object. The
download is not authority: each validator keeps and pins its own policy file
(including the Ed25519 key and master hotkey), verifies the signed envelope,
and performs the existing finalized-chain anchor checks. A payload can never
replace policy or keys.

Each successful aggregate carries `eligibleSucceeded`, `eligibleVideoMs`, and
`slowSucceeded`. Eligibility is per successful request against the policy's
`maxTtvMs`; M1 uses only the eligible values, so failed or slow work cannot be
rewarded through a favourable aggregate percentile.

`master-live-sign` reads a finalized testnet-578 registry snapshot and signs
only after its safety preflight. `light-live-plan` reads the latest commitment
for the master hotkey and the current finalized UID map. The signed registry
block and later commitment-observation block are distinct explicit fields. Both
commands only read, sign, or plan: they never publish a commitment or broadcast
weights. The master is registered but need not have a validator permit; it must
remain different from M0, while the eventual M1 weight publisher may be another
registered hotkey.

Offline commands (stdout is canonical JSON):

```sh
PYTHONPATH=src .venv/bin/python scripts/inference_gateway_epoch.py gateway-sign --body body.json --private-key gateway-ed25519.pem
PYTHONPATH=src .venv/bin/python scripts/inference_gateway_epoch.py master-sign --envelope gateway.json --policy policy.json --finalized-registry-block-hash HASH --wallet-name NAME --wallet-hotkey HOTKEY --wallet-path PATH
PYTHONPATH=src .venv/bin/python scripts/inference_gateway_epoch.py light-plan --master-envelope master.json --policy policy.json --anchor anchor.json --metagraph metagraph.json --state state.json
PYTHONPATH=src .venv/bin/python scripts/inference_gateway_epoch.py master-live-sign --envelope gateway.json --policy policy.json --wallet-name NAME --wallet-hotkey HOTKEY --wallet-path PATH --preflight-out master-preflight.json
PYTHONPATH=src .venv/bin/python scripts/inference_gateway_epoch.py light-live-plan --master-envelope master.json --policy policy.json --state state.json --preflight-out light-preflight.json
```
