# Validator guide

## Default validation: light

The intended default for the SN126 validator is **`light`**. A light validator
runs on a CPU host: it checks service accounting against the network policy and
registered miner identities, then deterministically calculates the proposed
weights. It does not serve models, download model checkpoints, rent a GPU, or
regenerate videos. It relies on the approved gateway for the reported work;
accounting checks do not independently prove model execution.

The deployment target is to install and start the validator with the network
defaults already supplied by the release: Finney, SN126, mechanism 0, the
reviewed gateway and policy, and indefinite one-hour UTC windows. Operators
should not have to configure miner endpoints, generation profiles, reward
thresholds, or the window length.

The light validator **publishes weights by default**. Start it with
`--wallet.name NAME --wallet.hotkey NAME`; `--live` is no longer needed.
It accepts fresh signed one-hour Finney/SN126/M0 live receipts with a fixed 500
bps service pool. It never chooses a wallet implicitly: both names are required
for publication. Use **`--dry-run`** to verify and simulate without loading a
wallet, signing, or sending weights. Give dry-run its own `--state-dir`; the
runner locks that directory to prevent simultaneous use by another process.

## What is available now

The shipped CPU-only `light` runner fetches the reviewed gateway feed, verifies
its pinned Ed25519 signature with the system OpenSSL provider, checks the
Finney/SN126/M0 namespace and fresh one-hour window, then reads the registry
block again from chain before producing the existing unsigned simulation. It
persists the accepted envelope, sequence, digest and artifact in SQLite, so a
replayed or changed sequence is rejected across restarts.

Historical daily-shadow registry checks use the Finney archive endpoint, since a
pinned block may be pruned from an ordinary head endpoint. The genesis
and finalized block hash are still verified independently.
You can select your own RPC with `--network` or `--subtensor-endpoint`; see
[Custom Subtensor RPC](#custom-subtensor-rpc) below.

Live uses a fixed 5% service / 95% owner-burn split. Eligible reward is measured
in completed video seconds, including trial and ramp deliveries; phase does not
discount it. Empty signed windows are burn. If the signed input remains missing
or unavailable after 10 minutes, that window is burn; the next valid window
resumes normally without backfill. Invalid input is unavailable, never
fabricated as empty work. A policy permits one non-revoked worker per hotkey;
paused workers still occupy that binding.

Dry-run's weight vector is an accounting proposal only. Default operation can
prepare a CRv4 M0 commit after its independent finalized guards pass, but actual
chain publication remains unconfirmed until its commit, automatic reveal, and
finalized weight readback are each recorded.

The release pins the gateway's base64 raw 32-byte Ed25519 **public** key in
package defaults. That is a public verification value, not a wallet key or a
provider secret.
The host needs OpenSSL 3.0 or later with `openssl pkeyutl` Ed25519 support.

Gateway operators review provider bindings and seal the service ledger.
Validators check those inputs, calculate the vector and publish it; they do not need to
operate the gateway or approve miners themselves.

## Run the validator

Install the chain extra in the release environment before starting it:

```bash
.venv/bin/python -m pip install -e '.[chain]'
attelier-validator --wallet.name NAME --wallet.hotkey NAME
```

The wallet directory defaults to `~/.bittensor/wallets` for the user running the
process. No wallet path flag is needed for a standard Bittensor installation.
For another location, add `--wallet.path PATH`. `--wallet`, `--hotkey`, and
`--wallet-path` remain aliases for the dotted options.

Use [scripts/inference/attelier-light.service.example](../scripts/inference/attelier-light.service.example)
as an installation template. Its dedicated service user has an isolated wallet
directory, so that example supplies `--wallet.path`. The state directory defaults
to `/var/lib/attelier-light`; provision it for the running user or supply another
writable `--state-dir`. Keep the original state directory when upgrading.
The runner rejects redirects and non-reviewed
production domains. Local HTTP is
reserved for the integration fixture; it is not an operator option.

To simulate one window without a wallet:

```bash
attelier-validator --dry-run --once --state-dir /var/lib/attelier-light-check
```

Dry-run records `simulationOnly:true`, `publishable:false`, and the proposed
vector in `live-dry-run.json`. Normal operation journals each window before
signing and cannot bypass an unknown submission. A commit, its reveal, and exact
M0 vector readback must all be recorded before publication can be claimed.

**Upgrading:** commands with wallet arguments now publish unless they include
`--dry-run`. Add that flag to any existing simulation-only service before
updating. Old commands containing `--live` still work; the flag is redundant,
and `--live --dry-run` still simulates without signing.

## Custom Subtensor RPC

Use `--network` with an absolute `ws://` or `wss://` URL to select your own node:

```bash
attelier-validator \
  --network wss://subtensor.example.com:9944 \
  --state-dir /var/lib/attelier-light \
  --wallet.name NAME --wallet.hotkey NAME
```

`--subtensor-endpoint` is an alias for the same option. It applies to
`--dry-run` and normal publication. An explicit URL is used for both finalized
registry reads and weight publication; it does not fall back to a public RPC.
Your node must retain the historical state at the signed feed's registry block.
If that state is unavailable, validation fails without publishing that feed's
weights. Before starting a live writer, check the connection without a wallet:

```bash
attelier-validator --network wss://subtensor.example.com:9944 \
  --dry-run --once --state-dir /var/lib/attelier-light-check
```

Omitting the option, or passing `--network finney`, preserves the defaults:
the SDK's Finney connection for publication and the Finney archive RPC for
registry checks. This selects a connection to **Finney SN126, mechanism 0**;
it does not enable testnet or another chain. The finalized genesis and registry
checks still apply, and new journal entries retain the `finney` identity.

When upgrading an installation that previously patched the RPC URL into the
source, stop its existing process, install the unmodified release, add the flag,
and keep its original live `--state-dir`. Existing URL-labelled journal entries
are reconciled in place without rewriting their intent or resubmitting an
unresolved transaction. Do not delete the journal or patch `NETWORK`.

See the [accounting and publication contract](INFERENCE_PRELAUNCH.md) for the
service policy and finalized readback requirements.
