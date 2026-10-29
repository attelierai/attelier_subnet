# Validator guide

## Default validation: light

`attelier-validator` runs the SN126 **light validator**. No mode flag is needed.
A light validator
runs on a CPU host: it checks service accounting against the network policy and
registered miner identities, then deterministically calculates the proposed
weights. It does not serve models, download model checkpoints, rent a GPU, or
regenerate videos. It relies on the approved gateway for the reported work;
accounting checks do not independently prove model execution.

The release supplies the network defaults: Finney, SN126, mechanism 0, the
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

Live uses a fixed 5% service / 95% owner-burn split when there is eligible work.
The 5% is shared among eligible miners in proportion to completed video seconds,
including trial and ramp deliveries; phase does not discount them. This is a
proposed mining-weight allocation, not a guaranteed payout or a share of total
subnet emissions: effective incentives depend on chain consensus.

Work that becomes eligible and terminal in UTC window **W** determines the
vector published during **W+1**. For example, eligible completions from
12:00–13:00 UTC determine the next window's vector after 13:00 UTC. The runner
polls every 60 seconds by default; commit-reveal and finalized readback can delay
when that vector becomes effective on chain. Work still running at the boundary
belongs to the window in which it finishes.

Empty signed windows burn 100%. If the signed input remains missing
or unavailable after 10 minutes, the outage fallback is also burn. A fresh,
authenticated live receipt resumes normal publication prospectively, including
if it arrives later in the same publication window after a fallback burn. It
does not reverse or compensate an earlier burn. Invalid input is unavailable, never
fabricated as empty work. A policy permits one non-revoked worker per hotkey;
paused workers still occupy that binding.

Dry-run's weight vector is an accounting proposal only. Default operation
journals, submits and reconciles a CRv4 M0 commit after its independent finalized
guards pass, but actual
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

Use Python 3.12 or later and OpenSSL 3.0 or later. From the repository root,
create the environment if it does not exist and install the chain extra:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[chain]'
```

The hotkey must already be registered on Finney SN126 and have a validator
permit. The process needs access to its native Bittensor wallet files; it does
not need a Provider API key, a GPU or miner credentials.

On a new host, provision the default state directory for the OS user that will
run the process:

```bash
sudo install -d -m 700 -o "$(id -u)" -g "$(id -g)" /var/lib/attelier-light
```

For an existing installation, retain the directory, its owner and its journal.
The systemd template provisions it for its own dedicated user instead. Start
the validator with the installed executable:

```bash
.venv/bin/attelier-validator --wallet.name NAME --wallet.hotkey NAME
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
Live publication also uses `/var/lib/attelier-light/publisher-locks`, even with
a custom `--state-dir`; the running user must be able to create and use that
directory. Run only one writer for a hotkey, including across different hosts.
The runner rejects redirects and non-reviewed
production domains. Local HTTP is
reserved for the integration fixture; it is not an operator option.

To simulate one window without a wallet:

```bash
.venv/bin/attelier-validator --dry-run --once \
  --state-dir "$HOME/.bittensor/sn126_attelier/dry-run"
```

Dry-run records `simulationOnly:true`, `publishable:false`, and the proposed
vector in `live-dry-run.json`. Normal operation journals each window before
signing and cannot bypass an unknown submission. A commit, its reveal, and exact
M0 vector readback must all be recorded before publication can be claimed.

**Upgrading:** commands with wallet arguments now publish unless they include
`--dry-run`. Add that flag to any existing simulation-only service before
updating. Old commands containing `--live` still work; the flag is redundant,
and `--live --dry-run` still simulates without signing.

### PM2

If you already use PM2, start the executable directly from the repository root:

```bash
pm2 start .venv/bin/attelier-validator \
  --name attelier-validator --interpreter none -- \
  --wallet.name NAME --wallet.hotkey NAME
pm2 save
pm2 logs attelier-validator
```

Append `--network wss://subtensor.example.com:9944` after the wallet arguments
if you use your own RPC. Keep the existing process manager and wallet/state
locations when upgrading; stop the old writer before replacing it. Do not run
the PM2 example alongside a systemd instance for the same hotkey.

## Custom Subtensor RPC

Use `--network` with an absolute `ws://` or `wss://` URL to select your own node:

```bash
.venv/bin/attelier-validator \
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
.venv/bin/attelier-validator --network wss://subtensor.example.com:9944 \
  --dry-run --once --state-dir "$HOME/.bittensor/sn126_attelier/dry-run"
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

## Check publication and recovery

Read these files in the configured state directory:

| File or table | What it establishes |
| --- | --- |
| `status.json` | `ready` means the current feed was accepted; it does not establish that weights are applied. `unavailable` includes a reason code. |
| `latest-artifact.json` | The checked source window, eligible work and calculated vector. Its `simulationOnly:true` and `publishable:false` describe the unsigned calculation, even when the separate live publisher is running. |
| `light.sqlite3`, table `live_intent` | The durable publication intent, status and chain receipt for each window. |
| `live-dry-run.json` | A dry-run proposal; never evidence of a submitted transaction. |

The publication states distinguish preparation from chain completion:

- `PREPARED`: the exact intent is journaled, but not yet confirmed submitted.
- `SUBMITTING` or `UNKNOWN_AFTER_SUBMIT`: submission is unresolved. The runner
  must reconcile it before another submission; never clear the journal to retry.
- `COMMITTED` or `REVEAL_PENDING`: submission has progressed, but reveal and
  finalized weight readback are not yet both confirmed.
- `REVEALED`: the journal contains the confirmed reveal and weight readback.

SDK rate limits and publication RPC failures report
`light_publication_unavailable` and retry on the next polling cycle. A preflight
failure leaves the intent `PREPARED`; an uncertain submission is reconciled
before another send. The same recovery applies to outage-burn publication.
Publication failures do not turn an accepted feed into a missing-feed burn.
With `--once`, a failed publication exits with status 1 instead of polling.

For a read-only publication check, if `sqlite3` is installed:

```bash
cat /var/lib/attelier-light/status.json
sqlite3 -readonly /var/lib/attelier-light/light.sqlite3 \
  'SELECT sequence, status, updated_at FROM live_intent ORDER BY updated_at DESC LIMIT 5;'
```

Use your configured state path if different. Preserve `light.sqlite3` and the
rest of that directory across restarts and upgrades. Recovery from an old
outage burn never bypasses an unresolved submission. Other validators showing
the same weight vector proves alignment of their applied weights, not which
software they run; a local `REVEALED` receipt does not establish network-wide
consensus or a miner payout.

See the [accounting and publication contract](INFERENCE_PRELAUNCH.md) for the
service policy and finalized readback requirements.
