# Validator guide

On Finney SN126, the default **light** validators copy the primary validator's finalized M0
row. The **primary** reads Attelier's signed gateway feed and applies the
selected incentive mechanism. Each validator submits from its own hotkey when
publication is enabled. A CPU host with Python 3.12 and OpenSSL is sufficient.

## Start a light validator

From the repository root, install the chain dependency and run one walletless
check. This reads finalized chain state and records the exact proposed vector.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[chain]'
.venv/bin/attelier-validator --role light --dry-run --once --state-dir ./state/check
```

For continuous publication, use your registered validator wallet and a durable
directory writable by its service user. Provision the shared publisher lock
directory as described below.

```bash
.venv/bin/attelier-validator --role light --wallet.name NAME --wallet.hotkey HOTKEY --state-dir ./state/light
```

You can omit `--role light`. The release pins the primary hotkey. Light checks Finney genesis, SN126/M0,
owner UID 0, the primary's identity and validator permit, its last update and
the target identities. It reads raw integer weights at the last finalized
100-block boundary and journals one copy per boundary. Commit-reveal means
changes become visible after the primary's reveal and the light validator's
own commit follows the normal reveal schedule. If the primary is stale or the
RPC cannot establish these checks, the light waits.

## Run the primary

| Mode | Input | Empty window |
| --- | --- | --- |
| `--role primary --im minimax` | H3 gateway receipt, profile policy and finalized registry | H3 continuity and burn policy below |
| `--role primary --im api` | Signed hourly API allocation, accepted organic receipts and finalized registry | 100% owner burn |

Use a distinct state directory per mode and preserve all publication journals.
With the API mechanism the release supplies the gateway URL and verifier key.
To inspect an allocation without a wallet, run:

```bash
.venv/bin/attelier-validator --role primary --im api --dry-run --once --state-dir ./state/api-check
```

The check writes `live-dry-run.json`. Its allocation shares sum to 65,535.
`relativeWeights` contains the max-normalized integer vector for M0. Only
the configured primary hotkey can publish the API allocation. The command for
live publication uses the same selector, removes `--dry-run --once`, and adds
the explicit wallet name, hotkey and a durable state directory.

## Keep 100% burn during a transition

Run the existing primary with `--burn-only` and its normal wallet arguments.
It journals one owner-only vector per UTC hour. Run the API primary check in a
separate walletless process and state directory. Switch the primary writer
only after reconciling its existing journal. Two writers sharing one hotkey
can collide even on different hosts. Installing this release performs no switch.

## MiniMax H3 policy and existing installations

The MiniMax primary (`--role primary --im minimax`) fetches the reviewed gateway feed, verifies
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

Live uses a 5% service ceiling. Eligible deliveries earn their frozen profile's
work units, discounted by the miner's frozen remuneration bid; any unused share
stays burn. Trial and ramp deliveries follow the same accounting. A miner with
only failed or ineligible deliveries keeps zero reward without blocking other
miners' valid work. This is a proposed mining-weight allocation, not a guaranteed
payout or a share of total subnet emissions: effective incentives depend on
chain consensus.

V4 validates each frozen bid group's work against its original admission
snapshot before combining rewards. The live publisher and dry-run use the same
discounted vector, including its residual burn. Already submitted historical
intents retain their exact bytes for reconciliation; an upgrade never rewrites
their published allocation.

Work that becomes eligible and terminal in UTC window **W** determines the
vector published during **W+1**. For example, eligible completions from
12:00–13:00 UTC determine the next window's vector after 13:00 UTC. The runner
polls every 60 seconds by default; commit-reveal and finalized readback can delay
when that vector becomes effective on chain. Work still running at the boundary
belongs to the window in which it finishes.

Empty signed windows or transport/age failures first try bounded continuity
with fresh signed availability, as described below. Without eligible continuity,
an empty window burns 100%; a missing or unavailable input falls back to burn
after 10 minutes. A fresh,
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
The MiniMax primary verifies those inputs and calculates the vector. Community
light validators follow the primary's finalized M0 row for either mechanism.

## Existing MiniMax primary command

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
.venv/bin/attelier-validator --role primary --im minimax --wallet.name NAME --wallet.hotkey NAME
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
.venv/bin/attelier-validator --role primary --im minimax --dry-run --once \
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
  --role primary --im minimax --wallet.name NAME --wallet.hotkey NAME
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

## Upgrading a live validator

A bare wallet command now follows the primary on chain. The primary operator selects
`--role primary --im minimax` or `--role primary --im api`. A dedicated
`--burn-only` service keeps its current command. Explicit legacy H3 feed URLs
and continuity settings retain their receipt-based behavior.

The current live feed uses V5. V4-only readers reject this schema and must be
upgraded even when the endpoint and signing key are unchanged. Update the code
and install its dependencies into the existing project virtual environment,
then restart the existing validator service once. Keep its wallet, hotkey and
state directory unchanged; do not delete the database or publication journals.
V1–V4 history remains readable and pending submissions continue reconciliation.

Continuity defaults to 2 hours and requires fresh signed availability. Choose
`--continuity-max-hours 6` only for an explicitly selected longer grace period,
or `0` to disable it. This changes neither the service ceiling nor the source
deadline, and it creates no delivered-work entries.
