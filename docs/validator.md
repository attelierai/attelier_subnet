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

Validation mode and publication are separate. Default `light` remains shadow,
read-only. The productive path is explicit: `--live --wallet NAME --hotkey NAME`.
It accepts only fresh signed one-hour Finney/SN126/M0 receipts with a fixed 500
bps service pool; it never discovers a wallet or enables itself from a feed.

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

Live uses a fixed 5% service / 95% owner-burn split. Eligible reward is measured
in completed video seconds, including trial and ramp deliveries; phase does not
discount it. Empty signed windows are burn. If the signed input remains missing
or unavailable after 10 minutes, that window is burn; the next valid window
resumes normally without backfill. Invalid input is unavailable, never
fabricated as empty work. A policy permits one non-revoked worker per hotkey;
paused workers still occupy that binding.

Shadow's weight vector is an accounting proposal only. Explicit live mode can
prepare a CRv4 M0 commit after its independent finalized guards pass, but actual
chain publication remains unconfirmed until its commit, automatic reveal, and
finalized weight readback are each recorded.

The release pins the gateway's base64 raw 32-byte Ed25519 **public** key in
package defaults. That is a public verification value, not a wallet key or a
provider secret.
The host needs OpenSSL 3.0 or later with `openssl pkeyutl` Ed25519 support.

Gateway operators review provider bindings and seal the service ledger.
Validators check those inputs and run the simulation; they do not need to
operate the gateway or approve miners themselves.

## Run the light review

Install the chain extra in the release environment before starting it:

```bash
.venv/bin/python -m pip install -e '.[chain]'
attelier-validator --state-dir /var/lib/attelier-light
```

Use [scripts/inference/attelier-light.service.example](../scripts/inference/attelier-light.service.example)
as an installation template. The runner rejects redirects and non-reviewed
production domains. Local HTTP is
reserved for the integration fixture; it is not an operator option.

Shadow simulation remains unsigned and records `simulationOnly:true`,
`publishable:false`, and no chain calls. Its one-hour dry-run is explicit:
`attelier-validator --live --dry-run --once`. Productive mode is explicit:
`--live --wallet NAME --hotkey NAME --wallet-path PATH`; it journals each
window before signing, has no retry loop, and cannot bypass an unknown
submission. A commit still requires separate authorization, reveal evidence,
and exact M0 vector readback before publication can be claimed.

Check the [prelaunch contract](INFERENCE_PRELAUNCH.md) before operating the
runner. A later chain action would require separately authorized commit/reveal,
consensus, and finalized readback.
