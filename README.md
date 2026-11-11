<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-banner.jpg" alt="Attelier" width="100%" /></a>
</p>

# Attelier Subnet · Inference

<p align="center">
  <a href="https://attelier.ai">Website</a> ·
  <a href="https://dashboard.attelier.ai">Dashboard</a> ·
  <a href="https://docs.attelier.ai">Docs</a> ·
  <a href="https://x.com/attelier_ai">X</a>
</p>

SN126 mechanism 0 serves H3 through a permissioned gateway. The default CPU
validator is **light**: it verifies the signed gateway feed and finalized
registry, calculates the weight vector, and **publishes weights by default**.
The gateway, public verification key and network policy ship with the release.

```bash
.venv/bin/attelier-validator --wallet.name NAME --wallet.hotkey NAME
```

Wallets default to `~/.bittensor/wallets`; `--wallet.path` is optional.
`--wallet` and `--hotkey` remain aliases. To simulate without loading a wallet
or sending weights, run `attelier-validator --dry-run` with a separate
`--state-dir`. Existing simulation services must add `--dry-run` when upgrading;
`--live` remains accepted but is no longer required. See the
[validator guide](docs/validator.md) for installation and custom RPCs.
Publication is confirmed only after commit, reveal and finalized readback.
The current live-policy contract uses indefinite one-hour UTC windows. Eligible
work completed in window W determines accounting for W+1; pending work belongs
to the window in which it becomes terminal.

The current V5 feed requires a V5-compatible reader; older V4-only readers
reject it. Upgrade the validator code and environment, then restart the existing
service with the same wallet and state directory. Preserve pending commits and
publication journals. The reader continues to verify historical V1–V4 inputs.

The signed policy uses 5% of the mining pool as a service ceiling, not a
guaranteed allocation. Eligible, timely, custodied exact-profile deliveries earn
work units; their frozen remuneration bid discounts the service allocation, and
the deterministic residual burns rather than being normalized back to 5%.
V5 gives organic work twice the reward units of equivalent synthetic work,
inside that same service ceiling. Work keeps its admission policy across an
hour boundary; cancelled work earns no units.

During an empty window or a transport/age failure, signed fresh availability
can retain the latest positive miner shares for up to 2 hours by default.
`--continuity-max-hours 6` explicitly selects a longer grace period; `0` disables
it. Restarts and heartbeats do not extend the historical source deadline.
Unavailable, paused, revoked or changed workers lose their shares to burn.
Bad signatures and identity faults cannot authorize continuity. Without eligible
work or valid continuity, the unused service share burns; a later valid window
resumes normally without retroactive catch-up.

Synthetic demand runs continuously and can fill healthy compatible capacity
when organic preemption is enabled. Admitted organic work has priority; a
cancelled synthetic retains its slot until release is confirmed. An owner-associated row has zero
expected mining payout. Historical daily and 10% shadow artifacts remain replay
history, not the current live policy.

New signed policies use per-job eligibility: a completion already eligible for
the configured profile, custody, and TTV contributes on its own. Policies
without that field retain the historic aggregate success-threshold rule.

## Start here

- [Prelaunch contract](docs/INFERENCE_PRELAUNCH.md)
- [Provider guide](docs/miner.md)
- [Validator guide](docs/validator.md)
- [Gateway architecture and trust boundary](docs/INFERENCE.md)

## Local checks

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,chain,vision]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

These commands verify local code only; they do not activate a worker, weight, or
chain action.

<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-mark.svg" alt="Attelier" width="56" /></a>
</p>
