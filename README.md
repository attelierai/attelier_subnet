<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-banner.jpg" alt="Attelier" width="100%" /></a>
</p>

# Attelier Subnet (Bittensor SN126)

<p align="center">
  <a href="https://attelier.ai">Website</a> ·
  <a href="https://providers.attelier.ai">Providers Dashboard</a> ·
  <a href="https://console.attelier.ai/manage">Provider Console</a> ·
  <a href="https://docs.attelier.ai">Docs</a> ·
  <a href="https://x.com/attelier_ai">X</a>
</p>

**Attelier (Subnet 126 on Bittensor Finney)** powers decentralized generative video inference behind one unified USD API (`https://inference.attelier.ai`), with passwordless Bittensor hotkey onboarding and TAO / SN126 Gamma billing.

## Dual Incentive Mechanism (Finney M0)

SN126 ships **two incentive mechanisms** on Finney Mechanism 0. The primary validator selects one active weight source (or runs in `--burn-only` observer mode during rollouts); community **light validators** automatically copy the primary's finalized on-chain weights across mechanism switches.

| Mechanism | Status | Miner provides | Pricing & Competition | Eligible work |
| --- | --- | --- | --- | --- |
| **Private API TEE** (`--im api`) | **Active** | Confidential Rust worker in a reviewed **Phala Intel TDX** CVM proxying private video APIs (starting with BytePlus Seedance) | Signed reverse-auction quotes (`quoteMicrousd`, `quoteMicrousdPerSecond`, or exact `quotesByDuration`) with configurable `ATTELIER_MAX_CONCURRENCY` | Completed, settled organic requests with a quote-bound Ed25519 receipt |
| **MiniMax H3 GPU** (`--im minimax`) | **Paused** | Public HTTPS GPU worker serving a pinned checkpoint and exact video profiles | Frozen remuneration bids (`0.0001`–`1.0000`) within the 5% service ceiling | Deliveries accepted under the signed H3 gateway policy |

Key invariants:
- **Zero-work burn**: an empty API window allocates zero to miners (100% owner burn). Operator trials and synthetic simulations never earn organic rewards.
- **Reverse-auction fairness**: the gateway selects the lowest compatible available offer per request (comparing miners and direct routes) and credits merit without penalizing cost savings.
- **Hardware attestation**: API miners authenticate via real Intel TDX DCAP quotes bound to ephemeral in-memory runtime keys (`Ed25519` + `X25519`), not static API tokens.
- **Rollout safety**: a `--burn-only` primary publisher can keep renewing 100% owner burn while running `--observe-api-allocation` in the same process to verify sealed hourly API allocations in dry-run.

## Quick Links

- **[Validator Guide](docs/validator.md)** — Run a community light validator (3 commands), primary validator (`--im api` / `--im minimax`), or `--burn-only` dual-IM observer.
- **[Miner Guide](docs/miner.md)** — Set up a Phala Intel TDX Private API miner (duration-aware offers, concurrency, DCAP attestation) or review H3 GPU worker requirements.
- **[API Miner Worker Runbook](workers/api_miner/README.md)** — Rust worker CLI, Phala Cloud (`phala@1.1.22` + `dstack-0.6.5`), secret sealing, enrollment, and recovery.
- **[H3 Policy & Prelaunch Contract](docs/INFERENCE_PRELAUNCH.md)** — Exact H3 worker, ledger, and eligibility contract.
- **[H3 Gateway Architecture](docs/INFERENCE.md)** — Gateway routing and verification architecture.
- **[TAO, Gamma & AI Agent Docs](https://docs.attelier.ai/tao-gamma/)** — Use SN126 from the CLI or an AI agent with TAO or SN126 Gamma.

## Local Verification

```bash
# Python validator / core checks
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,chain,vision]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .

# Rust Private API TEE worker checks
cargo test --manifest-path workers/api_miner/Cargo.toml
```

These commands verify local code only; they do not activate a worker, publish weights, or touch the chain.

<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-mark.svg" alt="Attelier" width="56" /></a>
</p>
