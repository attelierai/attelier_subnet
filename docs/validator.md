# Validator Guide — Attelier (Bittensor SN126)

Set weights on **Attelier (Subnet 126, Finney Mechanism 0)** in under 2 minutes.

Every hour, the Attelier gateway seals an Ed25519-signed allocation over a **6-hour rolling window (5 Finney tempos)** using dynamic cap-and-burn against live SN126 miner emissions:
- Paid organic inference deliveries earn emissions proportional to their winning reverse-auction quote.
- Any unused emission capacity burns to **UID 0** (100% burn when organic demand is zero).
- Your validator fetches `https://inference.attelier.ai/v1/subnet/126/final-allocation/latest`, verifies the pinned Ed25519 signature, Finney genesis, SN126/M0 namespace, and finalized registry block on chain, then submits the exact CRv4 u16 weight vector — **zero GPU, zero API keys, zero provider secrets, and zero 1-tempo weight-copy lag**.

---

## Prerequisites

- **Docker** with the `docker compose` v2 plugin (recommended), **or** **Python 3.12+** with **OpenSSL 3.0+**
- A Bittensor validator hotkey registered on **Finney netuid 126** with validator permit
- A small CPU host (1 vCPU / 512 MB RAM is enough)

---

## Option 1 — Docker Compose (Recommended, 3 steps)

**1. Clone the repository:**

```bash
git clone https://github.com/attelierai/attelier_subnet.git
cd attelier_subnet
```

**2. Create `.env` and set your wallet & hotkey names:**

```bash
cp .env.validator.example .env
# edit .env -> set WALLET_NAME and WALLET_HOTKEY (and HOST_WALLETS_DIR if not ~/.bittensor/wallets)
```

**3. Start detached (`restart: unless-stopped`):**

```bash
docker compose -f docker-compose.validator.yml up -d --build
docker compose -f docker-compose.validator.yml logs -f validator
```

That's it. Your wallet directory is mounted read-only (`/wallets:ro`), your SQLite publication journal persists in `./state/validator`, and the container survives SSH disconnects and reboots.

Manage it with standard Compose commands:

```bash
docker compose -f docker-compose.validator.yml ps
docker compose -f docker-compose.validator.yml logs -f
docker compose -f docker-compose.validator.yml down
```

---

## Option 2 — Native Python + PM2 (3 steps)

**1. Install and run a walletless dry-run check:**

```bash
git clone https://github.com/attelierai/attelier_subnet.git
cd attelier_subnet
./scripts/install-validator.sh
```

**2. Start live weight publication with PM2:**

```bash
pm2 start .venv/bin/attelier-validator \
  --name attelier-sn126-validator --interpreter none -- \
  --role validator --im api \
  --wallet.name <WALLET> --wallet.hotkey <HOTKEY> \
  --state-dir "$PWD/state/validator"
pm2 save
pm2 logs attelier-sn126-validator
```

*(Prefer systemd? Use [scripts/inference/attelier-light.service.example](../scripts/inference/attelier-light.service.example) and set `--role validator --im api`.)*

---

## Walletless dry-run check (verify before publishing)

Inspect the exact verified weight vector without loading any wallet or signing anything:

```bash
.venv/bin/attelier-validator --role validator --im api --dry-run --once --state-dir ./state/check
cat ./state/check/live-dry-run.json
```

`live-dry-run.json` shows `allocationShares` (sum to `65,535`) and `relativeWeights` (max-normalized u16 vector sent to Finney M0).

---

## Keep it updated (autoupdate)

Run the autoupdate loop to track `origin/main` every 5 minutes and rebuild/restart automatically when new commits land:

```bash
# Docker (default):
./scripts/autoupdate-validator.sh

# Or PM2:
AUTOUPDATE_MODE=pm2 ./scripts/autoupdate-validator.sh
```

Overrides: `AUTOUPDATE_INTERVAL` (seconds, default `300`), `AUTOUPDATE_BRANCH` (default `main`), `PM2_APP_NAME`.

---

## Validator modes & environment variables

| Mode | Flags / Env | What it does |
| --- | --- | --- |
| **Direct API Validator (Recommended)** | `--role validator --im api` (`VALIDATOR_ROLE=validator`) | Verifies the signed SN126 allocation feed + finalized chain registry directly and sets weights in the **same tempo** (zero copy lag). |
| **Light Copy Validator** | `--role light` (`VALIDATOR_ROLE=light`) | Copies the primary validator's finalized on-chain M0 weight row (1-tempo commit-reveal lag). |
| **Primary Operator** | `--role primary --im api` | Pinned to SN126's primary validator hotkey (`5FZD47Wh...`). |

Supported environment variables (CLI flags override env vars):

| Variable | Default | Description |
| --- | --- | --- |
| `WALLET_NAME` | — | Bittensor coldkey wallet name (required unless `--dry-run`) |
| `WALLET_HOTKEY` | — | Bittensor hotkey name (required unless `--dry-run`) |
| `WALLET_PATH` | `~/.bittensor/wallets` | Wallet directory (`/wallets` in Docker) |
| `BITTENSOR_NETWORK` | `finney` | `finney` or custom Subtensor RPC (`ws://...` / `wss://...`) |
| `STATE_DIR` | `/var/lib/attelier-light` | Durable SQLite journal and status directory (`/state` in Docker) |
| `LOCK_DIR` | `<STATE_DIR>/publisher-locks` | Single-writer lock directory (never run two writers for the same hotkey) |
| `INTERVAL_SECONDS` | `60` | Polling cadence in seconds |

---

## Custom Subtensor RPC

Set `BITTENSOR_NETWORK=wss://your-node.example.com:9944` in `.env` (or pass `--network wss://...`). Your node must retain state for the feed's finalized registry block (within the last hour). Test without a wallet first:

```bash
.venv/bin/attelier-validator --role validator --im api \
  --network wss://your-node.example.com:9944 \
  --dry-run --once --state-dir ./state/rpc-check
```

---

## Checking status & recovery

Inspect these files inside `STATE_DIR` (`./state/validator` by default):

| File | Meaning |
| --- | --- |
| `status.json` | Current validator status (`ready` / `unavailable`), last verified `sequence`, and digest |
| `latest-final-allocation.json` | Last accepted signed allocation body from Attelier |
| `light.sqlite3` | Durable commit/reveal publication journal (`PREPARED` → `COMMITTED` → `REVEAL_PENDING` → `REVEALED`) |

- **Upgrading:** always preserve `STATE_DIR` (`light.sqlite3`) across upgrades so pending CRv4 commits reconcile cleanly without duplicate submissions.
- **Safety:** never run two live validator processes for the same hotkey on the same or different hosts.
