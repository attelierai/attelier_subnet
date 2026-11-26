# Miner Guide (SN126)

Attelier SN126 supports **two miner lanes** feeding Finney Mechanism 0:
1. **Private API TEE (`api_miner`, Active)** — Lightweight CPU worker running inside a reviewed **Phala Intel TDX** enclave, supporting the full frontier video & image catalog (Seedance, Kling 4.0/3/2.6, Veo 3.1, MiniMax H3, Wan 3.0, FLUX 3 Max, Nano Banana, FLUX 2 Flash, Seedream 5 Lite) via signed reverse-auction offers. Production rolls out **Seedance first** (`enabledModels` gateway allowlist) and enables additional families incrementally with zero worker re-attestation.
2. **MiniMax H3 GPU (Paused)** — Dedicated GPU workers serving pinned open-weights checkpoints under reviewed profiles.

Both lanes require a registered Finney SN126 hotkey and operator review before receiving routed traffic. Monitor live status, attestation expiry, signed offers, 24h activity, and sealed hourly rewards in the **[Provider Console](https://console.attelier.ai/manage)** and **[Providers Dashboard](https://providers.attelier.ai)**.

---

## Lane 1: Private API Models (Phala Intel TDX) · Active

### How it works

Instead of hosting GPUs locally, an API miner runs the open-source Rust worker ([`workers/api_miner/`](../workers/api_miner/README.md)) inside a Phala Intel TDX confidential VM (e.g. `tdx.small`) with its own funded upstream provider credentials sealed inside the enclave.

- **Hardware-rooted trust**: on every start, the worker generates fresh in-memory `Ed25519` (receipts) and `X25519` (encrypted assignment payloads) keys and binds them into a real Intel TDX DCAP quote verified by Attelier's pinned DCAP verifier.
- **Duration-aware reverse auction**: the worker publishes signed offers every 2 minutes (5-minute TTL). For each organic customer request, the gateway selects the lowest compatible available bid (comparing miners and direct routes) and pins the winning bid at quote time.
- **Fair reference merit**: bidding lower wins more traffic without penalizing merit per unit of served work. Idle time, empty hours, failed deliveries, and operator trials earn zero organic weight.
- **Concurrency**: set `ATTELIER_MAX_CONCURRENCY` (e.g. `2`–`8`) to process multiple upstream video jobs concurrently within one CVM.

### Step-by-step setup

1. **Register & apply**: register your hotkey on Finney SN126, verify it in [Provider Console](https://console.attelier.ai/manage), and request operator review for your provider account, model IDs, and Phala deployment. Possessing an upstream API key alone does not grant admission.
2. **Prepare the Phala deployment**: follow the full [API Miner Operator Quick Start](../workers/api_miner/README.md#operator-quick-start) to:
   - Pin the reviewed worker image digest in `docker-compose.phala.yml` (with the named state volume `attelier-api-miner-state` mounted at `/data/attelier-api-miner`).
   - Include `--pre-launch-script phala-pre-launch.sh` (required on `dstack-0.6.5` for read-only dstack socket permissions for non-root UID `10001`).
   - Generate the persistent control SPKI (`attelier-api-miner identity`) and upload the one-shot `attelier/api-miner-enrollment/v2` evidence envelope to the private enrollment collector.
3. **Seal credentials and offers in `.phala.env`** (never in Git, shell args, or logs):
   - Set `ATTELIER_PROVIDER=byteplus` and your `BYTEPLUS_API_KEY`.
   - Set `ATTELIER_MAX_CONCURRENCY` (default `1`, recommended `2`–`4` depending on upstream rate limits).
   - Set `ATTELIER_MINER_OFFERS` using any of the three supported pricing modes (all amounts in integer USD micros, where `1,000,000` = `$1.00`):

   **Precedence per request:** `quotesByDuration[seconds]` → `quoteMicrousdPerSecond * seconds` → flat `quoteMicrousd`.

   ```json
   [
     {
       "model": "dreamina-seedance-2-5-260628",
       "modality": "text-to-video",
       "quoteMicrousdPerSecond": 20000,
       "quotesByDuration": { "5": 95000, "10": 180000 }
     },
     {
       "model": "dreamina-seedance-2-5-260628",
       "modality": "image-to-video",
       "quoteMicrousdPerSecond": 20000,
       "quotesByDuration": { "5": 95000, "10": 180000 }
     }
   ]
   ```

   *(A flat per-job `quoteMicrousd`—for example `500000` = `$0.50`—also remains supported and applies to any supported duration when duration-specific fields are omitted.)*
4. **Run and verify**: the compose entrypoint runs `attelier-api-miner run`. It attests on startup, renews attestation before expiry, publishes signed offers every 2 minutes, and processes up to `ATTELIER_MAX_CONCURRENCY` jobs in parallel.
   - Check live attestation status, `review_expires`, `inFlight / maxConcurrency`, active signed offers, `activity.last24h` (completed, failed, video seconds), and sealed hourly `apiMechanism` weight share in [Provider Console → Workers & Rewards](https://console.attelier.ai/manage).

### Operational & economic rules

- **Durability & zero double-spend**: the worker records `(job, attempt)` in durable state before submitting upstream. On restart or timeout, it reconciles via gateway `recover` against the canonical upstream reference instead of issuing a second paid upstream call.
- **Review expiry**: every admitted API miner carries an operator `review_expires` timestamp. Renew before expiry so the worker does not transition to `blocked`.
- **No guaranteed payout**: bids are USD micro service valuations. Subnet emissions are distributed via finalized Finney weights and depend on chain consensus; hosting a CVM incurs compute costs even during empty hours.

---

## Lane 2: MiniMax H3 GPU Workers · Currently Paused

> **Status:** H3 GPU admission and organic H3 routing are currently **paused** during the Dual-IM rollout while the Private API TEE lane is active. Existing H3 policy and accounting contracts remain intact below for reference and future re-enablement.

SN126 H3 inference is permissioned. A provider is considered only after operator review of its public hotkey, worker binding, profile, and expiry. Registration or a healthy endpoint alone does not admit a worker.

1. Start at [Providers](https://providers.attelier.ai) / [Provider Console](https://console.attelier.ai/manage): verify the hotkey, apply for approval, select the published checkpoint and a nonempty set of profiles, then install the offered worker package with its scoped `h3w_` credential.
2. Register the worker's authenticated public HTTPS endpoint and keep it active. Credentials belong in protected local worker state, never in this repository or shell logs.
3. The gateway reconciles approved active workers every minute. Once health, registration, identity, checkpoint, and selected profiles match, it binds the worker automatically; new coverage joins at the next frozen UTC-hour policy. Explicit pauses are respected.
4. A successful timely synthetic delivery must prove each profile before ordinary organic requests can use it; healthy status alone is insufficient.

### H3 accounting summary

- **One worker per hotkey**: the live-policy contract admits at most one non-revoked H3 worker per application/hotkey (pausing does not release that binding). TEE API workers have separate attestation-enrolled concurrency and do not consume H3 slots.
- **Hourly windows & 5% ceiling**: eligible completed exact-profile work is accounted in one-hour UTC windows with a 5% service ceiling. Frozen remuneration bids discount each miner's share; V5 organic work earns twice equivalent synthetic reward units within that ceiling. Cancelled, late, or invalid work earns none.
- **Pixel-frames basis**: deterministic `pixel_frames` weights equal `duration × pixels` normalized to the 5-second 480p landscape baseline (`= 1,000`).
- **Continuity & burn**: during partial outages or empty windows, validators may retain the latest positive allocation for currently available workers using fresh signed availability up to the source deadline; without eligible work or valid continuity, unused allocation burns and is never backfilled.

Read the [prelaunch contract](INFERENCE_PRELAUNCH.md) for the exact H3 worker, ledger, and eligibility fields.
