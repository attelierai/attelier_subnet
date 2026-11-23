# Miner guide

Choose the mechanism your worker can serve. Both require a registered SN126
hotkey and gateway admission. The same validator receives the weight vector.

## Private API models · BytePlus Seedance

1. Prepare a funded provider account authorized to serve the selected model.
   The miner runs a small CPU worker in a reviewed Phala TEE. BytePlus runs Seedance.
2. Follow the [attested worker setup](../workers/api_miner/README.md#operator-quick-start)
   to create the persistent identity, enroll the worker and bind its reviewed
   runtime to your hotkey and allowed model/modalities.
3. Set `ATTELIER_PROVIDER=byteplus`, the BytePlus key and `ATTELIER_MINER_OFFERS`
   in the encrypted deployment. For example, this quotes USD 0.50 per job:

   ```json
   [{"model":"dreamina-seedance-2-5-260628","modality":"text-to-video","quoteMicrousd":500000}]
   ```

4. Start the worker with `attelier-api-miner run`. It renews its signed offer
   while polling. For Seedance 2.5 text-to-video it serves 720p, five or ten
   seconds, landscape or portrait. The same total price covers either duration.
   First-frame requests use a separate image-to-video offer with adaptive ratio.

The gateway picks the lowest compatible available offer or direct provider
quote for each organic request. It pins the winner and price at quote time.
A completed job needs validated output, a signed receipt and settlement before
it can contribute to the miner's share. A bid is a service valuation in USD
micros. The subnet distributes emissions through weights. Idle time, an empty
window and operator trials earn zero in this mechanism. Hosting can still cost
money while the worker waits.

## MiniMax H3 GPU workers

SN126 inference is currently permissioned. A provider is considered only after
operator review of its public hotkey, worker binding, profile, and expiry. A
registration or health response alone does not admit a worker.

Start at [Providers](https://providers.attelier.ai): verify the hotkey, apply for
approval, select the published checkpoint and a nonempty set of profiles, then
install the offered worker package with its scoped credential. Register the
worker's authenticated public HTTPS endpoint and keep it active. Credentials
belong in protected local worker state, never this repository or shell logs.

The gateway retries approved active workers every minute. Once health,
registration, identity, checkpoint and selected profiles match, it binds the
worker automatically; installation delays do not revoke application approval.
New coverage may wait for the next frozen UTC-hour policy. Explicit pauses are
respected. A successful timely synthetic delivery must prove each profile before
ordinary organic requests can use it; healthy status alone is insufficient.

Providers do not submit weights or sign reward artifacts. Validators calculate
and publish them from the signed gateway feed. Eligible service can receive
chain rewards, whose effective amount depends on consensus; gateway accounting
alone never proves settled payment.

The worker must preserve one durable identity across a restart and return the
media and receipt data required by the prelaunch policy. Missing, expired, or
inconsistent bindings are excluded from the closed ledger rather than inferred
as successful service. A deterministic `pixel_frames` operator proposal may set
all 18 canonical tuples without a GPU benchmark; its weights are duration ×
pixels normalized to the 5-second 480p landscape baseline = 1,000. A
`reference_gpu` basis remains optional and must retain its reviewed evidence.
Install the worker version and immutable artifacts supplied by the selected
checkpoint's bootstrap; do not infer current runtime requirements from a
historical signed policy version. The worker's technical timeout is separate
from each profile's customer execution and delivery deadlines.

The current live-policy contract admits at most one non-revoked worker for each
application/hotkey; pausing does not release that binding. It accounts eligible
completed exact-profile work in indefinite one-hour windows with a 5% service
ceiling. Frozen remuneration bids discount each miner's share. V5 organic work
earns twice equivalent synthetic reward units within that ceiling. Cancelled,
late or invalid work earns none. Continuous synthetic demand targets equal
admission counts across published profiles, subject to compatible capacity;
it promises neither traffic nor equal GPU time per worker. Organic work has
priority, and synthetic cancellation retains its slot until release is confirmed.

During partial outages or empty windows, validators can retain the latest
positive allocation for currently available workers using fresh signed
availability, bounded by the original source deadline. Without eligible work
or valid continuity, unused service allocation burns. Later recovery never
backfills missed windows. See the validator guide for the continuity limits.

Read the [prelaunch contract](INFERENCE_PRELAUNCH.md) for the exact worker,
ledger, and eligibility fields.
