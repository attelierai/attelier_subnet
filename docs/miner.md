# Provider guide

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
