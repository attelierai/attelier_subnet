# Provider guide

SN126 inference is currently permissioned. A provider is considered only after
operator review of its public hotkey, worker binding, profile, and expiry. A
registration or health response alone does not admit a worker.

The current prelaunch has no provider-side weight submission. Providers do not
sign reward artifacts, submit weights, or receive a payout from this workflow.
Keep worker credentials outside this repository and provide only the reviewed
binding to the operator-controlled configuration.

The standalone miner archive `1.1.0` is a prepared candidate and has not been
published. Any additional license needed for a provider is confirmed by the
operator. Neither status establishes completed model verification.

The worker must preserve one durable identity across a restart and return the
media and receipt data required by the prelaunch policy. Missing, expired, or
inconsistent bindings are excluded from the closed ledger rather than inferred
as successful service.

The current live-policy contract admits at most one non-revoked worker for each
application/hotkey; pausing does not release that binding. It accounts eligible
completed video seconds in indefinite one-hour windows at a fixed 5% service
share. Trial and ramp work is not discounted. Synthetic demand is network-global
at 2, 5, then 10 requests/hour by available phase, not a per-worker promise.
An empty window, or a missing/unavailable feed after its 10-minute grace, is
burn; the next valid window resumes without backfill. This is accounting policy,
not evidence that weights have been published or that model execution was proven.

Read the [prelaunch contract](INFERENCE_PRELAUNCH.md) for the exact worker,
ledger, and eligibility fields.
