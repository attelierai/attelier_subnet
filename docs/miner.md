# Provider guide

SN126 inference is currently permissioned. A provider is considered only after
operator review of its public hotkey, worker binding, profile, and expiry. A
registration or health response alone does not admit a worker.

The current prelaunch has no provider-side weight submission. Providers do not
sign reward artifacts, submit weights, or receive a payout from this workflow.
Keep worker credentials outside this repository and provide only the reviewed
binding to the operator-controlled configuration.

The worker must preserve one durable identity across a restart and return the
media and receipt data required by the prelaunch policy. Missing, expired, or
inconsistent bindings are excluded from the closed ledger rather than inferred
as successful service.

Read the [prelaunch contract](INFERENCE_PRELAUNCH.md) for the exact worker,
ledger, and eligibility fields.
