# Gateway architecture and trust boundary

The H3 Gateway accepts work only for reviewed, permissioned worker bindings. It
records the selected worker and request context before dispatch, keeps terminal
ledger rows durable, and separates service evidence from any customer or chain
accounting.

The prelaunch registry projects finalized public identity into the reviewed
worker set. A worker must match its configured UID, hotkey, profile, and review
expiry. Missing or changed bindings fail closed for the next ledger window.

Gateway accounting has no validator wallet or chain publication authority.
The light validator independently checks its signed feed and finalized registry,
then publishes M0 weights by default with the operator's named wallet and hotkey.
`--dry-run` performs the same accounting without signing or submitting weights.
Simulation artifacts remain unsigned; they are not proof of a chain write.

The gateway treats an uncertain accepted request as the same durable request on
recovery; it does not create a replacement job. A closed window contains only
terminal work. Later terminal work belongs to the next period.

This boundary does not prove GPU ownership, checkpoint execution, availability,
or a provider payout. Those claims require separate evidence and authorization.
See [INFERENCE_PRELAUNCH.md](INFERENCE_PRELAUNCH.md) for the accounting contract
and [validator.md](validator.md) for publication and dry-run commands.
