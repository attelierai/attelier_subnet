# Gateway architecture and trust boundary

The H3 Gateway accepts work only for reviewed, permissioned worker bindings. It
records the selected worker and request context before dispatch, keeps terminal
ledger rows durable, and separates service evidence from any customer or chain
accounting.

The prelaunch registry projects finalized public identity into the reviewed
worker set. A worker must match its configured UID, hotkey, profile, and review
expiry. Missing or changed bindings fail closed for the next ledger window.

The current accounting path is local and unsigned. It has no wallet, signer,
commit, reveal, weight write, or reward payment. Its output is an input to human
review, not a chain instruction.

The gateway treats an uncertain accepted request as the same durable request on
recovery; it does not create a replacement job. A closed window contains only
terminal work. Later terminal work belongs to the next period.

This boundary does not prove GPU ownership, checkpoint execution, availability,
or a provider payout. Those claims require separate evidence and authorization.
See [INFERENCE_PRELAUNCH.md](INFERENCE_PRELAUNCH.md) for the executable
prelaunch contract.
