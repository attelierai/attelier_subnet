# Qualified testnet weight publication

`forge.chain.launch_publication` is the durable boundary between a qualified
competitive result and a separately supervised testnet signer. It currently
supports only `bittensor-sdk/testnet-qualified-v1`. This is an adapter mode,
not a claim that a testnet deployment is activated or ready.

Before the native signer receives a submission, SQLite stores the exact
intent and a `QualifiedPublicationContext`. The context binds the network,
genesis hash, netuid, runtime hash, mechanism, contract hash, frozen plan,
certified state, certificate when one exists, and activation-plan hash. A
context mismatch fails closed.

`QualifiedLaunchAuthority` accepts an activation only after recomputing
`activation_plan_hash` from its canonical body. The activation must name the
allowed external-GPU V2 evidence origin, contain a hash-valid
`forge/launch-policy/v2` plan, and include the raw `{plan, raw, result}`
qualification package whose result is replayed exactly. A V1 plan, synthetic
or fixture-labelled evidence, an asserted `QUALIFIED` field, or a changed
activation hash is not publishable. The origin label is provenance only; it
does not replace replay of the raw controls.

Its canonical body repeats the context bindings for network, netuid, genesis,
runtime, mechanism, contract, round-plan hash, certified-state hash and
certificate hash. The V2 candidate and king fingerprints must match the signed
round plan. This prevents an otherwise valid qualification package from being
reused for another campaign or certificate.

The worker never considers the SDK submission result sufficient. It first
persists the intent and the composed CRV3 timelocked-call ciphertext witness in
SQLite, then broadcasts. A timeout becomes `SUBMISSION_UNKNOWN`; recovery scans
from the persisted prepared-block cursor for that exact ciphertext and its
events, never blindly resubmits.

An intent becomes `FINALIZED` only after one reveal chain is proven: the unique
queued R-1 ciphertext, `TimelockedWeightsRevealed`, `WeightsSetR`, and the
effective `u16` vector in R. The target UID must still have its expected hotkey.
An old matching vector, `last_update`, or `last_step` is not evidence of this
new reveal.

For the initial no-winner state, the fallback target needs a
`BootstrapAuthority` signed by a trust anchor outside the activation JSON. That
authority binds the campaign genesis and activation plan to one UID/hotkey and
a target/runtime qualification receipt. UID `0` receives no special treatment:
it is valid only when that exact target has such qualification. The bootstrap
route is rejected if any ordinary certificate is supplied.

`forge.validator.launch_publish.publication_callback` connects a verified
certificate to the native signer for explicitly selected local testnet wallets.
It constructs and persists the activation plan, verifies public winner custody,
and checks the native fallback at a common finalized block. Every configured
validator has a separate journal and ciphertext witness. Aggregate completion
requires each validator's new vector to be effective.

The currently reviewed fallback is pinned to the official runtime v481 binary
and its exact SHA-256. `launch_fallback.qualify_fallback` reads `:code`,
`RecycleOrBurn`, `SubnetOwnerHotkey`, and the actual UID mapping. Another runtime
requires a new review. A matching UID alone does not qualify it.

`forge launch preflight` reports `native_fallback.status` separately from its
read-only identity verification. Require `QUALIFIED` before allocating a worker
for acceptance. Both `launch run --mode live` and `launch consensus --mode live`
reject an unqualified native fallback before accessing wallets or dispatching
work. Matching call names in upgraded metadata alone does not establish the
runtime's burn behavior; the supported binary and source proof must be reviewed.

`JsonHttpPublicationAdapter` remains an optional client transport. There is no
authenticated HTTP signer service in this change. The CLI uses the native
adapter; no endpoint URL or token is required.

`BittensorSdkTestnetSigner` persists the SDK-composed timelocked call before
submission and uses the local native reader for recovery and finalization. It
does not rely on injected RPC stubs or `BittensorChain.set_weights`.

`TestnetNativePublicationReader` verifies testnet genesis, SHA-256 of runtime
`:code`, raw finalized extrinsics (hash, signer, module/function, netuid,
mechanism and ciphertext), and the specific reveal/event/vector sequence.
Missing ciphertext identity remains unknown.
