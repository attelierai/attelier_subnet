# Inference prelaunch contract

The SN126 mechanism-0 light validator publishes weights by default using the
wallet named by `--wallet.name` and `--wallet.hotkey`. It reads a closed service
ledger and a finalized registry projection, calculates the vector, and journals
the exact encrypted M0 commit-reveal call before submit. Publication
requires finalized inclusion plus a reveal/weight readback before success can
be claimed. `--dry-run` selects unsigned simulation without a wallet or chain
writes. Old `--live` commands remain valid, but that flag is no longer needed.

## Window and eligibility

Historical shadow accepts one UTC day or hour; the current live contract accepts
exactly one UTC hour (`3,600,000` ms) indefinitely. The policy hash binds the cadence, worker pool, fixed
mining-pool share, signed network target, and service inputs to each closed
receipt. Live has a 500 bps service ceiling and at least 9,500 bps burn;
one eligible completion can contribute. Empty signed windows are valid; without
bounded continuity their unused service share also burns.

Every row binds a reviewed worker to its configured UID, hotkey, worker ID, and
profile. A hotkey/application has at most one non-revoked worker; pausing does
not free that binding, revocation does. Missing, duplicate, expired,
unregistered, or malformed rows fail closed. Eligible work completed in window
W determines proposed weights for W+1. Work that is still pending at close
belongs to the window in which it becomes terminal.

New policies may set `gatewayPolicy.rewardEligibility` to `per_job`: each
profile, custody, and TTV-eligible completion then contributes independently.
The historic interpretation remains when the field is absent: the aggregate
must meet its configured success threshold. Unknown values fail closed.

For signed v2 policy activation, each approved profile records `rewardBasis`
(`pixel_frames` or `reference_gpu`) and an operator `approvalHash`. The
deterministic `pixel_frames` basis uses duration × pixels normalized to the
5-second 480p landscape baseline = 1,000 and does not require a GPU benchmark.
The benchmark-backed `reference_gpu` basis is optional. The gateway must admit
only workers reporting protocol version 2 and runtime version 1.2.0; all v1
history remains unchanged.

The artifact separates service merit, unsigned proposed weights, counterfactual
allocation, and expected payout. The current live ceiling is 500 bps
(5%) of the mining pool. Without valid continuity, empty eligible service is a
full burn. If no verified feed is available after the 10-minute grace and
continuity is unavailable, that window is also burn, and the
next valid window resumes without retroactive accounting. An owner-associated
shadow row has zero expected mining payout. Historical 10% shadow artifacts are
kept only for reproducible replay. Simulation artifacts themselves are unsigned;
the live publisher separately validates and submits the calculated vector.

## V4 hourly service auction

`attelier/prelaunch-policy/v4`, `attelier/prelaunch-ledger/v4`, and the signed
`attelier/light-input/v4` / `attelier/light-feed/v4` keep the same 500 bps (5%)
service ceiling. They do not promise that the whole ceiling is paid: its unused
part is burned. A dispatch freezes its integer `remunerationBps` in the range
1..10,000 alongside its exact checkpoint/duration/resolution/aspect profile.
It cannot be changed by a later hourly policy or worker update.

For each eligible, timely and custodied exact-profile completion, the ledger
records both undiscounted `eligibleWorkUnits` and the integer numerator
`eligibleWeightedWorkUnitsBps = eligibleWorkUnits * remunerationBps`. For a
closed hour, with `E = 65,535`, raw total `R`, and weighted total `W`, the
service allocation is `floor(E * 500 / 10,000 * W / (R * 10,000))`. The
deterministic residual, the 95% non-service share, and a zero-eligible hour all
burn; discounted work is never renormalized back to 5%. Per-miner allocation is
the deterministic largest-remainder split of that allocated service amount by
its weighted numerator. This remains planning/accounting, not a claim of an
on-chain payout or finalized weight publication.

Admission policy snapshots are sealed with the job. A V4 close accepts terminal
work admitted under preserved V2, V3, or V4 snapshots and validates the original
worker/profile binding; historic rows without an explicit frozen bid are
interpreted only as the legacy 10,000 bps bid. Customer request priority
(`1`, `0`, `-1`) is gateway scheduling only: it is neither sent to workers nor a
reward-work multiplier. Quality remains exact-profile policy data and is
additive: `attelier/h3-quality/v1` records the required outcomes, success rate,
observation/evidence window, cooldown, duration tolerance, optional visual
threshold, calibration state, and reference job. It never introduces a GPU
requirement.

## V5 organic reward units

The current V5 policy, ledger and signed feed preserve the V4 service ceiling
and frozen remuneration bids. `organicRewardMultiplierBps` is exactly `20,000`.
Organic work admitted under V5 earns twice the reward units of equivalent
synthetic work. Aggregates bind their traffic kind and admission policy; work
admitted under V2–V4 keeps its original multiplier even if it finishes in V5.
The boosted units enter both the service allocation denominator and each
miner's discounted numerator. They never increase the 5% ceiling or turn failed,
late or cancelled work into eligible service. Historical V1–V4 inputs remain
readable; V4-only readers cannot verify a V5 envelope.

## Registry projection

Build the read-only registry projection before the simulation:

```sh
H3_SUBNET_MODE=prelaunch PYTHONPATH=src .venv/bin/python scripts/inference/registry.py \
  --config CONFIG --out REGISTRY --simulation-registry-out SIMULATION_REGISTRY
```

The projection reads one finalized snapshot and includes only enabled workers
whose configured identity and review expiry match that snapshot. A local
`--chain-snapshot` fixture exercises the projection code but does not establish
current chain state.

## Checkpoint catalog profiles

`attelier/subnet-h3/v3` configurations bind every profile to a published,
operator-owned checkpoint from `src/forge/inference/h3-checkpoint-catalog.json`.
The validator recomputes the immutable artifact-manifest hash and the same
service-profile-v2 identity used by the gateway; it never accepts a miner
checkpoint claim. Set `H3_CHECKPOINT_CATALOG_FILE` only to a reviewed native
catalog snapshot with the same schema when publishing a newer catalog release.
New v3 selections require a published checkpoint. Frozen v2 policy receipts may
still validate a catalog identity that has since been retired, so withdrawal
does not reinterpret history.

## Closed-ledger simulation

```sh
PYTHONPATH=src .venv/bin/python scripts/inference/prelaunch.py \
  --ledger CLOSED_LEDGER --policy POLICY \
  --registry SIMULATION_REGISTRY --out ARTIFACT
```

The output always records `simulationOnly:true`, `publishable:false`, and an
empty chain-call list. It is created atomically and accepts only a
byte-identical retry, so a changed input cannot overwrite an existing receipt.

The light runner fetches only the reviewed signed endpoint, verifies the pinned
public signature and its finalized registry readback, and persists its local
review state. Default publication requires wallet/hotkey names and independently
checks Finney/SN126/M0, CRv4,
validator permit, burn UID0 owner identity and `RecycleOrBurn=Burn` before the
SDK may compose a weight commit. `--dry-run` keeps all reads and accounting but
never loads a wallet or submits a transaction.

With `--continuity-max-hours 2` (or `6`), an empty live window or transport/age
failure can preserve the latest positive V4/V5 allocation until that source
window's end plus the configured limit. The separate signed
`/v1/subnet/126/light/availability` snapshot expires after 120 seconds and binds
recent authenticated health, approval, active state, worker identity and current
registration. Bad signatures, identity faults or absent/stale availability do
not authorize continuity. Down, paused, revoked or replaced workers lose their
share to burn; remaining shares retain their frozen bid discounts and organic
multiplier within the original 5% ceiling. Heartbeats and restarts never extend
the source deadline. The accepted work ledger remains unchanged; continuity has
its own input/artifact schemas and `continuity-status.json`, and uses the same
durable publication journal and final chain identity checks. The default is
`2`; an explicit `0` disables continuity.
