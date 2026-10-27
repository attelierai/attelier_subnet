# Inference prelaunch contract

The default SN126 mechanism-0 path is a permissioned, unsigned simulation. It
reads a closed service ledger and a finalized registry projection, then emits a
review artifact. Explicit `--live` is a separately wallet-gated M0
commit-reveal path; it journals the exact encrypted call before submit and
requires finalized inclusion plus a reveal/weight readback before success can
be claimed. No current publication is implied by this capability.

## Window and eligibility

Historical shadow accepts one UTC day or hour; the current live contract accepts
exactly one UTC hour (`3,600,000` ms) indefinitely. The policy hash binds the cadence, worker pool, fixed
mining-pool share, signed network target, and service inputs to each closed
receipt. Live is a fixed 500 bps service pool and 9,500 bps owner-burn split;
one eligible completion is enough. Empty signed windows are valid and burn
100%.

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

The artifact separates service merit, unsigned proposed weights, counterfactual
allocation, and expected payout. The current live share is fixed at 500 bps
(5%) of the mining pool; empty eligible service is a full burn. If no verified
feed is available after the 10-minute grace, that window is also burn, and the
next valid window resumes without retroactive accounting. An owner-associated
shadow row has zero expected mining payout. Historical 10% shadow artifacts are
kept only for reproducible replay. These values are simulation accounting, not a
chain instruction.

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
review state. Default mode remains shadow/read-only. Explicit `--live` requires
wallet/hotkey arguments and independently checks Finney/SN126/M0, CRv4,
validator permit, burn UID0 owner identity and `RecycleOrBurn=Burn` before the
SDK may compose a weight commit.
