# Inference prelaunch contract

The current SN126 mechanism-0 path is a permissioned, unsigned simulation. It
reads a closed service ledger and a finalized registry projection, then emits a
review artifact. It has no wallet, signer, commit, reveal, weight write, or
publication action.

## Window and eligibility

The normal policy cadence is one UTC day (`86,400,000` ms). A policy may set
`windowMs: 3,600,000` for one-hour rehearsal windows; no other cadence is
accepted. The policy hash binds the selected cadence, worker pool, and service
inputs to each closed receipt.

Every row binds a reviewed worker to its configured UID, hotkey, worker ID, and
profile. Missing, duplicate, expired, unregistered, or malformed rows fail
closed. Eligible work completed in window W determines proposed weights for
W+1. Work that is still pending at close belongs to the window in which it
becomes terminal.

The artifact separates service merit, unsigned proposed weights, counterfactual
allocation, and expected payout. An owner-associated test row has zero expected
mining payout. Empty eligible service is represented as simulated burn. These
values are scenario accounting, not a launch, emission, revenue, or payout
promise.

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

The daily target and nominal daily amount are pro-rated exactly for an hourly
window. Consumers receive the window duration and exact rational target fields;
they must not treat an hourly artifact as a full daily allocation.
