# Validator guide

## Default validation: light

The intended default for the SN126 validator is **`light`**. A light validator
runs on a CPU host: it checks service accounting against the network policy and
registered miner identities, then deterministically calculates the proposed
weights. It does not serve models, download model checkpoints, rent a GPU, or
regenerate videos. It relies on the approved gateway for the reported work;
accounting checks do not independently prove model execution.

The deployment target is to install and start the validator with the network
defaults already supplied by the release: Finney, SN126, mechanism 0, the
reviewed gateway and policy, and 24-hour UTC windows. Operators should not have
to configure miner endpoints, generation profiles, reward thresholds, or the
window length. One-hour windows are for operator-controlled rehearsals.

Validation mode and weight publication are separate. **`light` must not enable
live weights by itself.** The current prelaunch only produces unsigned
simulation artifacts. A future live validator will still need the operator's
registered wallet/hotkey configuration and explicit publication activation.
That identity cannot be supplied by a shared network default. A provider API
key and a miner application are not validator configuration.

## What is available now

**This release does not yet provide the automatic SN126 validator service.**
There is no shipped `light` mode selector or install-and-run command that
fetches the required inputs and continuously calculates or publishes weights.
The default above is the deployment contract; it is not a claim that this
service is already implemented.

The available path is a manual, CPU-only rehearsal. It needs three explicit
inputs: a closed service ledger, its policy, and a finalized registry
projection. It needs no wallet or signing key and performs no chain writes.
Automatic retrieval of authenticated inputs and the service runner still need
to be delivered before deployment can be described as requiring no network
configuration.

Gateway operators review provider bindings and seal the service ledger.
Validators check those inputs and run the simulation; they do not need to
operate the gateway or approve miners themselves.

## Run the current rehearsal

Use the registry projection before a closed-window simulation:

```bash
H3_SUBNET_MODE=prelaunch PYTHONPATH=src .venv/bin/python scripts/inference/registry.py \
  --config CONFIG --out REGISTRY --simulation-registry-out SIMULATION_REGISTRY
PYTHONPATH=src .venv/bin/python scripts/inference/prelaunch.py \
  --ledger CLOSED_LEDGER --policy POLICY --registry SIMULATION_REGISTRY --out ARTIFACT
```

The registry command is read-only. The simulation is unsigned and records
`simulationOnly:true`, `publishable:false`, and no chain calls. A daily window
is the normal cadence; one-hour windows are limited to rehearsal tests. Work
counts in the window in which it becomes terminal. Eligible work from window W
determines the proposed weights for W+1; an empty eligible window produces
simulated burn rather than carrying forward the previous weights.

Check the [prelaunch contract](INFERENCE_PRELAUNCH.md) before running either
command. A later chain action would require separately authorized commit/reveal,
consensus, and finalized readback.
