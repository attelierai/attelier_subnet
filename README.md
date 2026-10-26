<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-banner.jpg" alt="Attelier" width="100%" /></a>
</p>

# Attelier Subnet · Inference prelaunch

<p align="center">
  <a href="https://attelier.ai">Website</a> ·
  <a href="https://dashboard.attelier.ai">Dashboard</a> ·
  <a href="https://docs.attelier.ai">Docs</a> ·
  <a href="https://x.com/attelier_ai">X</a>
</p>

SN126 mechanism 0 is currently a permissioned H3 Gateway rehearsal. It closes
service ledgers, validates the reviewed worker set, and produces an unsigned
local simulation artifact.

The current state is **simulated prelaunch**: weights are off. The artifact is
`simulationOnly`, non-publishable, and performs no chain writes. A 24-hour UTC
window is normal; a one-hour window exists for rehearsal testing. Eligible work
completed in window W determines proposed weights for W+1. Pending work belongs
to the window in which it becomes terminal.

Service merit and payout are separate. An owner-associated row has zero expected
mining payout. Any future chain result requires its own authority, commit/reveal,
consensus, and finalized readback.

## Start here

- [Prelaunch contract](docs/INFERENCE_PRELAUNCH.md)
- [Provider guide](docs/miner.md)
- [Validator guide](docs/validator.md)
- [Gateway architecture and trust boundary](docs/INFERENCE.md)

## Local checks

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,chain,vision]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

These commands verify local code only; they do not activate a worker, weight, or
chain action.

<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-mark.svg" alt="Attelier" width="56" /></a>
</p>
