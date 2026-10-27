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

SN126 mechanism 0 is a permissioned H3 Gateway shadow prelaunch. The default
CPU validator runs as `.venv/bin/attelier-validator`, using release-pinned
public defaults; it needs no wallet or operator public-key configuration. It
checks the signed gateway feed and finalized public registry, then produces an
unsigned local simulation artifact.

The default state is **simulated prelaunch**: actual chain publication is not
confirmed. Default operation is shadow/read-only and its artifact is
`simulationOnly`, non-publishable, and performs no chain writes. The release
also contains an explicit, wallet-gated `--live` M0 commit-reveal path; its
availability does not mean that a commit, reveal, or finalized weight readback
has happened.
The current live-policy contract uses indefinite one-hour UTC windows. Eligible
work completed in window W determines accounting for W+1; pending work belongs
to the window in which it becomes terminal.

The signed fixed policy assigns exactly 5% of the mining pool to eligible work;
the remaining 95% is burn. An empty signed hourly window is a full burn. A
missing or unavailable feed after the 10-minute grace is burn for that window;
the next valid window resumes normally, with no retroactive catch-up. Initial
network-global synthetic demand is capped at 2 requests/hour while only trial
workers exist, 5 with ramp capacity, and 10 with active capacity. These are
network-global caps, not per miner. An owner-associated row has zero expected
mining payout. Historical daily and 10% shadow artifacts remain replay history,
not the current live policy.

New signed policies use per-job eligibility: a completion already eligible for
the configured profile, custody, and TTV contributes on its own. Policies
without that field retain the historic aggregate success-threshold rule.

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
