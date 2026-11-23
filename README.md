<p align="center">
  <a href="https://attelier.ai"><img src="assets/attelier-banner.jpg" alt="Attelier" width="100%" /></a>
</p>

# Attelier Subnet · Inference

<p align="center">
  <a href="https://attelier.ai">Website</a> ·
  <a href="https://dashboard.attelier.ai">Dashboard</a> ·
  <a href="https://docs.attelier.ai">Docs</a> ·
  <a href="https://x.com/attelier_ai">X</a>
</p>

SN126 ships two incentive mechanisms for Finney M0. The operator selects one
source of weights for the primary validator. Light validators follow the
primary's finalized on-chain weights across switches.

| Mechanism | Miner provides | Eligible work |
| --- | --- | --- |
| MiniMax H3 | A GPU serving an approved checkpoint and profile | Deliveries accepted under the signed H3 gateway policy |
| Private API models | Provider access inside an attested worker, starting with BytePlus Seedance | Completed, settled organic requests with a verified receipt |

In the API mechanism the miner quotes a total price per job. For supported
equivalent routes the gateway compares miners with its own provider accounts.
An empty API window allocates zero to miners. A dedicated burn-only publisher
can keep renewing 100% owner burn during a rollout while the new weights are
calculated in dry-run.

- [Run a validator](docs/validator.md)
- [Set up a miner](docs/miner.md)
- [H3 policy and worker contract](docs/INFERENCE_PRELAUNCH.md)
- [H3 gateway architecture](docs/INFERENCE.md)

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
