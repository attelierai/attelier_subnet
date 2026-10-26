# Historical M1 market simulation

## Current prelaunch economics

The current SN126 rehearsal uses the separate
[`INFERENCE_PRELAUNCH.md`](INFERENCE_PRELAUNCH.md) artifact, not this M1 event
simulator. It evaluates one 24-hour permissioned-service ledger without a reward
signature or chain transaction. Provider onboarding can separately verify an
existing hotkey through a bounded signed ownership challenge; that is an
admission proof, not a reward signature, self-approval or publication. The
illustrative target is 600 eligible H3 videos/day and a 125 bps service budget.
These are policy hypotheses for review, not an approved emission, a spend limit,
customer revenue, or a guaranteed
alpha/USD outcome.

The artifact keeps counterfactual service allocation separate from expected
chain payout. It models empty eligible service as simulated burn, and an
owner-associated test miner's expected mining payout as zero. It does not model
Yuma consensus or infer rewards from a single validator. The following M1 model
is historical sensitivity analysis only.

`scripts/inference_market_simulation.py` es un modelo CPU de eventos discretos:
no ejecuta GPU, API, wallet, chain, billing ni dashboard.

```bash
PYTHONPATH=src .venv/bin/python scripts/inference_market_simulation.py \
  --out .codex/market-simulation-v3
```

El config explícito declara flotas `owner`/`competitor`, GPU tier, conteo,
segundos/vídeo, USD/GPU-h y evidencia `assumed` o `measured`. Explora rejillas D
(Poisson), E (unidades/día) y 0..4 GPUs por flota con llegadas comunes por seed.
`summary.json` y `sensitivity.csv` dan shares y beneficio por tier/flota.

El router es FIFO con cursor persistente round-robin sobre slots libres: con D
baja y slots ociosos alterna; bajo saturación el slot rápido vuelve antes y
sirve más. No es el modelo fluido proporcional a capacidad: ése sólo sería una
cota superior y no debe usarse como share real.

E fijo se normaliza entre vídeos elegibles completados; cero éxitos queda sin
asignar. El coste cubre GPUs hasta el drain observado. No implementa pesos,
pagos ni subasta, y la tabla D/E/count no prueba un Nash.

Las observaciones de campañas cerradas no son parámetros por defecto. Los tiers
del ejemplo son supuestos, no benchmarks ni mediciones de hardware disponible.
