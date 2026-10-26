# Guía operativa del miner

El miner publica una candidatura pública, inmutable y reproducible. No ejecuta inferencia para el validator ni entrega Python, Docker, endpoint o resultados propios: el validator descarga los bytes publicados, hace el preflight y genera sus vídeos. La referencia es Wan2.1-T2V-1.3B; se admite un LoRA compatible o un `full_transformer` compatible con el base fijado por contrato.

## Preparar una candidatura V2

Trabaja desde un clon limpio y su propio entorno virtual. Los pesos, datasets y checkpoints viven en almacenamiento remoto; este repositorio sólo guarda receta y metadatos.

```bash
cd /workspace/forge/subnet
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[chain]"
.venv/bin/python -m forge.cli model-manifest \
  --backend hf --repo ORG/MODEL --revision IMMUTABLE_REVISION \
  --role lora --base-manifest STATE/wan-base.json \
  --steps STEPS --guidance-scale CFG --scheduler uniform_euler \
  --out STATE/candidate-manifest.json
.venv/bin/python -m forge.cli prepare-submission \
  --manifest STATE/candidate-manifest.json --out STATE/submission.json
```

Publica los pesos en Hugging Face o Hippius con revisión inmutable, licencia y todas las piezas declaradas. No modifiques pesos, receta o revisión después del commitment: será otra candidatura. El manifiesto y su preparación verifican identidad y formato, no una mejora de calidad.

Puedes ejecutar el mismo preflight público que usará el validator:

```bash
.venv/bin/python -m forge.cli preflight-submission \
  --manifest STATE/candidate-manifest.json --base-manifest STATE/wan-base.json \
  --contract configs/wan-mvp.json --cache-dir REMOTE_MODEL_CACHE \
  --out STATE/preflight.json
```

Descarga bytes públicos y comprueba pesos y receta, pero no admite la candidatura, consume la hotkey, consulta duplicados ni mide calidad.

## Commitment y elegibilidad

El commitment compacto une repositorio, revisión, rol, pasos, CFG y scheduler. Una receta actual con scheduler explícito usa `forge:v4`; los formatos anteriores se resuelven sólo para replay histórico. Sus reglas de bytes y compatibilidad están en [ARTIFACTS.md](ARTIFACTS.md).

En una ventana autorizada, prepara primero el recibo local sin `--execute`:

```bash
.venv/bin/python -m forge.cli testnet-publish-model \
  --netuid NETUID --wallet WALLET --hotkey HOTKEY --out STATE/commitment.json \
  --repo ORG/MODEL --revision IMMUTABLE_REVISION --role lora \
  --steps STEPS --guidance-scale CFG --scheduler uniform_euler \
  --contract STATE/contract.json
```

Cada linaje de hotkey tiene una candidatura por ventana V2 y una candidatura pendiente impide otra simultánea. Cambiar UID, reinscribirse o renombrar archivos no evita la deduplicación de pesos, receta y scheduler. Un fallo de infraestructura conserva el mismo artefacto, receta, panel y ronda; no permite repetir una muestra científica completa.

## Qué evalúa V2

[LAUNCH_V2.md](LAUNCH_V2.md) es la regla actual. El validator usa un panel de 32 prompts en cuatro familias: ocho estáticos y 24 de movimiento entre humano, animal, objeto y paisaje. Mide además ocho timings R,C,C,R y cuatro prompts de diversidad con tres semillas. VQ, MQ y TA se comprueban por separado contra la referencia vigente y Wan-50; el límite inferior unilateral del 95% de velocidad debe ser al menos 1.10x.

La selección se fija antes de observar la entropía. El código público permite auditar la métrica, pero no autoriza al miner a aportar sus propias puntuaciones o vídeos de evaluación. Sólo consenso firmado y publicación cualificada podrían convertir una evidencia completa en un `WeightIntent`; subir, entrenar, pasar el preflight o ganar un screening no da recompensa ni pesos activos.

## Inference y documentos históricos

Inference SN126 está en prelaunch privado y permissioned de mecanismo 0. La simulación de 24 horas es local, `simulationOnly`, no firmante y sin escritura de pesos; consulta [INFERENCE_PRELAUNCH.md](INFERENCE_PRELAUNCH.md). No es un segundo mecanismo activo junto a Optimization V2.

V1, el MVP y los ejemplos 16/128/32 se mantienen exclusivamente para lectura y replay de recibos históricos. No son una vía actual de admisión, evaluación ni recompensa. Tampoco se exige Luna por nombre en V2: lo exigible es la cualificación replayable definida por el contrato V2.
