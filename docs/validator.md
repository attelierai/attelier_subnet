# Validator operativo

Un validator descarga y ejecuta únicamente artefactos públicos y recetas congeladas. El miner entrega un manifiesto; no entrega código, contenedor, GPU ni resultados que el validator acepte como propios. La referencia inicial es Wan2.1-T2V-1.3B con la receta Wan-50. El candidato se mide contra el king para velocidad y contra el king y Wan-50 para calidad.

## Regla actual: Launch V2

[LAUNCH_V2.md](LAUNCH_V2.md) es el contrato vigente y acotado de testnet para la competición de optimización. No activa mainnet ni publica pesos por el mero hecho de que el código, una prueba local o una simulación pasen.

Antes de abrir una ronda, el validator debe:

1. Reconstruir comité, compromisos y bloque de entropía desde cadena finalizada, y guardar el plan y checkpoint de admisión antes de revelar prompts.
2. Ejecutar el preflight público de licencia, hashes, tensores y receta. Es CPU/red: no acepta endpoints privados ni resultados del miner.
3. Reservar el presupuesto completo y conservar el mismo artefacto, receta, panel, semillas y evidencia ante una recuperación de infraestructura.
4. Verificar una cualificación replayable del evaluador. La V2 actual usa cuatro familias y doce vídeos de cualificación (cuatro de desarrollo y ocho de confirmación); no hace de Luna un requisito obligatorio ni adopta por nombre un proveedor de evaluación.

El panel competitivo V2 contiene 32 prompts: ocho estáticos y 24 de movimiento, repartidos entre las familias humana, animal, objeto y paisaje. Añade ocho timings R,C,C,R y cuatro prompts de diversidad con tres semillas. VQ, MQ y TA se deciden por separado contra la referencia vigente y Wan-50; el límite inferior unilateral del 95% de velocidad debe llegar a 1.10x. Consulta [LAUNCH_V2.md](LAUNCH_V2.md) para márgenes, controles, deriva determinista y reglas exactas de `PASS`, `FAIL`, `INCONCLUSIVE` e `INCOMPLETE`.

```bash
.venv/bin/python -m forge.cli competitive-open \
  --db STATE/competitive.sqlite --plan ROUND.json --state STATE/head.json \
  --admission-checkpoint ADMISSION.json --out STATE/open.json
```

Una base SQLite vacía no prueba que una hotkey sea nueva. La historia firmada es la autoridad para consumo de candidaturas y swaps de hotkey.

## Ejecución, replay y publicación

El worker de V2 genera los vídeos en un entorno controlado del validator y guarda los recibos necesarios para reproducir panel, timing, diversidad y cualificación. Un fallo de infraestructura deja la ronda en recuperación o incompleta; no autoriza cambiar muestra, ampliar tolerancias ni volver a sortear el panel.

Antes de votar, cada validator reconstruye la evidencia desde recibos firmados e independientes. Una promoción exige la mayoría estricta del stake completo congelado; una fila ausente, evidencia alterada, desacuerdo de auditoría o cualificación incompleta bloquea la promoción. El resultado local puede crear un `WeightIntent`, pero no llama a `set_weights`.

La publicación cualificada es una puerta separada: conserva el intento, reconcilia respuestas desconocidas y verifica el reveal, el vector finalizado y el mapeo UID/hotkey. Una inclusión o una prueba de integración no son pesos efectivos. Véase [QUALIFIED_TESTNET_PUBLICATION.md](QUALIFIED_TESTNET_PUBLICATION.md).

## Inference SN126: prelaunch separado

Inference no convierte SN126 en dos mecanismos activos. El estado actual es un prelaunch privado y permissioned de mecanismo 0, con una simulación local de un ledger UTC de 24 horas. No firma, no escribe pesos y no publica en cadena. Su contrato de entrada es [INFERENCE_PRELAUNCH.md](INFERENCE_PRELAUNCH.md); [INFERENCE.md](INFERENCE.md) describe los límites de la integración.

## Rutas históricas

[LAUNCH_V1.md](LAUNCH_V1.md), [WAN_MVP.md](WAN_MVP.md) y la ruta distribuida 16/128/32 se conservan para replay de recibos y evidencia histórica. No son la regla vigente, no deben mezclarse con unidades, certificados ni calibración de V2, y no habilitan una ronda, un king ni publicación de pesos nuevos.

Para artefactos y capacidad histórica, consulta [ARTIFACTS.md](ARTIFACTS.md), [EVALUATION_CAPACITY.md](EVALUATION_CAPACITY.md) y [DISTRIBUTED_EVALUATION.md](DISTRIBUTED_EVALUATION.md).
