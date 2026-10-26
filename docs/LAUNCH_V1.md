# Model launch V1

Status: **testnet-only implementation and bounded real-GPU acceptance in progress**.
The current campaign authorizes no mainnet reads, configuration or transactions.
Models launch first; creator rewards follow later. This is the pragmatic initial
rule approved on 2026-10-14. Human annotation, a learned adversary and a large
perceptual calibration study are improvements, not prerequisites for this V1.
The historical calibrated 16/128/32 route remains replayable in `WAN_MVP.md`.

## Initial decision rule

- Public immutable safetensors + reusable license + bounded validator-owned
  recipe. No miner Python. Training on the current king is encouraged. Effective
  duplicate weights **and** recipe are rejected, including under another repo.
- Sixteen paired prompts from distinct generated scene families: four static
  shots and twelve motion shots. Eight also receive measured R,C,C,R timing on
  the same GPU. The public generator is deterministic from frozen post-close
  entropy; it is not a claim of novel concepts or resistance to overfitting.
- Speed must clear 1.10x at the one-sided 95% Student-t lower bound of the
  per-prompt log ratios. Full generation includes text encoding, denoising and
  VAE decoding; model loading, materialization, video encoding and warmups are
  excluded from that metric but included in operational cost.
- VideoReward evaluates **VQ, MQ and TA separately**, against both the current
  king and the original Wan-50 anchor. No sum or mean across heads buys a pass.
  Lower 95% paired-delta bounds must be >= -0.35 VQ, -0.20 MQ and -0.20 TA.
  These are **provisional normalized reward units**, not percentages or a
  calibrated guarantee that every visible loss is slight.
- Tagged motion/steady-light/continuous-shot prompts additionally check exact
  freezes, excessive luminance acceleration and abrupt frame transitions.
  These narrow guards are not a physics, diversity or comprehensive gaming test.
- Policy revision 2 additionally rejects a near-freeze only on a
  `motion_required` prompt when decoded `temporal_delta` is <= `0.1/255` and
  <= 1% of a moving reference (`temporal_delta` above `0.1/255`). This closes
  the observed codec-noise case where `frozen_fraction` was zero. Revision 1
  plans remain replayed with their original hash and rules; a new measurement
  uses revision 2.
- Missing, duplicate, foreign, changed or nonfinite evidence is INCOMPLETE.
  An observed mean failing a margin, or a temporal guard, is FAIL. Means passing
  while confidence bounds do not is INCONCLUSIVE. Only complete passing evidence
  is PROMOTE_ELIGIBLE; that result alone never activates rewards or crowns a king.

The fixed values and policy hash are owned by
`src/forge/evaluation/launch_policy.py`. Policy and prompts are frozen before
candidate measurements. Changing a threshold starts a new version and new round;
never reinterpret a failed candidate by silently moving its threshold.

## Execution and authority

`forge.evaluation.launch_worker` runs the existing strict paired Wan loader and
isolated pinned VideoReward environment. It binds scores to the exact generator
report, model, prompt, seed and video bytes. Heavy data stays on the remote worker;
small immutable receipts are retained with the private evidence archive.

`forge.validator.launch_bridge` reconstructs policy rows from raw receipts in an
operator-controlled local store, then uses the existing signed committee and
SQLite state. A miner-supplied score/result JSON is not trusted execution evidence.
Receipt hashing detects altered files; honest eligible validators remain the
execution trust boundary. No TEE or independent GPU attestation is claimed.

The dashboard exposes pinned public model revisions, license, file hashes and
weight download links. V1 scores use an additive `open_quality` object; legacy
[-1,1] judge axes and V1 normalized scores must never share a scale. A local pass,
a signed certificate and finalized chain activation are distinct states.

## Minimum testnet acceptance still to close

1. Complete real public-artifact generation/scoring under the frozen V1 rule;
   retain failures/inconclusives and recovery evidence. No fabricated winner is
   needed: a real KEEP is a valid outcome.
2. Exercise the real registered-miner path and the eligible-validator committee
   on testnet, including immutable model bindings, restart, tamper, missing
   evidence, stale parent, expiry and public model availability. Demonstrate
   certificate delivery and finalized weight readback with the actual adapter.
3. Verify testnet 578's eligible committee, capacity-bound queue and windows,
   immunity and supported weight publication from fresh testnet chain reads.
   Prove that admitted work fits the deadline on the deployed worker.
4. Publish the verified dashboard build and actual sanitized receipts/media;
   verify public downloads and the active deployed artifact.

A small launch accepts residual metric risk. Monitor submitted models, retain a
fixed anchor and version the rule when a concrete exploit or failure is observed.

Measured campaign results and their limits: [V1 testnet acceptance](LAUNCH_V1_ACCEPTANCE.md).

## Operador offline de ronda testnet

`python -m forge.validator.launch_operator` prepara archivos revisables; no abre
una conexión, no crea firmas, no ejecuta GPU ni publica pesos. Para una ronda
futura, `prepare` recibe el estado durable, el checkpoint de admisión ya firmado,
el snapshot del comité, los tres manifests, el panel V1 congelado, un recibo de
capacidad y el bloque/hash de entropía posterior al cierre. Reconstruye el panel completo mediante `synthetic_prompts.launch_panel` con ese
hash y el `round_id`, y selecciona los ocho timings por orden determinista;
un texto, familia, tag o seed que no coincida no se prepara. No abre aún la ronda: el recibo de
capacidad debe ligar el hash de ese plan ya revisable. `open` importa ese recibo
y llama a `LaunchV1Bridge`.

Después de que el worker haya ejecutado los dos shards y `launch_worker` haya
producido sus scores, `bundle` importa paneles, timings y scores a un
`LocalLaunchReceiptStore` por hash de contenido y escribe el bundle del
validador. `report` reevalúa ese bundle. Los commits, reveals y certificados se
intercambian como envelopes firmados existentes y se importan con
`receive-envelope`; `apply-certificate` sólo verifica y persiste un certificado
ya firmado. Finalmente `intent` prepara el `WeightIntent` testnet desde el
certificado persistido y recibos locales de custodia/burn; no presenta una
transacción ni sustituye el readback finalizado.

El operador no convierte campañas diagnósticas ni experimentos R3/R4 en rondas:
requiere un cierre C, entropía E>C, admisión firmada y recibos nuevos ligados a
ese `RoundPlan`.

`forge.validator.launch_voting.LaunchV1VotingWorkflow` conecta esos recibos con
el protocolo de voto existente. Reevalúa el bundle antes de firmar, conserva el
commit y su salt en SQLite para recuperar un proceso reiniciado y verifica las
firmas y los recibos de cada reveal recibido. La construcción del certificado
exige el quorum congelado y rechaza evidencia incompleta o contradictoria.
Los signers, el bloque finalizado y la entrega de envelopes los aporta el
operador; el módulo no contiene claves, clientes RPC ni ejecución GPU.

La operación real debe reconstruir `resolve_closed_admissions` al cierre y
comprobar el par exacto hotkey/manifiesto antes de abrir la ronda. El checkpoint
de historial firmado por sí solo no demuestra esa admisión. Los compromisos de
voto se publican en testnet y se releen al cierre; sólo después del reveal y la
activación se aplica el certificado. La publicación de pesos requiere además
custodia pública verificada y una prueba de burn ligada al runtime actual;
el recibo de envío queda pendiente hasta comprobar el vector finalizado.

Dos validadores que reevalúan los mismos recibos prueban firmas, transporte y
recuperación, pero comparten una ejecución GPU. No representan dos mediciones
independientes ni dos operadores independientes. Cuando king y anchor son
Wan-50, los dos jobs del panel requieren 48 generaciones medidas y 4 warmups;
sus primeros renders producen los 32 vídeos que puntúa VideoReward.

`forge.validator.launch_certified_dashboard.export_certified_snapshot` transforma
el snapshot ya reconstruido desde los vídeos y recibos en una ronda certificada
solamente después de verificar los dos almacenes SQLite, el certificado y los
readbacks finalizados. Reconstruye los intents desde el estado certificado y los
UIDs observados; no acepta como autoridad el ganador declarado en un journal.
Los dos journals deben corresponder a validadores distintos del comité firmado.
El dashboard conserva los experimentos anteriores y publica por separado la
ejecución compartida, la transición certificada y la activación de pesos. KEEP
conserva el king; PROMOTE necesita la corona real del estado certificado.
