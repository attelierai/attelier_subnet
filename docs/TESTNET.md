# Testnet evaluation records

`forge/evaluation-record/v1` is the handoff between a validator-owned automatic
Wan evaluation and a future multi-validator commit/reveal round. It is a signed
statement of what one validator recomputed. It is **not** a certificate, a
crown, an instruction to set weights, or proof that a model won.

## Observed testnet commitments

On testnet netuid 578, the dashboard's finalized-commitment snapshot records a
model commitment at block **8,143,985**, a sealed-pack commitment at **8,143,991**,
and close block **8,143,998**. They establish public testnet commitments only.
They do not establish an admitted winner, quality preservation, a local gate pass,
a quorum certificate, a crown or weights. The local gate, hotkey-signed record and
testnet anchoring command exist as an evidence path; their use does not create the
missing multi-validator certification ceremony.

The retained receipt for `forge:v3:eval:19a04bfc5b7e720d2dadb5c89d45b16f666030816e15b4e7ffed7825f98a4bda`
was finalized at block **8,145,097** (read back at **8,145,104**). Its recomputed
status is `DEVELOPMENT_DIAGNOSTIC`, not a pass or a reward instruction. The final
receipt is evidence that one hotkey anchored that exact diagnostic record; it does
not change any winner or emission state.

The record binds these exact values:

- `network`, `netuid` and `round_id`;
- the canonical hash of the complete confirmation plan;
- the incumbent manifest and its certified-state hash;
- the candidate manifest, miner hotkey, admission block and frozen close block;
- canonical hashes of the policy, quality, ABBA timing and calibration evidence;
- the evaluator identity and a SHA-256 of the retained Luna response journal;
- the recomputed automatic-gate result and its status; and
- whether the source reports were public and whether the run was development-only.

The confirmation plan must carry the following local bindings before a record
can be created:

```json
{
  "network": "test",
  "netuid": 578,
  "round_id": "r...",
  "close_block": 123,
  "admitted": [{"manifest_hash": "...", "hotkey": "...", "block": 122}],
  "candidate": {"manifest_hash": "...", "hotkey": "...", "admission_block": 122},
  "incumbent": {
    "manifest_hash": "...",
    "certified_state_hash": "...",
    "certified_activation_block": 100
  }
}
```

`create_record()` calls `evaluate_automatic()` itself; callers cannot supply a
claimed gate status. It rejects a mismatch between the candidate/incumbent in
the plan, quality report and ABBA timing receipt. It can record
`INCOMPLETE`, `FAIL`, `INCONCLUSIVE` and development diagnostics as well as
`PROMOTE_ELIGIBLE`; retaining a negative result prevents a validator from
silently replacing it with a different claim.

`sign_record()` signs a domain-separated canonical payload with the validator's
existing sr25519 hotkey. `verify_record()` verifies it from only the public
hotkey. Changing network, netuid, round, candidate, evidence hash or any other
bound field invalidates the signature, so a record cannot be replayed into a
different round.

The evaluator journal hash is a retention commitment, not a claim that this
module independently audited the provider. Operators must retain the matching
journal, policy and media/evidence objects for later reopening.

## Signed testnet anchor

`forge testnet-publish-evaluation` recomputes the automatic gate from local
`--plan`, `--policy`, `--quality`, `--timing` and `--calibration` evidence,
hashes the retained `--journal`, signs the resulting record with the validator
hotkey, and verifies that sr25519 signature before producing a receipt. It only
accepts `network=test`, requires the signer to be both registered and present
in the finalized testnet committee, and defaults to `DRY_RUN`.

`--execute` is an explicit second action. It anchors only
`forge:v3:eval:<record-hash>` and atomically updates the receipt through
submission and finalized readback. The command refuses to overwrite that
validator's sealed prompt-pack commitment while its close window is active.
The receipt and anchor remain signed evaluation evidence; an `INCOMPLETE`,
`FAIL` or diagnostic record is retained exactly as signed, and no invocation
can crown a model or submit weights.

## Next integration step

A real testnet coordinator may publish the signed record beside the
commit/reveal documents. It must construct `LocalEvaluationReport` only from a
verified record for the same confirmation plan. The pre-consensus layer must
then collect a quorum certificate and apply it transactionally before any WTA
weight plan is even prepared. This repository deliberately does not make that
connection here, and `testnet-round` remains simulation-only.


Antes de firmar un plan público con `testnet-publish-evaluation`, el ancla vuelve a
leer en cadena los hashes finalizados de cierre y entropía, el comité de cierre, el
commitment sellado del pack y el commitment del candidato. Para v3 comprueba
repo, revisión, rol y receta contra el manifiesto fijado; para v2 comprueba la
referencia de la declaración fijada. Un plan diagnóstico sin esos campos sólo puede
anclarse con `--development`; no afirma admisión pública en cadena. El lock de
finalista ya tiene un commitment específico y sus reglas canónicas se describen
abajo; sigue sin ser un quorum ni una promoción.

Para confirmación, publique `testnet-publish-finalist-lock` después del cierre. El
recibo contiene el digest del candidato, incumbent, contrato y hash finalizado de
cierre; el bloque `L` se toma del readback finalizado, no de una predicción. El plan
de confirmación usa `E=L+1` y verifica el digest, bloque exacto y comité histórico.
Un lock local, tardío, reutilizado o que cambie candidato no es elegible.


La regla canónica es `FIRST_VALID_LOCK_AFTER_C`: el planner recorre el historial
de commitments de cada miembro del comité de cierre, hasta 64 cambios por hotkey,
y elige el lock válido con menor `(block, hotkey)`. Si el historial es malformado o
supera ese límite falla cerrado. Por tanto republicar locks tras observar una
entropía no permite reinterpretar la misma ronda.

La admisión no se toma de la lista editable `plan.admitted`: para un plan público
actual (`admission_algorithm: forge/closed-admission/v1`), el planificador, el
publicador del lock y el ancla vuelven a resolver todos los commitments de modelo
que estaban en el snapshot finalizado C, con contrato, revisión HF inmutable y
orden `(block, manifest_hash, hotkey)`. Sólo después aplican el deduplicado y el
límite de candidatos. Por ello omitir un finalista temprano o sustituir el
manifiesto en JSON hace fallar el anclaje y la publicación.

Esto reduce reinterpretaciones accidentales del operador; no sustituye un estado
certificado del incumbent ni un quorum multi-validador, y un comité malicioso no
queda resuelto por este mecanismo. El límite de 64 cambios de historial sigue
siendo fail-closed y debe revisarse si el ritmo real de commitments lo exige.


Durante la ventana C..L un commitment `finalist_lock` de cualquier miembro del
comité que no corresponda exactamente a un candidato canónico de esa ronda hace
fallar cerrado. Esto cubre un snapshot HF que ya no pueda resolverse después de
C: no se ignora el lock temprano para aceptar otro posterior. En consecuencia el
comité no debe solapar locks de rondas distintas en esa ventana.
