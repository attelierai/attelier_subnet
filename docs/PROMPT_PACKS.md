# Sealed prompt packs

Each production round uses a new `forge/sealed-pack/v1` prompt pack. Its private 64-hex salt, round context (including `close_block`) and entries are hashed with `seal_pack`. The validator publishes only `{hash, block, hotkey}` by the close block, then reveals the entire pack after the entropy block. `verified_plan_prompts` accepts a reveal only when it hashes to that commitment, its network/netuid/round/contract/close context matches, `commit_block <= close_block < entropy_block`, and the observed `pack_reveal_block` is at or after entropy. Confirmation additionally requires the finalist lock in `[close_block, entropy_block)`.

This document defines pack semantics. The observed netuid 578 commitment blocks and
the separate local-record/anchor boundary are recorded in [`TESTNET.md`](TESTNET.md).

Here, a new pack means a new round context and commitment, not globally new text.
Uniqueness checks apply **within one pack**. There is no historical lookup of
previously revealed prompt texts or `source_group` values: an operator can place
the same entries under a new context and salt and pass the current structural
checks. Avoiding cross-round reuse is an operational requirement, not an enforced
guarantee. Activation needs an auditable history of revealed packs and an agreed
non-reuse rule checked by independent validators; changing the salt is insufficient.

Entries carry one unique normalized prompt text and one unique `source_group`. A source group represents an independently authored concept, never a paraphrase label. The pack builder verifies only these structural claims and the four subject classes, two camera modes and two style families. It cannot prove that metadata is honest, concepts are truly independent, prompts are semantically novel, or the pack was not leaked before its reveal. A post-entropy reveal publication is useful audit evidence, but does not establish secrecy. Operators should keep source provenance and have independent validators review the revealed pack; quorum and chain finality remain separate protocol requirements.

Selection is deterministic from finalized entropy, stage and the sealed-pack hash, with balanced subject classes. Entropy is fixed by stage: `screen = close + 1`, `duel = close + 2`, and `confirmation = finalist_lock + 1`; the confirmation lock must be after close. A pack must contain enough entries for every required class; there is no procedural fallback. The legacy procedural grammars remain resolvable solely to replay old receipts. Production admission must require `prompt_source: forge/sealed-pack/v1`. The confirmation lock is encoded and checked against finalized commitments by the planner and testnet anchor; its canonical history and evidence boundary are specified in [`TESTNET.md`](TESTNET.md). This is not a quorum certificate, incumbent certification or a winner publication.

The integration test includes a deliberately simple finite-vocabulary oracle to show why a renewed pack is useful. It is a simulated lookup attack, not evidence about a real model's generalization or resistance to prompt leakage.

## Historial entre rondas

La unicidad estructural de un pack sigue siendo sólo interna. Para una promoción de
producción, el verificador requiere además un `forge/prompt-history-checkpoint/v1`
proporcionado por estado certificado independiente, ligado al estado certificado del
incumbent de la confirmación, y un segmento
`forge/prompt-history-bundle/v1` que parta exactamente de esa raíz. Cada enlace
reproduce un plan revelado completo, su pack y sus identidades normalizadas NFKC de
texto y `source_group`; se rechazan colisiones, cierres fuera de orden, raíces
cambiadas, enlaces truncados desde el checkpoint y rollback frente al checkpoint
aportado. Se comparan todas las entradas reveladas del pack, no sólo las que fueron
seleccionadas para el panel. El segmento acepta como máximo 64 enlaces y 20 MB
canónicos; un estado certificado nuevo debe compactar el historial antes de superar
ese límite.

El checkpoint no se toma del plan ni de SQLite local: su autenticidad, actualidad y
completitud pertenecen al estado certificado/quórum que lo suministre. Mientras no
exista ese transporte, falta el checkpoint y la ruta de producción queda
`INCOMPLETE`; el bundle por sí solo no demuestra que no se omitieron rondas. No hay
un flag CLI que acepte un checkpoint arbitrario: la API sólo recibe ese objeto desde
un consumidor futuro del estado certificado. Esto impide prometer que contexto y salt nuevos constituyan novedad global, y tampoco
prueba equivalencia semántica más allá de la identidad textual normalizada.

