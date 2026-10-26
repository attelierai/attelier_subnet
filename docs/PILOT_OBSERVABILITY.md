# Pilot observability protocol

The development pilot may submit its 192 blinded judging calls only after a
complete pre-score observability receipt validates. This is a fail-closed
integrity gate for the 96 frozen comparisons; it is not a score, a perceptual
finding, evaluator calibration, or approval to spend. The receipt must be
created before any Luna call, fusion, or access to VBench score values. VBench
may compute and store sealed sheets beforehand; this protocol does not certify
who could access them, so the executor must enforce that separation.

The decision changed by this gate is narrow: a PASS permits blinded judging of
the already-rendered pilot media. Any other outcome stops the pilot before
judging. The independent unit remains one semantic source family per control;
`wrong_video` binds both disjoint source lineages. Reuse across control kinds
does not create extra independent observations.

`forge.evaluation.pilot_observability.freeze_pilot_observability_protocol`
recomputes the exact frozen pilot worklist and freezes one requirement for each
comparison. Each requirement binds its slot, control kind, reference and
candidate job IDs, source/donor lineage hash and treatment hash. The latter
includes the preregistered control specification and job identities, including
the frozen shuffle permutation in the worklist.

The protocol freezes a three-part perceptual rule for every control:
applicability, what must be perceptible, and when the row is invalid. `self`
requires identity rather than a visible defect; new-run and new-seed rows allow
ordinary legitimate variation; `static_scene` treats lack of motion as
legitimate. `freeze` and `shuffle` require meaningful reference motion; blur
and detail controls require perceptible degradation with comparable content;
wrong-video requires a perceptible semantic mismatch between bound lineages;
photometric attacks require a perceptible artifact. A transform, offset or
pixel digest alone is never perceptual evidence. These rules are frozen before
review and contain no post-score exceptions.

`validate_pilot_observability_receipt` accepts a receipt only when all of the
following are true:

- Exactly 96 pre-score review rows exist, one for each frozen slot.
- Exactly the media job IDs used by those comparisons have SHA-256 digests.
- Every review repeats the frozen kind, lineage, treatment and media digests.
- Every review has `review_status: PRE_SCORE_COMPLETE` and
  `observable: true`.
- The canonical pilot, worklist, protocol and receipt identities match.

The receipt is deliberately score-free: it has no Luna response, fusion axis,
threshold or pass/fail quality value. A changed pixel hash alone does not make
a control observable. The reviewer must inspect the rendered media before
scoring under the frozen rule: mild blur must be perceptible, shuffle/freeze
must have meaningful requested motion, and a photometric change must be a
perceptual defect rather than merely different pixels. If any control is not
observable, the only valid result is an incomplete pilot; it cannot be relabelled,
replaced, reseeded or judged anyway.

The pilot remains `qualification: prohibited` even after this gate passes. A
PASS does not demonstrate sensitivity, false-alarm control, quality retention,
worker agreement or candidate speed. Those claims require the later separately
funded calibration, panel and worker evidence described in
`docs/EVALUATOR_PILOT.md`.

Call `validate_pilot_observability_receipt` with both `pilot=` and `worklist=`
at the execution boundary. These parameters remain optional for historical
read-only inspection; without them, a validator can check canonical
self-consistency but cannot recompute and reject a coherently rehashed protocol.
