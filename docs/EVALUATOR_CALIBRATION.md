# Evaluator calibration and RTX 5090 probe

Qualified execution binds `source_implementation_hash` to the actual fusion,
VBench adapter/runner, vision judge, media sampling and raster-check sources.
Changing any of them requires a matching calibration; a nonempty arbitrary hash
does not authorize execution. Metric receipts must also match the calibrated
runner source. Diagnostic observations remain non-qualifying after a numerical
bug fix until the revised evaluator is independently calibrated.

The quality evaluator is a blocker for promotion.  It does not prove that a
finite panel has zero perceptual loss; it only decides whether the automatic
rule is calibrated enough to reject known regressions and avoid systematic false
alarms under the declared policy.

The reproducible sequence is:

1. Render and judge development controls.  `mvp-fit-policy` derives the only
   permitted quality margin from those rows and freezes the policy.
2. Run a fresh **calibration** split with new source and concept families.  It
   must meet the frozen false-alarm and sensitivity bounds.
3. Run a separate fresh **independent** split.  It must meet the same bounds;
   neither split can tune the margin.
4. Only then freeze separate screening and 128-prompt confirmation packages.

## V4 reusable-family campaign and pilot

New promotion-grade campaigns use `forge/mvp-calibration-campaign/v4` only.
Each development, calibration and independent partition has 64 source
families: 32 static and 32 requested-motion. A source may appear in different
control kinds, but never twice within one kind. Static controls use static
families; motion controls use requested-motion families. `wrong_video` binds a
different static target and motion donor lineage for every comparison, so
neither lineage repeats inside that control. V4 retains the applicable category
coverage: static families provide four examples in each static-camera
subject/style cell and motion families provide two examples in each
subject/camera/style cell, per partition. Labels describe the actual source;
a fixed-camera action is not relabelled as a moving-camera scene.

The development pilot is frozen in
`configs/evaluator-pilot-manifest-v1.json`.
It has eight static and eight motion families, 32 Wan-50 generations (16
references, eight new runs, eight new seeds), 96 blind comparisons and 192
Luna calls. It is explicitly `qualification: prohibited`: its source bank is
reviewed for prompt applicability but no control is observable until the
rendered evidence is judged. A full v4 campaign may bind it as
`development_pilot`; all pilot families and exact base/seed/control identities
are then confined to development and reused there. It cannot enter calibration
or independent holdouts.

Each of the twelve control kinds needs at least 32 observable, independent
source/concept families in both calibration and independent splits.  A control
is a hit only when a protected axis crosses the frozen negative margin.  The
legitimate controls (`self`, `same_model_new_run`, `independent_seed`, and
`static_scene`) instead bound false alarms across every protected quality axis.
The receipt must include two blind Luna orders per comparison, VBench and Luna
provenance, source/concept identities, and a hash-bound threshold derivation.

The validator rejects a receipt if sources, concepts or their families overlap
between development, calibration and independent splits.  It also now rejects a
campaign whose calibration split fails even if its final independent holdout
looks favorable.

## Why 32, and what it does not prove

This count is not a proxy for 128 competitive prompts.  It is the fixed
sample size for a *per-control-family* binomial claim at the frozen 0.05
interval level.  With the present 0.80 sensitivity and 0.15 false-alarm
targets, an unrealistically perfect campaign would need 17 all-hit rows to
bound sensitivity and 23 zero-false-alarm rows to bound false alarms.  At 32,
a degradation family can tolerate one miss (31/32) and still meet the lower
sensitivity bound; a legitimate family cannot have a false alarm and meet the
upper bound.  That is deliberately strict, but it does not establish that a
128-prompt challenger panel is powerful enough.  The panel still reports its
global and protected-category confidence bounds independently.

`forge.evaluation.calibration_campaign.freeze_calibration_campaign` freezes
the 960 slots before render: 32 development observations per legitimate
control and eight per degradation recipe, plus 12 controls × 32 source families
in each of calibration and independent holdouts. Development only selects the
margin and control recipes; its rows do not prove sensitivity. The larger
legitimate sample exposes more natural variation, but the current maximum-based
fit can still have its margin set by one extreme legitimate observation. It
does not make the fit robust to outliers or establish panel-level power. It records reuse at
the base-family level rather than pretending derived controls are independent
renders. There are 360 distinct base families; the number of GPU generations
depends on which comparison needs a second run and must be costed by the worker
before dispatch.  The preflight reports 768 held-out comparisons and 1,536
blinded judge calls.  It is a work count, never a spending authorization.

The command no longer invents prompts or calls numbered takes separate source
families. It requires a validator-owned, frozen `source-bank.json` with one
inspectable record per source/concept family:
`base_group`, `source_id`, `concept_id`, `source_family`, `concept_family` and
`prompt_text`. New v3 banks also freeze `subject_class`, `camera_type` and
`style_family` from the current Wan confirmation vocabulary (`rigid_object`,
`human`, `animal`, `scenery`; `static`/`moving`; `realistic`/`stylized`). The CLI
requires this v3 form for new plans. The planner assigns whole supplied families
to one split and
rejects repeated identities, exact normalized prompts, and simple numbered
take/version variants across the bank. This structural validation does **not**
establish semantic independence from metadata alone. Before GPU or Luna spend,
record an automated or agent-assisted semantic review of the corpus and its
taxonomy; until a reviewed corpus exists, no campaign is qualified.

For each v3 held-out split and control kind, the 32 independent families are
pre-allocated as two families in each of the 16 subject/camera/style cells.
Those labels are bound into the slots and later evidence, so a diagnostic cannot
categorize a result after it is known. This preserves the 960-slot campaign and
does not alter the quality margin, calibration hit rule, candidate gate or
promotion eligibility. Historical v2 campaigns remain readable, but lack these
labels and cannot support a future stratified diagnostic.

Before dispatch, run the CPU-only panel report as well:

```bash
.venv/bin/python -m forge.cli mvp-panel-power \
  --policy POLICY.json --probe .codex/5090-validation/outcome.json \
  --out /secure/state/panel-power.json
```

The current confirmation code examines 272 quality lower bounds: eight axes,
global plus 16 protected cells, against king and Wan-50. This is an
intersection-union noninferiority gate: promotion requires every bound, so
0.05 per bound already limits a false promotion whenever any protected axis is
truly below the margin. The cost of all those bounds is power: candidates near
the margin can become inconclusive because at least one valid bound misses.
The report is intentionally informational and must not be used to claim that
the three-prompt RTX probe establishes variance or power for eight prompts per
protected cell.

After scoring a complete v3 campaign, the same informational command can run a
conditional stratified replay of the exact candidate quality bounds:

```bash
.venv/bin/python -m forge.cli mvp-panel-power \
  --policy POLICY.json --calibration CALIBRATION.json --draws 64 \
  --out /secure/state/conditional-panel-replay.json
```

It samples eight whole-source score vectors per protected cell, with replacement
from the two frozen sources in that cell, preserving within-source axis
correlation. It applies the existing global and cell means/lower bounds without
fitting a threshold. It reports simulated quality-clear, FAIL and INCONCLUSIVE
frequencies for each held-out control and split, separately from the original
source-control calibration result. This is a **single-reference conditional
simulation**, not 64 new independent panels, a joint king/Wan-50 power estimate,
a measured acceptance rate, or permission to promote. Two sources per cell
provide a very limited empirical distribution. The report refuses post-hoc
labels, incomplete/hash-mismatched evidence and unobservable controls.

The intersection-union argument needs valid one-sided bounds for independent
prompt sampling; it does not require independence between the 272 bounds.
Neither that argument nor resampling establishes that the t-based bounds have
adequate coverage for every score distribution. Source-control sensitivity,
false-promotion control and the chance of an inconclusive panel are distinct
claims; there is no requirement here to buy 32 full panels per control.

## Calibration and candidate decision units

The calibration rule currently records a binary crossing for each control row,
then uses a binomial count by control kind. The candidate rule instead uses
prompt-level symmetric deltas, one global mean and 16 protected eight-prompt
cell means with one-sided lower bounds. Those are different decision units. A
maximum legitimate row delta can be a conservative development margin, but it
does not by itself show whether the 128-prompt gate distinguishes normal seed
variation from a small systematic regression.

Run `scripts/calibration_unit_diagnostic.py` on a completed probe before using
it in planning. The RTX 5090 result records independent-seed Luna deltas down
to -0.25 and a native imaging-quality delta of -6.82, while mild blur has zero
Luna deltas and a native imaging-quality delta of -3.26. These values are
diagnostic only: the source receipt has one family per kind, `observable: null`,
and lacks a qualified fused calibration receipt. The pinned VBench imaging
runner divides its per-video MUSIQ mean by100: these image-axis deltas are
-0.0682 and -0.0326 respectively, not percentages of perceptual loss. This
conversion alone cannot fit a margin, establish mild-blur sensitivity, or tell whether
more of the same blind control would help. The next paid work must use frozen,
reviewed families and the exact candidate fusion path; its outcome decides
whether qualification stays disabled.

The current fusion protects eight candidate axes. It uses Luna for prompt,
action and camera adherence plus motion-derived activity; VBench provides the
subject, background, image and temporal axes. Luna also records physical
plausibility, visual quality, temporal stability and its subject-consistency
assessment, but they are not additional protected candidate axes today. This
is a claim limit, especially for defects primarily visible as physical
implausibility. It is not a reason to add or retune axes during the current
candidate evaluation; any policy change needs a new frozen calibration.

The competitive distributed replay requires this frozen campaign as well as
the ordinary calibration receipt.  It recomputes a binding from the campaign,
all scored slots, the policy and the independent-check hash; the distributed
plan must contain that binding.  A diagnostic or legacy receipt without frozen
slots remains useful for research but cannot make a challenger eligible.

Freeze the campaign before a worker is provisioned.  This writes a new file
only and has no GPU, judge, upload or chain side effect:

```bash
.venv/bin/python -m forge.cli mvp-calibration-plan \
  --campaign-label calibration-YYYYMMDD-r1 \
  --entropy FINALIZED_CHAIN_ENTROPY \
  --source-bank /secure/state/reviewed-source-bank.json \
  --out /secure/state/calibration.private.json
```

The runner must later add its signed media, VBench and two blinded Luna orders
to those slots without changing any bound field.  `mvp-calibrate` checks the
ordinary receipt; competitive replay additionally calls the qualified campaign
binding and fails closed on any absent or substituted slot.

## Bounded RTX 5090 probe

Generate the manifest before rendering.  It is private until its controls have
been executed, and the output is immutable:

```bash
.venv/bin/python -m forge.evaluation.rtx5090_probe \
  --campaign-id rtx5090-YYYYMMDD-01 \
  --out /workspace/forge/rtx5090-probe.json
```

It defines three distinct synthetic concepts: development, calibration, and
independent. It requests 15 quality GPU renders, eight targeted locally-derived
defect/attack variants and two isolated ABBA timing quartets. That is 23 measured
GPU renders plus eight derived variants; the timing runtime retains one video
per side and prompt, so the probe stores 27 video files. The ABBA runtime records
four timed renders per prompt and one initial
warmup per side; warmups remain operational cost, not speed evidence. Every
timing quartet runs alone on the GPU; quality work must not overlap it. Record
the exact GPU, driver, precision, dependency lock, scheduler, recipe, prompt
seed, complete-pipeline time, decoded video hash, VBench receipt and both Luna
responses.

This probe deliberately has one source per control/split.  Its only valid
outcomes are diagnostic: it can expose a broken RTX 5090 runtime, a false alarm
on a static scene, or a missed obvious degradation.  It cannot meet the
32-family calibration requirement, establish a speed confidence interval, crown
a king, or publish weights.  A later full calibration requires new source and
concept families; none of the probe videos may be reused.

Use the execution helper only on the remote GPU after preparing the immutable
manifest:

```bash
.venv/bin/python scripts/rtx5090_probe.py prepare \
  --probe /workspace/forge/rtx5090-probe.json \
  --contract CONTRACT.json --base-manifest BASE.json --base-dir /workspace/forge/model \
  --validator-hotkey HOTKEY --max-generation-seconds 900 --out /workspace/forge/rtx5090-run
.venv/bin/python scripts/rtx5090_probe.py render --prepared /workspace/forge/rtx5090-run
.venv/bin/python scripts/rtx5090_probe.py derive --prepared /workspace/forge/rtx5090-run
```

`prepare` is CPU-only. `render` reuses the validator's diagnostic worker for
the 15 quality renders and its paired runtime for ABBA; it does not implement a
second generation path. `derive` records a SHA-256 for every transformed video
and leaves `observable: null`: only subsequent blinded judging can classify a
control as informative. The static-scene control is bound exclusively to the
static source; freeze and shuffle are bound exclusively to requested-motion
sources.

## Observations from the 2026-10-04 probe

The RTX 5090 completed all 23 measured generation calls and the five pinned
VBench dimensions. The three same-seed Wan-50 reruns were byte-identical to their
references. Blinded Luna comparisons of self, identical rerun and the legitimate
static-scene control gave zero difference on every axis in both orders.
Independent-seed generation did vary and produced order disagreement; this is
not evidence of a systematic regression or a calibrated false-alarm rate.

Luna registered negative motion scores for an incorrectly frozen action,
negative temporal scores for shuffled frames, negative visual quality for strong
blur, and negative prompt adherence for the wrong scene. It returned zero
differences for the mild blur, detail-loss and both photometric attack controls.
Those zero results must remain visible. A transformation name or changed pixel
hash does not establish perceptual observability; the probe leaves that field
unset and cannot count these transformations as validated sensitivity examples.

The 40-step CFG-5 recipe lost native MUSIQ points in all three source concepts:
approximately **-3.27**, **-18.35** and **-2.50**, averaging **-8.04 points**.
These are index differences, not percentages of perceptual loss. They do not
establish a calibrated non-inferiority verdict, and the measured speed gain
cannot be advertised as quality preservation.

All controls, disagreements and raw receipts are retained, including weak or
uninformative results. No threshold was adjusted using these outcomes. This is
one family per control, not the required 32 independent families in each held-out
split; it does not qualify the automatic evaluator or the 128-prompt policy.
The distributed promotion gate remains closed until calibration, independent
raw-evidence replay and the competitive certificate/custody path are qualified.

After all judging and metric work ended, a separate frozen one-prompt GPU smoke
exercised the final `screening_ab` runtime. It passed with exactly two synchronized
measurements in reference/candidate order, a plan-bound `forge/screen-timing/v1`
receipt and no ABBA confirmation receipt. Its two warmups plus measured calls and
loading/validation took 473.96 seconds. This verifies the new runtime path; it is
not a full 16-prompt screening or an eligible candidate evaluation.
