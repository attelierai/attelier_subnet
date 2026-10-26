# Bounded Wan LoRA MVP

This page documents the existing executable 16/128/32 path and historical evidence.
The current bounded testnet rule is [LAUNCH_V2.md](LAUNCH_V2.md). The broader
target is [MODEL_COMPETITION.md](MODEL_COMPETITION.md), with its
development comparison in [EVALUATOR_STUDY.md](EVALUATOR_STUDY.md). Neither replaces
the running policy or qualifies a model merely by being documented.

## Current acceptance status

Local implementation is available; **rewards are not activated**. No independently
calibrated Wan-50-preserving candidate or complete real testnet reward cycle has
been demonstrated. Synthetic integration fixtures test enforcement only. The
64-update LoRA remains a negative experiment; do not rename it as a winner.

The current decentralized diagnostic route, strict-majority protocol version and
its remaining activation limits are described in [Distributed evaluation](DISTRIBUTED_EVALUATION.md).

## Contract

The pinned base is Wan2.1-T2V-1.3B-Diffusers revision
`0fad780a534b6463e45facd96134c9f345acfa5b`; tokenizer, text encoder, VAE and
architecture remain fixed. Anonymous public safetensors LoRA and compatible
transformer weights are allowed. Validator code owns execution; miner Python,
containers and private model endpoints are never executed.

`configs/wan-mvp.json` specifies 832x480, 81 frames, 16 FPS and the initial original
Wan **50-step UniPC** quality/speed reference. The current certified winner's
exact recipe becomes the speed comparator; Wan-50 remains the quality anchor.
Candidates have at most 50 steps/100 forward passes; this is a resource ceiling,
not an implicit 16-step champion. The unused legacy duel field exists solely for
old contract readers. The new MVP has only screening and confirmation.

| Stage | Fixed work | Outcome |
| --- | --- | --- |
| Public preflight | Anonymous immutable revision, license, file/tensor/recipe hashes | Admit or reject before GPU |
| Screening | 16 prompts, one seed, paired lightweight comparison | Screen pass/fail/incomplete |
| Confirmation | 128 new prompts, one seed, candidate vs Wan-50 and current winner | FAIL/INCOMPLETE/INCONCLUSIVE/PROMOTE_ELIGIBLE |
| Timing | 16 screening AB slots; 32 predetermined confirmation slots with warmup and ABBA | Screening only tests potential; confirmation lower speedup confidence bound at least 1.1 |

Confirmation is 4 subject classes x2 camera types x2 styles x8 prompts. No panel
expansion or seed search is allowed. Timing covers text encoding, denoising and
VAE decode, with GPU synchronization. Downloads, model materialization and exports
are excluded from competitive time but included in operational cost. Clean model
materialization prevents accumulated LoRA state, including mixed LoRA/full pairs.

## Packages and receipt compatibility

`forge.protocol` owns contracts, hashes, histories, state and certificates;
`forge.validator` admission, preflight, budgets and orchestration;
`forge.evaluation` metrics, judges, calibration and gates;
`forge.models` public artifacts and Wan execution; `forge.chain` chain adapters.
CLI remains `forge`. All in-scope callers use these imports. Hash domains/serialized historical receipt formats remain unchanged.
Source-code identities necessarily change; a former evaluator calibration does
not qualify the changed evaluator. Legacy evaluation remains replayable under
`configs/wan-legacy-v3.json`, not the new 128-prompt gate.

Within those packages, keep the executable entry points separate from their data contracts:

| Responsibility | Modules |
| --- | --- |
| Frozen MVP policy, calibration, evidence binding, decisions | `evaluation/mvp_policy.py`, `mvp_calibration.py`, `mvp_evidence.py`, `mvp.py` |
| Manifest schema and anonymous artifact transport | `models/artifact_manifest.py`, `artifacts.py` |
| Shared generation recipe, stage-bound AB/ABBA execution | `models/generation.py`, `runtime.py`, `paired_runtime.py` |
| Consensus records and certificate/state transitions | `protocol/consensus_types.py`, `preconsensus.py` |
| Validator command registration and dispatch | `validator/validator_commands.py`, `validator_cli.py` |
| Public snapshot assembly, historical receipt readers, media | `validator/dashboard_export.py`, `dashboard_history.py`, `dashboard_media.py` |

`scripts/export_dashboard_snapshot.py` is only a command entry point. The historical
`automatic_*` evaluator is retained for receipt replay; it does not qualify the MVP.
Miners use the public artifact and submission commands; there is no separate miner validator or axon.

Local stabilization checks exercise SQLite admission/resume/budget persistence, evidence
validation, certificates, CLI boundaries and historical snapshot replay. They do not
replace remote GPU execution, independent quality calibration or finalized chain readback.

## Admission and budget

Historical V1 checkpoints use one lifetime hotkey-lineage admission. Windowed V2
checkpoints scope that right to their authenticated admission window: one attempt
per lineage and window, with no concurrent pending attempt across windows. A
technical retry reuses the same frozen admission, panel and full-evaluation round;
it cannot draw another sample. Effective artifact-and-recipe identities remain
permanently deduplicated, so a later window can admit only a genuinely new
candidate. Admission binds the weights, recipe and sealed panel.

An empty database does not prove eligibility. `admission_checkpoint` verifies
signatures exceeding two thirds of the full trusted committee and binds a trusted
certified state plus registration/lineage history. `mvp_admission` joins this to
public preflight; a JSON-supplied committee is not a trust root. Finalized chain
history ingestion and the initial independently trusted certified checkpoint still
need live qualification.

`EvaluationCoordinator` reserves before work, atomically claims FIFO screening,
and permits multiple screened challengers per round within the shared budget.
Previous SQLite round claims survive the migration. Infrastructure resumption retains the
exact artifact, recipe, panel and prior evidence. Unknown charges block retry
under every operation name until provider-receipt reconciliation. The default
validator allowance is10 USD; any higher validator allowance requires an explicit
campaign cap. Reservations count against both limits, with a warning at80%.
Insufficient budget leaves work incomplete.

## Automatic quality and calibration

The bounded policy replaces the legacy0.02 noninferiority claim; it does not
interpret a score delta as a perceptual percentage. Development controls determine
a frozen threshold and its derivation hash. An independent receipt must reproduce
that derivation, identify code/metric weights/Luna, and use disjoint source and
concept families across development/calibration/independent splits.
Verification reruns the declared development fitting procedure and compares the
entire policy and derivation; mutually consistent hashes alone are insufficient.
Within each independent control kind, source and concept IDs and their families
must be distinct. Multiple variants of one family cannot count as independent
observations in the binomial sensitivity/false-alarm intervals. The same source
may supply paired observations across different control kinds.

Twelve control families cover identical repetitions, fresh runs, independent
seeds, legitimate static scenes, incorrect freeze, mild/severe blur, temporal
shuffle, wrong prompt/video, detail loss and mild/severe evaluator attacks.
Nonobservable transformations are inconclusive. Independent false-alarm and
sensitivity bounds must pass for every family; missing evidence never passes.
The fixed128-prompt policy also requires real same-model and deliberately degraded
whole-panel checks before reward activation; this statistical power has not yet
been measured on Wan.
The current scalar margin is the largest legitimate development delta across all
axes. This can mask small regressions on quieter axes; it remains an unqualified
development rule, not a measured perceptual tolerance. Control-level PASS does
not establish whole-panel power or authorize reward activation by itself.

`evaluation.vbench_runner` calls the pinned official VBench implementation at
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, requiring clean source checkouts and
preverified local DINO/CLIP/MUSIQ/AMT/RAFT weights. Provision it in its own remote
metric environment. All five official metrics have now run in a separate batch
development pilot, with pinned source, weight hashes and a frozen dependency list.
The certification runner remains unqualified: the pilot provider did not attest
the runtime OCI digest, and the pilot produces diagnostic evidence only.
Native float32 scores can slightly exceed one on identical frames. The producer
clips boundary roundoff within1e-6, retains raw scores and their evidence hash,
and rejects larger excursions. Receipt consumers still require bounded values;
this is not a quality tolerance. Upstream implementation:
[API](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/__init__.py).

`evaluation.fusion` reconstructs paired metric differences and raw blinded Luna
A/B scores, authenticates journal context, and binds prompt/seed/video/model hashes.
Activity is judged against requested motion; larger dynamic degree is not always
better. `evaluation.mvp` protects every dimension and category, checks both quality
references and ABBA evidence, and exposes uncertainty. Raw scores and a signature
alone are not evidence of independent execution.

At bootstrap, identical Wan-50/incumbent manifests may share one canonical fused
comparison per prompt. `fuse_bootstrap_alias_row` binds both roles to the same
plan, seed, prompt, model, video and raw evidence hashes. Both quality inputs must
contain those exact same rows; this remains128 observations, not256. The two
blind-order judge calls and response replay protection remain mandatory.

## Reproducible local commands

From the owning repository, use its `.venv`:

```bash
.venv/bin/python -m forge.cli mvp-policy --out /tmp/wan-development-policy.json
.venv/bin/python -m forge.cli mvp-fit-policy --development DEVELOPMENT.json --provenance PROVENANCE.json --out-dir FROZEN_POLICY
.venv/bin/python -m forge.cli mvp-calibrate --policy FROZEN_POLICY/policy.json --evidence CALIBRATION.json --out CALIBRATION_RESULT.json
.venv/bin/python -m forge.cli mvp-freeze --policy FROZEN_POLICY/policy.json --screening SCREEN.json --confirmation CONFIRMATION.json --candidate-manifest-hash CANDIDATE_SHA --incumbent-manifest-hash INCUMBENT_SHA --wan50-manifest-hash WAN50_SHA --out-dir FROZEN_PACKAGES
.venv/bin/python -m forge.cli mvp-evaluate --bundle EVIDENCE.json --out REPLAY.json
.venv/bin/python -m forge.cli mvp-queue --db QUEUE.sqlite --network test --netuid NETUID
.venv/bin/python -m forge.cli mvp-dashboard --db QUEUE.sqlite --network test --netuid NETUID --out LOCAL_MVP_SNAPSHOT.json
```

Uppercase arguments denote real inputs, not included fixtures. Freeze writes
separate0600 screening/confirmation packages and a prompt-free commitment. Existing
packages are never overwritten. Revealing screening does not disclose confirmation.
`mvp-evaluate` produces a local replay, never a certificate or chain write.

For the frozen development/calibration/independent workflow and the bounded,
non-qualifying RTX 5090 runner probe, see [Evaluator calibration](EVALUATOR_CALIBRATION.md).

Miners can run the validator's own preflight against a candidate before
submitting it (see the [miner guide](miner.md)). A local training smoke is only a
viability check; it is not evidence of preserved quality or an admitted candidate.

## MVP validation scope

The current bounded diagnostic campaign uses **two real GPU validators** and
three public base-model recipes: an identical reference control, a deliberately
degraded control and a lower-step speed hypothesis. Work is divided equally with
audited overlap, rather than repeating the full panel at every validator. Both
validators are controlled by one operator; this is not operator independence.

Use separate test state, round identifiers and diagnostic receipts. Mocked results
must remain visibly synthetic and must never enter the real MVP reward verifier.
This tests distributed protocol behavior; it neither establishes independent
model evaluations nor relaxes the production quorum/execution requirements.
The earlier single-real-validator plus mocked-peer campaign remains historical.
Real calibration and full-panel cost qualification are separate from this small
diagnostic. See [Distributed evaluation](DISTRIBUTED_EVALUATION.md) for the actual
chain-derived assignment, signed evidence exchange and activation limits.

## Certificates, closure and weights

`protocol.transport` exchanges signed durable envelopes. Deadline expiry releases
a locally pending round without creating a KEEP/PROMOTE certificate or changing
the winner. Certificate synchronization remains parent-state checked.
`MVPCertificateVerifier` rejects simulated/diagnostic/missing execution evidence,
requires independent validator execution and public custody verification, and
uses the round's explicit quorum policy and retains double-sign protection.
New `stake_majority_v2` rounds require strictly more than half of the full frozen
committee stake. Historical rounds retain their original >2/3 rule and hashes.
The ledger can reopen an identical infrastructure-failed job; the runtime retains
failed outputs and does not blindly rerun them. Per-video remote recovery and
provider hard-stop/invoice reconciliation still require implementation qualification.
These trusted execution/custody/chain interfaces are not automatically provided by
loading a receipt file and still need real testnet integration qualification.

Only the current eligible certified winner gets100% miner weight. No historical
fallback and no reuse of a recycled UID. `chain.reward_publication` binds intent,
subnet, state, runtime, public custody and finalized transaction readback.
**Skipping set_weights does not clear previous weights.** The no-winner route
requires a separately qualified burn receipt. The real two-validator diagnostic
replaced both testnet weight rows with 100% to owner UID0, observed at block
8,149,032. A finalized readback at block8,149,387 confirmed 100% miner burn in a
subsequent epoch (8,149,386), with unchanged runtime and identities. This verifies
the observed testnet route, not mainnet or automatic certified-state integration;
model reward publication remains blocked by the unqualified evaluation and
execution/custody adapters. See [the measured distributed campaign](DISTRIBUTED_EVALUATION.md#measured-diagnostic-campaign-2026-10-04).
Runtime configuration background:
[official hyperparameters](https://guides.learnbittensor.org/subnets/subnet-hyperparameters).

## Cost qualification and remaining real work

The default 10 USD validator allowance includes screening, GPU, automatic
metrics, judge, startup, download, storage and shutdown. A higher allowance
needs an explicit campaign spending cap. The current two-validator diagnostic measures
real distributed work; it does not establish the cost of a full 128-prompt
evaluation or justify multiplying inference at every additional validator.

On 2026-10-04, one cold native Wan-50 video took 257.70 seconds on RTX 6000 Ada
and 77.00 seconds on H100, including text encoding, denoising and VAE decoding.
These are single development measurements, not competitive timing panels.
At the observed rental rates, 208 reference generations alone project 10.42 USD
on Ada or 5.78 USD on H100. A candidate taking half the H100 reference time would
project 8.68 USD combined GPU time, before the other costs. Its time is an assumption.
Exact bootstrap evidence aliasing reduces the judge projection from 544 to 288
calls; the historical mean implies another 1.16 USD. The whole 10 USD limit is
therefore still unqualified; do not launch a full panel from these projections.

A private experimental adapter completed a small training run with finite
gradients and frozen base weights. This establishes viability, not quality
preservation. The frozen adapter was then rendered on two fresh development prompts against
Wan-50 and untrained CFG-1 (six videos, eight blind-order Luna calls). It tied
untrained CFG-1 on both prompts; the watch scene regressed against Wan-50 in
prompt adherence, visual quality and physical plausibility. The judge is not yet
calibrated, so this is a negative development signal, not a population estimate.
No candidate was promoted and no full paid evaluation was dispatched. Both GPU
workers were removed after verified remote backups.
Real calibration and VBench qualification are
still required. Return INCOMPLETE if the bounded evaluation cannot finish, rather
than expanding its budget or inferring quality from training loss.

A subsequent bounded Luna control probe reused the two original Wan-50 development
videos (2026-10-04): 12 paired controls, 24 calls, eight predeclared primary checks
passed. Both identical-copy comparisons tied exactly; severe blur and unrelated
video content were penalized in both scenes. Freezing the moving cyclist was
penalized while freezing the legitimately static watch composition tied. Shuffling
the cyclist harmed temporal scores; shuffling the almost-static watch tied and is
not evidence of a missed visible defect. Both mild Gaussian-blur controls (radius1)
tied, so sensitivity to subtle losses remains unresolved. Some axes changed their
interpretation across blind orders, notably subject consistency on unrelated
content. These results do not calibrate the combined Luna/VBench evaluator or
qualify whole-panel power: only two reused source families were tested, with no
fresh runs, seeds or VBench scores. No promotion follows. Judge usage priced at
0.1026949 USD; storage/transfer charges remain unreconciled. The frozen plan,
raw response journal, decision replay and verification are retained in the private
experiment archive with a SHA-256-verified readback.

The subsequent VBench pilot measured five metrics on12 losslessly encoded files
from those same two sources, verifying decoded RGB hashes against the Luna inputs.
Metric processes took100.40 seconds in total on RTX4090, including model loading.
Identical-video repetitions matched exactly on every metric. Both severe-blur
controls reduced imaging quality. Mild blur changed imaging quality by-0.109198
for the cyclist and+0.002283 for the watch; these normalized score differences are
not perceptual percentages and do not establish sensitivity to subtle defects.
Freezing the cyclist improved consistency and smoothness while removing motion,
confirming that these metrics require the separate prompt-aware action gate.
The full pilot GPU bill was0.106247812125 USD, including setup and backup time;
metric-process compute alone corresponds to0.01255 USD at that rental rate.
Storage and transfer charges remain unreconciled. Neither figure is a measured
complete-candidate evaluation cost. Weights and raw receipts have verified private
backups; no fresh renders, independent calibration or promotion occurred.

The dashboard distinguishes diagnostic rounds from certified winners. A completed
local experiment is not automatically a published round or a certified king.
See [dashboard repository](https://github.com/attelierai/attelier_subnet_dashboard) for delivery and rendering checks,
and [historical diagnostics](WAN_DIAGNOSTICS_V3.md) for prior evidence.

## Measured UniPC development evaluation, 2026-10-05

A frozen 16-prompt development panel compared the unchanged Wan transformer at
40 versus 50 original UniPC steps, both CFG-5, on one RTX 5090. Each recipe
produced 16 videos at 832×480, 81 frames and 16 FPS, after two total warm-ups.
Generation timing includes text encoding, denoising and decoding; model loading
and downloads are operational costs. This is recipe optimization, not a trained
LoRA, independent confirmation or a distributed competitive round.

The prompt-paired geometric speedup was **1.247584×**. Individual ratios ranged
from 1.225323× to 1.473864×; that range is **not a confidence interval**. The
sequential A/B design does not replace the required 32-prompt ABBA timing panel.

All five VBench metrics completed. Luna completed both blind orders for every
pair (32 distinct settled responses), with no judgments marked uncertain.
Representative quality results were:

| Measurement | Mean candidate minus Wan-50 | Pairs with negative delta |
| --- | ---: | ---: |
| VBench subject consistency | -0.008843 | 11/16 |
| VBench background consistency | -0.005254 | 12/16 |
| VBench imaging quality, native MUSIQ | +1.053511 | 8/16 |
| Luna subject consistency | -0.046875 | 6/16 |
| Luna temporal stability | -0.054688 | 8/16 |
| Luna visual quality | -0.015625 | 3/16 |
| Luna physical plausibility | -0.031250 | 5/16 |

These are metric/ordinal differences, **not percentages of perceptual loss**.
One prompt per protected category cannot establish category-level preservation.
The mixed observations do not demonstrate preserved quality; the result is
`DIAGNOSTIC_COMPLETE_NOT_QUALIFIED`, with no promotion or winner. This panel
must not be reused as an independent holdout after selecting another recipe.

Evidence binds plan, video hashes, metric inputs and outputs, source versions,
metric weights and all judge response IDs. The archived DINO checkpoint actually
used matched its pinned hash. A separate public-worker smoke check failed before
GPU execution because its image provenance input was invalid; it is not a passed
runtime qualification. Private archives have full hash-verified readbacks, but
are not public model custody or public reproducibility evidence.

GPU cost is estimated at 2.739349 USD from elapsed rental time; Luna usage was
priced at 0.148043 USD, for **2.887392 USD known estimated experiment cost**.
Storage/transfer remain unreconciled. The GPU was deleted and deletion verified.
Neither this cost nor this development result qualifies the complete 128/32
candidate evaluation or its default 10 USD validator allowance.

## Public LongCat LoRA diagnostic, 2026-10-07

A public, pre-trained adapter was used as candidate work, with no new training:
`lightx2v/Wan2.1-T2V-1.3B-longcat-step1000`, revision
`c6b89d290401bba9587d532723544678c92f691e`. Its 700,000,376-byte safetensors file
has SHA-256 `50fab0581a5ba9c01a522ef9b4e223d6100df58f4ab829f9d4558e42f8fa6415`.
The loader verified 600 LoRA matrices across 300 targets and alpha/rank scaling
64/128. Public artifact download, manifest/weight admission, real generation,
native metrics and blind judgments were exercised. This was not a registered
competitive submission, independent confirmation or reward transaction.

The complete retained comparisons used an RTX PRO 4500 Blackwell, 832×480,
81 frames and 16 FPS. Both contenders used the same GPU within each paired
screening run, with separate warm-ups. Loading/download time is excluded from
generation timing and included in the operational cost estimate.

| Candidate recipe | Development pairs | Mean Wan-50 seconds | Mean candidate seconds | Geometric paired speed ratio | Motion-adherence regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA, 50 steps, CFG-4.5 | 16 | 285.416 | 322.962 | 0.883746× | 4/16 |
| LoRA, experimental 16-step UniPC, CFG-4.5 | 4 | 285.795 | 113.912 | 2.508886× | 1/4 |

The 50-step recipe was slower, despite positive visual-quality deltas on
10/16 prompts. The faster recipe does not establish preserved quality from four development
pairs, regardless of the average gains. These are development
observations, not perceptual percentages or confidence bounds. The 16-step
UniPC experiment is not the model card's default 50-step inference recipe.
Neither candidate was promoted. The result is
`DIAGNOSTIC_COMPLETE_NOT_QUALIFIED`.

The campaign completed 104 settled judge calls and 63 native metric jobs
(five dimensions per job). Twelve controls reuse two source families and do not
constitute independent calibration. Freeze, severe blur, shuffle and wrong-video
controls produced relevant negative deltas. Mild blur tied in the visual judge
while native imaging quality fell by 0.196873. A legitimate seed change also
produced a negative visual judgment, illustrating why a calibrated margin is
needed. The text-in-video injection smoke check passed its predeclared
prompt/motion bounds in both orders; it does not establish general attack resistance.

### Bugs found and verified

- A PEFT-formatted public LoRA could previously be accepted by the Diffusers
  loading call while activating zero adapters. The loader now canonicalizes its
  keys and verifies loaded matrices and effective scaling. Real public-weight
  execution and separate Diffusers/PEFT integration checks passed.
- FP16 cosine arithmetic produced a background-consistency value above 1.
  Feature arithmetic is now FP32 before the cosine, with unchanged score bounds
  and recorded runner provenance. Calibration is bound to the evaluator sources;
  these numerical fixes do not qualify old calibration receipts.
- The archive allowlist rejected legitimate Python files under `forge/models`.
  That source-only exception is fixed. An early backup failure now leaves a
  bounded repair window; an explicit reconciled retry is required before another
  transfer attempt, and the original spending deadline remains enforced.
- The dashboard exporter requires explicit current-king state. The locally
  reviewed UI distinguishes eligibility from a crown, snapshot inclusion from
  evidence replay, old snapshots from live data, and MUSIQ points from percentages.
  Native video controls work with pointer, touch and keyboard.

The final subnet suite passed 231 tests with four environment-dependent skips;
those four cases were checked separately with the actual LoRA runtime and local
Docker boundary. Dashboard values were compared with six signed reports and
native GPU/MUSIQ receipts; all 36 unique public media hashes matched. Six real
videos played in both 1920px desktop and 390px mobile views. These were local
client checks against the dated public snapshot, not a production deployment or
an assertion of live network data. Local media copies are not vendored: remote
playback/hash checks do not turn missing local media paths into passing checks.

### Custody and cost limits

An early worker archive attempt failed and 12 original GPU-generated video
files plus auxiliary metric logs were lost. Their numerical receipts were
recovered byte for byte. The key 16-step comparison was repeated as new evidence;
the main 16-prompt panel also covers the original two 50-step concepts. The final
40 retained GPU-generated videos and their complete native/judge inputs have a
hash-verified private backup. A real two-pass S3 stream replayed all 12 freshly
selected control comparisons using their 24 stored responses, with no new API calls.

Original randomized detail views could not be recovered from video hashes alone.
Fresh views used fresh request identities, while original results remain separate.
The new payloads and compact selection/image-hash ledgers are retained. The CPU
injection video was reconstructed byte for byte; its new detail views were judged
again. An archive's complete allowlist does not restore the lost original clips.

All three owned pods are absent from the provider, including one replacement
that failed to obtain a running container. Estimated total cost is
**6.27 USD / 50 USD authorized**, including
conservative GPU/disk rates, usage-priced API calls, known S3 transfers and a small
storage/request allowance. This is not a provider invoice. Unrelated account
spending and model-subscription token equivalents are excluded. Private evidence
remains in S3; no GPU remains active.

### Remaining launch gates

1. Calibrate the revised evaluator on frozen, semantically independent families,
   then validate the untouched independent split and actual panel power. The
   development prompts and newly inspected controls cannot become that holdout.
2. Complete a qualified real public-miner cycle with the required timing design,
   validator agreement, replay/custody and executing-runtime identity. A provider's
   pinned image specification alone is not runtime attestation. Verify the
   no-winner/burn and recovery paths on testnet before live reward activation.
3. Resolve the recorded-but-unprotected physical-plausibility axis in the quality
   policy before relying on it as a safeguard; any gate change requires matching
   fresh calibration. Publish the reviewed backend/UI only after those gates and
   verify the deployed data path. No deployment, mainnet activation or weight
   transaction was performed in this campaign.

### Temporal diagnostic follow-up, 2026-10-08

The expanded development run completed 88 native video measurements and 240
settled, distinct judge calls (96 base comparisons and 24 repeat comparisons).
Exact-image repetitions satisfied the predeclared tolerance in 11/12 cases.
This remains development evidence, not independent quality qualification.

A CPU-only follow-up measured all-frame temporal diagnostics on the 88 archived,
byte-verified videos. Mean luminance acceleration rose from 0.0746 to 3.8769 for
the mild photometric controls and to 31.0742 for severe controls, increasing in
8/8 of each kind. For shuffled frames, mean centered transition magnitude rose
from 3.0122 to 14.3117, increasing in 8/8 cases. These are continuous physical
signals, **not calibrated defect detections or a demonstrated natural-video
false-positive rate**. Actual codec integration checks also distinguish synthetic
flicker from gradual exposure and smooth movement; they do not replace human labels.

The four flagged independent-seed comparisons were positive differences at the
old diagnostic tolerance; none had negative differences beyond that tolerance.
They cannot be called four evaluator false positives. Human review must determine
quality equivalence before seed variation is used to fit a noise margin.

No new GPU or judge calls were needed for this follow-up. Human review, independent
calibration, protected physical-plausibility assessment and certified timing remain
pending. Production quality fusion and numerical acceptance thresholds are unchanged.
