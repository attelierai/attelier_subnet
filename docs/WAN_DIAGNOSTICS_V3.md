# Historical Wan diagnostics and legacy v3 evaluator

This document records the pre-MVP experiments and legacy commands. It is not the
current bounded evaluation contract. Use [WAN_MVP.md](WAN_MVP.md) for new work.

# Public Wan MVP

## Contract

The pinned base is `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at
`0fad780a534b6463e45facd96134c9f345acfa5b` (Apache-2.0).
`configs/wan-mvp.json` specifies 832x480, 81 frames, 16 FPS; the separate
`wan-pilot.json` uses 256x256 and 17 frames for inexpensive functional checks.
Pilot results do not establish 480p quality or production readiness.

Candidates replace the transformer or add one LoRA. Tokenizer, text encoder and
VAE always come from the approved base. No model-provided Python, pickle weights,
remote inference endpoint, private repository or gated download is accepted.
Generation configuration allows only step count and CFG scale (1–10).

Runtime: BF16 transformer/text encoder, FP32 tiled VAE, Torch SDPA, no compile,
FlowMatch Euler with shift 3 and uniform sigmas `1 - i / steps` before shifting.
The original base has a **UniPC** scheduler configuration: passing it blindly to
FlowMatch Euler loses `flow_shift`. Training and evaluation must construct the
Euler schedule explicitly. Runtime source and dependency versions enter its hash.

A full unmeasured warm-up precedes each panel. Wall time covers prompt encoding,
denoising and VAE decoding after model loading; download, load and MP4 export are
excluded. CUDA events measure transformer compute, hooks count every real forward
(CFG normally makes two per step), and CUDA records peak allocated VRAM. MP4 hashes
bind quality assessments to outputs. Pixel variance/frame delta are diagnostics,
not calibrated quality scores.

## GPU environment

On the authorized remote worker, create a project virtual environment. Install
Torch's CUDA wheel appropriate to the worker, then the pinned extra:

```bash
uv venv --python 3.12 /workspace/forge/.venv
uv pip install --python /workspace/forge/.venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /workspace/forge/.venv/bin/python -e '.[gpu]'
```

Hippius 0.6.0 needs Hub >=1 while the pinned Transformers needs Hub <1. Keep a
separate interpreter; do not combine `[gpu,hippius]` in one environment:

```bash
uv venv --python 3.12 /workspace/forge/.venv-hippius
uv pip install --python /workspace/forge/.venv-hippius/bin/python -e '.[hippius]'
export FORGE_HIPPIUS_PYTHON=/workspace/forge/.venv-hippius/bin/python
```

The bridge always uses `token=False`, immutable OCI digests and selective pulls.
It does not forward credentials or execute miner code.

## Public submission

1. Export safetensors and required configuration with an accepted license.
2. Upload the weights to a public HF or Hippius repository.
3. For the current v3 HF path, commit the immutable snapshot, candidate role and
   generation recipe in one compact claim. The validator derives files, hashes and
   public license directly from that revision. The older v2 declaration path keeps
   `forge-manifest.json` outside its own reference to avoid a self-hash cycle.
4. Obtain the **new** immutable snapshot revision after every uploaded artifact.
5. Commit the versioned claim on chain. One commitment is limited to 128 UTF-8
   bytes, so use a short repository name.
6. Validators resolve the pinned snapshot, check public access and all bytes again
   before execution. A mutable branch/tag is never sufficient.

`model-manifest` can derive HF file hashes from LFS metadata without fetching
weights. `resolve-submission` supports both providers. Reusing a valid local
cache still verifies size/hash; corrupt files are refused without overwriting.
HF SHA and Hippius OCI digest identify snapshots; the recipe also participates
in the candidate manifest fingerprint. Canonical effective-tensor deduplication
across differently packaged LoRAs remains a production admission task; a changed
manifest alone never constitutes measured improvement. [`ARTIFACTS.md`](ARTIFACTS.md)
is authoritative for public-snapshot, licensing and retention limits.

Before rendering a public round, run `forge preflight-round` over every closed
admission. Its validator-owned receipt hashes exact canonical tensor bytes, the
pinned base, contract and `{steps, guidance_scale}`. Repackaged or forked
snapshots with the same tensors and recipe are marked duplicate before GPU or
vision spend; the same tensors at a different step count remain a distinct
candidate. An unavailable or structurally invalid admitted snapshot receives an
`INCOMPLETE_CANDIDATE_PREFLIGHT` row (`fetch` or `validation`) and cannot render,
be judged or promote; it does not stop the remaining closed admissions. The receipt
does not identify algebraically equivalent LoRA factors or backfill an exhausted
candidate cap; distinct-weight Sybil quota exhaustion remains an open queue-design issue.
Public `render-panel`, `render-paired` and production `compare-automatic`
require the matching receipt; `--experimental-adapter` and `--development`
remain diagnostic-only paths.

Example on the remote worker:

```bash
forge model-manifest --repo Wan-AI/Wan2.1-T2V-1.3B-Diffusers --revision 0fad780a534b6463e45facd96134c9f345acfa5b --role pipeline --steps 50 --out /workspace/forge/base.json
forge panel-plan --contract configs/wan-mvp.json --round-id r123 --stage screen --block-hash <finalized-after-close-hash> --out /workspace/forge/plan.json
forge render-panel --manifest /workspace/forge/base.json --base-manifest /workspace/forge/base.json --cache-dir /workspace/forge/models --plan /workspace/forge/plan.json --reference --scheduler original_unipc --out /workspace/forge/golden
```

For chain-bound admission, use `public-round-plan --network test --netuid <N>
--close-block <B> --entropy-block <E> --contract configs/wan-mvp.json --out <file>`.
With the production sealed-pack contract, also supply `--prompt-pack <revealed-pack.json>`
and `--pack-hotkey <committing-validator-hotkey>`. It checks stage-specific entropy
and finality, freezes committee/panel and resolves eligible public commitments.
Pass the resulting plan to `render-panel`, which checks candidate membership and
regenerates prescribed prompts. This command does not publish weights or crowns.

## Quality and winner gate

`compare-panels` requires a trusted validator policy (`runtime_id`,
`evaluator_id`), a prescribed plan and reports for golden, incumbent and challenger.
An evaluator must supply all eight contractual axes on [0,1], exact prompt/seed
slots and each corresponding video hash. `calibrated: true` records the evaluator's
claim; it is not proof by itself. Only accept locally produced or authenticated
validator evidence, never a miner-supplied scored JSON. Raw runtime reports deliberately retain `quality: null`. The automatic Luna
workflow below writes separate blinded evidence and recomputes its control
calibration; a raw report or a self-declared boolean cannot promote a model.

Promotion eligibility requires a paired speed lower bound of at least 1.1 and
noninferiority of all quality axes against both the fixed golden and incumbent.
Missing slots, inconsistent runtimes/geometry, nonfinite scores, an unapproved
judge or infrastructure failure leave the comparison incomplete. Private local
adapters are marked `private_experiment` and are ineligible. Exact manifest copies
are rejected. Quality is anchored to the original reference to prevent drift.

The latest **certified** winner takes 100%; historical crowns do not split or
expire into rewards. If that hotkey is ineligible, the share is unallocated, not
redirected to an older winner. The baseline earns no crown.

## Remaining activation gates

- Validate automatic evaluator sensitivity and repeatability at 480p on a separate
  control panel. No human labeling is used or required by this MVP. A small
  diagnostic pilot is not a calibrated quality-preservation certificate.
- Connect real reports to distributed commit/reveal/confirmation; the local signed
  record and testnet anchor are evidence from one validator, not an operational
  validator network or a quorum certificate.
- Require fresh public-availability evidence immediately before any future WTA
  weight submission: credential-less resolution of the pinned winner reference,
  its reusable license and a fresh verified fetch of every manifest file to a
  clean destination. A valid cache or the admission-time receipt is insufficient:
  pinning a revision cannot stop an owner from deleting or gating a repository.
  If that check fails, the current winner's share must remain unassigned until a
  later independently verified availability check passes; it must never fall back
  to an older crown.
- Run independent validator reproduction and an authorized testnet round with
  real submissions before enabling real-model reward transactions.

Private experiments are not admitted candidates until their weights are published
and independently scored.

## Automatic evaluation without human labeling

The competition concerns public candidate/base weights; the operator may use the
explicitly selected external Luna judge. `gpt-6-luna`, reasoning `high`, receives
images through the Responses API with structured output, no tools, and `store:false`.
It does not accept video natively. The v3 evaluator supplies **every frame** as
three chronological sheets (81 thumbnails total), a native-resolution sheet of nine
secret post-render frames, three matched 4x `NEAREST` crops, and a sixth sheet of
nine secret stratified transitions. Each transition row shows `t`, `t+1`, and their
absolute pixel delta; it locates discontinuities but does not award motion or quality.
The crop and transition selections are made post-render, persisted, and shared by A
and B. This six-image-per-side package is 29,886 image patches per pair, below the
30,000-patch envelope. It cannot claim playback-level motion perception or universal
perceptual equivalence.

For each prompt/seed Luna compares anonymous A/B outputs twice, reversing their
order. Neither model identity, steps, speed nor training method enters the rubric.
Eight anchored ordinal scores are normalized to candidate-minus-reference deltas;
raw responses, provider request IDs, request-image hashes, prompt/seed, media hashes,
input recipe and billing usage remain in the receipt. V3 uses the symmetric mean of
both orders per axis, then clusters seeds within a prompt before the quality interval.
Order magnitude and sign-flip diagnostics remain explicit evidence; they do not turn a
single ordinal disagreement into a second pseudo-prompt or a minimum-score penalty.
Malformed schema, uncertainty, non-inverted image evidence, or missing hashes still
make the observation incomplete. Pixel diagnostics cover all frames but are not
quality scores. Before any paid judge request, every decoded source video also passes
the fixed raster-integrity preflight. A raster anomaly makes the artifact `INCOMPLETE`
with `RASTER_INTEGRITY_ANOMALY`; it is not a miner quality penalty or a model-failure
label.

Two references serve distinct purposes:

- **Golden quality anchor:** original public Wan base, original UniPC scheduler,
  50 steps. It prevents quality drifting down after repeated winner replacements.
- **Incumbent:** the current recipe in the fixed uniform Euler speed lane. Candidate
  quality must also be noninferior to it. Paired timing runs one pipeline, two warm
  calls, then ABBA (incumbent, candidate, candidate, incumbent) per prompt/seed with
  CUDA synchronization. Timing includes text encoding, denoising and VAE decoding;
  downloads, model loading and MP4 export are outside the interval. Both raw repeats
  remain auditable. The MVP supports base/LoRA and LoRA/LoRA, and its
  full-transformer lane swaps verified CPU states into one resident GPU transformer.
  Mixed full-transformer/LoRA timing and an experimental adapter in that lane fail
  explicitly.

The evaluator itself is tested using deterministic self/copy, frozen, blurred,
shuffled-frame and wrong-video controls. At least 32 independent, informative prompts
per control family are required. Exact binomial intervals must put sensitivity above
0.8 and the self-control false-alarm rate below 0.15, with a familywise 0.05 budget.
A non-self transformation whose decoded pixels equal its source is `N/A`, never a
miss or a success; it reduces the informative sample and can make calibration
inconclusive. V3 calibration evaluates the symmetric mean and records order
magnitude/sign-flips as diagnostics. Calibration sources must be fresh, machine-known
golden controls, separate from both candidate selection and the confirmation panel.
No v3 calibration has passed yet.
The control artifact is recomputed by the gate, including all blinded calls;
changing a `PASS` field is insufficient. These controls validate specified defects,
not agreement with human taste. They must be rechecked in each evaluation campaign:
Luna currently exposes a mutable alias, not an immutable model snapshot.

`configs/wan-mvp.json` retains grammar v3 only to reproduce existing pilot receipts.
Production requires a newly committed `forge/sealed-pack/v1`, then deterministic
post-entropy selection from its revealed entries; there is no procedural fallback.
The pack must balance the protected subject, camera and style slices, but its
metadata cannot itself prove semantic novelty or secrecy. See
[`PROMPT_PACKS.md`](PROMPT_PACKS.md) for the authoritative commitment/reveal rules.
Development, calibration and final confirmation panels must be separate. Never
select candidates or tune a judge on its final confirmation results.

The independent statistical unit is the **prompt**, averaging its seeds and timing
repeats. Quality uses bounded fixed-mixture betting intervals per axis globally and
per subject class, plus aggregate camera/style slices. Speed uses paired log ratios
and a Student-t interval (an approximate sampling model, not a hardware guarantee).
Bonferroni accounts for both tails, two quality anchors, protected comparisons
and up to 16 candidates. Fixed bets are selected before seeing observations; no
data-dependent bet or sample stopping rule is used. The method follows the
[bounded-mean betting framework](https://arxiv.org/abs/2010.09686). A candidate needs quality LCB >= -0.02 and speedup LCB >= 1.1, plus a
valid calibration and public artifacts. Evidence overlapping a threshold stays
inconclusive. Only a fresh confirmation panel may be eligible; local evidence is
not itself a distributed crown or a chain write.
`PROMOTE_ELIGIBLE` currently means that a trusted operator's local evaluation met
these gates. Its confirmation plan locally binds the admitted candidate, incumbent
state and sealed panel, and the resulting record can be hotkey-signed and anchored
on testnet. That is not a final chain admission, a certification ceremony or a
quorum. Its raw judge JSON and controls are operator-produced evidence, not remotely
authenticated provider attestations. Do not feed this status directly into rewards:
transport and independent validator reproduction remain activation gates. See
[`TESTNET.md`](TESTNET.md) for the exact record boundary.

There is **no magic sufficient prompt count**. The updated configuration uses
32 screen prompts, 256 duel prompts and 4,096 confirmation prompts, with two seeds
in the latter two stages. These are planned sizes, not a guarantee of sufficient
power. With the fixed mixture and current multiplicity correction, 628 tied
independent prompts per protected slice are enough to exclude an expected decline
of 0.02. Confirmation supplies 1,024 per subject class; observed score variability
can still require more or produce an inconclusive result. This avoids the former
Bernstein bound that needed over 3,400 ties per slice, without changing the margin
or dropping individual quality axes. The gate reports the tied-data precision
floor explicitly. Freeze sample size and thresholds before evaluating a candidate.
This production-sized confirmation can cost far more than a small GPU pilot; it is
never implied by the pilot's authorization or its budget. It contains 8,192 prompt/seed
slots. Quality reuses the videos from the ABBA reports and is 32,768 Luna calls
(two references x two orders). The v3 historical-Wan replay used 64 calls costing
an estimated 0.2877128 USD from reported usage, or 0.0044955125 USD per call.
Extrapolating that observed mean gives about 147.31 USD for confirmation judging,
before fresh calibration; this is not a quote or a maximum. A single validator
executes 40,960 unique renders: 8,192 golden renders plus 32,768 ABBA renders.
For the historical L40S 12-step-candidate versus 16-step-incumbent scenario, using
its measured 50-step UniPC golden mean of 219.6684193595 s and ABBA means of
79.541097 s and 63.027788 s, this is about 1,148.7145 GPU-hours:
`8,192 * (219.6684193595 + 2 * (79.541097 + 63.027788)) / 3,600`.
At the historically observed 0.38--0.42 USD/h rental rate, that illustrative
per-validator execution is 436.51--482.46 USD in GPU time. It assumes no shared
baseline-golden cache across candidates, even where a private fresh panel might
later permit one. Recipe changes, queueing, failure retries and provider
availability make this neither a bound nor a delivery-time estimate. The 32,768
judge calls also need measured provider throughput before asserting that they fit
the 24-hour calibration freshness window.
The 0.02 margin is on the normalized ordinal rubric, not a claim of "2% perceptual
quality loss". Neither a finite prompt panel nor a vision judge can prove zero
degradation for every future prompt. The result applies to the frozen challenge
population, measured axes and accepted automatic evaluator. The margin is an
initial engineering policy; it has not been calibrated to perceptual preference.

The design borrows validator-owned measurement, paired timing and separate fidelity
checks from [Cacheon's pinned scoring design](https://github.com/latent-to/cacheon/blob/5763f5643cc2d30aaa9bcb9146776bd960306ef5/docs/validator-guide/scoring.md).
That current SN14 repository targets contributions inside a validator-owned SGLang
engine. Its token teacher-forcing fidelity tests do not transfer directly to Wan
distillation: lower latent MSE cannot replace evaluation of rendered videos.
This is a comparison of code/design, not verification of SN14's live deployment.

The public-storage flow also follows the pattern in
[Albedo's pinned mining guide](https://github.com/unarbos/albedo/blob/7cf5f2a45df149a1b6fc76af4db3713a90f2d8d9/docs/MINING.md):
miners publish an immutable HF revision or Hippius digest and validators download
and evaluate it. Its two separate qualifying duels support keeping discovery and
confirmation apart. Attelier additionally requires every admitted artifact to be
public; a private research checkpoint cannot enter the reward competition.
Do not infer current Affine mechanics from old Chutes documentation: the
[SN120 checkout reviewed on 2026-10-04](https://github.com/AffineFoundation/affine/blob/887c467ce723a09f9268afc8dbc34752178719b1/README.md)
describes an epoch training/audit loop, and its
[scoring function](https://github.com/AffineFoundation/affine/blob/887c467ce723a09f9268afc8dbc34752178719b1/subnet/scoring.py)
normalizes unique accepted contributions among miners. That source does not
establish the requested winner-takes-all policy or a current Chutes billing flow.
Attelier's WTA is its own explicit protocol rule, not a claim about SN120's live state.

Install the optional evaluator in its own project environment:

```bash
uv pip install --python .venv/bin/python -e '.[dev,vision]'
forge vision-policy --out /workspace/forge/policy.json
forge preflight-round --plan /workspace/forge/plan.json --base-manifest /workspace/forge/base.json --cache-dir /workspace/forge/models --out /workspace/forge/preflight.json
forge render-paired --base-manifest /workspace/forge/base.json --incumbent-manifest /workspace/forge/incumbent.json --manifest /workspace/forge/candidate.json --cache-dir /workspace/forge/models --plan /workspace/forge/plan.json --preflight /workspace/forge/preflight.json --out /workspace/forge/duel
forge judge-controls --plan /workspace/forge/calibration-plan.json --golden /workspace/forge/calibration-golden/report.json --out /workspace/forge/calibration.json --ledger /workspace/forge/api-cost.json --api-budget 2
forge judge-panels --plan /workspace/forge/plan.json --golden /workspace/forge/golden/report.json --incumbent /workspace/forge/duel/reference/report.json --challenger /workspace/forge/duel/candidate/report.json --out /workspace/forge/quality.json --ledger /workspace/forge/api-cost.json --api-budget 2
forge compare-automatic --plan /workspace/forge/plan.json --policy /workspace/forge/policy.json --quality /workspace/forge/quality.json --timing /workspace/forge/duel/paired-timing.json --calibration /workspace/forge/calibration.json --preflight /workspace/forge/preflight.json --out /workspace/forge/decision.json
```

Render both golden reports with `render-panel --reference --scheduler original_unipc`.
Outside `--development`, create one matching `preflight-round` receipt before
public candidate render/timing and pass it again to `compare-automatic`; the judge
also requires `CLEAR` raster receipts binding every reviewed video hash and geometry.
Generate the separate calibration and evaluation plans with a common `--campaign-id`
and different entropy/stage or round IDs. Before running controls, hash the exact
calibration plan; the confirmation plan must declare that hash and production rejects
a receipt from another plan. Production also rejects reuse of the calibration panel and
requires its calibration to precede evaluation by at most 24 hours. This prevents
substituting a different calibration plan after it is declared; it does not prevent
retries of that same plan, ex-post confirmation-plan creation, or concealed failures
without an independent append-only publication mechanism.
Use one API ledger and a budget that is a subset of the authorized session limit;
its example `2` is not automatic spending authorization. Each dispatch reserves
0.08 USD against a bounded input/output envelope. Settled provider usage is priced
conservatively at standard uncached rates; failed/unknown calls retain reservations
and are never automatically retried. The project loads its own ignored `.env`
(or `--env-file`) for `OPENAI_API_KEY`; do not copy credentials to GPU workers.
Running the judge locally with temporary RAM-backed media is supported; durable
videos, checkpoints and their backups remain on the authorized remote storage.
Both judging commands accept `--workers 1..8` (default 1). Each worker completes
both orientations of one pair; the shared locked budget remains the total cap.
Small batches bound in-flight work, and every provider response is journaled before
parsing. No failed request is automatically retried. Choose concurrency for the
available memory and API limits; the full confirmation workload must fit within
the campaign's 24-hour calibration window.

### Frozen spatial/raster probe evidence

The frozen policy was probed through actual encoded-and-decoded media before the
32-per-family Luna controls began. The receipts
`.codex/receipts/frozen-probe-checker1.json`,
`.codex/receipts/frozen-probe-checker2.json` and
`.codex/receipts/frozen-probe-image-instructions.json` are all `COMPLETE` and their
specified probes were detected. `checker1` also produced `RASTER_INTEGRITY_ANOMALY`
while its source was `CLEAR`; `checker2` and `image-instructions` were `CLEAR` under
the raster rule. Historical fixed first/middle/last and native-nine reviews both
missed one pixel-level case, motivating the recorded secret crop coverage.

One probe per attack family is not a sensitivity calibration. The raster rule is a
conservative repeated-Nyquist heuristic; it can false-positive on legitimate
patterns and it does not establish detection of semantic, low-frequency, or other
adaptive attacks. The production action is therefore evidence incompleteness, never
a score deduction or automatic miner penalty. The subsequently completed
32-prompt control campaign failed the temporal-order sensitivity requirement;
the small spatial probes do not override that failure. The completed
`temporal-v3-probe-result.json` is a separate diagnostic: eight procedural scenes
from one family, 24 constructed pairs and 48 calls. It met its preregistered criteria:
4/4 detections each for shuffled motion, reversed direction and incorrectly frozen
motion; 0/12 false alarms across identical copies and legitimately static scenes;
no invalid pairs. Five pairs had order disagreement above 0.25, recorded as a
diagnostic rather than discarded. This supports observability of these constructions
in v3's six-sheet representation, not a Wan population calibration or promotion.

The same frozen v3 then re-examined **all 32 historical Wan shuffle cases**, without
selecting by outcome: 64 calls, 10/32 flags, no invalid pairs and four pairs with
order disagreement above 0.25. Recomputing the old v2 calls with the same mean
aggregation gives 11/32 flags (the historical minimum-based rule gave 19/32).
This replay shows no detection improvement in that material. It is not an
independent calibration or a controlled ablation of transition strips: the old
cases informed v3 design, and a shuffle may not visibly damage a nearly static
source. The constructed success therefore does not justify activation. Raw
receipts remain separate and unchanged; the new receipt is
`.codex/receipts/wan-temporal-v3-replay.json`. No reward or model promotion resulted.

## Evaluation research checked on 2026-10-04

[VBench](https://arxiv.org/abs/2311.17982) separates temporal smoothness, subject
consistency and other video dimensions. [VBench-2.0](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md)
extends evaluation to 18 dimensions, including physical and human-related
capabilities. Its implementation documents custom-input support for only a subset
of dimensions; it is not a drop-in judge for arbitrary sealed prompts.

[T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench) combines multimodal
judging, detection and tracking across seven compositional categories and 1,400
public prompts. In particular, motion binding uses object segmentation and
tracking. These are useful examples of distinct checks for different failure modes.

Our next design should test motion intent and continuity separately, including
legitimate static scenes and independent-seed same-model controls. This is a design
inference, not an implemented or calibrated replacement. Public benchmark prompts
can inform the taxonomy but must not become the subnet's fixed reward holdout.
No new evaluator weights or dependencies from these projects were installed;
their automatic components still require artifact/license checks and fresh
adversarial calibration before any adoption. Human labeling remains excluded.

## Measured overnight evaluation, 2026-10-04

The frozen evaluator completed 32 distinct source prompts, separate from the
eight-prompt candidate panel, with five controls and two blinded orders per source
(320 API calls). No observation was invalid. All unmodified source videos passed
the raster guard.

| Control | Detections / sources | Adjusted one-sided bound | Result |
| --- | ---: | ---: | --- |
| Identical copies | 0 / 32 false alarms | Upper 0.1340 | Pass |
| Frozen | 31 / 32 | Lower 0.8099 | Pass |
| Blurred | 32 / 32 | Lower 0.8660 | Pass |
| Shuffled frames | 19 / 32 | Upper 0.7881 | **Fail** |
| Wrong video | 32 / 32 | Lower 0.8660 | Pass |

These are exact binomial bounds adjusted across five control families. The
shuffle sensitivity is conclusively below the required 0.8 under this test. The
automatic gate therefore blocks promotion. It does not establish that every
shuffle is perceptually equally severe, nor measure quality on arbitrary future
video distributions. See [the threat model](THREAT_MODEL.md) for the temporal
failure, exploratory interpolation measurements and their limits.

On one L40S, the prescribed 32-call ABBA experiment over eight prompts measured:

| Public base recipe | Mean wall time | Transformer forwards |
| --- | ---: | ---: |
| Uniform Euler, 16 steps | 79.5411 s | 32 |
| Uniform Euler, 12 steps | 63.0278 s | 24 |

The paired geometric speedup was **1.2620x**, with descriptive 95% interval
**[1.26118, 1.26282]**. The eight-axis mean ordinal quality delta of the 12-step
recipe was **-0.1523** against original 50-step Wan and **-0.0938** against the
16-step incumbent recipe. These use the minimum of both blinded-order deltas;
they are uncalibrated diagnostics, not percentages or a quality guarantee. The
main dashboard exposes every axis and two preselected scenes with the exact
golden, incumbent and candidate videos used in the judgments.

Two public research checkpoints were also rendered on the same L40S with three
conditional forwards, using dense BF16 attention and the original FP32 Wan VAE:

| Exploratory adaptation | Mean generation time | Mean ordinal delta vs original Wan |
| --- | ---: | ---: |
| FastWan dense | 19.5064 s | -0.0664 |
| QAD dense | 19.4473 s | -0.0820 |

Each row has eight measured videos after a warm-up. These are unpaired timings
and adaptations of the authors' runtimes, so no paired speedup, native-runtime
equivalence or public-candidate admission is claimed. No trained model in this
session has demonstrated preserved quality.

The signed 12-step evaluation record was finalized on testnet netuid 578 at
block **8,145,097**. It explicitly records `DEVELOPMENT_DIAGNOSTIC`, the failed
calibration and the evidence hashes. It is neither a crown nor a reward update;
[testnet documentation](TESTNET.md) describes its verification boundary.

### Native 64-update training diagnostic

A separate RTX A6000 run trained a private experimental LoRA adapter for 64
updates at 81x480x832 with all base parameters frozen. The training loss is not
a quality metric.

On eight reserved prompts, exactly disjoint from the 64 training prompt hashes,
adapter8 versus unchanged base8 used identical initial latent seeds and the frozen
two-order Luna policy. There were no invalid pairs. The mean ordinal difference
was **+0.00390625** using the minimum of orders, or **+0.0078125** using their mean.
Seven axis means were zero; motion quality was +0.03125 under the minimum rule.
This small, uncalibrated result is practically a tie, not demonstrated improvement
or preservation of the original 50-step model. Semantic split independence has
not been established. The private adapter is not an admitted public candidate.

The render reports separately measured 76.6076 s for base8 and 79.3526 s for
adapter8 on that GPU. These single renders per prompt exclude cached text encoding
and were not an interleaved ABBA timing experiment; they do not establish a speedup.

The preregistered comparison against the frozen **16-step base recipe** subsequently
completed on all eight reserved prompts with no invalid judgments. The adapter's
mean ordinal difference was **-0.203125** under the minimum-of-orders rule and
**-0.154296875** under the mean-of-orders diagnostic. Visual quality was the worst
axis: **-0.34375** using minima and **-0.28125** using means. The negative signal
therefore persists without taking the minimum of judge orders. This experiment
does not demonstrate quality preservation even against the 16-step base recipe and
is not eligible for promotion. The 16-step recipe separately averaged **129.4403 s** on
the same GPU; these unpaired, text-embedding-cached renders do not establish a
certified speedup. No new 50-step preservation claim or reward follows from them.

A preregistered control generated the **same unchanged base8 model** with a second
independent seed for each of those eight prompts and randomized which realization
was called the candidate. Across the eight axes, averaging judge orders gave
**-0.0390625**, whereas taking their minimum gave **-0.125**: an additional
downward shift of **0.0859375** caused by aggregation. Individual seed realizations
need not have equal quality, and eight pairs do not estimate a calibrated
population false-rejection rate. The difference between aggregations nevertheless
demonstrates their behavior on unchanged-model outputs. It does not justify
retuning the frozen policy on this panel or promoting any model.

## Measured pilot, 2026-10-03

One H100 80GB ran the pinned public base and a small private experimental LoRA
targeting one-step uniform Euler generation. Its training diagnostics are not
evidence of preserving golden video quality.

An independent validator process loaded the exported safetensors successfully.
One full-format smoke prompt (832x480, 81 frames) measured:

| Model recipe | Wall time | Transformer forwards | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| Base, 16 steps | 28.72 s | 32 | 16.82 GB |
| Base, 1 step | 7.71 s | 2 | 16.82 GB |
| LoRA, 1 step | 7.81 s | 2 | 16.84 GB |

The reduction comes from fewer steps; LoRA adds slight overhead versus the
one-step base. Every row generated a valid MP4 with a recorded hash. There was
one unmeasured warm-up per recipe. One prompt does not establish a speed confidence
bound, and no calibrated video-quality result exists. The comparison gate returned
`NOT_ELIGIBLE` when applied to actual unscored reports. No model was crowned or
rewarded. Any public testnet commitment is documented in [`TESTNET.md`](TESTNET.md);
it is not a quality certificate for this private pilot.

### Automatic 480p follow-up

A second bounded H100 run evaluated the same fixed experimental LoRA on eight
new stratified v2 prompts, one seed each, at 832x480, 81 frames and 16 FPS. The
original 50-step UniPC anchor produced eight valid reference videos. The timed
comparison used two full warm-ups followed by 32 ABBA calls on the same GPU:

| Recipe | Mean wall time | Measured forwards per call |
| --- | ---: | ---: |
| Base, 16 uniform Euler steps | 28.6315 s | 32 |
| Experimental LoRA, 1 uniform Euler step | 7.8131 s | 2 |

The prompt-paired geometric speedup was **3.6645x**, with a descriptive 95% t
interval of **[3.6584, 3.6706]**. This small development interval is not a
multiple-candidate confirmation certificate. Reduction to one step is the main
source of the speed gain; it does not establish the LoRA's contribution.

Luna judged both references in both A/B orientations (32 calls). Raw mean visual
quality deltas were -0.6406 versus golden and -0.5469 versus the 16-step recipe;
subject consistency was -0.7188 against both. These are uncalibrated ordinal
diagnostics, including unstable judgments, not perceptual percentages. One of
eight golden comparisons and five of eight incumbent comparisons exceeded the
order-disagreement limit. No quality preservation was demonstrated.

Forty control pairs (80 calls) produced:

| Control | Stable detections / prompts | Unstable judgments |
| --- | ---: | ---: |
| Identical copies | 0 false alarms / 8 | 0 |
| Frozen video | 4 / 8 | 4 |
| Blurred video | 7 / 8 | 1 |
| Shuffled frames | 4 / 8 | 1 |
| Wrong video | 3 / 8 | 5 |

Calibration was **INCONCLUSIVE**: sample size is below 32 per family and several
judgments are unstable. The development gate returned `INCOMPLETE`; the production
gate also rejected the private artifact. This historical pilot used the v1
average/order-hard-failure diagnostic and cannot establish the later v2
minimum-of-orders policy. No threshold was relaxed to obtain a pass. This judge
configuration is not ready to award production rewards. Resolving temporal/order
sensitivity is necessary before a larger confirmation or a claim that an accelerated
model preserves quality. Human labeling remains excluded.
