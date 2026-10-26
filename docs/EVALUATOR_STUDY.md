# Open evaluator study

Status: **development tooling and A40 GPU execution verified; independent validation
and blinded human calibration pending**. No reward or chain publication
path consumes these results. Target competition: [MODEL_COMPETITION.md](MODEL_COMPETITION.md).

## Decision this study can change

Determine whether a local, open evaluator is reliable enough to pursue qualification
instead of the existing Luna/native fusion. Compare identical prompt/video bytes,
record failures, repeatability, GPU memory and latency, then challenge the evaluator
using score-optimized distortions. Reject the route if useful quality discrimination
cannot survive attacks; do not widen margins or keep adding prompts until it passes.

The candidate is [VideoReward](https://github.com/KlingAIResearch/VideoAlign), with
separate VQ, MQ and TA heads. Code and model revisions, all consumed asset hashes,
preprocessing and direct dependency versions are pinned in
[`configs/videoreward-study-v1.json`](../configs/videoreward-study-v1.json).
The model card declares Apache-2.0; the upstream code has an MIT license. This is a
candidate evaluator, not evidence of exploit resistance or complete physics coverage.

The adapter preserves upstream preprocessing/normalization and uses SDPA in place of
FlashAttention. It loads the exact checkpoint with `weights_only=True` and a strict
state dictionary; it never executes a miner-supplied module. Base processor and model
both use the verified local snapshot, avoiding upstream's independent `main` lookup.
This SDPA/runtime variant passed the A40 smoke test and 12 exact inference repeats
on 2026-10-14; that does not establish repeatability on other hardware/runtimes.
The initial pinned Torch environment targets A40/4090, not RTX 5090; Blackwell needs a
separately identified, measured runtime. No local weights are downloaded by reports.

VideoReward samples at 2 FPS by default. All-frame temporal diagnostics accompany
every score; they are not yet calibrated vetoes. Legitimate flashes, cuts and camera
motion must remain in validation. Do not turn high temporal energy into “bad quality”.
The three heads are not summed for quality decisions. A sum is used only as one attack
search objective, alongside each individual head.

## Implemented workflow

The following commands use the owning repo as the working directory. Historical
receipts/media are private operator inputs; no private corpus is checked into Git.
Choose new output names: completed plans/reports are never overwritten.

```bash
.venv/bin/python -m forge.evaluation.evaluator_study freeze-pilot \
  --pilot configs/evaluator-pilot-manifest-v1.json \
  --media-index /private/archive/media-index.json \
  --legacy /private/archive/result-memory.json \
  --spec configs/videoreward-study-v1.json --repeats 12 \
  --out /private/study/comparison-plan.json

.venv/bin/python -m forge.evaluation.evaluator_study report \
  --plan /private/study/comparison-plan.json \
  --out /private/study/pending.json
```

`freeze-pilot` reconstructs the original 96 comparisons, binds input digests and
deduplicates scoring by **video bytes plus the evaluated prompt**. A wrong-video
control is scored with the reference prompt, not its donor's generation prompt.
Exact inference repeats are distinct requests; they are not filled from the cache.
The October archive has 88 media jobs but 80 distinct byte hashes; including wrong
prompts yields 88 unique scoring samples, plus 12 repeats = **100 calls**. These are
16 already-known development families, not 96 independent examples.

Without GPU receipts the report is `INCOMPLETE`, all open deltas remain missing, and
historical fusion values remain visible. Complete reports still say `qualification:
prohibited` and emit no quality verdict. Invalid, duplicate, stale, mixed-runtime,
non-finite or wrong-media receipts are rejected. Current and old scores have different
units: compare error patterns after human labels, not their numeric scales.

On an **authorized remote worker only**, prepare a separate environment under the
owning checkout. Do not replace the generator's environment:

```bash
uv venv --python 3.12 .venv-video-reward
uv pip install --python .venv-video-reward/bin/python \
  -r configs/videoreward-requirements.txt
export PYTHONPATH="$PWD/src"
.venv-video-reward/bin/python -m forge.evaluation.video_reward download-assets \
  --spec configs/videoreward-study-v1.json --assets /workspace/forge/reward-assets
.venv-video-reward/bin/python -m forge.evaluation.video_reward measure \
  --plan /workspace/forge/study/comparison-plan.json \
  --assets /workspace/forge/reward-assets --media-root /workspace/forge/restored \
  --out /workspace/forge/study/receipts --deadline-unix <approved-absolute-deadline> \
  --max-new-requests 4
```

The four-request smoke run returns exit **3** (`INCOMPLETE`); inspect its receipts and
project the remaining lease before resuming the same plan/output without that limit.
Do not mistake this expected partial exit for approval to repeat completed requests.
Asset downloads are explicit and pinned. `measure` has no download path, verifies all
source bytes, records implementation/environment identity and resumes only compatible
receipts. A process deadline bounds execution; the existing external provider guard
must separately terminate paid compute at the approved lease/budget limit. A process
exit does not stop provider billing. Preserve receipts before termination.

Transfer only small receipts back, then run `report --receipts <receipt-directory>`.
Runtime totals measure the reward call and exclude the separate all-frame CPU
decode/diagnostics, installation/download, model loading, warmup and idle time.
Reconcile those against the provider lease rather than treating reward-call latency
as end-to-end capacity or total cost. Receipt write timestamps can bound observed
throughput, but do not individually measure all these stages.

For the first adaptive screen, prepare **four development families** on the worker:

```bash
.venv/bin/python -m forge.evaluation.evaluator_attacks prepare \
  --plan /workspace/forge/study/comparison-plan.json \
  --media-root /workspace/forge/restored --families 4 \
  --out /workspace/forge/attack-study
```

This writes a sealed plan and real encoded variants: freeze/shuffle, three hold-frame
strides, four brightness phase/amplitude combinations, contrast, sharpness and a
reward-text overlay. At most 52 samples including references. Run `video_reward
measure` on that plan, with `--media-root /workspace/forge/attack-study`, then:

```bash
.venv/bin/python -m forge.evaluation.evaluator_attacks select \
  --plan /workspace/forge/attack-study/plan.json \
  --receipts /workspace/forge/attack-receipts \
  --out /workspace/forge/attack-selection.json
```

Selection requires the complete frozen query budget. It retains all receipts and
selects each head's maximum plus the sum maximum, with deterministic ties. Selected
cases remain `UNREVIEWED`; high reward on a distortion is a review candidate, not
automatically a successful exploit. This bounded black-box search does **not** test
white-box trained adversaries or general semantic diversity collapse.

## Independent validation, after this development screen

1. Construct new source/concept families with recorded provenance. Separate training/
   attack development, margin calibration and final validation at the family level;
   descendants and transformations cannot cross splits. Existing October families
   stay development permanently. A public benchmark/model may already know particular
   examples; document that risk rather than claim universal unseen prompts.
2. Include legitimate difficult motion, camera moves, flashes/cuts, exposure changes,
   physics/interaction, prompt composition, static tasks and genuine different seeds.
   Add model/recipe improvements and obvious controls, with several seeds for diversity.
3. Train or adapt a miner against the public evaluator on development families, including
   reward maximization and collapse/retrieval strategies. Freeze the attack and all
   thresholds before final families are scored. The implemented distortion search is
   only the inexpensive first screen for this larger test.
4. Human review comes last, with full videos and blinded identities/order/scores. Label
   acceptable quality, critical regression and uncertain cases; do not infer equal
   quality from identical model names. Include random unselected cases as well as attack
   maxima. Freeze the rubric, reviewer allocation and adjudication before review.
5. Estimate discrimination and non-inferiority with family-level uncertainty, keeping
   missing/uncertain cases in the denominator. Learn margins on calibration only;
   evaluate once on the held-out families. Freeze multiple-comparison and stopping
   rules before scoring. Exact sample size depends on required precision and measured
   variance; 16/128/32 is not inherited as a scientific requirement.

Failures: systematic high-reward unacceptable videos, unacceptable false alarms on
legitimate variation, poor repeatability, or inability to fit measured capacity.
Inconclusive: missing media/scores, unreviewed equivalence, inadequate independent
families or confidence intervals too wide. Neither result may activate rewards.

## Measured development pilot — 2026-10-14

The authorized RunPod A40 pilot completed **100 comparison calls and 52 attack calls**,
plus three unrecorded warmup calls. The comparison plan was frozen before scoring;
all receipts share the same evaluator/runtime identity. There were no new Wan renders,
training, commercial-judge calls or paid human labels. Both reports are `COMPLETE`,
with `qualification: prohibited` and `quality_decision: NOT_EMITTED`.

| Check | Observation |
| --- | --- |
| Exact inference repeats | 12/12 identical on all three heads; maximum absolute change 0 |
| Mild blur, severe blur, detail loss | VQ decreases in 8/8 pairs of each kind |
| Wrong-video controls with reference prompt | TA decreases in 8/8 pairs |
| Frozen motion controls | MQ increases in 6/8; TA decreases in 7/8 |
| Archived photometric controls | No head decreases in 3/16 transformations, from two static source families |
| Additional bounded search | 48 transformations on four motion families; no case has all three deltas nonnegative |
| Individual/sum search objectives | Frozen clips win the MQ search in all four families, improving MQ in three; the best sum increases in three families, twice with reward text and once with frame holding |

These are **raw delta signs**, not calibrated pass/fail rates. A rising sum cannot
compensate for a protected dimension losing quality, and the sum is not a proposed
reward rule. The attack-family result does not cancel the archived photometric
finding: these are different, already-known source families. Re-encoding is part of
each generated treatment; without a matched no-op encode and neutral-text controls,
the text result cannot establish instruction following or isolate a prompt-injection
effect. Selected clips remain unreviewed. The comparison's self/static/same-run zero
deltas use identical media and score reuse; only the 12 separate inference repeats
measure scorer repeatability.

The reward calls took **98.25 s** for the comparison and **52.65 s** for attacks,
with peak CUDA allocation **6.01 GiB**. These exclude all-frame decoding/diagnostics
and startup. Observed successive-receipt intervals average approximately **2.84 s**
in the resumed comparison and **2.90 s** in the attack run; these exclude the first
sample and model loading, so they are not complete validator capacity measurements.

The first attempt failed before scoring because upstream TRL required the undeclared
`rich` dependency. `rich==13.7.1` is now pinned in the requirements and runtime spec;
the corrected plan was frozen before any GPU scores. The retry completed. Both
attempts were archived with SHA-256 readback verification and terminated; provider
absence was checked. The successful archive contains 224 members, including all
attack media and receipts; 158 small JSON files were also verified locally against
the archive manifest. Heavy artifacts remain in private remote storage.

Total create-request-to-confirmed-absence time across both attempts was **20.47 min**.
The actual GPU quote was **0.59 USD/hour**. A conservative estimate including the
failed setup, disk allowance and approximately 0.02 USD for S3/transfer is **0.24 USD
against the authorized 5 USD cap**. This is an estimate, not an itemized settled bill.
Reward-call time alone corresponds to about 0.025 USD and excludes most setup costs.

Decision: runtime feasibility and this environment's repeatability are established.
**VideoReward is not qualified as the sole reward judge.** Preserve the temporal
counterexamples, build the independent validation and blinded review described above,
and test a frozen joint quality rule before adopting it. Do not tune margins until
these known controls pass and call that validation. No mainnet/reward path changed.

## Repository changes kept small

Study logic stays under `src/forge/evaluation/`, immutable specs under `configs/`, and
two behavior-level integration tests exercise CLI/files/media boundaries. Existing
fusion/protocol/runtime paths retain their behavior. Evidence stays in ignored local
operator state and remote artifact storage. No new package hierarchy, service, UI or
repository split is needed. Consolidate old docs/scripts only once actual consumers
are migrated; do not delete historical receipt readers as “junk”.
