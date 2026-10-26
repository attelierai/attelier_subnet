# Inference prelaunch and historical pilot

## Current prelaunch contract

The active work is a private rehearsal on Finney SN126, mechanism 0, for one
fixed H3 profile: text-to-video, 5 seconds, 480p, 16:9, 24 fps, audio and 20
steps. It is **not a launch**. The reward-accounting simulator creates no new
identity, accesses no wallet, and never signs, commits, reveals or sets weights.
Mainnet information is read only.

All providers are permissioned. A provider may create and manage its own
application manually in the console or with an `attp_` provider-management key,
but neither path approves or activates that provider. Operator review and a
tested worker binding remain required. As part of that review, the provider must
answer a bounded ownership challenge signed by its existing public hotkey. This
proves control of that hotkey to the operator; it neither discloses a private
wallet key nor approves the provider or writes chain state. The `h3w_` worker
credential is separate: the Gateway uses it to authenticate to an admitted
worker. Neither credential is a private wallet key or authority to publish chain
state.

Each closed rehearsal ledger is exactly one UTC day. The offline-only
[`INFERENCE_PRELAUNCH.md`](INFERENCE_PRELAUNCH.md) simulator validates the
permissioned binding and finalized identity snapshot, then emits an unsigned
review artifact. It explicitly records `simulationOnly: true`,
`publishable: false`, and no chain calls. Its proposed 600 videos/day and 125 bps
service budget are scenario inputs, not approved emissions, prices, revenue, or
a payout promise. Empty eligible work produces simulated burn. Service merit,
unsigned proposed weights, and expected chain payout are separate values. An
owner-associated test identity can receive counterfactual service merit while its
expected mining payout is zero.

The pinned worker manifest is in the inference API's H3 worker package. Its
public model files, revisions and hashes are the required release inputs; health
declarations do not replace installation or media checks. The measured 5090
candidate is useful only for its tested shape and remains subject to the
prelaunch gateway test.

The rest of this document records the **closed testnet mechanism-1 pilot** and
trace research. It is preserved as historical evidence and must not be used as a
SN126 launch recipe, current reward policy, or permission to publish.

## Boundaries

- `forge.inference`: miner registration/identity, measured service challenges,
  evidence, scoring, mechanism-specific chain operations and dashboard export.
- The H3 worker owns its Comfy process, admitted jobs and GPU result files. A
  worker restart retains its ledger identity.
- The inference API owns customer quotes, budgets, jobs and routing. A separate
  H3 subnet provider dispatches across miners and persists its chosen miner before
  sending a job. Unknown acceptance is reconciled with that miner; it never causes
  a second miner to execute the same request. Other providers remain registered.
- The dashboard renders inference separately from optimization. No customer
  prompts, private endpoints, authentication material or raw logs enter its public
  snapshot. Hardware names are operator declarations, not attestation.

## Acceptance

Real testnet acceptance requires registered distinct miner hotkeys, independently
verified media from both GPUs through the API, recovery and cancellation checks,
validator measurements, and finalized mechanism-1 weights read back from chain.
An HTTP health check, synthetic fixture, submitted transaction or allocated GPU
does not establish these results. Quality heuristics are reported with their
limits; they cannot be promoted to production reward qualification implicitly.

The validator counts failed, cancelled and timed-out challenges in availability.
Only verified successful results contribute useful service. Identical challenges
use equal model settings; elapsed time includes queue and transport. Repeated
result hashes, wrong dimensions/duration, undecodable media and missing audio
fail validation. Semantic checks and negative controls must be retained with the
assessment. An unavailable judge is unresolved, never an automatic pass.

## Pilot operation and score

Use one operator-owned configuration shared by the API and validator. It contains
`schema: "attelier/subnet-h3/v1"`, `network: "test"`, `netuid`, `mechanism: 1`,
`source: "live"`, an absolute `registrationFile`, and a `miners` array. Each miner
needs its distinct registered `uid`/`hotkey`, stable `id`, authenticated private
`url`, persisted `workerId`, `tokenEnv`, pinned `modelManifestHash`, and absolute
`attestationFile`. `hardware` is an optional operator-declared display label.
The token field is an environment-variable name, never a credential value.

After native wallet registration and a worker deployment, `attest --wallet NAME
--hotkey MINER --miner-id ID --expires UNIX_SECONDS --out ATTESTATION` signs the
binding with that miner's local hotkey. It cannot extend beyond 24 hours. The
private key stays in the native wallet; copy only the signed public attestation.
`export-registry --config CONFIG --out REGISTRY` verifies those attestations
against one finalized block. Refresh this projection within 120 seconds. A
registered UID alone does not make a worker eligible for admission or rewards.
Neither the public attestation nor a worker's service receipt proves GPU
ownership, exclusive use, the named checkpoint, or absence of outsourcing.

Run the validator with the project environment and the isolated source tree, for
example `PYTHONPATH=/absolute/isolated/src .venv/bin/python -m forge.inference
step --netuid 578 --config CONFIG --directory STATE --budget-micros LIMIT`.
Use `enqueue` with `--input`, `--batch` and `--ceiling-micros`; then independently
review each decoded result with `review --job-id ID --input JUDGMENT` before
asking `scores --batch NAME`. These commands only manage local durable evidence
unless the separate `publish --execute` command is explicitly used.

An installed package also provides the `forge-inference` entry point with the same
arguments. `dashboard --input BUDGET --publication RECEIPT --observations HEALTH
--out SNAPSHOT` writes the separate dashboard projection. The budget contains
measured or explicitly estimated spend; health maps miner IDs to measured `state`
and Unix `observedAt`. Missing observations remain unknown/offline. The exporter
derives emission shares from observed chain configuration, rather than assuming
that requested shares are active. Publish the snapshot atomically to the
dashboard's configured data directory as `inference.json`.

`publish --execute` requires a local validator wallet and live, complete,
registered evidence. It persists the exact SDK call hash and encrypted commit
witness before broadcasting. `reconcile --out RECEIPT` is read-only: it can
recover a lost inclusion acknowledgement by the prepared call hash, then binds
the automatic reveal to the unique queued ciphertext, attributed reveal/weights
events and exact finalized weights row. Commit-time `LastUpdate` is not a reveal
proof. Retain the receipt and continue reconciliation; never blindly resubmit.
Finalized commit evidence is cached before RPC pruning can remove its events.

Every attempted job reserves its quoted ceiling against the explicit pilot
budget, including cancelled, failed, timed-out and unknown-acceptance jobs. The
validator records POST intent before sending it and only rechecks the same
worker/job ID after a restart. Immediately before every execution POST, including
a same-ID recovery POST, it reads a fresh finalized registration projection and
validates that worker's live hotkey attestation. GET and cancellation recovery
remain available if that admission later becomes stale. A missing or stale
finalized projection, a local worker credential failure, or an operator worker
configuration drift is `inconclusive`: it blocks publication and is not counted
as a miner service failure. Before the deadline such an unsent job remains
recoverable; after it, the durable inconclusive row remains for audit. An invalid
current worker attestation, malformed response, replay or invalid media remains
a miner-evidence failure.

For legacy batches, at least two equal challenge parameter sets are required.
Its score is `sum(min(1, 60 / elapsed_seconds) for verified attempts) / all
attempts`. Failed, rejected, cancelled and timed-out attempts therefore remain
in the denominator; `inconclusive` and semantic-review-pending attempts make the
batch incomplete.

`referenceVerification: true` is the availability-first H3 audit mode. It is
fail-closed: each worker pins `modelManifestHash`, `workflowHash` and an Ed25519
service public key. Its completion receipt covers `jobId`, `workerId`, POST-only
random nonce, parameter hash, both roots and delivered result SHA-256. The service
signer is protected worker material, never a wallet. The validator decodes video
to yuv420p and audio to stereo 48 kHz f32le, hashes both streams, then invokes the
configured full reference replay. Only exact decoded stream matches verify. A
missing or malformed receipt is `invalid`. A renderer/custody fault is
`inconclusive`. An exact audio or video mismatch during either calibration is
also `inconclusive`: calibration does not qualify and no weights can be derived.
A confirmed holdout substitution zeroes that miner's entire window; it is not
treated as a partial latency loss.

The command receives canonical JSON on stdin and emits JSON with an existing MP4
`path`; it runs outside the miner latency clock and has a default 600-second cap.
`calibrationBatch`, `calibrationPerMiner` (2) and `holdoutsPerMiner` (6) gate
publication. The score is `A * (0.8 + 0.2 * mean(Tref / (Tref + Tminer)))`, where
`A` is verified-reference successes / attempted holdouts and `Tminer` begins at
the first execution POST and ends at completed download plus receipt collection.
Reference rendering and validator decode time are excluded. This establishes
sampled agreement with a controlled reference, not a TOPLOC-style checkpoint
proof, remote GPU ownership proof, or cryptographic proof of every execution.
The model/workflow hashes and signature bind an untrusted worker declaration to
its receipt; they do not independently vouch for what ran. One receipt directory allows only one pending M1
publication per validator hotkey; reconcile it through reveal before another.

Complete miners receive score-proportional weights only when the total positive
score is nonzero.

## Dashboard contract: `forge/inference-dashboard/v1`

The page fetches same-origin `/data/inference.json`, refreshes every 15 seconds and
marks snapshots older than 120 seconds stale. Missing and null values stay
unknown, never become zero. Its test data must be visibly labelled synthetic.

```json
{
  "schema": "forge/inference-dashboard/v1",
  "generatedAt": "2026-10-23T00:00:00Z",
  "source": "live",
  "network": "test",
  "netuid": 578,
  "mechanism": 1,
  "status": "preparing",
  "emissionSplit": {"optimization": 0.5, "inference": 0.5},
  "chain": {"block": null, "hash": null, "runtime": null, "mechanismCount": null, "weightsStatus": "not_published", "weightsBlock": null},
  "budget": {"limitUsd": 50, "spentUsd": 0, "estimated": true, "hourlyUsd": 0, "deadline": "2026-10-23T06:00:00Z"},
  "miners": [],
  "jobs": [],
  "checks": [],
  "issues": []
}
```

Each `miners` row contains `id`, `uid` (nullable), `hotkey`, `hardware`, `model`,
`state` (`offline|starting|ready|busy|draining|stopped|failed`), `workerId`,
`attempts`, `succeeded`, `failed`, `successRate` (0..1 or null),
`medianLatencySeconds` (nullable), `score` (nullable), `availabilityRate`,
`speedScore`, `verificationStatus`, `excluded`, `weight` (0..1 or null),
`publishedWeight` (0..1 or null), `lastSeen` (ISO timestamp or null) and
`evidenceStatus` (plain text).

Each `jobs` row contains `id`, `minerId`, `kind` (`challenge|api`), `status`,
`startedAt`, `completedAt` (nullable), `latencySeconds` (nullable),
`generationSeconds` (nullable), `valid` (boolean or null), `resultSha256`
(nullable), `receiptStatus`, `referenceStatus`, `canonicalMatch` (boolean or
null) and `reason` (nullable).
Each `checks` row contains `id`, `label`, `status` (`pending|pass|fail|blocked`)
and `detail`. `issues` is a list of strings. No media URL is implied by a hash.

The dashboard separates measured score, planned weight, submission and finalized
weight observation. A 50/50 configured split is not evidence of finalized chain
configuration. Loading, missing data, stale, empty, active, partial failure and
closed campaign states must work on desktop and mobile.

## Experimental trace verification — not qualified for rewards

The private H3 experiment separates three questions: whether each denoiser
prediction matches the approved model on the submitted state, whether the states
obey the approved sampler, and whether the delivered media agrees with the final
latent decoded by a validator-owned VAE. The existing service verifier and weight
publication gate are unchanged. The research scripts do not publish weights.

For the pinned Comfy `res_multistep` / `simple`, 20-step, CFG=1 profile, the
validator fixes the request conditioning, initial noise, sigma schedule and
video/audio layout. It checks the first-state and terminal-latent bindings and
all sampler transitions. Only after the trace and delivered-media hashes have
been committed does it select model replay challenges. Replaying a prediction
on the miner's supplied state avoids comparing entire trajectories after
numerical error has accumulated. Algebraically consistent transitions alone
are insufficient: the experiment includes a forged trajectory that passes
recurrence checks but fails canonical model replay.

A three-step sample is explicitly an incomplete audit. If one of 20 predictions
is substituted, three uniformly selected distinct challenges detect it with
probability 3/20. The single-skip control demonstrates this bypass. The initial
availability qualification candidate therefore requires all 20 predictions,
transition and endpoint checks, and delivered-media verification. Full replay
has substantial validator compute cost; it is not a cheap proof of computation.
The current trusted conditioning fixture comes from a controlled honest capture;
independent request preprocessing must be implemented before remote admission.

Checkpoint identity must include an approved numerical execution profile.
The same NVFP4 checkpoint runs through native operations on the tested Blackwell
GPU and emulated operations on A100. In paired experiments, some unapproved FP8
outputs were closer to the A100 reference than the honest Blackwell outputs.
A single relaxed scalar tolerance cannot separate all of these individual
cases. Unknown or unqualified profiles are inconclusive, rather than evidence
of checkpoint substitution. Hardware names and miner-supplied runtime hashes
do not attest the execution path. Portability requires qualification across
independent devices with a matching reference implementation.

The media audit compares all decoded video frames against the trusted VAE's
RGB24 encoder input and compares AAC-decoded audio against the same duration
crop used by the pinned encoder. Codec distortion needs its own frozen
calibration; exact MP4 or decoded-stream hashes can disagree after identical
latent and VAE computation. Pixel/audio error bounds are approximate agreement,
not semantic quality or cryptographic output identity. The audit process,
model files, graph and evidence directory are validator-owned in this closed
experiment. Miner-authored numeric receipts would not establish these checks.

The report binds the media schema and audio crop/sample contract to the six
hash-frozen honest calibration receipts. Mixing evaluator versions makes that
media check inconclusive. Measured replay coverage is reported separately from
prediction agreement: checking all 20 predictions and finding one wrong value
is complete coverage with a failed model check. Neither coverage nor the
report's `MEASURED` status is an admission decision.

Before deployment, the gateway must retain immutable delivered bytes and bind
that hash, the fresh request/seed, miner identity, execution profile and trace
commitment to one job. The latency clock must end after both media delivery and
trace commitment, so a miner cannot earn a faster time by constructing evidence
later. Reference audit time remains outside miner latency. A fresh nonce that
does not affect the actual model input would not prevent cached generations.
Outsourced correct computation can satisfy functional verification; this design
does not establish remote GPU ownership or exclusive use.

Availability rewards require unpredictable, successfully verified generation
challenges, including failures and timeouts in the denominator. Apply the
existing bounded efficiency component only after verification succeeds. A
confirmed substitution may exclude the window; an unsupported backend or
validator failure must remain inconclusive. The frozen research holdouts,
negative controls and remaining media/profile defects must be resolved before
this experimental path can authorize rewards.

The related [Kuno verified-mode design](https://github.com/concil859856/kunoworld-subnet/blob/b7c04e353a8807497f34f2773f6124693816b68a/VERIFIED_MODE.md)
commits denoising states, opens sampled steps and sometimes repeats the entire
trajectory. At that pinned revision its documentation explicitly reports that
the H3 GPU hooks/executors have not run on a GPU. It supplies useful protocol
ideas, rather than qualification evidence for Attelier's checkpoint or runtime.
Its [Turbo competition](https://github.com/concil859856/kunoworld-subnet/blob/b7c04e353a8807497f34f2773f6124693816b68a/TURBO.md)
commits hidden evaluation sets before a window and reveals them afterward. That
can make challenge selection auditable, but cannot independently prove model
execution. Similarly, hardware attestation is a separate trust layer: the
[documented CVM status](https://github.com/concil859856/kunoworld-subnet/blob/b7c04e353a8807497f34f2773f6124693816b68a/image/CVM.md)
at that revision reports no boot on TDX hardware. None of those assurances are
implicitly inherited by these GPU experiments.

The public [Engy miner at `ed84d0b`](https://github.com/hanlinai/engy/blob/ed84d0b43a9de19f8611a2321c45acc09e570b4c/miner/engy_miner.py#L570-L578)
builds TOPLOC commitments from transformer hidden states. Its
[one-pager](https://github.com/hanlinai/engy/blob/ed84d0b43a9de19f8611a2321c45acc09e570b4c/docs/SN53_ONE_PAGER.md#L50-L80)
reports an FP8/INT4 experiment and identifies TOPLOC as its current gate;
sampled GEMM is described separately. The
[repository status](https://github.com/hanlinai/engy/blob/ed84d0b43a9de19f8611a2321c45acc09e570b4c/README.md#L24-L26)
says that the audit verifier and full incentive mechanism are not yet public.
Its useful design lessons are internal commitments, calibration across hardware
and a separate audit process. Those LLM results do not qualify H3 denoising or
prove that a scalar error bound separates H3 numerical backends. Engy's public
`model_root` hashes configuration and shard names/sizes; it is a declared
identifier, not a hash of every weight byte or an execution attestation.

## Closed GPU study — 2026-10-23

**The combined verifier did not qualify for rewards.** Its frozen video-error
limit rejected two honest runs from one held-out prompt lineage. The full model
replay component matched every honest prediction in the held-out set; this does
not remove the downstream media false positive.

The study contains 88 cases: 28 calibration/development cases from three
lineages and 60 held-out cases from six additional lineages. Thresholds and the
case index were frozen before held-out generation. Each lineage contributes
correlated variants; neither the 88 cases nor individual model calls are 88
independent statistical trials.

| Held-out condition | Runs | Frozen combined verdict |
| --- | ---: | --- |
| Honest baseline, two repeats per lineage | 12 | 10 pass; 2 false rejections |
| Unapproved FP8 checkpoint variant | 6 | 6 rejected |
| Unapproved LoRA, still 20 steps | 6 | 6 rejected |
| Native 10-step or 3-step schedule | 12 | 12 rejected |
| 10 steps padded to claim 20 | 6 | 6 rejected |
| Algebraically forged sampler recurrence | 6 | 6 rejected |
| Honest trace paired with a different delivered video | 6 | 6 rejected |
| One model call skipped, with 20 declared sampler steps | 6 | 6 rejected |

All 240 canonical prediction comparisons for the 12 held-out honest runs were
bit-exact. Full replay detected the skipped call in each single-skip case. None
of the six preselected three-step samples included that bad call: five attacks
passed the partial rule; the sixth was rejected by the same video criterion
that incorrectly rejected its honest controls. That sixth rejection is not
evidence that sampling detected the missing computation.

The two honest media failures had video maximum frame RMSE
`0.018528164854504224`, above the frozen `0.016550245569790196` limit. Their
complete raw VAE video tensor hash matched the trusted audit exactly. The
disagreement therefore arises downstream of VAE computation; its precise
encoding cause has not been established. Changing the threshold after seeing
these runs would not validate the criterion.

A separate post-hoc diagnostic compared already-encoded honest repeats across
all nine lineages. Their largest video error was `0.009036479555079507`, while
the smallest error against the corresponding 3-step video was
`0.11387683633522654`; honest decoded audio repeats matched exactly. This is a
candidate direction for an encoded-reference check, not a replacement
calibration or an unseen validation set. Another design option is to construct
the delivered media from the validated final latent through a trusted renderer;
that removes miner-encoded media comparison but adds trusted serving compute.
Neither alternative has been integrated into the gateway by this study.

The separate media-audit v3 supplement preserved every measured video/audio
value and trusted VAE hash for two controls selected before the supplement. It
rejected an appended 2-second audio payload through the stream-timing check.
The prior v2 audit had also rejected that concrete payload; no successful audio
tail bypass was observed. The bounded decoder/timing correction does not fix
the frozen video-threshold false positive above.

The second physical RTX PRO 6000 Blackwell device reproduced all three sampled
predictions exactly for six honest cases and rejected the twelve corresponding
FP8/LoRA cases. A further full 20-call honest control matched exactly and a
single-skip control differed only at the skipped call. The A100 experiment used
the same checkpoint through an emulated numerical backend; its honest errors
overlapped those of the unapproved variant, so it remains unqualified for the
native Blackwell comparison. These are prediction checks, not proof of VAE or
video portability across devices. RTX 5090 and RTX 6000 Ada were not qualified
by this study.

The original report SHA256 is
`d02f656a29145f2d8b112b63d331dbfe841e677c0344f207199ad0d7be8c7bce`.
A separate report with evaluator-contract validation and corrected coverage
semantics has SHA256
`569518695a0f611dd26ed890238967eb5c288695ab52a203a060e87bb6ef7993`;
all 88 full/partial verdicts, source hashes and thresholds are unchanged.
This was a controlled GPU harness, not a remote-miner reward or chain test.
Generation and audit work shared the GPU, so measured campaign times are not
a production time-to-video benchmark. No reward publication was enabled.
