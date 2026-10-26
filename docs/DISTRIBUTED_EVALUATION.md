# Distributed evaluation without a coordinator

## Implemented boundary

Validators derive equal assignments from the same finalized chain snapshot and
future block entropy. They sign partial measurements, publish their hashes through
their hotkeys, fetch peer evidence, and reconstruct a global result. There is no
trusted scheduling API. A script starting the two controlled test validators is
an integration harness, not a protocol authority.

The current real GPU route is **diagnostic only**. It cannot certify a winner or
publish reward weights. Independent quality calibration, calibrated cross-worker
audit tolerances and the full 128-prompt power check remain activation blockers.
A signature authenticates a claim; it does not prove that a GPU executed it.

## Frozen inputs and allocation

1. Close miner submissions at a finalized block. Bind immutable public weights
   and recipe; snapshot committee identities and stake at that block.
2. Use a later finalized block for entropy. Each validator independently derives
   the same synthetic prompts, seeds, candidate identities and assignments.
3. Assign an equal number of work items to every committee member, including a
   small deterministic overlap. Impossible equal allocations are rejected rather
   than silently padding the sample or favoring a faster machine.
4. Keep assignments identical across challengers sharing a panel, so the exact
   king renders can be reused. Candidate-specific plan hashes still bind every
   signed report to its own weights and recipe.

`distributed-round-plan` reconstructs the diagnostic plan directly from finalized
testnet data, including the actual candidate commitment. Its six unique prompts
and two repeated audit prompts give two validators four prompts each. The small
panel is deliberately different from competitive screening (16), confirmation
(128) and repeated timing (32); it establishes integration, not benchmark power.

The deterministic template generator separates screening concepts from
confirmation concepts, covers subject/camera/style categories and includes
legitimate static scenes. Its finite vocabulary is not claimed to resist trained
specialists; the generator and population still require independent validation.
The diagnostic committee uses real validator permits and positive stake. A
production minimum-stake admission policy is not inferred from this small test.

## Measurements, exchange and aggregation

The single-GPU worker verifies public file and tensor identities before loading
Wan. It loads a clean transformer for each recipe, warms up, synchronizes CUDA,
and times text encoding, denoising and VAE decode. Download, load, warmup and export
remain operational costs. Per-job receipts bind prompt, seed, model, scheduler,
geometry, video hash, runtime and device. The current assigned worker requires a
new output directory, so it does not resume interrupted renders; it can reuse a
settled Luna order only when its durable journal matches the frozen input-image
hash. Interrupted or failed receipts cannot be blindly rerun under a new seed or
resent to Luna.

Luna compares both blind orientations; official pinned VBench supplies native
subject/background/image/smoothness metrics. Activity remains conditional on the
requested action. `diagnostic_fusion` retains raw evidence hashes and explicitly
cannot emit certification-grade quality receipts.

`distributed-report` signs assigned observations using the wallet's native
sr25519 hotkey. A frozen `DiagnosticBatch` includes every challenger plus commit
and reveal deadlines. `distributed-batch-index` signs the complete mapping of
plan hashes to partial-report hashes. `distributed-batch-anchor` commits that one
index hash and verifies finalized readback without resubmission. Peers reveal the
index, reports and raw evidence together after the commit deadline, using their
own content-addressed HTTP(S) endpoints. `distributed-batch-resolve` reads the
common finalized commitment cut and checks committee, entropy, hash, signature
and assignment before aggregating. Hash-addressed raw sidecars let each peer recompute the
reported deltas and detect provider-response reuse without paying for a second
judge call. Numerical replay does not prove honest execution. A peer URL is never
a trust root.

Each unique prompt contributes once globally. Overlapping measurements are
stake-weighted **within that prompt**, with contribution hashes retained. Missing
work remains incomplete; missing stake is not silently treated as a favorable
score. Equivocation, altered signatures and out-of-assignment data fail closed.
Uncalibrated audit tolerances are reported as uncalibrated, not as evidence that
two validators disagreed. ABBA measurements are combined as dimensionless local
speed ratios; absolute seconds from different GPUs are never averaged together
as a competitive speed measurement.

The diagnostic campaign serves no results before the frozen commit cutoff.
Missing peer data remains pending during the reveal window; after its deadline
the round expires without promotion. Late inspection cannot reopen an expired
round. A signed partial report is required for each challenger, even when work
is incomplete. This bounded CLI exchange is not an autonomous production round
daemon; sustained round rollover and production availability still need live
qualification.

## Rounds, state and weights

### Qualified lifecycle boundary

`forge.validator.competitive_lifecycle.CompetitiveLifecycle` is the durable
bridge used once a competitive (not diagnostic) bundle exists.  Opening a
round requires a >2/3 committee-signed admission-history checkpoint whose
root, parent certified state and frozen committee exactly match the plan.  It
persists one pending round only; after a certificate or an explicit expired
receipt, the next plan may extend the resulting head.  This preserves the
hotkey lineage history across validator restart and prevents local SQLite from
being mistaken for admission authority.

Closing a promotion requires `DistributedMVPCertificateVerifier`, which
rebuilds the signed distributed aggregate, validates the campaign-bound
calibration and the frozen 128/32 MVP gate, then verifies public custody.  It
does not trust a caller-supplied verdict or precomputed aggregate. It returns a
pure testnet `WeightIntent` from the certified state and current UID snapshot:
either 100% to the current publicly verified king or the separately qualified
burn target.  It does **not** sign or call `set_weights`; publishing remains a
separate infrastructure action.  Diagnostic reports, an ordinary certificate
verifier, missing calibration, missing raw independent panel validation,
missing raw cross-worker tolerance evidence, or a missing custody receipt
cannot cross this boundary.

The bridge accepts neither a hash nor a self-declared `PASS` as qualification
for those new blockers. It replays all three independent 128-prompt panels
(same model, mild blur and wrong video) from fused rows, requiring quality
clearance for the same-model panel and quality rejection for both degradations.
The same-model panel need not improve speed or promote. It derives
worker tolerances from eight development families, then checks eight disjoint
fresh families. The evidence must bind policy,
evaluator provenance, runtime, hardware, data families and the plan's exact
tolerances. It qualifies only that controlled environment.

`forge competitive-open` also requires `--distributed-plan` and `--capacity-inputs`:
the latter contains the raw `CapacityInputs` fields. Opening recomputes capacity,
checks its binding to the frozen round, and rejects commit/reveal windows too
short for the assigned work and delivery reserves. `forge competitive-expire`
closes a timed-out round. They make restart
and timeout handling operational without accepting a local quality JSON as a
certificate. `competitive-distributed-replay` constructs the immutable
execution bundle for a full confirmation only. Every certifying validator
replays that bundle again before making its local commit/reveal report. It
requires the round plan, the exact MVP plan, report signatures, campaign-bound
calibration, raw fused quality/timing receipts and public custody. It also
requires raw panel-validation and cross-worker tolerance evidence; declared
statuses alone do not satisfy either gate. The command
does not create a certificate or call `set_weights`.

The execution bundle is version 2. Its initial worker qualification requires
at least two validators with identical GPU type, precision, dependency lock and
runtime identity. The worker emits that identity in every signed quality and
timing row; it binds the image digest, dependency lock, VBench revision/weights,
DINO revision and evaluator provenance. Replay checks it against qualification.
A declared digest still needs verification against the actual executing container;
the hash alone is not hardware attestation. Panel source/concept families and
prompts must be fresh relative to calibration and the candidate panel, with
the same 128 source families in all three derived validation panels.

The existing six-prompt diagnostic route remains excluded: it cannot satisfy
the confirmation, campaign calibration or custody requirements and therefore
cannot cross this bridge. A successful synthetic integration fixture proves the
protocol boundary, never quality preservation by a Wan candidate.

Admission remains FIFO and consumes one opportunity per hotkey lineage. Several
screened challengers may enter full evaluation in one round, subject to budget.
The SQLite migration preserves previous round claims; a resumed challenger cannot
move to a different round or panel. It does not restore a consumed hotkey.

New consensus records explicitly support `stake_majority_v2`: a strict majority
of the **full frozen committee stake**. Exactly 50%, missing messages and double
signatures cannot constitute a majority. Historical plans omit that new field
and preserve their original greater-than-two-thirds rule and hashes. The global
certified winner overrides a validator's conflicting local verdict. Eligible
winner weights remain one-hot (100% of the miner allocation).

The diagnostic aggregate cannot enter this promotion path. The existing certified
weight publisher also requires current hotkey-to-UID eligibility and a qualified
no-allocation/burn route. Skipping `set_weights` does not clear old chain weights.
Do not describe a successful signed diagnostic exchange as a certified reward
cycle, or controlled validators as independent operators.

## Commands and checks

Use the repository's `.venv/bin/python -m forge.cli`:

- `competitive-open`: verify the signed admission checkpoint and persist one frozen round.
- `competitive-expire`: close a finalized timeout without inventing a KEEP/PROMOTE result.
- `distributed-round-plan`: reconstruct a frozen diagnostic round from chain.
- `distributed-plan`: inspect equal assignments without CUDA.
- `distributed-worker-plan`: validate exact jobs without inference.
- `distributed-worker`: run that frozen diagnostic bundle on one GPU.
- `distributed-report`: sign a partial result with a native wallet hotkey.
- `distributed-batch-plan`: freeze all challengers and common delivery deadlines.
- `distributed-batch-index`: sign the complete report-hash index.
- `distributed-batch-anchor`: commit the index on testnet (`--execute` required).
- `distributed-batch-resolve`: fetch peers, replay raw evidence and aggregate.

The CLI integration test uses two real temporary wallet signatures, two separate
HTTP servers and persisted reports. Numerical fixtures in it test protocol
behavior only. Real Wan experiments and their provider/chain receipts are stored
separately from tests and large artifacts remain on remote storage.

## Measured diagnostic campaign, 2026-10-04

Two controlled validators and three registered miner hotkeys completed a real
testnet campaign: six unique prompts, four assigned to each validator, two shared
audit prompts, 32 generated videos, five pinned VBench metrics and 48 blind-order
Luna calls. All candidates used the same public base weights with frozen recipes;
this was not a trained LoRA improvement experiment.

| Control | Observed result | Interpretation |
| --- | --- | --- |
| Identical Wan-50/UniPC | Eight byte-identical video pairs; every Luna and VBench delta zero; speed ratios 0.9974/0.9985 | Equality control passed; no speed improvement |
| One-step Euler, CFG-1 | Strong prompt/image degradation detected on all six unique prompts; mean MUSIQ delta -30.83 points | Deliberate degradation control passed |
| Forty-step Euler, CFG-5 | Speed ratios 1.2346/1.2373; mean MUSIQ delta -10.03 points, despite more favorable Luna observations in several scenes | Faster recipe; preserved quality not established |

Speed ratios are within-GPU descriptive measurements, not ABBA confidence bounds.
MUSIQ points are not perceptual percentages. Six prompt families cannot qualify
the 128-prompt gate or calibrate disagreements between automatic metrics.

Both validators finalized one signed all-candidate index before the fixed cutoff.
The real chain resolver rejected premature resolution, remained pending when
committed peer files were absent, and reconstructed identical global aggregates
on both GPU hosts after reveal. Missing peers became `EXPIRED` after the deadline,
with verification flags false. Every result remained diagnostic and ineligible
for rewards.

Both validators' native testnet weights were replaced with 100% to the owner burn
UID at observed block 8,149,032. A separate finalized readback at block 8,149,387
confirmed the same rows and identities, epoch block 8,149,386 and a 100% miner-burn
fraction, with the runtime code hash unchanged. The earlier burn counter predated
weight application and was not accepted as proof of the replacement's subsequent
effect. This qualifies the observed testnet/runtime route only; it does not enable
automatic reward publication or establish a mainnet burn destination.

A clean-materialization regression probe completed two model loads under a
24 GiB CUDA allocator cap, with 15.6 GiB peak generation allocation. It does not
qualify every recipe or metric configuration for a 24 GiB deployment.

Using these measured generation times and the observed rental price, an
illustrative two-validator allocation of the full confirmation panel plus ABBA
projects 10.65–11.84 USD GPU cost per validator, before screening and overhead.
This projection is not a measured full run or a future price quote. The 10 USD
default is conservative rather than a hard per-validator ceiling: a frozen
campaign may raise it only within its explicit global allowance. Calibration
and autonomous production round closure remain
activation blockers; this campaign produced no new certified winner.

## Time, cost and dashboard delivery

Use the [capacity calculation](EVALUATION_CAPACITY.md) before dispatching a frozen panel.
The capacity receipt derives the commit/reveal windows from measured assignments
and timings. The default 10 USD value is conservative; a campaign can raise it
only through its explicit global allowance. A GPU name alone does not qualify throughput;
unknown costs or insufficient capacity leave the evaluation incomplete.

Validators can send their signed reports directly to the [dashboard API](DASHBOARD_API.md).
The dashboard records receipt independently of chain anchoring and evidence replay.
An API acknowledgement never changes assignments, consensus, the king or weights.
