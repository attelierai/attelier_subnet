# Model competition: agreed target and activation boundary

Status: **pragmatic launch V1 implemented in part; real acceptance in progress;
window runtime not activated**. [LAUNCH_V1.md](LAUNCH_V1.md) owns the initial
provisional constants and current acceptance boundary. The initial launch no
longer requires human labels or full scientific calibration.
This document owns the target for the first model-only launch. Creator rewards are
a later phase. The current executable 16/128/32 path remains documented in
[WAN_MVP.md](WAN_MVP.md); it does not already implement this new window contract.

## What wins

One text-to-video family, initially Wan2.1-T2V-1.3B. Miners publish immutable,
anonymously downloadable weights and a bounded approved inference recipe under a
reusable license. Training on the current king is encouraged; training code is
optional. A recipe-only improvement can win. Identical effective weights and
recipe, repackaging and equivalent submissions cannot dethrone the king.

A challenger needs at least **1.10x speedup** over the current king, with acceptable
quality against **both the king and the original fixed quality anchor**. The anchor
prevents small tolerated losses accumulating across kings. Quality means acceptable
outputs for the requested task, not matching the reference pixels or seed result.
Every protected dimension must pass; improvements in one cannot buy a failure in
another. No commercial API participates in the intended reward decision.

Use paired timings interleaved on the same GPU/runtime. The lower confidence bound
of the speed ratio, not its single-run mean, must clear 1.10. Warmup, loading,
decoding and generation timing boundaries must be frozen explicitly. Different
hardware is not compared through raw seconds or a guessed scaling factor.

## One closed window

1. At a finalized close block, freeze the queue, FIFO chain positions, effective
   artifact/recipe identities, evaluator version and eligible validator committee.
   Eligibility requires validator permit and the configured minimum stake.
2. Derive prompt instances and seeds from a public generator and future finalized
   entropy after closure. New submissions wait for the next window. Public code
   does not imply future samples are precomputable; unpredictable samples do not
   make a public metric resistant to optimization.
3. Assign unique prompts to validator shards. Validators are initially trusted to
   measure honestly; stake is an admission/consensus weight, not proof of GPU work.
   No mandatory overlap, TEE or systematic re-execution in this version.
4. Reuse the king and anchor **quality videos** across challengers only for exactly
   the same window, prompt, seed, artifact, recipe, geometry and runtime. Alias
   king/anchor when identical. Timing references remain contemporaneous/interleaved.
5. Commit signed measurement reports before revealing/aggregating them. Count each
   assigned prompt once, irrespective of stake or duplicate report delivery. Missing
   evidence makes that candidate incomplete, never better. An infrastructure retry
   resumes the same frozen identities and sample set, not a fresh attempt.
6. The window starts with a frozen king. Process complete challengers in FIFO order;
   if a challenger replaces the provisional king, later candidates require paired
   evidence against that provisional king before replacing it. Missing duel work
   cannot be inferred from comparisons with the old king. Limit admission to the
   capacity for these dependencies; incomplete work carries forward with its frozen
   identity, without borrowing scores from another window.
7. Certify the deterministic state using a strict majority of the **full frozen
   eligible stake**, including nonresponders in the denominator. Publish the latest
   certified king's weights through the supported chain mechanism. Distinguish local
   decision, certificate, weight submission and finalized on-chain activation.

Retain the existing one-submission-per-hotkey-lineage admission, persistent history,
effective-weight deduplication, artifact verification, signatures, report replay and
outbox recovery. Re-registration or hotkey swaps do not reset the allowance. Different
registered hotkeys can submit; registration cost plus bounded queue capacity are the
initial economic controls, not a claim of Sybil elimination. Derivatives are legitimate.

Continue refreshing the existing king's weights during a long window when required
for validator activity. A new king is not active merely because a local certificate
exists. No-winner, missing-owner and burn behavior must use the explicit chain adapter
and tested policy, never an assumed special UID.

## Initial constants and later refinement

The initial panel, separate reward-head margins and statistical gate are frozen
in [LAUNCH_V1.md](LAUNCH_V1.md). The table below describes desirable refinement
evidence, not mandatory human-study or learned-adversary gates for V1. Capacity,
committee eligibility and chain activation still require operational evidence.

| Parameter | Evidence required before freezing it |
| --- | --- |
| Quality axes and margins | Blinded acceptable/unacceptable judgments, including motion, physics, text, appearance and diversity failure modes |
| Final panel size | Independent family-level precision/power, measured disagreement and attack sensitivity |
| Statistical gate | One-sided simultaneous non-inferiority bounds by family; multiplicity across protected dimensions, two references and tested candidates; no unplanned repeated peeking |
| Early rejection | Frozen futility rule that can reject, never promote; final panel remains complete |
| Window length N epochs | Measured generation/evaluator/duel time, shard count, startup/recovery and report/certificate lag |
| Queue capacity | Worst-case useful work fitting that N and retention/recovery capacity; explicit FIFO overflow rule |
| Minimum stake | Actual committee availability and trust/participation requirement |
| Immunity and weight activation | Queue wait + evaluation + certification + actual commit/reveal/finality delay fit the chain parameters |

The first study measures whether the open evaluator is worth qualifying; it does not
pick these constants by fitting known controls. Initial A40/4090/5090 capacity should
be used effectively while work is queued. A 1–2 hour evaluation is acceptable if its
quality evidence is sufficient; it is not a hard bound or a reason to shrink a panel.

## Current testnet acceptance

Use the provisional V1 rule under its distinct version and retain the known
metric limitations. Run a real public
miner through distributed testnet, including duplicate hotkeys/artifacts, changed
post-close recipes, late arrivals, wrong reference reuse, missing shards, same-sample
resume, cumulative degradation, king changes and actual finalized weights. Align
queue/immunity and chain activation using current testnet chain reads. Mainnet
research, configuration and activation are outside the current campaign. Local fixture tests
and simulated `testnet-round` do not satisfy this acceptance gate.

Historical overlapping audits, sealed private packs and implementation attestations
remain replayable in their original formats. They are not requirements of the new
trusted-validator contract. Remove obsolete runtime branches only when the replacement
has real acceptance evidence; keep the historical readers needed by stored receipts.

The current rule and measured acceptance work are specified in
[LAUNCH_V1.md](LAUNCH_V1.md) and [LAUNCH_V1_ACCEPTANCE.md](LAUNCH_V1_ACCEPTANCE.md).
[EVALUATOR_STUDY.md](EVALUATOR_STUDY.md) retains the earlier diagnostic study.
