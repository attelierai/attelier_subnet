# Wan launch V2 — testnet

V2 is the current bounded testnet contract. It does not activate mainnet or
qualify an evaluator merely because the code and integration checks pass.
[V1](LAUNCH_V1.md) and the older [MVP](WAN_MVP.md) remain available for historical
receipt replay. Model artifacts and license signals follow [ARTIFACTS.md](ARTIFACTS.md).

## Admission and frozen rounds

`configs/launch-wan-testnet.json` binds testnet genesis, netuid 578, the Wan
contract, 300-block admission windows, one active candidate, and a campaign
budget. It is not spending or signing authority by itself.

Each hotkey lineage may submit one candidate per admission window. A pending
candidate prevents a concurrent submission in another window. After completion,
a later window may admit a different effective model. The effective identity
includes weights, generation recipe and scheduler; changing a UID, filename or
container cannot bypass global duplicate detection.

Selection is persisted at the finalized close C, before E = C + 10. Its round
identity derives from the close, candidate, parent state and frozen
configuration; the operator cannot choose another round identifier to draw
different prompts. Infrastructure recovery retains the same artifact, recipe,
panel, seeds and request. A completed scientific result cannot be rerolled.
New rounds explicitly freeze `stake_majority_v2`.

## Automatic evaluator qualification

Before candidate comparisons, qualification version 2 binds four development
videos (one per family) and eight fresh confirmation videos (two per family).
The independent unit is a source video, not each of its 12 transformations.
Confirmation actions and scene lineages must be reviewed before scoring and
disjoint from development and every previously inspected confirmation source.
The manifest binds those lineage labels and exclusions; labels do not themselves
prove semantic independence. Before scoring, every decoded source must exceed
the preregistered temporal-delta floor `0.1 / 255`. An ineligible source interrupts
preparation without assigning a scientific evaluator result. Each source has
the following controls:

| Controls | Required response |
| --- | --- |
| Exact identity copy and pinned H.264 transcode | No decoded guard hit; transcodes demonstrate score equivalence on the fresh panel |
| Freeze, black frame, magenta frame, corrupted segment and flicker | Detect with applicable decoded-frame guards |
| Gaussian blur | Demonstrate material perceptual loss on the fresh panel |
| Video from a different subject family with the original prompt | Demonstrate text-alignment loss on the fresh panel |
| Identical decoded videos across three seeds | Reject copied diversity |
| Altered file hash and missing media | Reject before scoring |

Frame-corruption and flicker tolerances use only development positives. Freeze
and near-freeze rules remain fixed. Missing media, altered hashes, exact copied
diversity and severe frame controls still require their declared response.

Perceptual checks use paired candidate-minus-identity score differences over
the eight confirmation sources and the existing product margins: VQ `0.35`,
MQ `0.20`, TA `0.20`:

- Benign transcodes require both one-sided 95% Student-t bounds within the
  positive and negative margin on all three axes (a two-sided 90% interval).
  This checks both loss and artificial score inflation.
- Blur requires the upper one-sided 97.5% bound below the negative margin on
  VQ or TA. The 97.5% level accounts for these two opportunities to detect it.
  Family means are reported, and no family may improve on both VQ and TA.
- Incompatible prompts require the upper one-sided 95% TA bound below `-0.20`.

An uncalibrated high-frequency ratio is a diagnostic, not a hard blur veto in
version 2. Naturally soft images are not disqualified solely by that proxy;
competitive quality continues to require separate VQ/MQ/TA non-inferiority.
No per-video score invariance or detection is inferred from a pooled result.

`QUALIFIED` requires all checks. A deterministic failure or a failed point
estimate is `FAILED`; a point estimate that meets the rule with insufficient
confidence is `INCONCLUSIVE`. Invalid or missing evidence is `INCOMPLETE`.
The last two states confer no qualification authority. A new sample, rather
than retuning on inspected confirmation scores, is required after a design change.

The immutable package contains the source/control plan, raw score receipts with
pinned evaluator spec and runtime, and the derived result. Validators replay the
entire package. Version 2 retains the receipt envelope schema and binds an
explicit `qualification_version: 2` plus the complete versioned decision rule.
A `QUALIFIED` flag or hashes without replayable raw measurements
have no authority. A failed or inconclusive gate stops candidate comparisons.
Historical qualification v1 receipts retain their exact replay behavior and
results, including the stricter per-video transcode tolerance and blur proxy;
version 2 never relabels a historical failure as a pass.

### Composite flicker qualification v3

V3 is a separate, additive receipt for the repeated-photometric flicker guard.
It preserves the complete immutable V2 package and replays it before accepting
any V3 result; V2's three statistical decisions and all retained decoded-frame
guards are unchanged. A V2 result is usable only when it is `QUALIFIED`, or
when every V2 rejection is explicitly an old `:flicker` reason. No historical
receipt is relabelled.

Before CPU measurements, the V3 plan freezes eight independently labelled
natural sources, their lineages, source/control byte hashes, decoded hashes,
geometry, the 64-cell fresh matrix and the 36-cell V2 regression matrix. The
fresh controls expect no guard hit for identity, transcode, pan, zoom and ramp;
they expect a hit for inversion, luminance and chroma. Regression remeasures
identity, transcode and flicker against the exact V2 control bytes. A raw
receipt contains all 100 per-file guard envelopes and is invalid when a hash,
geometry, matrix cell, algorithm identity or envelope is missing or changed.

The result retains the existing `forge/evaluator-qualification/v1` envelope
for the narrow V2 launch-policy consumer, with `qualification_version: 3` and
the frozen `flicker_guard_v1` algorithm marker. On V3 competitive receipts the
new envelope-based guard replaces only the former global-brightness flicker
branch; black, magenta, segment, freeze, quality and speed checks remain in
force. V1 and V2 retain their historical replay paths.

Qualification tests these controls, not general perceptual quality, physical
plausibility or resistance to every metric exploit. Human calibration is not
claimed.

## Competitive panel and result

The public generator derives 32 confirmation prompts from E: 8 static and
24 motion prompts across human, animal, object and scenery families. Development
and confirmation combinations are disjoint. Generator `v2.1` assigns whole-shot
static/motion semantics through separate catalogs, before shuffling. Each family
has two static and six motion prompts. The splits use different lighting
conditions; these public compositions do not constitute an unseen holdout.
Steady-lighting tags enable the flicker guard.

| Measurement | Decision |
| --- | --- |
| Separate VQ, MQ and TA against current reference and original Wan50 | Each one-sided 95% lower confidence bound must meet its margin |
| Eight predetermined R,C,C,R timings | One-sided 95% lower speed ratio must be at least 1.10 |
| Four predetermined prompts, three seeds each | Reject exact copied decoded videos |
| Qualified frame guards | Reject applicable severe defects |

Quality margins remain provisional: VQ 0.35, MQ 0.20, TA 0.20 in evaluator score
units. They are not percentages of perceived quality. Each prompt is one
statistical unit; three diversity seeds do not triple the sample size. Freeze
checks require a motion-tagged prompt, and flicker checks require steady lighting.

`PASS` is locally promotion-eligible. `FAIL` and `INCONCLUSIVE` retain the current
reference. Missing or invalid evidence is `INCOMPLETE`, not a scientific failure.
Only the separate signed consensus and qualified publication path can change
reward weights.

## Baseline comparison

Screen at most three predeclared alternatives on the same six development
prompts. The initial candidates are Wan44, Wan40 and the public
[LongCat1500 manifest](../configs/longcat1500-testnet.json). Rank candidates that
pass provisional mean quality and severe-defect checks by measured speed, with
manifest identity breaking ties.

The pinned [LongCat model card](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-longcat-step1500/blob/791dc13f1e672d4643c736b273d8c91a85da6b40/README.md)
reports VideoAlign rewards during training. Scores from that evaluator therefore
are not independent evidence of general quality. The experimental 16-step,
guidance-4.5 recipe is fixed before screening; the card recommends 50 inference
steps, so the faster recipe's quality must be measured.

Persist that choice once, then run exactly one full fresh confirmation panel.
Only a complete `PASS` imports the selected baseline. Failure, inconclusive or
incomplete confirmation retains Wan50; no second candidate gets another
confirmation opportunity. An imported baseline is a reference without a reward
hotkey. It is never presented as a certified miner king.

Baseline confirmation replays the same raw panel, timing, score and diversity
receipts as competitive rounds. `replay_baseline_campaign` binds those receipts
to the deterministic selection-derived panel without creating a consensus round
or validator identity. A confirmation that has started consumes the single
attempt even when an interruption leaves no validated rows.

## Durability, publication and observability

The SQLite service reserves cost before issuing a worker request and keeps
evaluation, certification, submission and effective weights separate. A restart
reconciles the original request; it does not select another panel, issue another
vote or manufacture a crown. Capacity deadlines are measured from E.
The execution allowance includes rendering, scoring, receipt import and replay.
Dispatch must leave more than the consensus signer's 10-block minimum commit
margin. Operational phase reservations and round reservations retain distinct
budget identities across restarts.

[Qualified publication](QUALIFIED_TESTNET_PUBLICATION.md) binds the exact
activation, round plan, certificate, state, chain runtime and fallback
qualification. The native signer persists the SDK's encrypted commit before
broadcast. Unknown responses require reconciliation. Inclusion alone is not
effective weights: the reader must attribute the automatic reveal and verify
the finalized vector and UID/hotkey mapping.

The dashboard consumes an atomic runtime JSON file outside its build directory.
`forge/launch-operations/v1` reports the four separate states, the bootstrap
reference, active queue, recorded capacity/worker health and costs. Unrecorded
cost is unknown; reservations are not measured spend. A missing media file is
shown as unavailable. The certified exporter remains responsible for any king.

## Operator entry points

`forge launch preflight` reads the finalized testnet identity and separately
reports native fallback qualification. Require `native_fallback.status=QUALIFIED`
before paid allocation; an upgraded unreviewed runtime blocks live mode before
wallet access or worker dispatch. `forge launch run`
defaults to shadow mode; live operation requires explicit local wallet names,
validator labels, a frozen worker command and a positive cost reservation.
Wallet names are handles to the existing wallet mechanism, never secrets.

Before the first round, initialize the empty campaign directory offline with the
reviewed raw qualified evaluator package and replayable bootstrap reference:

```bash
forge launch initialize --config /absolute/launch-config.json \
  --contract /absolute/wan-mvp.json --root /absolute/operator-root \
  --bootstrap-reference /absolute/bootstrap-reference.json \
  --qualification-package /absolute/qualification-package.json \
  --activation-block FINALIZED_BLOCK
```

`FINALIZED_BLOCK` is a positive finalized height supplied by the operator. The
command replays the qualified evaluator package and complete bootstrap receipt,
creates genesis from the bootstrap reference, preserves the original Wan50
quality anchor, and reuses the one canonical `budget.sqlite`. It is restart-safe
only for identical inputs and rejects a progressed campaign or changed evidence.
It creates no worker-ready receipt and grants no GPU, chain-signing, rendering,
or publication authority. When a confirmation exists, its replayed qualification
receipt must exactly match the supplied qualification package. A v1 `KEEP_WAN50` bootstrap
receipt contains screening rows but not the raw screening evaluator custody;
the operator must review that separate baseline-study receipt index/import/replay
before this offline initialization.

One campaign directory holds `budget.sqlite` for both service state and every
compute reservation, `state.json`, `worker-ready.json`, the raw
`qualification-package.json`, and `manifests/{king,anchor}.json` with their
fingerprints. The ready receipt binds evaluator/runtime identity and measured
capacity. An imported baseline additionally requires the replayable
`bootstrap-reference.json`; its Wan50 quality anchor remains unchanged after
later promotions. Canonical validator SQLite snapshots retain the entire state
ancestry across rounds.

The worker command is a JSON argv array supplied by the operator. It receives
`--request <immutable request>` and `--result <result path>`. The operator must
run it against the authorized remote resource with its deadline and closure
guard. The service writes a dispatch intent first; an interrupted dispatch is
unknown and is never automatically repeated.

Remote `forge.evaluation.launch_campaign` provides `prepare`, `render` and
`score` stages. Generation and VideoReward run in their separate pinned Python
environments. The workload includes all 32 quality prompts, the eight ABBA
timings, three diversity seeds, and a separate original-Wan50 panel when the
current reference differs. It reports the complete call count, including
warmups, before execution. Artifacts stay under `/workspace/forge`.

`render` and `score` require `--custody-deadline <unix time>`. Every completed
render batch freezes its video/report hashes and waits for the operator's matching
backup acknowledgement before the next batch starts; score receipts use the same
barrier. This wait is outside measured generation timings. The SSH worker config
requires `custody: {pod_id, worklist_id, archive_argv, archive_bucket, timeout_seconds}`. The fixed
absolute `archive_argv` invokes `scripts/evaluator_pilot_archive.py`; the timeout
is 1–900 seconds per batch and must fit the campaign budget/deadline. Identities
must match the already owned pod and campaign. The adapter supplies exact per-batch
allowlists, streams to private S3 and checks every archived member hash against the
request before releasing the worker. Local `media-custody/` retains only manifests
and receipts. Failure or uncertain upload blocks progress; restart reuses a verified
receipt and never blindly repeats an upload intent. Old remote-worker configs must
be migrated before dispatch. This neither rents a GPU nor changes model scoring.

The operator fetches only the indexed JSON receipts from its authenticated
worker. `forge.validator.launch_receipts` verifies their byte hashes, imports
the graph into validator custody, replays it, and emits a result bound to the
original request and exact frozen committee. Miner-provided scores are not an
input to that importer.

To enable certified publication, `--publication-config` names a JSON object
with `evidence_origin: "external_gpu_qualified_v2"` and the explicit
`validator_labels`. This declares provenance; raw replay and native chain
checks still enforce authority. Without that configuration the service stops
at `CERTIFIED_AWAITING_PUBLICATION`. `--watch` reconciles every 12 seconds by
default. `forge launch export` can merge operational data into a compatible
dashboard snapshot; `--base-snapshot` preserves the certified king data.
Before the first certified snapshot, omit `--base-snapshot` and export the
operations document alone. The bootstrap argument accepts the exact replayable
reference receipt; only its validated public fingerprint, recipe label and
pinned model source enter the projection. Neither output grants a crown.

## Acceptance boundary

Acceptance requires a fresh testnet commitment, new C/E, complete GPU evidence,
commit/reveal, certificate, an actual process restart without duplicate actions,
qualified publication or fallback, attributed finalized readback, and a
dashboard transition after the build. A real certified `KEEP` is valid.
Synthetic `PROMOTE` integration fixtures exercise enforcement but cannot
authorize the live publisher.

The experimental subnet campaign has a user-authorized cumulative cap of 60 USD.
The canonical ledger counts setup, interrupted studies, GPU, disk and transfer
costs together. Phase allocations are operational reservations, not additional
spending authority or measured charges. No new work may commit more than
55 USD cumulatively, preserving 5 USD for closure. Warn at 48 USD. The private
miner campaign has its own ledger and does not contribute funds to this cap.
An independent guard archives evidence and closes only the campaign's owned
resource. GPU, disk and transfer charges must be reconciled before claiming the
campaign complete. Recurring operation requires its own budget.

Heavy evidence from this campaign expires in its scoped private S3 prefixes
after 30 days. Small reports, artifact hashes and archive receipts remain in the
project. S3 lifecycle deletion is asynchronous; unrelated bucket prefixes keep
their existing policies.
