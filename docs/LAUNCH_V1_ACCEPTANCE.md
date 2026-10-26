# V1 testnet acceptance — 2026-10-15

Scope: model competition only, testnet 578. This record distinguishes public
submissions, diagnostic measurements, signed certification and active rewards.
The rule is in [LAUNCH_V1.md](LAUNCH_V1.md); creator rewards are outside this phase.

## Public submissions

Three immutable public LoRAs were committed through registered testnet miner
hotkeys. Finalized commitments were read back; anonymous weight downloads returned
HTTP 200. None is a certified winner.

| Candidate | Pinned revision | License | UID | Finalized block |
| --- | --- | --- | --- | --- |
| [LongCat step1500](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-longcat-step1500) | `791dc13f1e672d4643c736b273d8c91a85da6b40` | MIT | 6 | 8,220,500 |
| [LongCat step1000](https://huggingface.co/lightx2v/Wan2.1-T2V-1.3B-longcat-step1000) | `c6b89d290401bba9587d532723544678c92f691e` | MIT | 4 | 8,220,505 |
| [SOLACE](https://huggingface.co/wookiekim/Wan2.1-T2V-1.3B-SOLACE) | `a68333b95a3c8f9f4ef56c4a432c8cac98f76a1b` | Apache-2.0 | 5 | 8,220,510 |

Wan UniPC44 and UniPC40 are operator recipe controls, not independent miner
submissions. The common base is Wan2.1 T2V 1.3B, revision
`0fad780a534b6463e45facd96134c9f345acfa5b`.

LongCat's reward-based training overlaps the VideoAlign/VideoReward family.
Its score is therefore especially sensitive to reward-model gaming; it is not an
independent demonstration of general perceptual quality. Custom-pipeline models
requiring untrusted code were excluded from this bounded loader experiment.

## First small batch: completed diagnostic

RunPod A100 80GB PCIe, one independent smoke prompt: a rotating paper pinwheel,
seed 630214. This prompt is outside the frozen 16-prompt panel. Each recipe was
rendered once after warmup; the comparison is not an ABBA confidence estimate.

| Measurement | Wan50 reference | Wan44 control |
| --- | ---: | ---: |
| Full generation seconds | 208.538569 | 184.515403 |
| VideoReward VQ | -0.085124 | -0.015606 |
| VideoReward MQ | -0.393531 | -0.102210 |
| VideoReward TA | 0.622786 | 0.597906 |

The observed single-pair speed ratio was 1.1302x. This proves rendering and
scoring work, not that the 1.10x lower-confidence gate or the quality panel passes.
Both models peaked at approximately 16.75GB allocated GPU memory.

All five candidate artifacts passed CPU admission. No full panel fit the remaining
small-lease capacity: the conservative Wan44 estimate was 14,870.71 seconds versus
6,138.51 seconds available. Full candidate gates are **INCOMPLETE**, not failures
or wins. The capacity guard correctly refused to begin a partial panel.

Conservative compute/storage exposure: approximately 1.06 USD; exact provider
billing was not available. The pod was terminated and provider absence verified.
The 622,936-byte private evidence archive was read back and hash-verified:
`ba38b45e0cfa082b74e18b6f4ffb6a29766c4c08dc2eacae5e68f9c9b0752c8a`.
It contains source, manifests, raw reports and both generated videos, not weights.

## Follow-up batch: completed

The H100 NVL campaign completed three full panels under the same frozen revision1
policy and panel. Each measured 16 quality prompts and eight ABBA timing prompts.
The quoted GPU rate was 3.19 USD/h; the supervisor budgeted 3.30 USD/h including
disk. The earlier smoke measured 147.471081 seconds for Wan50 and 131.059752 for
Wan44; the full-panel estimates below replace that single-pair speed observation.

| Candidate | Speed ratio | Lower 95% speed bound | Frozen-policy outcome |
| --- | ---: | ---: | --- |
| Wan UniPC44 | 1.128272x | 1.127283x | PROMOTE_ELIGIBLE, diagnostic only |
| LongCat step1500, 16 steps | 2.613627x | 2.609989x | INCONCLUSIVE |
| LongCat step1000, 16 steps | 2.622482x | 2.619257x | FAIL |
| SOLACE, 16 steps | — | — | INCOMPLETE: not run for capacity |
| Wan UniPC40 | — | — | INCOMPLETE: not run for capacity |

Wan44 passed the three provisional quality margins against both the king and
the fixed Wan50 anchor, which were the same model in this campaign. Its mean
VQ/MQ/TA deltas were -0.067346 / +0.096474 / -0.059417, with lower 95% bounds
-0.161092 / -0.018154 / -0.170023. No applicable temporal guard fired. This is
eligibility under the diagnostic's frozen rule, not a signed competitive win.

LongCat1500's TA lower bound was -0.234925, outside the -0.20 allowance, despite
a mean TA delta of -0.074833. The panel therefore could not establish the
required text-alignment tolerance. LongCat1000 also missed that confidence gate
(TA lower bound -0.218918) and triggered the steady-lighting temporal guard on
`confirmation-object-moving-stylized-5`. This flag is a provisional metric
decision, not a human-confirmed perceptual judgment.

The remaining lease had 5,471.95 seconds; the conservative full-panel estimates
were 8,651.92 seconds for SOLACE and 10,262.31 for Wan40. Neither was started.
The pod was terminated, and its absence was checked again through the provider
(404) on 2026-10-15. Conservative compute/storage exposure was approximately
18.79 USD for this batch, or 19.85 USD including the first batch, within 30 USD.
Exact provider billing and ancillary storage/transfer charges were unavailable.

The complete private archive was read back and verified: 48,375,364 bytes,
SHA-256 `ecbd6e5e448e2b07d26c3d1e381a3004e60dbfe226e98f5dcd1ce1364e77c238`.
It retains raw reports, score receipts, source, logs and videos. The three policy
results were reproduced from the byte-preserved archived evidence on 2026-10-15.

Policy hash: `67745893df7246e2adade5e755f2f04ce5dd6dcd23ef24d305387c09e53674fd`.
Panel hash: `fda13665b9d081dc783afd09b66d3c621814acd7b22fa9b4c4839cc47254c2e8`.
These operator experiments use explicit diagnostic provenance and cannot be
retroactively converted into rounds with post-close chain entropy.

## Matched attack diagnostic

One H100 smoke video was scored in its original form, after a matched no-op
re-encode, with neutral text, and under twelve fixed transformations: 15 queries
total, one source family. The campaign was paused between GPU jobs for this
64-second interlude; no generation-timing call overlapped it. Raw receipts and
all transformed videos are retained in the same remote campaign archive.

| Transformation | Delta VQ | Delta MQ | Delta TA | Frozen revision-1 guard |
| --- | ---: | ---: | ---: | --- |
| Freeze, then encode | -0.0348 | +1.4156 | -1.6732 | No flag |
| Shuffle frames | -0.2433 | +0.1259 | -0.1866 | No flag |
| Alternate brightness +3, phase0 | +0.0973 | +0.0226 | -0.0249 | Lighting flag |
| Alternate brightness +16, phase0 | +0.0139 | 0.0000 | -0.0373 | Lighting flag |
| Reward-seeking text, vs neutral text | +0.1390 | -0.0226 | +0.1120 | No flag |

Compression prevented the exact-pixel freeze diagnostic from firing:
`frozen_fraction` was 0 even for the repeated frame. Its decoded mean pixel change
(`temporal_delta`, normalized to 0–1) was 0.0000125033 versus 0.0290312786 for
the matched control. This is a concrete gap in revision1, despite the large TA
penalty. Revision2 now adds a separately versioned near-freeze guard for prompts
tagged `motion_required`: candidate mean pixel change at most 0.1/255 and at most
1% of a moving reference. Replaying the 15 diagnostic receipts catches this
encoded freeze; the other control outcomes are unchanged. Revision1 measurements
continue to replay with their original rule, including the completed H100 campaign.
Revision2 policy hash:
`c112f72206b8710f59d5a03847d27ee8cf3fe9f822a310a0a446030903779cd8`.

Shuffling and holding frames can also improve MQ; MQ alone is not evidence of
correct motion. The text comparison shows score sensitivity, not a demonstrated
successful instruction attack. Contrast/sharpness score gains do not establish
quality loss without perceptual evidence. This diagnostic is too small to claim
gaming resistance or detection rates across models and prompts.

## Software and dashboard checks

- Core suite: 247 passed, 4 skipped; Ruff passed. Three local LoRA runtime checks
  need Torch, absent from this CPU checkout; one container check needs an explicitly
  supplied pinned local image. These skips are not GPU qualification. Integration exercises filesystem
  receipts, strict replay, tamper rejection, SQLite restart, signed committee
  decisions, fixed genesis anchor and latest durable certificate-bound intents.
- The operator CLI exercises prepare, open, bundle and report across real subprocesses,
  native signatures and SQLite. Raw receipt bytes survive import unchanged. The
  bridge and certificate replay reject a changed post-close entropy panel, even
  if the caller recomputes the plan hash. CLI checks alone cannot bypass that boundary.
- Dashboard build passed; 1,140 browser checks passed at desktop 1920 and mobile390,
  including all 16 slides of the current-rule deck and its scrollable mobile states.
  Claude Opus 5.5 reviewed the rendered states, including failed, inconclusive,
  incomplete and eligible-but-not-crowned cases.
- Latest staged release strict check: 1,711 passed, zero failures or unverified
  media files, including the completed H100 panels and the legacy clips.
- Real diagnostic snapshot preserves the existing king and crowns, distinguishes
  operator recipes from finalized testnet submissions, and publishes pinned
  weight URLs, hashes, licenses and LoRA base references.

The dashboard is deployed and its public HTML, JavaScript, CSS and snapshot
bytes match the verified release, including the deck after following its canonical
URL redirect. Active result release: `20261015T075712Z-launch-v1-r4-results`.
Public browser checks passed for all five H100 candidate pages at 1920 and 390px:
exact download links, operator-only provenance, correct incomplete states for
unrun candidates, no overflow or page exceptions. Both videos played successfully
for a prompt from each of the three completed panels at each viewport. All 16
deck slides were also checked publicly at both widths. Earlier legacy playback
checks remain separate evidence.

The real-data check caught and fixed an exporter defect: a frozen plan without
measurements emitted no V1 quality object, causing the legacy display to appear.
It now emits INCOMPLETE with null measurements and the exact policy/evaluator
identity. The fix passed the actual JavaScript contract and public browser check.
Deployment also exposed unreadable public file modes; only generated public
files were changed to 0644, and staging now checks web-server readability.

The first batch's private S3 archive was restored on the existing Attelier server
under its remote evidence workspace. The complete archive hash and size matched;
raw receipts were extracted without JSON reserialization and the diagnostic
snapshot replayed successfully. This recovery check did not alter rewards or
publish fabricated measurements. The local worker's convenience JSON copies are
not substitutes for byte-preserved receipt evidence.

The automatic post-campaign publication stopped before activation on a recovery
bug: it treated the matched-attack diagnostic's `result.json` as a candidate.
Candidate discovery now follows the frozen plans. Recovery also exposed an
exporter mismatch with the native timing schema: it required an aggregate
`status`, while the runner records status per measurement. The exporter now
checks the native paired-timing schema and every measurement status. Integration
coverage verifies successful native receipts and rejection of a failed timing
measurement. No scores, margins or historical receipts were changed to publish
the results.

Local tests and a working dashboard do not prove a real V1 round has closed or
weights have been activated. No new reward publication was performed in this campaign.

## Fresh certified-round preparation, 2026-10-15

The signed V1 voting workflow and certified dashboard exporter pass the complete
CPU suite: 249 passed, four explicitly skipped dependency/image checks. New
integration covers durable reservations before external signatures, process
restart, missing/altered receipt rejection, two distinct validator journals and
rejection of a hash-consistent journal redirected to the wrong winner UID. The
serialized certified export also passes the dashboard's actual JavaScript
contract. These are integration fixtures, not measured GPU qualification.

The dashboard's shared-execution and certification states pass 1,415 browser
checks. Claude Opus 5.5 reviewed desktop 1920 and mobile 390 renders, including
KEEP, PROMOTE, pending activation and a certificate without a matching crown.
Certification and finalized weight activation are displayed separately.

A fresh A100 capacity pilot completed with 188.756 s for Wan-50 and 167.380 s
for Wan-44. Opening the real round then failed because the native SDK client
was accessed before its lazy connection opened. No close/entropy selection,
vote, certificate or candidate weight publication occurred. The guard archived
the pilot, verified its backup and deleted the pod; conservative compute/disk
exposure was approximately 0.852 USD. Archive SHA-256:
`832baaacc957942e601809c43c1af8b529e73f328d078765b48bf996ec1e60cc`.
The corrected connection order subsequently passed fresh native read-only
connections for freeze, certification and weight publication, including the
actual miner admission and unchanged runtime. A new bounded attempt must use
its own future entropy and measurements; the smoke result cannot certify it.

### Fresh panel completed; certification expired

The second A100 attempt completed all 48 measured generations plus four warmups
and the 32 VideoReward scores. Its native testnet close was block 8226174 and
entropy block 8226179, under the unchanged revision-2 rule. Wan-44's result was
**INCONCLUSIVE / NOT_ELIGIBLE**:

| Check | Measured value | Required bound | Result |
| --- | --- | --- | --- |
| Speed ratio | 1.127375, lower 95% bound 1.127221 | At least 1.10 | Pass |
| VQ lower confidence bound | -0.178150 | At least -0.35 | Pass |
| MQ lower confidence bound | -0.240141 | At least -0.20 | Inconclusive |
| TA lower confidence bound | -0.114626 | At least -0.20 | Pass |

Quality bounds were identical against king and anchor because both were Wan-50.
These are normalized reward units, not quality percentages. No tagged temporal
guard fired; this does not establish general motion or physics correctness.
The earlier favorable diagnostic is not a substitute for this fresh panel.

Both validator report commitments finalized, at blocks 8226957 and 8226960.
The deliberately restarted process then rejected its own saved commit envelope:
SQLite returned identical signed JSON with a different key order, while the
operator compared pretty-printed text. The commit/reveal deadline passed before
recovery; no reveals, certificate or new model weights were produced. This is an
operator integration failure, separate from the inconclusive quality result.

The operator now compares canonical JSON values while preserving the original
signed file bytes. Running the actual restarted path against both durable vote
stores passed without new transactions. The round was closed with
`EXPIRED_NO_CERTIFICATE`; it must not be backdated or retrocertified. Finalized
chain readback still showed both validators' owner-only burn vector.

The complete GPU archive is 13,362,215 bytes with SHA-256
`edb337a6c776446ef8efaac87f9f52b1a58618443cfb74fdb075eb2310eb5ebc`.
Backup verification completed and the provider subsequently confirmed the pod
was deleted. This attempt's conservative compute/disk exposure was about
5.722 USD, bringing the campaign total to about 26.425 USD of the authorized
30 USD. No additional GPU was launched. The public dashboard still requires a
separate truthful projection of this expired round; it must not display a
certified KEEP or promotion.
