# Evaluator development pilot: staged authority

This is a local preparation, not an authorization to rent a GPU, call Luna,
publish artifacts or change testnet. The entire objective remains incomplete
until a qualified faster candidate closes a real distributed round.

### Measured payload capacity (2026-10-07)

A twelve-comparison diagnostic using the current six-sheet representation
produced 328,185,047 compressed bytes and a largest JSON row of 50,038,687 bytes;
nine rows exceeded the former 32 MiB row limit. The judge now allows bounded
128 MiB rows and offers `--archived-payload` for larger campaigns. This reads the
verified private S3 archive twice with bounded memory: the first pass validates
every comparison and both archive/member hashes before any API dispatch; the
second checks each row against its prevalidated hash before using it. It never
downloads a local model/media archive. Payload and archive ceilings are 4 GiB
and 8 GiB; individual network reads time out after 60 seconds, with 900 seconds
of cumulative network wait per pass, excluding time spent judging rows.

The original stdin mode remains capped at 512 MiB compressed. Archive capacity,
backup duration and transfer cost must be budgeted for the actual media workload;
the small diagnostic's archive limits are not full-pilot defaults. Successful
transport or diagnostic calls do not waive the full-pilot custody/review gates,
qualify its 96 comparisons, or establish perceptual calibration.

### Randomized media selections and replay custody

Detail frames, crops and transition views are selected randomly. Identical video
bytes alone do not reproduce an earlier judge request. Retain the exact payload,
its hash, and a compact copy of its selection metadata and encoded-image hashes
before releasing the worker. If those inputs are lost, a newly selected payload
requires new request identities and fresh judgments; old settled responses cannot
be reused as if they evaluated the new images. Native video metrics may be reused
only after their receipts and source video hashes are verified independently.

## Decision and independent unit

The pilot asks whether the **same fusion used by competitive validators** can
separate the twelve declared controls from legitimate variation. It does not
qualify the evaluator or test UniPC-44. Each control counts a semantic source
family once; wrong-video comparisons count both disjoint source lineages.
Reusing the same source across different control kinds saves renders, not
independent observations. Static sources must retain static camera labels.

The development-only pilot contains eight motion and eight static families:
16 reference renders, eight new executions, eight new seeds; eight comparisons
for each of twelve controls, with both blind orientations. That is **32 measured
generations, 96 comparisons, 192 Luna calls**, plus separately budgeted warmups.
Media and model files stay on the remote worker and private backup storage named
in AGENTS.md. Only manifests and small receipts belong in this repository.

Before scores are available, freeze prompts, source lineage, seed, transformations,
applicability and the observability protocol. Inspect transformed media without
access to evaluation scores. An imperceptible blur, a shuffled clip without
meaningful motion, or a brightness change that is not a perceptual defect must
not become an observable positive because its pixels changed. Missing or
unobservable evidence stops the pilot; do not replace rows, relabel categories,
pick favorable seeds or inspect independent partitions to rescue it.

## Complete cost envelope

`configs/evaluator-pilot-costs.json` separates historical measured timing from
proposed operating allowances. The provider catalog returned RTX 5090 at
0.99 USD/hour with low stock on 2026-10-05; this is not a reserved offer.
The proposed envelope includes two warmups, 15 minutes startup, 30 minutes
metrics, 5 minutes encoding, 5 minutes transfer buffer and 8 minutes backup/closure,
plus storage, transfer and contingency. Luna runs after GPU closure. Its
**approximately 4.80 USD** total at the proposed combined compute/disk ceiling is
an estimate, not a demonstrated upper bound or a promise of completion.

Reproduce the all-in calculation with the project interpreter:

```sh
.venv/bin/python scripts/evaluator_pilot_preflight.py \
  --pilot configs/evaluator-pilot-manifest-v1.json \
  --costs configs/evaluator-pilot-costs.json \
  --guard-envelope configs/evaluator-pilot-guard-envelope.json \
  --out /absolute/path/to/new-preflight.json
```

The command checks the actual frozen pilot ledger, preserves existing output,
and returns exit 2 for incomplete or over-cap preflights. Unknown values cannot
silently become zero. Even an estimate within the cap never authorizes dispatch.
The committed example remains incomplete because its allowances and execution
prerequisites have not been qualified.

Its receipt includes the exact worklist: prompt, seed and immutable job identity
for every render, 56 local transformations (including explicit frame permutations),
96 comparisons, 192 blind calls and 88 unique metric-video jobs with five native
VBench dimensions each. Warmups are separate and ineligible as observations.
The worklist describes preparation; no GPU/API executor is invoked by this CLI.

The pilot reserves 0.30 USD before each Luna request (legacy callers retain
0.08 USD), much more than the measured
mean 0.00462634375 USD/call. The 0.006 USD/call planning allowance is **not** its
worst-case charge. Reserve each request, settle actual usage, retain uncertain
charges and stop before the next request would exceed the remaining campaign
envelope. A limited run may stop before 192 calls; that is an incomplete pilot,
not permission to exceed 5 USD. Judging now requires a verified closed-pod receipt,
so no GPU is intentionally retained for API latency.

Before rental, verify the exact offer and disk/transfer rates; bind the actual
container digest, dependencies, loaded weights and DINO cache verification;
finish an all-in cost guard that warns at 4 USD and closes resources before
5 USD, retaining closure/backup funds. Archive evidence and verify resource
closure. No new or existing resource is implicitly authorized by this document.

### Local closure supervisor

`scripts/evaluator_pilot_guard.py plan --envelope
configs/evaluator-pilot-guard-envelope.json --out /absolute/new-envelope.json`
is offline. The proposed partition withholds 1.20 USD for API, 0.35 USD for
ancillary costs and 0.50 USD contingency, leaving 2.95 USD for compute and
running disk. At the proposed combined ceiling of 1.01 USD/hour, this permits
10,514.85 billable seconds, including 300 seconds for backup and 180 for closure.
Only 10,034.85 seconds remain for work. The old plan, which kept the GPU during
judging, exceeded this window by 55.03 seconds. Moving judging after verified
closure reduces the work estimate to 9,489.88 seconds, leaving **544.97 seconds**.
This changes the execution order; it is not a new timing measurement.
These allocations are proposals, not measured charges or new authority.

### Prepared execution boundaries

The proposed worker is one Secure Cloud RTX5090 with 40 GB container disk and
an 80 GB **new pod volume** mounted at `/workspace`; no existing/network volume
is attached. The 28.93 GB model snapshot alone does not fit a default 20 GB
workspace. Confirm both disk allocations and their combined running rate before
accepting placement. The native CLI exposes `--container-disk-in-gb 40` and
`--volume-in-gb 80`; the existing MCP create wrapper does not expose pod volume
size. An uncertain create response requires reconciling the unique pod name,
never blindly creating a second resource.

The registry-verified Linux amd64 base is
`runpod/base@sha256:c65777a69057e5872abdebdeb7929a1e137db2f712c9614f9a0784959f95a5bb`.
That registry observation is not proof of a running container. Compare the
provider's deployed immutable image, GPU, disks and rate after placement; preserve
the provider receipt alongside actual dependency freezes and loaded-weight checks.
Abort before observations if they differ from the reviewed launch proposal.

On the owned worker, `scripts/evaluator_pilot_bootstrap.sh` prepares separate
generation/metric environments on local container storage at `/opt/forge-pilot-runtime`,
pinned source checkouts and verified weights. It requires at least 32 GB free
container storage; models and evidence remain under `/workspace/forge`. Both
environments share a local pip download cache. After generation dependencies
are ready, verified asset downloads overlap metrics installation. Both must
succeed before the real DINO probe. `bootstrap-events.jsonl` and
`bootstrap-assets.log` preserve startup stages and download errors for backup.
Run it from `/workspace/forge` under a **900-second process-group timeout** while
the independent closure supervisor is already running. It refuses existing
environment/model destinations; a partial bootstrap needs reconciliation.
It finishes by executing the pinned DINO cache probe on CUDA and recording actual
feature shape, hashes, GPU and dependency version. The script has only shell
syntax/import checks locally; neither installation duration nor GPU behavior
has been verified for this new pilot.

The public Wan-50 reference and its 19 file hashes are in
`configs/wan50-reference.json` (28,928,887,859 bytes before runtime dependencies).
`configs/evaluator-pilot-observability-v1.json` freezes the 12 control rules and
96 review identities; it contains no completed review. See
[the review protocol](PILOT_OBSERVABILITY.md).

`scripts/evaluator_pilot_render.py` prepares the exact 34 generation jobs without
GPU execution unless `--execute` is supplied. The pilot explicitly permits the
50-step reference; the existing worker's implicit warmup is disabled because
the two warmups are already jobs. Execution binds the original absolute deadline
across restarts, preserves terminal uncertain jobs, derives all 56 frozen controls
and writes the 88-video index. Each transformation binds source/output hashes,
kind and exact options (including the frame permutation). Real CPU codec tests
cover derivation and resume behavior; they do not demonstrate Wan inference.

`scripts/evaluator_pilot_measure.py` consumes a complete 88-video media index
on `/workspace/forge`. It runs the existing isolated VBench implementation with
each timeout bounded by the remaining absolute deadline, then builds 96 compressed
sheet payloads with the competitive media-selection functions. Its terminal
journal binds every metric receipt, media hash and prepared payload hash. It makes
zero API calls; its scores must remain inaccessible to the observability reviewer.
An interrupted output is retained and never blindly retried.

`scripts/evaluator_pilot_judge.py` accepts the compressed payload stream on stdin,
or reads the archived member directly with `--archived-payload --archive-bucket
<private-bucket>`, after verified backup and owned-pod closure. The standalone
`python -m forge.evaluation.pilot_payloads` archive route likewise requires
`--archive-bucket <private-bucket>` unless `--local-archive` is supplied. Both
compare that explicit bucket with the receipt before invoking the S3 client. It checks the archived manifests,
all 88 metric receipts, all 96 reviewed comparisons and the complete bounded
payload stream before creating an API ledger. It then uses the existing blind
`judge_pair`, a 1.20 USD API partition and the **same numerical fusion function**
as competitive evaluation. An uncertain response, reused response or exhausted
reservation stops the run. Development separation never grants qualification.
Both entrypoints have local import/CLI checks; CUDA, DINO kernels, actual archive
delivery and paid judging still require the bounded pilot and have not run here.

RunPod lists running container disk at 0.10 USD/GB/month and no ingress/egress
fee in its [official pricing](https://docs.runpod.io/pods/pricing). The 1.01 ceiling
must cover the actual selected GPU and disks together. Backup storage and readback
costs are separate; an external storage provider's transfer costs are not zero
just because RunPod egress is free. Do not attach an unbudgeted persistent volume.

`supervise --ownership /absolute/ownership.json --state /absolute/guard.json`
acts on an **already owned pod**, using the installed `runpodctl`. Its private
ownership receipt binds `pod_id`, `name` (prefix `attelier-evaluator-pilot-`),
`worklist_id`, `request_started_unix` (recorded before creation), the complete
`envelope`, absolute `archive_argv` and `repair_notify_argv`, and explicit
`terminate_on_backup_failure_authorized`. The archive program must verify remote
readback and return matching `pod_id`, `worklist_id`, `status: VERIFIED`,
`readback_verified: true` and `evidence_complete: true`. Partial archives are
retained but do not permit an early successful closure.
The example config supplies neither a real ownership receipt nor authority to
invoke the archive program.

The supervisor locks one durable state file, emits a warning at 80% reserved
exposure, starts backup before the closing window and checks exact provider ID
and name immediately before deletion. A `.close` sibling requests early closure.
Deletion acknowledgement alone is insufficient: the full provider list must
confirm absence. Restart does not repeat a recorded backup attempt. Failed backup
does not silently become verified, and closure still proceeds only under the
explicit loss-at-deadline authorization. Unrelated pods are never selected.
Provider failures keep retrying cleanup, including after the deadline, while
recording any overrun. Ownership mismatch requires operator reconciliation.

This is **not a provider-enforced spending cap**. A dead supervisor, network outage,
unbounded API request, or provider billing discrepancy can defeat a local deadline.
The actual launcher must enforce the API/ancillary envelopes, stop new work before
backup, supply the verified archive implementation, supervise this process and
reconcile final provider charges. Those integrations remain dispatch blockers;
the passing local process-boundary test is not proof of a live guard or backup.

`scripts/evaluator_pilot_archive.py` is the reusable archive boundary. It accepts
an exact ownership object (`pod_id`, `worklist_id`, `ssh_argv`) and an exact
archive configuration rooted only at `/workspace/forge`: a private `s3_bucket`
name (not a URL or path), a new remote `.tar.gz` name, an unissued
`subnet/experiments/...` S3 key prefix, an explicit relative-file
allowlist, per-member and total-archive byte caps, and a bounded timeout. The
archiver appends a fresh UUID to the configured key before every transfer. It
first creates `<out>.intent.json` with exclusive creation; that durable intent
blocks a retry after any failed or interrupted transfer until reconciliation.
The allowlist rejects credentials,
environment files, model/weight extensions and unsafe paths. It streams the
remote archive straight to the private bucket after a definitive S3 404, then
reads the complete object back before writing its local receipt. S3's streaming
CLI path cannot make a conditional PUT, so the caller must use an unissued,
exclusive experiment key; a concurrent writer makes the archive result unsafe
and requires reconciliation. That receipt
contains `status: VERIFIED`, the pod/worklist identities, `uri`, `bytes`, and
`archive_sha256`; it is absent after any failure while the intent remains.
The total cap bounds both archive storage and the full readback the judge will
later consume. Supply only small state,
manifests, metric receipts, observability records, selected evidence media and
the compressed Luna payload stream that the post-GPU judge needs. It neither
starts a worker nor creates a real archive until explicitly invoked with reviewed
files and current spending authority.

The allowlist distinguishes the runtime's direct Python sources under
`subnet/src/forge/models/` from downloaded model data. Weight files, other
contents of that package directory, credentials and secret paths remain rejected.
Validate the actual allowlist before relying on an unattended close.

An early backup failure records `BACKUP_FAILED_AWAITING_REPAIR`, the error
type, child exit code and a private per-attempt process receipt. It retains the
owned pod until the budget-derived backup deadline, reserving closure time;
it no longer deletes after a short repair interval while unused allowance remains.
This may consume the remaining authorized compute allowance during repair.
Reconcile the archive intent and any object transfer before
creating the guard state's `.retry-backup` marker. That marker authorizes one
new attempt, not an automatic retry of an uncertain upload. Without repair the
guard still closes at that deadline and records `backup_verified: false` and
`BUDGET_DEADLINE_WITH_UNVERIFIED_BACKUP`.
Resource closure therefore never substitutes for a verified evidence backup.

`repair_notify_argv` receives a safe JSON event on stdin containing the pod,
ownership identity, archive attempt, repair deadline and `alert_id`. The reviewed
notification receiver must return exactly
`{"status":"DELIVERED","alert_id":"<same id>"}` after accepting delivery.
The guard persists `<state>.alert.json` and records the acknowledged alert ID;
a command exit alone is not delivery. This proves receiver acceptance, not human
reading or successful repair. A real receiver and observed delivery must be
checked before a future unattended launch. Historical frozen ownership files
require explicit migration.

For a guard running in the operator's desktop session, the concrete adapter is
`python -m forge.evaluation.pilot_notify --receipts /absolute/private/receipts`.
It sends a persistent critical notification through the session's
`org.freedesktop.Notifications` service and requires a positive notification ID
before acknowledging the exact alert. Its private receipt survives a lost guard
acknowledgement; retries of that ID reuse the receipt. A
`CUSTODY_NOTIFICATION_PROBE` event exercises the same delivery path while clearly
labeling the notification as a preparation test. The adapter does not prove a
human read the alert and does not monitor failure of the operator's host itself.
Verify the real desktop receiver in the same session/environment that will run
the guard; a headless service without that session is not covered.

The archiver persists `<out>.status.json` before compression, destination lookup,
upload and full readback, plus `<out>.manifest.json` before upload. Failures retain
phase, elapsed time, timeout classification and available exit code at mode0600.
Raw provider stderr, command arguments and exception messages are not persisted.
Keep the original intent after failures; reconcile rather than repeating uploads.

Launch V2 now backs up each completed render batch and the final score receipts
before advancing. Each batch is immutable and has its own S3 object and readback
receipt. A later archive or optional checkpoint failure cannot erase those earlier
backups. Model weights remain excluded from this evidence archiver; any separate
checkpoint transfer must follow verified evidence custody and requires its own
size/timeout validation. No multi-GB checkpoint transfer is certified by the small
offline process tests.

## Exit and next stage

Continue only if all twelve controls are observable under the preregistered
rules and development fitting separates every degradation kind from legitimate
variation. Otherwise stop and record the limitation. No candidate-specific
threshold tuning is permitted.

Only after a successful pilot and separately approved funding: complete
development, freeze the evaluator, calibrate on 32 independent families/control,
then check other independent families. Preserve sensitivity lower bound 0.80,
false-alarm upper bound 0.15 and the policy's interval level. A changed evaluator
requires new qualification and independent checks. Full 128-prompt same-model
and degraded-panel evidence and cross-worker receipts remain separate gates.

`configs/wan-unipc44-cfg5.json` declares the next candidate using the existing
public original Wan transformer at 44 UniPC steps and CFG 5. Its manifest parses
locally; public availability, execution, speed and quality remain unverified for
this candidate. Its eventual screening is fresh and conditional on a qualified
evaluator. Rejection or inconclusion retains the king and does not authorize a
second experiment.

### Price verification and storage horizon

The pilot reserve bounds a complete 1,050,000-token context at the published
long-context cache-write rate, 8,192 output tokens and 10% regional headroom:
`(1,050,000 * 0.25 + 8,192 * 0.75) / 1,000,000 * 1.1 = 0.2955084 USD`.
The actual request uses the global endpoint and default service tier, without
tools. Ledger settlement applies long-context pricing above 272,000 input
tokens and the cache-write premium; missing write detail is conservatively
charged as all-input writes. Cached reads receive no assumed discount.
This is a usage-based conservative accounting estimate, not a provider invoice.
Sources checked 2026-10-05: [model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [cache accounting](https://developers.openai.com/api/docs/guides/prompt-caching).
The 1.20 USD API allocation is not a guarantee that all 192 calls can complete.

The private backup bucket is in eu-west-1. For at most 1 GiB, three complete
readbacks, one month of standard storage and 0.01 USD of request allowance,
using 0.09 USD/GB egress and 0.023 USD/GB-month storage gives about 0.325 USD,
within the 0.35 USD ancillary allocation. Sources: [S3 pricing](https://aws.amazon.com/s3/pricing/)
and the [regional price catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/eu-west-1/index.json).
That horizon does not authorize indefinite storage. Continued retention requires
a later funded stage; no bucket lifecycle or existing object is modified here.
Partial backups preserve available evidence before closure, but
`evidence_complete=false` explicitly blocks all pilot Luna dispatch.

### Cold-start deployment checks

Before accepting a placement, inspect its actual download/upload specification
and mounted filesystem, not only the GPU model. The operator path requires at
least 1,000 Mbps advertised download and 500 Mbps upload; this still does not
prove sustained model-download or shared-filesystem performance. The 900-second
startup timeout includes image readiness, dependencies, weights and DINO. A cold
bootstrap that misses it is an incomplete pilot and must not enter generation.

Transfer source with `tar --no-same-owner --no-same-permissions`, retain transfer
stderr, and require the exit status to succeed. Restoring a local archive's
UID/GID is unnecessary and can fail on provider shared storage. Download weights
with `fetch_verified`'s supported default deadline; do not pass an override above
its bounded maximum. Previously closed attempts remain charged to the same5USD
campaign: subtract their conservative exposure from the next compute allowance,
without resetting the cap or weakening backup/API reserves. A new paid attempt
needs a resolved cause and a reviewable remaining-budget preflight.

### Retrying within a cumulative campaign budget

Pass `--prior-campaign-result /absolute/path/to/pilot-campaign-result.json` to
the preflight after failed attempts. Every prior pod must have verified closure;
duplicate pods, invalid exposures and inconsistent totals are rejected. The
preflight adds prior compute and running-disk exposure to the campaign estimate
and withholds it in the emitted `guard_config`. Use that exact guard configuration
for a subsequently authorized retry; do not add the exposure a second time.
The full ancillary envelope remains reserved for unbilled storage and requests.

The default cap remains 5 USD. A different explicit planning cap can be evaluated
offline, but neither a successful preflight nor an edited cap grants authority
to spend. The guard warning follows 80% of its configured campaign cap. Preserve
the original create-time startup deadline, archive/closure reserves, source-upload
`--no-same-owner --no-same-permissions` flags and transfer stderr. Refresh the
actual combined GPU/disk rate before any rental. Faster startup is an unmeasured
hypothesis: retain the 900-second allowance until real evidence supports a change.

### CUDA host admission before dependency installation

The frozen Torch build is cu130. Request `allowedCudaVersions: ["13.0"]`
when placing the worker, and run `scripts/evaluator_pilot_cuda_preflight.py`
through SSH before transferring/installing the runtime. The standard-library
probe checks the actual RTX5090 identity, NVIDIA driver branch580+, initialized
CUDA driver API>=13000 and one visible CUDA device. Its nonzero exit must request
owned closure. Bootstrap repeats the probe before creating either environment
and archives `cuda-host-probe.json`; provider placement metadata alone is not proof.
A passing driver probe does not replace the real DINO CUDA forward pass.
See [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and [RunPod placement filters](https://github.com/runpod/docs/blob/main/sdks/graphql/manage-pods.mdx).

### Lium development pilot

The Lium adapter uses the same durable supervisor through
`evaluator_pilot_guard.py supervise --provider-executable /absolute/path/to/provider`.
The fixed wrapper invokes `evaluator_pilot_lium_provider.py --ownership ...`;
its Lium client loads only this project's ignored `.env` (`LIUM_API_KEY`).
Deletion checks the frozen pod ID, name and executor again at the HTTP boundary,
then the supervisor checks `/pods` and requires HTTP404 from the owned pod detail
endpoint when it disappears from that active list. `DELETING` remains pending and
does not trigger repeated DELETE calls. Other account pods are preserved. A deletion acknowledgment alone does not establish closure.

Use a private, one-time template pinned to the existing `runpod/base` manifest.
Check the owned pod's executor inventory for the exact `pod_<owned-id>` container,
its container ID and its nonempty digest matching the pinned registry manifest.
An absent container is pending evidence, never a match. This inventory comes from
[Docker inspect scraping](https://github.com/Datura-ai/lium-io/blob/main/neurons/validators/src/miner_jobs/machine_scrape.py),
but the live pilot found it can remain older than pod creation. The available-node
catalog (`/executors?view=full`) excludes a fully rented host and cannot resolve
this gap. The pilot remains blocked until the actual owned container is evidenced;
a verified template alone does not satisfy this requirement. Bound any wait within
the startup budget, stop before installing dependencies, and do not repeat rentals
to test this same unavailable evidence path.
Check the template digest,
CUDA driver initialization and exact GPU before dependency installation. Keep
Torch2.11/cu130 and the frozen model manifests. Provider-reported metadata is not
hardware attestation; the real DINO forward probe remains required.

A whole-node single RTX5090 quote at0.75USD/hour allows a1200-second startup
allowance and network admission at800Mbps download/500Mbps upload; GPU timing,
API, ancillary and contingency allocations remain unchanged. These are planning
inputs, not new measurements or universal defaults. Require at least120GB free
workspace and32GB free container storage. No paid persistent Lium volume is
attached. Retain backups in the existing private S3 experiment prefix.

Freeze a unique creation intent before renting; reconcile an uncertain response
by exact name and executor without issuing another rental. Start the independent
supervisor immediately after the exact pod ID is known. Set and read back the
provider's precise scheduled removal at or before the guard's hard deadline.
Reassert and read back the same absolute deadline after RUNNING, before transfer;
creation can overwrite an earlier schedule with the initial termination-hours value.
Failure to verify the price or removal schedule requests immediate owned closure.
The source/model transfer and inference start only after the live guard is verified.

Lium bills from creation by the second; a provisioning failure can still cost
money. Refresh the offer and campaign's prior costs before each authorized attempt.
The local supervisor and provider removal schedule bound the planned exposure,
but neither guarantees billing during provider outages or a changed price.
See [Lium billing](https://docs.lium.io/pod-users/billing) and
[scheduled removal](https://docs.lium.io/developers/sdk/reference/client/lium#schedule_termination).

### Controlled inner Docker runtime

When Lium's executor inventory cannot identify the running pod image, a controlled
inner Docker container provides a separate inspectable workload. Sysbox supports
[an isolated inner Docker daemon](https://github.com/nestybox/sysbox/blob/master/docs/user-guide/dind.md).
`scripts/evaluator_pilot_dind.sh` uses only `/run/forge-pilot-docker.sock` and a
fresh Sysbox-managed `/var/lib/docker` directory on the owned disposable pod. It exposes
only the assigned NVIDIA devices, driver files and experiment directory to the
inner workload; it does not use the provider host's Docker socket.

The workload still uses the frozen `runpod/base@sha256:...` image.
`scripts/evaluator_pilot_container.py` checks its actual running container ID,
requested digest, loaded image configuration ID and repository digest against the
registry manifest. `evaluator_pilot_cuda_preflight.py` must then pass inside that
same container, before model downloads or evaluator installation. Run the worker
in that container and archive the inspect/CUDA receipts with the existing media
custody records. These checks establish operational provenance under the same
trusted-host assumption, not protection against a malicious provider.

Budget inner-daemon installation/pull explicitly: the current proposal allows
360seconds for this step inside an1800-second total startup window. The remaining
bootstrap time is calculated from original pod creation. Missing provenance,
failed GPU passthrough or exhausted startup time requests owned closure. Do not
fall back to running the workload directly in an unverified outer container.

The opt-in integration test uses an already installed local image without pulling:
`FORGE_TEST_LOCAL_IMAGE=python@sha256:<installed-digest> .venv/bin/pytest -q tests/integration/test_pilot_container_integration.py`.
It verifies matching identity, digest substitution rejection and rejection after
stopping the exact test container, then removes that container by ID.

Sysbox requires the inner Docker data root at `/var/lib/docker`; custom roots fail
with nested overlay mount errors. Refuse nonempty prior daemon state. This is the
private mount inside the rented system container, never the provider host cache.

Docker29 with the containerd image store can report the manifest digest as ImageID.
For that case the verifier also hashes the local manifest and config blobs and
checks their linkage. Legacy stores must expose the expected config digest as ID.
Both paths require the same pinned repository digest and actual running container.

### Recovering measurements without regenerating observations

If generation completed but metrics failed, preserve the original archive before
closing the worker. A separately budgeted recovery can reuse those exact videos;
it creates no new independent observations and must not count them again.
`scripts/evaluator_pilot_restore.py` requires a verified archive receipt, checks
the complete archive SHA-256 and byte count, and checks each selected member's
hash and size before restoring it into a new destination. It restores generation
media and provenance, not a failed measurement journal. Traversal, duplicate
members, nonregular files and an existing destination are rejected.

Use `scripts/evaluator_pilot_metrics_bootstrap.sh` inside the verified inner
container for this recovery. It installs only orchestration and metric environments,
checks pinned metric assets, and repeats the actual CUDA and DINO probes. Keep the
original generation runtime/freeze and the new metric runtime/freeze separately,
alongside the recovery-origin receipt. All transfers and extraction stay on the
remote worker; use the same cumulative campaign guard, including previous failures.

Pinned DINO source has no ignore rule for Python bytecode. A normal import can
leave `__pycache__` and make the next strict clean-checkout check fail. The DINO
probe and metric subprocess therefore disable bytecode writes (`-B`); source
integrity checks remain strict. Metric subprocesses preserve bounded stdout/stderr
sidecars on completion and timeout, next to their output directories. Include these in the
archive allowlist so a failed metric can be diagnosed after owned resource closure.
For every prescribed `pilot-measurements/metrics/<job-id>/receipt.json`, the
archiver also requires `metrics_eval_results.json` and `metrics_full_info.json`
from that directory. Missing files make the backup incomplete. Luna verifies the
archived native result against the receipt's `raw_sha256`; a receipt alone is
insufficient custody. This also retains native output when normalization fails
before a receipt can be written. Earlier archives without native outputs remain
diagnostic and cannot be upgraded by editing their metadata.

The recovery does not waive prescore observability. Timestamped contact sheets
can establish some spatial defects but do not automatically establish temporal
flicker. Record insufficient evidence as incomplete, preserve the sampled review,
and keep Luna blocked until all prescribed reviews are complete. Never substitute
pixel differences or metric scores for the required perceptual observation.

### Diagnosing native cosine precision

`scripts/evaluator_pilot_precision_probe.py` is a remote diagnostic for a frozen
freeze-control video. Run it under the metrics interpreter with the original
pilot, media index, verified media root, VBench config, job ID, a new output
directory, and the guarded absolute deadline. The supervising process must also
enforce that deadline with a process timeout. Expired or nonfinite deadlines
are rejected before GPU dependencies load.

The probe verifies the pinned VBench checkout and CLIP weight hash. It encodes
the video once, then calls the same upstream background-consistency function
twice with identical features: native dtype and float32 before normalization.
It preserves the features and both native results, their hashes, hardware and
probe identity. Include `precision-probe/diagnostic.json` and
`precision-probe/features.npy` in the private backup. The probe makes no API
calls and emits no quality or observability receipt.

This tests a numerical hypothesis, not a new acceptance threshold. A synthetic
float16 example or successful float32 calculation does not establish what
happened in a lost native result. A runtime precision change requires its own
provenance and fresh calibration; do not combine it with old qualified evidence.
The upstream calculation is documented in the
[pinned VBench background metric](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/background_consistency.py).

### All-frame temporal diagnostics (2026-10-08)

`load_video` now records all-frame temporal signals through `video_diagnostics`.
For clips with at least three frames, luminance acceleration is the absolute second
frame difference of mean BT.709 luminance after 32x32 BOX reduction; median and
95th percentile retain the 8-bit luminance scale. Centered transition magnitude
removes each frame's mean luminance before measuring spatial change. Pilot payloads
and their compact selection ledgers preserve these diagnostics alongside video
hashes; the automatic quality runner already includes decoded-video diagnostics.
They are not sent to Luna as instructions and do not change quality fusion or
promotion thresholds. Intentional flashes, cuts and camera motion need semantic
context. Short clips omit these fields rather than reporting fabricated zeros.

Independent seeds are not automatically quality-equivalent controls. Pre-score
observability review must establish equivalence on protected quality axes before
such samples contribute to evaluator-noise fitting or false-alarm calibration.
Real improvements and regressions remain generation variation. Updated perceptual
rules change the observability protocol identity; old review receipts cannot be
silently reused. Changed media source also changes evaluator identity, requiring
new calibration before certification.
