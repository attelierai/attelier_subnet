# Dashboard validator-report API

`forge dashboard-api` exposes a loopback-only observability endpoint for reports
that validators have already signed.  It is designed to sit behind the dashboard
reverse proxy; it is not a scheduler, a coordinator, a resolver or an authority
for a king, certification or weights.

```sh
forge dashboard-api --database /var/lib/forge/dashboard.sqlite \
  --plan /var/lib/forge/frozen-plan.json --host 127.0.0.1 --port 8088 \
  --snapshot /srv/attelier/dashboard/public/data/status.json
```

The operator supplies the frozen plan files.  The server verifies the report's
Sr25519 hotkey signature and that its assigned evidence belongs to that plan.
That establishes authorship and plan membership only.  `chain_verified` and
`evidence_replayed` remain `false`; the distributed resolver separately verifies
finalized chain context and raw evidence before a result can be used elsewhere.

`POST /api/v1/validator-reports` accepts one exact
`forge/distributed-evaluation-report/v1` JSON object, up to 4 MiB.  The first
report for `(plan_hash, validator_hotkey)` is immutable.  Retrying identical
bytes returns `200` with `duplicate`; a conflicting report returns `409`.
SQLite WAL makes accepted reports durable across restarts.
It rejects transfer encodings, duplicate content lengths, incomplete bodies and
non-finite JSON.  Responses are `no-store`, close the connection, and never
include filesystem paths in errors.

`GET /api/v1/validator-reports` returns only the public report index:
round, plan, validator hotkey, content hash, work counts and verification
limits.  It never returns signatures, prompt text, evidence rows, paths,
provider data or private infrastructure.  `GET /data/status.json` can serve a
read-only non-fixture `forge/dashboard/v2` exporter artifact; it cannot modify
that snapshot.

Validators call `forge dashboard-push --plan PLAN --report REPORT --endpoint
URL --outbox OUTBOX`.  The command writes canonical report bytes to `OUTBOX`
before networking.  If delivery fails, rerun the identical command: it reuses
the same bytes and the server's immutable idempotency rule makes the retry safe.
The endpoint must be HTTPS except for a loopback development origin.

For unattended retry, run `forge dashboard-flush --plan-directory PLANS
--outbox-directory OUTBOX --endpoint URL` from the validator's timer. The dedicated
outbox directory contains only already-signed report JSON files. A pass stops on
an API delivery failure, retains every payload, and never needs a wallet or
repeats inference. Repeating a successful pass returns duplicate acknowledgements.

The `distributed-report` command can deliver automatically after saving its signed
report: add `--dashboard-endpoint https://dashboard.attelier.ai` and
`--dashboard-outbox /path/to/immutable-outbox.json`. A failed delivery returns
`PENDING` in its delivery field while report creation still succeeds, so dashboard
availability cannot block chain consensus. Keep the local report and retry
`dashboard-push`, without signing a
new result or repeating inference. The client identifies itself as
`forge-validator/0.1`, and the public endpoint has been exercised with real signed
reports, exact retries and a server restart.

The API accepts reports only for plans in the server's registered plan set.
Use `--plan-directory /etc/forge/dashboard-plans` to discover operator-installed
frozen plans without a restart. Install each JSON through an atomic rename; only
that private directory can register plans, never an HTTP client. Accepted plans
are retained in SQLite so retiring a registry file does not erase report history
after a restart. This does not discover or authorize chain rounds automatically. The certified snapshot is still produced by the
resolver/exporter, separately from the receipt feed.

For bootstrap diagnostic snapshots, `dashboard-export-v2 --raw-evidence DIR`
adds native MUSIQ point deltas after checking the signed reports, raw evidence
hashes and complete assigned coverage, including audit replicas. The mean of
these native points is independent of the mean across the normalized judge axes;
it is not a perceptual percentage or a calibrated preservation-of-quality claim.

Deployment keeps the published HTML byte-identical to the reviewed build. The
dashboard virtual host sends `Cache-Control: public, no-transform` to prevent
automatic edge injection, as documented in [Cloudflare Web Analytics](https://developers.cloudflare.com/web-analytics/get-started/).
Keep `/api/v1/validator-reports` uncached and proxy it to the loopback service;
serve static media only by the verified hashes in the exported snapshot.

A non-baseline king requires explicit `--king-state` from the finalized-state
workflow. A plan's comparator is not a crown; the exporter refuses to infer one.
The exporter projects the supplied state and does not authenticate its chain
origin itself. Keep that input under the resolver's operational trust boundary.
`PROMOTE` in a resolved evaluation can denote `PROMOTE_ELIGIBLE`; it alone is not
a coronation. The page requires a matching king manifest and crown before saying
that the candidate became king. Diagnostic measurements remain uncalibrated.

## Provisional model launch V1 fields

`model.files` optionally exposes the admitted manifest's files as
`{path, sha256, size_bytes, url}` with anonymous immutable HF `resolve/<revision>`
links. The model also carries `backend`, `manifest_hash`, `license`, `role` and optional
`base: {repo, revision, url}` for a LoRA's required base. No private evidence
bucket or local artifact path is part of this object.

`evaluations[].open_quality` is optional and uses its own scale:
`{evaluator: "VideoReward", policy_id, evaluator_id, units: "normalized_reward",
status, provisional: true, quality, speed, temporal_violations}`.
`quality.king` and `quality.anchor` each contain VQ/MQ/TA objects with
`mean_delta`, `lower_ci`, `margin`; `speed` contains `ratio` and `lower_ci`.
An INCOMPLETE result may lack numerical summaries. Legacy quality fields remain
null in this mode; scores are never clamped to [-1,1], averaged across heads or
shown as perceptual percentages. `PROMOTE_ELIGIBLE` (V1), or `PASS` with an exact
`forge/launch-policy/v2` in `contract.launch_v1`, is only a local policy result;
the existing certified state remains authoritative for the king and rewards.
With a known policy and all three validated head margins, the legacy
`contract.quality_margin` may be null: it must not substitute a single averaged
margin for VQ/MQ/TA. V2 uses 32 quality prompts, 8 timing prompts and 4 diversity
prompts with 3 seeds each. A qualified evaluator and provisional competitive
margins are separate facts.

An operator experiment is explicit: its round has `origin: "operator_experiment"`,
`diagnostic: true`, an actual ISO `created_at`, and null `close_block`,
`entropy_block` and `commit_close_block`. Its evaluation verdict is always
`DIAGNOSTIC`; a completed `open_quality` decision is still visible separately.
Real model submission blocks do not turn an operator-generated panel into a
chain-derived competition round. The exporter preserves the certified king and
crown history from its input snapshot. `contract.launch_v1` carries the exact
frozen provisional policy so the UI can explain its rules without reusing the
legacy quality margin. Hippius metadata may have null public browser URLs;
the UI never substitutes an unverified download link.

A certified V1 snapshot is produced by `launch_certified_dashboard` only after
certificate replay, reconstructed weight intents and finalized readbacks. The
round uses `origin: "testnet_chain"`, actual close/entropy/commit blocks and
`diagnostic: false`. `shared_execution: true` and `independent_operators: false`
identify the controlled shared-worker path. Each evaluation carries
`execution: {shared, independent_operators, validator_count}` and
`certification: {status, transition, certificate_hash, certified_state_hash,
activation: {status, finalized_block, runtime_hash, validator_count}}`.
`launch_v1_certification` exposes the same summary with its `round_id`.
The finalized activation status is `FINALIZED_WEIGHTS_VERIFIED`; a certificate
alone never means active weights. Valid consensus reports carry the signed
report hash, their actual commit anchor block and the prompts replayed. Their
presence does not imply independent GPU generation. No signatures, salts or
private receipt locations are exported.
