# Attelier API Miner

This is the API-provider worker for Attelier SN126. It is separate from an H3
GPU worker, does not submit Bittensor weights, and does not promise an alpha
reward. An operator must review the hotkey, provider account, models, offer,
and Phala deployment before the gateway can route work here.

## Security boundary

Deploy only the reviewed image/configuration in a Phala Intel TDX environment.
Before routing paid work, the Attelier gateway verifies Phala attestation,
measurement, a fresh session challenge, the persistent control public key, and
fresh runtime signing/encryption public keys bound into that quote. This binary
does not interpret an environment variable as TDX proof.

The runtime-signed receipt binds the assignment, one-use grant, upstream request
reference, upstream response hash and output bytes. It demonstrates that the
approved worker handled that exchange. It does not prove the provider's internal
model execution, invoice, service availability, or commercial permission.

## Operator quick start

1. Register a hotkey and request review for the provider account, model IDs and
   Phala deployment. Possessing an API key does not grant admission.
2. Create an owner-only directory, make it writable by UID `10001` (the
   reviewed non-root image user), and set `ATTELIER_MINER_STATE_DIR` to it.
   The deployed CVM must also permit that user to access the read-only dstack
   socket; `doctor` can check configuration but cannot manufacture TDX access.
3. Encrypt only keys that the miner is authorized to use in the Phala deployment:
   `BYTEPLUS_API_KEY`, `KLING_ACCESS_KEY` plus `KLING_SECRET_KEY`,
   `GOOGLE_CLOUD_ACCESS_TOKEN`, `FAL_KEY`, or `ATLASCLOUD_API_KEY`. Kling V2.6 uses its legacy
   AccessKey/SecretKey flow: the worker creates a short-lived HS256 JWT inside
   the CVM (cached only until shortly before its 30-minute expiry); do not
   replace that pair with a browser/API token.
   Never put a key in Git, the image, logs, or a coordinator request. Google
   Cloud uses a short-lived Cloud access token plus `GOOGLE_CLOUD_PROJECT`,
   `GOOGLE_CLOUD_LOCATION`, and an owner-controlled
   `GOOGLE_CLOUD_OUTPUT_BUCKET`; Gemini preview keys are not a Veo production
   credential. The worker writes only under its `attelier-api-miner/` prefix
   and reads output back using that same Cloud token. Rotate the sealed Cloud token before its expiry,
   then restart and attest the reviewed deployment; a missing or expired token
   makes that provider unavailable and never falls back to another account.
   V1 does not refresh Google access tokens: unattended Veo mining needs a
   reviewed token-refresh sidecar/identity design before it can be relied on.
   fal's current Veo 3.1 Fast endpoint is marked for deprecation on
   2026-11-17. Do not enroll it for a production miner without rechecking the
   reviewed catalogue and replacing it with a supported endpoint if required.
4. Run `cargo run --manifest-path workers/api_miner/Cargo.toml -- doctor`, then
   `cargo run --manifest-path workers/api_miner/Cargo.toml -- identity`.
   It creates the one persistent owner-only **control** signing key and prints
   its public SPKI only. Give that control SPKI, plus worker/miner IDs, to
   Attelier review before deploy. The command never prints a private key. The
   runtime Ed25519 receipt key and X25519 assignment key are generated only in
   memory for each worker process; they must never be registered from a local
   development machine or copied into state.
5. Pin the worker image digest in `docker-compose.phala.yml` and deploy its
   rendered compose through Phala. It mounts the dstack socket plus the fixed
   Docker named volume `attelier-api-miner-state` at
   `/data/attelier-api-miner`; the root-only `state-init` service gives that
   volume to the non-root worker before it starts. dstack persists Docker data
   on its encrypted data disk, whereas an arbitrary guest path such as
   `/var/lib/attelier-api-miner` is not a durable identity store across a CVM
   redeploy. Do not replace the named volume with a host-path bind or run the
   compose unencrypted on an ordinary host.
6. Production Phala CVMs may not expose SSH. The measured bootstrap can run
   `attelier-api-miner evidence` once with a fresh, public
   `ATTELIER_ENROLLMENT_CHALLENGE`, writing its single JSON line to a protected
   temporary file. It emits `attelier/api-miner-enrollment/v2` with the
   persistent control SPKI, fresh runtime signing/encryption SPKIs, their exact
   V2 report-data binding, raw quote, original event log and only dstack's
   `vm_config` object; it never
   prints provider keys, private keys, environment values or durable state.
   Send that file only to Attelier's scoped private enrollment collector, then
   remove it. The collector and its pinned DCAP verifier must independently
   validate the quote and policy. The VM config is observational input for
   reproducing the pinned OS measurement, never a replacement for quote
   verification or an admission decision.
7. Run `attelier-api-miner run` (or `run-once`). On every process start it
   creates fresh in-memory runtime keys and performs a fresh remote attestation
   before it can claim work. It renews that attestation with the same runtime
   keys before expiry. `attest` is a diagnostic command only: its one-off
   runtime keys are discarded, so running it separately does not pre-authorize
   a later `run` process.

`pause`, `resume`, and `status` are safe lifecycle commands. The durable state
marks a `(job, attempt)` accepted before an upstream request. A restart or lost
response is reconciled from that exact identifier; it never causes an automatic
second paid generation. `pause` writes an independent control flag rather than
rewriting the live durable ledger, so it can safely stop a running loop. Only
one `run`, `run-once`, `attest`, or `evidence` process can hold `worker.lock`; after an
unclean host loss, `flock` is released by the OS even though the lock file may
remain. Inspect the durable state and ensure no worker process holds the lock;
do not delete a harmless lock file merely because it exists. Never run a second
worker against the same state directory.

## Reviewed model catalog & staged rollout

The single audited worker binary supports all 5 upstream provider families (`byteplus`, `fal`, `atlascloud`, `kling`, `google-cloud`) across both video (`text-to-video`, `image-to-video`) and image (`text-to-image`, `image-to-image`) modalities so enabling new model families later requires no CVM image rebuild or re-attestation:

| Family | Providers | Reviewed Models | Modalities | Rollout |
|---|---|---|---|---|
| **Seedance** | `byteplus`, `fal` | Seedance 2.5, 2.0, 2.0 Fast, 2.0 Mini | `text-to-video`, `image-to-video` | **Wave 1 (Launch)** |
| **Kling** | `atlascloud`, `fal`, `kling` | Kling 4.0 (Video & Image), Kling 3 Turbo, Kling 2.6 | `text-to-video`, `image-to-video`, `text-to-image`, `image-to-image` | Staged |
| **Veo 3.1** | `google-cloud`, `fal` | Veo 3.1, Veo 3.1 Fast | `text-to-video`, `image-to-video` | Staged |
| **MiniMax H3** | `fal`, `atlascloud` | MiniMax H3 Max, MiniMax H3 Developer | `text-to-video` | Staged |
| **Wan 3.0** | `fal` | Wan 3.0 | `text-to-video` | Staged |
| **FLUX 3 Max** | `fal` | FLUX 3 Max Video | `text-to-video`, `image-to-video` | Staged |
| **Nano Banana** | `fal` | Nano Banana Pro & 2.1 (Create & Edit) | `text-to-image`, `image-to-image` | Staged |
| **FLUX 2 Flash** | `fal` | FLUX 2 Flash (Create & Edit) | `text-to-image`, `image-to-image` | Staged |
| **Seedream 5 Lite** | `fal` | Seedream 5 Lite (Create & Edit) | `text-to-image`, `image-to-image` | Staged |

Production starts with **Seedance only** via the gateway's `enabledModels` allowlist in the API miner config (e.g. `["dreamina-seedance-2-5-260628", "bytedance/seedance-2.5/text-to-video"]`). Additional families are promoted incrementally by adding their upstream or canonical model IDs to `enabledModels`.

## Gateway protocol

### Advertise an API capability

Set `ATTELIER_MINER_OFFERS` (and optionally `ATTELIER_MAX_CONCURRENCY`, default `1`)
in the sealed deployment. All prices are integer millionths of a US dollar
(`1,000,000` = `$1.00`). Each offer supports three pricing fields with strict
precedence:

1. `quotesByDuration`: exact total price per duration in seconds (e.g. `{"5": 95000, "10": 180000}`).
2. `quoteMicrousdPerSecond`: per-second rate multiplied by requested duration (e.g. `20000` = `$0.02/s`).
3. `quoteMicrousd`: flat total per accepted request for any supported duration (e.g. `500000` = `$0.50`).

```json
[
  {
    "model": "dreamina-seedance-2-5-260628",
    "modality": "text-to-video",
    "quoteMicrousdPerSecond": 20000,
    "quotesByDuration": { "5": 95000, "10": 180000 }
  }
]
```

The worker takes the provider from `ATTELIER_PROVIDER`, signs inside its live
attested session and refreshes offers every two minutes while polling. Each
offer expires within five minutes and by the end of that session. To change the
price or concurrency, update the sealed config and restart the worker with its
existing state volume. An empty list publishes no offers. A price change applies
to future quotes. Note that on Phala `dstack-0.6.5`, keep `--pre-launch-script phala-pre-launch.sh`
so non-root UID `10001` can access the read-only dstack socket.

Assignments are received only as an AES-GCM envelope for the session's X25519
key. Its plaintext is `attelier/api-miner/v1`, bound to a worker/miner,
session, provider/model, canonical `requestHash`, one-use nonce, bid and
deadline. Workers reject generic URLs, model substitutions, expired grants and
providers outside the reviewed catalog.

Receipts are `attelier/api-miner/v1`, signed by the fresh quote-bound runtime
key. They
bind `jobId`, `attemptId`, `sessionId`, `requestHash`, provider/model, provider
reference and response hash, and output hash/size. The gateway's verifier-owned
session, rather than a worker-declared attestation marker, is authoritative.
The coordinator owns the provider-specific
assignment and reconciliation contract. The worker itself polls the original
upstream reference and downloads accepted output from a provider allowlist. It
must never retry an uncertain accepted attempt as a new paid submission.
Raw media is capped at 100 MiB before the signed base64 completion payload, in
lockstep with the gateway body limit; oversized output is treated as unknown
and reconciled, never retried as a new generation.

`run-once` performs one full pull sequence: claim → durable grant → upstream
submission → gateway start with the original provider reference → upstream poll
and allowlisted download → signed receipt plus bytes. It reports `unknown` to
the gateway whenever a paid submission or result is ambiguous. The production loop
`run` is the compose default: it performs the same sequence repeatedly with a
two-second idle backoff, renews an attestation within 60 seconds of expiry, and
honours `pause` without resubmitting any accepted provider request. `run-once`
remains the diagnostic/reconciliation command. If a provider call outlives the
current session, the next loop first renews with the same in-memory runtime key
and retries only the idempotent gateway action; it never submits the provider
request again.

A definitive provider rejection (HTTP 400, 401, 403, 404 or 422 before a
provider reference) or an explicit upstream `failed`/`cancelled` state is sent
as signed `fail`, then durably closed as `provider_failed` after the gateway
acknowledges it. The only optional diagnostic is the provider's constrained
`error.code`; responses, messages and credentials are never forwarded. Network
errors, 5xx, deadlines and output-download errors remain `unknown` because
their billed state is not proven.

## Secret rotation and recovery

Create the Phala application from the rendered compose in its console or its
current official CLI, using that interface's encrypted environment/secret
feature. Do not use shell `-e KEY=value` flags, paste credentials into a
terminal history, or submit a `.env` file to Git. The only plaintext input is
the local operator prompt provided to Phala; verify the resulting application
measurement and image digest before registering its persistent control identity
with Attelier.

For an API-key rotation, pause the worker, update the sealed variable in Phala,
restart the reviewed image, then use `status` and `run-once`; the new process
attests itself automatically. Do not delete the state volume: it records a
provider reference after paid acceptance. On a new process, durable state is
only an untrusted hint: the worker calls gateway `recover` with its job/attempt
grant tuple and accepts only a gateway-canonical provider reference, response
hash and freshly encrypted assignment. It then polls that original reference
and never submits another generation. Local `assigned`, `submitting`,
missing-reference, or non-canonical rows are left for gateway
expiry/reconciliation, never promoted to a first start from disk.

## Phala CLI runbook

This is the current Phala Cloud CLI flow, pinned to `phala@1.1.22`. It is a
runbook only: these commands can create or update a CVM, so run them only after
the miner/operator admission is approved.

```sh
npm install -g phala@1.1.22
phala login
phala instance-types
test ! -e .phala.env && (umask 077; : > .phala.env)
# Edit .phala.env locally with only the reviewed variables from the compose.
# Never put its values in shell flags, Git, or command history.
phala deploy -n attelier-api-miner-REVIEWED_ID \
  -c docker-compose.phala.yml -e .phala.env \
  --pre-launch-script phala-pre-launch.sh --kms phala \
  --no-dev-os --no-public-logs --no-public-sysinfo --no-listed --wait
phala link
phala ps
phala cvms attestation --json > attestation.json
```

If selecting a region, use the exact case-sensitive identifier returned by
`phala cvms list-nodes` (for example, `US-WEST-1`). Do not use legacy
`--node-id` selection: current Phala Cloud rejects node/teepod IDs for
placement, and a lowercase region value can resolve to no Teepod.

`-e .phala.env` seals the file's environment to the deployed compose
measurement; the CLI stores no plaintext in `phala.toml`. Before gateway
admission, compare the returned compose/image measurement and the image digest
with the reviewed release, then submit the *worker-generated* dstack quote to
the pinned Attelier verifier. Do not treat `phala cvms attestation` alone as
gateway admission.

`--no-public-logs` intentionally makes `phala logs -f` unavailable. Do not
turn on public logs or `--dev-os` to inspect a production CVM. Use `phala ps`
for container state and the one-shot private enrollment bootstrap below for the
worker's quote. It uploads evidence to the scoped collector; a successful
upload remains non-admission until independent verification completes.

For a non-SSH production bootstrap, copy
`docker-compose.phala.enroll.example.yml` to an ignored local override:

```yaml
services:
  api-miner:
    entrypoint: ["/usr/local/bin/attelier-api-miner-enroll"]
    command: []
    environment:
      ATTELIER_ENROLLMENT_CHALLENGE: ${ATTELIER_ENROLLMENT_CHALLENGE:?fresh public challenge}
      ATTELIER_ENROLLMENT_TOKEN: ${ATTELIER_ENROLLMENT_TOKEN:?sealed scoped collector token}
```

Those two values are sealed variables in the Phala environment file, never
shell arguments or tracked compose content. The bootstrap runs the measured
worker's `evidence` command, posts its public envelope over HTTPS using the
scoped token, then starts the usual worker loop. It does not create an HTTP
listener or enable logs.

For a release or secret update, edit the owner-only local `.phala.env` or the
pinned image reference, then re-run `phala deploy -e .phala.env --wait` from
the linked directory (so the linked compose/env paths are used), `phala cvms
attestation --json`, and `phala ps` from the linked directory. Keep
`--pre-launch-script phala-pre-launch.sh`, `--no-dev-os`, and the private
visibility flags on every update.
The sealed variables are re-encrypted on each deploy. Keep the named worker
state volume: deleting it discards the persistent control identity and pending
paid-request reconciliation. A new CVM also has a new encrypted disk and
therefore needs a new enrollment; an update must target the existing CVM rather
than create one.

The compose `state-init` service prepares the dedicated non-root named volume;
the measured pre-launch script only handles optional private-registry login and
socket permissions. For a private image registry, seal
`DSTACK_DOCKER_REGISTRY`, `DSTACK_DOCKER_USERNAME` and
`DSTACK_DOCKER_PASSWORD` with the worker variables; the script feeds the password
to Docker through stdin. Pin the image by digest. Confirm socket permissions
and the real worker attestation after boot; a successful provisioning response
does not prove that the container started. Updates must target the existing
`--cvm-id`; if Phala reports another operation in progress, reconcile that
operation instead of creating a second paid instance.

## Verification

```sh
cargo test --manifest-path workers/api_miner/Cargo.toml
cargo run --manifest-path workers/api_miner/Cargo.toml -- doctor
```

Tests make no provider or Phala request. They cover canonical request binding,
receipt tampering, durable replay prevention, and paused-worker rejection. A
real Phala attestation qualification remains an operational admission step.

The cross-repository encrypted lifecycle test is deliberately opt-in: it
starts the separately built local Node gateway fixture and has no production
URL, credential, or absolute home-directory dependency. The fixture finds
`ffmpeg` and `ffprobe` through its environment or `PATH`.

```sh
ATTELIER_API_MINER_NODE_RUNNER=/absolute/path/to/api-miner-node-runner.mjs \
  cargo test --manifest-path workers/api_miner/Cargo.toml \
  rust_worker_completes_real_node_gateway_encrypted_lifecycle -- --ignored --nocapture
```
