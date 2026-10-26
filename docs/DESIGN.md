# Design v0.2 — public Wan acceleration MVP

Historical v0.2 design, retained for protocol/evidence context. The agreed next
target is [MODEL_COMPETITION.md](MODEL_COMPETITION.md); the current executable MVP
is documented in [WAN_MVP.md](WAN_MVP.md). Do not infer implementation or launch
readiness from this older design. Anything that reaches the chain or a public
artifact uses the codename **`forge`**; never the product name.

## Roles

- **Miner:** trains offline (LoRA, fine-tune, distillation) and submits weights.
  Miners never serve inference.
- **Validator:** admits artifacts, generates videos with the approved runtime,
  evaluates them, commits and reveals its report, verifies certificates and
  sets weights derived only from certified state.
- **Serving:** consumes certified winners later. Out of scope here.

The implemented GPU/storage workflow and remaining activation gates are in
[`WAN_MVP.md`](WAN_MVP.md); public artifact and panel rules live in
[`ARTIFACTS.md`](ARTIFACTS.md) and [`PROMPT_PACKS.md`](PROMPT_PACKS.md). The sections
below distinguish the target protocol from the legacy CPU simulation. A new contract
hash requires a fresh challenge; never reuse historical simulation state as a
real-model certificate. The repository currently has a local signed evaluation
record and testnet anchoring path, not a quorum, crown or weight-setting path.

## Challenge contract (one per challenge, hashed and versioned)

- Base: Wan2.1-T2V-1.3B, pinned revision and file hashes.
- Task: text to video, 832x480, 81 frames, 16 FPS, no audio.
- Frozen: public pinned text encoder, VAE, tokenizer and runtime. Candidates are
  safetensors LoRA or full transformers; generation steps and CFG are committed.
- Candidate limit: 16 denoising steps, 32 actual transformer forwards, 900 seconds,
  24 GiB peak allocated VRAM. The fixed quality anchor may use 50 steps/100 forwards.
- Admission binds immutable snapshots, declared file hashes and generation recipes.
  Preflight computes canonical effective-tensor identity to reject duplicate candidates;
  the future real-model consensus bridge must bind that identity to its round plan.

## Benchmark: public contract, sealed production panels

- The contract, rubrics, metrics and weights are public.
- A production panel is selected from a new `forge/sealed-pack/v1` whose hash is
  committed by the close and whose content is revealed after its finalized entropy.
  Each candidate in the cohort receives the same selected prompt/seed slots. The
  exact commitment, reveal and selection checks are authoritative in
  [`PROMPT_PACKS.md`](PROMPT_PACKS.md).
- Procedural grammars v1--v3 remain only to replay historical receipts. They are
  not a production fallback and do not claim secret or out-of-distribution coverage.

## Evaluation

- Axes (v1): prompt adherence, motion adherence, camera adherence, subject
  consistency, temporal stability, motion quality, visual quality, physical
  plausibility. Each axis maps to [0,1] with fixed, versioned normalizers
  (never normalized against the current cohort).
- The implemented automatic path uses blinded Luna vision comparisons in both
  A/B orders, full-frame sheets and native detail frames. It requires synthetic
  self/copy, frozen, blurred, shuffled and wrong-video controls; no human labeling.
  These controls test specified defects, not universal perceptual quality. Judge
  uncertainty blocks promotion. Historical v1/v2 policies remain replay-only.
  Production v3 uses `symmetric_mean_prompt_clustered`: it averages each blinded
  orientation per axis and clusters seeds within a prompt; orientation disagreement
  and sign flips are explicit diagnostics. Fresh machine-known controls still gate
  promotion. See `WAN_MVP.md` for the exact policy, calibration conditions and
  limitations of image-sequence evaluation.
- The automatic gate protects each quality axis globally and per subject class,
  plus aggregate camera/style slices. It does not collapse quality to one mean.
  The legacy simulation uses `0.6 * mean(axes) + 0.4 * mean(worst 3 axes)`;
  that scalar is not the real-model automatic promotion rule.
- Model failures (NaN, OOM, timeout, wrong format) count as score 0. Infra or
  judge failures make the comparison incomplete; they are never a miner loss.

## Duel: king vs challenger

1. Screen: 32 prompts x 1 seed.
2. Duel: 256 prompts x 2 seeds.
3. Confirmation: 4,096 prompts x 2 seeds, selected from a later sealed-pack
   reveal only after the finalist is locked. The history verifier rejects normalized
   text/source-group reuse against a supplied checkpoint and revealed-plan chain.
   Authenticating that checkpoint from certified state remains unimplemented;
   production fails closed without it. See `PROMPT_PACKS.md` for the trust boundary.

The real GPU path targets speed with quality noninferiority: paired, prompt-clustered
speed lower bound >=1.1, protected quality axes no worse than their fixed margin,
against both incumbent and original golden. A trusted prescribed panel and
calibrated evaluator are required; missing evidence never promotes. The legacy
simulation still tests quality-superiority decisions for the state-machine core.
At most one promotion per challenge per round; ties go to earliest commitment,
then artifact id. Confirmation uses fresh post-finalist entropy.

## Pre-consensus: `current_king` is shared state

- Round plan (fixed before evaluation): parent state hash, king and challenger
  fingerprints, contract hash, prompt seeds, committee (validator hotkeys and
  voting power from a finalized-block snapshot), commit, reveal and activation
  blocks.
- Each validator evaluates locally and publishes
  `commit = H(canonical(network, netuid, round_id, validator_hotkey,
  parent_state_hash, contract_hash, report_hash, salt))` before the commit
  deadline, then reveals report and salt before the reveal deadline.
- The decision is a quorum of verdicts backed by valid reveals, not a new
  average of scores. Promote only if `3 * favorable_power > 2 * total_power`,
  where the denominator is the whole committee snapshot, not the responders.
- A v2 `RoundCertificate` names one transition (PROMOTE or KEEP) from one parent
  state, includes the exact canonical digest of its commit/reveal set, and signs
  that digest with enough validators. Verifiers recompute the decision from the
  included evidence; a declared transition alone never suffices. This still
  expresses consensus over local reports, not proof that model inference or a
  judge was independently reproduced. Any node, including a dissenter, applies
  a valid certificate and keeps its own report unchanged.
- The state hash depends on the transition, never on which signer subset a node
  received. Applying a certificate is transactional and idempotent.
- No quorum means no promotion. Missing data means "pending sync", never a
  negative vote. One pending comparison per challenge.
- A node returns KEEP only when promotion is impossible even if every reveal it
  lacks were favorable; otherwise it waits. Certificates do not trust a local
  boolean claim that the reveal window closed: missing or incomplete evidence
  remains pending until an independently verified closure transport exists.
- Screen and duel run once, by the round coordinator, on the shared panels; the
  finalist it locks goes into the round plan. Every committee member evaluates
  only the confirmation panel. Validators sign the transition they computed.

## Chain mapping (SDK 11.0.1)

- Public model submissions use a compact, versioned commitment to an immutable HF
  SHA or Hippius OCI digest. The pinned snapshot contains `forge-manifest.json`.
  Admission reads commitments at the finalized close block; the precise artifact
  format and its limits are authoritative in [`ARTIFACTS.md`](ARTIFACTS.md).
- Report commits: validator hotkey commitment `forge:v1:rep:<round>:<hash>`,
  read as of the commit-deadline block (commitments keep only the latest value
  per hotkey, so historical reads are required).
- Reveals, plans and certificates are off-chain documents (write-once board).
- Committee: hotkeys with a validator permit and stake in the metagraph at the
  close block; voting power is total stake in rao.
- Entropy: the sealed-pack selection uses hashes finalized after the close and,
  for confirmation, after the public finalist lock.
- Limitation: SDK reads expose the commitment block, not the extrinsic index,
  so ties inside one block fall back to the artifact id.

## Rewards

- The king is the reference; a crown is a historical event; only crowns that
  are eligible get paid.
- Winner takes all: 100% to the latest certified crown's eligible hotkey. No
  decay, expiry or historical sharing. If that winner is ineligible, allocation
  remains unassigned; never resurrect an older winner. The baseline earns no crown.
- Weights resolve per hotkey first, then map to UIDs from a finalized snapshot.
  Unallocated share is returned explicitly; nothing is assumed about UID 0.
- Weights are computed only from certified state. Never from a local verdict.

## Non-goals for v0.1

No commercial serving by miners, no custom kernels or miner code execution, no
multi-GPU miner requirements, no mainnet until explicitly authorized.
