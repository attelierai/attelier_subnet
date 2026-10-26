# Public artifact admission

Every admitted candidate is an immutable public HF SHA or Hippius OCI digest.
The base is the pinned Wan pipeline. A candidate is either one standard LoRA
adapter or a complete transformer with the same Wan architecture.

HF must expose its SPDX-compatible license in public model metadata. Hippius
must expose a small public `LICENSE`, `LICENSE.txt`, `LICENSE.md`, or `COPYING`
file identifying an allowed reusable license through an SPDX line or standard
license text. This is an admission signal, not a legal-rights determination.
Public snapshot access does not verify ownership, redistribution rights, downstream
obligations or licence compatibility beyond that signal. Admission rechecks current
access and all declared bytes immediately before use. It does not prove that a
provider will retain an artifact after a win; winner retention needs a separate
public mirror/archive policy.

The public requirement applies to every admitted candidate, not only the winner.
Another miner can download a candidate or certified winner, continue training it,
and publish a new immutable revision for a later round. The challenger must still
pass fresh evaluation against the incumbent and the fixed original reference.
The current runtime accepts one adapter over the pinned base or a full compatible
transformer, not an adapter stack. To continue from a LoRA winner with another
adapter, merge the resulting weights and submit the full transformer. The pinned
`base_reference` identifies runtime compatibility and fixed text-encoder/VAE
components; it is not a verified claim about the checkpoint's training ancestry.

The manifest recipe permits `steps`, `guidance_scale`, and an optional `scheduler`
selected from `uniform_euler` or `original_unipc`. The validator owns both
implementations; miners cannot supply scheduler code or arbitrary configuration.
Euler shift belongs to the global challenge contract; UniPC configuration comes
from the pinned original pipeline. Every numeric value is finite. Validators fetch only declared regular
files and validate hashes before loading any model code.

HF recipe commitments with an explicit scheduler use
`forge:v4:h:REPO:REVISION:ROLE:STEPS:GUIDANCE:SCHEDULER`, where the final token is
`e` for Euler or `u` for UniPC and the role is `l` (LoRA) or `t` (full transformer).
The entire commitment must fit the 128-byte chain field. Legacy v3 commitments
retain their original implicit scheduler behavior. Scheduler selection participates
in candidate identity and cannot change after admission. A recipe-only candidate
can reference the original public HF revision as a full transformer: validators
select its transformer files and keep the pinned text encoder and VAE.
Candidate JSON/TXT configuration files are capped at 2 MiB before downloading,
including HF LFS entries. The pipeline-only `tokenizer/tokenizer.json` allowance
is bounded at 32 MiB to support the pinned Wan tokenizer. Hippius fetches also require one exact per-file OCI metadata
entry with the declared size and SHA-256 before calling the SDK downloader;
verification of the downloaded bytes remains mandatory. The isolated bridge
keeps SDK progress on stderr and reserves stdout for its JSON protocol.

The same metadata boundary limits the pinned Wan 1.3B layouts before any weight
download: a pipeline is at most 32 GiB total and 6 GiB per file; a complete
transformer is at most 6 GiB total and per file; a rank-128 LoRA is at most 1 GiB.
The pinned pipeline is 28,928,887,859 bytes, while the inspected public complete
transformers are 5,676,070,784 and 5,959,377,056 bytes. These are capacity limits,
not a claim that every artifact below them is safe or compatible.

The 1 GiB LoRA limit is derived from this exact pinned T2V architecture, not from
the observed rank-16 adapter. It has 30 blocks at width 1,536 and FFN width 8,960.
At rank 128, adapting every permitted two-dimensional weight consumes
`30 * (8 * 128 * (1,536 + 1,536) + 2 * 128 * (1,536 + 8,960)) + 3,317,760`
= 178,298,880 F32 values = 713,195,520 bytes. The final term covers all six
outside-block linear weights; optional scalar alphas for the 306 targets and the
maximum 16 MiB safetensors header keep the file below 729,975,184 bytes even if
those scalar alphas use the permitted F64 encoding. Conv3D,
norms, biases and tables are not LoRA targets under the validator's two-dimensional
weight rule.

HF downloads have a 300-second monotonic deadline per file, checked as data arrives,
and each blocking read is capped at 60 seconds. A silent peer can therefore delay a
failure by at most one final read after the monotonic deadline. Hippius retains its
isolated bridge's 300-second subprocess timeout.

`forge preflight-round` first performs a conservative cache-capacity check for the
whole closed batch, before downloading the base or any candidate. It counts each
missing declared path once per immutable cache reference, reserves those final bytes,
adds the largest missing file once for the temporary HF download, and retains a fixed
2 GiB filesystem floor. The maximum cold batch therefore requires 136 GiB free:
32 GiB base + 16 × 6 GiB full-transformer candidates + one 6 GiB temporary file +
2 GiB floor. This is a preflight observation from `statvfs`, not an atomic reservation;
other writers, filesystem quotas, or provider-side capacity can still make a later
fetch fail. The command fails before any artifact HTTP request when that capacity is
not presently available, and does not arbitrarily omit a closed admission.

Before a candidate reaches Diffusers, `forge.models.weights.validate_candidate_weights`
parses safetensors directly. It rejects malformed headers, overlapping ranges,
undeclared bytes, non-finite floats, unsupported LoRA keys, aliases resolving to
the same key, unpaired A/B matrices, rank mismatches, and full-transformer
shape/dtype/config changes against the pinned base. Full-transformer configs may
differ only in `_class_name`, `_diffusers_version`, and `_name_or_path`.

The tensor-content fingerprint hashes canonical tensor names, dtype, shape, and
content hash. It catches repackaging and standard prefix aliases such as
`base_model.model.transformer.*` versus `transformer.*`. It is exact-tensor
identity, not a mathematical equivalence proof: scaled LoRA factorizations
(`A*c`, `B/c`) can produce the same update with different tensors. Promotion
must retain the runtime's base-plus-adapter evaluation and cannot treat this fingerprint
alone as a proof of model equivalence.

LoRA admission also binds executable scaling. `adapter_config.json` and embedded
Diffusers adapter metadata must agree; alpha/rank is preserved, and nondefault
scaling changes the tensor-content fingerprint. Unsupported scaling variants or
additional trainable modules are rejected. Runtime loading canonicalizes admitted
PEFT/Diffusers names and verifies every loaded A/B matrix and its scale. A loader
returning without an exception is insufficient: Diffusers can otherwise silently
ignore a checkpoint whose prefix does not match its pipeline filter.
