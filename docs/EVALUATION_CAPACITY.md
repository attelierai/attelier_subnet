# Evaluation capacity

`forge distributed-capacity` is an offline admission calculation for one frozen
distributed plan. It does not rent a GPU, alter a queue, or make a candidate
eligible. Give it measured full-pipeline seconds for the exact GPU and recipes,
the current hourly price, and explicit time and remaining-budget limits.

```bash
.venv/bin/python -m forge.cli distributed-capacity \
  --plan /path/to/frozen-confirmation-plan.json \
  --king-seconds 278.14 --wan50-seconds 278.14 \
  --candidate-seconds 225.28 --gpu-usd-hour 0.70 \
  --node-overhead-seconds 600 --node-overhead-usd 0 --metric-seconds-per-video 4 \
  --judge-usd-per-quality-comparison 0.01 --storage-usd-per-video 0.001 \
  --validator-candidate-budget-usd 10 --validator-screening-spent-usd 0 \
  --network-remaining-usd 100 --commit-delivery-seconds 60 \
  --reveal-delivery-seconds 30 --chain-block-seconds 12 \
  --out /path/to/capacity.json
```

The receipt is only a projection. `ADMISSIBLE_PROJECTION` says a complete
frozen panel fits the supplied estimates and limits; it does not authorize a
GPU rental, a budget charge, a promotion, or weights. Every receipt has
`dispatch_authorized: false` and `promotion_possible: false`.

The CLI defaults to the conservative 10 USD candidate cap per validator. It
does not assume a two-hour job: it calculates the required commit window from
the slowest actual allocation, measured GPU/metric/overhead time and explicit
report-delivery reserve. Supply `--max-node-hours` only when a separate
operational cap has been approved. The screening spend and commit/reveal
delivery reserves have no defaults: pass reconciled or measured values
explicitly, even when they are 0; otherwise the receipt is incomplete.

`--validator-candidate-budget-usd` is a per-validator ceiling, and
`--validator-screening-spent-usd` is the already reconciled screening cost on
that validator; the planner compares each node's confirmation total to their
difference. `--network-remaining-usd` is separate and compares the sum of all
validator costs to an explicitly supplied campaign allowance. All money values
are USD, all seconds are wall-clock GPU seconds, and no omitted cost category
is treated as zero: omitted overhead, metrics, judge, storage or reconciled
screening spend inputs produce
`PROJECTION_INCOMPLETE_INPUTS`.

The receipt's `windows` field gives the minimum commit/reveal duration in
seconds and, when a block interval is supplied, in blocks. It replaces the old
two-hour planning assumption: a frozen batch must use windows at least that
large.

The receipt counts each validator's actual assignments, rather than dividing a
global average by the number of validators. A quality assignment creates one
candidate video and a reference video. It needs a second reference only when
the current king and Wan-50 are different immutable manifests; in that case the
same candidate video may be compared to both references because its prompt,
seed, recipe and artifact are frozen. A **screening** timing assignment is one
isolated AB pair (`R,C`); a **confirmation** timing assignment is an ABBA
quartet (`R,C,C,R`, two paired repeats). They use different stage-bound receipt
schemas and neither reuses quality renders. VBench metric runtime is charged
only for quality videos (including distinct king/Wan-50 references); timing
AB/ABBA renders are stored but do not invoke VBench. Metric runtime,
warmup/load/shutdown reserve, judge calls and storage are separately visible.
The overhead seconds must also include network transfers, any serial judge/API
waiting and GPU idle time until shutdown; the GPU remains billable during these
waits. The 30-minute example is an assumption for all of that overhead, not a
measured allowance.

Keeping 128 prompts preserves the current design; it does not establish
statistical power. Independent calibration and protected-category sensitivity
must still pass before rewards can be enabled.

The diagnostic pilot's measured RTX 6000 Ada values were approximately
278.14 seconds for Wan-50 and 225.28 seconds for the 40-step candidate, at an
observed historical price of 0.70 USD/hour. The old 2-validator projection had
79 quality + 21 timing jobs on one GPU and 84 + 16 on the other: 16.92 and
15.22 GPU-hours respectively, or 10.65--11.84 USD per validator before
screening, overhead, metrics, judging or storage. Thus 278 seconds per video
does make a full 128+32 confirmation take many hours on that hardware.

For a 2-hour confirmation target, run the command with `--max-node-hours 2`.
Its `fit_target.per_node.required_uniform_generation_speedup` reports the
measured speedup required for each actual allocation. It includes no imagined
hardware multiplier. A historical cold H100 observation measured Wan-50 at
about 77 seconds, versus about 278 seconds on the Ada pilot, but there is no
corresponding 40-step H100 measurement. The planner therefore accepts separate
King, Wan-50 and challenger measurements rather than extrapolating the latter.

An eight-validator H100 capacity-only scenario preserves all 128 quality and
32 ABBA timing prompts with 25% audit replication. It uses the 77.00-second
historical Wan-50 H100 observation, a clearly labeled *assumption* that the
challenger takes 81% of that time, 30 minutes of node overhead, and the H100 80 GB
price observed on 2026-10-04 of 3.49 USD/hour. On one deterministic allocation it
projects about 1.8 hours makespan and at most 6.34 USD of GPU cost per
validator; two H100 validators do not meet the two-hour target. The network
GPU charge is about 46 USD because eight GPUs reduce elapsed time, not work.
Metrics, judging and storage are unmeasured in that scenario, so the receipt is
`PROJECTION_INCOMPLETE_INPUTS` and cannot authorize a deployment or spend.

More GPUs reduce wall-clock time by distributing fixed work; they do not erase
network GPU cost. With eight equally loaded GPUs, the same roughly 32.1 total
GPU-hours would ideally approach 4.0 hours of makespan before overhead, while
the GPU charge remains about 22.5 USD at that historical price. This is only a
throughput illustration: the present allocator equalizes total assignments, not
quality and timing work separately, so the planner emits a warning when their
mix differs. A faster GPU changes the measured seconds and often price; submit
fresh measurements rather than applying an assumed multiplier.

Screening is a distinct frozen plan and needs its own capacity receipt. Its AB
receipt is rejected for confirmation, which still requires exactly 32 ABBA
slots. Screening may stop a losing challenger before confirmation, but it may
never promote one.
If the full frozen panel cannot finish within either declared time or budget,
the result is `INCOMPLETE_INSUFFICIENT_CAPACITY`; executing only part of it
cannot yield a crown or weights.

## Measured RTX 5090 probe, 2026-10-04

The bounded probe completed 15 quality renders and eight synchronized timing
renders, with five additional warmups, on one RTX 5090. The format was 832×480,
81 frames at 16 fps. Runtime: PyTorch 2.11.0+cu130, Diffusers 0.36.0,
bfloat16 transformer/text encoder and float32 tiled VAE. Wan weights were pinned
to revision `0fad780a534b6463e45facd96134c9f345acfa5b`.

The two ABBA quartets measured median complete-pipeline latency of **176.48 s**
for Wan-50 and **142.97 s** for the 40-step CFG-5 uniform-Euler recipe. These are
not measurements of a trained LoRA. Peak allocated memory across quality and
ABBA was **18.22 GiB**. This establishes that the tested runtime fits a 32 GB
5090; it does not establish a competitive speed confidence bound from 32 prompts.
The historical H100 observation used another runtime and is not a controlled
hardware comparison.

Using those measured medians, 128 quality prompts, 32 ABBA prompts, 25% audit
replication and the exact deterministic allocations produces the following
**projections**, at the observed rental price of 0.99 USD/hour. Each includes an
*assumed* 30 minutes per node for overhead. Metrics, judging, storage and prior
screening are excluded here and remain explicitly unknown in the receipts.

| Validators | Slowest node, hours | Network GPU cost, USD | Highest node GPU cost, USD |
|---:|---:|---:|---:|
| 2 | 11.15 | 21.99 | 11.04 |
| 4 | 6.09 | 23.50 | 6.03 |
| 8 | 3.34 | 24.96 | 3.31 |
| 10 | 2.98 | 26.12 | 2.95 |
| 20 | 1.74 | 30.98 | 1.72 |

These are capacity examples, not a recommendation to rent that many GPUs or
permission to spend. Two validators exceed the two-hour target and the most
loaded node already exceeds 10 USD before omitted costs. Twenty validators fit
only the supplied generation/overhead assumptions: the receipt remains
`PROJECTION_INCOMPLETE_INPUTS`. Equal total assignments do not guarantee equal
quality/timing work; the audited work mix also changes between committees.

The five VBench functions took 103.77 seconds in total for 23 source/derived
videos in this probe, excluding interpreter/import startup. This small mixture
of controls does not establish full-panel metric cost. GPU rental also includes
provisioning, downloads, model loading, API waiting and evidence backup.

Screening's new AB path cuts its timing renders from four to two per prompt.
For the initial king equal to Wan-50, separate quality and timing work falls
from six to four renders per screening prompt (33.3%). Confirmation is unchanged.
This reduction does not by itself make a two-validator full evaluation viable
within two hours; stronger model acceleration or more distributed capacity is
still required, with quality qualification preserved.


## Measured RTX 5090 qualification projection, 2026-10-05

The completed 16-prompt UniPC development experiment supplies paired full-pipeline
runtime and 32 settled Luna calls; see [its evidence and limits](WAN_MVP.md#measured-unipc-development-evaluation-2026-10-05).
At the observed 0.99 USD/hour rental rate, the current 552-generation calibration
plan projects 27.64 USD of generation. Its 960 comparisons require 1,920 blind
calls; applying this experiment's mean usage price projects another 8.88 USD.
Together these two components are **36.52 USD**. This is a projection, not an
invoice, a spending cap or a guarantee of successful calibration.

A separate bootstrap candidate scenario uses 208 generations per side: 16
screening, 128 quality and 64 timing generations (two per side for each of the
32 ABBA prompts). With the measured Wan-50 and UniPC-40 times and 288 blind
calls, generation plus Luna project **20.09 USD network-wide**, before audit
replication. This assumes a candidate as fast as UniPC-40; the measured recipe
has **not** demonstrated preserved quality and is not qualified for dispatch.
The sum of these calibration and candidate components is **56.60 USD**.

All figures exclude metric execution, warm-up, loading, encoding overhead,
GPU idle time during API/transfers, storage, transfers, additional training and
failed experiments. Two validators divide assigned work; they do not halve
network cost, and replicated audits add work. These measurements therefore do
not establish either the per-validator 10 USD ceiling or affordable end-to-end
qualification. Reconcile the operator's remaining campaign allowance separately
before any rental. No smaller diagnostic or synthetic certificate substitutes
for independent calibration and real candidate confirmation.
