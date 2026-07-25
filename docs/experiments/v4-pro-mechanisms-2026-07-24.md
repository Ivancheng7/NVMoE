# V4-Pro Rust canonical mechanism experiments

Date: 2026-07-24

Updated: 2026-07-25

Status: decision-sufficient Rust canonical mechanism PoC

## Decision boundary

This report records storage, caching, and layer-stationary mechanisms on the
pinned DeepSeek-V4-Pro checkpoint. It is self-contained: the conclusions below
do not require opening the raw JSONL traces.

The supported claim is deliberately narrow:

> On one RTX 4080 Laptop GPU under WSL2, the Rust/cuda-oxide path executed all
> 61 layers with stable canonical Rust traces, and the measured M3, M4, and B2
> mechanisms, the B4 gate4 throughput cohort, and three independent
> single-user prefill confirmations reduced their prespecified estimands
> without changing the recorded routes, weights, hidden/KV state, compressor
> state, or final Rust output.

This is a **decision-sufficient Rust canonical mechanism PoC**. It is not an
official-PyTorch bit-exact acceptance, a generated-token result, a long-context
result, a B8 result, a single-request B4 latency result, or a general 2.6x
runtime claim.

The local mechanisms stay on three separate axes: mHC linear work, CSA/HCA
attention, and HC layer-stationary fixed-work sharing. Their estimands are not
pooled into a single throughput claim.

## Frozen environment and source

| Item | Recorded value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Pro` |
| Revision | `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Model path | `/mnt/colibri-native/deepseek-v4-pro-b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Model filesystem | `/dev/sde`, ext4, mounted at `/mnt/colibri-native` with `noatime` |
| Host | WSL2, Linux `6.18.33.2-microsoft-standard-WSL2`, x86-64 |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB, Ada `sm_89` |
| Driver | `610.74` |
| Repository HEAD | `81f08a09e5651ce52616dc720f68810f9021c0be` |
| Rust toolchain | `nightly-2026-04-03`, pinned by [`rust-toolchain.toml`](../../rust/rust-toolchain.toml) |
| Source identity | Not recorded by the early M1-B2 launchers |

The early M1-B2 traces predate the fail-closed source-identity launcher and
therefore cannot be tied to the final reviewed source hash. Their trace hashes
below identify the artifacts, not the exact compiled worktree. Any headline
claim must be rerun through `rust/run-identified.sh`, whose JSON run header
records the verified source manifest, binary, build mode, and token E2E timing.
Raw traces remain outside the repository.

## Estimands and fair comparisons

The measurements answer three different questions and must not be pooled:

- **First token:** one position-0 token across 61 layers, with no cross-token
  fixed-cache reuse.
- **Steady token:** token 1 after token 0 filled the same-process cache.
- **Throughput slice:** two independent position-0 requests through only L3
  HCA and L4 CSA, sharing fixed layer work but not semantic state or routed
  work.

| Fair comparison | Scope | Repeats | Control median / p95 | Mechanism median / p95 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| M1 vs M3 | First token, full 61 layers, token 0 | 3 per arm | M1 wall `94.060 / 96.103 s` | M3 wall `62.020 / 62.713 s` | M3 pass; `1.517x` wall |
| M2 local gate | Steady token, L3/L4 within layers 0--4 | 1 per arm | cache-off `5.890190 s` | decoded cache `3.894631 s` | local pass; `1.512x` |
| Decoded vs M4 raw composition | Steady token, full 61 layers, tokens `[0,1]` | 3 per arm | decoded M2+M3 `30.480 / 30.950 s` | raw M4+M3 `24.546 / 25.041 s` | M4 pass; 19.468% lower |
| B2 serial vs layer-stationary | Two independent requests, L3/L4 only | 20 paired | serial `3.334721 / 3.533139 s` | B2 `2.002625 / 2.053425 s` | B2 pass; `1.665x` |

The decoded and raw compositions had comparable first-token fill:
`58.977 / 62.509 s` and `60.138 / 61.924 s` median/p95 respectively. Their
reported win is therefore the second-token estimand, not first-token latency.

The M4 steady median is also `2.595x` faster than a same-sequence, uncached
M3-only token-2 control of `63.688 s`, but that control has `n=1`. It is
retained as secondary context, not promoted as a general or equally repeated
claim.

B4 gate4 is a separate fair full61 throughput cohort: four ABBA strata each
closed the Rust-canonical exactness gate and were positive. Cohort completion
was `251.901 s` serial versus `75.146 s` layer-stationary, or `3.352152x`.
The attached artifact is
[`gate4-abba.jsonl`](../assets/readme/data/gate4-abba.jsonl), SHA-256
`1c0936ae885e563bb3190b505f9b574f102370c9dfc1e0e9aedd84bbf9362db5`.
This is a throughput comparison for a four-request cohort, not a claim about
one request's latency or official-PyTorch parity.

## Post-milestone chat mechanism evidence

The tracked
[`official-single-user-hi-greedy-1-after-validator-fix.jsonl`](../assets/readme/data/official-single-user-hi-greedy-1-after-validator-fix.jsonl)
artifact records one official single-user V4 `chat`-mode request with content
`hi`. Its SHA-256 is
`7ad3a351392fb01de36848a1de101892a9629bd9e1a59034e8756adb6254932c`.
Five prompt IDs each crossed 61 layers for 305 layer records; the canonical
greedy head emitted one token decoded as `效果好`, and the request E2E was
`500.407 s`. This is a state and wiring mechanism pass. It is not interactive,
and it does not establish system messages, tools, multiturn behavior, sampling,
quality, accepted-token rates, tok/s, or long-context support.

## Confirmatory causal layer-stationary prefill

This gate measures complete prefill for the official single-user `hi` prompt,
not the preceding single-trace wiring result. The protocol was frozen before
confirmation results. It compares token-major serial prefill with causal
layer-stationary prefill, which loads each layer's fixed runtime once and then
advances the five prompt positions through that layer. It retains independent
routes, routed experts, hidden state, attention, KV, compressor state, and the
final head.

The exploratory `pair-smoke.json` pilot is excluded from final statistics. The
protocol is
[`confirmation-abba-protocol.json`](../assets/readme/data/confirmation-abba-protocol.json),
SHA-256 `03447f79dba8550345b9a4b713d589148b203b238a4e6c81efe973128af262b7`.
The frozen summary is
[`confirmation-abba-summary.json`](../assets/readme/data/confirmation-abba-summary.json),
SHA-256 `d0e62825d0a041f3c37a4c2339dfa171b10ab84f91d4a1f4e79d30fd1715bd8e`.
The repository tracks all eight raw traces used by the four pairs.

| Measure | Serial prefill | Causal layer-stationary prefill | Result |
| --- | ---: | ---: | --- |
| Prompt E2E median | 473.309382 s | 287.927884 s | `1.643847x` ratio of medians |
| Prompt E2E p95 | 483.201318 s | 288.770390 s | stationary below serial |
| Fixed H2D | 145.307855 GB | 29.061571 GB | fivefold fixed-work reuse |
| Routed H2D | 64.222986 GB | 64.222986 GB | unchanged by design |
| Semantic gate | 305 layers plus final | 305 layers plus final | `4/4` exact, `4/4` positive |

The paired ratios are `1.634432x`, `1.632435x`, `1.648528x`, and `1.684915x`.
The experiment supports Rust-canonical A/B equality for this official
single-user prefill workload. It does not establish official PyTorch full61
parity, interactive latency, tok/s, text quality, system messages, tools,
multiturn behavior, sampling, or long context.

## Confirmatory exact expert-slot reuse

This independent frozen protocol begins with causal layer-stationary prefill in
both arms. Arm A has no routed expert-slot reuse. Arm B enables exact
layer-local reuse. It is therefore a narrower question than the prior serial
versus layer-stationary confirmation.

The protocol is
[`expert-slot-reuse/protocol.json`](../assets/readme/data/expert-slot-reuse/protocol.json),
SHA-256 `63d1237e928a25aaf2a2c95abe529789f6fa2111453806b234bffd00e4a57d97`.
The summary is
[`expert-slot-reuse/summary.json`](../assets/readme/data/expert-slot-reuse/summary.json),
SHA-256 `155d6ac26df6dca03a9b13c76a63a951d13abe30b07923be1c817060527878ea`.
The exploratory development pilot is excluded. One failed attempt exited with
code 101 and left no result artifact. It contributes no trace or measurement to
the summary and is excluded from the confirmatory statistics. The incident is
tracked at
[`pair1-B-preinference-env-var.json`](../assets/readme/data/expert-slot-reuse/incidents/pair1-B-preinference-env-var.json),
SHA-256 `0afe2cce157a3873d7f229b5a3f75b95b50adf1fa094ddc02aa7d69a626a1dc1`.
Its original "pre-inference" and `inference_started=false` fields are not used
as evidence. The artifact contains no captured stderr, launcher or source
identity, or timing record that can establish the failure stage. The
[`incident audit amendment`](../assets/readme/data/expert-slot-reuse/incidents/incident-audit-amendment.json)
records that limitation without changing the original artifact. Current source
inspection at the time of the amendment found the result-path read after the
complete layer-position loop and canonical final head; this does not
retroactively identify the source or binary used by the failed attempt. No
runtime fix is part of this evidence-only amendment.

| Measure | A: no reuse | B: exact slot reuse | Result |
| --- | ---: | ---: | --- |
| Prompt E2E median | 283.197157 s | 237.053777 s | `1.194654x` ratio of medians |
| Prompt E2E p95 | 283.903281 s | 239.591386 s | B below A |
| Fixed source / H2D | 24.968176 / 29.061571 GB | 24.968176 / 29.061571 GB | unchanged by design |
| Routed source / H2D | 64.222986 GB | 29.619782 GB | reduced with reuse |
| Semantic gate | 305 layers plus final | 305 layers plus final | `4/4` exact, `4/4` positive |

The paired ratios are `1.184948x`, `1.199045x`, `1.192034x`, and `1.197298x`.
All traces keep the frozen source identity, model revision, and cuBLAS identity.
This result is Rust-canonical A/B equality for one official single-user prefill
workload. It is not official PyTorch full61 parity, interactive latency, tok/s,
text quality, or general chat support. The preceding `473.309382 s →
287.927884 s` result and this `283.197157 s → 237.053777 s` result are separate
confirmation lineages. Do not multiply their ratios. `473.309382 s →
237.053777 s` is a descriptive cross-study endpoint only, not a paired
factorial estimate.

A [post-review current-source bridge](../assets/readme/data/expert-slot-reuse/post-review-bridge/summary.json) is a no-regression gate, not a fifth confirmatory observation. It matched pair0-B at all 305 layer records and the final record, with fixed-source, fixed-H2D, routed-source, and routed-H2D bytes equal. The maximum five-position expert union was 29 under the 32-slot retained cap, so fallback did not execute, and the transactional `create_new` plus flush-and-sync result commit completed. Its 239.107 s wall time is descriptive only and supports no new performance claim.

A [cap-32 fallback execution gate](../assets/readme/data/expert-slot-reuse/cap32-fallback/summary.json) used six prompt positions and is excluded from the confirmatory n=4 and all performance statistics. Arm A reached a maximum layer union of 36, crossing the 32-slot cap, and Arm B remained exact across all 366 layer records plus the final record with equal fixed bytes. The frozen pre-B simulation reproduced 883 routed loads and 30,988,468,224 routed bytes and recorded 10 fallback witnesses; the first was layer 0, position 5, route slot 1, expert 77, after 32 retained slots. Both result transactions completed their `create_new`, flush, and sync checks. The 335.977 s to 267.019 s wall observation is descriptive only and supports no performance claim.

A [corrected stationary phase profile](../assets/readme/data/stationary-phase-profile/analysis.json) adds 61 fixed scopes, 305 scopes each for attention, router, shared expert, and routed MoE, and one final scope to the exact post-review bridge path. All 305 layer records and the final record match after removing `phase_profile`; physical bytes match, per-record sums close the header, and the monotonic and nested-parent gates pass. Exclusive request shares are 49.81% attention, 22.28% fixed-layer loading, and 19.87% routed MoE, so attention is the next measured bottleneck. The 244.419 s profile is 1.0222x the 239.107 s bridge reference, below the 1.10 intrusion gate, but this single diagnostic run is outside the confirmatory n=4 and supports no performance claim. The [first schema-failed trace status](../assets/readme/data/stationary-phase-profile/excluded-first/schema-status.json) freezes its SHA and exclusion without shipping that large trace.

## Confirmatory one-layer-ahead CPU fixed prefetch

The frozen [protocol](../assets/readme/data/fixed-cpu-prefetch/confirmatory/protocol.json)
compares the same layer-stationary, expert-reuse, profiled, cache-off path with
and without one-layer-ahead CPU fixed-weight read/decode. Its SHA-256 is
`fdade20a329794ae3c691a2c8096b3aefb0fcab09668faba19ad4ffb7c1cab65`.
The [summary](../assets/readme/data/fixed-cpu-prefetch/confirmatory/summary.json)
has SHA-256
`be0b369e37c26da8455564e2616964065dd289402e45ff9f6c7952c89cc519b0`.
The development pilot is excluded and not copied into the confirmatory packet.

| Measure | A: no CPU prefetch | B: one layer ahead | Result |
| --- | ---: | ---: | --- |
| Prompt E2E median | 245.425660 s | 195.743459 s | `1.253812829x` ratio of medians |
| Linear empirical p95 | 248.535317 s | 195.907536 s | B/A `0.788248x` |
| Fixed source / H2D | 24.968176 / 29.061571 GB | 24.968176 / 29.061571 GB | unchanged |
| Routed source / H2D | 29.619782 / 29.619782 GB | 29.619782 / 29.619782 GB | unchanged |
| Semantic gate | 305 layers plus final | 305 layers plus final | `4/4` exact, `4/4` positive |

The paired ratios are `1.271873x`, `1.259494x`, `1.248007x`, and
`1.245291x`. Each B trace has 60 transitions and all 60 are ready before
consume, with zero aggregate consume wait. Worker errors and 4 GiB startup
`MemAvailable` guard violations are zero. Raw transition records reproduce the
reported read, decode, and overlap sums. The frozen runner did not instrument
per-run minimum RAM, so its `null` value cannot support a minimum-memory claim.
This is prompt-E2E confirmation on one laptop and one frozen identity, not
tok/s or interactive latency. The `473.309382 s → 195.743459 s` cross-lineage
endpoint is descriptive; no ratio from the three confirmation lineages is
multiplied.

## Confirmatory persistent routed QD4

The frozen
[`protocol`](../assets/readme/data/qd4-full61/confirmatory/protocol.json),
SHA-256 `617d7b099e2067a7cf544432a1e146b18630f6c4309d17da2afe527d45806b97`,
keeps the official single-user `hi` prompt, five positions, all 61 layers,
phase telemetry, and fixed CPU prefetch in both arms. Arm A uses exact union
expert reuse. Arm B disables reuse and runs six routed jobs per layer through
four persistent QD4 lanes. The development pilot, old controls, and incident
artifacts are excluded.

| Measure | A: exact union reuse | B: persistent QD4 | Result |
| --- | ---: | ---: | --- |
| Prompt E2E median | 199.431954 s | 148.555131 s | `1.342477723x` ratio of medians |
| Linear empirical p95 | 201.077111 s | 150.062579 s | B/A `0.746294x` |
| Routed source / H2D | 29.619782 / 29.619782 GB | 64.222986 / 64.222986 GB | B executes all six jobs |
| Routed expert loads | 844 | 1,830 | four persistent lanes, six jobs per layer |
| Semantic gate | 305 layers plus final | 305 layers plus final | `4/4` exact, `4/4` positive |

Arm A times were 199.832706 s, 197.576603 s, 199.031202 s, and
201.296711 s. Arm B times were 148.140518 s, 146.336024 s, 148.969744 s,
and 150.255432 s. Paired ratios were 1.348940x, 1.350157x, 1.336051x, and
1.339697x. Every run records 60 fixed-prefetch transitions, all ready before
consume, with zero consume wait. Arm A v1 and Arm B v2 phase records close
their per-record and request accounting.

The runner did not emit a summary. The tracked
[`derived_from_raw` summary](../assets/readme/data/qd4-full61/confirmatory/summary.json),
SHA-256 `65ce6de19236a591af7aeffd6eab31c2e706663b77e8c81b4f2d0ee6a016993d`,
was independently replayed from the eight unchanged traces and eight PASS
logs. The packet
[`SHA256SUMS`](../assets/readme/data/qd4-full61/confirmatory/SHA256SUMS)
checks every tracked artifact. The protocol STOP gate declares zero
capacity-guard violations and all logs report PASS; the traces do not record
per-run memory readings.

This is prompt E2E under Rust-canonical equality, not tok/s, TTFT, interactive
latency, complete official generation, or official-reference full61 parity.
The `473.309382 s` and `148.555131 s` medians are descriptive endpoints from
different confirmation lineages, not a paired factorial comparison.

The supporting [L3/L4 five-position gate](../assets/readme/data/qd4-stationary-gate/review-boundary.json)
has `4/4` exact pairs per layer and 40 contribution/state records. Its
routed-scope ratios are 3.108982x and 4.117849x. Fixed prefetch was not
exercised there, and the local gate is excluded from the full61 statistics.

### STOP: device-attention expansion

The [local gate](../assets/readme/data/device-attention-gate/gate-attempt-1.jsonl)
contains 30/30 exact pairs for each of L3 and L4 at positions 0/1, with p50
wall ratios of `1.236661x` and `1.326521x`. Expanding to positions 0 through 4
was stopped at the CSA persistent-state boundary. The
[review wrapper](../assets/readme/data/device-attention-gate/review-boundary.json)
records that no expansion artifact exists and that the local result is not a
full61 or headline claim.

## M1: full61 baseline

M1 executed token 0 at position 0 through all 61 Rust layers. One run followed
`POSIX_FADV_DONTNEED`; two were ordinary warm repeats. The cold hint did not
produce a faster-cache interpretation: wall times were `89.93`, `94.06`, and
`96.33 s`.

| M1 stage, 61-layer total | Median | p95 |
| --- | ---: | ---: |
| Layer elapsed | 91.652 s | 93.806 s |
| Fixed read + allocation | 34.380 s | 35.396 s |
| Fixed decode | 20.803 s | 21.298 s |
| Fixed H2D | 3.528 s | 3.599 s |
| Routed inclusive | 21.661 s | 22.157 s |

Each run had a corrected unique fixed-source footprint of
`24,968,175,688 B`, fixed H2D of `29,061,571,000 B`, routed source/H2D of
`12,844,597,248 B`, and total fixed+routed H2D of `41,906,168,248 B`.

The original top-level telemetry covered only a partial manifest. The old
`7.57x` decoded/source ratio is withdrawn. The corrected ratio is
`29,061,571,000 / 24,968,175,688 = 1.164x`.

## M2: decoded fixed cache, local positive and full61 negative

M2 statically ranked whole-layer decoded fixed bundles by expected time saved
per byte. Device hits avoid source read, allocation, decode, and H2D; host hits
avoid source read, allocation, and decode but retain H2D. Preflight preserved
2 GiB device and 4 GiB host headroom and failed closed to a smaller budget
rather than reducing headroom.

The real L3/L4 local gate cached `953,416,752 B`. On token 2 it reduced E2E
from `5.890190 s` to `3.894631 s` and made L3/L4 fixed H2D exactly `0 B`.
Canonical fields matched.

The full61 capacity model was negative. Safe effective budgets of about
7.765 GiB device and 9.546 GiB host could retain 17 device layers and 21
additional host layers. Even assuming all modeled savings materialized, the
steady upper bound was only `1.631x`, below the frozen `1.8x` continuation
gate. M2 was killed as a standalone full61 experiment; no full61 M2-only run
was forced by reducing headroom. This negative result motivated composition
with M3 rather than retrospective threshold changes.

## M3: four persistent routed QD4 lanes

M3 allocates four persistent routed-expert lanes once. Each lane owns a CUDA
stream, pinned source buffers, device buffers, scratch, and events. The first
four of six canonical routed experts are submitted immediately; completion
drives the remaining two. Reduction stays in canonical route order.

The real L3/L4 gate used 30 serial and 30 QD4 samples per layer after one
warmup. Routed inclusive medians fell from `306.926` to `23.950 ms` at L3 and
from `311.304` to `24.259 ms` at L4. QD4 read throughput was about
`9.9 GB/s`, and steady lane setup/allocation was zero.

The full61 expansion ran three times. Routed total median was `3.322 s`, down
84.66% from M1's `21.661 s`; all-observation routed p95 was `63.977 ms`,
versus M1's `401.609 ms`. Full wall median improved from `94.060` to
`62.020 s`. Fixed and routed byte volume remained the M1 volume; M3 changes
overlap and allocation/setup lifetime, not the mathematical workload.

## Decoded M2 + M3 composition

The composed run used tokens `[0,1]` in one process. Token 0 filled the decoded
fixed cache and token 1 measured hits while QD4 stayed enabled. Three token-2
times were `30.480`, `30.377`, and `31.002 s`.

At the median live placement, token 2 had 19 device hits, 22 host hits, and 20
misses. It read `8,368,468,544 B` of fixed source, transferred
`20,127,591,408 B` of fixed H2D, and retained canonical routed source/H2D of
`12,844,597,248 B`. Actual host availability changed one host-hit/miss between
runs and is preserved in each trace.

All 122 layer records in all three runs matched the uncached two-token M3
control for the canonical fields listed below. Final output was stable at
argmax `40341`.

## M4: raw fixed cache and GPU E4M3 decode

M4 retains `wo_a` as source E4M3FN plus scale instead of doubled BF16. One
128 MiB BF16 device scratch is allocated once. A raw hit expands E4M3FN on the
GPU into that scratch, executes the layer, and swaps ownership back without a
steady decode allocation.

The conversion gate checked all 256 byte codepoints and matched the canonical
CPU conversion exactly. At an equal 1.2 GiB device-cache budget, the local
L3/L4/L5 pair retained all three layers in raw form versus L3/L5 decoded,
reducing token-2 E2E from `6.387384` to `4.685945 s` (26.637%). Raw-hit reads,
H2D, and decode allocations were zero; decode p95 was `0.581 ms`.

The full61 raw-device + raw-host + M3 composition ran three times. Token-2
times were `24.546`, `24.414`, and `25.095 s`. At the median live placement it
had 22 raw-device hits, 25 raw-host hits, and 14 misses; fixed source reads were
`5,946,128,256 B`, fixed H2D was `17,037,487,016 B`, and routed source/H2D
remained `12,844,597,248 B`. Across the three runs, GPU expansion p95 was
`0.633 ms`; raw-decode allocation and QD4 lane allocation/setup stayed zero.

The mechanism passed because it was 19.468% below the equally repeated decoded
composition while preserving all 122 canonical layer records and final output.

## Host24 token-2 max-plus closure

The paired host-cap archive (historically stored outside the repository) has
an exact outer timing closure for token position 1. Its reported
`token.e2e_ns` is `27,137,582,440 ns`; the 61 contiguous
`layer_elapsed_ns` values sum to `27,137,480,746 ns`. Thus the serial
layer-DAG replay misses by only `101,694 ns` (`0.000375%`). This certificate is
for the outer precedence chain, not an inferred per-lane GPU/SSD schedule.

For token 2, the exclusive/opaque critical-path accounting is:

| Stage | Time | E2E share |
| --- | ---: | ---: |
| Fixed H2D plus raw `wo_a` H2D | `2.455013 s` | `9.047%` |
| Attention plus commit | `11.268225 s` | `41.523%` |
| Router scope | `9.376578 s` | `34.552%` |
| Shared expert | `0.474176 s` | `1.747%` |
| Routed opaque inclusive scope | `3.281017 s` | `12.090%` |
| Decode, hash, and closed residual | `0.282471 s` | `1.041%` |

The run had 22 raw-device and 39 raw-host fixed-cache hits, with no fixed
source read. Consequently, eliminating every fixed transfer can save at most
`2.455013 s/token` (`9.047%`); it does not justify a large double-buffering
layer. A hypothetical 2x reduction of the whole attention scope saves
`5.634113 s`, router scope 2x saves `4.688289 s`, and the additive no-overlap
upper bound is `10.322402 s` (`38.037%`), leaving `16.815181 s`. These are
stage-scale upper bounds, not performance predictions.

Persistent QD4 was already enabled, so it has no incremental token-2 claim.
The existing exact CDQW L3/L4 smoke supports only its two-layer local result
(`14.845`--`19.391 ms/token` by the observed minimum/p50 pairing), not a
full61 extrapolation. Expert cache/prefetch cannot be identified from this
trace: `routed_ns` is inclusive, while `routed_read_ns`, `routed_h2d_ns`, and
`routed_kernel_ns` aggregate six jobs across four lanes and must not be added.
The token-to-token route overlap is 223 of 366 `(layer, expert)` slots
(`7,826,079,744 B`), but the host fixed cache already holds `16.278 GB` under
a `19.095 GB` effective budget, so this observation alone neither fits nor
prices an expert-cache counterfactual.

### Router exclusive probe

The prior `router_ns` was an opaque wall interval spanning FFN mHC pre,
host preparation, router buffers/copies, selection, and route download. An
identified release probe added exclusive wall fields under
`COLIBRI_V4_BENCH_TELEMETRY=1` only. It ran the existing exact token `[0,1]`,
`max_layers=5`, persistent-QD4 harness three times; only L3/L4 at position 1
are reported. The binary source identity was
`ee706045638fdafe24adb02d834b60f8083d919905353c10e52e31bdc5032d1a`.
All three runs matched frozen host24 exactly for route IDs, route-weight bits,
attention top-k, hidden, KV, and compressor/indexer digests.

| Field | Meaning | L3 median | L4 median |
| --- | --- | ---: | ---: |
| `router_ns` | Parent exclusive wall interval | `120.136 ms` | `119.856 ms` |
| `router_mhc_ns` | FFN mHC-pre opaque wall interval | `115.332 ms` | `115.139 ms` |
| `router_host_prep_ns` | Hash-route host preparation | `0.000111 ms` | `0.000020 ms` |
| `router_alloc_h2d_sync_ns` | Router input/hash `DeviceBuffer::from_host` allocation, H2D, and required sync | `0.055 ms` | `0.050 ms` |
| `router_workspace_alloc_ns` | Router output/logit workspace allocation | `0.085 ms` | `0.037 ms` |
| `router_kernel_select_sync_ns` | RMSNorm, gate, and select wall interval through event completion | `4.560 ms` | `4.496 ms` |
| `router_d2h_sync_ns` | Normalized input, IDs, and weights download/sync | `0.126 ms` | `0.104 ms` |
| `router_other_ns` | Audit/timer residual | `0.016 ms` | `0.018 ms` |

`router_kernel_select_gpu_ns` is an event-clock diagnostic nested in the
selection wall interval, not an additive field: it was `4.489 ms` for L3 and
`4.431 ms` for L4. The additive wall fields close exactly to `router_ns` for
all six observations. `DeviceBuffer::from_host` deliberately remains one
honest allocation+H2D+sync field because that existing API combines those
operations. The main sub-bottleneck is therefore FFN mHC pre (96.00%/96.07%
of the L3/L4 router wall), not router selection, router allocation, or D2H.
This is a stage diagnosis only; it is not a full61 timing claim.

The subsequent exact device-handoff gate was **KILL**: L3 was `0.9626x` and
L4 was `1.00094x`. Thus the 28,832-B mHC round trip is not the 115-ms root
cause; the candidate was removed rather than retained as an optimization.

### mHC internal exclusive probe

An identified release probe then split the FFN mHC-pre scope at its existing
same-stream kernel-launch boundaries. It ran the same token `[0,1]`,
`max_layers=5`, persistent-QD4 harness three times and reports L3/L4 at
position 1 only. Source identity was
`5e5fa7ebb31fbeffbcb549591469301629f7aa125ce36005b4640f46f60bf0c6`.
All six observations closed their additive mHC wall fields exactly and matched
frozen host24 routes, route-weight bits, attention top-k, hidden, KV, and
compressor/indexer digests.

| Field | L3 median | L4 median |
| --- | ---: | ---: |
| `router_mhc_ns` | `57.376 ms` | `57.236 ms` |
| `router_mhc_input_alloc_h2d_sync_ns` | `0.042 ms` | `0.040 ms` |
| `router_mhc_workspace_alloc_sync_ns` | `0.066 ms` | `0.056 ms` |
| `router_mhc_mixing_linear_sync_ns` | `56.695 ms` | `56.547 ms` |
| `router_mhc_sinkhorn_normalize_collapse_sync_ns` | `0.446 ms` | `0.451 ms` |
| `router_mhc_output_d2h_sync_ns` | `0.134 ms` | `0.137 ms` |
| `router_mhc_other_ns` | `0.003 ms` | `0.003 ms` |

The two GPU-event diagnostics are intentionally non-additive: workspace init
was `0.037/0.030 ms`, mixing/linear `56.149/56.144 ms`, and the fused
Sinkhorn+normalization+collapsed/post/comb kernel `0.366/0.367 ms` for L3/L4.
They are bracketed on the same stream and synchronized before the next launch,
so their two-kernel sum describes device execution, but it is not mixed with
host-wall closure. The source has only those two launches: splitting
Sinkhorn/normalization from collapsed/post/comb would require a kernel change,
so this probe does not pretend to measure nonexistent exclusive kernels.

Mixing/linear is `98.81%`/`98.80%` of mHC wall. Holding that one kernel fixed,
eliminating every other mHC component gives only a `1.012x` router-stage
Amdahl upper bound for either layer. The next smallest separately-approved
mechanism is therefore an exact SM89 parallel reduction for the existing
`mhc_linear_v4_bf16` contract, with the current kernel as the oracle. The
official fused prenorm+Sinkhorn path is SM90/100-only reference material, not
an implementation or comparison for this Ada run. No optimization was made
by this probe.

The actual target is `kernels::mhc_linear_v4_bf16` in
`rust/colibri-cuda/src/main.rs`. For the measured B=1 row it is exactly one
launch: `LaunchConfig::for_num_elems(1)` gives grid `(1,1,1)`, block
`(256,1,1)`, and zero dynamic shared memory. Only thread 0 owns the row; it
normalizes `[1,4,7168]` BF16 input and evaluates 24 FP32 dot products against
the contiguous output-major `hc_ffn_fn` layout `[24,28672]`
(`2,752,512 B`), with 192-element serial K blocks. `mhc_sinkhorn_v4_bf16` is
the sole following launch. A minimal parallel-reduction candidate must retain
the current scalar order as its exact oracle: changing its FMA/reduction order
can move FP32 mixes, mHC gates, collapsed BF16 output, routes, and persistent
state. It therefore requires the existing route/state digest gate before any
performance claim.

To identify a lane/resource replay, record per-route lane/stream ID,
`submit`, read/H2D/kernel/D2H start/end, reduction start/end, buffer ID, and
fence dependencies. The current 1-second sidecar telemetry is health evidence
only and cannot supply those edges. The routed-timeline work is therefore
**STOP** for this mechanism loop, rather than a reason to add scheduler
instrumentation or a larger fixed buffer.

## B2: two-request layer-stationary L3/L4

B2 tested a throughput mechanism separately from first-token and steady-token
latency. The control fully loaded and ran L3/L4 for request 0, then did the
same for request 1. B2 loaded, decoded, and transferred each layer's raw fixed
weights once, then ran the two independent requests against that read-only
runtime.

Requests had separate hidden, attention, router, KV, compressor, and output
state. Routed QD4 execution remained per request. Expert unions were telemetry
only; no route or routed tensor was shared.

| Per repeat | Serial | B2 |
| --- | ---: | ---: |
| Fixed source | 1,906,866,272 B | 953,433,136 B |
| Fixed H2D | 2,175,285,344 B | 1,087,642,672 B |
| Routed source | 842,268,672 B | 842,268,672 B |
| Routed H2D | 842,268,672 B | 842,268,672 B |

Fixed physical traffic was halved exactly; routed work was unchanged. Every
one of 20 repeats exceeded the 1.20 gate (range `1.610973x`--`1.926813x`).
Median request completion under B2 was `1.837023 s` for request 0 and
`1.997880 s` for request 1; median service latency was `2.002625 s`. Peak
process VmHWM was `883,703,808 B`, and peak sampled GPU memory used was
`2,041,053,184 B`.

B4 gate4 is **RECOVERED/ATTACHED**: all four ABBA strata are
Rust-canonical-exact and positive, with cohort completion `251.901 -> 75.146 s`
(`3.352152x`). Its scope is four-request throughput, not single-request latency
or official-PyTorch parity. B8 was not evaluated.

## Canonical exactness contract

The mechanism comparisons require exact equality within the Rust canonical
path for:

- ordered route IDs and FP32 route-weight bits;
- attention top-k digest where recorded;
- BF16 hidden, output, and KV digests;
- main/indexer compressed-KV digests;
- main/indexer compressor KV and score FP32-state digests;
- final logits/normalized/collapsed digests and argmax for full61 runs.

M1 and M3 matched across 183 layer observations. The decoded and raw two-token
compositions matched their uncached M3 control across 122 layers per run. B2
matched serial execution per request across all 20 repeats. No OOM, reclaim,
pinning, or QD4 lane-deadlock failure was observed.

“Exact” in this paragraph is internal canonical mechanism parity. It does not
mean that every Rust FP32/BF16 reduction is bit-identical to the official
PyTorch/TileLang implementation.

## Claim registry and limitations

| Claim | Status | Boundary |
| --- | --- | --- |
| Rust canonical all-61-layer execution | Positive | Token 0, position 0; three deterministic traces; not official-reference bit parity |
| Persistent routed QD4 | Positive | Full61 first-token wall `1.517x`; routed stage improved more, but total wall is the relevant first-token claim |
| Decoded fixed cache alone | Negative at full61 | Local L3/L4 passed; safe full61 upper bound `1.631x` missed the `1.8x` gate |
| Raw fixed cache + GPU E4M3 decode | Positive | Full61 steady token `19.468%` below decoded composition; not a first-token win |
| Layer-stationary B2 | Positive | L3/L4 two-request throughput slice only; `1.665x` ratio of medians |
| Official single-user prefill | Positive | `n = 4` ABBA pairs, 305 layers plus final exact; `1.643847x` prompt-E2E ratio only, not interactive latency, tok/s, or official parity |
| Expert-slot reuse prefill | Positive | `n = 4` ABBA pairs, 305 layers plus final exact; `1.194654x` from no-reuse layer-stationary prefill only, not multiplicative with the prior confirmation |
| One-layer-ahead CPU fixed prefetch | Positive | `n = 4` ABBA pairs, 305 layers plus final exact; `1.253812829x` prompt-E2E ratio, with RAM minimum not instrumented |
| Persistent routed QD4 prefill | Positive | `n = 4` ABBA pairs, 305 layers plus final exact; `1.342477723x` prompt-E2E ratio with 4 lanes and 6 jobs per layer |
| Device-resident attention | Local positive, expansion stopped | L3/L4 positions 0/1 only; `60/60` exact pairs; positions 0 through 4 cross the CSA persistent-state boundary |
| B4 gate4 | Positive | 4/4 ABBA strata exact and positive; `3.352152x` four-request throughput only, not single-request latency or official parity |
| B8, CUDA Graph, QUIC, federation | Not evaluated | Deferred |
| Long context or generated tokens | Not evaluated | Position 0/1 mechanism traces do not establish either |
| Official PyTorch bit-exact full61 | Not established | Official runner remains an oracle, not a passed full61 acceptance |
| General `2.6x` speedup | Withdrawn/unsupported | `2.595x` is only M4 steady token versus one same-sequence M3-only control |
| Decoded/source ratio `7.57x` | Withdrawn | Correct full61 unique-source ratio is `1.164x` |

The pinned official TileLang sparse-attention kernel requests `344,064 B` of
dynamic shared memory for `h=128,d=512`; Ada `sm_89` cannot launch it. The
independently cross-checked scalar PyTorch formula is therefore oracle-only and
never enters the Rust runtime or performance timing. Vendor-specific mHC
FP32/BF16 reduction differences are bounded and reported rather than relabeled
as bit equality; discrete routes/top-k and transactional state remain hard
gates.

The M2 kill, the absence of a first-token M4 improvement, the single-control
qualification on `2.595x`, the official sparse-kernel limit, and the earlier
mHC vendor-reduction difference are retained negative evidence.

### KILL: L3/L4 FFN mHC device-handoff gate

An identified local L3/L4 position-1 experiment tested the minimal proposed
handoff: FFN mHC and router input remained on the main GPU stream, while the
existing host shared-expert/QD4 canonical reduction path and final output
ownership were unchanged. The candidate added no persistent fields or kernels;
its only temporary router workspace was 1,584 B. All 30 paired A/B samples per
layer matched the canonical output, ordered route IDs and FP32 route-weight
bits, attention top-k digest, KV digest, and main/indexer compressor digests.

The performance gate was killed. L3 was `0.962581x` at p50 with `0/30` positive
pairs (`164.826102` versus `158.658429 ms`) and p95 worsened from `162.002156`
to `173.622540 ms`. L4 was `1.000937x` at p50 with `18/30` positive pairs
(`161.673031` versus `161.824502 ms`) and p95 was `167.434031` versus
`166.809668 ms`. Both miss the frozen `1.05x`, `24/30`, and no-more-than-5%
p95-reversal gate. No full61 run was made; the candidate code and harness were
removed after recording the artifacts.

Frozen identified artifacts: [gate30 JSONL](../assets/readme/data/mhc-device-handoff-gate30.jsonl)
(9,228 B, 64 lines, schema `colibri-v4-mhc-device-handoff-v1`), SHA-256
`ac29e3653f4c8623fad3d19bb162103794d773863303dc29eaf463d0dea14318`; and
[smoke JSONL](../assets/readme/data/mhc-device-handoff-smoke.jsonl) (2,124 B,
10 lines, the same schema), SHA-256
`0b454d5ff0d4822bc172eedb41f081aee64543eaa880756984e24554b5c76979`.

## Fail-closed reproduction

Run from the repository root. The preflight refuses a non-pinned path, any
non-native-ext4 filesystem, a missing oracle environment, or a snapshot that
fails pinned validation. The recorded `/dev/sde` mount is provenance only;
reproduction is parameterized by `COLIBRI_MODEL_DIR`.

```bash
set -euo pipefail
REV=b5968e9190ef611bbf34a7229255be88a0e937c1
: "${COLIBRI_MODEL_DIR:?set COLIBRI_MODEL_DIR to the pinned native-ext4 snapshot}"
MODEL_DIR=$COLIBRI_MODEL_DIR
: "${COLIBRI_ORACLE_PYTHON:?set COLIBRI_ORACLE_PYTHON to the pinned oracle venv Python}"
test -x "$COLIBRI_ORACLE_PYTHON"
test "$(basename "$MODEL_DIR")" = "deepseek-v4-pro-$REV"
read -r SOURCE FSTYPE < <(findmnt -nro SOURCE,FSTYPE -T "$MODEL_DIR")
test "$FSTYPE" = ext4
"$COLIBRI_ORACLE_PYTHON" rust/colibri-cuda/tools/v4_full61_oracle.py self-test
"$COLIBRI_ORACLE_PYTHON" rust/colibri-cuda/tools/v4_full61_oracle.py validate \
  --model-dir "$MODEL_DIR"
cd rust
cargo fmt --check
cargo check -p colibri-cuda
./run-identified.sh test --arch sm_89
```

The following are single-run templates. Use distinct output filenames for
repeats; do not overwrite evidence.

```bash
# M1: first-token full61 baseline
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_ID=0 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=0 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=0 \
COLIBRI_V4_RESULT_JSONL=/absolute/result/m1.jsonl \
./run-identified.sh run --arch sm_89

# M2 local gate, cache-off arm
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_IDS=0,1 \
COLIBRI_V4_MAX_LAYERS=5 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=0 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=0 \
COLIBRI_V4_RESULT_JSONL=/absolute/result/m2-local-off.jsonl \
./run-identified.sh run --arch sm_89

# M2 local gate, decoded device-cache arm
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_IDS=0,1 \
COLIBRI_V4_MAX_LAYERS=5 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=1 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=0 \
COLIBRI_V4_FIXED_CACHE_CANDIDATE_LAYERS=3,4 \
COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL=/absolute/result/m1.jsonl \
COLIBRI_V4_RESULT_JSONL=/absolute/result/m2-local-on.jsonl \
./run-identified.sh run --arch sm_89

# M3: first-token full61 persistent routed QD4
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_ID=0 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=0 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=0 \
COLIBRI_V4_ROUTED_QD4=1 \
COLIBRI_V4_RESULT_JSONL=/absolute/result/m3.jsonl \
./run-identified.sh run --arch sm_89

# Decoded M2+M3 steady token. Requested budgets are fail-closed to live
# availability while retaining 2 GiB device, 4 GiB host, and QD4 reserves.
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_IDS=0,1 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=64 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=64 \
COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL=/absolute/result/m1.jsonl \
COLIBRI_V4_ROUTED_QD4=1 \
COLIBRI_V4_RESULT_JSONL=/absolute/result/decoded-combo.jsonl \
./run-identified.sh run --arch sm_89

# M4 raw fixed cache + M3
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_IDS=0,1 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=64 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=64 \
COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL=/absolute/result/m1.jsonl \
COLIBRI_V4_RAW_FIXED_CACHE=1 \
COLIBRI_V4_ROUTED_QD4=1 \
COLIBRI_V4_RESULT_JSONL=/absolute/result/raw-combo.jsonl \
./run-identified.sh run --arch sm_89

# M4 equal-budget local gate. Run once without the RAW flag for the decoded
# arm, then once with COLIBRI_V4_RAW_FIXED_CACHE=1 for the raw arm.
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_IDS=0,1 \
COLIBRI_V4_MAX_LAYERS=6 \
COLIBRI_V4_BENCH_TELEMETRY=1 \
COLIBRI_V4_DEVICE_FIXED_CACHE_GIB=1.2 \
COLIBRI_V4_HOST_FIXED_CACHE_GIB=0 \
COLIBRI_V4_FIXED_CACHE_CANDIDATE_LAYERS=3,4,5 \
COLIBRI_V4_FIXED_CACHE_BASELINE_JSONL=/absolute/result/m1.jsonl \
COLIBRI_V4_RAW_FIXED_CACHE=1 \
COLIBRI_V4_RESULT_JSONL=/absolute/result/m4-local-raw.jsonl \
./run-identified.sh run --arch sm_89

# B2: two independent requests, L3/L4, at least 20 paired repeats
COLIBRI_DSV4_MODEL_DIR="$MODEL_DIR" \
COLIBRI_V4_B2_LAYER_STATIONARY_BENCH=1 \
COLIBRI_V4_B2_REPEATS=20 \
COLIBRI_V4_B2_RESULT_JSONL=/absolute/result/b2.jsonl \
./run-identified.sh run --arch sm_89
```

Live cache placement depends on memory available at preflight. A reproduction
must retain its emitted effective budgets and manifest rather than silently
normalizing them after the run.

## Artifact integrity

Historical M1 through M4 raw JSONL traces are not copied into the repository.
The current QD4 confirmatory packet is tracked in full. The older evidence
archive was stored outside the repository; its device-specific location is
provenance, not a reproduction setting. Each dated directory contained its own
`SHA256SUMS`. Principal trace hashes are reproduced here so the report remains
auditable if that archive is moved:

| Evidence | Principal artifact SHA-256 |
| --- | --- |
| M1 | cold `379a0c052c7ffea9121a58e3a0a116697e6785e36f8a44aeb257c25ace589518`; warm1 `a3eb014c29d5698febc816a818af2276ecc8e79e4417120265c193a59d6b93d8`; warm2 `f9e1b33d3fe37169c0b1591a957a01c7008974bb7cd0192c59a44c759f6aa73d` |
| M2 local | off `c55fb4ff36bd71a0d6b802d0d3254f9d796057b57bfef276723486683004137a`; on `2196682331ca609fe10b732f2fe11f7216dd5bbca1b3561c06565f6f0695f4b2` |
| M3 full61 | run1 `aa3ec1dea7df86413965180c42c12be3aafb0af6823c3683989bb546449e6cde`; run2 `c66a931fe619036872992c0d6d43bb1b8b2feff10cb0593dc27dfac928cf7d63`; run3 `6ee163ace75b2a3691d859a4bba9f0731292e2e9b799bd66dd090c70af222620` |
| Decoded M2+M3 | run1 `3dd28e61738e8d9551efff42be4d1704801b62dd7a39603ba626faf3650dc838`; run2 `2d4739c16bdbcf24c64f192bd2e001fdf1c1ac6e961dcb1c7f82cd193422fdd3`; run3 `cfcc629f96144598ed4606c9b1cb33f2adb0aeb2072e6fc4920683689053b33f` |
| M3-only two-token control | `11f4db9f960e723610a8ca3ed8f62e85bb0983f5321494baa5280786e73cca3b` |
| M4 local raw | decoded `5f0e6be00cc1653bebe4c5eeca10ad6c216bd58e7ea2136d5bbc5fce59783359`; raw `7af65edb0df6cf40c02d718adf1df4c413f0a19b7b8f970833476962076b18dd` |
| M4 full61 | run1 `dc88dbcc9445c2b1398a81c78b25a17f4d36e987aabb128b874dbe060441aaff`; run2 `7d02dc9686a7b2eb86f0796c992b15493a297c563b05a4587777a2ddec6611e7`; run3 `09c30912a74b9a17020a266ac6ee81e26debe17f89ecb6f38e6c1453fd96e07c` |
| B2 | benchmark `bf99662c33f2c30e7acb9582365865cc5344739022df938969cc38b88911308c`; B1 regression `f666565ba54d38780f73dc58b290a04b935a41089017eaef0aac544ea8dfd4f1` |
| QD4 full61 confirmatory packet | protocol `617d7b099e2067a7cf544432a1e146b18630f6c4309d17da2afe527d45806b97`; derived summary `65ce6de19236a591af7aeffd6eab31c2e706663b77e8c81b4f2d0ee6a016993d`; packet SHA256SUMS `53f2c821b53703e32e084c0b1e6175e100d3316bca66158e4f1df2561a5c14ea` |

## Current stage and deferred work

## SM89 warp mHC linear promotion

The retained `mHC` linear kernel maps one row to one SM89 warp: lane 0 keeps
the original BF16 RMS reduction/Newton order and broadcasts its scalar, while
lanes 0--23 keep one original 192-element FMA/block-order output each. The
same-revision scalar kernel remains only as the focused fixture oracle. No
Tensor Core, TF32, Sinkhorn, cache, or persistent-state behavior changed.

The identified full61 A/B gate used fixed tokens `[0,1]`, raw fixed cache,
QD4, live safe cache budgets, and one frozen binary
`df3c66d1f28a61840f700703407c2fe13e7890d5f7769ceeed314641d702a18a`
with source identity
`87ca30e7b4a7529fb87f3675a3c157a09020461e830a13b71bd89c6cd0b6f0f0`.
Three balanced pairs (AB/BA/AB) matched all 122 layer records per pair,
including route IDs/weight bits, attention top-k, hidden/KV, both compressor
states, both final token states, and final logits. Candidate/scalar request
p50/p95 were `78.061 / 79.146 s` and `88.092 / 90.079 s`; token-2 p50/p95
were `10.051 / 10.233 s` and `15.881 / 15.981 s`. All three pairs improved
request and token-2 latency, with no p95 regression. The frozen JSONL and
logs are historical evidence artifacts; callers choose their own result path
for reproduction.

## SM89 sparse-attention score-reuse crossover

The unconditional score-reuse candidate is **not** retained: its three-kernel
split (`scores`, `probabilities`, `values`) was exact in isolation but was
killed as an always-on full61 mechanism. The retained production path is the
SM89 crossover: K<4 is scalar and K>=4 is score reuse. `reference` remains an
explicit `COLIBRI_V4_SPARSE_ATTN_ARM` oracle arm.

An identified CUDA-event sweep on SM89 used the real `128 x 512` L3 HCA and
L4 CSA attention geometry at K=`1,2,4,8,16,32,64,129`, 15 AB/BA pairs per
geometry/K (240 pairs). Each pair closed BF16 output plus raw FP32 score,
probability, denominator, and valid-count bits. Candidate p95 won at every
sampled K. The retained `sm89_crossover` arm
keeps K<4 on scalar and uses score reuse only at K>=4, giving a single,
conservative internal threshold with a short-context reference path.

The post-dispatch 30-pair L3/L4 gate passed all 60 pairs exactly: L3 p50/p95
was `28.869 / 30.858 ms` scalar versus `28.691 / 30.705 ms` dispatch, and L4
was `32.216 / 32.418 ms` versus `32.070 / 32.311 ms`. A fresh full61
reference-only sanity trace recorded only actual top-k widths `1,128`, used
`reference:sparse_attn_v4_bf16` throughout, and reproduced the prior 122-layer
CMEG digest (`5bf394a22abed068e8067f05d767fa99cd46ed3a1c3df600663179682f286a12`)
and final logits/argmax. It is therefore evidence of no reference-oracle
regression.

The final identified full61 gate used the same binary and command, changing
only `COLIBRI_V4_SPARSE_ATTN_ARM`, across AB/BA/AB. All six legs closed 122
layer CMEG records, two token records, and final logits/argmax exactly. The
actual widths were K=1 (61 layer/token calls) and K=128 (61): reference used
scalar 122 times; crossover used scalar 61 times and score reuse 61 times.
Request p50/p95 was `79.523 / 79.807 s` reference versus `75.549 / 78.385 s`
crossover; token-1 was `9.401 / 12.897 s` versus `8.110 / 9.128 s`; summed
attention per token was `1.368 / 2.360 s` versus `1.328 / 2.193 s`. This met
the GO gate, so unset `COLIBRI_V4_SPARSE_ATTN_ARM` now selects crossover.
The gate binary/source identity was
`556ee41ab38ffb69d9d81664fedd06a9d79f926e6a76fd6cf89b523d094c2168` /
`b7bd9c0a36b0b77c2e27d397a1c4725bd3586b7aba18a1f0eaeafeefb3353ff8`.
The production-default-only rebuild is
`6f7f81323f5f9ce3da6b8a1fabd41ee451c99bd513405d35e05268352bae054c` /
`e1c1838d06a775f9d73e2cdde77bcb54a4564f74f2e020d5d0a9165e1ed4afcc`.

Frozen artifacts (binary SHA-256
`556ee41ab38ffb69d9d81664fedd06a9d79f926e6a76fd6cf89b523d094c2168`, source
identity `b7bd9c0a36b0b77c2e27d397a1c4725bd3586b7aba18a1f0eaeafeefb3353ff8`):

| Evidence | SHA-256 |
| --- | --- |
| CUDA-event crossover sweep | `f1425cf39b5c04a623f1ee336a9e78a05fedce377bc81788b90c20e73126a3fb` |
| L3/L4 30-pair dispatch gate | `564ff3795b8b9c96a35f746c8ad736ce8442e2c35cd7cc41813e0b0e4d247769` |
| full61 reference-only sanity | `549ac87449bdb49e6b26889a6b53136051a239f330479f7acc151bac020233f4` |
| full61 crossover AB/BA/AB | pair1 `4e57802e1c9dcf6b1ad95fe658620df1bac59bae92a0ddad365d5f8a732998ef` / `01317cb631063fad76018a44509ada6713260b6c7a0de4e59818697dfc3cc924`; pair2 `94fa367869c10fa8757ad53ac1ae21bd3aca1c33b311c904ae8003c0656cca10` / `7661518dc3b2b79fae77e57f419b94dc37e19376fbe56d2667ccb8afa7181737`; pair3 `7b7f9d58e46f904a61c96ea8a044c4719174c93c3159746a8219d8ff724508a0` / `0f28b1d8bf0d1ca94c3aa0eb9895bb63286486e5dc469441f908994961310fde` |

### ID-only greedy decode: two-step semantic comparison

The ID-only greedy stage is complete as a narrow sequence-mechanics check. Its
frozen [greedy trace](../assets/readme/data/greedy-seed-0-steps-2.jsonl) is
270,221 B and 127 lines (SHA-256
`2552a08a9d273838e200ca5e2ef20bc403950057903e6f1aa0abba84693f5d84`);
the [teacher-forced trace](../assets/readme/data/teacher-forced-seed-0-pred1.jsonl)
is 269,831 B and 126 lines (SHA-256
`75cdf0ddebfbe23a8829325ecce511ee2418736cf50505132504e6d849287677`).
Both record the pinned V4-Pro revision, release build mode, source identity
`65d2f7fc3b9edc0c785ae968084db2c0d8164ea243677af4b88e179bd86cab65`, and
the same exact-cuBLAS hash/version. They record the binary path but not a
binary SHA-256; this is a provenance limitation, not an inferred identity.

Greedy prompt ID `[0]` emitted `[0, 22604, 40341]`. The teacher trace consumed
`[0, 22604]` and its final argmax was `40341`. Each trace has 122 layer records
(layers 0--60 at positions 0 and 1). Their token IDs, route IDs and weights,
attention top-k digest, hidden/KV digest, and both compressor-state digest
families match at all 122 keyed records; the greedy position-1 final also
matches the teacher final logits and argmax.

The recorded request E2E times are 214.973445133 s greedy (114.523327505 s,
100.450117628 s per input token) and 204.169905335 s teacher-forced
(103.687323813 s, 99.265580885 s). They are separate traces, not an ABBA
comparison, so they establish neither a speedup nor tok/s. This is ID-only:
it does not bind a tokenizer or chat template, establish accepted-token or
generation throughput, or test long context. A coarse RSS observation was not
stored in either frozen artifact and makes no memory claim.

### Raw-text greedy bridge: one untemplated step

The frozen [raw-text trace](../assets/readme/data/hello-greedy-1.jsonl) is
136,346 B and 64 lines (SHA-256
`a493ed050e41c30930e1224a59ca181c3627b5a88e98ee7906ba699429567ed1`). It
records the pinned revision, a release build, source identity
`f2054520ed819dc2bf45d7034304633642f4428eacc7ef021841be3fe9d06a9c`, and the
recorded binary path; no binary SHA-256 was captured in the artifact.

The raw tokenizer path has `chat_template=false` and `add_special_tokens=false`.
For input text `Hello`, it records the chain `Hello -> 19923 -> 43883 ->
enment`: `Hello` encoded to ID `19923`, the one 61-layer forward selected ID
`43883`, and raw decoding produced `enment`, giving cumulative text
`Helloenment`. This is one raw, no-template greedy step, not chat behavior or
a quality result. Its recorded 107.860390223 s E2E is one trace, not a tok/s,
throughput, accepted-token, or generation-rate measurement.

The local canonical mechanism stage is complete enough to choose the retained
path: raw fixed device/host caching with GPU E4M3 decode, persistent routed QD4
lanes, and layer-stationary B2 as a bounded throughput mechanism.

Next work is ordered, not implied by this report:

1. **Tokenizer and transactional decode:** bind pinned tokenizer/template data
   and commit verified KV/mHC state across tokens.
2. **Official numerical-policy acceptance:** compare full61 Rust and official
   traces under the documented discrete/numerical policy.
3. **Core kernel migration:** add retained-C parity and same-shape performance
   evidence for the outstanding primitives and critical GEMM/MoE/MLA work.
4. **Second-model-driven abstraction:** only after the V4-Pro sequence, admit
   one pinned second model and implement the smallest justified metadata path.

Long-context, sustained thermal, and accepted-token measurements follow a real
decode loop. Keep the attached B4 gate4 result scoped to its ABBA throughput
estimator. B8, CUDA Graph, QUIC, and federated/multi-worker work remain
separate deferred experiments.
