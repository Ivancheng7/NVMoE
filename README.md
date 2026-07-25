[English](README.md) | [简体中文](README.zh-CN.md)

# localMoE: Want to run a full 1.6T-parameter MoE on your gaming laptop?

Today's laptop result is a DeepSeek-V4-Pro stress test. The pinned model has 1.6T total parameters, 49B activated parameters, and an `805.33 GiB` checkpoint. On a laptop with `12,282 MiB` of VRAM and `32 GB` of RAM, the prototype traversed all 61 layers and produced one real next token at position 5 in `36.979170611 s`, or `0.02704225 token/s`. The evidence tier is **MEASURED DEVELOPMENT**, based on one decode. Comfortable interaction and sustained throughput remain outside the current evidence.

The future product target for home interaction is the smaller DeepSeek-V4-Flash model. Its 24 GiB gate is `0.5 to 1.0 committed token/s` for useful batch and background work. The evidence tier is **TARGET**. Flash support remains unimplemented, and fluid interactive chat remains outside this target.

localMoE is a Rust and CUDA research runtime for this placement problem. The full checkpoint stays on local SSD while a bounded working set moves through RAM and VRAM. Decode, transfer, and compute overlap. Routed contributions follow canonical reduction order, and the separate speculative-state path commits only an accepted prefix.

**MEASURED DEVELOPMENT ROUTED-SCOPE SPEEDUP: `4.117849x`**

At layer 4, the five-position QD4 gate reduced routed-scope median wall time from `486.604659 ms` to `118.169615 ms`. All `4/4` pairs were exact. This is the largest speedup claimed here. Its scope is one local routed-layer mechanism gate. Prompt E2E and token throughput have separate measurements, and production evidence remains pending. This gate ran with fixed prefetch disabled. The [frozen review artifact](docs/assets/readme/data/qd4-stationary-gate/review-boundary.json) records the scope and exclusions.

[See the consumer hardware path](#from-pro-proof-to-flash-product-target) | [Reproduce the evidence](#reproduce-the-evidence) | [Review the evidence map](#evidence-map)

## Why this exists

All 64 pinned Pro shards stay on local SSD. RAM and VRAM serve as bounded working windows.

![The 805.33 GiB V4-Pro checkpoint is much larger than the recorded laptop's 32 GB RAM and 12 GiB VRAM](docs/assets/readme/01-capacity.svg)

| Recorded system | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop, `12,282 MiB` VRAM |
| Host memory | `32 GB` RAM |
| Pro checkpoint | 64 pinned shards, `864,721,029,744 B` actual files |
| Executed scope | Complete 61-layer Rust-canonical mechanism path |

Complete has a narrow meaning here: every pinned source shard is local and the recorded Rust path traverses all 61 layers. Official-reference numerical parity, a general generation runtime, and a finished service remain open.

## What Pro proves

The position-5 run closes one full decode timing ledger. Fixed layer work dominates at `29.606 s`; routed work takes `3.407 s`; the final head takes `1.535 s`. The recorded stages plus a `0.005 s` unattributed gap close exactly to the measured outer span.

![Measured development timing for one V4-Pro next-token decode at position 5 across all 61 layers](docs/assets/readme/16-pos5-one-token-dag.svg)

This run shows that the prototype can drive one real next-token path through the full model under a checked capacity budget. With `n=1`, the reciprocal rate is descriptive. Thermal steady state, latency distributions, and chat speed remain unmeasured.

A separate confirmatory packet tests the persistent four-lane QD4 mechanism on the official single-user `hi` prompt. Across four paired ABBA runs, median prompt E2E fell from `199.431954 s` to `148.555131 s`, a `1.342477723x` ratio of medians. All `4/4` pairs match at 305 layer records plus the final output.

![Four paired full61 QD4 prompt runs with exact semantic records](docs/assets/readme/12-qd4-full61.svg)

That result measures prompt E2E for one Rust-canonical mechanism comparison. Next-token throughput, TTFT, official-reference full61 parity, and other MoE models remain outside its scope.

### What the current full-model path can do

The prototype can load all pinned Pro shards, format and tokenize the official single-user `hi` prompt, execute the complete 61-layer Rust-canonical path, and run a narrow ID-only greedy loop with a two-step teacher-forced check and EOS test. The position-5 pilot retains one bounded state per layer for five prompt tokens plus one generated input.

Useful generated text, system messages, tools, multiturn chat, sampling, long context, sustained throughput, and general model compatibility remain unsupported or unmeasured. Official numerical-policy acceptance against the pinned reference remains open.

## What moving fewer bytes buys

The recent development traces isolate two costs that a home runtime has to remove.

Raw fixed-weight transfer keeps `wo_a` in FP8 across PCIe and decodes it on the GPU. In two paired full61 development runs per arm, fixed H2D fell from `29.062 GB` to `24.968 GB`, saving `4.094 GB` or `14.1%`.

![Measured fixed-weight H2D bytes for host BF16 decode versus raw GPU FP8 decode](docs/assets/readme/13-fixed-h2d-savings.svg)

A strict immutable 32-slot routed cache avoids repeated source loads and H2D misses. In two paired development runs per arm, routed source bytes and routed H2D each fell from `64.223 GB` to `29.620 GB`, saving `34.603 GB` or `53.9%` per field. The independent audit recorded 1,830 routes, 844 loads, 986 hits, maximum occupancy 29, all 61 layers complete, zero fallbacks, and zero worker errors.

![Measured routed source and H2D byte reduction from the strict 32-slot immutable cache](docs/assets/readme/14-routed-cache-savings.svg)

The byte counts are mechanism evidence. The paired timings show where the gains survived: raw GPU decode improved the E2E median by `1.171788x`; the routed cache improved routed-wall median by `1.430045x`; its E2E median improved by `1.068977x`.

![All paired n equals 2 development timings for raw GPU decode and the immutable routed cache](docs/assets/readme/15-development-paired-timings.svg)

Every point and pair line is shown. Each arm has `n=2`, so confidence intervals are unavailable. Production benchmarking remains pending.

## Why Rust

GEMM speed still comes from the kernels. Rust makes disk to CPU decode to PCIe to GPU overlap easier to express and keeps state ownership explicit.

![The bounded SSD to RAM to VRAM execution path used by the Rust prototype](docs/assets/readme/02-execution-path.svg)

| Runtime contract | Concrete mechanism |
| --- | --- |
| A cache slot stays exclusively leased through completion | A move-only lease takes the slot out of the resident map and returns it after job completion, preventing premature or ABA-style reuse |
| Capacity is checked before the run | An exact byte ledger covers the QD4 cache, device state, RoPE, transients, and `2 GiB` headroom; Rust RAII keeps scoped allocations tied to their owners |
| PCIe carries the smaller representation | Raw `wo_a` stays FP8 through H2D and is decoded to BF16 on the GPU |
| Canonical model order survives parallel completion | A bounded pool of four stable worker lanes may finish in any order, then the canonical reducer applies routed expert IDs in ascending order and the shared expert last |
| Hot-path storage keeps stable geometry | Preallocated device state pools and strict reusable host arenas enforce geometry stability across reuse |
| Only accepted speculative rows reach state | Prefix commit checks the authoritative base and commits the accepted rows; the rejected suffix remains outside semantic state |

This is why the rewrite matters. Placement policy can change where bytes wait, while ownership and reduction order keep the accepted computation auditable.

## From Pro proof to Flash product target

Flash is the smaller future product candidate, and its port remains unimplemented. The pinned checkpoint has `284B` total and `13B` activated parameters, 43 layers, 46 shards, and `159,617,149,040 B` of actual shard files. Flash inference runs in this repository: `0`.

![Evidence map separating the measured Pro result, calibrated Flash floor, and future Flash hardware gates](docs/assets/readme/17-hardware-evidence-ladder.svg)

Each row below has its own workload and evidence tier. The Pro row is observed. The Flash rows use Pro development traces for calibration or define hard acceptance targets for future work.

| Desktop VRAM | Model and path | Rate | Evidence boundary |
| ---: | --- | ---: | --- |
| `12 GiB` | Pro, H1 position-5 decode | `0.02704 token/s` | **MEASURED DEVELOPMENT**, one token; sustained rate unmeasured |
| `12 GiB` | Flash, cold H1 | `0.08 to 0.12 token/s` | **CALIBRATED**, Flash run count `0` |
| `16 GiB` | Flash, H1 | `0.25 to 0.40 token/s` | **TARGET**, conditional on unimplemented fixed/head residency |
| `24 GiB` | Flash, H2 at `q about 0.8` | `0.5 to 1.0 committed token/s` | **TARGET**; current-path calibration is only `0.20 to 0.29` |
| `32 to 48 GiB` | Flash, H5 at `q about 0.8` | `0.7 to 1.5 committed token/s` | **TARGET**; route overlap dependent, current-path calibration is only `0.26 to 0.37` |

`H` is the speculative horizon, the number of drafted positions verified together. `q` is the planning assumption that the next drafted token is accepted given that the preceding drafted tokens were accepted. Cache hit rate is a separate metric, and runtime measurement of `q` remains pending. Rejection rollback, draft cost, and the real acceptance distribution can erase the projected gain.

For smaller MoE checkpoints, fewer fixed bytes, routed bytes, and layers make the same placement problem less severe. The method still requires exact per-model metadata, state schedules, routing semantics, and numerical gates. The current runtime is Pro-specific, and compatibility requires a separate implementation and acceptance path.

The `24 GiB` target covers useful full-model batch and background inference on a home workstation. Fluid interactive chat remains outside the target. The first Flash work remains gated on exact metadata and byte closure, one representative routed layer, then exact 43-layer H1 and H2 runs.

## Planned `colibri plan`

The intended product entry point is a hardware planner. Its implementation remains pending, and current builds omit the command.

```bash
# Planned interface. This command does not exist yet.
colibri plan \
  --model deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1 \
  --vram 24GiB --ram HOST_GIB \
  --ssd-read MEASURED_SSD_GBPS --pcie-h2d MEASURED_PCIE_GBPS \
  --context CONTEXT_TOKENS --concurrency REQUESTS \
  --spec-horizon H --acceptance Q
```

Its inputs are the exact model revision, VRAM, RAM, measured SSD direct-read bandwidth, measured PCIe H2D bandwidth, context, concurrency, `H`, and `q`. Its output should be a placement plan with mandatory bytes, capacity reserve and margin, the expected bottleneck, and a rate tier labeled **MEASURED**, **CALIBRATED**, or **TARGET**.

Until that command exists, use the evidence quickstart below to check what is real.

## Reproduce the evidence

Use WSL2/Linux, CUDA 12.x, LLVM/NVPTX, libclang, and an Ada `sm_89` GPU. Put the checkpoint on WSL-native ext4. The environment contract rejects `/mnt/c` and `/mnt/d` for model storage. Set `COLIBRI_DSV4_MODEL_DIR`, `COLIBRI_ORACLE_PYTHON`, and `COLIBRI_CUBLAS_EXACT_LIB` as described in the [full61 environment contract](rust/colibri-cuda/tools/V4_FULL61_ORACLE.md).

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust
cargo oxide doctor
cargo fmt --check
cargo check --locked -p colibri-cuda
./run-identified.sh test --arch sm_89
```

Validate the compact evidence envelopes, then regenerate figures `13` through
`17` if the identified raw development logs are available:

```bash
cd ..
python -m json.tool docs/assets/readme/data/qd4-full61/confirmatory/summary.json >/dev/null
python -m json.tool docs/assets/readme/data/qd4-stationary-gate/review-boundary.json >/dev/null
python docs/assets/readme/generate_cost_figures.py \
  --artifact-root /mnt/colibri-native/.logs \
  --check
```

The repository carries compact review and summary envelopes. Bulky raw traces
stay external. The cost-figure command requires those identified raw development
logs locally. Their expected paths, SHA-256 identities, source fields,
formulas, units, and evidence classes are frozen in
[`cost-figures-provenance.json`](docs/assets/readme/cost-figures-provenance.json).
The generator rejects an identity mismatch and renders twice before accepting
deterministic output.

The [mechanism experiment record](docs/experiments/v4-pro-mechanisms-2026-07-24.md) is authoritative for exact model-run commands and the fail-closed preflight.

## Evidence map

| Evidence | Role |
| --- | --- |
| [Position-5 trace identity and timing formulas](docs/assets/readme/cost-figures-provenance.json) | One real next-token development decode |
| [Raw fixed and cache development provenance](docs/assets/readme/cost-figures-provenance.json) | Paired `n=2` byte and timing figures; CI unavailable |
| [QD4 confirmatory summary](docs/assets/readme/data/qd4-full61/confirmatory/summary.json) | Four paired prompt-E2E runs |
| [QD4 L3/L4 support gate](docs/assets/readme/data/qd4-stationary-gate/review-boundary.json) | Local routed-scope mechanism gate, excluded from full61 statistics |

The figures `13` through `17` are generated from frozen raw traces and pinned target metadata. Their provenance file records trace hashes and calculations for measured values, then marks the future hardware bands as hardcoded **TARGET** gates. Their predictive formula remains undefined. The Flash ladder gives measured current evidence and future gates separate rate frames.

Ratios from different lineages remain separate. The `4.117849x` routed-layer gate, the `1.342477723x` paired prompt result, and the `0.02704225 token/s` position-5 observation answer different questions.

## Repository layout

```text
rust/colibri-cuda/          Rust + cuda-oxide V4-Pro prototype
rust/colibri-cuda/tools/    oracle, trace, and evidence tools
docs/assets/readme/         deterministic figures and tracked artifacts
c/                          retained upstream C/CUDA oracle and baseline
```

## Research status

This repository is a research prototype. Current work targets one pinned DeepSeek-V4-Pro revision on one Ada laptop. Multi-family weights, official-reference full61 acceptance, generation quality, long context, sustained throughput, and service behavior remain open.

Earlier layer-stationary, expert-slot reuse, phase-profile, and fixed-prefetch experiments remain in the technical report and experiment record. Their estimands and frozen source identities differ, so their ratios are kept separate.

## License

Code remains under [Apache License 2.0](LICENSE). Model weights and their associated materials follow the terms on the model release page. Retained upstream code and assets keep their existing notices and license terms.

## Inspiration

localMoE was inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri). We are taking its local MoE ambition forward through exact streamed execution, explicit Rust/CUDA ownership, and consumer-hardware performance work.
