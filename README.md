[English](README.md) | [简体中文](README.zh-CN.md)

# Want to Try Deploying a Full 1.6T MoE Model on Your Gaming Laptop?

Does Your Next LLM Infrastructure Really Need to Be an H100?

| **1.6T parameters** | **12GB VRAM** | **3.58x faster** |
| --- | --- | --- |
| Full 810GiB model directory | Exact bounded GPU working set | 198.948s to 55.550s |

![Exact full-request speedup](docs/assets/readme/19-full-request-speedup.svg)

localMoE runs the complete DeepSeek-V4-Pro checkpoint from a WSL-native ext4 volume and keeps only a bounded working set in RAM and VRAM. On the measured laptop, the accepted P5/N3 candidate completed a full request in a geometric mean of `55.550s`, down from the recorded `198.948s` exact baseline. That is a descriptive `3.58x` cross-lineage speedup and a `72.1%` reduction in wall time.

The final candidate itself passed a sealed ABBA gate with `n=2` per configuration. Every arm emitted the same three token IDs and the same normalized BF16 digest sequence.

## The 1.6T problem on a 12GB GPU

An MoE activates a sparse subset of experts for each token. V4-Pro still crosses 61 serial layers, and every layer combines fixed work with router-selected expert payloads. The full expert set must remain addressable even though only a small fraction can live on the GPU.

localMoE turns that constraint into a tiered execution path:

`ext4 SSD -> CPU read and decode -> pinned RAM -> retained VRAM -> canonical GPU reduction -> transactional state commit`

![Bounded SSD, RAM, and VRAM execution path](docs/assets/readme/02-execution-path.svg)

The runtime prepares future payloads while the GPU consumes the current layer. An immutable expert cache loads each prefill expert union once, host retention keeps more fixed payloads away from storage, and device retention removes another part of the H2D path. The route order and state transition remain canonical throughout the request.

![Checkpoint capacity versus laptop memory tiers](docs/assets/readme/01-capacity.svg)

## What the sealed experiments support

All four gates use full61 exact execution. The three single-request configurations use P5/N3 and two complete arms in the frozen `A1, B1, B2, A2` order. The B2 packet uses two independent position-0 requests, with two repeats in each order stratum.

| Sealed mechanism gate | Candidate change | Symmetric E2E ratio | Registered timing gate |
| --- | --- | ---: | ---: |
| Immutable expert cache | cache off to 32 slots | `0.7007` | TTFT `0.7901`, hot total `0.6328` |
| Host retention | `12GiB` to `13GiB` | `0.8946` | TTFT `0.8862`, hot total `0.9039` |
| Device retention | `3.5GiB` to `4.0GiB` | `0.9260` | TTFT `0.9268`, hot total `0.9248` |
| B2 row set | serial two-request execution to a layer-stationary row set | `1.9417x` aggregate completion throughput | `2.0413x`, `1.8469x` by order; 4/4 exact pairs |

The cache gate cut physical expert loads from `2,562` to `1,576` and removed `34.603GB` from both routed storage reads and routed H2D traffic. Moving to `13GiB` host retention removed another `2.563GB` of fixed source reads. Moving to `4.0GiB` device retention removed `1.709GB` of fixed source reads and `1.568GB` of fixed H2D traffic.

The sealed N6 device-retention overlay reduced the measured per-token interval from `17.8409426951s` to `16.5992631988s`, a change of `-1.2416794963s/token` (`6.96%`). Per token, it avoided `1,708,974,176 B` of source reads and `783,919,152 B` of H2D traffic while preserving the engine's declared exact semantics. This estimate has `n=2` per configuration. Paired drift bounds the supported claim to the recorded P5/N3 setup.

### Fixed work shared across two independent requests

![B2 row-set aggregate throughput](docs/assets/readme/21-b2-rowset-full61.svg)

The sealed B2 packet runs two independent L0-L60 requests. Each keeps its router metadata and routed payloads. The row set shares fixed work. Across four exact ABBA pairs, aggregate completion throughput had a symmetric geometric mean of `1.9417x`; the serial-then-stationary and stationary-then-serial strata were `2.0413x` and `1.8469x`.

Per pair, fixed source traffic fell from `58,123,632,496 B` to `29,061,816,248 B`, and fixed H2D fell from `66,310,423,408 B` to `33,155,211,704 B`. Router metadata remained `288 B` per request, while routed source and H2D traffic each remained `25,689,194,496 B`. Single-request completion latency remains outside this result.

The current product B2 mode passes the L0 strict gate. This evidence chain joins that product check to a sealed full61 mechanism measurement. A full product B2 E2E rerun remains pending. Resource closure is weak: GPU telemetry was `1,241 MiB` before and after, GNU time recorded `925,472 KiB` maximum RSS with zero swaps, and system SwapFree declined `27,688 KiB`. The continuous child RSS/VmSwap trace is missing because of a wrapper sampler defect.

Use two semicolon-separated token-ID lists and set greedy steps explicitly:

```bash
COLIBRI_V4_B2_GENERATE=1 \
COLIBRI_V4_B2_TOKEN_IDS='0;1' \
COLIBRI_V4_GREEDY_STEPS=1 \
cargo run --manifest-path rust/Cargo.toml --release -p colibri-cuda
```

## Expected timing on the measured laptop

P5/N3 means a five-token prompt followed by three emitted tokens. The final accepted configuration uses `13GiB` host retention and `4.0GiB` device retention.

| P5/N3 measurement | Geometric center | Derived rate |
| --- | ---: | ---: |
| TTFT | `27.631s` | one first emitted token |
| Two decode intervals | `27.713s` total | `0.0722tok/s` |
| Complete request E2E | `55.550s` | `0.0540 emitted tok/s` |

The decode rate comes from two measured intervals. It is short-run evidence, and sustained generation needs a longer benchmark. The E2E number covers the whole recorded request, including TTFT and both decode intervals.

The measured starting configuration is:

| Resource | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU, `12,282MiB` VRAM |
| Host | `32GB` RAM |
| WSL2 | about `24GiB` visible memory |
| Retention | `13GiB` host and `4.0GiB` device budgets |
| Model storage | WSL-native ext4, at least `810GiB` for the model directory |

This measured configuration is the current reference point. General minimum requirements still need independent validation. Leave additional storage for builds and result artifacts.

## What exact means here

`exact` refers to the Rust engine's declared canonical contract. The single-request gates at p4, p5, and p6 emitted token `115287` with normalized BF16 digests `519e76cfedc01504`, `7017c61decbc4151`, and `3bed9be0e5ebd27c`. The sealed B2 packet matched each independent request's 61-layer records and final argmax plus collapsed, normalized, and logits digests within its frozen lineage. Route order, reductions, KV updates, lease release, and transactional state commits also passed their recorded gates.

Official PyTorch full-model parity, long-context behavior, text quality across prompts, sampling, and sustained throughput remain separate acceptance targets. The headline `3.58x` compares different source lineages, so it is descriptive evidence. The three single-request ABBA ratios and the frozen B2 aggregate ratio carry their respective paired mechanism claims.

## Why Rust

Rust gives the runtime useful constraints at the language boundary. Move-only leases keep cache slots owned until GPU work releases them. RAII closes pinned-memory, CUDA-event, and device-buffer lifetimes on every exit path. Typed FP4 and FP8 layouts make byte budgets explicit. Bounded channels and CUDA streams overlap storage, CPU decode, H2D, and GPU execution without relaxing canonical reduction order.

The result is a resource scheduler whose ownership rules are inspectable in code, while GEMM performance still comes from CUDA kernels and the GPU.

## Quickstart

The current path targets WSL2/Linux, CUDA 12.x, LLVM/NVPTX, libclang, and Ada `sm_89`. Store model weights on a WSL-native ext4 block device. The environment contract rejects `/mnt/c` and `/mnt/d` for model storage.

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust
cargo oxide doctor
cargo fmt --check
cargo check --locked -p colibri-cuda
./run-identified.sh test --arch sm_89
```

Code is available under the [Apache License 2.0](LICENSE). Model weights follow their own release terms.

## Inspiration

localMoE was inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri) and continues that line of work toward exact, full-weight MoE inference on home hardware.
