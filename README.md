[English](README.md) | [简体中文](README.zh-CN.md)

# localMoE: Want to try a full 1400B-class MoE on your gaming laptop?

localMoE runs a complete DeepSeek-V4-Pro checkpoint on an RTX 4080 Laptop GPU with 12 GB of VRAM. The pinned model is a 1.6T-parameter MoE with official mixed FP4-FP8 weights totaling about 864.7 GB. Its 64 shards remain on local SSD while the Rust runtime moves a bounded working set through RAM and VRAM.

The current P6 development trace completed all 61 layers and emitted token `115287` in `14.789027100 s/token`, or `0.0676177 token/s`. The full61 audit at P4 through P6 returned `PASS`; the certified GPU argmax needed no CPU fallback. This is one development trace (`n=1`). It has no confidence interval, and cache state and major-fault conditions were not controlled.

![P6 full61 development throughput across three versions of the same timing estimand](docs/assets/readme/18-p6-throughput.svg)

The chart uses one P6 development trace per condition and one timing estimand: full61 next-token wall time. The combined path is `1.1605x` faster than the pre-pipeline trace by this descriptive comparison. It does not establish a causal effect, sustained throughput, or an upstream C++ comparison.

## What runs today

| Recorded configuration | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop, `12,282 MiB` VRAM |
| Host memory | `32 GB` RAM |
| Checkpoint | 64 pinned shards, `864,721,029,744 B` of model files |
| Model path | Complete 61-layer DeepSeek-V4-Pro Rust execution |
| Current P6 trace | `14.789027100 s/token`, token `115287`, full61 audit P4-P6 `PASS` |

The implementation executes an exact, narrow full-model path for the recorded prompt and greedy token step. General chat, sampling, long context, sustained thermals, service behavior, and official-reference parity remain open evaluation work.

## Why the Rust path matters

The runtime keeps the data movement visible in code.

- Explicit resource ownership holds device and host buffers for the lifetime of each request.
- A pinned double-buffer worker prepares fixed layer inputs while the current layer computes.
- Metadata-driven bounded readahead advises the following layer and records a skip when the advisory path cannot run.
- Certified GPU argmax accepts a certificate when its interval is decisive, then uses the canonical CPU path when it is not.
- Colibri-owned CUDA kernels are implemented with `cuda-oxide`.

These mechanisms let the runtime place a checkpoint far larger than laptop memory without relaxing the recorded execution order or capacity checks.

## P6 benchmark

All three values below use the same P6 full61 next-token timing estimand. Each condition has `n=1`; the table is descriptive and has no CI.

| P6 condition | Seconds per token | Tokens per second |
| --- | ---: | ---: |
| Pre-pipeline | `17.162779092` | `0.0582668` |
| Pinned pipeline | `15.852466305` | `0.0630817` |
| Combined path | `14.789027100` | `0.0676177` |

## Future targets

DeepSeek-V4-Flash support and a home-platform 1400B-class deployment target are future work. They have no measured result in this README. Any new model family needs its own metadata, capacity ledger, semantic checks, and benchmark evidence.

## Reproduce the current path

Use WSL2/Linux, CUDA 12.x, LLVM/NVPTX, libclang, and an Ada `sm_89` GPU. Store the checkpoint on WSL-native ext4. The environment contract rejects `/mnt/c` and `/mnt/d` for model storage. Set `COLIBRI_DSV4_MODEL_DIR`, `COLIBRI_ORACLE_PYTHON`, and `COLIBRI_CUBLAS_EXACT_LIB` as described in the [full61 environment contract](rust/colibri-cuda/tools/V4_FULL61_ORACLE.md).

```bash
git clone https://github.com/andyzpb/localMoE.git
cd localMoE/rust
cargo oxide doctor
cargo fmt --check
cargo check --locked -p colibri-cuda
./run-identified.sh test --arch sm_89
```

The [mechanism experiment record](docs/experiments/v4-pro-mechanisms-2026-07-24.md) contains the exact model-run commands and fail-closed preflight. The repository also carries the compact [evidence envelope](docs/assets/readme/data/evidence-v1.json) for review.

## Repository layout

```text
rust/colibri-cuda/          Rust + cuda-oxide V4-Pro prototype
rust/colibri-cuda/tools/    oracle, trace, and evidence tools
docs/assets/readme/         public figures and compact evidence artifacts
c/                          retained upstream C/CUDA oracle and baseline
```

## License

Code remains under [Apache License 2.0](LICENSE). Model weights and associated materials follow the terms on the model release page. Retained upstream code and assets keep their existing notices and license terms.

## Inspiration

Inspired by [JustVugg/colibri](https://github.com/JustVugg/colibri) and pushing further.
