# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `make_glm_oracle.py`, `make_glm_bench_model.py`: deterministic fixtures
- `benchmark_cuda_fixture.py`, `bench_ux.sh`, `laptop_benchmark.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `gen_unicode.py`: tokenizer table generation

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/colibri-bench
```

## Laptop A/B capture

`laptop_benchmark.py` wraps an existing benchmark without changing it. It writes
`run.json`, raw `samples.jsonl`/stdout/stderr, and 1 Hz `telemetry.jsonl`
(power, temperature, clocks, VRAM, utilization). Supply all fair-comparison
fields every time: implementation, checkpoint hash or manifest, quant, prompt
IDs, batch, context, cold/warm state, phase, and power state. `--kv-layout`
and the paired `--active-experts`/`--total-experts` fields cover MLA+MoE
checkpoints without naming a model family.

```sh
cd c
python3 tools/laptop_benchmark.py --out results/c-cuda-warm \
  --implementation c-cuda --model /tmp/colibri-bench \
  --model-manifest /tmp/colibri-bench/bench_manifest.json --quant bf16 \
  --prompt-ids /tmp/colibri-bench/ref_glm.json --batch 1 --context 4 \
  --cache-state warm --phase decode --power-state ac-performance -- \
  python3 tools/benchmark_cuda_fixture.py --model /tmp/colibri-bench --gpu 0
```

Use `--nsys` or `--ncu` in a separate capture when profiling; neither is a
timing run. `--no-telemetry` is only for CPU-only smoke tests.

## V4 ext4 read matrix

`v4_io_matrix.py` runs the fixed `35/210/421 MiB × 1/2/4/8 threads ×
direct/cached` matrix and writes both JSONL and CSV. Threads approximate
concurrent range reads; they are not an io_uring queue-depth measurement.
`direct_uncached` means successful `O_DIRECT`, not a claimed globally cold
page cache. Cached cells get one unmeasured matching warmup.

```sh
make iobench
python3 tools/v4_io_matrix.py \
  --model-dir /mnt/colibri-native/deepseek-v4-pro-b5968e9190ef611bbf34a7229255be88a0e937c1 \
  --device /dev/sde \
  --jsonl /mnt/colibri-native/.logs/v4-io-matrix.jsonl \
  --csv /mnt/colibri-native/.logs/v4-io-matrix.csv
```

The driver exits before reading weights unless the model is on the requested
ext4 device and the pinned index, 64 regular shard headers, tensor mapping, and
`864721029744` shard bytes all agree.
