#!/usr/bin/env bash
# Reversed-order A/B: pinned arm runs FIRST this time, to test whether the
# shared-window growth seen with baseline-first was an ordering artifact.
set -u
BASE=/srv/localmoe/targets/localmoe-baseline-routed-target/release/colibri-cuda
PIN=/srv/localmoe/targets/localmoe-pinned-target/release/colibri-cuda
OUT=/srv/localmoe/targets/io-ceiling
env_for() {
  export COLIBRI_DSV4_MODEL_DIR=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb
  export COLIBRI_CUBLAS_EXACT_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5
  export COLIBRI_V4_FLASH_BASE=1
  export COLIBRI_V4_FLASH_TOKEN_IDS=0
  export COLIBRI_V4_GREEDY_STEPS=8
  export COLIBRI_V4_STREAM_READ_MODE=buffered
  export COLIBRI_V4_FLASH_PREFETCH=1
  export COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas
  export COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas
  export COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice
  export COLIBRI_V4_FLASH_FP4_ACCUM_ARM=block32
  export COLIBRI_V4_FLASH_FINAL_HEAD_ARM=gpu_exact
}
for r in $(seq 1 2); do
  for arm in pinned baseline; do
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    env_for
    if [ "$arm" = baseline ]; then BIN="$BASE"; else BIN="$PIN"; fi
    export COLIBRI_V4_RESULT_JSONL="$OUT/pinned-rev-${arm}-r${r}.jsonl"
    "$BIN" >/dev/null 2>&1
    echo "rev run $r $arm done"
  done
done
