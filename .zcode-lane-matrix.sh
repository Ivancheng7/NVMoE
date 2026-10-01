#!/usr/bin/env bash
set -euo pipefail
TREE=/srv/localmoe/build/localmoe-clean-git
BIN=/srv/localmoe/targets/lane-contention-4b0c6dc-target/release/colibri-cuda
OUT=/srv/localmoe/targets/lane-contention-1001-r1
MODEL=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb
CUBLAS=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5
mkdir -p "$OUT"
cd "$TREE"
for pair in 0:a 3:a 2:a 1:a 6:a 0:b; do
  width=${pair%:*}
  run=${pair#*:}
  label=width${width}-${run}
  printf '%s width=%s start\n' "$(date -Iseconds)" "$width"
  env COLIBRI_DSV4_MODEL_DIR="$MODEL" COLIBRI_CUBLAS_EXACT_LIB="$CUBLAS" \
    COLIBRI_V4_FLASH_BASE=1 COLIBRI_V4_FLASH_TOKEN_IDS=0 COLIBRI_V4_GREEDY_STEPS=8 \
    COLIBRI_V4_STREAM_READ_MODE=buffered COLIBRI_V4_FLASH_PREFETCH=1 \
    COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
    COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice COLIBRI_V4_FLASH_FP4_ACCUM_ARM=block32 \
    COLIBRI_V4_FLASH_FINAL_HEAD_ARM=cpu COLIBRI_V4_BENCH_TELEMETRY=1 \
    COLIBRI_V4_FLASH_FANOUT_GPU_WINDOW="$width" COLIBRI_V4_RESULT_JSONL="$OUT/$label.jsonl" \
    "$BIN" >"$OUT/$label.stdout" 2>"$OUT/$label.stderr"
  printf '%s width=%s finished\n' "$(date -Iseconds)" "$width"
done
