#!/usr/bin/env bash
# Three-arm interleaved A/B: pre-change baseline, routed-only (pinned lanes),
# both (pinned lanes + attention device chain). Rotating so drift hits each arm
# equally; cold cache before every run; fails loudly if an arm writes nothing.
# Usage: run_three_arm_ab.sh <rounds> <tag>
set -u
ROUNDS="${1:-3}"
TAG="${2:-3arm}"
BASE=/srv/localmoe/targets/localmoe-baseline-routed-target/release/colibri-cuda
ROUTED=/srv/localmoe/targets/localmoe-pinned-target/release/colibri-cuda
BOTH=/srv/localmoe/targets/localmoe-both-changes-target/release/colibri-cuda
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
for r in $(seq 1 "$ROUNDS"); do
  for arm in baseline routed both; do
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    env_for
    case "$arm" in
      baseline) BIN="$BASE" ;;
      routed)   BIN="$ROUTED" ;;
      both)     BIN="$BOTH" ;;
    esac
    RESULT="$OUT/three-${TAG}-${arm}-r${r}.jsonl"
    rm -f "$RESULT"
    export COLIBRI_V4_RESULT_JSONL="$RESULT"
    if ! "$BIN" >"$OUT/three-${TAG}-${arm}-r${r}.stdout" 2>&1 || [ ! -s "$RESULT" ]; then
      echo "FAIL run $r $arm"; tail -3 "$OUT/three-${TAG}-${arm}-r${r}.stdout"; exit 1
    fi
    echo "run $r $arm ok"
  done
done
