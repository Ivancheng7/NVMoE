#!/usr/bin/env bash
# 一次跑完：同步、构建、开 shared 探针跑默认帧和关预取帧，再看相位账。
#   usage: run_shared_probe.sh <commit-ish> [tag]
set -uo pipefail

if ps -eo comm | grep -qE '^(colibri-cuda|cargo)$'; then
  echo "refusing to run: another colibri-cuda or cargo is on the machine"
  ps -eo pid,etime,comm | grep -E 'colibri-cuda|cargo' | head -5
  exit 9
fi

COMMIT=${1:?commit required}
TAG=${2:-$COMMIT}
Z=/mnt/c/Users/27236/Desktop/localmoe-clean/.zcode
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=/srv/localmoe/build/merged-prompt-chunk
export CARGO_TARGET_DIR=/srv/localmoe/targets/merged-prompt-chunk-target
BIN=$CARGO_TARGET_DIR/release/colibri-cuda
MODEL=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb
CUBLAS=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5
OUT=/srv/localmoe/targets/sharedprobe-$TAG
mkdir -p "$OUT"
exec > >(tee "$OUT/batch.log") 2>&1

echo "commit=$COMMIT  tag=$TAG  out=$OUT"

echo "################ 1/3 sync and build ################"
git -C "$TREE" fetch --quiet origin codex/business-restart || exit 1
git -C "$TREE" checkout --quiet --detach "$COMMIT" || exit 1
git -C "$TREE" log --oneline -1 | cat
git -C "$TREE" status --porcelain | head -3
export RUSTC_WRAPPER="$TREE/rust/rustc-human-wrapper.sh"
( cd "$TREE/rust" && cargo oxide build --arch sm_120 -- --release ) \
  > /tmp/build_sharedprobe.log 2>&1
status=$?
echo "build exit=$status  warnings=$(grep -c '^warning' /tmp/build_sharedprobe.log)"
if [ $status -ne 0 ]; then
  tail -60 /tmp/build_sharedprobe.log
  exit $status
fi
sha256sum "$BIN"

run() { # label steps telemetry [extra VAR=value ...]
  local label=$1 steps=$2 telemetry=$3
  shift 3
  echo "################ run $label  steps=$steps  $* ################"
  (
    cd "$TREE" || exit 1
    /usr/bin/time -v env \
      COLIBRI_V4_FLASH_BASE=1 \
      COLIBRI_V4_FLASH_TOKEN_IDS=0 \
      COLIBRI_V4_GREEDY_STEPS="$steps" \
      COLIBRI_V4_STREAM_READ_MODE=buffered \
      COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas \
      COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
      COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice \
      COLIBRI_V4_FLASH_SHARED_PROBE=1 \
      COLIBRI_V4_BENCH_TELEMETRY="$telemetry" \
      COLIBRI_DSV4_MODEL_DIR="$MODEL" \
      COLIBRI_CUBLAS_EXACT_LIB="$CUBLAS" \
      COLIBRI_V4_RESULT_JSONL="$OUT/$label.jsonl" \
      "$@" \
      "$BIN" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr"
  )
  echo "exit=$?  $(grep -E 'Elapsed \(wall' "$OUT/$label.stderr" | head -1)"
  grep -E 'Maximum resident' "$OUT/$label.stderr" | head -1
  echo "SHAREDPROBE lines: $(grep -c SHAREDPROBE "$OUT/$label.stderr")"
}

run 1step-sharedprobe 1 0
run 8step-sharedprobe 8 1
run 8step-sharedprobe-noprefetch 8 1 COLIBRI_V4_FLASH_PREFETCH=0

echo "################ 2/3 pinned values (1 step) ################"
python3 "$Z/verify_baseline.py" "$OUT/1step-sharedprobe.jsonl" | tail -12

echo "################ 3/3 shared window summary ################"
python3 "$TOOLS/shared_window_sum.py" "$OUT/8step-sharedprobe.stderr" \
  "$OUT/8step-sharedprobe-noprefetch.stderr"

echo "################ result files ################"
ls -l "$OUT"
