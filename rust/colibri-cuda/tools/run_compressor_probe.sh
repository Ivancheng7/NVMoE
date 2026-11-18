#!/usr/bin/env bash
# 压缩器两档：连着跑 N 次同样的 8 步，两次探针都开，事后按档分组比对。
#   usage: run_compressor_probe.sh <commit-ish> [tag] [repeats]
set -uo pipefail

if ps -eo comm | grep -qE '^(colibri-cuda|cargo)$'; then
  echo "refusing to run: another colibri-cuda or cargo is on the machine"
  ps -eo pid,etime,comm | grep -E 'colibri-cuda|cargo' | head -5
  exit 9
fi

COMMIT=${1:?commit required}
TAG=${2:-$COMMIT}
REPEATS=${3:-5}
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$TOOLS/../../.." && pwd)
Z=${VERIFY_DIR:-$TOOLS/verify}
TREE=${TREE:-$REPO_ROOT}
BRANCH=${BRANCH:-main}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$TREE/rust/target}
BIN=$CARGO_TARGET_DIR/release/colibri-cuda
MODEL=${COLIBRI_DSV4_MODEL_DIR:?set COLIBRI_DSV4_MODEL_DIR to the Flash-0731 checkpoint on ext4}
CUBLAS=${COLIBRI_CUBLAS_EXACT_LIB:-/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5}
OUT_ROOT=${OUT_ROOT:-$CARGO_TARGET_DIR/bench-runs}
OUT=$OUT_ROOT/compressorprobe-$TAG
mkdir -p "$OUT"
# Fresh output directory only: the engine refuses to overwrite a result jsonl.
if ls "$OUT"/*.jsonl >/dev/null 2>&1; then
  echo "refusing to run: $OUT already holds result files, pick another tag"
  exit 9
fi
exec > >(tee "$OUT/batch.log") 2>&1

echo "commit=$COMMIT  tag=$TAG  repeats=$REPEATS  out=$OUT"

echo "################ 1/3 sync and build ################"
git -C "$TREE" fetch --quiet origin "$BRANCH" 2>/dev/null || true
git -C "$TREE" checkout --quiet --detach "$COMMIT" || exit 1
git -C "$TREE" log --oneline -1 | cat
git -C "$TREE" status --porcelain | head -3
export RUSTC_WRAPPER="$TREE/rust/rustc-human-wrapper.sh"
( cd "$TREE/rust" && cargo oxide build --arch sm_120 -- --release ) \
  > /tmp/build_compressorprobe.log 2>&1
status=$?
echo "build exit=$status  warnings=$(grep -c '^warning' /tmp/build_compressorprobe.log)"
if [ $status -ne 0 ]; then
  tail -60 /tmp/build_compressorprobe.log
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
      COLIBRI_V4_FLASH_COMPRESSOR_PROBE=1 \
      COLIBRI_V4_BENCH_TELEMETRY="$telemetry" \
      COLIBRI_DSV4_MODEL_DIR="$MODEL" \
      COLIBRI_CUBLAS_EXACT_LIB="$CUBLAS" \
      COLIBRI_V4_RESULT_JSONL="$OUT/$label.jsonl" \
      "$@" \
      "$BIN" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr"
  )
  echo "exit=$?  $(grep -E 'Elapsed \(wall' "$OUT/$label.stderr" | head -1)"
  grep -E 'Maximum resident' "$OUT/$label.stderr" | head -1
}

run 1step 1 0
for i in $(seq 1 "$REPEATS"); do
  run 8step-r$i 8 1
done

echo "################ 2/3 pinned values (1 step) ################"
python3 "$Z/verify_baseline.py" "$OUT/1step.jsonl" | tail -12

echo "################ 3/3 compressor segments per run ################"
python3 "$TOOLS/compressor_segments.py" "$OUT"/8step-r*.jsonl

echo "################ result files ################"
ls -l "$OUT"
