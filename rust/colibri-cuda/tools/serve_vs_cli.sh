#!/usr/bin/env bash
# 服务面与命令行的同一批对照：同一个二进制、交错跑，两个面都量稳态每 token。
#
#   usage: serve_vs_cli.sh <commit-ish> [tag] [rounds]
#
# 服务那面用 tools/flash_serve_client.py（它自己设 COLIBRI_V4_FLASH_SERVE=1，
# 不设 FP4_ACCUM_ARM / FINAL_HEAD_ARM，所以两个交付面的数值臂都是代码默认）。
# 命令行的 8 步用状态页那套配方。两面的稳态定义都是"扣掉第一个 token"，因为
# 那个 token 付的是冷页缓存（命令行）或预填充（服务）。
set -uo pipefail

if ps -eo comm | grep -qE '^(colibri-cuda|cargo)$'; then
  echo "refusing to run: another colibri-cuda or cargo is on the machine"
  ps -eo pid,etime,comm | grep -E 'colibri-cuda|cargo' | head -5
  exit 9
fi

COMMIT=${1:?commit required}
TAG=${2:-$COMMIT}
ROUNDS=${3:-3}
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
OUT=$OUT_ROOT/servevscli-$TAG
mkdir -p "$OUT"
if ls "$OUT"/*.log >/dev/null 2>&1; then
  echo "refusing to run: $OUT already holds logs, pick another tag"
  exit 9
fi
exec > >(tee "$OUT/batch.log") 2>&1

echo "commit=$COMMIT  tag=$TAG  rounds=$ROUNDS  out=$OUT"

echo "################ 1/3 sync and build ################"
git -C "$TREE" fetch --quiet origin "$BRANCH" 2>/dev/null || true
git -C "$TREE" checkout --quiet --detach "$COMMIT" || exit 1
git -C "$TREE" log --oneline -1 | cat
git -C "$TREE" status --porcelain | head -3
export RUSTC_WRAPPER="$TREE/rust/rustc-human-wrapper.sh"
( cd "$TREE/rust" && cargo oxide build --arch sm_120 -- --release ) \
  > /tmp/build_servevscli.log 2>&1
status=$?
echo "build exit=$status  warnings=$(grep -c '^warning' /tmp/build_servevscli.log)"
if [ $status -ne 0 ]; then
  tail -60 /tmp/build_servevscli.log
  exit $status
fi
sha256sum "$BIN"

serve() { # label
  local label=$1
  echo "################ serve $label  $(date +%H:%M:%S) ################"
  python3 "$TOOLS/flash_serve_client.py" \
    --binary "$BIN" \
    --model-dir "$MODEL" \
    --cublas-lib "$CUBLAS" \
    --prompt-json '"\u4f60\u597d\uff0c\u4f60\u662f\u8c01\uff1f"' \
    --max-tokens 8 --repeat 1 \
    > "$OUT/serve-$label.log" 2>&1
  echo "exit=$?  $(date +%H:%M:%S)"
  grep -E "^(stats|request|wall|token)" "$OUT/serve-$label.log" | head -4
}

cli() { # label
  local label=$1
  echo "################ cli $label  $(date +%H:%M:%S) ################"
  (
    cd "$TREE" || exit 1
    /usr/bin/time -v env \
      COLIBRI_V4_FLASH_BASE=1 \
      COLIBRI_V4_FLASH_TOKEN_IDS=0 \
      COLIBRI_V4_GREEDY_STEPS=8 \
      COLIBRI_V4_STREAM_READ_MODE=buffered \
      COLIBRI_V4_FLASH_MHC_LINEAR_ARM=cublas \
      COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM=cublas \
      COLIBRI_V4_FLASH_ROUTER_SCORE_ARM=libdevice \
      COLIBRI_V4_BENCH_TELEMETRY=1 \
      COLIBRI_DSV4_MODEL_DIR="$MODEL" \
      COLIBRI_CUBLAS_EXACT_LIB="$CUBLAS" \
      COLIBRI_V4_RESULT_JSONL="$OUT/cli-$label.jsonl" \
      "$BIN" > "$OUT/cli-$label.stdout" 2> "$OUT/cli-$label.stderr"
  )
  echo "exit=$?  $(grep -E 'Elapsed \(wall' "$OUT/cli-$label.stderr" | head -1)"
}

for i in $(seq 1 "$ROUNDS"); do
  serve r$i
  cli r$i
done

echo "################ 2/3 pinned values (serve side carries no digest) ################"
python3 "$Z/verify_baseline.py" "$OUT/cli-r1.jsonl" | tail -12

echo "################ 3/3 steady per token: serve vs cli ################"
python3 "$TOOLS/serve_vs_cli_sum.py" "$OUT"

echo "################ result files ################"
ls -l "$OUT"
