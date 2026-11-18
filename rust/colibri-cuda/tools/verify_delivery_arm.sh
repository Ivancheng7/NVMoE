#!/usr/bin/env bash
# 交付面的开关 A/B 一条龙：同步、构建、三跑、逐位比对、相位账，一次跑完。
# 全部输出同时写进结果目录（`$OUT/batch.log`）：/tmp 是 tmpfs，证据别只留在那儿。
#
#   usage: verify_delivery_arm.sh <commit-ish> [tag]
#          ARM=COLIBRI_V4_FLASH_COMPRESSOR_ARENA LABEL=comparena verify_delivery_arm.sh <commit> t1
#          ORDER=off-first ...        # 先跑关掉那臂
#
# 默认臂是固定层槽位（这个脚本最早就是为它写的）。**换序要跑两次**：冷帧（前两个
# token）里先跑的那臂要付冷页缓存的 6.7 GB 固定权重读，只跑一个顺序的话，前两个
# token 的差是顺序的差，不是改动的差。
#
# 判据（跑完看最后四段输出）：
#   1. 四个钉死值 + 43 层 + block32 / canonical 两个臂，1 步跑逐项 OK。
#   2. 8 步两臂的 8 个 hidden digest、8 条 43 层摘要链、8 个 argmax 与参照逐项一致。
#      **显存峰值允许不同**（槽位自己要占设备缓冲，那是代价不是数值）。
#   3. 相位账：收益要出现在它该出现的相位上，而且两臂 `outer` 的差应约等于那个相位的差
#      （不然就是被别的相位吃掉了，或者计时被推迟了）。
#   4. 数值两臂都不动（在第 2 段里一起看）。
#
# 脚本刻意不删任何结果文件：引擎自己拒绝覆盖已存在的 COLIBRI_V4_RESULT_JSONL，
# 重跑请换 tag。检查脚本在 tools/verify/ 下。
set -uo pipefail

# 两个会话同时跑引擎会毁掉双方的计时，这条是双向的；构建也一样会污染。
# 所以先看机器空不空，别靠记性。
if ps -eo comm | grep -qE '^(colibri-cuda|cargo)$'; then
  echo "refusing to run: another colibri-cuda or cargo is on the machine"
  ps -eo pid,etime,comm | grep -E 'colibri-cuda|cargo' | head -5
  exit 9
fi

COMMIT=${1:?commit required}
TAG=${2:-$COMMIT}
ARM=${ARM:-COLIBRI_V4_FLASH_FIXED_SLOT}
LABEL=${LABEL:-slot}
ORDER=${ORDER:-on-first}
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
OUT=$OUT_ROOT/ab-$LABEL-$TAG
REF8=${REF8:-}
mkdir -p "$OUT"
exec > >(tee "$OUT/batch.log") 2>&1

echo "arm=$ARM  label=$LABEL  order=$ORDER  out=$OUT"

echo "################ 1/4 sync and build $COMMIT ################"
git -C "$TREE" fetch --quiet origin "$BRANCH" 2>/dev/null || true
git -C "$TREE" checkout --quiet --detach "$COMMIT" || exit 1
git -C "$TREE" log --oneline -1 | cat
git -C "$TREE" status --porcelain | head -3
export RUSTC_WRAPPER="$TREE/rust/rustc-human-wrapper.sh"
( cd "$TREE/rust" && cargo oxide build --arch sm_120 -- --release ) \
  > /tmp/build_arm.log 2>&1
status=$?
echo "build exit=$status  warnings=$(grep -c '^warning' /tmp/build_arm.log)"
if [ $status -ne 0 ]; then
  tail -40 /tmp/build_arm.log
  exit $status
fi
sha256sum "$BIN"

run() { # label steps telemetry [extra VAR=value ...]
  local label=$1 steps=$2 telemetry=$3
  shift 3
  echo "################ run $label  steps=$steps  telemetry=$telemetry  $* ################"
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

if [ "$ORDER" = "off-first" ]; then
  run 1step-$LABEL-on 1 0
  run 8step-$LABEL-off 8 1 "$ARM=0"
  run 8step-$LABEL-on 8 1
else
  run 1step-$LABEL-on 1 0
  run 8step-$LABEL-on 8 1
  run 8step-$LABEL-off 8 1 "$ARM=0"
fi

echo "################ 2/4 pinned values (1 step) ################"
python3 "$Z/verify_baseline.py" "$OUT/1step-$LABEL-on.jsonl" | tail -12

echo "################ 3/4 bit-exact comparison ################"
if [[ -n "$REF8" && -f "$REF8" ]]; then
  python3 "$Z/outputs_match.py" "$REF8" "$OUT/8step-$LABEL-on.jsonl" "$OUT/8step-$LABEL-off.jsonl"
else
  python3 "$Z/outputs_match.py" "$OUT/8step-$LABEL-on.jsonl" "$OUT/8step-$LABEL-off.jsonl"
fi
python3 "$Z/verify_baseline.py" "$OUT/8step-$LABEL-on.jsonl" | tail -12

echo "################ 4/4 phase table: arm on vs off ################"
python3 "$Z/attention_share.py" "$OUT/8step-$LABEL-on.jsonl" "$OUT/8step-$LABEL-off.jsonl"

echo "################ result files ################"
ls -l "$OUT"
