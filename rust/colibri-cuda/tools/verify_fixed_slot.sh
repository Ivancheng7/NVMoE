#!/usr/bin/env bash
# 固定层设备槽位的一条龙验证：同步、构建、三跑、逐位比对、相位账，一次跑完。
# 全部输出同时写进结果目录（`$OUT/batch.log`）：/tmp 是 tmpfs，证据别只留在那儿。
#
#   usage: verify_fixed_slot.sh <commit-ish> [tag]
#          ORDER=off-first verify_fixed_slot.sh <commit-ish> [tag]
#
#   顺序是可换的：`ORDER=off-first` 先跑关掉那臂。**冷帧（前两个 token）里先跑的那臂
#   要付冷页缓存的 6.7 GB 固定权重读**，所以只跑一个顺序时前两个 token 的差是顺序的
#   差，不是改动的差。要判冷帧就得两个顺序各跑一次。
#
# 判据（跑完看最后四段输出）：
#
#   1. 四个钉死值 + 43 层 + block32/canonical 臂，1 步跑逐项 OK。
#   2. 8 步开关两臂的 8 个 hidden digest、8 条 43 层摘要链、8 个 argmax 与
#      defaults-pc-8step 逐项 IDENTICAL。**唯一允许的差异是显存峰值**：槽位
#      自己占三个层型的设备缓冲（实测 +216 MB），那是代价不是数值。
#   3. 相位账里 fixed 的墙：开着约 0.7 s/token，关掉应回到约 1.0 s/token。
#      这一项才是槽位改动的证据——fixed_h2d_bytes 是按缓冲尺寸算的，
#      restage 拷的是同一批字节，所以那个字节账两臂相同。
#   4. 数值两臂都不动（比对在第 2 段里一起看）。
#
# 脚本刻意不删任何结果文件：引擎自己拒绝覆盖已存在的 COLIBRI_V4_RESULT_JSONL，
# 重跑请换 tag。检查脚本在量测目录 .zcode/ 下（未跟踪，但它们是判据本身，
# 别改；改它们等于改判据）。
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
Z=/mnt/c/Users/27236/Desktop/localmoe-clean/.zcode
TREE=/srv/localmoe/build/merged-prompt-chunk
export CARGO_TARGET_DIR=/srv/localmoe/targets/merged-prompt-chunk-target
BIN=$CARGO_TARGET_DIR/release/colibri-cuda
MODEL=/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb
CUBLAS=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5
OUT=/srv/localmoe/targets/fixed-slot-$TAG
REF8=/srv/localmoe/targets/defaults-pc-8step.jsonl
mkdir -p "$OUT"
# 输出同时进控制台和结果目录：这份 log 是这次跑的判据的一部分，跟 jsonl 放一起。
exec > >(tee "$OUT/batch.log") 2>&1

echo "################ 1/4 sync and build $COMMIT ################"
git -C "$TREE" fetch --quiet origin codex/business-restart || exit 1
git -C "$TREE" checkout --quiet --detach "$COMMIT" || exit 1
git -C "$TREE" log --oneline -1 | cat
git -C "$TREE" status --porcelain | head -3
export RUSTC_WRAPPER="$TREE/rust/rustc-human-wrapper.sh"
( cd "$TREE/rust" && cargo oxide build --arch sm_120 -- --release ) \
  > /tmp/build_fixed_slot.log 2>&1
status=$?
echo "build exit=$status  warnings=$(grep -c '^warning' /tmp/build_fixed_slot.log)"
if [ $status -ne 0 ]; then
  tail -40 /tmp/build_fixed_slot.log
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
}

if [ "${ORDER:-on-first}" = "off-first" ]; then
  run 1step-slot-on 1 0
  run 8step-slot-off 8 1 COLIBRI_V4_FLASH_FIXED_SLOT=0
  run 8step-slot-on 8 1
else
  run 1step-slot-on 1 0
  run 8step-slot-on 8 1
  run 8step-slot-off 8 1 COLIBRI_V4_FLASH_FIXED_SLOT=0
fi

echo "################ 2/4 pinned values (1 step) ################"
python3 "$Z/verify_baseline.py" "$OUT/1step-slot-on.jsonl" | tail -12

echo "################ 3/4 bit-exact comparison (reference $REF8) ################"
python3 "$Z/outputs_match.py" "$REF8" "$OUT/8step-slot-on.jsonl" "$OUT/8step-slot-off.jsonl"
python3 "$Z/verify_baseline.py" "$OUT/8step-slot-on.jsonl" | tail -12

echo "################ 4/4 phase table: fixed wall, slot on vs off ################"
python3 "$Z/attention_share.py" "$OUT/8step-slot-on.jsonl" "$OUT/8step-slot-off.jsonl"

echo "################ result files ################"
ls -l "$OUT"
