#!/usr/bin/env bash
# 服务面的换臂验证：同一个二进制、一个 flag 两臂交错，比生成 id 串与稳态每字。
#
#   usage: ARM=COLIBRI_V4_FLASH_ROUTED_FANOUT LABEL=fanout serve_arm.sh <commit-ish> [tag] [rounds]
#          ARMS="default" ...      # 只跑"不设臂变量"的默认面（翻默认后确认发货形状）
#
# 服务面不带 digest，它的数值证据是引擎 `stats` 行里的 `ids=`。判据两条：
#   1. 所有臂、所有轮次的 8 个 id 串逐字符相同（服务面唯一的数值闸门）。
#   2. 稳态每字（引擎自报的 per_char）按臂聚合，看差是否在跑间离散之内。
# 两臂都开 `COLIBRI_V4_FLASH_SERVE_PHASE_JSONL`（默认关的落盘口），顺手拿服务面相位账；
# 它对行为只多写一个文件，两臂条件相同。
# 引擎拒绝覆盖已存在的 COLIBRI_V4_RESULT_JSONL；本脚本用新 tag，且结果目录非空就拒跑。
set -uo pipefail

if ps -eo comm | grep -qE '^(colibri-cuda|cargo)$'; then
  echo "refusing to run: another colibri-cuda or cargo is on the machine"
  ps -eo pid,etime,comm | grep -E 'colibri-cuda|cargo' | head -5
  exit 9
fi

COMMIT=${1:?commit required}
TAG=${2:-$COMMIT}
ROUNDS=${3:-3}
ARM=${ARM:-COLIBRI_V4_FLASH_ROUTED_FANOUT}
LABEL=${LABEL:-fanout}
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$TOOLS/../../.." && pwd)
TREE=${TREE:-$REPO_ROOT}
BRANCH=${BRANCH:-main}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$TREE/rust/target}
BIN=$CARGO_TARGET_DIR/release/colibri-cuda
MODEL=${COLIBRI_DSV4_MODEL_DIR:?set COLIBRI_DSV4_MODEL_DIR to the Flash-0731 checkpoint on ext4}
CUBLAS=${COLIBRI_CUBLAS_EXACT_LIB:-/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5}
OUT_ROOT=${OUT_ROOT:-$CARGO_TARGET_DIR/bench-runs}
OUT=$OUT_ROOT/servearm-$LABEL-$TAG
mkdir -p "$OUT"
if ls "$OUT"/*.log >/dev/null 2>&1; then
  echo "refusing to run: $OUT already holds logs, pick another tag"
  exit 9
fi
exec > >(tee "$OUT/batch.log") 2>&1

echo "arm=$ARM  label=$LABEL  rounds=$ROUNDS  commit=$COMMIT  out=$OUT"

echo "################ 1/3 sync and build ################"
git -C "$TREE" fetch --quiet origin "$BRANCH" 2>/dev/null || true
git -C "$TREE" checkout --quiet --detach "$COMMIT" || exit 1
git -C "$TREE" log --oneline -1 | cat
echo "dirty lines: $(git -C "$TREE" status --porcelain | wc -l)"
git -C "$TREE" status --porcelain | head -3
export RUSTC_WRAPPER="$TREE/rust/rustc-human-wrapper.sh"
( cd "$TREE/rust" && cargo oxide build --arch sm_120 -- --release ) \
  > /tmp/build_servearm.log 2>&1
status=$?
echo "build exit=$status  warnings=$(grep -c '^warning' /tmp/build_servearm.log)"
if [ $status -ne 0 ]; then
  tail -60 /tmp/build_servearm.log
  exit $status
fi
sha256sum "$BIN"

serve_arm() { # name value
  local name=$1 value=$2
  echo "################ serve $name  $ARM=$value  $(date +%H:%M:%S) ################"
  if [ "$value" = "default" ]; then
    # No arm variable at all: this is the shipping default face.
    env COLIBRI_V4_FLASH_SERVE_PHASE_JSONL="$OUT/serve-$name-phase.jsonl" \
      python3 "$TOOLS/flash_serve_client.py" \
        --binary "$BIN" \
        --model-dir "$MODEL" \
        --cublas-lib "$CUBLAS" \
        --prompt-json '"\u4f60\u597d\uff0c\u4f60\u662f\u8c01\uff1f"' \
        --max-tokens 8 --repeat 1 \
        > "$OUT/serve-$name.log" 2>&1
  else
    env "$ARM=$value" COLIBRI_V4_FLASH_SERVE_PHASE_JSONL="$OUT/serve-$name-phase.jsonl" \
      python3 "$TOOLS/flash_serve_client.py" \
        --binary "$BIN" \
        --model-dir "$MODEL" \
        --cublas-lib "$CUBLAS" \
        --prompt-json '"\u4f60\u597d\uff0c\u4f60\u662f\u8c01\uff1f"' \
        --max-tokens 8 --repeat 1 \
        > "$OUT/serve-$name.log" 2>&1
  fi
  echo "exit=$?  $(date +%H:%M:%S)"
  grep -E "^stats" "$OUT/serve-$name.log" | head -1 | cut -c1-220
}

# `ARMS` picks which arms each round runs; `default` means "do not set the arm
# variable at all", which is how the shipping default face gets measured.
ARMS=${ARMS:-"off on"}
for i in $(seq 1 "$ROUNDS"); do
  for arm in $ARMS; do
    case "$arm" in
      off) serve_arm off-r$i 0 ;;
      on) serve_arm on-r$i 1 ;;
      default) serve_arm default-r$i default ;;
      *) echo "unknown arm: $arm (use off, on or default)"; exit 9 ;;
    esac
  done
done

echo "################ 2/3 id strings and steady per arm ################"
python3 "$TOOLS/serve_arm_sum.py" "$OUT"

echo "################ result files ################"
ls -l "$OUT"
