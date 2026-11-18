#!/usr/bin/env bash
# Run DeepSeek-V4-Flash-0731 CLI generation with the shipping default configuration.
#
# Required environment variable:
#   COLIBRI_DSV4_MODEL_DIR  Path to the pinned Flash-0731 checkpoint on a native ext4 volume
#
# Optional overrides:
#   COLIBRI_V4_FLASH_TOKEN_IDS  Comma-separated prompt token IDs (default: "0")
#   COLIBRI_V4_GREEDY_STEPS     Number of greedy tokens to generate (default: 8)
#   COLIBRI_V4_RESULT_JSONL     Output JSONL path (default: ./flash-result-<timestamp>.jsonl)
#   COLIBRI_CUBLAS_EXACT_LIB    Path to CUDA 12.8+ libcublas.so.<ver>
#   BIN                         Path to built colibri-cuda binary
set -euo pipefail

TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUST_DIR=$(cd -- "$TOOLS_DIR/../.." && pwd)
TARGET_DIR="${CARGO_TARGET_DIR:-$RUST_DIR/target}"
BIN="${BIN:-$TARGET_DIR/release/colibri-cuda}"

: "${COLIBRI_DSV4_MODEL_DIR:?COLIBRI_DSV4_MODEL_DIR must point to the Flash-0731 checkpoint on ext4}"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not found at $BIN" >&2
  echo "build it first from rust/: ./run-identified.sh build --arch sm_120 -- --release -p colibri-cuda" >&2
  exit 1
fi

if [[ -z "${COLIBRI_CUBLAS_EXACT_LIB:-}" ]]; then
  for cand in /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.* \
              /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12.*; do
    if [[ -f "$cand" ]]; then
      export COLIBRI_CUBLAS_EXACT_LIB="$cand"
      break
    fi
  done
fi
: "${COLIBRI_CUBLAS_EXACT_LIB:?set COLIBRI_CUBLAS_EXACT_LIB to your CUDA 12.8+ libcublas.so.<version>}"

CUBLAS_DIR=$(dirname -- "$COLIBRI_CUBLAS_EXACT_LIB")
export LD_LIBRARY_PATH="$CUBLAS_DIR:${LD_LIBRARY_PATH:-}"

RESULT_JSONL="${COLIBRI_V4_RESULT_JSONL:-$PWD/flash-result-$(date +%Y%m%d-%H%M%S).jsonl}"

exec env \
  COLIBRI_V4_FLASH_BASE=1 \
  COLIBRI_V4_FLASH_TOKEN_IDS="${COLIBRI_V4_FLASH_TOKEN_IDS:-0}" \
  COLIBRI_V4_GREEDY_STEPS="${COLIBRI_V4_GREEDY_STEPS:-8}" \
  COLIBRI_V4_STREAM_READ_MODE="${COLIBRI_V4_STREAM_READ_MODE:-buffered}" \
  COLIBRI_V4_FLASH_PREFETCH="${COLIBRI_V4_FLASH_PREFETCH:-1}" \
  COLIBRI_V4_FLASH_MHC_LINEAR_ARM="${COLIBRI_V4_FLASH_MHC_LINEAR_ARM:-cublas}" \
  COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM="${COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM:-cublas}" \
  COLIBRI_V4_FLASH_ROUTER_SCORE_ARM="${COLIBRI_V4_FLASH_ROUTER_SCORE_ARM:-libdevice}" \
  COLIBRI_V4_RESULT_JSONL="$RESULT_JSONL" \
  "$BIN" "$@"
