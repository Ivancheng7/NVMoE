#!/usr/bin/env bash
# Launch the OpenAI-compatible HTTP API server backed by DeepSeek-V4-Flash-0731.
#
# Required environment variable:
#   COLIBRI_DSV4_MODEL_DIR  Path to the pinned Flash-0731 checkpoint on a native ext4 volume
#
# Optional overrides:
#   HOST                        Bind address (default: 127.0.0.1)
#   PORT                        Bind port (default: 8000)
#   MAX_TOKENS                  Max completion tokens per request, <= 64 (default: 64)
#   COLIBRI_CUBLAS_EXACT_LIB    Path to CUDA 12.8+ libcublas.so.<ver>
#   BIN                         Path to built colibri-cuda binary
set -euo pipefail

TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$TOOLS_DIR/../../.." && pwd)
RUST_DIR="$REPO_ROOT/rust"
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

export COLIBRI_V4_STREAM_READ_MODE="${COLIBRI_V4_STREAM_READ_MODE:-buffered}"
export COLIBRI_V4_FLASH_PREFETCH="${COLIBRI_V4_FLASH_PREFETCH:-1}"
export COLIBRI_V4_FLASH_MHC_LINEAR_ARM="${COLIBRI_V4_FLASH_MHC_LINEAR_ARM:-cublas}"
export COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM="${COLIBRI_V4_FLASH_ROUTER_LINEAR_ARM:-cublas}"
export COLIBRI_V4_FLASH_ROUTER_SCORE_ARM="${COLIBRI_V4_FLASH_ROUTER_SCORE_ARM:-libdevice}"

exec python3 "$REPO_ROOT/c/openai_server.py" \
  --flash-base \
  --model "$COLIBRI_DSV4_MODEL_DIR" \
  --engine "$BIN" \
  --model-id "${COLI_MODEL_ID:-deepseek-v4-flash-0731}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --kv-slots 1 \
  --max-tokens "${MAX_TOKENS:-64}" \
  "$@"
