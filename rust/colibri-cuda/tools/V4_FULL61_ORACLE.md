# V4 full61 B=1 acceptance

Run from `$COLIBRI_REPO_DIR/rust`. The validator and both runners
fail closed unless the snapshot is the pinned revision on a native ext4 block
device. The official runner owns one dense layer,
one selected routed expert, and one 256-row output-head chunk at a time; it
requires at least 9 GiB free VRAM and 8 GiB available host RAM.

```bash
COLIBRI_REPO_DIR=/absolute/path/to/colibri
COLIBRI_MODEL_DIR=/native/ext4/deepseek-v4-pro-b5968e9190ef611bbf34a7229255be88a0e937c1
COLIBRI_ORACLE_PYTHON=/absolute/path/to/oracle-venv/bin/python
COLIBRI_CUBLAS_EXACT_LIB=/absolute/path/to/pinned/libcublas.so.12
TOKEN_ID=0

: "${COLIBRI_REPO_DIR:?set the repository path}"
: "${COLIBRI_MODEL_DIR:?set the native ext4 model path}"
: "${COLIBRI_ORACLE_PYTHON:?set the oracle Python path}"
: "${COLIBRI_CUBLAS_EXACT_LIB:?set the pinned cuBLAS library path}"
cd "$COLIBRI_REPO_DIR/rust"

"$COLIBRI_ORACLE_PYTHON" colibri-cuda/tools/v4_full61_oracle.py self-test

"$COLIBRI_ORACLE_PYTHON" colibri-cuda/tools/v4_full61_oracle.py validate \
  --model-dir "$COLIBRI_MODEL_DIR"

"$COLIBRI_ORACLE_PYTHON" colibri-cuda/tools/v4_full61_oracle.py run \
  --model-dir "$COLIBRI_MODEL_DIR" \
  --token-id "$TOKEN_ID" \
  --start-pos 0

COLIBRI_DSV4_MODEL_DIR="$COLIBRI_MODEL_DIR" \
COLIBRI_CUBLAS_EXACT_LIB="$COLIBRI_CUBLAS_EXACT_LIB" \
COLIBRI_V4_FULL61_B1=1 \
COLIBRI_V4_TOKEN_ID="$TOKEN_ID" \
./run-identified.sh run --arch sm_89

"$COLIBRI_ORACLE_PYTHON" colibri-cuda/tools/v4_full61_oracle.py compare \
  "$COLIBRI_MODEL_DIR/oracles/v4_full61_token_${TOKEN_ID}_pos_0.official.jsonl" \
  "$COLIBRI_MODEL_DIR/oracles/v4_full61_token_${TOKEN_ID}_pos_0.rust.jsonl"
```

The two JSONL files each contain one pinned-run record, 61 ordered layer
records, and one final-head record. The current digest comparator is the
strict diagnostic; use the evidence policy below for the mechanism decision.
This B=1 acceptance entry intentionally accepts only `start_pos=0`; later
positions require a replayed prefix rather than a fabricated empty cache.

For a bounded prefix audit, pass `--max-layers N` to the official runner and
set `COLIBRI_V4_MAX_LAYERS=N` for Rust, then use `compare-prefix`. Prefix mode
does not execute the final head.

## Evidence policy and known limitation

Routes and selected top-k indices are discrete gates and must match exactly.
Quantized payloads/scales and transactional cache state are checked exactly
where the two runners expose the same logical slice. Vendor-specific FP32/BF16
reduction differences are reported with max absolute/relative error and ULP
distribution; final logits also require the same argmax. Such numerical
differences remain high risk if they change a later discrete gate.

The pinned TileLang sparse-attention kernel requests 344064 bytes of dynamic
shared memory for `h=128,d=512`, which Ada `sm_89` cannot launch. The official
audit therefore uses the independently cross-checked scalar PyTorch formula.
That fallback is oracle-only and must not enter the Rust runtime or benchmark.
