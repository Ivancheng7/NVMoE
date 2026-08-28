#!/usr/bin/env bash
set -euo pipefail

# The pinned nightly can ICE while rendering a diagnostic produced by the
# cuda-oxide backend. Keep the compiler invocation identical, but request the
# short renderer so a real source error remains visible instead of hiding it
# behind the renderer panic.
rustc_bin=${1:?rustc path is required}
shift
filtered=()
for argument in "$@"; do
    case "$argument" in
        --error-format=json)
            filtered+=(--error-format=short)
            ;;
        --json=*)
            ;;
        *)
            filtered+=("$argument")
            ;;
    esac
done
exec "$rustc_bin" "${filtered[@]}"
