#!/usr/bin/env bash
set -euo pipefail

rust_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest="$rust_root/.source-identity.manifest"
tmp=$(mktemp "$rust_root/.source-identity.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT

mapfile -d '' sources < <(
  find "$rust_root" -type f \
    \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name 'Cargo.toml' \
       -o -name 'Cargo.lock' -o -name 'rust-toolchain.toml' \
       -o -path '*/fixtures/*.json' -o -path '*/fixtures/*.bin' \) \
    ! -path "$rust_root/target/*" -print0 | sort -z
)
test "${#sources[@]}" -gt 0
for source in "${sources[@]}"; do
  hash=$(sha256sum -- "$source" | cut -d' ' -f1)
  printf '%s  %s\n' "$hash" "${source#"$rust_root/"}" >>"$tmp"
done

source_identity=$(sha256sum -- "$tmp" | cut -d' ' -f1)
mv -f -- "$tmp" "$manifest"
trap - EXIT
export COLIBRI_SOURCE_IDENTITY="$source_identity"
export COLIBRI_SOURCE_IDENTITY_MANIFEST="$manifest"

cd -- "$rust_root"
if (($# == 0)); then
  set -- run --arch "${COLIBRI_CUDA_ARCH:-sm_120}"
fi
# CUDA-Oxide uses a fixed LLVM artifact name for bin and test codegen.
if [[ ${1:-} == test ]]; then
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
fi
# The pinned nightly panics inside the diagnostic renderer on a span the
# cuda-oxide backend emits, which replaces the real diagnostic with an ICE.
# The wrapper keeps the compiler invocation identical but asks for the short
# renderer, so errors stay visible. Absolute path because cargo resolves a
# wrapper relative to its working directory.
export RUSTC_WRAPPER="$rust_root/rustc-human-wrapper.sh"
exec cargo oxide "$@"
