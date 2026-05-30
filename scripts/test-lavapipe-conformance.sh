#!/bin/sh
set -eu

if [ -z "${LAVAPIPE_ICD:-}" ]; then
  echo "LAVAPIPE_ICD must point to an lvp_icd*.json file" >&2
  exit 77
fi

if [ ! -f "$LAVAPIPE_ICD" ]; then
  echo "LAVAPIPE_ICD does not exist: $LAVAPIPE_ICD" >&2
  exit 77
fi

dyld_library_path="/opt/homebrew/lib"
if [ -n "${DYLD_LIBRARY_PATH:-}" ]; then
  dyld_library_path="$DYLD_LIBRARY_PATH:$dyld_library_path"
fi

MESA_SHADER_CACHE_DISABLE=1 \
  DYLD_LIBRARY_PATH="$dyld_library_path" \
  VK_DRIVER_FILES="$LAVAPIPE_ICD" \
  WGPU_BACKEND=vulkan \
  WGPU_ADAPTER_NAME=llvmpipe \
  FUSOR_CONFORMANCE_REQUIRE_GPU=1 \
  RUST_MIN_STACK=16777216 \
  cargo test -p fusor-conformance "$@"
