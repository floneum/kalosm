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

validation_args=""
if [ -n "${FUSOR_VK_VALIDATION:-}" ]; then
  validation_args="--validation"
fi

dyld_library_path="/opt/homebrew/lib"
if [ -n "${DYLD_LIBRARY_PATH:-}" ]; then
  dyld_library_path="$DYLD_LIBRARY_PATH:$dyld_library_path"
fi

cargo build -p fusor-lavapipe-subgroups

cargo_target_dir="${CARGO_TARGET_DIR:-target}"
runner="$cargo_target_dir/debug/fusor-lavapipe-subgroups"

lavapipe_ran=""
emulated_ran=""

for sg in 4 8 16 32 64; do
  echo "lavapipe subgroup sweep: $sg"
  set +e
    MESA_SHADER_CACHE_DISABLE=1 \
    DYLD_LIBRARY_PATH="$dyld_library_path" \
    VK_DRIVER_FILES="$LAVAPIPE_ICD" \
    "$runner" --backend vulkan --subgroup-size "$sg" $validation_args
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    lavapipe_ran="$lavapipe_ran $sg"
    continue
  elif [ "$status" -ne 77 ]; then
    exit "$status"
  fi

  echo "lavapipe skipped subgroup $sg; running local emulation fallback"
  "$runner" --backend emulated --subgroup-size "$sg"
  emulated_ran="$emulated_ran $sg"
done

if [ -z "$lavapipe_ran" ]; then
  echo "no subgroup width ran on Lavapipe; check LAVAPIPE_ICD" >&2
  exit 1
fi

echo "lavapipe-covered subgroups:$lavapipe_ran"
if [ -n "$emulated_ran" ]; then
  echo "emulated fallback subgroups:$emulated_ran"
fi
