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

cargo build -p fusor-lavapipe-subgroups

for sg in 4 8 16 32 64; do
  echo "lavapipe subgroup sweep: $sg"
  set +e
  VK_DRIVER_FILES="$LAVAPIPE_ICD" \
    target/debug/fusor-lavapipe-subgroups --subgroup-size "$sg" $validation_args
  status=$?
  set -e
  if [ "$status" -eq 77 ]; then
    echo "skip subgroup $sg"
    continue
  fi
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
