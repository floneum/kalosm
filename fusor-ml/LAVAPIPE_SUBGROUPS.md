# Lavapipe Subgroup Smoke Tests

This is a local pre-CI smoke test for SPIR-V compute kernels under Mesa
Lavapipe. It uses raw Vulkan so it can request a specific subgroup width with
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, which wgpu does not
currently expose.

## Setup

Install Lavapipe and point the Vulkan loader at its ICD manifest:

```sh
export LAVAPIPE_ICD=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

On macOS arm64, use the Lavapipe build archive and point at its manifest:

```sh
export LAVAPIPE_ICD=$PWD/lavapipe-macos-arm64/lvp_icd.aarch64.json
```

The script sets `VK_DRIVER_FILES="$LAVAPIPE_ICD"` for each run. If no CPU
Vulkan device is visible, or a requested width is outside the device's reported
subgroup-size range, the runner exits `77` and the script treats that width as
skipped.

## Run

```sh
scripts/test-lavapipe-subgroups.sh
```

To run one width or one kernel directly:

```sh
VK_DRIVER_FILES="$LAVAPIPE_ICD" \
  cargo run -p fusor-lavapipe-subgroups -- --subgroup-size 32 --kernel flash-attention
```

Enable validation locally with:

```sh
FUSOR_VK_VALIDATION=1 scripts/test-lavapipe-subgroups.sh
```

The runner currently covers:

- `subgroup-probe`: verifies the shader observes the requested subgroup size.
- `subgroup-reduce`: checks `subgroup_reduce_sum` across one workgroup.
- `flash-attention`: runs the real Fusor streaming flash-attention tile kernel
  for the requested width and compares against a CPU reference.

Hardcoded-32 qgemv/qmatmul kernels are intentionally not part of the default
sweep yet; they should be added once those kernels are parameterized by subgroup
width.
