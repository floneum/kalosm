# Lavapipe Conformance

Use Lavapipe as a software Vulkan adapter for the existing Fusor conformance
harness:

```sh
export LAVAPIPE_ICD=$PWD/.lavapipe/lavapipe-macos-arm64/lvp_icd.aarch64.json
scripts/test-lavapipe-conformance.sh
```

The script sets:

- `VK_DRIVER_FILES="$LAVAPIPE_ICD"` so the Vulkan loader uses Lavapipe.
- `WGPU_BACKEND=vulkan` so wgpu does not pick Metal.
- `FUSOR_CONFORMANCE_REQUIRE_GPU=1` so the run fails if the conformance harness
  cannot create a GPU device.
- `MESA_SHADER_CACHE_DISABLE=1` to avoid Mesa shader-cache writes during local
  sandboxed runs.

This runs `cargo test -p fusor-conformance`. It does not run a separate raw
Vulkan smoke harness.

## Subgroup Coverage

The current macOS arm64 Lavapipe build reports subgroup range `4..=4`. Through
the conformance harness, this means Lavapipe coverage is whatever Fusor/wgpu can
run on that adapter. The harness does not currently request alternate subgroup
widths per pipeline with
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, so `8/16/32/64` are not
executed as real Vulkan subgroup widths here.
