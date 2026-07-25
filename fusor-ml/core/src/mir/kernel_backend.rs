pub(crate) use fusor_tile_ir_runtime::{
    DirectKernel, DirectKernelTemplate, KernelCacheKey, KernelVariantKey, PreparedDirectDispatch,
    dynamic_kernel_from_ir, run_direct_kernel, run_kernel, three_buffer_pipeline_from_ir,
};

/// Marker returned by device-specific direct-kernel builders when the current
/// device cannot support the required IR capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceNotSupported;

impl std::fmt::Display for DeviceNotSupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("device does not support this kernel variant")
    }
}

impl std::error::Error for DeviceNotSupported {}

pub(crate) mod sampling;
