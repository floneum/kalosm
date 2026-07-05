//! Device abstraction for CPU and GPU

use crate::Error;

/// Represents a compute device (CPU or GPU).
#[derive(Clone, Debug, Default)]
pub enum Device {
    /// CPU device - uses fusor-cpu for SIMD-accelerated operations.
    #[default]
    Cpu,
    /// GPU device - uses fusor-core (wgpu) for GPU-accelerated operations.
    Gpu(crate::gpu::Device),
}

impl Device {
    /// Create a new CPU device.
    pub fn cpu() -> Self {
        Device::Cpu
    }

    /// Create a new GPU device asynchronously.
    ///
    /// This is an alias for `gpu()` to match the fusor-core API.
    pub async fn new() -> Result<Self, Error> {
        Self::gpu().await
    }

    /// Create a new GPU device asynchronously.
    pub async fn gpu() -> Result<Self, Error> {
        #[cfg(feature = "gpu")]
        {
            let device = crate::gpu::Device::new().await?;
            Ok(Device::Gpu(device))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(Error::msg("GPU backend is disabled"))
        }
    }

    /// Create a new GPU device, blocking until ready.
    pub fn gpu_blocking() -> Result<Self, Error> {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            pollster::block_on(Self::gpu())
        }
        #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
        {
            Err(Error::msg(
                "blocking GPU initialization is not supported on wasm; use Device::gpu().await",
            ))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(Error::msg("GPU backend is disabled"))
        }
    }

    /// Resolve every pending lazy tensor now, submitting the work to the GPU
    /// without waiting for it or downloading anything. Call at iteration
    /// boundaries in training-style loops to keep the pending graph bounded
    /// while the GPU runs ahead of the host. No-op on CPU, where lazy
    /// expressions are evaluated when they are consumed.
    pub fn flush(&self) {
        match self {
            Device::Cpu => {}
            Device::Gpu(device) => device.flush(),
        }
    }

    /// Create a device, preferring GPU if available, otherwise falling back to CPU.
    pub async fn auto() -> Self {
        #[cfg(not(feature = "gpu"))]
        {
            return Device::Cpu;
        }
        #[cfg(feature = "gpu")]
        match Self::gpu().await {
            Ok(gpu) => gpu,
            Err(err) => {
                if std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
                    || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
                    || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some()
                {
                    tracing::warn!("fusor_device_auto_gpu_error={err}");
                }
                Device::Cpu
            }
        }
    }

    /// Returns true if this is a CPU device.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns true if this is a GPU device.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    /// Returns a reference to the GPU device if this is a GPU device.
    #[inline]
    pub fn as_gpu(&self) -> Option<&crate::gpu::Device> {
        match self {
            Device::Gpu(d) => Some(d),
            _ => None,
        }
    }

    /// Return a handle to the same device that reports no subgroup support, so
    /// the no-subgroup kernel fallbacks (the browser's only path) are exercised.
    /// A no-op for the CPU device.
    pub fn without_subgroups(&self) -> Self {
        match self {
            Device::Cpu => Device::Cpu,
            Device::Gpu(d) => Device::Gpu(d.without_subgroups()),
        }
    }

    /// Return a handle whose tensor allocations poison kernel-output buffers,
    /// reproducing the app's reused buffer pool. A no-op for the CPU device.
    pub fn with_poisoned_allocations(&self) -> Self {
        match self {
            Device::Cpu => Device::Cpu,
            Device::Gpu(d) => Device::Gpu(d.with_poisoned_allocations()),
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => true,
            // GPU devices from the same Arc are equal
            (Device::Gpu(_), Device::Gpu(_)) => true,
            _ => false,
        }
    }
}
