//! Device abstraction for CPU and GPU

use crate::Error;

/// Represents a compute device (CPU or GPU).
#[derive(Clone, Debug, Default)]
pub enum Device {
    /// CPU device - uses fusor-cpu for SIMD-accelerated operations.
    #[default]
    Cpu,
    /// GPU device - uses fusor-core (wgpu) for GPU-accelerated operations.
    Gpu(fusor_core::Device),
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
        let device = fusor_core::Device::new().await?;
        Ok(Device::Gpu(device))
    }

    /// Create a new GPU device, blocking until ready.
    pub fn gpu_blocking() -> Result<Self, Error> {
        pollster::block_on(Self::gpu())
    }

    /// Create a device, preferring GPU if available, otherwise falling back to CPU.
    pub async fn auto() -> Self {
        match Self::gpu().await {
            Ok(gpu) => gpu,
            Err(_) => Device::Cpu,
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
    pub fn as_gpu(&self) -> Option<&fusor_core::Device> {
        match self {
            Device::Gpu(d) => Some(d),
            _ => None,
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
