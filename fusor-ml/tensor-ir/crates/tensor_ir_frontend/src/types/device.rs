//! Device profile: hardware/runtime parameters that bound how the
//! compiler may dispatch a kernel. Threaded through `StageConfig`/
//! `RunnerConfig` so optimization phases and the cost model can consult
//! real device limits instead of hardcoded constants.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Lanes per simdgroup (a.k.a. subgroup / warp).
    pub simd_width: u32,
    /// Maximum threadgroup (shared) memory in bytes a single dispatch may use.
    pub max_threadgroup_bytes: u32,
    /// Soft cap on registers per lane before occupancy degrades.
    pub max_registers_per_lane: u32,
    /// Maximum simdgroups per physical workgroup.
    pub max_simdgroups: u32,
    /// Maximum threads per workgroup (typically `simd_width * max_simdgroups`).
    pub max_workgroup_size: u32,
}

impl DeviceProfile {
    /// A conservative default targeting Apple Silicon (M-series) GPUs.
    #[must_use]
    pub const fn default_apple_silicon() -> Self {
        Self {
            simd_width: 32,
            max_threadgroup_bytes: 32 * 1024,
            max_registers_per_lane: 128,
            max_simdgroups: 8,
            max_workgroup_size: 256,
        }
    }
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self::default_apple_silicon()
    }
}
