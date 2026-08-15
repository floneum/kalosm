//! What a device *can* do. Legality only — rates live in
//! [`crate::cost::DeviceFacts`].

use crate::dtype::Dtype;
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// One cooperative-matrix configuration the adapter reports.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoopKind {
    pub operand: Dtype,
    pub acc: Dtype,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// Subgroup width range. `min == max` is the *fixed* case every
/// subgroup-size-aware kernel requires.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupWidths {
    pub min: u32,
    pub max: u32,
}

impl SubgroupWidths {
    pub const fn is_fixed(self) -> bool {
        self.min == self.max
    }
    /// The width to assume; every policy derived from it is a floor.
    pub const fn assumed(self) -> u32 {
        self.min
    }
}

/// The wgpu limits the compiler actually reads, mirrored so `fusor2-ir` has
/// no wgpu dependency. Defaults are the **WebGPU baseline**, not
/// `adapter.limits()`: a plan legal on one device is then legal on another,
/// and the cost model's filters mean the same thing everywhere. A backend
/// widens a field only when a selected kernel proves it needs the headroom.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Limits {
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size: [u32; 3],
    pub max_compute_workgroups_per_dimension: u32,
    /// Part of the plan-cache fingerprint.
    pub max_compute_workgroup_storage_size: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_buffer_binding_size: u64,
}

impl Default for Limits {
    /// WebGPU baseline.
    fn default() -> Self {
        Self {
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size: [256, 256, 64],
            max_compute_workgroups_per_dimension: 65535,
            max_compute_workgroup_storage_size: 16384,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_buffer_binding_size: 128 << 20,
        }
    }
}

/// Broad device class. Used only to seed calibration and pick a fallback
/// rate table; never to route a kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Gpu,
    Cpu,
}

/// Everything a legality predicate may read about a device. Every
/// performance feature is probed and optional, each with a working fallback
/// (shared-memory reduction trees for subgroups, f32 for f16,
/// sgemm/sgemv/generic fold for cooperative matrix, cold compile, no
/// profiling).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Caps {
    pub kind: DeviceKind,
    /// Stable adapter/backend name; part of the calibration cache key.
    pub name: String,
    pub limits: Limits,
    pub subgroups: Option<SubgroupWidths>,
    pub f16: bool,
    pub bf16: bool,
    pub coop: SmallVec<[CoopKind; 4]>,
    /// `atomicAdd` on f32 in storage. Gates `ScatterMode::Atomic`.
    pub atomic_f32: bool,
    /// Byte-arena workgroup packing (the `fork-metal` feature).
    pub workgroup_alias: bool,
    /// Cooperative store of an f32 accumulator into f16 memory (also
    /// `fork-metal`). Without it such a kernel pays a staging tile plus a
    /// per-lane cast — footprint, never correctness.
    pub mixed_precision_coop_store: bool,
    pub pipeline_cache: bool,
    pub timestamp_query: bool,
    /// SIMD lane counts the CPU emitter may instantiate (4, 8, 16).
    pub simd_widths: SmallVec<[u32; 3]>,
    /// Worker threads available to the CPU backend. 1 on wasm32.
    pub threads: u32,
}

impl Caps {
    /// A coop config, a *fixed* subgroup width, and enough workgroup width.
    pub fn coop_supported(&self) -> bool {
        !self.coop.is_empty()
            && self.subgroups.is_some_and(|s| s.is_fixed())
            && self.limits.max_compute_workgroup_size[0] >= 64
    }

    pub fn coop_for(&self, operand: Dtype, acc: Dtype) -> Option<CoopKind> {
        self.coop
            .iter()
            .copied()
            .find(|c| c.operand == operand && c.acc == acc)
    }

    /// 32 when subgroups are unsupported — the narrowest width on hardware
    /// fusor2 targets, so a wrong guess only keeps more parallelism.
    pub fn subgroup_width(&self) -> u32 {
        self.subgroups.map_or(32, |s| s.assumed())
    }

    /// Stable digest folded into `PlanHash` and the disk-cache salt.
    pub fn fingerprint(&self) -> u64 {
        let mut h = FxHasher::default();
        self.hash(&mut h);
        h.finish()
    }
}
