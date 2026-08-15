//! Schedule-domain generators. Each `legal` *generates* the complete legal
//! parameter space of one node under one `Caps`, filtered by structural
//! predicates and the exact [`crate::arena`] footprint.

pub mod coop;
pub mod fold;
pub mod map;
pub mod sgemm;
pub mod sgemv;

pub use coop::legal as coop_legal;
pub use fold::legal as fold_legal;
pub use map::legal as map_legal;
pub use sgemm::legal as sgemm_legal;
pub use sgemv::legal as sgemv_legal;

pub use coop::{coop_domain, coop_tiles, stage_element};
pub use fold::{emitted_block, fold_blocks, fold_domain, fold_domain_for};
pub use map::map_domain;
pub use sgemm::sgemm_domain;
pub use sgemv::sgemv_domain;

use fusor2_ir::device::Caps;
use fusor2_ir::ir::level1::{FoldStrat, MapTiling, SgemmParams, SgemvParams};
use fusor2_ir::ir::level2::ArenaPlanner;

/// Everything a generator reads. `planner` is the *exact* footprint
/// function — never an estimator, because a geometry admitted here must
/// pass `verify_l1` unchanged.
pub struct DomainCtx<'a> {
    pub caps: &'a Caps,
    pub planner: &'a dyn ArenaPlanner,
}

impl<'a> DomainCtx<'a> {
    pub fn new(caps: &'a Caps, planner: &'a dyn ArenaPlanner) -> Self {
        Self { caps, planner }
    }
}

/// Hard ceiling on split-K candidates, matching the reference's
/// `split_candidates` bound. Bounds the candidate count; it is not a
/// profitability judgement.
pub const MAX_SPLITS: u32 = 64;

/// A process-wide memo for a shape-independent candidate table.
///
/// The heavy generators (`coop::candidate_geoms_for`, `sgemm::sgemm_domain`)
/// are pure functions of `(Caps, element, planner)` — no extent reaches them,
/// which is the whole point of carrying a domain instead of a chosen point.
/// Regenerating them per contraction cost 2.3 ms for coop and 0.33 ms for
/// sgemm, so a graph with two matmuls could not saturate inside *any*
/// sensible budget and the driver truncated at a wall-clock-dependent point.
/// Memoizing makes the enumeration lazy in the only sense that matters:
/// once per device, not once per node.
pub(crate) struct DomainMemo<K, V> {
    slots: std::sync::Mutex<Vec<(K, V)>>,
}

impl<K: Clone + PartialEq, V: Clone> DomainMemo<K, V> {
    pub(crate) const fn new() -> Self {
        Self {
            slots: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// The memoized value for `key`, computing it on a miss. A device count
    /// in the low single digits makes a linear scan the right structure and
    /// keeps the key free of a `Hash` bound.
    pub(crate) fn get_or_insert(&self, key: &K, build: impl FnOnce() -> V) -> V {
        if let Ok(slots) = self.slots.lock()
            && let Some((_, v)) = slots.iter().find(|(k, _)| k == key)
        {
            return v.clone();
        }
        let value = build();
        if let Ok(mut slots) = self.slots.lock()
            && !slots.iter().any(|(k, _)| k == key)
        {
            slots.push((key.clone(), value.clone()));
        }
        value
    }
}

/// Identity of the planner a `DomainCtx` carries, for memo keys. Two
/// `ArenaPlanner`s may report different footprints, so a cached candidate
/// table is only valid for the planner that filtered it.
pub(crate) fn planner_id(planner: &dyn ArenaPlanner) -> usize {
    std::ptr::from_ref(planner) as *const () as usize
}

/// Rank of a point no bench has ever visited. Deliberately far from the
/// measured band and deliberately below 255, so a future seed table can
/// still order below it.
pub(crate) const UNMEASURED: u8 = 200;

/// Ascending-tuple tiebreak for the sgemm cap.
pub(crate) fn sgemm_order(p: &SgemmParams) -> (u32, u32, u32, u32, u32, bool) {
    (p.bm, p.bn, p.bk, p.tm, p.tn, p.double_buffer)
}

/// Ascending-tuple tiebreak for the sgemv cap.
pub(crate) fn sgemv_order(p: &SgemvParams) -> (u32, u32, u32, u32, u32) {
    (p.vector, p.subgroups, p.cols, p.parts, p.gap)
}

/// Ascending-tuple tiebreak for the fold cap.
pub(crate) fn fold_order(s: &FoldStrat) -> (u8, u32, u32) {
    match s {
        FoldStrat::Subgroup => (0, 0, 0),
        FoldStrat::WgTree { lane_group } => (1, *lane_group, 0),
        FoldStrat::LoopThenTree {
            iterations,
            lane_group,
        } => (2, *lane_group, *iterations),
    }
}

/// Ascending-tuple tiebreak for the map cap.
pub(crate) fn map_order(t: &MapTiling) -> (u32, u32, u32) {
    (t.dim.map_or(u32::MAX, |d| d), t.tm, t.vector)
}

// ---------------------------------------------------------------------------
// The planner the L1 rules reach for when the caller supplies none
// ---------------------------------------------------------------------------

/// The [`ArenaPlanner`] the rules in [`crate::rules`] reach for: the one
/// memoized [`crate::Planner`], the same object `verify_l1` admits against
/// and the L2 emitter lays out with. A geometry this crate admits therefore
/// passes `verify_l1` unchanged, because both read the same number from the
/// same function.
pub fn default_planner() -> &'static dyn ArenaPlanner {
    crate::Planner::global()
}

#[cfg(test)]
pub(crate) mod testing {
    //! Shared `Caps` fixtures for the domain and rule tests.

    use fusor2_ir::device::{Caps, CoopKind, DeviceKind, Limits, SubgroupWidths};
    use fusor2_ir::dtype::Dtype;
    use smallvec::smallvec;

    /// WebGPU baseline limits, subgroups fixed at 32, f32 coop available.
    pub fn baseline_caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "baseline".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: false,
            coop: smallvec![
                CoopKind {
                    operand: Dtype::F32,
                    acc: Dtype::F32,
                    m: 8,
                    n: 8,
                    k: 8
                },
                CoopKind {
                    operand: Dtype::F16,
                    acc: Dtype::F16,
                    m: 8,
                    n: 8,
                    k: 8
                },
            ],
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: smallvec![1],
            threads: 1,
        }
    }

    /// Apple-class: 32 KiB threadgroup memory, 1024 lanes, subgroup 32.
    pub fn apple_caps() -> Caps {
        let mut caps = baseline_caps();
        caps.name = "Apple M2 Max".into();
        caps.limits.max_compute_invocations_per_workgroup = 1024;
        caps.limits.max_compute_workgroup_size = [1024, 1024, 64];
        caps.limits.max_compute_workgroup_storage_size = 32768;
        caps
    }

    /// Apple-class with no cooperative-matrix configurations at all.
    pub fn no_coop_caps() -> Caps {
        let mut caps = apple_caps();
        caps.coop = smallvec![];
        caps
    }
}

