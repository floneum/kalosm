//! Schedule-domain generators. Each `legal` *generates* the complete legal
//! parameter space of one node under one `Caps`, filtered by structural
//! predicates and the exact `arena` footprint.

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
pub use fold::{emitted_block, fold_domain, fold_domain_for};
pub use map::map_domain;
pub use sgemm::sgemm_domain;
pub use sgemv::sgemv_domain;

use fusor_ir::device::Caps;
use fusor_ir::ir::kernel::ArenaPlanner;
use fusor_ir::ir::launch::{FoldStrat, MapTiling, SgemmParams, SgemvParams};

/// Everything a generator reads. `planner` is the *exact* footprint
/// function — never an estimator, because a geometry admitted here must
/// pass `verify_launch` unchanged.
pub struct DomainCtx<'a> {
    pub caps: &'a Caps,
    pub planner: &'a dyn ArenaPlanner,
}

impl<'a> DomainCtx<'a> {
    pub fn new(caps: &'a Caps, planner: &'a dyn ArenaPlanner) -> Self {
        Self { caps, planner }
    }
}

/// Hard ceiling on split-K candidates. Bounds the candidate count; it is not
/// a profitability judgement.
pub const MAX_SPLITS: u32 = 64;

/// A process-wide memo for a shape-independent candidate table.
///
/// The heavy generators (`coop::candidate_geoms_for`, `sgemm::sgemm_domain`)
/// are pure functions of `(Caps, element, planner)` — no extent reaches them
/// — so the enumeration runs once per device, not once per node.
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

/// Rank of a point no bench has ever visited. Far from the measured band and
/// below 255, so a future seed table can still order below it.
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
    (t.dim.unwrap_or(u32::MAX), t.tm, t.vector)
}

/// The [`ArenaPlanner`] the rules in [`crate::rules`] reach for: the one
/// memoized [`crate::Planner`], the same object `verify_launch` admits against
/// and the Kernel emitter lays out with, so a geometry this crate admits
/// passes `verify_launch` unchanged.
pub fn default_planner() -> &'static dyn ArenaPlanner {
    crate::Planner::global()
}
