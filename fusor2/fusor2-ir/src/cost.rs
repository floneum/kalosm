//! One scalar, picoseconds, on a roofline — and the facts it is parameterized
//! on.

use crate::device::Caps;
use crate::dtype::Dtype;
use crate::egraph::Id;
use crate::extract::{Extraction, PlanHash};
use crate::facts::{ValueFacts, Work};
use crate::ir::Node;
use crate::ir::level1::SchedPoint;
use crate::shape::Dims;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// Modelled time in picoseconds. One scalar, not a lexicographic tuple: the
/// reference's own unit test shows the tuple gives the wrong verdict, and
/// its own doc concedes dispatches are 0.2% of modelled time while the
/// tuple will pay unbounded bandwidth to remove one.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Picoseconds(pub u64);

impl std::ops::Add for Picoseconds {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}
impl std::ops::AddAssign for Picoseconds {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}
impl std::ops::Sub for Picoseconds {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}
impl std::iter::Sum for Picoseconds {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0), |a, b| a + b)
    }
}

/// Which functional unit issues a MAC. Indexes [`DeviceFacts::mac_per_us`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum MacUnit {
    Fma = 0,
    Coop = 1,
    Dp4a = 2,
}

impl MacUnit {
    pub const ALL: [MacUnit; 3] = [MacUnit::Fma, MacUnit::Coop, MacUnit::Dp4a];
}

/// Dtype slots in [`DeviceFacts::mac_per_us`], in a fixed order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum RateDtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    U32 = 3,
    I32 = 4,
}

impl RateDtype {
    /// Quantized formats price at their dequantized compute dtype.
    pub const fn of(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F32 | Dtype::Q(_) => Self::F32,
            Dtype::F16 => Self::F16,
            Dtype::BF16 => Self::BF16,
            Dtype::U32 => Self::U32,
            Dtype::I32 => Self::I32,
        }
    }
    pub const COUNT: usize = 5;
}

/// The device rates the cost model prices its terms in, built by
/// `fusor2-cost::facts::seed_facts` from the [`Caps`] a backend reports.
/// The table is per device *class* and physically dimensioned, which is what
/// keeps it portable: the reference picked five integers fitted on one M2 Max
/// by an adapter-name string test and shared them with every other GPU on
/// earth.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceFacts {
    pub launch_ps: u64,
    pub dram_bytes_per_us: u64,
    /// Feeds the continuous LLC reread term *and* the grid swizzle term —
    /// one source, no private constants.
    pub llc_bytes: u64,
    pub wg_bytes_per_us: u64,
    pub mac_per_us: [[u64; RateDtype::COUNT]; 3],
    pub trans_ps: u64,
    /// Accumulator zeroing, fragment shuffles and the store, per padded
    /// output element per emitting subgroup. `score_fs`'s T3.
    pub store_ps_per_element: u64,
    pub saturation_lanes: u32,
    pub single_buffered_traffic_pct: u32,
    pub compile_ps_per_kernel: u64,
    /// Cost of waking the CPU worker pool for one parallel region.
    /// Replaces `PARALLEL_THRESHOLD = 16_777_216`.
    pub thread_wake_ps: u64,
    pub caps: Caps,
}

impl DeviceFacts {
    pub fn mac_rate(&self, unit: MacUnit, dtype: Dtype) -> u64 {
        self.mac_per_us[unit as usize][RateDtype::of(dtype) as usize].max(1)
    }

    /// Digest folded into `PlanHash` and the calibration cache key.
    /// Includes `max_compute_workgroup_storage_size` via [`Caps`].
    pub fn fingerprint(&self) -> u64 {
        let mut h = FxHasher::default();
        self.hash(&mut h);
        h.finish()
    }
}

/// One launch in the realized DAG: a connected component cut at
/// materialization boundaries, index-space mismatches and merged waves.
/// `reads` is `(bytes, reread_factor)` per distinct operand; `wg_bytes`
/// comes from the exact `ArenaPlan::total_bytes`.
#[derive(Clone, Debug)]
pub struct LaunchPlan<'a> {
    pub members: &'a [Id],
    pub root: Id,
    pub theta: &'a FxHashMap<Id, SchedPoint>,
    pub reads: &'a [(u64, u32)],
    pub writes: u64,
    pub work: Work,
    pub resident_lanes: u64,
    pub wg_bytes: u64,
    pub grid: [u32; 3],
}

/// The cost model. Object-safe: extraction holds it as `&dyn CostModel`.
/// Every method returns picoseconds so terms are commensurable. Precision
/// is **not** a cost term — it is a verifier property
/// (`NumericContract`), because a time-only model eliminates f32 everywhere.
pub trait CostModel: Send + Sync {
    fn facts(&self) -> &DeviceFacts;

    /// `launch_ps + max(dram_ps, math_ps, wg_ps) + drain_ps`.
    fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds;

    /// Arithmetic cost of one node at one schedule point, ignoring traffic.
    /// The admissible lower bound is built from this.
    fn node_math(
        &self,
        node: &Node,
        ins: &[ValueFacts],
        out: &ValueFacts,
        theta: Option<SchedPoint>,
    ) -> Picoseconds;

    /// Traffic for `bytes` read `rereads` times. Continuous in `llc_bytes`,
    /// not a strict `>` cliff.
    fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds;

    /// `compile_ps_per_kernel / expected_reuse(plan, binding)`.
    fn compile_amortized(&self, plan: PlanHash, expected_reuse: u32) -> Picoseconds;

    /// Total cost of a realized extraction. The accept test for every
    /// local-search move is this, never a local delta heuristic.
    fn total(&self, extraction: &Extraction, launches: &[LaunchPlan<'_>]) -> Picoseconds;
}

/// Bounded per-process record of which dim bindings a plan has been seen
/// at, so specialization is a decision recorded in the key rather than an
/// accident of shape. On first sighting the generic symbolic variant wins
/// outright — nothing compiles per length bucket.
#[derive(Default, Debug, Clone)]
pub struct ShapeStats {
    seen: FxHashMap<PlanHash, SmallVec<[(Dims, u32); 8]>>,
}

impl ShapeStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, plan: PlanHash, binding: &[crate::shape::Dim]) -> u32 {
        let entry = self.seen.entry(plan).or_default();
        if let Some(slot) = entry.iter_mut().find(|(d, _)| d.as_slice() == binding) {
            slot.1 += 1;
            return slot.1;
        }
        if entry.len() < 8 {
            entry.push((binding.iter().copied().collect(), 1));
        }
        1
    }

    /// `1` on first sighting, so nothing compiles speculatively.
    pub fn expected_reuse(&self, plan: PlanHash, binding: &[crate::shape::Dim]) -> u32 {
        self.seen
            .get(&plan)
            .and_then(|e| e.iter().find(|(d, _)| d.as_slice() == binding))
            .map_or(1, |(_, n)| *n)
    }
}

