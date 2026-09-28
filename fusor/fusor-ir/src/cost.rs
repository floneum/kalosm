//! One scalar, picoseconds, on a roofline — and the facts it is parameterized
//! on.

use crate::device::Caps;
use crate::dtype::Dtype;
use crate::egraph::Id;
use crate::facts::{ValueFacts, Work};
use crate::ir::Node;
use crate::ir::launch::SchedPoint;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::{Hash, Hasher};

/// Modelled time in picoseconds. One scalar, not a lexicographic tuple.
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

/// The device rates the cost model prices its terms in, per device class and
/// physically dimensioned, built by `fusor-cost::facts::seed_facts`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceFacts {
    pub launch_ps: u64,
    pub dram_bytes_per_us: u64,
    /// Feeds both the LLC reread term and the grid swizzle term.
    pub llc_bytes: u64,
    pub wg_bytes_per_us: u64,
    pub mac_per_us: [[u64; RateDtype::COUNT]; 3],
    pub trans_ps: u64,
    /// Accumulator zeroing, shuffles and store, per padded output element
    /// per emitting subgroup.
    pub store_ps_per_element: u64,
    pub saturation_lanes: u32,
    pub single_buffered_traffic_pct: u32,
    /// Cost of waking the CPU worker pool for one parallel region.
    pub thread_wake_ps: u64,
    /// Latency of one dependent step of a tiled contraction's k loop.
    pub coop_step_ps: u64,
    /// Latency of one dependent step of a per-lane loop.
    pub lane_step_ps: u64,
    /// Floor per launched lane, idle or not: an over-launched grid pays for
    /// every invocation it schedules.
    pub lane_launch_ps: u64,
    pub caps: Caps,
}

impl DeviceFacts {
    pub fn mac_rate(&self, unit: MacUnit, dtype: Dtype) -> u64 {
        self.mac_per_us[unit as usize][RateDtype::of(dtype) as usize].max(1)
    }

    /// Digest folded into `PlanHash` and the calibration cache key.
    pub fn fingerprint(&self) -> u64 {
        let mut h = FxHasher::default();
        self.hash(&mut h);
        h.finish()
    }
}

/// One launch in the realized DAG. `reads` is `(bytes, reread_factor)` per
/// distinct operand.
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
    /// Cache-line traffic beyond the useful bytes of uncoalesced reads.
    pub line_bytes: u64,
    /// The longest dependent chain one workgroup runs, which occupancy
    /// cannot shorten.
    pub coop_steps: u64,
    pub lane_steps: u64,
    pub grid: [u32; 3],
}

/// The cost model, object-safe, in picoseconds. Precision is a verifier
/// property (`NumericContract`), never a cost term.
pub trait CostModel: Send + Sync {
    fn facts(&self) -> &DeviceFacts;

    /// `launch_ps + max(dram_ps, math_ps, wg_ps) + drain_ps`.
    fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds;

    /// Arithmetic cost of one node at one schedule point, ignoring traffic.
    fn node_math(
        &self,
        node: &Node,
        ins: &[ValueFacts],
        out: &ValueFacts,
        theta: Option<SchedPoint>,
    ) -> Picoseconds;

    /// Traffic for `bytes` read `rereads` times, continuous in `llc_bytes`.
    fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds;

    /// Total cost of a realized extraction: every search move's accept test.
    fn total(&self, launches: &[LaunchPlan<'_>]) -> Picoseconds;
}
