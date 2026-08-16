//! [`Roofline`] — the [`CostModel`] implementation.
//!
//! Per launch:
//! `launch_ps + max(dram_ps, occupancy * (math_ps + wg_ps)) + occupancy *
//! drain_ps + combine_ps`.
//!
//! T1 and T2 are **summed** inside the `max` because they contend for the
//! same per-core issue and load/store slots; DRAM overlaps them; the combine
//! dispatch sits behind its own barrier and adds.
//!
//! One scalar. Precision is a verifier property (`NumericContract`), not a
//! cost term, because a time-only model eliminates f32 everywhere.

use crate::terms;
use fusor2_ir::cost::{CostModel, DeviceFacts, LaunchPlan, MacUnit, Picoseconds, ShapeStats};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::extract::{Extraction, PlanHash};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::launch::SchedPoint;
use fusor2_ir::ir::Node;
use fusor2_ir::shape::Dim;
use parking_lot::RwLock;
use rustc_hash::{FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};

/// The one cost model. `score_fs` maps onto its terms one for one:
/// T1 -> math, T2 -> wg, T3 -> drain, T4 -> the `max`, T5 -> the combine
/// launch.
pub struct Roofline {
    facts: DeviceFacts,
    stats: RwLock<ShapeStats>,
}

/// The schedule-dependent inputs one launch's terms need, decoded from its
/// root's [`SchedPoint`].
#[derive(Copy, Clone, Debug)]
struct Sched {
    unit: MacUnit,
    /// 1 loses the load/MMA overlap the threadgroup rate was fitted on.
    staging: u8,
    splits: u32,
    /// Emitting subgroups per workgroup — the epilogue drain is per element
    /// *and* per subgroup.
    subgroups: u32,
}

impl Default for Sched {
    fn default() -> Self {
        Self {
            unit: MacUnit::Fma,
            staging: 2,
            splits: 1,
            subgroups: 1,
        }
    }
}

impl Sched {
    fn of(theta: Option<SchedPoint>) -> Self {
        match theta {
            Some(SchedPoint::Coop {
                geom,
                splits,
                staging,
            }) => Self {
                unit: MacUnit::Coop,
                staging,
                splits,
                subgroups: (geom.rg * geom.cg).max(1),
            },
            Some(SchedPoint::Sgemm(p)) => Self {
                staging: if p.double_buffer { 2 } else { 1 },
                ..Self::default()
            },
            Some(SchedPoint::Sgemv(p)) => Self {
                subgroups: p.subgroups.max(1),
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

impl Roofline {
    pub fn new(facts: DeviceFacts) -> Self {
        Self {
            facts,
            stats: RwLock::new(ShapeStats::new()),
        }
    }

    /// Record that this plan ran at this dim binding, and return how many
    /// times that pair has now been seen.
    ///
    /// Also bumps a plan-level counter (the empty binding), which is what
    /// [`CostModel::total`] amortizes compilation against — a plan is
    /// compiled once per plan, not once per binding.
    pub fn observe_binding(&self, plan: PlanHash, binding: &[Dim]) -> u32 {
        let mut stats = self.stats.write();
        stats.observe(plan, &[]);
        stats.observe(plan, binding)
    }

    /// How many times this plan has been seen at any binding. `1` on first
    /// sighting, so nothing compiles speculatively and the generic symbolic
    /// variant wins outright.
    pub fn expected_reuse(&self, plan: PlanHash) -> u32 {
        self.stats.read().expected_reuse(plan, &[])
    }

    /// The compile identity of one launch: its root, its members, the
    /// schedule points they resolved to, whether the root is materialized,
    /// and the device fingerprint.
    ///
    /// This is a *stand-in*. The authoritative `PlanHash` is
    /// `plan::plan_hash` over the whole realized term; a launch alone
    /// cannot see that term.
    pub fn launch_plan_hash(&self, launch: &LaunchPlan<'_>, materialized: bool) -> PlanHash {
        let mut h = FxHasher::default();
        self.facts.fingerprint().hash(&mut h);
        launch.root.hash(&mut h);
        materialized.hash(&mut h);
        for id in launch.members {
            id.hash(&mut h);
            launch.theta.get(id).hash(&mut h);
        }
        launch.grid.hash(&mut h);
        let lo = h.finish();
        // A second lane so a 64-bit collision is not a plan collision.
        let mut h2 = FxHasher::default();
        (lo, 0x9e37_79b9_7f4a_7c15u64).hash(&mut h2);
        launch.work.hash(&mut h2);
        PlanHash((u128::from(h2.finish()) << 64) | u128::from(lo))
    }

    /// [`CostModel::launch_cost`] at an explicit operand dtype.
    ///
    /// [`LaunchPlan`] carries no dtype, so the trait method assumes f32.
    /// Callers that know better — every real lowering does — should come
    /// through here, so an f16 contraction is priced at the f16 MAC rate.
    pub fn launch_cost_at(&self, launch: &LaunchPlan<'_>, dtype: Dtype) -> Picoseconds {
        let f = &self.facts;
        let sched = Sched::of(launch.theta.get(&launch.root).copied());
        let elem_bytes = dtype.byte_size().max(1);

        let math = terms::math_ps(f, launch.work, sched.unit, dtype);
        let wg = terms::wg_ps(f, launch.work.wg_bytes, sched.staging);
        let (num, den) = terms::occupancy_scale_num_den(f, launch.resident_lanes);
        let issue = terms::scaled(math + wg, num, den);

        // `writes` is the padded output the launch actually stores,
        // including every split's full tile — exactly the reference's
        // `workgroups * bm * bn` once divided by the element size.
        let padded_out_elems = launch.writes / elem_bytes;
        let drain = terms::scaled(
            terms::drain_ps(
                f,
                padded_out_elems,
                sched.subgroups,
                u32::try_from(launch.wg_bytes).unwrap_or(u32::MAX),
                f.caps.limits.max_compute_workgroup_storage_size,
            ),
            num,
            den,
        );

        let dram = terms::dram_ps(f, launch.reads, launch.writes);
        // One split's padded output; `(splits + 1)` then counts reading
        // every partial and writing the result.
        let combine = terms::combine_ps(
            f,
            sched.splits,
            launch.writes / u64::from(sched.splits.max(1)),
        );

        Picoseconds(f.launch_ps) + dram.max(issue) + drain + combine
    }
}

/// Which functional unit and dtype a node's MACs issue on.
fn unit_and_dtype(
    ins: &[ValueFacts],
    out: &ValueFacts,
    theta: Option<SchedPoint>,
) -> (MacUnit, Dtype) {
    // MACs issue at the operand dtype; `acc` is a separate attribute and
    // does not set the issue rate.
    let dtype = ins.first().map_or(out.dtype, |f| f.dtype);
    match theta {
        Some(SchedPoint::Coop { .. }) => (MacUnit::Coop, dtype),
        _ => (MacUnit::Fma, dtype),
    }
}

impl CostModel for Roofline {
    fn facts(&self) -> &DeviceFacts {
        &self.facts
    }

    fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds {
        self.launch_cost_at(launch, Dtype::F32)
    }

    fn node_math(
        &self,
        node: &Node,
        ins: &[ValueFacts],
        out: &ValueFacts,
        theta: Option<SchedPoint>,
    ) -> Picoseconds {
        let (unit, dtype) = unit_and_dtype(ins, out, theta);
        let mut work = fusor2_ir::semantics::work::work_of(&node.op, ins, out);
        // A tiled point issues MACs on the *padded* tile. Padding is real
        // work the theta performs — the kernels stage zero-filled tiles and
        // run the whole tile's MACs — so charging it keeps the bound
        // admissible.
        if let (
            Some(tile),
            fusor2_ir::ir::Op::Launch(fusor2_ir::ir::launch::Launch::Contract {
                m, n, k, batch, ..
            }),
        ) = (
            theta.and_then(|t| match t {
                SchedPoint::Coop { geom, .. } => Some((geom.bm, geom.bn)),
                SchedPoint::Sgemm(p) => Some((p.bm, p.bn)),
                _ => None,
            }),
            &node.op,
        )
        {
            let geom_bm = tile.0;
            let geom_bn = tile.1;
            let priced = |d: &Dim| d.as_const().unwrap_or(1).max(1);
            let (m, n, k, batch) = (priced(m), priced(n), priced(k), priced(batch));
            let m_pad = m
                .div_ceil(u64::from(geom_bm.max(1)))
                .saturating_mul(u64::from(geom_bm.max(1)));
            let n_pad = n
                .div_ceil(u64::from(geom_bn.max(1)))
                .saturating_mul(u64::from(geom_bn.max(1)));
            let extra = m_pad
                .saturating_mul(n_pad)
                .saturating_sub(m.saturating_mul(n))
                .saturating_mul(k)
                .saturating_mul(batch);
            work.macs = work.macs.saturating_add(extra);
        }
        // Zero traffic, no occupancy scaling. The admissible lower bound is
        // built from this, and either addition would break admissibility.
        terms::math_ps(&self.facts, work, unit, dtype)
    }

    fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds {
        terms::dram_ps(&self.facts, &[(bytes, rereads)], 0)
    }

    fn compile_amortized(&self, plan: PlanHash, expected_reuse: u32) -> Picoseconds {
        let _ = plan;
        Picoseconds(self.facts.compile_ps_per_kernel / u64::from(expected_reuse.max(1)))
    }

    fn total(&self, extraction: &Extraction, launches: &[LaunchPlan<'_>]) -> Picoseconds {
        let mut total = Picoseconds(0);
        let mut compiled = FxHashSet::default();
        for launch in launches {
            total += self.launch_cost(launch);
            let hash = self.launch_plan_hash(launch, extraction.is_materialized(launch.root));
            if compiled.insert(hash) {
                total += self.compile_amortized(hash, self.expected_reuse(hash));
            }
        }
        total
    }
}
