//! [`Roofline`] — the [`CostModel`] implementation.
//!
//! Per launch:
//! `launch_ps + max(dram_ps, occupancy * (math_ps + wg_ps)) + occupancy *
//! drain_ps_ps`.
//!
//! T1 and T2 are **summed** inside the `max` because they contend for the
//! same per-core issue and load/store slots; DRAM overlaps them; the combine
//! dispatch sits behind its own barrier and adds.
//!
//! One scalar. Precision is a construction invariant (`NumericContract`), not a
//! cost term, because a time-only model eliminates f32 everywhere.

use crate::nodes::{Mnkb, Tiling, fold_lane_group, fold_theta};
use crate::realize::fold_line_amplification;
use crate::terms;
use fusor_ir::cost::{CostModel, DeviceFacts, LaunchPlan, MacUnit, Picoseconds};
use fusor_ir::dtype::Dtype;
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::launch::{Launch, SchedPoint};
use fusor_ir::ir::{Node, Op};

/// The one cost model. `score_fs` maps onto its terms one for one:
/// T1 -> math, T2 -> wg, T3 -> drain, T4 -> the `max`. Split-K combine
/// work is an ordinary reduction launch in the graph.
pub struct Roofline {
    facts: DeviceFacts,
}

/// The schedule-dependent inputs one launch's terms need, decoded from its
/// root's [`SchedPoint`].
#[derive(Copy, Clone, Debug)]
struct Sched {
    unit: MacUnit,
    /// 1 loses the load/MMA overlap the threadgroup rate was fitted on.
    staging: u8,
    /// Emitting subgroups per workgroup — the epilogue drain is per element
    /// *and* per subgroup.
    subgroups: u32,
}

impl Default for Sched {
    fn default() -> Self {
        Self {
            unit: MacUnit::Fma,
            staging: 2,
            subgroups: 1,
        }
    }
}

impl Sched {
    fn of(theta: Option<SchedPoint>) -> Self {
        match theta {
            Some(SchedPoint::Coop { geom, staging }) => Self {
                unit: MacUnit::Coop,
                staging,
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
        Self { facts }
    }

    /// Line traffic beyond the useful bytes a fold moves at `theta`'s lane
    /// group, over the operands it walks at its own iteration space.
    fn fold_line_floor(
        &self,
        node: &Node,
        ins: &[ValueFacts],
        theta: Option<SchedPoint>,
    ) -> Picoseconds {
        let Op::Launch(op @ Launch::Fold { space, axis, .. }) = &node.op else {
            return Picoseconds(0);
        };
        let Some(total) = space.iterations() else {
            return Picoseconds(0);
        };
        let dims: Vec<u64> = space.dims.iter().filter_map(|d| d.as_const()).collect();
        let caps = &self.facts.caps;
        let lane_group = fold_lane_group(fold_theta(op, theta, caps), caps);
        let mut extra = 0u64;
        for f in ins {
            let elems = f
                .shape
                .iter()
                .try_fold(1u64, |a, d| d.as_const().map(|d| a * d));
            if elems != Some(total) {
                continue;
            }
            let elem = f.dtype.byte_size().max(1);
            let amp = fold_line_amplification(&dims, *axis as usize, lane_group, caps, elem);
            extra = extra.saturating_add(total.saturating_mul(elem).saturating_mul(amp - 1));
        }
        terms::dram_ps(&self.facts, &[], 0, extra)
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

    /// [`LaunchPlan`] carries no dtype, so a launch prices at f32.
    fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds {
        let f = &self.facts;
        let dtype = Dtype::F32;
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

        // Bandwidth is not free of parallelism: a launch resident on a
        // fraction of the device cannot keep DRAM busy, and the shortfall
        // that throttles issue throttles the memory pipe too. Without it a
        // reduction is priced on bytes alone, so every lane group reads the
        // same total and occupancy never enters the comparison.
        let dram = terms::scaled(
            terms::dram_ps(f, launch.reads, launch.writes, launch.line_bytes),
            num,
            den,
        );
        // What one workgroup cannot finish faster than: its dependent chain.
        let serial = terms::serial_ps(f, launch.coop_steps, launch.lane_steps);
        Picoseconds(f.launch_ps) + dram.max(issue).max(serial) + drain
    }

    fn node_math(
        &self,
        node: &Node,
        ins: &[ValueFacts],
        out: &ValueFacts,
        theta: Option<SchedPoint>,
    ) -> Picoseconds {
        let (unit, dtype) = unit_and_dtype(ins, out, theta);
        let mut work = fusor_ir::semantics::work::work_of(&node.op, ins, out);
        // A tiled point issues MACs on the *padded* tile: the kernels stage
        // zero-filled tiles and run the whole tile's MACs.
        let tile = match theta {
            Some(SchedPoint::Coop { geom, .. }) => Some((geom.bm, geom.bn)),
            Some(SchedPoint::Sgemm(p)) => Some((p.bm, p.bn)),
            _ => None,
        };
        if let (Some((bm, bn)), Some(c)) = (
            tile,
            Mnkb::of(&node.op, |d| d.as_const().unwrap_or(1).max(1)),
        ) {
            let t = Tiling::new(c.m, c.n, bm, bn);
            let extra = t
                .padded_m()
                .saturating_mul(t.padded_n())
                .saturating_sub(c.m.saturating_mul(c.n))
                .saturating_mul(c.k)
                .saturating_mul(c.batch);
            work.macs = work.macs.saturating_add(extra);
        }
        // Zero traffic, no occupancy scaling. The admissible lower bound is
        // built from this, and either addition would break admissibility.
        // The one memory term that is a floor of the node itself: a fold's
        // line amplification at this point, which every plan through the
        // point pays whatever it inlines around it.
        let t =
            terms::math_ps(&self.facts, work, unit, dtype) + self.fold_line_floor(node, ins, theta);
        // A workgroup's dependent chain is a floor of the node at this point
        // too: nothing around it shortens the k loop.
        if crate::debug::flags().no_seed_floor {
            return t;
        }
        let (coop, lane) = crate::realize::node_serial_steps(&node.op, theta, &self.facts.caps);
        t.max(terms::serial_ps(&self.facts, coop, lane))
    }

    fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds {
        terms::dram_ps(&self.facts, &[(bytes, rereads)], 0, 0)
    }

    fn total(&self, launches: &[LaunchPlan<'_>]) -> Picoseconds {
        launches.iter().map(|launch| self.launch_cost(launch)).sum()
    }
}
