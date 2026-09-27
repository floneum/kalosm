//! [`Roofline`]: per launch `launch + max(dram, occupancy * (math + wg),
//! serial) + occupancy * drain`. Math and workgroup traffic share issue slots,
//! so they sum; DRAM overlaps them.

use crate::nodes::{Mnkb, Tiling, fold_lane_group, fold_theta};
use crate::realize::fold_line_amplification;
use crate::terms;
use fusor_ir::cost::{CostModel, DeviceFacts, LaunchPlan, MacUnit, Picoseconds};
use fusor_ir::dtype::Dtype;
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::launch::{Launch, SchedPoint};
use fusor_ir::ir::{Node, Op};

/// The one cost model.
pub struct Roofline {
    facts: DeviceFacts,
}

impl Roofline {
    pub fn new(facts: DeviceFacts) -> Self {
        Self { facts }
    }

    /// Line traffic past the useful bytes a fold moves at `theta`.
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

impl CostModel for Roofline {
    fn facts(&self) -> &DeviceFacts {
        &self.facts
    }

    /// [`LaunchPlan`] carries no dtype, so a launch prices at f32.
    fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds {
        let f = &self.facts;
        let dtype = Dtype::F32;
        // Single staging loses the load/MMA overlap the rates were fitted on;
        // the drain is per emitting subgroup.
        let (unit, staging, subgroups) = match launch.theta.get(&launch.root) {
            Some(SchedPoint::Coop { geom, staging }) => {
                (MacUnit::Coop, *staging, (geom.rg * geom.cg).max(1))
            }
            Some(SchedPoint::Sgemm(p)) => (MacUnit::Fma, if p.double_buffer { 2 } else { 1 }, 1),
            Some(SchedPoint::Sgemv(p)) => (MacUnit::Fma, 2, p.subgroups.max(1)),
            _ => (MacUnit::Fma, 2, 1),
        };
        let elem_bytes = dtype.byte_size().max(1);

        let math = terms::math_ps(f, launch.work, unit, dtype);
        let wg = terms::wg_ps(f, launch.work.wg_bytes, staging);
        let (num, den) = terms::occupancy_scale_num_den(f, launch.resident_lanes);
        let issue = terms::scaled(math + wg, num, den);

        // `writes` is the padded output the launch stores.
        let padded_out_elems = launch.writes / elem_bytes;
        let drain = terms::scaled(
            terms::drain_ps(
                f,
                padded_out_elems,
                subgroups,
                u32::try_from(launch.wg_bytes).unwrap_or(u32::MAX),
                f.caps.limits.max_compute_workgroup_storage_size,
            ),
            num,
            den,
        );

        // A launch resident on a fraction of the device cannot keep DRAM
        // busy either.
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
        // MACs issue at the operand dtype, not `acc`.
        let dtype = ins.first().map_or(out.dtype, |f| f.dtype);
        let unit = match theta {
            Some(SchedPoint::Coop { .. }) => MacUnit::Coop,
            _ => MacUnit::Fma,
        };
        let mut work = fusor_ir::semantics::work::work_of(&node.op, ins, out);
        // A tiled point issues MACs on the padded tile.
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
        // No traffic or occupancy, to stay an admissible bound; a fold's line
        // amplification is the one memory floor every plan through it pays.
        let t =
            terms::math_ps(&self.facts, work, unit, dtype) + self.fold_line_floor(node, ins, theta);
        // So is a workgroup's dependent chain.
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
