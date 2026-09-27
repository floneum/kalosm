//! Read-only views of graph nodes shared by pricing, extraction and
//! verification.

use crate::realize::dim_extent;
use fusor_ir::device::Caps;
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::extract::{Extraction, Plan};
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{AccessPlan, Launch, SchedPoint, ScheduleDomain, SgemvParams};
use fusor_ir::scalar::ScalarKind;
use fusor_ir::shape::Dim;
use smallvec::SmallVec;

/// A composite's members: a slab's stages or a group's launches.
pub(crate) fn composite_members(graph: &EGraph, id: Id) -> Option<&SmallVec<[Id; 8]>> {
    match &graph.node(id).op {
        Op::Launch(Launch::Slab { members, .. } | Launch::Group { members, .. }) => Some(members),
        _ => None,
    }
}

pub(crate) fn is_composite(graph: &EGraph, id: Id) -> bool {
    composite_members(graph, id).is_some()
}

pub(crate) fn is_group(graph: &EGraph, id: Id) -> bool {
    matches!(graph.node(id).op, Op::Launch(Launch::Group { .. }))
}

/// An identity map reading one strided view of another value.
pub(crate) fn is_view_copy(op: &Op) -> bool {
    matches!(
        op,
        Op::Launch(Launch::Map { body, ops, .. })
            if ops.len() == 1
                && matches!(body.kind(), ScalarKind::Arg(0))
                && matches!(ops[0].access, AccessPlan::Alias)
    )
}

/// The schedule domain a launch node carries.
pub(crate) fn domain_of(graph: &EGraph, id: Id) -> Option<&ScheduleDomain> {
    match &graph.node(id).op {
        Op::Launch(launch) => launch.schedule(),
        _ => None,
    }
}

/// `id`'s operands as the selection resolves them, paired with the operand:
/// a composite names its members by id, anything else reads its operand's
/// selected member, `None` when that class has none.
pub(crate) fn resolved_children<'a>(
    graph: &'a EGraph,
    ex: &'a Extraction,
    id: Id,
) -> impl Iterator<Item = (Id, Option<Id>)> + 'a {
    let by_id = is_composite(graph, id);
    graph.node(id).children.iter().map(move |&c| {
        let resolved = if by_id {
            Some(c)
        } else {
            ex.selected(graph.class_of(c))
        };
        (c, resolved)
    })
}

/// Every value a plan touches: each launch's members, then its bindings.
pub(crate) fn plan_values(plan: &Plan) -> impl Iterator<Item = Id> + '_ {
    plan.launches.iter().flat_map(|l| {
        l.members
            .iter()
            .copied()
            .chain(l.bindings.iter().map(|b| b.value))
    })
}

/// A contraction's extents.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Mnkb {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub batch: u64,
}

impl Mnkb {
    /// A `Contract`'s extents at `extent`; `None` for any other node.
    pub(crate) fn of(op: &Op, extent: impl Fn(Dim) -> u64) -> Option<Self> {
        let Op::Launch(Launch::Contract { m, n, k, batch, .. }) = op else {
            return None;
        };
        Some(Self {
            m: extent(*m),
            n: extent(*n),
            k: extent(*k),
            batch: extent(*batch),
        })
    }

    /// The extents the realized cost prices at: nominal symbols, never zero.
    pub(crate) fn priced(op: &Op) -> Option<Self> {
        Self::of(op, |d| dim_extent(d).max(1))
    }
}

/// Output tiles of a `bm x bn` tiling over an `m x n` matrix.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Tiling {
    pub bm: u64,
    pub bn: u64,
    pub tiles_m: u64,
    pub tiles_n: u64,
}

impl Tiling {
    pub(crate) fn new(m: u64, n: u64, bm: u32, bn: u32) -> Self {
        let (bm, bn) = (u64::from(bm.max(1)), u64::from(bn.max(1)));
        Self {
            bm,
            bn,
            tiles_m: m.div_ceil(bm),
            tiles_n: n.div_ceil(bn),
        }
    }

    pub(crate) fn padded_m(&self) -> u64 {
        self.tiles_m.saturating_mul(self.bm)
    }

    pub(crate) fn padded_n(&self) -> u64 {
        self.tiles_n.saturating_mul(self.bn)
    }

    /// Workgroups over `batch` matrices, one per tile.
    pub(crate) fn groups(&self, batch: u64) -> u64 {
        batch
            .saturating_mul(self.tiles_m)
            .saturating_mul(self.tiles_n)
    }
}

/// Lanes of a one-column sgemv workgroup: `subgroups` subgroups, capped by
/// the device.
pub(crate) fn sgemv_block(p: SgemvParams, caps: &Caps) -> u32 {
    (p.subgroups.max(1) * caps.subgroup_width())
        .min(caps.limits.max_compute_invocations_per_workgroup)
        .max(1)
}

/// Lanes one sgemv output reduces its k across: multi-column schedules
/// reduce each column within one subgroup, the one-column path across the
/// block.
pub(crate) fn sgemv_lanes(p: SgemvParams, caps: &Caps) -> u32 {
    if p.cols > 1 {
        caps.subgroup_width()
    } else {
        sgemv_block(p, caps)
    }
}

/// The point a fold lowers at: its GPU fold strategy, else `theta` itself.
pub(crate) fn fold_theta(
    op: &Launch,
    theta: Option<SchedPoint>,
    caps: &Caps,
) -> Option<SchedPoint> {
    op.fold_schedule(theta, caps)
        .map(|s| SchedPoint::Fold(s.strategy))
        .or(theta)
}

/// The lane group a fold lowers at under `theta`. A point that is not a fold
/// strategy takes the emitters' default, `emitted_block(1)`.
pub(crate) fn fold_lane_group(theta: Option<SchedPoint>, caps: &Caps) -> u32 {
    match theta {
        Some(SchedPoint::Fold(s)) => s.lane_group(caps.subgroup_width()),
        _ => fusor_ir::ir::launch::emitted_block(1, caps),
    }
}
