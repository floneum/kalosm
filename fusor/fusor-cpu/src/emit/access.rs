//! The four strided-access lowerings, selected **once at compile time** from
//! the `TileLayout` / `MultiFlattenMap`, never per vector. The form is a
//! property of the emitted `Instr`, so the inner loop has no branch.

use fusor_ir::ir::kernel::{Addr, Builtin, TileExpr, TileExprKind, TileLayout, TileLiteral};
use fusor_ir::scalar::BinOp;
use fusor_ir::shape::MultiFlattenMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// How one operand's lanes are gathered into a register.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AccessForm {
    /// Unit stride from the lane index: one vector load.
    Contiguous,
    /// Stride 0: splat of one scalar.
    Broadcast,
    /// Unit stride from an outer base: a contiguous sub-slice at an offset.
    UnitInnerStride,
    /// Anything else: an index vector built from the declared divmod chain,
    /// then a gather.
    Gather,
}

impl AccessForm {
    pub const ALL: [AccessForm; 4] = [
        AccessForm::Contiguous,
        AccessForm::Broadcast,
        AccessForm::UnitInnerStride,
        AccessForm::Gather,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Contiguous => 0,
            Self::Broadcast => 1,
            Self::UnitInnerStride => 2,
            Self::Gather => 3,
        }
    }
}

/// Per-form selection counter, so `four_access_lowerings` can assert each case
/// picked its own compiled form rather than all four collapsing to `Gather`.
pub(crate) static FORM_COUNTS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub(crate) fn note_form(form: AccessForm) {
    FORM_COUNTS[form.index()].fetch_add(1, Ordering::Relaxed);
}

/// The affine dependence of an index expression on the lane index:
/// `index = base + coeff * lane`, when that shape can be proven statically.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct LaneAffine {
    pub coeff: i64,
    /// True when `base` is itself lane-invariant (it usually is: a program id,
    /// a loop index, a uniform).
    pub base_uniform: bool,
}

/// Prove `e = base + coeff*lane` with a lane-invariant `base`, or report the
/// general case. Purely structural — no runtime probe.
pub(crate) fn lane_affine(e: &TileExpr) -> Option<LaneAffine> {
    fn go(e: &TileExpr) -> Option<(i64, bool)> {
        use TileExprKind as K;
        Some(match e.kind() {
            K::Builtin(Builtin::Lane) | K::Builtin(Builtin::SubgroupLane) => (1, true),
            K::Literal(_) | K::LoadLocal(_) | K::Builtin(_) => (0, true),
            K::Binary {
                op, left, right, ..
            } => {
                let (lc, lu) = go(left)?;
                let (rc, ru) = go(right)?;
                let uniform = lu && ru;
                match op {
                    BinOp::Add => (lc.checked_add(rc)?, uniform),
                    BinOp::Sub => (lc.checked_sub(rc)?, uniform),
                    BinOp::Mul => {
                        if rc == 0 && lc != 0 {
                            (lc.checked_mul(const_u32(right)? as i64)?, uniform)
                        } else if lc == 0 && rc != 0 {
                            (rc.checked_mul(const_u32(left)? as i64)?, uniform)
                        } else if lc == 0 && rc == 0 {
                            (0, uniform)
                        } else {
                            return None;
                        }
                    }
                    _ => {
                        if lc == 0 && rc == 0 {
                            (0, uniform)
                        } else {
                            return None;
                        }
                    }
                }
            }
            _ => return None,
        })
    }
    let (coeff, base_uniform) = go(e)?;
    Some(LaneAffine {
        coeff,
        base_uniform,
    })
}

/// Whether an expression can vary across lanes of one chunk. Conservative: a
/// `Load` may read anything, so it counts as divergent.
pub(crate) fn is_lane_uniform(e: &TileExpr) -> bool {
    use TileExprKind as K;
    match e.kind() {
        K::Builtin(Builtin::Lane)
        | K::Builtin(Builtin::SubgroupLane)
        | K::Builtin(Builtin::SubgroupId) => false,
        K::Literal(_) | K::Builtin(_) | K::LoadLocal(_) => true,
        K::Load { .. } | K::LoadTile { .. } => false,
        K::Unary { value, .. } => is_lane_uniform(value),
        K::Binary { left, right, .. } | K::Compare { left, right, .. } => {
            is_lane_uniform(left) && is_lane_uniform(right)
        }
        K::Round { value, .. } | K::Cast { value, .. } | K::Bitcast { value, .. } => {
            is_lane_uniform(value)
        }
        K::Select {
            condition,
            accept,
            reject,
        } => is_lane_uniform(condition) && is_lane_uniform(accept) && is_lane_uniform(reject),
        K::Vec { parts, .. } => parts.iter().all(is_lane_uniform),
        K::VecComponent { vector, .. } => is_lane_uniform(vector),
        K::Dot { left, right } => is_lane_uniform(left) && is_lane_uniform(right),
        // A cross-lane reduce is uniform across its group by construction.
        K::Reduce { .. } => true,
        K::CoopLoad { .. } | K::CoopMma { .. } | K::CoopZero { .. } => false,
    }
}

/// Constant-fold a u32-valued expression, when it is one.
pub(crate) fn const_u32(e: &TileExpr) -> Option<u32> {
    match e.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => Some(*v),
        TileExprKind::Literal(TileLiteral::I32(v)) => Some(*v as u32),
        _ => None,
    }
}

/// Pick the narrowest form an index expression admits under `layout`.
///
/// `Addr::Linear` addresses the buffer's elements directly, so the layout only
/// contributes when the address is rank-2 (`Addr::Rc2`), where the row and
/// column coordinates run through the `MultiFlattenMap`'s divmod chains.
pub(crate) fn form_of(layout: &TileLayout, addr: &Addr) -> AccessForm {
    match addr {
        Addr::Linear(e) => match lane_affine(e) {
            Some(LaneAffine { coeff: 0, .. }) => AccessForm::Broadcast,
            Some(LaneAffine {
                coeff: 1,
                base_uniform: true,
            }) => {
                if is_bare_lane(e) {
                    AccessForm::Contiguous
                } else {
                    AccessForm::UnitInnerStride
                }
            }
            _ => AccessForm::Gather,
        },
        Addr::Rc2 { row, col } => {
            let unit_col = layout
                .indexing
                .groups
                .get(1)
                .is_some_and(|g| g.sub_axes.len() == 1 && g.sub_axes[0].stride == 1);
            let row_uniform = matches!(lane_affine(row), Some(LaneAffine { coeff: 0, .. }));
            let col_aff = lane_affine(col);
            match (unit_col, row_uniform, col_aff) {
                (_, true, Some(LaneAffine { coeff: 0, .. })) => AccessForm::Broadcast,
                (true, true, Some(LaneAffine { coeff: 1, .. })) => AccessForm::UnitInnerStride,
                _ => AccessForm::Gather,
            }
        }
    }
}

fn is_bare_lane(e: &TileExpr) -> bool {
    matches!(e.kind(), TileExprKind::Builtin(Builtin::Lane))
}

/// Divmod one logical coordinate through an `AxisGroup`'s sub-axes,
/// most-significant-first. Only the divmods the map actually declares are
/// performed — `divmod_ops()` is the cost term the extractor prices, so
/// emitting more than that would make the model wrong.
#[inline(always)]
pub(crate) fn apply_group(map: &MultiFlattenMap, axis: usize, coord: u32) -> u32 {
    let group = &map.groups[axis];
    if group.sub_axes.len() == 1 {
        return coord.wrapping_mul(group.sub_axes[0].stride);
    }
    let mut rest = coord;
    let mut acc = 0u32;
    for (i, sub) in group.sub_axes.iter().enumerate() {
        let below: u32 = group.sub_axes[i + 1..]
            .iter()
            .map(|s| s.extent)
            .product::<u32>()
            .max(1);
        let q = rest / below;
        rest %= below;
        acc = acc.wrapping_add(q.wrapping_mul(sub.stride));
    }
    acc
}

/// Physical element offset of a rank-2 address.
#[inline(always)]
pub(crate) fn rc2_offset(map: &MultiFlattenMap, row: u32, col: u32) -> u32 {
    let r = if map.groups.is_empty() {
        row
    } else {
        apply_group(map, 0, row)
    };
    let c = if map.groups.len() > 1 {
        apply_group(map, 1, col)
    } else {
        col
    };
    r.wrapping_add(c)
}
