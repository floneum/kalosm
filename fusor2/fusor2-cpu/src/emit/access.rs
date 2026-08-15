//! The four strided-access lowerings, selected **once at compile time** from
//! the `TileLayout` / `MultiFlattenMap`, never per vector. The form is a
//! property of the emitted `Instr`, so the inner loop has no branch.

use fusor2_ir::ir::level2::{Addr, Builtin, TileExpr, TileExprKind, TileLayout, TileLiteral};
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::MultiFlattenMap;
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
pub static FORM_COUNTS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub fn note_form(form: AccessForm) {
    FORM_COUNTS[form.index()].fetch_add(1, Ordering::Relaxed);
}

pub fn form_counts() -> [u64; 4] {
    core::array::from_fn(|i| FORM_COUNTS[i].load(Ordering::Relaxed))
}

pub fn reset_form_counts() {
    for c in &FORM_COUNTS {
        c.store(0, Ordering::Relaxed);
    }
}

/// The affine dependence of an index expression on the lane index:
/// `index = base + coeff * lane`, when that shape can be proven statically.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LaneAffine {
    pub coeff: i64,
    /// True when `base` is itself lane-invariant (it usually is: a program id,
    /// a loop index, a uniform).
    pub base_uniform: bool,
}

/// Prove `e = base + coeff*lane` with a lane-invariant `base`, or report the
/// general case. Purely structural — no runtime probe.
pub fn lane_affine(e: &TileExpr) -> Option<LaneAffine> {
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
pub fn is_lane_uniform(e: &TileExpr) -> bool {
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
pub fn const_u32(e: &TileExpr) -> Option<u32> {
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
pub fn form_of(layout: &TileLayout, addr: &Addr) -> AccessForm {
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
pub fn apply_group(map: &MultiFlattenMap, axis: usize, coord: u32) -> u32 {
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
pub fn rc2_offset(map: &MultiFlattenMap, row: u32, col: u32) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::ir::level2::{ElementType, MemoryLevel, ScalarElement};
    use fusor2_ir::shape::{AxisGroup, SubAxis};

    fn lane() -> TileExpr {
        TileExpr::new(
            TileExprKind::Builtin(Builtin::Lane),
            ElementType::Scalar(ScalarElement::U32),
        )
    }
    fn lit(v: u32) -> TileExpr {
        TileExpr::new(
            TileExprKind::Literal(TileLiteral::U32(v)),
            ElementType::Scalar(ScalarElement::U32),
        )
    }
    fn bin(op: BinOp, a: TileExpr, b: TileExpr) -> TileExpr {
        TileExpr::new(
            TileExprKind::Binary {
                op,
                left: a,
                right: b,
                numeric: fusor2_ir::dtype::NumericContract::RELAXED,
            },
            ElementType::Scalar(ScalarElement::U32),
        )
    }

    #[test]
    fn lane_affine_recognises_the_easy_forms() {
        assert_eq!(lane_affine(&lane()).unwrap().coeff, 1);
        assert_eq!(lane_affine(&lit(7)).unwrap().coeff, 0);
        assert_eq!(
            lane_affine(&bin(BinOp::Add, lit(7), lane())).unwrap().coeff,
            1
        );
        assert_eq!(
            lane_affine(&bin(BinOp::Mul, lane(), lit(4))).unwrap().coeff,
            4
        );
    }

    #[test]
    fn form_of_distinguishes_all_four() {
        let l = TileLayout::contiguous(MemoryLevel::Storage, &[8, 8]);
        assert_eq!(form_of(&l, &Addr::Linear(lane())), AccessForm::Contiguous);
        assert_eq!(form_of(&l, &Addr::Linear(lit(3))), AccessForm::Broadcast);
        assert_eq!(
            form_of(&l, &Addr::Linear(bin(BinOp::Add, lit(64), lane()))),
            AccessForm::UnitInnerStride
        );
        assert_eq!(
            form_of(&l, &Addr::Linear(bin(BinOp::Mul, lane(), lit(3)))),
            AccessForm::Gather
        );
    }

    #[test]
    fn divmod_chain_matches_a_hand_decomposition() {
        // One logical axis of extent 6 decomposed as 2 x 3, strides 100 and 1.
        let map = MultiFlattenMap {
            groups: smallvec::smallvec![AxisGroup {
                sub_axes: smallvec::smallvec![
                    SubAxis {
                        extent: 2,
                        stride: 100
                    },
                    SubAxis {
                        extent: 3,
                        stride: 1
                    }
                ],
            }],
        };
        for c in 0..6u32 {
            assert_eq!(apply_group(&map, 0, c), (c / 3) * 100 + (c % 3));
        }
    }
}
