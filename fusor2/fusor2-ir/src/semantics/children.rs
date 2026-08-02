//! Operand ids of every `Op`, in the order inference, verification, work
//! accounting and the cost model all expect. The one place that order is
//! written down.
//!
//! Owned by W1.

use crate::ir::level0::L0;
use crate::ir::level1::L1;
use crate::ir::{Children, Op};

/// Operand ids of `op`. `Op::Union(a, b)` yields `[a, b]`.
pub fn children_of(op: &Op) -> Children {
    match op {
        Op::L0(o) => children_l0(o),
        Op::L1(o) => children_l1(o),
        Op::Union(a, b) => Children::from_slice(&[*a, *b]),
    }
}

/// Operand ids of an L0 node.
pub fn children_l0(op: &L0) -> Children {
    match op {
        L0::Leaf(_) => Children::new(),
        L0::Map { ins, .. } => ins.iter().copied().collect(),
        L0::Fold { ins, .. } => ins.iter().copied().collect(),
        L0::Contract { a, b, .. } => Children::from_slice(&[*a, *b]),
        L0::Restride { x, .. } => Children::from_slice(&[*x]),
        L0::Window { x, .. } => Children::from_slice(&[*x]),
        L0::Gather { x, idx, .. } => Children::from_slice(&[*x, *idx]),
        L0::Scatter { base, idx, upd, .. } => Children::from_slice(&[*base, *idx, *upd]),
        L0::Dequant { x, .. } => Children::from_slice(&[*x]),
        L0::Project { x, .. } => Children::from_slice(&[*x]),
    }
}

/// Operand ids of an L1 node, taken from its `Operand` lists. `KContract`
/// and `KQContract` are `[a.src, b.src]`; a region is its members and a
/// merged wave is its segments.
pub fn children_l1(op: &L1) -> Children {
    match op {
        L1::KMap { ops, .. }
        | L1::KFold { ops, .. }
        | L1::KGather { ops, .. }
        | L1::KScatter { ops, .. }
        | L1::Ext { ops, .. } => ops.iter().map(|o| o.src).collect(),
        L1::KContract { a, b, .. } | L1::KQContract { a, b, .. } => {
            Children::from_slice(&[a.src, b.src])
        }
        L1::KRegion { members, .. } => members.iter().copied().collect(),
        L1::KMerged(m) => m.segments().iter().copied().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::egraph::Id;
    use crate::carrier::Carrier;
    use crate::ir::level0::{LeafKind, ScatterCombine, TiePolicy};
    use crate::scalar::BinOp;
    use crate::ir::level1::{
        AccessPlan, IndexSpace, KMerged, MergeKey, MergeSegment, Operand, ScheduleDomain, WaveCat,
    };
    use crate::scalar::ScalarExpr;
    use crate::shape::{Dim, Layout};
    use smallvec::smallvec;

    fn operand(src: u32) -> Operand {
        Operand {
            src: Id(src),
            layout: Layout::contiguous(&[Dim::Const(4)]),
            access: AccessPlan::Alias,
        }
    }

    #[test]
    fn l0_operand_order() {
        assert!(
            children_l0(&L0::Leaf(LeafKind::Const {
                value: crate::dtype::Splat::F32(0.0),
                shape: smallvec![],
            }))
            .is_empty()
        );

        assert_eq!(
            &children_l0(&L0::Map {
                expr: ScalarExpr::arg(0, Dtype::F32),
                ins: smallvec![Id(1), Id(2), Id(3)],
                outs: 1,
            })[..],
            &[Id(1), Id(2), Id(3)]
        );

        assert_eq!(
            &children_l0(&L0::Scatter {
                axis: 0,
                combine: ScatterCombine::Add,
                base: Id(7),
                idx: Id(8),
                upd: Id(9),
                unique: false,
            })[..],
            &[Id(7), Id(8), Id(9)]
        );

        assert_eq!(
            &children_l0(&L0::Gather {
                axis: 0,
                x: Id(4),
                idx: Id(5)
            })[..],
            &[Id(4), Id(5)]
        );

        assert_eq!(
            &children_l0(&L0::Fold {
                carrier: Carrier::binop(
                    BinOp::Max,
                    Carrier::binop_identity(BinOp::Max, Dtype::F32).unwrap(),
                    Dtype::F32,
                )
                .with_tie(TiePolicy::FirstWins),
                axis: 0,
                acc: Dtype::F32,
                ins: smallvec![Id(2)],
            })[..],
            &[Id(2)]
        );
    }

    #[test]
    fn l1_operand_order() {
        let ops = vec![operand(3), operand(4)];
        assert_eq!(
            &children_l1(&L1::KMap {
                space: IndexSpace::new([Dim::Const(4)]),
                body: ScalarExpr::arg(0, Dtype::F32),
                ops,
                sched: ScheduleDomain::Point,
            })[..],
            &[Id(3), Id(4)]
        );

        let key = MergeKey {
            m: Dim::Const(4),
            n: Dim::Const(4),
            k: Dim::Const(4),
            batch: Dim::Const(1),
            splits: 1,
            dtype: Dtype::F32,
            family: crate::ir::level1::Family::Sgemm,
        };
        let merged = KMerged::new(
            WaveCat::Matmul,
            [
                MergeSegment {
                    id: Id(11),
                    key,
                    has_epilogue: false,
                },
                MergeSegment {
                    id: Id(12),
                    key,
                    has_epilogue: false,
                },
            ],
            ScheduleDomain::Point,
        )
        .unwrap();
        assert_eq!(&children_l1(&L1::KMerged(merged))[..], &[Id(11), Id(12)]);
    }

    #[test]
    fn union_children() {
        assert_eq!(&children_of(&Op::Union(Id(1), Id(2)))[..], &[Id(1), Id(2)]);
    }
}
