//! Operand ids of every `Op`, in the order inference, verification, work
//! accounting and the cost model all expect. The one place that order is
//! written down.

use crate::ir::logical::Logical;
use crate::ir::launch::Launch;
use crate::ir::{Children, Op};

/// Operand ids of `op`. `Op::Union(a, b)` yields `[a, b]`.
pub fn children_of(op: &Op) -> Children {
    match op {
        Op::Logical(o) => children_logical(o),
        Op::Launch(o) => children_launch(o),
        Op::Union(a, b) => Children::from_slice(&[*a, *b]),
    }
}

/// Operand ids of an Logical node.
pub fn children_logical(op: &Logical) -> Children {
    match op {
        Logical::Leaf(_) => Children::new(),
        Logical::Map { ins, .. } => ins.iter().copied().collect(),
        Logical::Fold { ins, .. } => ins.iter().copied().collect(),
        Logical::Contract { a, b, .. } => Children::from_slice(&[*a, *b]),
        Logical::Restride { x, .. } => Children::from_slice(&[*x]),
        Logical::Window { x, .. } => Children::from_slice(&[*x]),
        Logical::Gather { x, idx, .. } => Children::from_slice(&[*x, *idx]),
        Logical::Scatter { base, idx, upd, .. } => Children::from_slice(&[*base, *idx, *upd]),
        Logical::Dequant { x, .. } => Children::from_slice(&[*x]),
        Logical::Project { x, .. } => Children::from_slice(&[*x]),
    }
}

/// Operand ids of an Launch node, taken from its `Operand` lists. `Contract`
/// is its A-side operands followed by its B-side ones — one each in the
/// two-buffer case that reads `[a.src, b.src]`, more once a multi-edge
/// producer has been absorbed. A region is its members and a merged wave is
/// its segments.
pub fn children_launch(op: &Launch) -> Children {
    match op {
        Launch::Map { ops, .. }
        | Launch::Fold { ops, .. }
        | Launch::Gather { ops, .. }
        | Launch::Scatter { ops, .. }
        | Launch::Ext { ops, .. } => ops.iter().map(|o| o.src).collect(),
        Launch::Contract { a, b, .. } => a
            .ops
            .iter()
            .chain(b.ops.iter())
            .map(|o| o.src)
            .collect(),
        Launch::Region { members, .. } => members.iter().copied().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::egraph::Id;
    use crate::carrier::Carrier;
    use crate::ir::logical::{LeafKind, ScatterCombine, TiePolicy};
    use crate::scalar::BinOp;
    use crate::ir::launch::{
        AccessPlan, IndexSpace, Operand, ScheduleDomain,
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
            children_logical(&Logical::Leaf(LeafKind::Const {
                value: crate::dtype::Splat::F32(0.0),
                shape: smallvec![],
            }))
            .is_empty()
        );

        assert_eq!(
            &children_logical(&Logical::Map {
                expr: ScalarExpr::arg(0, Dtype::F32),
                ins: smallvec![Id(1), Id(2), Id(3)],
                outs: 1,
            })[..],
            &[Id(1), Id(2), Id(3)]
        );

        assert_eq!(
            &children_logical(&Logical::Scatter {
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
            &children_logical(&Logical::Gather {
                axis: 0,
                x: Id(4),
                idx: Id(5)
            })[..],
            &[Id(4), Id(5)]
        );

        assert_eq!(
            &children_logical(&Logical::Fold {
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
            &children_launch(&Launch::Map {
                space: IndexSpace::new([Dim::Const(4)]),
                body: ScalarExpr::arg(0, Dtype::F32),
                ops,
                sched: ScheduleDomain::Point,
            })[..],
            &[Id(3), Id(4)]
        );
    }

    #[test]
    fn union_children() {
        assert_eq!(&children_of(&Op::Union(Id(1), Id(2)))[..], &[Id(1), Id(2)]);
    }
}
