//! Recognize a consecutive index vector from its scalar expression.
use fusor_ir::dtype::Splat;
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{AccessPlan, Launch, Operand};
use fusor_ir::ir::logical::Logical;
use fusor_ir::scalar::{BinOp, ScalarExpr, ScalarKind};
use fusor_ir::shape::Dim;
use fusor_ir::target::LowerCtx;

fn range_start(expr: &ScalarExpr) -> Option<u64> {
    match expr.kind() {
        ScalarKind::IndexOf(0) => Some(0),
        ScalarKind::Bin {
            op: BinOp::Add,
            a,
            b,
        } => {
            let literal = |e: &ScalarExpr| match e.kind() {
                ScalarKind::Lit(lit) if matches!(lit.0, Splat::U32(_)) => {
                    Some(u64::from(lit.0.bits()))
                }
                _ => None,
            };
            range_start(a)
                .zip(literal(b))
                .or_else(|| range_start(b).zip(literal(a)))
                .and_then(|(a, b)| a.checked_add(b))
        }
        _ => None,
    }
}

/// The origin of `index_of(0) + constant` through an unchanged one-dimensional
/// operand. Callers must additionally prove that the complete range does not
/// wrap and fits their destination before replacing elementwise indexing.
pub fn index_start(cx: &LowerCtx<'_>, operand: &Operand) -> Option<u64> {
    if operand.layout.shape().len() != 1
        || operand.layout.offset() != Dim::Const(0)
        || operand.layout.strides() != [Dim::ONE]
        || operand.access != AccessPlan::Alias
    {
        return None;
    }

    match &cx.graph.node(cx.selected(operand.src)).op {
        Op::Launch(Launch::Map { body, .. }) => range_start(body),
        Op::Logical(Logical::Map { expr, .. }) => range_start(expr),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_unit_stride_unsigned_index_sums_have_an_origin() {
        let index = ScalarExpr::index_of(0);
        let offset = |x| ScalarExpr::lit(Splat::U32(x));
        let add = |a, b| ScalarExpr::bin(BinOp::Add, a, b);
        assert_eq!(range_start(&index), Some(0));
        assert_eq!(range_start(&add(offset(7), index.clone())), Some(7));
        assert_eq!(
            range_start(&add(add(index.clone(), offset(7)), offset(13))),
            Some(20)
        );
        // Keep overflow visible so callers can reject the entire range; never
        // confuse a wrapped unsigned expression with an ordinary slice.
        assert_eq!(
            range_start(&add(add(index.clone(), offset(u32::MAX)), offset(1))),
            Some(1u64 << 32)
        );
        for other in [
            offset(4),
            ScalarExpr::index_of(1),
            add(index.clone(), index.clone()),
            ScalarExpr::bin(BinOp::Mul, index.clone(), offset(2)),
        ] {
            assert_eq!(range_start(&other), None);
        }
    }
}
