//! Constant-folding helpers used by the index lowering fast paths.
//!
//! After the unification of `TileIndexExpr` into `Expr`, the lowerer can no
//! longer pattern-match `TileIndexExpr::Mul(_, u32)` to recognize a divisor or
//! multiplier that is statically known. These helpers re-extract that
//! information from the unified expression so the fast paths in
//! `lower::indexing` keep emitting the same Naga code.
use crate::ir::{Expr, TileLiteral};

/// Returns `Some(n)` when `expr` is statically a `u32` literal. Used by the
/// index lowering to recognize const-divisor / const-multiplier shapes after
/// unification.
#[allow(dead_code)]
pub(crate) fn as_const_u32(expr: &Expr) -> Option<u32> {
    match expr {
        Expr::Literal(TileLiteral::U32(value)) => Some(*value),
        _ => None,
    }
}
