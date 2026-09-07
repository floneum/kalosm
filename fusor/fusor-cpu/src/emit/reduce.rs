//! Reduction identities shared by lowering and the native tree emitter.

use fusor_ir::ir::kernel::TileReduceOp;

#[inline(always)]
pub(crate) const fn identity_f32(op: TileReduceOp) -> f32 {
    match op {
        TileReduceOp::Sum => 0.0,
        TileReduceOp::Product => 1.0,
        TileReduceOp::Max => f32::NEG_INFINITY,
        TileReduceOp::Min => f32::INFINITY,
    }
}
