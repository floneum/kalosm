use super::*;
use crate::ir::KernelIr;

/// Build a standalone tile IR.
///
/// This is the smallest entry point when runtime binding ownership is handled
/// by the caller. Use [`crate::KernelBuilder`] instead when you want tile-ir to
/// return the binding list alongside the IR.
///
/// ```
/// use fusor_tile_ir::{tile, Shape, TileLiteral, F32};
///
/// let ir = tile::build(|program| {
///     let x = program.storage_read::<F32, 1>(Shape::new([32]));
///     let y = program.storage_write::<F32, 1>(Shape::new([32]));
///     program.program_grid::<32>([1, 1, 1], |block| {
///         let lane = block.lane();
///         let mask = lane.clone().lt(32u32);
///         let value = block.load_linear(x.at(lane.clone()), mask.clone(), TileLiteral::f32(0.0));
///         block.store_linear(y.at(lane), value, mask);
///     });
/// });
/// # let _ = ir;
/// ```
pub fn build(f: impl FnOnce(&mut Program)) -> KernelIr {
    let mut program = Program::new();
    f(&mut program);
    program.ir
}
