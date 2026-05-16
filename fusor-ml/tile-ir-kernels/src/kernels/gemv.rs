//! Dense GEMV program kernels.

use fusor_tile_ir::tile::{Program, Storage, Tile};
use fusor_tile_ir::{TileLiteral, TileReduceOp, WorkgroupAxis, F32};

use crate::grid::dot4_sum;
use crate::types::matrix_shape;

/// Dense F32 GEMV: a single-output-column matmul specialized for the
/// K-reduction along the input vector.
pub fn gemv<const ROWS_PER_WORKGROUP: usize, const VALUES_PER_LANE: usize, const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    x: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
) {
    const SUBGROUP_SIZE: u32 = 32;
    assert!(
        ROWS_PER_WORKGROUP > 0 && VALUES_PER_LANE > 0,
        "gemv tile shape must be non-zero"
    );
    assert_eq!(
        ROWS_PER_WORKGROUP as u32 * SUBGROUP_SIZE,
        BLOCK as u32,
        "gemv maps one output row to each subgroup"
    );
    assert!(
        VALUES_PER_LANE.is_multiple_of(4),
        "gemv values per lane must be divisible by dot4 width"
    );
    let [m, k] = matrix_shape(&a.view().layout);
    let [x_k, n] = matrix_shape(&x.view().layout);
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(k, x_k, "gemv K dimensions must match");
    assert_eq!(n, 1, "gemv expects a single RHS column");
    assert_eq!(m, y_m, "gemv output row count must match A");
    assert_eq!(y_n, 1, "gemv output must have a single column");

    let k_per_iter = SUBGROUP_SIZE * VALUES_PER_LANE as u32;
    let k_iterations = k.div_ceil(k_per_iter);
    program.program_grid::<BLOCK>([m.div_ceil(ROWS_PER_WORKGROUP as u32), 1, 1], |program| {
        let row = program.program_id(WorkgroupAxis::X) * ROWS_PER_WORKGROUP as u32
            + program.subgroup_id();
        let lane = program.subgroup_lane();
        let row_in_bounds = row.lt(m);
        let zero = TileLiteral::f32(0.0);
        let [sum] = program.loop_fold_n::<1, _>(
            TileReduceOp::Sum,
            k_iterations,
            [zero],
            |program, loop_index| {
                let k_base = loop_index * k_per_iter + lane.clone() * VALUES_PER_LANE as u32;
                let a_values: [Tile<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let k_index = k_base.clone() + i as u32;
                    let mask = row_in_bounds.clone().and(k_index.lt(k));
                    program.load(a.at((&row, k_index)), mask, 0.0)
                });
                let x_values: [Tile<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let k_index = k_base.clone() + i as u32;
                    program.load(x.at((k_index.clone(), 0)), k_index.lt(k), 0.0)
                });
                [dot4_sum(program, &a_values, &x_values)]
            },
        );
        let reduced = program.subgroup_reduce_sum(sum);
        let mask = lane.eq(0).and(row_in_bounds);
        program.store(y.at((row, 0)), reduced, mask);
    });
}
