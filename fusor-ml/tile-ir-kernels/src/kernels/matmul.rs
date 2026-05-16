//! Dense matrix multiply program kernels.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{TileLiteral, TileReduceOp, WorkgroupAxis, F32};

use crate::types::{apply_optional_epilogue, matrix_shape, DenseMatmulEpilogues};

/// Dense F32 matmul, scalar lane-mapped body.
///
/// ```
/// use fusor_tile_ir::{tile, Shape, F32};
/// use fusor_tile_ir_kernels::matmul;
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([8, 64]));
///     let b = program.storage_read::<F32, 2>(Shape::new([64, 16]));
///     let y = program.storage_write::<F32, 2>(Shape::new([8, 16]));
///     matmul::<64>(program, &a, &b, &y);
/// });
/// # let _ = ir;
/// ```
pub fn matmul<const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
) {
    matmul_with_epilogues::<BK>(program, a, b, y, &DenseMatmulEpilogues::empty());
}

/// Dense F32 matmul with optional pre/post element-wise epilogues.
pub fn matmul_with_epilogues<const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
    epilogues: &DenseMatmulEpilogues<'_>,
) {
    const TILE_ROWS: usize = 16;
    const TILE_COLS: usize = 16;
    const LANES: usize = TILE_ROWS * TILE_COLS;
    assert!(BK > 0, "matmul K tile shape must be non-zero");
    let [m, k] = matrix_shape(&a.view().layout);
    let [b_k, n] = matrix_shape(&b.view().layout);
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(k, b_k, "matmul K dimensions must match");
    assert_eq!(m, y_m, "matmul output row count must match A");
    assert_eq!(n, y_n, "matmul output column count must match B");

    program.program_grid::<LANES>(
        [
            n.div_ceil(TILE_COLS as u32),
            m.div_ceil(TILE_ROWS as u32),
            1,
        ],
        |program| {
            let [tile_row, tile_col] = program.lane_tiles(&[TILE_ROWS as u32, TILE_COLS as u32]);
            let row = program.program_id(WorkgroupAxis::Y) * TILE_ROWS as u32 + tile_row;
            let col = program.program_id(WorkgroupAxis::X) * TILE_COLS as u32 + tile_col;
            let mask = row.lt(m).and(col.lt(n));
            let sum = program.loop_fold(
                TileReduceOp::Sum,
                k,
                TileLiteral::f32(0.0),
                |program, k_index| {
                    let a_value = apply_optional_epilogue(
                        epilogues.pre_a,
                        program.load(a.at((&row, &k_index)), mask.clone(), 0.0),
                    );
                    let b_value = apply_optional_epilogue(
                        epilogues.pre_b,
                        program.load(b.at((&k_index, &col)), mask.clone(), 0.0),
                    );
                    a_value * b_value
                },
            );
            let sum = apply_optional_epilogue(epilogues.post, sum);
            program.store(y.at((row, col)), sum, mask);
        },
    );
}
