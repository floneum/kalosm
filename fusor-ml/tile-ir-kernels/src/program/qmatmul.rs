//! Quantized matrix multiply program kernels.

use fusor_tile_ir::tile::{CoopAcc, CoopFragment, Program, ScalarIndex, Storage, TileBlock};
use fusor_tile_ir::{F32, QuantizedMatrix, TileLiteral, TileReduceOp, TileRef, WorkgroupAxis};

use crate::types::{apply_optional_epilogue, cooperative_store_layout_supported, matrix_shape};

/// Top-level quantized matrix multiply.
///
/// Picks the qgemv path when `m == 1`, otherwise the scalar/cooperative tiled
/// qmatmul body.
///
/// ```
/// use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
/// use fusor_tile_ir_kernels::{qmatmul, quantized_matrix};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([8, 256]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 16);
///     let y = program.storage_write::<F32, 2>(Shape::new([8, 16]));
///     qmatmul::<8, 4, 8>(program, &a, &b, &y, 4);
/// });
/// # let _ = ir;
/// ```
pub fn qmatmul<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
) {
    qmatmul_options::<BM, BN, BK>(program, a, b, y, vector_width, true, 1);
}

/// Top-level quantized matrix multiply with optional activation/output
/// epilogues. Single-row inputs keep using qgemv; multi-row inputs use the
/// generalized qmatmul body.
pub fn qmatmul_with_epilogue<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    qmatmul_options_with_epilogue::<BM, BN, BK>(program, a, b, y, vector_width, true, 1, epilogues);
}

pub(crate) fn qmatmul_options<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    use_qgemv: bool,
    workgroups_x: u32,
) {
    qmatmul_options_with_epilogue::<BM, BN, BK>(
        program,
        a,
        b,
        y,
        vector_width,
        use_qgemv,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

pub(crate) fn qmatmul_options_with_epilogue<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    use_qgemv: bool,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    assert!(
        BM > 0 && BN > 0 && BK > 0,
        "qmatmul tile shape must be non-zero"
    );
    assert!(vector_width > 0, "qmatmul vector width must be non-zero");
    assert!(workgroups_x > 0, "qmatmul workgroups_x must be non-zero");
    let [m, k] = matrix_shape(&a.view().layout);
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(k, b.rows, "qmatmul K dimensions must match");
    assert_eq!(m, y_m, "qmatmul output row count must match A");
    assert_eq!(b.cols, y_n, "qmatmul output column count must match B");

    if m == 1 && use_qgemv {
        super::qgemv::qgemv_with_epilogue::<BN, BK>(program, a, b, y, workgroups_x, epilogues);
    } else {
        qmatmul_tile_with_epilogue::<BM, BN, BK>(program, a, b, y, epilogues);
    }
}

/// Scalar lane-mapped qmatmul body. Public so downstream crates can reproduce
/// or replace the variant-selection layer above (`qmatmul_options` /
/// `qmatmul`).
pub(crate) fn qmatmul_tile<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
) {
    qmatmul_tile_with_epilogue::<BM, BN, BK>(
        program,
        a,
        b,
        y,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

pub(crate) fn qmatmul_tile_with_epilogue<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    const LANES: usize = 256;
    assert!(
        BM > 0 && BN > 0 && BK > 0,
        "qmatmul tile shape must be non-zero"
    );
    let [m, k] = matrix_shape(&a.view().layout);

    if epilogues.pre.is_none()
        && epilogues.post.is_none()
        && qmatmul_try_coop::<BM, BN, BK>(program, a, b, y)
    {
        return;
    }

    if BM * BN * BK != LANES || !BK.is_power_of_two() {
        qmatmul_tile::<8, 4, 8>(program, a, b, y);
        return;
    }
    let k_iterations = k.div_ceil(BK as u32);
    program.program_grid::<LANES>(
        [b.cols.div_ceil(BN as u32), m.div_ceil(BM as u32), 1],
        |program| {
            let lane = program.lane();
            let k_lane = lane.clone() % BK as u32;
            let output_lane = lane / BK as u32;
            let row_lane = output_lane.clone() / BN as u32;
            let col_lane = output_lane % BN as u32;
            let row = program.program_id(WorkgroupAxis::Y) * BM as u32 + row_lane;
            let col = program.program_id(WorkgroupAxis::X) * BN as u32 + col_lane;
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                TileLiteral::f32(0.0),
                |program, loop_index| {
                    let k_index = loop_index * BK as u32 + k_lane.clone();
                    let mask = row.lt(m).and(col.lt(b.cols)).and(k_index.lt(k));
                    let a_value = apply_optional_epilogue(
                        epilogues.pre,
                        program.load(a.at((&row, &k_index)), mask.clone(), 0.0),
                    );
                    let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
                    a_value * b_value
                },
            );
            let sum =
                apply_optional_epilogue(epilogues.post, program.group_reduce_sum::<BK>(partial));
            let store_mask = k_lane.eq(0).and(row.lt(m)).and(col.lt(b.cols));
            program.store(y.at((row, col)), sum, store_mask);
        },
    );
}

/// Emit the cooperative-matrix qmatmul body when the requested tile shape
/// matches a supported fast tile geometry. All branches instantiate the same
/// generic body; only the tile dimensions differ.
pub(crate) fn qmatmul_try_coop<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
) -> bool {
    let [m, k] = matrix_shape(&a.view().layout);
    if BK != 32
        || !(m as usize).is_multiple_of(BM)
        || !(b.cols as usize).is_multiple_of(BN)
        || !(k as usize).is_multiple_of(BK)
        || !cooperative_store_layout_supported(&y.view().layout)
    {
        return false;
    }
    match (BM, BN) {
        (64, 64) => qmatmul_perf::<64, 64, 32, 2, 2, 128>(program, a, b, y),
        (64, 128) => qmatmul_perf::<64, 128, 32, 2, 4, 256>(program, a, b, y),
        (128, 64) => qmatmul_perf::<128, 64, 32, 4, 2, 256>(program, a, b, y),
        (128, 128) => qmatmul_perf::<128, 128, 32, 4, 4, 512>(program, a, b, y),
        _ => return false,
    }
    true
}

fn zero_coop_acc_grid<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    rows: u32,
    cols: u32,
) -> Vec<Vec<CoopAcc<F32, 8, 8>>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let acc = program.alloc_coop_acc_typed::<F32, 8, 8>();
                    program.zero_coop_acc(&acc);
                    acc
                })
                .collect()
        })
        .collect()
}

fn coop_load_a_fragments<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    tile: TileRef,
    sg_row_base: &ScalarIndex,
    kk: u32,
    rows: u32,
) -> Vec<CoopFragment<F32, 8, 8>> {
    const COOP_DIM: u32 = 8;
    (0..rows)
        .map(|r| {
            program.coop_load_a_typed::<F32, 8, 8>(
                tile,
                sg_row_base.clone() + r * COOP_DIM,
                kk * COOP_DIM,
            )
        })
        .collect()
}

fn coop_load_b_fragments<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    tile: TileRef,
    sg_col_base: &ScalarIndex,
    kk: u32,
    cols: u32,
) -> Vec<CoopFragment<F32, 8, 8>> {
    const COOP_DIM: u32 = 8;
    (0..cols)
        .map(|c| {
            program.coop_load_b_typed::<F32, 8, 8>(
                tile,
                kk * COOP_DIM,
                sg_col_base.clone() + c * COOP_DIM,
            )
        })
        .collect()
}

fn coop_mma_grid<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    accs: &[Vec<CoopAcc<F32, 8, 8>>],
    a_frags: &[CoopFragment<F32, 8, 8>],
    b_frags: &[CoopFragment<F32, 8, 8>],
) {
    for (r, a) in a_frags.iter().enumerate() {
        for (c, b) in b_frags.iter().enumerate() {
            program.coop_mma(&accs[r][c], a, b);
        }
    }
}

fn coop_store_acc_grid<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    accs: &[Vec<CoopAcc<F32, 8, 8>>],
    y: &Storage<F32, 2>,
    row_base: &ScalarIndex,
    col_base: &ScalarIndex,
    sg_row_base: &ScalarIndex,
    sg_col_base: &ScalarIndex,
) {
    const COOP_DIM: u32 = 8;
    for (r, row_accs) in accs.iter().enumerate() {
        for (c, acc) in row_accs.iter().enumerate() {
            let row = row_base.clone() + sg_row_base.clone() + r as u32 * COOP_DIM;
            let col = col_base.clone() + sg_col_base.clone() + c as u32 * COOP_DIM;
            program.coop_store(acc, y, row, col);
        }
    }
}

/// Cooperative-matrix qmatmul body. Each workgroup produces one BMxBN output
/// tile via an interleaved `ROW_GROUPS x COL_GROUPS` grid of subgroups, each
/// holding `(32*32)/(8*8)` = 16 cooperative-matrix accumulators.
/// `BLOCK == ROW_GROUPS * COL_GROUPS * 32`.
pub(crate) fn qmatmul_perf<
    const BM: usize,
    const BN: usize,
    const BK: usize,
    const ROW_GROUPS: u32,
    const COL_GROUPS: u32,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    const SUBGROUP_ROWS: u32 = 32;
    const SUBGROUP_COLS: u32 = 32;
    debug_assert_eq!(ROW_GROUPS * SUBGROUP_ROWS, BM as u32);
    debug_assert_eq!(COL_GROUPS * SUBGROUP_COLS, BN as u32);
    debug_assert_eq!(ROW_GROUPS * COL_GROUPS * SUBGROUP_SIZE, BLOCK as u32);

    let [m, k] = matrix_shape(&a.view().layout);
    let n = b.cols;
    let n_grid_x = n / BN as u32;
    let n_grid_y = m / BM as u32;
    let k_iterations = k / BK as u32;

    let a_tile = program.alloc_workgroup_tile_f32(BM as u32, BK as u32);
    let b_tile = program.alloc_workgroup_tile_f32(BK as u32, BN as u32);
    let b_clone = b.clone();
    let a_clone = a;
    let y_clone = y;

    const TILE_ROWS_PER_SG: u32 = SUBGROUP_ROWS / 8;
    const TILE_COLS_PER_SG: u32 = SUBGROUP_COLS / 8;

    program.program_grid::<BLOCK>([n_grid_x, n_grid_y, 1], |program| {
        let row_base = program.program_id(WorkgroupAxis::Y) * BM as u32;
        let col_base = program.program_id(WorkgroupAxis::X) * BN as u32;
        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / COL_GROUPS;
        let sg_col = subgroup_id % COL_GROUPS;
        let sg_row_base = sg_row * SUBGROUP_ROWS;
        let sg_col_base = sg_col * SUBGROUP_COLS;

        let accs = zero_coop_acc_grid(program, TILE_ROWS_PER_SG, TILE_COLS_PER_SG);

        program.while_true(k_iterations, |program, loop_index| {
            let k_base = loop_index * BK as u32;
            program.copy_storage_to_tile(a_tile, a_clone, &row_base, &k_base);
            program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &col_base);
            program.workgroup_barrier();

            let kk_steps = (BK as u32) / COOP_DIM;
            for kk in 0..kk_steps {
                let a_frags =
                    coop_load_a_fragments(program, a_tile, &sg_row_base, kk, TILE_ROWS_PER_SG);
                let b_frags =
                    coop_load_b_fragments(program, b_tile, &sg_col_base, kk, TILE_COLS_PER_SG);
                coop_mma_grid(program, &accs, &a_frags, &b_frags);
            }
            program.workgroup_barrier();
        });

        coop_store_acc_grid(
            program,
            &accs,
            y_clone,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base,
        );
    });
}
