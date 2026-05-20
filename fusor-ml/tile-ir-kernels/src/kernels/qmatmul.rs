//! Quantized matrix multiply program kernels.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{QuantizedMatrix, TileLiteral, TileReduceOp, WorkgroupAxis, F32};

use crate::{
    kernels::helpers::{
        coop_load_a_fragments, coop_load_b_fragments, coop_mma_grid, coop_store_acc_grid,
        zero_coop_acc_grid,
    },
    types::{
        apply_qmatmul_post_epilogue, apply_qmatmul_pre_epilogue,
        cooperative_store_layout_supported, matrix_shape,
    },
};

/// Top-level quantized matrix multiply with optional activation/output
/// epilogues. Single-row inputs keep using qgemv; multi-row inputs use the
/// generalized qmatmul body. Callers with no epilogue pass
/// `&QmatmulEpilogues::empty()`.
///
/// ```
/// use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
/// use fusor_tile_ir_kernels::{qmatmul_with_epilogue, quantized_matrix, QmatmulEpilogues};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([8, 256]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 16);
///     let y = program.storage_write::<F32, 2>(Shape::new([8, 16]));
///     qmatmul_with_epilogue::<8, 4, 8>(program, &a, &b, &y, 4, &QmatmulEpilogues::empty());
/// });
/// # let _ = ir;
/// ```
pub fn qmatmul_with_epilogue<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    assert!(
        BM > 0 && BN > 0 && BK > 0,
        "qmatmul tile shape must be non-zero"
    );
    assert!(vector_width > 0, "qmatmul vector width must be non-zero");
    let [m, k] = matrix_shape(&a.view().layout);
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(k, b.rows, "qmatmul K dimensions must match");
    assert_eq!(m, y_m, "qmatmul output row count must match A");
    assert_eq!(b.cols, y_n, "qmatmul output column count must match B");

    if m == 1 {
        super::qgemv::qgemv_with_epilogue::<BN, BK>(program, a, b, y, 1, epilogues);
    } else {
        qmatmul_tile_with_epilogue::<BM, BN, BK>(program, a, b, y, epilogues);
    }
}

/// Scalar lane-mapped qmatmul body with optional pre/post epilogues. Public
/// so downstream crates can reproduce or replace the variant-selection layer
/// above (`qmatmul_options_with_epilogue` / `qmatmul_with_epilogue`).
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
        && epilogues.pre_with_extras.is_none()
        && epilogues.post.is_none()
        && epilogues.post_with_extras.is_none()
        && qmatmul_try_coop::<BM, BN, BK>(program, a, b, y)
    {
        return;
    }

    if BM * BN * BK != LANES || !BK.is_power_of_two() {
        qmatmul_tile_with_epilogue::<8, 4, 8>(program, a, b, y, epilogues);
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
                    let loaded = program.load(a.at((&row, &k_index)), mask.clone(), 0.0);
                    let pre_extras = epilogues
                        .pre_extra_col_vectors
                        .iter()
                        .map(|extra| program.load(extra.at(&k_index), k_index.lt(k), 0.0))
                        .collect::<Vec<_>>();
                    let a_value = apply_qmatmul_pre_epilogue(epilogues, loaded, pre_extras);
                    let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
                    a_value * b_value
                },
            );
            let reduced = program.group_reduce_sum::<BK, _>(partial);
            let extras = epilogues
                .post_extra_col_vectors
                .iter()
                .map(|extra| program.load(extra.at(&col), col.lt(b.cols), 0.0))
                .collect::<Vec<_>>();
            let sum = apply_qmatmul_post_epilogue(epilogues, reduced, extras);
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
            None,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base,
        );
    });
}
