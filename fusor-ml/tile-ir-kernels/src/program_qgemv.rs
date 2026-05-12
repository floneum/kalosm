//! Quantized GEMV (single-row qmatmul) kernel constructors. Free functions
//! over `&mut fusor_tile_ir::Program`.

use fusor_tile_ir::tile::{Bound, Mask, Program, ScalarIndex, Storage, Tile, TileBlock};
use fusor_tile_ir::{
    GgmlQuantFormat, QuantizedMatrix, TileLiteral, TileReduceOp, WorkgroupAxis, F32,
};

use crate::dispatch::{
    q4k_default_large, q4k_default_mid, q4k_default_tall, q4k_large_override, q4k_mid_override,
    q4k_tall_override, q6k_default_large, q6k_default_tall, q6k_large_override, q6k_tall_override,
    QgemvShapeQ4K, QgemvShapeQ6K,
};
use crate::grid::{dot4_sum, q4k_ggml_activations, q4k_lane_decomposition, qgemv_grid, Q4KLane};
use crate::types::{matrix_shape, PairedEpilogue};

macro_rules! q4k_paired_entrypoints {
    ($(($name:ident, $subgroups:literal, $pairs:literal, $dots:literal, $block:literal)),+ $(,)?) => {
        $(
            pub fn $name(
                program: &mut Program,
                a: &Storage<F32, 2>,
                b: &QuantizedMatrix,
                y: &Storage<F32, 2>,
                pair_cols: u32,
                m_rows: u32,
                workgroups_x: u32,
                epilogue: &PairedEpilogue,
                extras: &[Storage<F32, 1>],
            ) {
                qgemv_q4k_paired_ggml::<$subgroups, $pairs, $dots, $block>(
                    program,
                    a,
                    b,
                    y,
                    pair_cols,
                    m_rows,
                    workgroups_x,
                    epilogue,
                    extras,
                );
            }
        )+
    };
}

q4k_paired_entrypoints!(
    (qgemv_q4k_paired_4x2, 4, 2, 4, 128),
    (qgemv_q4k_paired_4x1, 4, 1, 2, 128),
    (qgemv_q4k_paired_4x4, 4, 4, 8, 128),
    (qgemv_q4k_paired_8x1, 8, 1, 2, 256),
    (qgemv_q4k_paired_8x2, 8, 2, 4, 256),
    (qgemv_q4k_paired_2x2, 2, 2, 4, 64),
    (qgemv_q4k_paired_2x4, 2, 4, 8, 64),
);

/// Variant-dispatched Q4K ggml qgemv. Picks the right
/// `qgemv_q4k_ggml::<S, C, B>` monomorphization for the supplied shape.
pub fn qgemv_q4k_dispatch(
    program: &mut Program,
    shape: QgemvShapeQ4K,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
) {
    qgemv_q4k_dispatch_with_epilogue(
        program,
        shape,
        a,
        b,
        y,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

/// Variant of [`qgemv_q4k_dispatch`] that threads an optional unary epilogue
/// through the chosen `qgemv_q4k_ggml` monomorphization.
pub fn qgemv_q4k_dispatch_with_epilogue(
    program: &mut Program,
    shape: QgemvShapeQ4K,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    macro_rules! emit {
        ($s:literal, $c:literal, $b:literal) => {
            qgemv_q4k_ggml_with_epilogue::<$s, $c, $b>(program, a, b, y, workgroups_x, epilogues)
        };
    }
    match shape {
        QgemvShapeQ4K::Ggml1x4_32 => emit!(1, 4, 32),
        QgemvShapeQ4K::Ggml1x8_32 => emit!(1, 8, 32),
        QgemvShapeQ4K::Ggml2x2_64 => emit!(2, 2, 64),
        QgemvShapeQ4K::Ggml2x3_64 => emit!(2, 3, 64),
        QgemvShapeQ4K::Ggml2x4_64 => emit!(2, 4, 64),
        QgemvShapeQ4K::Ggml2x8_64 => emit!(2, 8, 64),
        QgemvShapeQ4K::Ggml4x1_128 => emit!(4, 1, 128),
        QgemvShapeQ4K::Ggml4x2_128 => emit!(4, 2, 128),
        QgemvShapeQ4K::Ggml4x3_128 => emit!(4, 3, 128),
        QgemvShapeQ4K::Ggml4x4_128 => emit!(4, 4, 128),
        QgemvShapeQ4K::Ggml4x8_128 => emit!(4, 8, 128),
        QgemvShapeQ4K::Ggml8x1_256 => emit!(8, 1, 256),
        QgemvShapeQ4K::Ggml8x2_256 => emit!(8, 2, 256),
        QgemvShapeQ4K::Ggml8x4_256 => emit!(8, 4, 256),
    }
}

/// Variant-dispatched Q6K ggml qgemv. Same role as `qgemv_q4k_dispatch` for
/// the Q6K format.
pub fn qgemv_q6k_dispatch(
    program: &mut Program,
    shape: QgemvShapeQ6K,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
) {
    qgemv_q6k_dispatch_with_epilogue(
        program,
        shape,
        a,
        b,
        y,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

/// Variant of [`qgemv_q6k_dispatch`] that threads an optional unary epilogue
/// through the chosen `qgemv_q6k_ggml` monomorphization.
pub fn qgemv_q6k_dispatch_with_epilogue(
    program: &mut Program,
    shape: QgemvShapeQ6K,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    macro_rules! emit {
        ($s:literal, $c:literal, $b:literal) => {
            qgemv_q6k_ggml_with_epilogue::<$s, $c, $b>(program, a, b, y, workgroups_x, epilogues)
        };
    }
    match shape {
        QgemvShapeQ6K::Ggml2x2_64 => emit!(2, 2, 64),
        QgemvShapeQ6K::Ggml2x4_64 => emit!(2, 4, 64),
        QgemvShapeQ6K::Ggml2x8_64 => emit!(2, 8, 64),
        QgemvShapeQ6K::Ggml4x2_128 => emit!(4, 2, 128),
        QgemvShapeQ6K::Ggml4x4_128 => emit!(4, 4, 128),
        QgemvShapeQ6K::Ggml4x8_128 => emit!(4, 8, 128),
        QgemvShapeQ6K::Ggml8x2_256 => emit!(8, 2, 256),
        QgemvShapeQ6K::Ggml8x4_256 => emit!(8, 4, 256),
    }
}

/// Format-dispatched qgemv body. Picks a `qgemv_perf` / `qgemv_q*_dispatch`
/// monomorphization for the format/shape of `b`. Requires `m == 1`.
pub fn qgemv_tile<const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
) {
    qgemv_tile_with_epilogue::<BN, BK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

/// Variant of [`qgemv_tile`] that threads optional pre/post unary epilogues
/// through whichever underlying qgemv variant is chosen for the format/shape.
pub fn qgemv_tile_with_epilogue<const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    ep: &crate::types::QmatmulEpilogues<'_>,
) {
    let [m, _] = matrix_shape(&a.view().layout);
    assert_eq!(m, 1, "qgemv requires a single input row");

    match b.format {
        GgmlQuantFormat::Q8_0 => {
            if b.cols >= 8192 {
                return qgemv_perf_with_epilogue::<4, 8, 8, 128>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                );
            }
            qgemv_perf_with_epilogue::<4, 4, 8, 128>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q8_1 => {
            qgemv_perf_with_epilogue::<4, 4, 8, 128>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q4K => {
            if b.rows <= 4096 && b.cols >= 4096 && b.cols < 8192 {
                let shape = q4k_mid_override(q4k_default_mid(b.rows, b.cols));
                return qgemv_q4k_dispatch_with_epilogue(program, shape, a, b, y, workgroups_x, ep);
            }
            if b.rows <= 4096 && b.cols <= 4096 {
                return qgemv_perf_with_epilogue::<8, 4, 16, 256>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                );
            }
            if b.rows <= 4096 && b.cols >= 8192 {
                let shape = q4k_large_override(q4k_default_large(b.rows, b.cols));
                return qgemv_q4k_dispatch_with_epilogue(program, shape, a, b, y, workgroups_x, ep);
            }
            if b.rows > 4096 && b.cols <= 4096 {
                let shape = q4k_tall_override(q4k_default_tall(b.rows, b.cols));
                return qgemv_q4k_dispatch_with_epilogue(program, shape, a, b, y, workgroups_x, ep);
            }
            if b.format
                .qgemv_subgroups_per_workgroup_for_shape(b.rows, b.cols)
                == 8
            {
                return qgemv_perf_with_epilogue::<8, 8, 8, 256>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                );
            }
            qgemv_perf_with_epilogue::<4, 8, 8, 128>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q5_0 => {
            qgemv_perf_with_epilogue::<2, 4, 16, 64>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q4_0
        | GgmlQuantFormat::Q4_1
        | GgmlQuantFormat::Q5_1
        | GgmlQuantFormat::Q2K => {
            qgemv_perf_with_epilogue::<2, 4, 8, 64>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q3K | GgmlQuantFormat::Q8K => {
            qgemv_perf_with_epilogue::<2, 2, 8, 64>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q5K => {
            qgemv_perf_with_epilogue::<2, 1, 8, 64>(program, a, b, y, workgroups_x, ep)
        }
        GgmlQuantFormat::Q6K => {
            if b.rows <= 4096 && b.cols >= 8192 {
                let shape = q6k_large_override(q6k_default_large(b.rows, b.cols));
                return qgemv_q6k_dispatch_with_epilogue(program, shape, a, b, y, workgroups_x, ep);
            }
            if b.rows > 4096 && b.cols <= 4096 {
                let shape = q6k_tall_override(q6k_default_tall(b.rows, b.cols));
                return qgemv_q6k_dispatch_with_epilogue(program, shape, a, b, y, workgroups_x, ep);
            }
            if b.format
                .qgemv_subgroups_per_workgroup_for_shape(b.rows, b.cols)
                == 4
            {
                return qgemv_perf_with_epilogue::<4, 4, 8, 128>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                );
            }
            qgemv_perf_with_epilogue::<8, 4, 16, 256>(program, a, b, y, workgroups_x, ep)
        }
    }
}

/// Q4K ggml-format qgemv body. Public so downstream crates can call a
/// specific monomorphization directly (see `qgemv_q4k_dispatch` for the
/// shape-driven entry point).
pub fn qgemv_q4k_ggml<const SUBGROUPS: u32, const COLS_PER_SUBGROUP: usize, const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
) {
    qgemv_q4k_ggml_with_epilogue::<SUBGROUPS, COLS_PER_SUBGROUP, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

/// Variant of [`qgemv_q4k_ggml`] that applies optional pre/post unary
/// epilogues.
pub fn qgemv_q4k_ggml_with_epilogue<
    const SUBGROUPS: u32,
    const COLS_PER_SUBGROUP: usize,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert_eq!(SUBGROUPS * SUBGROUP_SIZE, BLOCK as u32);
    debug_assert_eq!(b.format, GgmlQuantFormat::Q4K);

    let [_, k] = matrix_shape(&a.view().layout);
    let grid = qgemv_grid::<SUBGROUPS, COLS_PER_SUBGROUP>(b.cols, workgroups_x);
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(4);
    let full_block_iterations = block_count.is_multiple_of(4);
    let b_cloned = b.clone();

    program.program_grid::<BLOCK>([grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let workgroup = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid.workgroups_x;
        let col_group_base = workgroup * grid.cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * COLS_PER_SUBGROUP as u32;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();
        let Q4KLane { ix, iq, ir } = q4k_lane_decomposition(&lane);

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile<BLOCK>; COLS_PER_SUBGROUP] = program.loop_fold_n::<COLS_PER_SUBGROUP, _>(
            TileReduceOp::Sum,
            block_iterations,
            [zero; COLS_PER_SUBGROUP],
            |program| {
                let block = program.loop_index() * 4 + ix.clone();
                let in_bounds = if full_block_iterations {
                    Mask::all()
                } else {
                    block.clone().lt(block_count)
                };
                let vector_base = block.clone() * 256 + iq.clone() * 64 + ir.clone() * 8;

                let activations =
                    q4k_ggml_activations(program, a, 0, &vector_base, in_bounds.clone());

                std::array::from_fn(|c| {
                    let col = col0.clone() + c as u32;
                    let mask = grid.mask(full_block_iterations, in_bounds.clone(), &col);
                    let dot_inputs = activations.clone();
                    program.quantized_q4k_ggml_dot(
                        dot_inputs.low,
                        dot_inputs.high,
                        dot_inputs.sums,
                        &b_cloned,
                        &block,
                        &iq,
                        &ir,
                        &col,
                        mask,
                        0.0,
                    )
                })
            },
        );

        crate::grid::store_qgemv_sums_with_epilogue(
            program,
            y,
            col0,
            lane,
            sums,
            grid.full_cols,
            grid.n_cols,
            epilogues,
        );
    });
}

/// Q4K paired-epilogue qgemv body. The kernel reduces the gate and up halves
/// of a `[gate; up]` matmul output and applies the supplied `PairedEpilogue`
/// in-register before the single output store. Per-shape entry points are
/// generated by the `q4k_paired_entrypoints!` macro.
pub fn qgemv_q4k_paired_ggml<
    const SUBGROUPS: u32,
    const PAIRS_PER_SUBGROUP: usize,
    const DOTS_PER_SUBGROUP: usize,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    pair_cols: u32,
    m_rows: u32,
    workgroups_x: u32,
    epilogue: &PairedEpilogue,
    extras: &[Storage<F32, 1>],
) {
    debug_assert_eq!(
        extras.len(),
        epilogue.extras_count(),
        "kernel extras count must match epilogue arity"
    );
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert_eq!(SUBGROUPS * SUBGROUP_SIZE, BLOCK as u32);
    debug_assert_eq!(DOTS_PER_SUBGROUP, PAIRS_PER_SUBGROUP * 2);
    debug_assert_eq!(b.format, GgmlQuantFormat::Q4K);
    debug_assert_eq!(b.cols, pair_cols * 2);

    let [_, k] = matrix_shape(&a.view().layout);
    let cols_per_workgroup = SUBGROUPS * PAIRS_PER_SUBGROUP as u32;
    let cols_workgroups = pair_cols.div_ceil(cols_per_workgroup);
    let m_rows = m_rows.max(1);
    let total_workgroups = cols_workgroups * m_rows;
    let workgroups_x = workgroups_x.min(total_workgroups.max(1));
    let dispatch_y = total_workgroups.div_ceil(workgroups_x);
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(4);
    let full_block_iterations = block_count.is_multiple_of(4);
    let full_cols = pair_cols.is_multiple_of(cols_per_workgroup);
    let b_cloned = b.clone();

    program.program_grid::<BLOCK>([workgroups_x, dispatch_y, 1], |program| {
        let workgroup_idx = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let row = workgroup_idx.clone() / cols_workgroups;
        let col_workgroup = workgroup_idx % cols_workgroups;
        let row_in_bounds = row.clone().lt(m_rows);
        let col_group_base = col_workgroup * cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * PAIRS_PER_SUBGROUP as u32;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();
        let Q4KLane { ix, iq, ir } = q4k_lane_decomposition(&lane);

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile<BLOCK>; DOTS_PER_SUBGROUP] = program.loop_fold_n::<DOTS_PER_SUBGROUP, _>(
            TileReduceOp::Sum,
            block_iterations,
            [zero; DOTS_PER_SUBGROUP],
            |program| {
                let block = program.loop_index() * 4 + ix.clone();
                let in_bounds = if full_block_iterations {
                    row_in_bounds.clone()
                } else {
                    row_in_bounds.clone().and(block.clone().lt(block_count))
                };
                let vector_base = block.clone() * 256 + iq.clone() * 64 + ir.clone() * 8;

                let activations =
                    q4k_ggml_activations(program, a, row.clone(), &vector_base, in_bounds.clone());
                let a_low_vec = activations.low;
                let a_high_vec = activations.high;
                let sum_vec = activations.sums;

                let dot =
                    |program: &mut TileBlock<'_, BLOCK>, col: ScalarIndex, mask: Mask<BLOCK>| {
                        program.quantized_q4k_ggml_dot(
                            a_low_vec.clone(),
                            a_high_vec.clone(),
                            sum_vec.clone(),
                            &b_cloned,
                            &block,
                            &iq,
                            &ir,
                            &col,
                            mask,
                            0.0,
                        )
                    };

                std::array::from_fn(|idx| {
                    let offset = idx % PAIRS_PER_SUBGROUP;
                    let gate = col0.clone() + offset as u32;
                    let col = if idx < PAIRS_PER_SUBGROUP {
                        gate.clone()
                    } else {
                        gate.clone() + pair_cols
                    };
                    let mask = if full_cols {
                        in_bounds.clone()
                    } else {
                        in_bounds.clone().and(gate.lt(pair_cols))
                    };
                    dot(program, col, mask)
                })
            },
        );

        for offset in 0..PAIRS_PER_SUBGROUP {
            let col = col0.clone() + offset as u32;
            let gate = program.subgroup_reduce_sum(sums[offset].clone());
            let up = program.subgroup_reduce_sum(sums[offset + PAIRS_PER_SUBGROUP].clone());
            let store_lane = if full_cols {
                lane.eq(0)
            } else {
                lane.eq(0).and(col.lt(pair_cols))
            };
            let mask = store_lane.and(row_in_bounds.clone());
            // Load any per-column extras (e.g. bias vectors) at the current
            // output column. Indexing is `extras[k][col]` — extras are 1D
            // tensors of length `pair_cols`.
            let extra_tiles: Vec<Tile<BLOCK>> = extras
                .iter()
                .map(|extra| {
                    program.load_linear(
                        extra.at(col.clone()),
                        mask.clone(),
                        fusor_tile_ir::TileLiteral::f32(0.0),
                    )
                })
                .collect();
            let value = epilogue.apply(gate, up, &extra_tiles);
            program.store(y.at(row.clone(), col), value, mask);
        }
    });
}

/// Q6K ggml-format qgemv body. Public so downstream crates can call a
/// specific monomorphization directly (see `qgemv_q6k_dispatch` for the
/// shape-driven entry point).
pub fn qgemv_q6k_ggml<const SUBGROUPS: u32, const COLS_PER_SUBGROUP: usize, const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
) {
    qgemv_q6k_ggml_with_epilogue::<SUBGROUPS, COLS_PER_SUBGROUP, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

/// Variant of [`qgemv_q6k_ggml`] that applies optional pre/post unary
/// epilogues.
pub fn qgemv_q6k_ggml_with_epilogue<
    const SUBGROUPS: u32,
    const COLS_PER_SUBGROUP: usize,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert_eq!(SUBGROUPS * SUBGROUP_SIZE, BLOCK as u32);
    debug_assert_eq!(b.format, GgmlQuantFormat::Q6K);

    let [_, k] = matrix_shape(&a.view().layout);
    let grid = qgemv_grid::<SUBGROUPS, COLS_PER_SUBGROUP>(b.cols, workgroups_x);
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(2);
    let full_block_iterations = block_count.is_multiple_of(2);
    let b_cloned = b.clone();

    program.program_grid::<BLOCK>([grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let workgroup = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid.workgroups_x;
        let col_group_base = workgroup * grid.cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * COLS_PER_SUBGROUP as u32;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();
        let tid = lane.clone() / 2;
        let ix = lane.clone() % 2;
        let ip = tid.clone() / 8;
        let il = tid % 8;
        let l0 = il.clone() * 4;

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile<BLOCK>; COLS_PER_SUBGROUP] = program.loop_fold_n::<COLS_PER_SUBGROUP, _>(
            TileReduceOp::Sum,
            block_iterations,
            [zero; COLS_PER_SUBGROUP],
            |program| {
                let block = program.loop_index() * 2 + ix.clone();
                let in_bounds = if full_block_iterations {
                    Mask::all()
                } else {
                    block.clone().lt(block_count)
                };
                let vector_base = block.clone() * 256 + ip.clone() * 128 + l0.clone();

                let a_bound: [Bound<BLOCK>; 16] = std::array::from_fn(|j| {
                    let offset = (j / 4) as u32 + (j % 4) as u32 * 32;
                    let scalar = program.load(
                        a.at(0, vector_base.clone() + offset),
                        in_bounds.clone(),
                        0.0,
                    );
                    program.bind(scalar)
                });

                std::array::from_fn(|c| {
                    let col = col0.clone() + c as u32;
                    let mask = grid.mask(full_block_iterations, in_bounds.clone(), &col);
                    let a_vec: [Tile<BLOCK>; 16] = std::array::from_fn(|i| a_bound[i].get());
                    program
                        .quantized_q6k_ggml_dot(a_vec, &b_cloned, &block, &ip, &il, &col, mask, 0.0)
                })
            },
        );

        crate::grid::store_qgemv_sums_with_epilogue(
            program,
            y,
            col0,
            lane,
            sums,
            grid.full_cols,
            grid.n_cols,
            epilogues,
        );
    });
}

/// Generic subgroup-partitioned qgemv body covering the formats that don't
/// have a dedicated `qgemv_q*_ggml` path.
pub fn qgemv_perf<
    const SUBGROUPS: u32,
    const COLS_PER_SUBGROUP: usize,
    const VALUES_PER_LANE: usize,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
) {
    qgemv_perf_with_epilogue::<SUBGROUPS, COLS_PER_SUBGROUP, VALUES_PER_LANE, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        &crate::types::QmatmulEpilogues::empty(),
    );
}

/// Variant of [`qgemv_perf`] that applies optional pre- and post-reduce
/// epilogues. `pre` is applied to each loaded activation tile before the dot;
/// `post` is applied to each per-output tile before the store.
pub fn qgemv_perf_with_epilogue<
    const SUBGROUPS: u32,
    const COLS_PER_SUBGROUP: usize,
    const VALUES_PER_LANE: usize,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert_eq!(SUBGROUPS * SUBGROUP_SIZE, BLOCK as u32);
    debug_assert!(VALUES_PER_LANE == 8 || VALUES_PER_LANE == 16 || VALUES_PER_LANE == 32);
    debug_assert!(
        COLS_PER_SUBGROUP == 1
            || COLS_PER_SUBGROUP == 2
            || COLS_PER_SUBGROUP == 4
            || COLS_PER_SUBGROUP == 8
    );
    let [_, k] = matrix_shape(&a.view().layout);
    let grid = qgemv_grid::<SUBGROUPS, COLS_PER_SUBGROUP>(b.cols, workgroups_x);
    let k_per_iter = SUBGROUP_SIZE * VALUES_PER_LANE as u32;
    let k_iterations = k.div_ceil(k_per_iter);
    let k_size = k;
    let full_k_iterations = k.is_multiple_of(k_per_iter);
    let b_cloned = b.clone();
    let q6k_vocab_f32_dot = b.format == GgmlQuantFormat::Q6K && b.rows <= 4096 && b.cols >= 65_536;
    program.program_grid::<BLOCK>([grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let workgroup = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid.workgroups_x;
        let col_group_base = workgroup * grid.cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * COLS_PER_SUBGROUP as u32;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile<BLOCK>; COLS_PER_SUBGROUP] = program.loop_fold_n::<COLS_PER_SUBGROUP, _>(
            TileReduceOp::Sum,
            k_iterations,
            [zero; COLS_PER_SUBGROUP],
            |program| {
                let k_base =
                    program.loop_index() * k_per_iter + lane.clone() * VALUES_PER_LANE as u32;
                let in_bounds_k = if full_k_iterations {
                    Mask::all()
                } else {
                    k_base.lt(k_size)
                };

                let a_bound: [Bound<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let scalar =
                        program.load(a.at(0, k_base.clone() + i as u32), in_bounds_k.clone(), 0.0);
                    let scalar = crate::types::apply_optional_epilogue(epilogues.pre, scalar);
                    program.bind(scalar)
                });

                let a8 = || -> [Tile<BLOCK>; 8] { std::array::from_fn(|i| a_bound[i].get()) };
                let an = || -> [Tile<BLOCK>; VALUES_PER_LANE] {
                    std::array::from_fn(|i| a_bound[i].get())
                };
                std::array::from_fn(|c| {
                    let col = col0.clone() + c as u32;
                    let mask = grid.mask(full_k_iterations, in_bounds_k.clone(), &col);
                    if b_cloned.format == GgmlQuantFormat::Q8_0
                        && VALUES_PER_LANE == 8
                        && grid.n_cols >= 8192
                    {
                        return program.quantized_q8_0_dot8(
                            a8(),
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        );
                    }
                    if b_cloned.format == GgmlQuantFormat::Q4K
                        && (VALUES_PER_LANE == 8 || VALUES_PER_LANE == 16 || VALUES_PER_LANE == 32)
                    {
                        return program.quantized_q4k_f32_dot::<VALUES_PER_LANE>(
                            an(),
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        );
                    }
                    if b_cloned.format == GgmlQuantFormat::Q6K && VALUES_PER_LANE == 8 {
                        return program.quantized_q8_0_dot8(
                            a8(),
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        );
                    }
                    if b_cloned.format == GgmlQuantFormat::Q6K && !q6k_vocab_f32_dot {
                        return program.quantized_q8_activation_dot::<VALUES_PER_LANE>(
                            an(),
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        );
                    }
                    let bs: [Tile<BLOCK>; VALUES_PER_LANE] = program
                        .load_quantized_block::<VALUES_PER_LANE>(
                            &b_cloned, &k_base, &col, mask, 0.0,
                        );
                    dot4_sum(program, &an(), &bs)
                })
            },
        );

        crate::grid::store_qgemv_sums_with_epilogue(
            program,
            y,
            col0,
            lane,
            sums,
            grid.full_cols,
            grid.n_cols,
            epilogues,
        );
    });
}
