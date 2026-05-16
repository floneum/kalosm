//! Q4K paired-epilogue GEMV program kernels.

use fusor_tile_ir::tile::{Mask, Program, QuantizedDot, Storage, Tile, TileBlock};
use fusor_tile_ir::{
    GgmlQuantFormat, QuantizedMatrix, TileLiteral, TileReduceOp, WorkgroupAxis, F32,
};

use crate::grid::{q4k_ggml_iteration, q4k_lane_decomposition, Q4KGgmlIterationRequest};
use crate::types::{matrix_shape, PairedEpilogue};

const PAIRED_SUBGROUPS: u32 = 8;
const PAIRED_PAIRS_PER_SUBGROUP: usize = 2;
const PAIRED_DOTS_PER_SUBGROUP: usize = PAIRED_PAIRS_PER_SUBGROUP * 2;
const PAIRED_BLOCK: usize = 256;
const PAIRED_COLS_PER_WORKGROUP: u32 = PAIRED_SUBGROUPS * PAIRED_PAIRS_PER_SUBGROUP as u32;

/// Compute launch geometry for the paired Q4K GEMV kernel.
pub fn qgemv_q4k_paired_dispatch(
    pair_cols: u32,
    m_rows: u32,
    max_workgroups_per_dimension: u32,
) -> Option<([u32; 3], u32)> {
    let cols_workgroups = pair_cols.div_ceil(PAIRED_COLS_PER_WORKGROUP);
    let total_workgroups = cols_workgroups.checked_mul(m_rows.max(1))?;
    let workgroups_x = total_workgroups.min(max_workgroups_per_dimension).max(1);
    let dispatch_size = [workgroups_x, total_workgroups.div_ceil(workgroups_x), 1];
    dispatch_size
        .iter()
        .all(|dim| *dim <= max_workgroups_per_dimension)
        .then_some((dispatch_size, workgroups_x))
}

/// Inputs and launch geometry for the Q4K paired-epilogue GEMV kernels.
///
/// These kernels consume a Q4K matrix whose columns are laid out as
/// `[gate columns, up columns]`. Each kernel computes both halves for a
/// column pair, applies `epilogue` in-register, and writes the paired result.
///
/// ```no_run
/// # use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
/// # use fusor_tile_ir_kernels::{
/// #     PairedEpilogue, Q4KPairedGgml, qgemv_q4k_paired, quantized_matrix,
/// # };
/// let epilogue =
///     PairedEpilogue::with_extras("mul", 0, |tiles| tiles[0].clone() * tiles[1].clone());
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([1, 4096]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 4096, 8192);
///     let y = program.storage_write::<F32, 2>(Shape::new([1, 4096]));
///     qgemv_q4k_paired(
///         program,
///         Q4KPairedGgml {
///             a: &a,
///             b: &b,
///             y: &y,
///             pair_cols: 4096,
///             m_rows: 1,
///             workgroups_x: 1,
///             epilogue: &epilogue,
///             extras: &[],
///         },
///     );
/// });
/// ```
pub struct Q4KPairedGgml<'a> {
    /// Single-row or batched activation matrix.
    pub a: &'a Storage<F32, 2>,
    /// Q4K matrix with `pair_cols * 2` columns.
    pub b: &'a QuantizedMatrix,
    /// Output matrix with `pair_cols` columns.
    pub y: &'a Storage<F32, 2>,
    /// Number of gate/up pairs in `b`.
    pub pair_cols: u32,
    /// Number of rows from `a` and `y` covered by the launch.
    pub m_rows: u32,
    /// Preferred dispatch width on X. Clamped to the kernel's total workgroup count.
    pub workgroups_x: u32,
    /// Register-level operation applied to each `(gate, up)` pair.
    pub epilogue: &'a PairedEpilogue,
    /// One-dimensional extra tensors consumed by `epilogue`.
    pub extras: &'a [Storage<F32, 1>],
}

/// Build a Q4K paired-epilogue GEMV body.
pub fn qgemv_q4k_paired(program: &mut Program, spec: Q4KPairedGgml<'_>) {
    qgemv_q4k_paired_ggml(program, spec);
}

/// Q4K paired-epilogue qgemv body. The kernel reduces the gate and up halves
/// of a `[gate; up]` matmul output and applies the supplied `PairedEpilogue`
/// in-register before the single output store.
fn qgemv_q4k_paired_ggml(program: &mut Program, spec: Q4KPairedGgml<'_>) {
    let Q4KPairedGgml {
        a,
        b,
        y,
        pair_cols,
        m_rows,
        workgroups_x,
        epilogue,
        extras,
    } = spec;
    debug_assert_eq!(
        extras.len(),
        epilogue.extras_count(),
        "kernel extras count must match epilogue arity"
    );
    debug_assert_eq!(b.format, GgmlQuantFormat::Q4K);
    debug_assert_eq!(b.cols, pair_cols * 2);

    let [_, k] = matrix_shape(&a.view().layout);
    let cols_workgroups = pair_cols.div_ceil(PAIRED_COLS_PER_WORKGROUP);
    let m_rows = m_rows.max(1);
    let total_workgroups = cols_workgroups * m_rows;
    let workgroups_x = workgroups_x.min(total_workgroups.max(1));
    let dispatch_y = total_workgroups.div_ceil(workgroups_x);
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(4);
    let full_block_iterations = block_count.is_multiple_of(4);
    let full_cols = pair_cols.is_multiple_of(PAIRED_COLS_PER_WORKGROUP);
    let b_cloned = b.clone();

    program.program_grid::<PAIRED_BLOCK>([workgroups_x, dispatch_y, 1], |program| {
        let workgroup_idx = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let row = workgroup_idx.clone() / cols_workgroups;
        let col_workgroup = workgroup_idx % cols_workgroups;
        let row_in_bounds = row.clone().lt(m_rows);
        let col_group_base = col_workgroup * PAIRED_COLS_PER_WORKGROUP;
        let subgroup_col_base = program.subgroup_id() * PAIRED_PAIRS_PER_SUBGROUP as u32;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();
        let q4k_lane = q4k_lane_decomposition(&lane);

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile; PAIRED_DOTS_PER_SUBGROUP] = program
            .loop_fold_n::<PAIRED_DOTS_PER_SUBGROUP, _>(
                TileReduceOp::Sum,
                block_iterations,
                [zero; PAIRED_DOTS_PER_SUBGROUP],
                |program, loop_index| {
                    let pass = q4k_ggml_iteration(
                        program,
                        Q4KGgmlIterationRequest {
                            loop_index,
                            a,
                            row: row.clone(),
                            block_count,
                            full_block_iterations,
                            lane: &q4k_lane,
                            base_mask: row_in_bounds.clone(),
                        },
                    );

                    let dot = |program: &mut TileBlock<'_>, col: Tile, mask: Mask| {
                        program.quantized_dot(QuantizedDot::q4k_block(
                            pass.activations.low.clone(),
                            pass.activations.high.clone(),
                            pass.activations.sums.clone(),
                            &b_cloned,
                            &pass.block,
                            &q4k_lane.iq,
                            &q4k_lane.ir,
                            &col,
                            mask,
                            0.0,
                        ))
                    };

                    std::array::from_fn(|idx| {
                        let offset = idx % PAIRED_PAIRS_PER_SUBGROUP;
                        let gate = col0.clone() + offset as u32;
                        let col = if idx < PAIRED_PAIRS_PER_SUBGROUP {
                            gate.clone()
                        } else {
                            gate.clone() + pair_cols
                        };
                        let mask = if full_cols {
                            pass.in_bounds.clone()
                        } else {
                            pass.in_bounds.clone().and(gate.lt(pair_cols))
                        };
                        dot(program, col, mask)
                    })
                },
            );

        for offset in 0..PAIRED_PAIRS_PER_SUBGROUP {
            let col = col0.clone() + offset as u32;
            let gate = program.subgroup_reduce_sum(sums[offset].clone());
            let up = program.subgroup_reduce_sum(sums[offset + PAIRED_PAIRS_PER_SUBGROUP].clone());
            let store_lane = if full_cols {
                lane.eq(0)
            } else {
                lane.eq(0).and(col.lt(pair_cols))
            };
            let mask = store_lane.and(row_in_bounds.clone());
            // Load any per-column extras (e.g. bias vectors) at the current
            // output column. Indexing is `extras[k][col]` — extras are 1D
            // tensors of length `pair_cols`.
            let extra_tiles: Vec<Tile> = extras
                .iter()
                .map(|extra| {
                    program.load(
                        extra.at(col.clone()),
                        mask.clone(),
                        fusor_tile_ir::TileLiteral::f32(0.0),
                    )
                })
                .collect();
            let value = epilogue.apply(gate, up, &extra_tiles);
            program.store(y.at((row.clone(), col)), value, mask);
        }
    });
}
