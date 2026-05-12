//! Program-level qmatmul / matmul / gemv / qdequantize kernel constructors.
//! Free functions over `&mut fusor_tile_ir::Program`.

use fusor_tile_ir::tile::{CoopAcc, Program, Storage, Tile};
use fusor_tile_ir::KernelBuilder;
use fusor_tile_ir::{
    GgmlQuantFormat, Layout, MemoryLevel, Shape, StorageView, TileLiteral, TileReduceOp,
    WorkgroupAxis, F32, U32,
};

use fusor_tile_ir::QuantizedMatrix;

use crate::dispatch::{qmatmul_path, QmatmulPath};
use crate::grid::dot4_sum;
use crate::program_qgemv;
use crate::types::{cooperative_store_layout_supported, matrix_shape};

pub trait IntoQgemvEpilogues<'a> {
    fn into_qgemv_epilogues(self) -> crate::types::QmatmulEpilogues<'a>;
}

impl<'a> IntoQgemvEpilogues<'a> for Option<&'a crate::UnaryEpilogue> {
    fn into_qgemv_epilogues(self) -> crate::types::QmatmulEpilogues<'a> {
        crate::types::QmatmulEpilogues {
            pre: None,
            post: self,
        }
    }
}

impl<'a> IntoQgemvEpilogues<'a> for &'a crate::types::QmatmulEpilogues<'a> {
    fn into_qgemv_epilogues(self) -> crate::types::QmatmulEpilogues<'a> {
        self.clone()
    }
}

/// Convenience: declare a quantized matrix on a [`KernelBuilder`] and remember
/// its runtime binding. Equivalent to pushing `binding` and then calling
/// [`quantized_matrix`] on the underlying [`Program`].
pub fn quantized_matrix_for<B>(
    kb: &mut KernelBuilder<B>,
    binding: B,
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
) -> QuantizedMatrix {
    kb.push_binding(binding);
    quantized_matrix(kb.program(), format, rows, cols)
}

/// Allocate a quantized matrix backing buffer + return a kernel handle.
pub fn quantized_matrix(
    program: &mut Program,
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
) -> QuantizedMatrix {
    assert!(
        rows > 0 && cols > 0,
        "quantized matrix shape must be non-zero"
    );
    assert_eq!(
        rows % format.block_elements(),
        0,
        "quantized rows/K dimension must be a multiple of the format block size"
    );
    let blocks_per_col = rows / format.block_elements();
    let words = blocks_per_col
        .checked_mul(cols)
        .and_then(|blocks| blocks.checked_mul(format.block_words()))
        .expect("quantized matrix word count overflow");
    let data: Storage<U32, 1> = program.storage_read(Shape::new([words]));
    QuantizedMatrix {
        data: data.view().clone(),
        format,
        rows,
        cols,
    }
}

/// Top-level qmatmul entry. Picks the qgemv path when `m == 1`, otherwise the
/// scalar/coop tiled qmatmul body.
pub fn qmatmul<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
) {
    qmatmul_options::<BM, BN, BK>(program, a, b, y, vector_width, true, 1);
}

/// Top-level qgemv entry. Equivalent to [`qmatmul`] with `BM = 1`.
pub fn qgemv<const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    workgroups_x: u32,
) {
    qmatmul_options::<1, BN, BK>(program, a, b, y, vector_width, true, workgroups_x);
}

/// Variant of [`qgemv`] that threads optional pre/post unary epilogues through
/// the underlying qgemv variant chosen by `qgemv_tile`.
pub fn qgemv_with_epilogue<'a, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: impl IntoQgemvEpilogues<'a>,
) {
    let epilogues = epilogues.into_qgemv_epilogues();
    program_qgemv::qgemv_tile_with_epilogue::<BN, BK>(program, a, b, y, workgroups_x, &epilogues);
}

pub fn qmatmul_options<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    use_qgemv: bool,
    workgroups_x: u32,
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
        program_qgemv::qgemv_tile::<BN, BK>(program, a, b, y, workgroups_x);
    } else {
        qmatmul_tile::<BM, BN, BK>(program, a, b, y);
    }
}

/// Scalar lane-mapped qmatmul body. Public so downstream crates can reproduce
/// or replace the variant-selection layer above (`qmatmul_options` /
/// `qmatmul`).
pub fn qmatmul_tile<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
) {
    const LANES: usize = 256;
    assert!(
        BM > 0 && BN > 0 && BK > 0,
        "qmatmul tile shape must be non-zero"
    );
    let [m, k] = matrix_shape(&a.view().layout);

    let path = qmatmul_path(
        BM,
        BN,
        BK,
        (m as usize).is_multiple_of(BM),
        (b.cols as usize).is_multiple_of(BN),
        (k as usize).is_multiple_of(BK),
        cooperative_store_layout_supported(&y.view().layout),
    );
    if qmatmul_dispatch(program, path, a, b, y) {
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
            let lane = program.arange();
            let k_lane = lane.clone() % BK as u32;
            let output_lane = lane / BK as u32;
            let row_lane = output_lane.clone() / BN as u32;
            let col_lane = output_lane % BN as u32;
            let row = program.program_id(WorkgroupAxis::Y) * BM as u32 + row_lane;
            let col = program.program_id(WorkgroupAxis::X) * BN as u32 + col_lane;
            let loop_index = program.loop_index();
            let k_index = loop_index * BK as u32 + k_lane.clone();
            let mask = row.lt(m).and(col.lt(b.cols)).and(k_index.lt(k));
            let a_value = program.load(a.at(&row, &k_index), mask.clone(), 0.0);
            let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                a_value * b_value,
                TileLiteral::f32(0.0),
            );
            let sum = program.group_reduce_sum::<BK>(partial);
            let store_mask = k_lane.eq(0).and(row.lt(m)).and(col.lt(b.cols));
            program.store(y.at(row, col), sum, store_mask);
        },
    );
}

/// Variant-dispatched cooperative-matrix qmatmul. Returns `true` if a perf
/// monomorphization was emitted; `false` for `QmatmulPath::Scalar`, in which
/// case the caller is responsible for emitting a scalar body
/// (see `qmatmul_tile`).
pub fn qmatmul_dispatch(
    program: &mut Program,
    path: QmatmulPath,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
) -> bool {
    match path {
        QmatmulPath::Coop64x64 => {
            qmatmul_perf::<64, 64, 32, 2, 2, 128>(program, a, b, y);
            true
        }
        QmatmulPath::Coop64x128 => {
            qmatmul_perf::<64, 128, 32, 2, 4, 256>(program, a, b, y);
            true
        }
        QmatmulPath::Coop128x64 => {
            qmatmul_perf::<128, 64, 32, 4, 2, 256>(program, a, b, y);
            true
        }
        QmatmulPath::Coop128x128 => {
            qmatmul_perf::<128, 128, 32, 4, 4, 512>(program, a, b, y);
            true
        }
        QmatmulPath::Scalar => false,
    }
}

/// Cooperative-matrix qmatmul body. Each workgroup produces one BMxBN output
/// tile via an interleaved `ROW_GROUPS x COL_GROUPS` grid of subgroups, each
/// holding `(32*32)/(8*8)` = 16 cooperative-matrix accumulators.
/// `BLOCK == ROW_GROUPS * COL_GROUPS * 32`.
pub fn qmatmul_perf<
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

        let accs: Vec<Vec<CoopAcc>> = (0..TILE_ROWS_PER_SG)
            .map(|_| {
                (0..TILE_COLS_PER_SG)
                    .map(|_| {
                        let acc = program.alloc_coop_acc();
                        program.zero_coop_acc(&acc);
                        acc
                    })
                    .collect()
            })
            .collect();

        program.while_true(k_iterations, |program| {
            let k_base = program.loop_index() * BK as u32;
            program.copy_storage_to_tile(a_tile, a_clone, &row_base, &k_base);
            program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &col_base);
            program.workgroup_barrier();

            let kk_steps = (BK as u32) / COOP_DIM;
            for kk in 0..kk_steps {
                let a_frags: Vec<_> = (0..TILE_ROWS_PER_SG)
                    .map(|r| {
                        program.coop_load_a(
                            a_tile,
                            sg_row_base.clone() + r * COOP_DIM,
                            kk * COOP_DIM,
                        )
                    })
                    .collect();
                let b_frags: Vec<_> = (0..TILE_COLS_PER_SG)
                    .map(|c| {
                        program.coop_load_b(
                            b_tile,
                            kk * COOP_DIM,
                            sg_col_base.clone() + c * COOP_DIM,
                        )
                    })
                    .collect();
                for r in 0..TILE_ROWS_PER_SG {
                    for c in 0..TILE_COLS_PER_SG {
                        program.coop_mma(
                            &accs[r as usize][c as usize],
                            &a_frags[r as usize],
                            &b_frags[c as usize],
                        );
                    }
                }
            }
            program.workgroup_barrier();
        });

        for r in 0..TILE_ROWS_PER_SG {
            for c in 0..TILE_COLS_PER_SG {
                let row = row_base.clone() + sg_row_base.clone() + r * COOP_DIM;
                let col = col_base.clone() + sg_col_base.clone() + c * COOP_DIM;
                program.coop_store(&accs[r as usize][c as usize], y_clone, row, col);
            }
        }
    });
}

/// Dense F32 matmul, scalar lane-mapped body.
pub fn matmul<const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
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
            let lane_tile = program.lane_tile_2d::<TILE_ROWS, TILE_COLS>();
            let row = program.program_id(WorkgroupAxis::Y) * TILE_ROWS as u32 + lane_tile.row();
            let col = program.program_id(WorkgroupAxis::X) * TILE_COLS as u32 + lane_tile.col();
            let mask = row.lt(m).and(col.lt(n));
            let k_index = program.loop_index();
            let a_value = program.load(a.at(&row, &k_index), mask.clone(), 0.0);
            let b_value = program.load(b.at(&k_index, &col), mask.clone(), 0.0);
            let sum = program.loop_fold(
                TileReduceOp::Sum,
                k,
                a_value * b_value,
                TileLiteral::f32(0.0),
            );
            program.store(y.at(row, col), sum, mask);
        },
    );
}

/// Dense F32 GEMV (matrix × vector) — single-output-column matmul specialized
/// for the K-reduction along the input vector.
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
        let [sum] =
            program.loop_fold_n::<1, _>(TileReduceOp::Sum, k_iterations, [zero], |program| {
                let k_base =
                    program.loop_index() * k_per_iter + lane.clone() * VALUES_PER_LANE as u32;
                let a_values: [Tile<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let k_index = k_base.clone() + i as u32;
                    let mask = row_in_bounds.clone().and(k_index.lt(k));
                    program.load(a.at(&row, k_index), mask, 0.0)
                });
                let x_values: [Tile<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let k_index = k_base.clone() + i as u32;
                    program.load(x.at(k_index.clone(), 0), k_index.lt(k), 0.0)
                });
                [dot4_sum(program, &a_values, &x_values)]
            });
        let reduced = program.subgroup_reduce_sum(sum);
        let mask = lane.eq(0).and(row_in_bounds);
        program.store(y.at(row, 0), reduced, mask);
    });
}

/// Lane-per-element dequantization: emits one f32 per quantized element of
/// `b` and writes them to a row-major `y` of `b.rows * b.cols` floats.
pub fn qdequantize(
    program: &mut Program,
    b: &QuantizedMatrix,
    y: &Storage<F32, 1>,
    workgroups_x: u32,
) {
    const BLOCK: usize = 256;
    assert!(
        workgroups_x > 0,
        "qdequantize workgroups_x must be non-zero"
    );
    assert_eq!(
        y.view().layout.element_count().get(),
        b.rows
            .checked_mul(b.cols)
            .expect("qdequantize output element count overflow"),
        "qdequantize output must contain one dense f32 per quantized element"
    );
    assert!(
        y.view().layout.is_row_major(),
        "qdequantize output must be row-major"
    );

    let total = b
        .rows
        .checked_mul(b.cols)
        .expect("qdequantize output element count overflow");
    let workgroups = total.div_ceil(BLOCK as u32);
    let dispatch_y = workgroups.div_ceil(workgroups_x);
    let y = Storage::<F32, 2>::from_view(StorageView {
        buffer: y.view().buffer,
        offset: y.view().offset,
        layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([1, total])),
    });
    program.program_grid::<BLOCK>([workgroups_x, dispatch_y, 1], |program| {
        let lane = program.arange();
        let linear_group = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let flat = linear_group * BLOCK as u32 + lane;
        let mask = flat.lt(total);
        let value = program.load_quantized(
            b,
            flat.clone() % b.rows,
            flat.clone() / b.rows,
            mask.clone(),
            0.0,
        );
        program.store(y.at(0, flat), value, mask);
    });
}
