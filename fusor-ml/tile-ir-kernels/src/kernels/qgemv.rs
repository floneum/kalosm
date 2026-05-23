//! Quantized GEMV program kernels.

use fusor_tile_ir::tile::{BlockCoord, Mask, Program, Q4KActivations, QuantizedDot, Storage, Tile, TileBlock};
use fusor_tile_ir::{GgmlQuantFormat, QuantizedMatrix, TileLiteral, TileReduceOp, F32, U32};

use crate::dispatch::{
    q4k_default_large, q4k_default_mid, q4k_default_tall, q4k_large_override, q4k_mid_override,
    q4k_tall_override, q6k_default_large, q6k_default_tall, q6k_large_override, q6k_tall_override,
    qgemv_subgroups_per_workgroup_for_shape, QgemvShapeQ4K, QgemvShapeQ6K,
};
use crate::grid::{
    dot4_sum, q4k_ggml_iteration, q4k_lane_decomposition, q6k_ggml_iteration,
    q6k_lane_decomposition, qgemv_grid, qgemv_program_scope, Q4KGgmlIterationRequest, Q4KLane,
    Q6KGgmlIterationRequest, Q6KLane, QgemvStoreTarget,
};
use crate::types::matrix_shape;

/// Converts qgemv epilogue inputs into the internal pre/post epilogue bundle.
///
/// Public callers normally pass either `Option<&UnaryEpilogue>` for a post-only
/// epilogue or `&QmatmulEpilogues` for explicit pre/post control.
pub trait IntoQgemvEpilogues<'a> {
    /// Convert into a qgemv epilogue bundle.
    fn into_qgemv_epilogues(self) -> crate::types::QmatmulEpilogues<'a>;
}

impl<'a> IntoQgemvEpilogues<'a> for Option<&'a crate::UnaryEpilogue> {
    fn into_qgemv_epilogues(self) -> crate::types::QmatmulEpilogues<'a> {
        crate::types::QmatmulEpilogues {
            pre: None,
            pre_with_extras: None,
            pre_extra_inputs: &[],
            post: self,
            post_with_extras: None,
            post_extra_inputs: &[],
            post_acc_init_col_vector: None,
        }
    }
}

impl<'a> IntoQgemvEpilogues<'a> for &'a crate::types::QmatmulEpilogues<'a> {
    fn into_qgemv_epilogues(self) -> crate::types::QmatmulEpilogues<'a> {
        self.clone()
    }
}

/// Top-level quantized GEMV with optional pre/post unary epilogues.
///
/// Equivalent to [`crate::qmatmul_with_epilogue`] with `BM = 1`. Callers
/// with no epilogue pass `None` (or `Option::<&UnaryEpilogue>::None`).
///
/// ```
/// use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
/// use fusor_tile_ir_kernels::{qgemv_with_epilogue, quantized_matrix, UnaryEpilogue};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([1, 256]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 128);
///     let y = program.storage_write::<F32, 2>(Shape::new([1, 128]));
///     qgemv_with_epilogue::<4, 64>(program, &a, &b, &y, 1, Option::<&UnaryEpilogue>::None);
/// });
/// # let _ = ir;
/// ```
pub fn qgemv_with_epilogue<'a, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: impl IntoQgemvEpilogues<'a>,
) {
    let epilogues = epilogues.into_qgemv_epilogues();
    qgemv_tile_with_epilogue::<BN, BK>(program, a, b, y, workgroups_x, &epilogues);
}

/// Variant-dispatched Q4K ggml qgemv. Picks the right monomorphization for
/// the supplied shape and threads optional unary epilogues through it.
pub(crate) fn qgemv_q4k_dispatch_with_epilogue(
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

/// Variant-dispatched Q6K ggml qgemv. Same role as the Q4K dispatch helper,
/// but for Q6K shapes.
pub(crate) fn qgemv_q6k_dispatch_with_epilogue(
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

/// Format-dispatched qgemv body with optional pre/post unary epilogues.
pub(crate) fn qgemv_tile_with_epilogue<const BN: usize, const BK: usize>(
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
            if qgemv_subgroups_per_workgroup_for_shape(b.format, b.rows, b.cols) == 8 {
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
            if qgemv_subgroups_per_workgroup_for_shape(b.format, b.rows, b.cols) == 4 {
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

mod ggml_family_sealed {
    pub trait Sealed {}
}

/// One ggml-format iteration's outputs as consumed by
/// `qgemv_ggml_with_epilogue`. Mirrors `Q{4,6}KGgmlIteration` but parameterized
/// over the format's activation payload.
struct GgmlIteration<A> {
    block: Tile<U32>,
    in_bounds: Mask,
    activations: A,
}

/// Sealed local trait abstracting the per-format pieces shared by the two
/// `qgemv_q{4,6}k_ggml_with_epilogue` functions: the block-count divisor used
/// to derive `block_iterations`, the lane decomposition, the per-pass
/// iteration helper, and the format-specific `QuantizedDot` constructor.
///
/// All trait items are `pub(crate)`-visible only via the impls below; the
/// `Sealed` supertrait keeps the implementer set closed to Q4K/Q6K.
trait GgmlQuantFamily: ggml_family_sealed::Sealed {
    const FORMAT: GgmlQuantFormat;
    /// `block_iterations = block_count.div_ceil(BLOCK_DIV)` — 4 for Q4K, 2 for Q6K.
    const BLOCK_DIV: u32;
    type Lane;
    type Activations: Clone;

    fn lane_decomposition(lane: &Tile<U32>) -> Self::Lane;

    fn iteration(
        program: &mut TileBlock<'_>,
        loop_index: Tile<U32>,
        a: &Storage<F32, 2>,
        block_count: u32,
        full_block_iterations: bool,
        lane: &Self::Lane,
    ) -> GgmlIteration<Self::Activations>;

    fn quantized_dot(
        activations: Self::Activations,
        b: &QuantizedMatrix,
        block: &Tile<U32>,
        lane: &Self::Lane,
        col: Tile<U32>,
        mask: Mask,
    ) -> QuantizedDot;
}

/// Q4K ggml family marker.
enum Q4KFamily {}
impl ggml_family_sealed::Sealed for Q4KFamily {}
impl GgmlQuantFamily for Q4KFamily {
    const FORMAT: GgmlQuantFormat = GgmlQuantFormat::Q4K;
    const BLOCK_DIV: u32 = 4;
    type Lane = Q4KLane;
    type Activations = Q4KActivations;

    fn lane_decomposition(lane: &Tile<U32>) -> Self::Lane {
        q4k_lane_decomposition(lane)
    }

    fn iteration(
        program: &mut TileBlock<'_>,
        loop_index: Tile<U32>,
        a: &Storage<F32, 2>,
        block_count: u32,
        full_block_iterations: bool,
        lane: &Self::Lane,
    ) -> GgmlIteration<Self::Activations> {
        let pass = q4k_ggml_iteration(
            program,
            Q4KGgmlIterationRequest {
                loop_index,
                a,
                row: 0,
                block_count,
                full_block_iterations,
                lane,
                base_mask: Mask::all(),
            },
        );
        GgmlIteration {
            block: pass.block,
            in_bounds: pass.in_bounds,
            activations: pass.activations,
        }
    }

    fn quantized_dot(
        activations: Self::Activations,
        b: &QuantizedMatrix,
        block: &Tile<U32>,
        lane: &Self::Lane,
        col: Tile<U32>,
        mask: Mask,
    ) -> QuantizedDot {
        QuantizedDot::q4k_block(
            activations,
            b,
            BlockCoord::new(block, &lane.iq, &lane.ir),
            col,
            mask,
            0.0,
        )
    }
}

/// Q6K ggml family marker.
enum Q6KFamily {}
impl ggml_family_sealed::Sealed for Q6KFamily {}
impl GgmlQuantFamily for Q6KFamily {
    const FORMAT: GgmlQuantFormat = GgmlQuantFormat::Q6K;
    const BLOCK_DIV: u32 = 2;
    type Lane = Q6KLane;
    type Activations = [Tile<F32>; 16];

    fn lane_decomposition(lane: &Tile<U32>) -> Self::Lane {
        q6k_lane_decomposition(lane)
    }

    fn iteration(
        program: &mut TileBlock<'_>,
        loop_index: Tile<U32>,
        a: &Storage<F32, 2>,
        block_count: u32,
        full_block_iterations: bool,
        lane: &Self::Lane,
    ) -> GgmlIteration<Self::Activations> {
        let pass = q6k_ggml_iteration(
            program,
            Q6KGgmlIterationRequest {
                loop_index,
                a,
                row: 0,
                block_count,
                full_block_iterations,
                lane,
                base_mask: Mask::all(),
            },
        );
        GgmlIteration {
            block: pass.block,
            in_bounds: pass.in_bounds,
            activations: pass.activations,
        }
    }

    fn quantized_dot(
        activations: Self::Activations,
        b: &QuantizedMatrix,
        block: &Tile<U32>,
        lane: &Self::Lane,
        col: Tile<U32>,
        mask: Mask,
    ) -> QuantizedDot {
        QuantizedDot::q6k_block(
            activations,
            b,
            BlockCoord::new(block, &lane.ip, &lane.il),
            col,
            mask,
            0.0,
        )
    }
}

/// Format-generic ggml-format qgemv body with optional pre/post unary
/// epilogues. Monomorphizes to the same IR the per-format
/// `qgemv_q{4,6}k_ggml_with_epilogue` shims previously produced by hand.
fn qgemv_ggml_with_epilogue<
    F: GgmlQuantFamily,
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
    debug_assert_eq!(b.format, F::FORMAT);

    let [_, k] = matrix_shape(&a.view().layout);
    let grid = qgemv_grid::<SUBGROUPS, COLS_PER_SUBGROUP>(b.cols, workgroups_x);
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(F::BLOCK_DIV);
    let full_block_iterations = block_count.is_multiple_of(F::BLOCK_DIV);
    let b_cloned = b.clone();

    program.program_grid::<BLOCK>([grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let scope = qgemv_program_scope::<COLS_PER_SUBGROUP>(program, grid);
        let col0 = scope.col0;
        let lane = scope.lane;
        let fmt_lane = F::lane_decomposition(&lane);

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile; COLS_PER_SUBGROUP] = program.loop_fold_n::<COLS_PER_SUBGROUP, _, _>(
            TileReduceOp::Sum,
            block_iterations,
            [zero; COLS_PER_SUBGROUP],
            |program, loop_index| {
                let pass = F::iteration(
                    program,
                    loop_index,
                    a,
                    block_count,
                    full_block_iterations,
                    &fmt_lane,
                );

                std::array::from_fn(|c| {
                    let col = col0.clone() + c as u32;
                    let mask = grid.mask(full_block_iterations, pass.in_bounds.clone(), &col);
                    program.quantized_dot(F::quantized_dot(
                        pass.activations.clone(),
                        &b_cloned,
                        &pass.block,
                        &fmt_lane,
                        col,
                        mask,
                    ))
                })
            },
        );

        crate::grid::store_qgemv_sums_with_epilogue(
            program,
            sums,
            QgemvStoreTarget {
                y,
                col0,
                lane,
                full_cols: grid.full_cols,
                n_cols: grid.n_cols,
                epilogues,
            },
        );
    });
}

/// Q4K ggml-format qgemv body with optional pre/post unary epilogues.
pub(crate) fn qgemv_q4k_ggml_with_epilogue<
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
    qgemv_ggml_with_epilogue::<Q4KFamily, SUBGROUPS, COLS_PER_SUBGROUP, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        epilogues,
    )
}

/// Q6K ggml-format qgemv body with optional pre/post unary epilogues.
pub(crate) fn qgemv_q6k_ggml_with_epilogue<
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
    qgemv_ggml_with_epilogue::<Q6KFamily, SUBGROUPS, COLS_PER_SUBGROUP, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        epilogues,
    )
}

/// Generic subgroup-partitioned qgemv body with optional pre- and post-reduce
/// epilogues, covering the formats that don't have a dedicated `qgemv_q*_ggml`
/// path. `pre` is applied to each loaded activation tile before the dot;
/// `post` is applied to each per-output tile before the store.
pub(crate) fn qgemv_perf_with_epilogue<
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
        let scope = qgemv_program_scope::<COLS_PER_SUBGROUP>(program, grid);
        let col0 = scope.col0;
        let lane = scope.lane;

        let zero = TileLiteral::f32(0.0);
        let sums: [Tile; COLS_PER_SUBGROUP] = program.loop_fold_n::<COLS_PER_SUBGROUP, _, _>(
            TileReduceOp::Sum,
            k_iterations,
            [zero; COLS_PER_SUBGROUP],
            |program, loop_index| {
                let k_base = loop_index * k_per_iter + lane.clone() * VALUES_PER_LANE as u32;
                let in_bounds_k = if full_k_iterations {
                    Mask::all()
                } else {
                    k_base.lt(k_size)
                };

                let a_bound: [Tile; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let scalar = program.load(
                        a.at((0, k_base.clone() + i as u32)),
                        in_bounds_k.clone(),
                        0.0,
                    );
                    let k_index = k_base.clone() + i as u32;
                    let pre_extras = epilogues
                        .pre_extra_inputs
                        .iter()
                        .map(|extra| match extra {
                            crate::types::QmatmulExtra::Column(vector) => {
                                program.load(vector.at(&k_index), k_index.lt(k_size), 0.0)
                            }
                            crate::types::QmatmulExtra::Pointwise(tensor) => {
                                program.load(tensor.at((0, &k_index)), k_index.lt(k_size), 0.0)
                            }
                        })
                        .collect::<Vec<_>>();
                    let scalar =
                        crate::types::apply_qmatmul_pre_epilogue(epilogues, scalar, pre_extras);
                    program.bind(scalar)
                });

                let a8 = || -> [Tile; 8] { std::array::from_fn(|i| a_bound[i].clone()) };
                let an =
                    || -> [Tile; VALUES_PER_LANE] { std::array::from_fn(|i| a_bound[i].clone()) };
                std::array::from_fn(|c| {
                    let col = col0.clone() + c as u32;
                    let mask = grid.mask(full_k_iterations, in_bounds_k.clone(), &col);
                    if b_cloned.format == GgmlQuantFormat::Q8_0
                        && VALUES_PER_LANE == 8
                        && grid.n_cols >= 8192
                    {
                        return program.quantized_dot(QuantizedDot::f32_activations(
                            a8(),
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        ));
                    }
                    if b_cloned.format == GgmlQuantFormat::Q4K
                        && (VALUES_PER_LANE == 8 || VALUES_PER_LANE == 16 || VALUES_PER_LANE == 32)
                    {
                        return program.quantized_dot(QuantizedDot::f32_activations::<
                            VALUES_PER_LANE,
                        >(
                            an(), &b_cloned, &k_base, &col, mask, 0.0
                        ));
                    }
                    if b_cloned.format == GgmlQuantFormat::Q6K && VALUES_PER_LANE == 8 {
                        return program.quantized_dot(QuantizedDot::f32_activations(
                            a8(),
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        ));
                    }
                    if b_cloned.format == GgmlQuantFormat::Q6K && !q6k_vocab_f32_dot {
                        return program.quantized_dot(QuantizedDot::q8_activations::<
                            VALUES_PER_LANE,
                        >(
                            an(), &b_cloned, &k_base, &col, mask, 0.0
                        ));
                    }
                    let bs: [Tile; VALUES_PER_LANE] = program
                        .load_quantized_block::<VALUES_PER_LANE>(
                            &b_cloned, &k_base, &col, mask, 0.0,
                        );
                    dot4_sum(program, &an(), &bs)
                })
            },
        );

        crate::grid::store_qgemv_sums_with_epilogue(
            program,
            sums,
            QgemvStoreTarget {
                y,
                col0,
                lane,
                full_cols: grid.full_cols,
                n_cols: grid.n_cols,
                epilogues,
            },
        );
    });
}
