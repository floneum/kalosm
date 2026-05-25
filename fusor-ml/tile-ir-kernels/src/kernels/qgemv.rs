//! Quantized GEMV program kernels.

use fusor_tile_ir::tile::{
    BlockCoord, Mask, Program, Q4KActivations, QuantizedDot, Storage, Tile, TileBlock,
};
use fusor_tile_ir::{F32, GgmlQuantFormat, QuantizedMatrix, TileLiteral, TileReduceOp, U32};

use crate::dispatch::{
    QgemvShape, q4k_default_large, q4k_default_mid, q4k_default_tall, q4k_large_override,
    q4k_mid_override, q4k_tall_override, q6k_default_large, q6k_default_tall, q6k_large_override,
    q6k_tall_override, qgemv_subgroups_per_workgroup_for_shape,
};
use crate::grid::{
    Q4KGgmlIterationRequest, Q4KLane, Q6KGgmlIterationRequest, Q6KLane, QgemvStoreTarget, dot4_sum,
    q4k_ggml_iteration, q4k_lane_decomposition, q6k_ggml_iteration, q6k_lane_decomposition,
    qgemv_grid, qgemv_program_scope,
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
///     qgemv_with_epilogue(program, &a, &b, &y, 1, Option::<&UnaryEpilogue>::None);
/// });
/// # let _ = ir;
/// ```
pub fn qgemv_with_epilogue<'a>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: impl IntoQgemvEpilogues<'a>,
) {
    let epilogues = epilogues.into_qgemv_epilogues();
    qgemv_tile_with_epilogue(program, a, b, y, workgroups_x, &epilogues);
}

/// Variant-dispatched Q4K ggml qgemv. Picks the right monomorphization for
/// the supplied shape and threads optional unary epilogues through it.
pub(crate) fn qgemv_q4k_dispatch_with_epilogue(
    program: &mut Program,
    shape: impl Into<QgemvShape>,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    let shape = shape.into();
    let s = shape.subgroups;
    let c = shape.cols_per_subgroup;
    match shape.block {
        32 => qgemv_q4k_ggml_with_epilogue::<32>(program, a, b, y, workgroups_x, epilogues, s, c),
        64 => qgemv_q4k_ggml_with_epilogue::<64>(program, a, b, y, workgroups_x, epilogues, s, c),
        128 => qgemv_q4k_ggml_with_epilogue::<128>(program, a, b, y, workgroups_x, epilogues, s, c),
        256 => qgemv_q4k_ggml_with_epilogue::<256>(program, a, b, y, workgroups_x, epilogues, s, c),
        other => panic!("unsupported Q4K qgemv BLOCK {other}"),
    }
}

/// Variant-dispatched Q6K ggml qgemv. Same role as the Q4K dispatch helper,
/// but for Q6K shapes.
pub(crate) fn qgemv_q6k_dispatch_with_epilogue(
    program: &mut Program,
    shape: impl Into<QgemvShape>,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
) {
    let shape = shape.into();
    let s = shape.subgroups;
    let c = shape.cols_per_subgroup;
    match shape.block {
        64 => qgemv_q6k_ggml_with_epilogue::<64>(program, a, b, y, workgroups_x, epilogues, s, c),
        128 => qgemv_q6k_ggml_with_epilogue::<128>(program, a, b, y, workgroups_x, epilogues, s, c),
        256 => qgemv_q6k_ggml_with_epilogue::<256>(program, a, b, y, workgroups_x, epilogues, s, c),
        other => panic!("unsupported Q6K qgemv BLOCK {other}"),
    }
}

/// Format-dispatched qgemv body with optional pre/post unary epilogues.
pub(crate) fn qgemv_tile_with_epilogue(
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
                return qgemv_perf_with_epilogue::<128>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                    4,
                    8,
                    8,
                );
            }
            qgemv_perf_with_epilogue::<128>(program, a, b, y, workgroups_x, ep, 4, 4, 8)
        }
        GgmlQuantFormat::Q8_1 => {
            qgemv_perf_with_epilogue::<128>(program, a, b, y, workgroups_x, ep, 4, 4, 8)
        }
        GgmlQuantFormat::Q4K => {
            if b.rows <= 4096 && b.cols >= 4096 && b.cols < 8192 {
                let shape = q4k_mid_override(q4k_default_mid(b.rows, b.cols));
                return qgemv_q4k_dispatch_with_epilogue(program, shape, a, b, y, workgroups_x, ep);
            }
            if b.rows <= 4096 && b.cols <= 4096 {
                return qgemv_perf_with_epilogue::<256>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                    8,
                    4,
                    16,
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
                return qgemv_perf_with_epilogue::<256>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                    8,
                    8,
                    8,
                );
            }
            qgemv_perf_with_epilogue::<128>(program, a, b, y, workgroups_x, ep, 4, 8, 8)
        }
        GgmlQuantFormat::Q5_0 => {
            qgemv_perf_with_epilogue::<64>(program, a, b, y, workgroups_x, ep, 2, 4, 16)
        }
        GgmlQuantFormat::Q4_0
        | GgmlQuantFormat::Q4_1
        | GgmlQuantFormat::Q5_1
        | GgmlQuantFormat::Q2K => {
            qgemv_perf_with_epilogue::<64>(program, a, b, y, workgroups_x, ep, 2, 4, 8)
        }
        GgmlQuantFormat::Q3K | GgmlQuantFormat::Q8K => {
            qgemv_perf_with_epilogue::<64>(program, a, b, y, workgroups_x, ep, 2, 2, 8)
        }
        GgmlQuantFormat::Q5K => {
            qgemv_perf_with_epilogue::<64>(program, a, b, y, workgroups_x, ep, 2, 1, 8)
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
                return qgemv_perf_with_epilogue::<128>(
                    program,
                    a,
                    b,
                    y,
                    workgroups_x,
                    ep,
                    4,
                    4,
                    8,
                );
            }
            qgemv_perf_with_epilogue::<256>(program, a, b, y, workgroups_x, ep, 8, 4, 16)
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
fn qgemv_ggml_with_epilogue<F: GgmlQuantFamily, const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    subgroups: u32,
    cols_per_subgroup: u32,
) {
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert_eq!(subgroups * SUBGROUP_SIZE, BLOCK as u32);
    debug_assert_eq!(b.format, F::FORMAT);

    let [_, k] = matrix_shape(&a.view().layout);
    let grid = qgemv_grid(subgroups, cols_per_subgroup, b.cols, workgroups_x);
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(F::BLOCK_DIV);
    let full_block_iterations = block_count.is_multiple_of(F::BLOCK_DIV);
    let b_cloned = b.clone();
    let cols_per_subgroup_usize = cols_per_subgroup as usize;

    program.program_grid::<BLOCK>([grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let scope = qgemv_program_scope(program, grid, cols_per_subgroup);
        let col0 = scope.col0;
        let lane = scope.lane;
        let fmt_lane = F::lane_decomposition(&lane);

        let zero = TileLiteral::f32(0.0);
        let sums: Vec<Tile> = program.loop_fold_vec(
            TileReduceOp::Sum,
            block_iterations,
            vec![zero; cols_per_subgroup_usize],
            |program, loop_index| {
                let pass = F::iteration(
                    program,
                    loop_index,
                    a,
                    block_count,
                    full_block_iterations,
                    &fmt_lane,
                );

                (0..cols_per_subgroup)
                    .map(|c| {
                        let col = col0.clone() + c;
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
                    .collect()
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
pub(crate) fn qgemv_q4k_ggml_with_epilogue<const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    subgroups: u32,
    cols_per_subgroup: u32,
) {
    qgemv_ggml_with_epilogue::<Q4KFamily, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        epilogues,
        subgroups,
        cols_per_subgroup,
    )
}

/// Q6K ggml-format qgemv body with optional pre/post unary epilogues.
pub(crate) fn qgemv_q6k_ggml_with_epilogue<const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    subgroups: u32,
    cols_per_subgroup: u32,
) {
    qgemv_ggml_with_epilogue::<Q6KFamily, BLOCK>(
        program,
        a,
        b,
        y,
        workgroups_x,
        epilogues,
        subgroups,
        cols_per_subgroup,
    )
}

/// Generic subgroup-partitioned qgemv body with optional pre- and post-reduce
/// epilogues, covering the formats that don't have a dedicated `qgemv_q*_ggml`
/// path. `pre` is applied to each loaded activation tile before the dot;
/// `post` is applied to each per-output tile before the store.
pub(crate) fn qgemv_perf_with_epilogue<const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    workgroups_x: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    subgroups: u32,
    cols_per_subgroup: u32,
    values_per_lane: u32,
) {
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert_eq!(subgroups * SUBGROUP_SIZE, BLOCK as u32);
    debug_assert!(values_per_lane == 8 || values_per_lane == 16 || values_per_lane == 32);
    debug_assert!(matches!(cols_per_subgroup, 1 | 2 | 4 | 8));
    let [_, k] = matrix_shape(&a.view().layout);
    let grid = qgemv_grid(subgroups, cols_per_subgroup, b.cols, workgroups_x);
    let k_per_iter = SUBGROUP_SIZE * values_per_lane;
    let k_iterations = k.div_ceil(k_per_iter);
    let k_size = k;
    let full_k_iterations = k.is_multiple_of(k_per_iter);
    let b_cloned = b.clone();
    let q6k_vocab_f32_dot = b.format == GgmlQuantFormat::Q6K && b.rows <= 4096 && b.cols >= 65_536;
    let cols_per_subgroup_usize = cols_per_subgroup as usize;
    program.program_grid::<BLOCK>([grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let scope = qgemv_program_scope(program, grid, cols_per_subgroup);
        let col0 = scope.col0;
        let lane = scope.lane;

        let zero = TileLiteral::f32(0.0);
        let sums: Vec<Tile> = program.loop_fold_vec(
            TileReduceOp::Sum,
            k_iterations,
            vec![zero; cols_per_subgroup_usize],
            |program, loop_index| {
                let k_base = loop_index * k_per_iter + lane.clone() * values_per_lane;
                let in_bounds_k = if full_k_iterations {
                    Mask::all()
                } else {
                    k_base.lt(k_size)
                };

                let a_bound: Vec<Tile> = (0..values_per_lane)
                    .map(|i| {
                        let scalar =
                            program.load(a.at((0, k_base.clone() + i)), in_bounds_k.clone(), 0.0);
                        let k_index = k_base.clone() + i;
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
                    })
                    .collect();

                (0..cols_per_subgroup)
                    .map(|c| {
                        let col = col0.clone() + c;
                        let mask = grid.mask(full_k_iterations, in_bounds_k.clone(), &col);
                        if b_cloned.format == GgmlQuantFormat::Q8_0
                            && values_per_lane == 8
                            && grid.n_cols >= 8192
                        {
                            return program.quantized_dot(QuantizedDot::f32_activations_vec(
                                a_bound.clone(),
                                &b_cloned,
                                &k_base,
                                &col,
                                mask,
                                0.0,
                            ));
                        }
                        if b_cloned.format == GgmlQuantFormat::Q4K
                            && (values_per_lane == 8
                                || values_per_lane == 16
                                || values_per_lane == 32)
                        {
                            return program.quantized_dot(QuantizedDot::f32_activations_vec(
                                a_bound.clone(),
                                &b_cloned,
                                &k_base,
                                &col,
                                mask,
                                0.0,
                            ));
                        }
                        if b_cloned.format == GgmlQuantFormat::Q6K && values_per_lane == 8 {
                            return program.quantized_dot(QuantizedDot::f32_activations_vec(
                                a_bound.clone(),
                                &b_cloned,
                                &k_base,
                                &col,
                                mask,
                                0.0,
                            ));
                        }
                        if b_cloned.format == GgmlQuantFormat::Q6K && !q6k_vocab_f32_dot {
                            return program.quantized_dot(QuantizedDot::q8_activations_vec(
                                a_bound.clone(),
                                &b_cloned,
                                &k_base,
                                &col,
                                mask,
                                0.0,
                            ));
                        }
                        let bs = program.load_quantized_block_vec(
                            values_per_lane,
                            &b_cloned,
                            &k_base,
                            &col,
                            mask,
                            0.0,
                        );
                        dot4_sum(program, &a_bound, &bs)
                    })
                    .collect()
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
