//! Quantized GEMV program kernels.
//!
//! Runtime-typed port: the const-generic markers (`Storage<F32, 2>` /
//! `Tile<U32>` / `program_grid::<BLOCK>`) are gone — every handle carries its
//! [`fusor_tile_ir::ElementType`] / dims as data and the workgroup `block` is a
//! runtime `u32`.
//!
//! The over-fused multi-format `QuantizedDot` op surface (the
//! `q{4,6}k_block` GGML fused dots) was removed from the IR. The composable
//! quantized primitive is `Expr::Dequantize` (one `Shared(Dequantize)` node
//! projected per-lane with `LaneOf`), surfaced as
//! [`load_quantized_block_vec`](fusor_tile_ir::tile::TileBlock::load_quantized_block_vec);
//! the kernel composes an ordinary dot over it via [`dot4_sum`]. The two fused
//! dots that composition cannot match compactly — the f32 block dot and the Q8
//! DP4a dot — are kept as `quantized_dot_f32` / `quantized_dot_q8`. The
//! `(format, values_per_lane)` choice that picks between them stays in the
//! kernel as [`select_qgemv_dot`]; `values_per_lane` is a caller TILING choice,
//! not `(format, dims)`-derivable, so it is not pushed into the lowerer.

use fusor_tile_ir::tile::{range, Mask, Program, Storage, Tile, TileBlock};
use fusor_tile_ir::{GgmlQuantFormat, QuantizedMatrix, TileLiteral};

use crate::dispatch::{
    q4k_default_large, q4k_default_mid, q4k_default_tall, q4k_large_override, q4k_mid_override,
    q4k_tall_override, q6k_default_large, q6k_default_tall, q6k_large_override, q6k_tall_override,
    qgemv_subgroups_per_workgroup_for_shape, QgemvShape, SubgroupConfig,
};
use crate::grid::{
    dot4_sum, qgemv_grid, qgemv_program_scope, store_qgemv_sums_with_epilogue, QgemvStoreTarget,
};
use crate::kernels::qgemv_q4k_ggml::{
    load_q4k_ggml_activations, q4k_ggml_dot_tiles, q4k_lane_decomposition,
};
use crate::kernels::qgemv_q6k::qgemv_q6k_ggml;
use crate::types::{apply_qmatmul_pre_epilogue, matrix_shape, QmatmulEpilogues, QmatmulExtra};

/// Converts qgemv epilogue inputs into the internal pre/post epilogue bundle.
///
/// Public callers normally pass either `Option<&UnaryEpilogue>` for a post-only
/// epilogue or `&QmatmulEpilogues` for explicit pre/post control.
pub trait IntoQgemvEpilogues<'a> {
    /// Convert into a qgemv epilogue bundle.
    fn into_qgemv_epilogues(self) -> QmatmulEpilogues<'a>;
}

impl<'a> IntoQgemvEpilogues<'a> for Option<&'a crate::UnaryEpilogue> {
    fn into_qgemv_epilogues(self) -> QmatmulEpilogues<'a> {
        QmatmulEpilogues {
            pre: None,
            pre_with_extras: None,
            pre_extra_inputs: &[],
            post: self,
            post_with_extras: None,
            post_extra_inputs: &[],
            post_accumulator_offsets: &[],
            post_acc_init_col_vector: None,
        }
    }
}

impl<'a> IntoQgemvEpilogues<'a> for &'a QmatmulEpilogues<'a> {
    fn into_qgemv_epilogues(self) -> QmatmulEpilogues<'a> {
        self.clone()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QgemvTensors<'a> {
    pub(crate) a: &'a Storage,
    pub(crate) b: &'a QuantizedMatrix,
    pub(crate) y: &'a Storage,
}

fn qgemv_shape(subgroups: u32, cols_per_subgroup: u32) -> QgemvShape {
    QgemvShape {
        subgroups,
        cols_per_subgroup,
    }
}

/// Top-level quantized GEMV with optional pre/post unary epilogues.
///
/// Equivalent to [`crate::qmatmul_with_epilogue`] with `BM = 1`. Callers
/// with no epilogue pass `None` (or `Option::<&UnaryEpilogue>::None`).
///
/// ```
/// use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, ScalarElement};
/// use fusor_tile_ir_kernels::{qgemv_with_epilogue, quantized_matrix, UnaryEpilogue};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read(ScalarElement::F32.element(), Shape::new([1, 256]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 128);
///     let y = program.storage_write(ScalarElement::F32.element(), Shape::new([1, 128]));
///     let subgroup = fusor_tile_ir::SubgroupToken::new_unchecked();
///     qgemv_with_epilogue(
///         program,
///         &a,
///         &b,
///         &y,
///         1,
///         fusor_tile_ir_kernels::SubgroupConfig::fixed(subgroup, 32),
///         Option::<&UnaryEpilogue>::None,
///     );
/// });
/// # let _ = ir;
/// ```
pub fn qgemv_with_epilogue<'a>(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    workgroups_x: u32,
    subgroups: SubgroupConfig,
    epilogues: impl IntoQgemvEpilogues<'a>,
) {
    let epilogues = epilogues.into_qgemv_epilogues();
    qgemv_tile_with_epilogue(program, a, b, y, workgroups_x, subgroups, &epilogues);
}

/// Format-dispatched qgemv body with optional pre/post unary epilogues.
///
/// Every arm routes to [`qgemv_perf_with_epilogue`], the single surviving body.
/// The per-format Q4K/Q6K "ggml" specializations consumed the over-fused
/// `QuantizedDot::q{4,6}k_block` dot, which the IR no longer exposes (the
/// `q{4,6}k_ggml_dot` lowering helpers are dead with no builder front-door); the
/// composable `Dequantize` + [`dot4_sum`] path replaces them. The shape tables
/// (`q4k_*_override` / `q6k_*_override`) are kept so the workgroup geometry for
/// the large/mid/tall regimes is unchanged; only the dot composition differs.
pub(crate) fn qgemv_tile_with_epilogue(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    workgroups_x: u32,
    subgroups: SubgroupConfig,
    ep: &QmatmulEpilogues<'_>,
) {
    let [m, _] = matrix_shape(a.layout());
    assert_eq!(m, 1, "qgemv requires a single input row");
    let tensors = QgemvTensors { a, b, y };
    let output_cols = ep.post_output_cols(b.cols);

    match b.format {
        GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native => {
            if output_cols >= 8192 {
                return qgemv_perf_with_epilogue(
                    program,
                    tensors,
                    workgroups_x,
                    subgroups,
                    ep,
                    qgemv_shape(4, 8),
                    8,
                );
            }
            qgemv_perf_with_epilogue(
                program,
                tensors,
                workgroups_x,
                subgroups,
                ep,
                qgemv_shape(4, 4),
                8,
            )
        }
        GgmlQuantFormat::Q8_1 => qgemv_perf_with_epilogue(
            program,
            tensors,
            workgroups_x,
            subgroups,
            ep,
            qgemv_shape(4, 4),
            8,
        ),
        GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative => {
            let shape = if b.rows <= 4096 && (4096..8192).contains(&output_cols) {
                q4k_mid_override(q4k_default_mid(b.rows, output_cols))
            } else if b.rows <= 4096 && output_cols <= 4096 {
                qgemv_shape(8, 4)
            } else if b.rows <= 4096 && output_cols >= 8192 {
                q4k_large_override(q4k_default_large(b.rows, output_cols))
            } else if b.rows > 4096 && output_cols <= 4096 {
                q4k_tall_override(q4k_default_tall(b.rows, output_cols))
            } else if qgemv_subgroups_per_workgroup_for_shape(b.format, b.rows, output_cols) == 8 {
                qgemv_shape(8, 8)
            } else {
                qgemv_shape(4, 8)
            };
            // The decode matmuls (no pre-epilogue) take the ggml super-block-
            // amortized dot, which decodes each 256-element super-block's
            // scale/min once per lane instead of re-decoding per 16-element
            // chunk. The rare pre-epilogue case keeps the generic dequant dot
            // (the strided ggml gather has no per-`k` index to feed a pre-op).
            if b.rows.is_multiple_of(b.format.block_elements())
                && qgemv_pre_epilogue_is_empty(ep)
                && subgroups.supports_lanes_per_item(8)
            {
                return qgemv_q4k_ggml(program, tensors, workgroups_x, subgroups, ep, shape);
            }
            let values_per_lane = if shape.cols_per_subgroup == 8 { 8 } else { 16 };
            qgemv_perf_with_epilogue(
                program,
                tensors,
                workgroups_x,
                subgroups,
                ep,
                shape,
                values_per_lane,
            )
        }
        GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native => qgemv_perf_with_epilogue(
            program,
            tensors,
            workgroups_x,
            subgroups,
            ep,
            qgemv_shape(2, 4),
            16,
        ),
        GgmlQuantFormat::Q4_0
        | GgmlQuantFormat::Q4_0Native
        | GgmlQuantFormat::Q4_1
        | GgmlQuantFormat::Q5_1
        | GgmlQuantFormat::Q2K => qgemv_perf_with_epilogue(
            program,
            tensors,
            workgroups_x,
            subgroups,
            ep,
            qgemv_shape(2, 4),
            8,
        ),
        GgmlQuantFormat::Q3K | GgmlQuantFormat::Q8K => qgemv_perf_with_epilogue(
            program,
            tensors,
            workgroups_x,
            subgroups,
            ep,
            qgemv_shape(2, 2),
            8,
        ),
        GgmlQuantFormat::Q5K | GgmlQuantFormat::Q5KNative => qgemv_perf_with_epilogue(
            program,
            tensors,
            workgroups_x,
            subgroups,
            ep,
            qgemv_shape(2, 1),
            8,
        ),
        GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative => {
            if b.rows <= 4096 && output_cols >= 8192 {
                let shape = q6k_large_override(q6k_default_large(b.rows, output_cols));
                return qgemv_perf_with_epilogue(
                    program,
                    tensors,
                    workgroups_x,
                    subgroups,
                    ep,
                    shape,
                    8,
                );
            }
            if b.rows > 4096 && output_cols <= 4096 {
                let shape = q6k_tall_override(q6k_default_tall(b.rows, output_cols));
                return qgemv_perf_with_epilogue(
                    program,
                    tensors,
                    workgroups_x,
                    subgroups,
                    ep,
                    shape,
                    8,
                );
            }
            let (shape, values_per_lane) =
                if qgemv_subgroups_per_workgroup_for_shape(b.format, b.rows, output_cols) == 4 {
                    (qgemv_shape(4, 4), 8)
                } else {
                    (qgemv_shape(8, 4), 16)
                };
            // The decode matmuls (no pre-epilogue) take the ggml super-block-
            // amortized dot, which decodes each 256-element super-block's `d`
            // and sub-block scales once per 16-element lane region instead of
            // re-decoding per 8-element chunk. Only the word-aligned f32-scale
            // `Q6K` layout uses raw-word addressing; the 210-byte `Q6KNative`
            // block and the rare pre-epilogue case keep the generic dequant dot.
            if b.rows.is_multiple_of(b.format.block_elements())
                && b.format == GgmlQuantFormat::Q6K
                && qgemv_pre_epilogue_is_empty(ep)
                && subgroups.supports_lanes_per_item(16)
            {
                return qgemv_q6k_ggml(program, tensors, workgroups_x, subgroups, ep, shape);
            }
            qgemv_perf_with_epilogue(
                program,
                tensors,
                workgroups_x,
                subgroups,
                ep,
                shape,
                values_per_lane,
            )
        }
    }
}

/// Which dot the perf body emits for a `(format, values_per_lane)` pair.
/// `values_per_lane` is a caller TILING choice, not `(format, dims)`-derivable,
/// so the selection stays in the kernel rather than the lowerer.
///
/// `F32Vec` and `BlockThenDot4` both dequantize the block to f32 and compose
/// [`dot4_sum`] (the over-fused `f32_activations_vec` builder lowered
/// identically to dequant+dot4, so they are folded together). `Q8Vec` is the
/// irreducible Q8 DP4a path — it keeps the weights quantized and emits
/// `Dot4I8Packed` via [`quantized_dot_q8`](TileBlock::quantized_dot_q8), which
/// `Dequantize` + `dot4_sum` cannot express.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum QgemvDot {
    /// Dequantize the block and dot against the f32 activation vector (folds
    /// into `BlockThenDot4` — byte-identical lowering).
    F32Vec,
    /// Q8 DP4a dot against int8-packed activations, weights kept quantized.
    Q8Vec,
    /// Dequantize the block to `values_per_lane` f32 tiles via one
    /// `Shared(Dequantize)`, then compose `dot4_sum`.
    BlockThenDot4,
}

/// Classify the qgemv dot path for `(format, values_per_lane, q6k_vocab_f32_dot)`.
/// Mirrors the original `qgemv_perf` format if-chain. `n_cols`-derived gates
/// (the q8 `>= 8192` accumulate split) live in the lowerer, so they are not an
/// input here.
fn select_qgemv_dot(
    format: GgmlQuantFormat,
    values_per_lane: u32,
    q6k_vocab_f32_dot: bool,
) -> QgemvDot {
    if format.is_q8_0_family() && values_per_lane == 8 {
        return QgemvDot::F32Vec;
    }
    if format.is_q4k_family()
        && (values_per_lane == 8 || values_per_lane == 16 || values_per_lane == 32)
    {
        return QgemvDot::F32Vec;
    }
    if format.is_q6k_family() && values_per_lane == 8 {
        return QgemvDot::F32Vec;
    }
    if format.is_q6k_family() && !q6k_vocab_f32_dot {
        return QgemvDot::Q8Vec;
    }
    QgemvDot::BlockThenDot4
}

/// Subgroup-partitioned qgemv body with optional pre- and post-reduce
/// epilogues. `pre` is applied to each loaded activation tile before the dot;
/// `post` is applied to each per-output tile before the store.
fn qgemv_perf_with_epilogue(
    program: &mut Program,
    tensors: QgemvTensors<'_>,
    workgroups_x: u32,
    subgroups: SubgroupConfig,
    epilogues: &QmatmulEpilogues<'_>,
    shape: QgemvShape,
    values_per_lane: u32,
) {
    let QgemvTensors { a, b, y } = tensors;
    let subgroup = subgroups.token();
    let block = subgroups.block_for_subgroups(shape.subgroups);
    let dispatch_subgroups = shape.subgroups;
    let cols_per_subgroup = shape.cols_per_subgroup;
    debug_assert!(values_per_lane == 8 || values_per_lane == 16 || values_per_lane == 32);
    debug_assert!(matches!(cols_per_subgroup, 1 | 2 | 3 | 4 | 8));
    let [_, k] = matrix_shape(a.layout());
    let output_cols = epilogues.post_output_cols(b.cols);
    let grid = qgemv_grid(
        dispatch_subgroups,
        cols_per_subgroup,
        output_cols,
        workgroups_x,
    );
    let k_size = k;
    let b_cloned = b.clone();
    let q6k_vocab_f32_dot = b.format.is_q6k_family() && b.rows <= 4096 && b.cols >= 65_536;
    let dot_path = select_qgemv_dot(b.format, values_per_lane, q6k_vocab_f32_dot);
    let cols_per_subgroup_usize = cols_per_subgroup as usize;
    let post_accumulator_offsets = (!epilogues.post_accumulator_offsets.is_empty())
        .then(|| epilogues.post_accumulator_offsets().to_vec());

    program.program_grid(block, [grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let scope = qgemv_program_scope(program, grid, cols_per_subgroup, subgroup);
        let col0 = scope.col0;
        let lane = scope.lane;
        let k_per_iter = subgroup.subgroup_size(program) * values_per_lane;
        let k_iterations = (Tile::u32(k) + k_per_iter.clone() - 1u32) / k_per_iter.clone();

        let zero = Tile::literal(TileLiteral::f32(0.0));
        let sums: Vec<Tile> = if let Some(post_accumulator_offsets) = &post_accumulator_offsets {
            let value_arity = post_accumulator_offsets.len();
            program.fold_vec(
                range(k_iterations),
                vec![zero; cols_per_subgroup_usize * value_arity],
                |program, loop_index, accs| {
                    let k_base = loop_index * k_per_iter.clone() + lane.clone() * values_per_lane;
                    let in_bounds_k = k_base.lt(k_size);

                    let a_bound = load_qgemv_activations(
                        program,
                        a,
                        epilogues,
                        values_per_lane,
                        &k_base,
                        k_size,
                        in_bounds_k.clone(),
                    );

                    accs.into_iter()
                        .enumerate()
                        .map(|(idx, acc)| {
                            let c = idx / value_arity;
                            let value_idx = idx % value_arity;
                            let output_col = col0.clone() + c as u32;
                            let matrix_col =
                                output_col.clone() + post_accumulator_offsets[value_idx];
                            let mask = grid.mask(in_bounds_k.clone(), &output_col);
                            let part = qgemv_dot_part(
                                program,
                                dot_path,
                                &a_bound,
                                &b_cloned,
                                values_per_lane,
                                &k_base,
                                &matrix_col,
                                mask,
                            );
                            acc + part
                        })
                        .collect()
                },
            )
        } else {
            program.fold_vec(
                range(k_iterations),
                vec![zero; cols_per_subgroup_usize],
                |program, loop_index, accs| {
                    let k_base = loop_index * k_per_iter.clone() + lane.clone() * values_per_lane;
                    let in_bounds_k = k_base.lt(k_size);

                    let a_bound = load_qgemv_activations(
                        program,
                        a,
                        epilogues,
                        values_per_lane,
                        &k_base,
                        k_size,
                        in_bounds_k.clone(),
                    );

                    accs.into_iter()
                        .enumerate()
                        .map(|(c, acc)| {
                            let col = col0.clone() + c as u32;
                            let mask = grid.mask(in_bounds_k.clone(), &col);
                            let part = qgemv_dot_part(
                                program,
                                dot_path,
                                &a_bound,
                                &b_cloned,
                                values_per_lane,
                                &k_base,
                                &col,
                                mask,
                            );
                            acc + part
                        })
                        .collect()
                },
            )
        };

        store_qgemv_sums_with_epilogue(
            program,
            sums,
            QgemvStoreTarget {
                y,
                subgroup,
                col0,
                lane,
                n_cols: grid.n_cols,
                epilogues,
            },
        );
    });
}

fn load_qgemv_activations(
    program: &mut TileBlock<'_>,
    a: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    values_per_lane: u32,
    k_base: &Tile,
    k_size: u32,
    in_bounds_k: Mask,
) -> Vec<Tile> {
    (0..values_per_lane)
        .map(|i| {
            let scalar = program.load(a.at((0u32, k_base.clone() + i)), in_bounds_k.clone(), 0.0);
            let k_index = k_base.clone() + i;
            let pre_extras = epilogues
                .pre_extra_inputs
                .iter()
                .map(|extra| match extra {
                    QmatmulExtra::Column(vector) => {
                        program.load(vector.at(&k_index), k_index.lt(k_size), 0.0)
                    }
                    QmatmulExtra::Pointwise(tensor) => {
                        program.load(tensor.at((0u32, &k_index)), k_index.lt(k_size), 0.0)
                    }
                })
                .collect::<Vec<_>>();
            let scalar = apply_qmatmul_pre_epilogue(epilogues, scalar, pre_extras);
            program.bind(scalar)
        })
        .collect()
}

/// `true` when no pre-reduce epilogue is attached, so the activation stream can
/// be gathered with the ggml strided pattern (which has no per-`k` index to feed
/// a pre-epilogue).
fn qgemv_pre_epilogue_is_empty(epilogues: &QmatmulEpilogues<'_>) -> bool {
    epilogues.pre.is_none()
        && epilogues.pre_with_extras.is_none()
        && epilogues.pre_extra_inputs.is_empty()
}

/// Q4K qgemv built on the ggml super-block-amortized decode. Each 8-lane
/// chunk covers one super-block; wider subgroups cover proportionally more
/// super-blocks per pass. This restores the per-super-block decode
/// amortization that the generic `quantized_dot_f32` path lost when it
/// re-decodes the metadata for every 8/16-element chunk. Only valid with an
/// empty pre-epilogue.
fn qgemv_q4k_ggml(
    program: &mut Program,
    tensors: QgemvTensors<'_>,
    workgroups_x: u32,
    subgroups: SubgroupConfig,
    epilogues: &QmatmulEpilogues<'_>,
    shape: QgemvShape,
) {
    let QgemvTensors { a, b, y } = tensors;
    let subgroup = subgroups.token();
    let block = subgroups.block_for_subgroups(shape.subgroups);
    let dispatch_subgroups = shape.subgroups;
    let cols_per_subgroup = shape.cols_per_subgroup;
    debug_assert!(subgroups.supports_lanes_per_item(8));
    debug_assert!(b.format.is_q4k_family());
    let output_cols = epilogues.post_output_cols(b.cols);
    let grid = qgemv_grid(
        dispatch_subgroups,
        cols_per_subgroup,
        output_cols,
        workgroups_x,
    );
    let [_, k] = matrix_shape(a.layout());
    let block_count = k.div_ceil(256);
    let blocks_per_col = b.rows / b.format.block_elements();
    let block_words = b.format.block_words();
    let native = b.format == GgmlQuantFormat::Q4KNative;
    let qwords = Storage::from_view(b.data.clone());
    let cols_usize = cols_per_subgroup as usize;
    let post_accumulator_offsets = (!epilogues.post_accumulator_offsets.is_empty())
        .then(|| epilogues.post_accumulator_offsets().to_vec());
    let row = Tile::u32(0);

    program.program_grid(block, [grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let scope = qgemv_program_scope(program, grid, cols_per_subgroup, subgroup);
        let col0 = scope.col0;
        let lane = scope.lane;
        let q4k_lane = q4k_lane_decomposition(&lane);
        let blocks_per_pass = subgroup.subgroup_size(program) / 8u32;
        let block_iterations =
            (Tile::u32(block_count) + blocks_per_pass.clone() - 1u32) / blocks_per_pass.clone();

        let sums: Vec<Tile> = if let Some(post_accumulator_offsets) = &post_accumulator_offsets {
            let value_arity = post_accumulator_offsets.len();
            program.fold_vec(
                range(block_iterations),
                vec![Tile::f32(0.0); cols_usize * value_arity],
                |program, loop_index, accs| {
                    let block_idx = loop_index * blocks_per_pass.clone() + q4k_lane.ix.clone();
                    let in_bounds = block_idx.clone().lt(block_count);
                    let vector_base = block_idx.clone() * 256u32
                        + q4k_lane.iq.clone() * 64u32
                        + q4k_lane.ir.clone() * 8u32;
                    let acts =
                        load_q4k_ggml_activations(program, a, &row, &vector_base, &in_bounds);
                    accs.into_iter()
                        .enumerate()
                        .map(|(idx, acc)| {
                            let c = idx / value_arity;
                            let value_idx = idx % value_arity;
                            let output_col = col0.clone() + c as u32;
                            let matrix_col =
                                output_col.clone() + post_accumulator_offsets[value_idx];
                            acc + q4k_ggml_dot_tiles(
                                program,
                                &qwords,
                                blocks_per_col,
                                block_words,
                                native,
                                &block_idx,
                                &matrix_col,
                                &q4k_lane,
                                &acts,
                            )
                        })
                        .collect()
                },
            )
        } else {
            program.fold_vec(
                range(block_iterations),
                vec![Tile::f32(0.0); cols_usize],
                |program, loop_index, accs| {
                    let block_idx = loop_index * blocks_per_pass.clone() + q4k_lane.ix.clone();
                    let in_bounds = block_idx.clone().lt(block_count);
                    let vector_base = block_idx.clone() * 256u32
                        + q4k_lane.iq.clone() * 64u32
                        + q4k_lane.ir.clone() * 8u32;
                    let acts =
                        load_q4k_ggml_activations(program, a, &row, &vector_base, &in_bounds);
                    accs.into_iter()
                        .enumerate()
                        .map(|(c, acc)| {
                            let col = col0.clone() + c as u32;
                            acc + q4k_ggml_dot_tiles(
                                program,
                                &qwords,
                                blocks_per_col,
                                block_words,
                                native,
                                &block_idx,
                                &col,
                                &q4k_lane,
                                &acts,
                            )
                        })
                        .collect()
                },
            )
        };

        store_qgemv_sums_with_epilogue(
            program,
            sums,
            QgemvStoreTarget {
                y,
                subgroup,
                col0,
                lane,
                n_cols: grid.n_cols,
                epilogues,
            },
        );
    });
}

/// Emit one column's quantized dot contribution for the perf body.
///
/// The [`QgemvDot::Q8Vec`] path keeps the weights quantized and emits the Q8
/// DP4a dot ([`quantized_dot_q8`](TileBlock::quantized_dot_q8)) — the fast path
/// that `Dequantize` + [`dot4_sum`] cannot express. The `F32Vec` and
/// `BlockThenDot4` paths both dequantize the block to f32 and compose
/// [`dot4_sum`]: `f32_activations_vec` lowered identically to dequant+dot4, so
/// folding them together is byte-identical (verified by the qgemv goldens).
#[allow(clippy::too_many_arguments)]
fn qgemv_dot_part(
    program: &mut TileBlock<'_>,
    dot_path: QgemvDot,
    a_bound: &[Tile],
    b: &QuantizedMatrix,
    values_per_lane: u32,
    k_base: &Tile,
    col: &Tile,
    mask: Mask,
) -> Tile {
    match dot_path {
        QgemvDot::F32Vec => program.quantized_dot_f32(a_bound, b, k_base, col, mask, 0.0),
        QgemvDot::Q8Vec => program.quantized_dot_q8(a_bound, b, k_base, col, mask, 0.0),
        QgemvDot::BlockThenDot4 => {
            let bs = program.load_quantized_block_vec(values_per_lane, b, k_base, col, mask, 0.0);
            dot4_sum(program, a_bound, &bs)
        }
    }
}
