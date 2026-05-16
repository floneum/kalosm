use fusor_tile_ir::{F32, TileLiteral, TileReduceOp, Vector, WorkgroupAxis};

use super::types::RmsNormVec4Meta;
use fusor_tile_ir::tile::Tile;

const RMS_NORM_VEC4_BLOCK: usize = 128;

/// Tensor bindings and shape metadata for [`rms_norm_vec4`].
///
/// The offsets and strides live in [`RmsNormVec4Meta`]; these tensor refs only
/// describe the bound buffers and base layouts.
///
/// ```no_run
/// # use fusor_tile_ir::{F32Bits, KernelBuilder, KernelTensorRef};
/// # use fusor_tile_ir_kernels::{linear_storage_layout, rms_norm_vec4, RmsNormVec4, RmsNormVec4Meta};
/// let layout = linear_storage_layout();
/// let mut kb = KernelBuilder::<()>::new();
/// let input = KernelTensorRef::new((), layout.clone());
/// let weight = KernelTensorRef::new((), layout.clone());
/// let output = KernelTensorRef::new((), layout);
/// rms_norm_vec4(
///     &mut kb,
///     RmsNormVec4 {
///         input,
///         residual: None,
///         weight,
///         bias: None,
///         output,
///         meta: RmsNormVec4Meta {
///             cols: 4,
///             cols_vec: 1,
///             eps: F32Bits::new(1e-5),
///             input_offset_vec: 0,
///             input_row_stride_vec: 1,
///             residual_offset_vec: None,
///             residual_row_stride_vec: 0,
///             weight_offset_vec: 0,
///             bias_offset_vec: None,
///             output_offset_vec: 0,
///             output_row_stride_vec: 1,
///         },
///         rows: 1,
///     },
/// );
/// ```
pub struct RmsNormVec4<B> {
    /// Input tensor, read as packed `vec4<f32>` values.
    pub input: fusor_tile_ir::KernelTensorRef<B>,
    /// Optional residual tensor added before normalization.
    pub residual: Option<fusor_tile_ir::KernelTensorRef<B>>,
    /// Per-column weight tensor, read as packed `vec4<f32>` values.
    pub weight: fusor_tile_ir::KernelTensorRef<B>,
    /// Optional per-column bias tensor.
    pub bias: Option<fusor_tile_ir::KernelTensorRef<B>>,
    /// Output tensor, written as packed `vec4<f32>` values.
    pub output: fusor_tile_ir::KernelTensorRef<B>,
    /// Column counts and vec4 offsets/strides.
    pub meta: RmsNormVec4Meta,
    /// Number of rows to normalize.
    pub rows: u32,
}

/// Build a packed-vec4 F32 RMS-norm kernel.
///
/// Returns `None` when the row/column metadata is empty or the optional
/// residual/bias bindings do not match their optional metadata offsets.
pub fn rms_norm_vec4<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    spec: RmsNormVec4<B>,
) -> Option<()> {
    let RmsNormVec4 {
        input,
        residual,
        weight,
        bias,
        output,
        meta,
        rows,
    } = spec;
    if rows == 0 || meta.cols == 0 || meta.cols_vec == 0 {
        return None;
    }
    if meta.residual_offset_vec.is_some() != residual.is_some()
        || meta.bias_offset_vec.is_some() != bias.is_some()
    {
        return None;
    }

    let chunks = meta.cols_vec.div_ceil(RMS_NORM_VEC4_BLOCK as u32);
    let eps = meta.eps.get();

    let input = kb.read::<Vector<F32, 4>, 1>(input);
    let residual = residual.map(|r| kb.read::<Vector<F32, 4>, 1>(r));
    let weight = kb.read::<Vector<F32, 4>, 1>(weight);
    let bias = bias.map(|b| kb.read::<Vector<F32, 4>, 1>(b));
    let output = kb.write::<Vector<F32, 4>, 1>(output);
    let phase = kb.program();

    phase.program_grid::<RMS_NORM_VEC4_BLOCK>([rows, 1, 1], |program| {
        let row = program.program_id(WorkgroupAxis::X);
        let lane = program.lane();
        let partial_sum = program.loop_fold(
            TileReduceOp::Sum,
            chunks,
            TileLiteral::f32(0.0),
            |program, loop_index| {
                let reduce_col = loop_index * RMS_NORM_VEC4_BLOCK as u32 + lane.clone();
                let reduce_mask = reduce_col.lt(meta.cols_vec);
                let input_index = row.clone() * meta.input_row_stride_vec + reduce_col.clone();
                let mut value = program.load_vector::<F32, 4>(
                    input.at(input_index),
                    reduce_mask.clone(),
                    TileLiteral::f32(0.0),
                );
                if let Some(residual) = &residual {
                    let residual_index =
                        row.clone() * meta.residual_row_stride_vec + reduce_col.clone();
                    value = value
                        + program.load_vector::<F32, 4>(
                            residual.at(residual_index),
                            reduce_mask,
                            TileLiteral::f32(0.0),
                        );
                }
                program.vector_dot::<F32, 4>(value.clone(), value)
            },
        );
        let total_sum = program.group_reduce_sum::<RMS_NORM_VEC4_BLOCK>(partial_sum);
        let mean =
            total_sum / Tile::<RMS_NORM_VEC4_BLOCK>::literal(TileLiteral::f32(meta.cols as f32));
        let scale =
            (mean + Tile::<RMS_NORM_VEC4_BLOCK>::literal(TileLiteral::f32(eps))).inverse_sqrt();
        let scale = program.bind(scale);

        for chunk in 0..chunks {
            let col = lane.clone() + chunk * RMS_NORM_VEC4_BLOCK as u32;
            let mask = col.lt(meta.cols_vec);
            let input_index = row.clone() * meta.input_row_stride_vec + col.clone();
            let mut value = program.load_vector::<F32, 4>(
                input.at(input_index),
                mask.clone(),
                TileLiteral::f32(0.0),
            );
            if let Some(residual) = &residual {
                let residual_index = row.clone() * meta.residual_row_stride_vec + col.clone();
                value = value
                    + program.load_vector::<F32, 4>(
                        residual.at(residual_index),
                        mask.clone(),
                        TileLiteral::f32(0.0),
                    );
            }
            let scale = program.vector_splat::<F32, 4>(scale.get());
            let weight = program.load_vector::<F32, 4>(
                weight.at(col.clone()),
                mask.clone(),
                TileLiteral::f32(0.0),
            );
            let mut normalized = value * scale * weight;
            if let Some(bias) = &bias {
                normalized = normalized
                    + program.load_vector::<F32, 4>(
                        bias.at(col.clone()),
                        mask.clone(),
                        TileLiteral::f32(0.0),
                    );
            }
            let output_index = row.clone() * meta.output_row_stride_vec + col;
            program.store_linear(output.at(output_index), normalized, mask);
        }
    });
    Some(())
}
