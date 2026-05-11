use crate::{F32Bits, TileLiteral, TileReduceOp, WorkgroupAxis};

use super::types::RmsNormVec4Meta;
use crate::tile::Tile;

const RMS_NORM_VEC4_BLOCK: usize = 128;

pub fn rms_norm_vec4<B>(
    kb: &mut crate::kernel_builder::KernelBuilder<B>,
    input: crate::kernel_builder::KernelTensorRef<B>,
    residual: Option<crate::kernel_builder::KernelTensorRef<B>>,
    weight: crate::kernel_builder::KernelTensorRef<B>,
    bias: Option<crate::kernel_builder::KernelTensorRef<B>>,
    output: crate::kernel_builder::KernelTensorRef<B>,
    meta: RmsNormVec4Meta,
    rows: u32,
) -> Option<()> {
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

    let input = kb.read::<crate::F32Vec4, 1>(input);
    let residual = residual.map(|r| kb.read::<crate::F32Vec4, 1>(r));
    let weight = kb.read::<crate::F32Vec4, 1>(weight);
    let bias = bias.map(|b| kb.read::<crate::F32Vec4, 1>(b));
    let output = kb.write::<crate::F32Vec4, 1>(output);
    let phase = kb.program();

    phase.program_grid::<RMS_NORM_VEC4_BLOCK>([rows, 1, 1], |program| {
            let row = program.program_id(WorkgroupAxis::X);
            let lane = program.arange();
            let reduce_col = program.loop_index() * RMS_NORM_VEC4_BLOCK as u32 + lane.clone();
            let reduce_mask = reduce_col.lt(meta.cols_vec);
            let input_index = row.clone() * meta.input_row_stride_vec + reduce_col.clone();
            let mut value = program.load_vec4(input.at(input_index), reduce_mask.clone(), 0.0);
            if let Some(residual) = &residual {
                let residual_index =
                    row.clone() * meta.residual_row_stride_vec + reduce_col.clone();
                value = value + program.load_vec4(residual.at(residual_index), reduce_mask, 0.0);
            }
            let dot = program.vec4_dot(value.clone(), value);
            let partial_sum = program.loop_fold(
                TileReduceOp::Sum,
                chunks,
                dot,
                TileLiteral::f32(0.0),
            );
            let total_sum = program.group_reduce_sum::<RMS_NORM_VEC4_BLOCK>(partial_sum);
            let mean = total_sum
                / Tile::<RMS_NORM_VEC4_BLOCK>::literal(TileLiteral::F32(F32Bits::new(
                    meta.cols as f32,
                )));
            let scale = (mean
                + Tile::<RMS_NORM_VEC4_BLOCK>::literal(TileLiteral::f32(eps)))
            .inverse_sqrt();
            let scale = program.bind(scale);

            for chunk in 0..chunks {
                let col = lane.clone() + chunk * RMS_NORM_VEC4_BLOCK as u32;
                let mask = col.lt(meta.cols_vec);
                let input_index = row.clone() * meta.input_row_stride_vec + col.clone();
                let mut value = program.load_vec4(input.at(input_index), mask.clone(), 0.0);
                if let Some(residual) = &residual {
                    let residual_index = row.clone() * meta.residual_row_stride_vec + col.clone();
                    value =
                        value + program.load_vec4(residual.at(residual_index), mask.clone(), 0.0);
                }
                let scale = program.vec4_splat(scale.get());
                let weight = program.load_vec4(weight.at(col.clone()), mask.clone(), 0.0);
                let mut normalized = value * scale * weight;
                if let Some(bias) = &bias {
                    normalized =
                        normalized + program.load_vec4(bias.at(col.clone()), mask.clone(), 0.0);
                }
                let output_index = row.clone() * meta.output_row_stride_vec + col;
                program.store_linear(output.at(output_index), normalized, mask);
            }
        });
    Some(())
}
