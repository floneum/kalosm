use std::{
    any::{Any, TypeId},
    hash::Hash,
    sync::OnceLock,
};

use crate::{
    DataTypeEnum, Layout,
    compute_graph::{ComputeGraphInner, GraphOperation, NodeIndex},
    kernel_selection::{
        Axis, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, multiple_of,
    },
    mir::{
        direct_kernel::DirectKernel,
        inputs::MirValue,
        kernel_backend,
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_direct::apply_unary_function_chain,
    nary_wise::UnaryFunctionChain,
    tensor::{TensorData, TensorLayoutInfo},
};
use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;

const BLOCK: usize = 1024;
const RMS_NORM_MODULE_CACHE_SIZE: usize = 128;

fn rms_norm_module_cache() -> &'static kernel_backend::ModuleCache {
    static CACHE: OnceLock<kernel_backend::ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| kernel_backend::module_cache(RMS_NORM_MODULE_CACHE_SIZE))
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RmsNormKernelVariant {
    Tile,
    Vec4,
}

struct RmsNormDirectKernelVariant;

const RMS_COLS: Axis<1> = Axis;

#[derive(Clone, Copy, Debug)]
struct RmsNormSelectionCtx {
    vec4_supported: bool,
}

fn rms_norm_selector() -> ShapeSelector<2, RmsNormSelectionCtx, RmsNormKernelVariant> {
    ShapeSelector::new()
        .rule(
            RmsNormKernelVariant::Vec4,
            ShapeRule::new()
                .axis(RMS_COLS, multiple_of(4))
                .when_ctx(|ctx: &RmsNormSelectionCtx| ctx.vec4_supported),
        )
        .rule(RmsNormKernelVariant::Tile, ShapeRule::new())
}

fn select_rms_norm_variant(
    rows: u32,
    cols: u32,
    ctx: &RmsNormSelectionCtx,
    caps: KernelDeviceCaps,
) -> RmsNormKernelVariant {
    rms_norm_selector()
        .select(KernelShape::new([rows as usize, cols as usize]), ctx, caps)
        .expect("rms norm selector has a catch-all rule")
}

#[derive(Clone, Debug)]
pub(crate) struct RmsNormOperation {
    pub(crate) input: NodeIndex,
    pub(crate) residual: Option<NodeIndex>,
    pub(crate) weight: NodeIndex,
    pub(crate) bias: Option<NodeIndex>,
    shape: Box<[usize]>,
    eps: f32,
    /// Unary chain applied to each normalized output element in-register
    /// before the store. Populated by `try_fuse_into_rmsnorm` when a
    /// downstream `Nary` matches the element-wise fusion pattern.
    pub(crate) post_element_wise: UnaryFunctionChain,
}

impl RmsNormOperation {
    pub(crate) fn new(
        input: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        shape: &[usize],
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual: None,
            weight,
            bias,
            shape: shape.into(),
            eps,
            post_element_wise: UnaryFunctionChain::empty(DataTypeEnum::F32),
        }
    }

    pub(crate) fn new_with_residual(
        input: NodeIndex,
        residual: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        shape: &[usize],
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual: Some(residual),
            weight,
            bias,
            shape: shape.into(),
            eps,
            post_element_wise: UnaryFunctionChain::empty(DataTypeEnum::F32),
        }
    }

    fn rows_cols(&self) -> Option<(u32, u32)> {
        let cols = *self.shape.last()?;
        let rows = self.shape[..self.shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        Some((rows.try_into().ok()?, cols.try_into().ok()?))
    }
}

impl Operation for RmsNormOperation {
    fn hash_kernel_signature(&self, state: &mut FxHasher) {
        TypeId::of::<Self>().hash(state);
        self.residual.is_some().hash(state);
        self.bias.is_some().hash(state);
        self.shape.hash(state);
        self.eps.to_bits().hash(state);
        self.post_element_wise.hash(state);
    }

    fn workgroup_shape_constraints(&self, _device: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(1));
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        let (rows, _) = self
            .rows_cols()
            .expect("rms norm requires a non-empty shape");
        [rows, 1, 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        if let Some(residual) = self.residual {
            f(residual);
        }
        f(self.weight);
        if let Some(bias) = self.bias {
            f(bias);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap();
        let residual = self
            .residual
            .map(|residual| nodes.get_cached_result(residual).unwrap());
        let weight = nodes.get_cached_result(self.weight).unwrap();
        let output =
            TensorData::new_for_shape(input.device(), input.layout().shape(), input.datatype());

        let mut inputs = vec![input.clone().into()];
        if let Some(residual) = residual {
            inputs.push(residual.clone().into());
        }
        inputs.push(weight.clone().into());
        if let Some(bias) = self.bias {
            inputs.push(nodes.get_cached_result(bias).unwrap().clone().into());
        }
        inputs.push(output.into());
        inputs
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let input = inputs.first()?.as_tensor()?;
        let (residual, weight_index) = if self.residual.is_some() {
            (Some(inputs.get(1)?.as_tensor()?), 2)
        } else {
            (None, 1)
        };
        let weight = inputs.get(weight_index)?.as_tensor()?;
        let (bias, output_index) = if self.bias.is_some() {
            (
                Some(inputs.get(weight_index + 1)?.as_tensor()?),
                weight_index + 2,
            )
        } else {
            (None, weight_index + 1)
        };
        let output = inputs.get(output_index)?.as_tensor()?;

        if input.datatype() != DataTypeEnum::F32
            || residual.is_some_and(|residual| residual.datatype() != DataTypeEnum::F32)
            || weight.datatype() != DataTypeEnum::F32
            || output.datatype() != DataTypeEnum::F32
            || bias.is_some_and(|bias| bias.datatype() != DataTypeEnum::F32)
        {
            return None;
        }

        let input_view = flatten_matrix_layout(input.layout())?;
        let residual_view = match residual {
            Some(residual) => Some(flatten_matrix_layout(residual.layout())?),
            None => None,
        };
        let output_view = flatten_matrix_layout(output.layout())?;
        let rows = input_view.rows;
        let cols = input_view.cols;
        if rows != output_view.rows || cols != output_view.cols {
            return None;
        }
        if let Some(residual_view) = residual_view.as_ref()
            && (rows != residual_view.rows || cols != residual_view.cols)
        {
            return None;
        }
        if weight.layout().shape() != [cols as usize] {
            return None;
        }
        if let Some(bias) = bias
            && bias.layout().shape() != [cols as usize]
        {
            return None;
        }

        // The vec4 path is a pre-built kernel that doesn't accept an arbitrary
        // post-element-wise chain; force the tile path when fusion has
        // attached one. (The tile path applies the chain inline below.)
        let post_chain_nonempty = !self.post_element_wise.functions.is_empty();
        let vec4_meta = graph
            .device()
            .subgroups_supported()
            .then(|| {
                if post_chain_nonempty {
                    return None;
                }
                build_vec4_rms_norm_meta(
                    input_view.clone(),
                    residual_view.clone(),
                    weight,
                    bias,
                    output_view.clone(),
                    self.eps,
                )
            })
            .flatten();
        let selection_ctx = RmsNormSelectionCtx {
            vec4_supported: vec4_meta.is_some(),
        };
        let variant = select_rms_norm_variant(
            rows,
            cols,
            &selection_ctx,
            KernelDeviceCaps::from_device(&graph.device()),
        );
        let dispatch_size = [rows, 1, 1];
        let kernel_label = match variant {
            RmsNormKernelVariant::Tile => "rms_norm",
            RmsNormKernelVariant::Vec4 => "rms_norm_vec4",
        };
        let cache_variant =
            kernel_backend::KernelVariantKey::with_payload::<RmsNormDirectKernelVariant>(|state| {
                variant.hash(state);
            });
        let module_key = self.kernel_module_key_with_dispatch(
            cache_variant,
            Some(workgroup_shape),
            dispatch_size,
            inputs,
        );

        if let Some(meta) = vec4_meta {
            // Collect buffers in the SAME order as the IR builder declares
            // them (input, residual?, weight, bias?, output), so the closure
            // can be deferred to cache-miss only.
            let mut buffers = Vec::with_capacity(5);
            buffers.push(input.buffer().clone());
            if let Some(residual) = residual {
                buffers.push(residual.buffer().clone());
            }
            buffers.push(weight.buffer().clone());
            if let Some(bias) = bias {
                buffers.push(bias.buffer().clone());
            }
            buffers.push(output.buffer().clone());

            let has_residual = residual.is_some();
            let has_bias = bias.is_some();
            kernel_backend::dynamic_kernel_from_hashed_ir(
                &graph.device(),
                rms_norm_module_cache(),
                kernel_label,
                module_key,
                buffers,
                dispatch_size,
                move || {
                    let vec_layout = tile_ir::Layout::strided(
                        tile_ir::MemoryLevel::Storage,
                        tile_ir::Shape::new([1]),
                        &[1],
                    );
                    let mut kb = tile_ir::KernelBuilder::<()>::new();
                    let input_ref = tile_ir::KernelTensorRef::with_offset(
                        (),
                        vec_layout.clone(),
                        meta.input_offset_vec,
                    );
                    let residual_ref = if has_residual {
                        meta.residual_offset_vec.map(|offset| {
                            tile_ir::KernelTensorRef::with_offset((), vec_layout.clone(), offset)
                        })
                    } else {
                        None
                    };
                    let weight_ref = tile_ir::KernelTensorRef::with_offset(
                        (),
                        vec_layout.clone(),
                        meta.weight_offset_vec,
                    );
                    let bias_ref = if has_bias {
                        meta.bias_offset_vec.map(|offset| {
                            tile_ir::KernelTensorRef::with_offset((), vec_layout.clone(), offset)
                        })
                    } else {
                        None
                    };
                    let output_ref = tile_ir::KernelTensorRef::with_offset(
                        (),
                        vec_layout,
                        meta.output_offset_vec,
                    );
                    tile_ir_kernels::rms_norm_vec4(
                        &mut kb,
                        tile_ir_kernels::RmsNormVec4 {
                            input: input_ref,
                            residual: residual_ref,
                            weight: weight_ref,
                            bias: bias_ref,
                            output: output_ref,
                            meta,
                            rows,
                        },
                    )?;
                    Some(kb.finish().0)
                },
            )
        } else {
            let mut buffers = Vec::with_capacity(5);
            buffers.push(input.buffer().clone());
            if let Some(residual) = residual {
                buffers.push(residual.buffer().clone());
            }
            buffers.push(weight.buffer().clone());
            if let Some(bias) = bias {
                buffers.push(bias.buffer().clone());
            }
            buffers.push(output.buffer().clone());

            let post_chain = self.post_element_wise.clone();
            kernel_backend::dynamic_kernel_from_hashed_ir(
                &graph.device(),
                rms_norm_module_cache(),
                kernel_label,
                module_key,
                buffers,
                dispatch_size,
                || {
                    build_rms_norm_tile_ir(
                        input_view,
                        residual_view,
                        weight,
                        bias,
                        output_view,
                        self.eps,
                        post_chain,
                    )
                },
            )
        }
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs.last().unwrap().clone()
    }

    fn name(&self) -> String {
        let op = if self.residual.is_some() {
            "rms_norm_residual"
        } else {
            "rms_norm"
        };
        format!(
            "{op}_f32_{}",
            self.shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl GraphOperation for RmsNormOperation {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn category(&self) -> &'static str {
        "rms_norm"
    }

    fn output_layout(
        &self,
        input_layouts: &FxHashMap<NodeIndex, TensorLayoutInfo>,
    ) -> Option<TensorLayoutInfo> {
        let input_layout = input_layouts.get(&self.input)?;
        Some(TensorLayoutInfo::new(
            Layout::contiguous(input_layout.shape()),
            input_layout.datatype(),
        ))
    }
}

fn build_vec4_rms_norm_meta(
    input_view: crate::mir::tile_direct::DirectMatrixLayout,
    residual_view: Option<crate::mir::tile_direct::DirectMatrixLayout>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output_view: crate::mir::tile_direct::DirectMatrixLayout,
    eps: f32,
) -> Option<tile_ir_kernels::RmsNormVec4Meta> {
    if !input_view.layout.is_affine()
        || !output_view.layout.is_affine()
        || residual_view
            .as_ref()
            .is_some_and(|residual| !residual.layout.is_affine())
        || !input_view.cols.is_multiple_of(4)
    {
        return None;
    }

    let [input_row_stride, input_col_stride] = matrix_strides(input_view.layout.affine_strides())?;
    let [output_row_stride, output_col_stride] =
        matrix_strides(output_view.layout.affine_strides())?;
    if input_col_stride != 1
        || output_col_stride != 1
        || !input_view.offset.is_multiple_of(4)
        || !output_view.offset.is_multiple_of(4)
        || !input_row_stride.is_multiple_of(4)
        || !output_row_stride.is_multiple_of(4)
    {
        return None;
    }

    let (residual_offset_vec, residual_row_stride_vec) = if let Some(residual_view) = residual_view
    {
        let [residual_row_stride, residual_col_stride] =
            matrix_strides(residual_view.layout.affine_strides())?;
        if residual_col_stride != 1
            || !residual_view.offset.is_multiple_of(4)
            || !residual_row_stride.is_multiple_of(4)
        {
            return None;
        }
        (Some(residual_view.offset / 4), residual_row_stride / 4)
    } else {
        (None, 0)
    };

    let weight_stride = *weight.layout().strides().first()?;
    if weight.layout().shape() != [input_view.cols as usize]
        || weight_stride != 1
        || !weight.layout().offset().is_multiple_of(4)
    {
        return None;
    }
    let bias_offset_vec = if let Some(bias) = bias {
        let bias_stride = *bias.layout().strides().first()?;
        if bias.layout().shape() != [input_view.cols as usize]
            || bias_stride != 1
            || !bias.layout().offset().is_multiple_of(4)
        {
            return None;
        }
        Some((bias.layout().offset() / 4).try_into().ok()?)
    } else {
        None
    };

    Some(tile_ir_kernels::RmsNormVec4Meta {
        cols: input_view.cols,
        cols_vec: input_view.cols / 4,
        eps: tile_ir::F32Bits::new(eps),
        input_offset_vec: input_view.offset / 4,
        input_row_stride_vec: input_row_stride / 4,
        residual_offset_vec,
        residual_row_stride_vec,
        weight_offset_vec: (weight.layout().offset() / 4).try_into().ok()?,
        bias_offset_vec,
        output_offset_vec: output_view.offset / 4,
        output_row_stride_vec: output_row_stride / 4,
    })
}

fn matrix_strides(strides: Vec<u32>) -> Option<[u32; 2]> {
    strides.try_into().ok()
}

fn build_rms_norm_tile_ir(
    input_view: crate::mir::tile_direct::DirectMatrixLayout,
    residual_view: Option<crate::mir::tile_direct::DirectMatrixLayout>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output_view: crate::mir::tile_direct::DirectMatrixLayout,
    eps: f32,
    post_chain: UnaryFunctionChain,
) -> Option<tile_ir::KernelIr> {
    let rows = input_view.rows;
    let cols = input_view.cols;
    let input_storage_layout = input_view.layout.clone();
    let residual_storage_layout = residual_view
        .as_ref()
        .map(|residual_view| residual_view.layout.clone());
    let residual_offset = residual_view.as_ref().map(|residual| residual.offset);
    let output_storage_layout = output_view.layout.clone();
    let weight_layout = vector_as_row_layout(weight.layout())?;
    let bias_layout = match bias {
        Some(bias) => Some(vector_as_row_layout(bias.layout())?),
        None => None,
    };
    let weight_offset = weight.layout().offset().try_into().ok()?;
    let bias_offset = match bias {
        Some(bias) => Some(bias.layout().offset().try_into().ok()?),
        None => None,
    };

    Some(tile_ir::tile::build(move |phase| {
        let input = tile_storage_read_with_direct_layout(
            phase,
            crate::mir::tile_direct::DirectMatrixLayout {
                rows,
                cols,
                offset: input_view.offset,
                layout: input_storage_layout,
            },
        );
        let residual = residual_storage_layout.map(|layout| {
            tile_storage_read_with_direct_layout(
                phase,
                crate::mir::tile_direct::DirectMatrixLayout {
                    rows,
                    cols,
                    offset: residual_offset.expect("residual offset exists with layout"),
                    layout,
                },
            )
        });
        let weight =
            phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(weight_layout, weight_offset);
        let bias = bias_layout.map(|layout| {
            phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(
                layout,
                bias_offset.expect("bias offset exists when bias layout exists"),
            )
        });
        let output = tile_storage_write_with_direct_layout(
            phase,
            crate::mir::tile_direct::DirectMatrixLayout {
                rows,
                cols,
                offset: output_view.offset,
                layout: output_storage_layout,
            },
        );

        let chunks = cols.div_ceil(BLOCK as u32);
        phase.program_grid::<BLOCK>([rows, 1, 1], |program| {
            let row = program.program_id(tile_ir::WorkgroupAxis::X);
            let lane = program.lane();
            let sum_square = program.loop_reduce_sum(chunks, |program, loop_index| {
                let reduce_col = loop_index * BLOCK as u32 + lane.clone();
                let reduce_mask = reduce_col.clone().lt(cols);
                let mut value =
                    program.load(input.at((&row, &reduce_col)), reduce_mask.clone(), 0.0);
                if let Some(residual) = &residual {
                    value =
                        value + program.load(residual.at((&row, &reduce_col)), reduce_mask, 0.0);
                }
                value.clone() * value
            });
            let rms = (sum_square / tile_ir::tile::Tile::literal(cols as f32)
                + tile_ir::tile::Tile::literal(eps))
            .unary(tile_ir::TileUnaryOp::Sqrt);
            for chunk in 0..chunks {
                let col = lane.clone() + chunk * BLOCK as u32;
                let mask = col.lt(cols);
                let mut value = program.load(input.at((&row, &col)), mask.clone(), 0.0);
                if let Some(residual) = &residual {
                    value = value + program.load(residual.at((&row, &col)), mask.clone(), 0.0);
                }
                let weight = program.load(weight.at((0, &col)), mask.clone(), 0.0);
                let mut normalized = value / rms.clone() * weight;
                if let Some(bias) = &bias {
                    let bias_value = program.load(bias.at((0, &col)), mask.clone(), 0.0);
                    normalized = normalized + bias_value;
                }
                // Apply any fused post-element-wise chain in-register before
                // the store. Empty chain is a no-op (the loop below short-
                // circuits and returns the value unchanged).
                if !post_chain.functions.is_empty() {
                    let (val, _) =
                        apply_unary_function_chain(normalized, DataTypeEnum::F32, &post_chain)
                            .expect("rms_norm post-chain validated at fuse time");
                    normalized = val;
                }
                program.store(output.at((&row, col)), normalized, mask);
            }
        });
    }))
}

fn vector_as_row_layout(layout: &crate::Layout) -> Option<tile_ir::Layout> {
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() != 1 {
        return None;
    }
    Some(tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([1, (*shape.first()?).try_into().ok()?]),
        &[0, (*strides.first()?).try_into().ok()?],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor, kernel_selection::assert_selector_generates};

    fn caps() -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: true,
            cooperative_matrix_supported: false,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_storage_size: 64 * 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroups_per_dimension: 65_535,
        }
    }

    #[test]
    fn rms_norm_selector_generates_each_variant() {
        let selector = rms_norm_selector();
        let cases = [
            (
                RmsNormKernelVariant::Vec4,
                RmsNormSelectionCtx {
                    vec4_supported: true,
                },
            ),
            (
                RmsNormKernelVariant::Tile,
                RmsNormSelectionCtx {
                    vec4_supported: false,
                },
            ),
        ];
        assert_selector_generates(
            &selector,
            cases.map(|(variant, ctx)| (variant, ctx, caps())),
        );
    }

    #[tokio::test]
    async fn rms_norm_direct_matches_reference() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let input = Tensor::new(&device, &vec![vec![1.0f32, 2.0, 3.0, 4.0]]);
        let weight = Tensor::new(&device, &vec![0.5f32, 1.0, 1.5, 2.0]);
        let output = input.try_rms_norm_direct(&weight, None, 1e-5).unwrap();
        let output = output.as_slice().await.unwrap();

        let mean_square = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let rms = f32::sqrt(mean_square + 1e-5);
        let expected = [1.0 / rms * 0.5, 2.0 / rms, 3.0 / rms * 1.5, 4.0 / rms * 2.0];

        for (i, expected) in expected.into_iter().enumerate() {
            let actual = output[[0, i]];
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }
}
