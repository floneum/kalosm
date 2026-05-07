use std::{
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    sync::OnceLock,
};

use crate::{
    DataTypeEnum,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        kernel_backend::{self, CompiledKernelModule},
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::TensorData,
};
use lru::LruCache;
use parking_lot::RwLock;
use phase_token_prototype as tile_ir;
use rustc_hash::{FxBuildHasher, FxHasher};

const BLOCK: usize = 1024;
const VEC4_BLOCK: u32 = 128;
const VEC4_SUBGROUP_WIDTH: u32 = 32;
const RMS_NORM_MODULE_CACHE_SIZE: usize = 128;

fn rms_norm_module_cache()
-> &'static RwLock<LruCache<[u64; 2], CompiledKernelModule, FxBuildHasher>> {
    static CACHE: OnceLock<RwLock<LruCache<[u64; 2], CompiledKernelModule, FxBuildHasher>>> =
        OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(RMS_NORM_MODULE_CACHE_SIZE).unwrap(),
            Default::default(),
        ))
    })
}

fn hash_layout<H: Hasher>(state: &mut H, layout: &crate::Layout) {
    layout.offset().hash(state);
    layout.shape().hash(state);
    layout.strides().hash(state);
}

fn rms_norm_module_key(
    variant: RmsNormKernelVariant,
    rows: u32,
    cols: u32,
    eps_bits: u32,
    has_bias: bool,
    has_residual: bool,
    dispatch_size: [u32; 3],
    input: &TensorData,
    residual: Option<&TensorData>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output: &TensorData,
) -> [u64; 2] {
    std::array::from_fn(|salt| {
        let mut hasher = FxHasher::default();
        (salt as u64).hash(&mut hasher);
        variant.hash(&mut hasher);
        rows.hash(&mut hasher);
        cols.hash(&mut hasher);
        eps_bits.hash(&mut hasher);
        has_bias.hash(&mut hasher);
        has_residual.hash(&mut hasher);
        dispatch_size.hash(&mut hasher);
        hash_layout(&mut hasher, input.layout());
        residual
            .map(|residual| hash_layout(&mut hasher, residual.layout()))
            .hash(&mut hasher);
        hash_layout(&mut hasher, weight.layout());
        bias.map(|bias| hash_layout(&mut hasher, bias.layout()))
            .hash(&mut hasher);
        hash_layout(&mut hasher, output.layout());
        hasher.finish()
    })
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RmsNormKernelVariant {
    Tile,
    Vec4,
}

#[derive(Clone, Debug)]
pub(crate) struct RmsNormOperation {
    pub(crate) input: NodeIndex,
    pub(crate) residual: Option<NodeIndex>,
    pub(crate) weight: NodeIndex,
    pub(crate) bias: Option<NodeIndex>,
    shape: Box<[usize]>,
    eps: f32,
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
        _workgroup_shape: &WorkgroupShape,
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

        let has_bias = bias.is_some();
        let has_residual = residual.is_some();
        let vec4_meta = graph
            .device()
            .subgroups_supported()
            .then(|| {
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
        let variant = if vec4_meta.is_some() {
            RmsNormKernelVariant::Vec4
        } else {
            RmsNormKernelVariant::Tile
        };
        let dispatch_size = [rows, 1, 1];
        let module_key = rms_norm_module_key(
            variant,
            rows,
            cols,
            self.eps.to_bits(),
            has_bias,
            has_residual,
            dispatch_size,
            input,
            residual,
            weight,
            bias,
            output,
        );
        let kernel_label = match variant {
            RmsNormKernelVariant::Tile => "rms_norm",
            RmsNormKernelVariant::Vec4 => "rms_norm_vec4",
        };
        let cache_key = format!(
            "{kernel_label}:{:016x}{:016x}",
            module_key[0], module_key[1]
        );
        let module = if let Some(module) = rms_norm_module_cache().write().get(&module_key) {
            module.clone()
        } else {
            let verbose_cache_key = format!(
                "{}:tile-program:rows={rows}:cols={cols}:eps={:?}:bias={has_bias}:residual={has_residual}:dispatch={dispatch_size:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
                self.name(),
                self.eps.to_bits(),
                input.layout(),
                residual.map(|residual| residual.layout()),
                weight.layout(),
                bias.map(|bias| bias.layout()),
                output.layout()
            );
            let module = if let Some(meta) = vec4_meta {
                kernel_backend::cached_backend_naga_module(
                    &graph.device(),
                    verbose_cache_key,
                    || build_rms_norm_vec4_naga_module(meta),
                )?
            } else {
                kernel_backend::cached_kernel_ir(&graph.device(), verbose_cache_key, || {
                    build_rms_norm_tile_ir(
                        input_view,
                        residual_view,
                        weight,
                        bias,
                        output_view,
                        self.eps,
                    )
                })?
            };
            rms_norm_module_cache()
                .write()
                .get_or_insert(module_key, || module.clone())
                .clone()
        };

        let mut bindings = vec![DirectKernelBinding::Storage {
            binding: 0,
            buffer: input.buffer().clone(),
            read_only: true,
        }];
        let mut binding = 1;
        if let Some(residual) = residual {
            bindings.push(DirectKernelBinding::Storage {
                binding,
                buffer: residual.buffer().clone(),
                read_only: true,
            });
            binding += 1;
        }
        bindings.push(DirectKernelBinding::Storage {
            binding,
            buffer: weight.buffer().clone(),
            read_only: true,
        });
        binding += 1;
        if let Some(bias) = bias {
            bindings.push(DirectKernelBinding::Storage {
                binding,
                buffer: bias.buffer().clone(),
                read_only: true,
            });
            binding += 1;
        }
        bindings.push(DirectKernelBinding::Storage {
            binding,
            buffer: output.buffer().clone(),
            read_only: false,
        });

        Some(kernel_backend::dynamic_kernel_from_module(
            kernel_label,
            cache_key,
            module,
            bindings,
            dispatch_size,
        ))
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

#[derive(Clone, Copy)]
pub(crate) struct RmsNormVec4Meta {
    cols: u32,
    cols_vec: u32,
    eps: f32,
    input_offset_vec: u32,
    input_row_stride_vec: u32,
    residual_offset_vec: Option<u32>,
    residual_row_stride_vec: u32,
    weight_offset_vec: u32,
    bias_offset_vec: Option<u32>,
    output_offset_vec: u32,
    output_row_stride_vec: u32,
}

fn build_vec4_rms_norm_meta(
    input_view: crate::mir::tile_direct::DirectMatrixLayout,
    residual_view: Option<crate::mir::tile_direct::DirectMatrixLayout>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output_view: crate::mir::tile_direct::DirectMatrixLayout,
    eps: f32,
) -> Option<RmsNormVec4Meta> {
    if input_view.index_map.is_some()
        || output_view.index_map.is_some()
        || residual_view
            .as_ref()
            .is_some_and(|residual| residual.index_map.is_some())
        || !input_view.cols.is_multiple_of(4)
    {
        return None;
    }

    let [input_row_stride, input_col_stride] = matrix_strides(input_view.layout.strides())?;
    let [output_row_stride, output_col_stride] = matrix_strides(output_view.layout.strides())?;
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
            matrix_strides(residual_view.layout.strides())?;
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

    Some(RmsNormVec4Meta {
        cols: input_view.cols,
        cols_vec: input_view.cols / 4,
        eps,
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

fn matrix_strides(strides: &tile_ir::Strides) -> Option<[u32; 2]> {
    strides.values().try_into().ok()
}

#[path = "rms_norm_vec4.rs"]
mod rms_norm_vec4;

use rms_norm_vec4::build_rms_norm_vec4_naga_module;

fn build_rms_norm_tile_ir(
    input_view: crate::mir::tile_direct::DirectMatrixLayout,
    residual_view: Option<crate::mir::tile_direct::DirectMatrixLayout>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output_view: crate::mir::tile_direct::DirectMatrixLayout,
    eps: f32,
) -> Option<tile_ir::KernelIr> {
    let rows = input_view.rows;
    let cols = input_view.cols;
    let input_storage_layout = input_view.layout.clone();
    let residual_storage_layout = residual_view
        .as_ref()
        .map(|residual_view| residual_view.layout.clone());
    let residual_offset = residual_view.as_ref().map(|residual| residual.offset);
    let residual_index_map = residual_view
        .as_ref()
        .and_then(|residual_view| residual_view.index_map.clone());
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
                index_map: input_view.index_map,
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
                    index_map: residual_index_map,
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
                index_map: output_view.index_map,
            },
        );

        let chunks = cols.div_ceil(BLOCK as u32);
        phase.program_grid::<BLOCK>([rows, 1, 1], |program| {
            let row = program.program_id(tile_ir::WorkgroupAxis::X);
            let lane = program.arange();
            let reduce_col = program.loop_index() * BLOCK as u32 + lane.clone();
            let reduce_mask = reduce_col.lt(cols);
            let mut value = program.load(input.at(&row, &reduce_col), reduce_mask.clone(), 0.0);
            if let Some(residual) = &residual {
                value = value + program.load(residual.at(&row, &reduce_col), reduce_mask, 0.0);
            }
            let sum_square = program.loop_reduce_sum(chunks, value.clone() * value);
            let rms = (tile_ir::tile::Tile::<BLOCK>::from(sum_square)
                / tile_ir::tile::Scalar::literal(cols as f32)
                + tile_ir::tile::Scalar::literal(eps))
            .unary(tile_ir::TileUnaryOp::Sqrt);
            for chunk in 0..chunks {
                let col = lane.clone() + chunk * BLOCK as u32;
                let mask = col.lt(cols);
                let mut value = program.load(input.at(&row, &col), mask.clone(), 0.0);
                if let Some(residual) = &residual {
                    value = value + program.load(residual.at(&row, &col), mask.clone(), 0.0);
                }
                let weight = program.load(weight.at(0, &col), mask.clone(), 0.0);
                let mut normalized = value / rms.clone() * weight;
                if let Some(bias) = &bias {
                    let bias_value = program.load(bias.at(0, &col), mask.clone(), 0.0);
                    normalized = normalized + bias_value;
                }
                program.store(output.at(&row, col), normalized, mask);
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
        tile_ir::Strides::new([0, (*strides.first()?).try_into().ok()?]),
    ))
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

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
