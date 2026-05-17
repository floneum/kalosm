use fusor_tile_ir as tile_ir;

use crate::{
    mir::{
        inputs::MirValue, kernel_backend, kernel_backend::DirectKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::{
        ValueTile, apply_unary_function_chain, layout_index, linear_group, output_dims_from_flat,
    },
    nary_wise::NaryScalar,
    reduce::{ReduceOp, ReduceOperation},
    tensor::{DataTypeEnum, TensorData},
};

const BLOCK: usize = 256;

struct ReduceDirectKernelVariant;

pub(crate) fn build_reduce_direct_kernel(
    operation: &ReduceOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    let input = inputs[0].as_tensor()?.clone();
    let output = inputs[1].as_tensor()?.clone();
    let reduce_size = match inputs.get(2)? {
        MirValue::Integer(value) => *value,
        _ => return None,
    };
    let reduce_stride = match inputs.get(3)? {
        MirValue::Integer(value) => *value,
        _ => return None,
    };

    if (input.datatype() == DataTypeEnum::F16 || output.datatype() == DataTypeEnum::F16)
        && !graph.device().f16_supported()
    {
        return None;
    }

    let input_meta = TensorMeta::new(&input)?;
    let output_meta = TensorMeta::new(&output)?;
    if operation.pre_element_wise.input_datatype() != input_meta.datatype {
        return None;
    }
    let reduce_dtype = operation.pre_element_wise.out_datatype();
    if reduce_dtype != operation.function.datatype()
        || operation.post_element_wise.input_datatype() != reduce_dtype
    {
        return None;
    }

    let output_shape = output
        .layout()
        .shape()
        .iter()
        .copied()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let total_outputs = output_shape
        .iter()
        .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;
    let reduce_op = tile_reduce_op(operation.function.op);
    let initial = operation.function.initial_value;
    let dispatch_size = operation.dispatch_size(workgroup_shape, inputs);
    let cache_key = operation.kernel_cache_key_with_dispatch(
        kernel_backend::KernelVariantKey::of::<ReduceDirectKernelVariant>(),
        Some(workgroup_shape),
        dispatch_size,
        inputs,
    );

    let input_buffer = input.buffer().clone();
    let output_buffer = output.buffer().clone();
    let input_layout = flat_layout(input_meta.allocation_len);
    let output_layout = flat_layout(output_meta.allocation_len);
    let input_meta_body = input_meta.clone();
    let output_meta_body = output_meta.clone();
    let pre_chain = operation.pre_element_wise.clone();
    let post_chain = operation.post_element_wise.clone();

    kernel_backend::run_kernel(
        graph.device().kernel_cache(),
        operation.name(),
        cache_key,
        dispatch_size,
        move |kb| {
            let input_tensor =
                tile_ir::KernelTensorRef::new(input_buffer.clone(), input_layout.clone());
            let output_tensor =
                tile_ir::KernelTensorRef::new(output_buffer.clone(), output_layout.clone());
            let input_storage = match input_meta_body.datatype {
                DataTypeEnum::F32 => {
                    crate::nary_direct::Storage2::F32(kb.read::<tile_ir::F32, 2>(input_tensor))
                }
                DataTypeEnum::F16 => {
                    crate::nary_direct::Storage2::F16(kb.read::<tile_ir::F16, 2>(input_tensor))
                }
                DataTypeEnum::U32 => {
                    crate::nary_direct::Storage2::U32(kb.read::<tile_ir::U32, 2>(input_tensor))
                }
            };
            let output_storage = match output_meta_body.datatype {
                DataTypeEnum::F32 => {
                    crate::nary_direct::Storage2::F32(kb.write::<tile_ir::F32, 2>(output_tensor))
                }
                DataTypeEnum::F16 => {
                    crate::nary_direct::Storage2::F16(kb.write::<tile_ir::F16, 2>(output_tensor))
                }
                DataTypeEnum::U32 => {
                    crate::nary_direct::Storage2::U32(kb.write::<tile_ir::U32, 2>(output_tensor))
                }
            };

            kb.program()
                .program_grid::<BLOCK>(dispatch_size, |program| {
                    let lane = program.lane();
                    let group = linear_group(program, dispatch_size);
                    let flat = group * BLOCK as u32 + lane.clone();
                    let in_bounds = flat.lt(total_outputs);
                    let dims = output_dims_from_flat_usize(flat.clone(), &output_shape);
                    let base = layout_index(&input_meta_body.as_nary_meta(), &dims);
                    let value_at =
                        |program: &mut tile_ir::tile::TileBlock<'_>,
                         loop_index: tile_ir::tile::Tile<tile_ir::U32>| {
                            let value_index = base.clone() + loop_index * reduce_stride;
                            let value = input_storage.load(program, value_index, in_bounds.clone());
                            let (value, value_ty) = apply_unary_function_chain(
                                value.into_f32(),
                                input_meta_body.datatype,
                                &pre_chain,
                            )
                            .expect("validated reduce pre_element_wise chain");
                            ValueTile::F32(value)
                                .cast_to(value_ty)
                                .cast_to(reduce_dtype)
                        };

                    let reduced = match reduce_dtype {
                        DataTypeEnum::F32 => ValueTile::F32(program.loop_fold(
                            reduce_op,
                            reduce_size,
                            tile_literal_for(initial, DataTypeEnum::F32),
                            |program, loop_index| value_at(program, loop_index).into_f32(),
                        )),
                        DataTypeEnum::F16 => ValueTile::F16(program.loop_fold(
                            reduce_op,
                            reduce_size,
                            tile_literal_for(initial, DataTypeEnum::F16),
                            |program, loop_index| value_at(program, loop_index).into_f16(),
                        )),
                        DataTypeEnum::U32 => ValueTile::U32(program.loop_fold(
                            reduce_op,
                            reduce_size,
                            tile_literal_for(initial, DataTypeEnum::U32),
                            |program, loop_index| value_at(program, loop_index).into_u32(),
                        )),
                    };

                    let (reduced, reduced_ty) =
                        apply_unary_function_chain(reduced.into_f32(), reduce_dtype, &post_chain)
                            .expect("validated reduce post_element_wise chain");
                    let reduced = ValueTile::F32(reduced)
                        .cast_to(reduced_ty)
                        .cast_to(output_meta_body.datatype);
                    let output_index = layout_index(&output_meta_body.as_nary_meta(), &dims);
                    output_storage.store(program, output_index, reduced, in_bounds);
                });
            Some(())
        },
    )
}

fn tile_literal_for(value: NaryScalar, target: DataTypeEnum) -> tile_ir::TileLiteral {
    match target {
        DataTypeEnum::F32 => match value {
            NaryScalar::F32(value) => tile_ir::TileLiteral::f32(value),
            NaryScalar::F16(value) => tile_ir::TileLiteral::f32(value.to_f32()),
            NaryScalar::U32(value) => tile_ir::TileLiteral::f32(value as f32),
        },
        DataTypeEnum::F16 => match value {
            NaryScalar::F32(value) => {
                tile_ir::TileLiteral::F16(half::f16::from_f32(value).to_bits())
            }
            NaryScalar::F16(value) => tile_ir::TileLiteral::F16(value.to_bits()),
            NaryScalar::U32(value) => {
                tile_ir::TileLiteral::F16(half::f16::from_f32(value as f32).to_bits())
            }
        },
        DataTypeEnum::U32 => match value {
            NaryScalar::F32(value) => tile_ir::TileLiteral::U32(value as u32),
            NaryScalar::F16(value) => tile_ir::TileLiteral::U32(value.to_f32() as u32),
            NaryScalar::U32(value) => tile_ir::TileLiteral::U32(value),
        },
    }
}

fn output_dims_from_flat_usize(
    flat: tile_ir::tile::Tile<tile_ir::U32>,
    shape: &[u32],
) -> Vec<tile_ir::tile::Tile<tile_ir::U32>> {
    let shape = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    output_dims_from_flat(flat, &shape)
}

fn tile_reduce_op(op: ReduceOp) -> tile_ir::TileReduceOp {
    match op {
        ReduceOp::Sum => tile_ir::TileReduceOp::Sum,
        ReduceOp::Product => tile_ir::TileReduceOp::Product,
        ReduceOp::Max => tile_ir::TileReduceOp::Max,
        ReduceOp::Min => tile_ir::TileReduceOp::Min,
    }
}

fn flat_layout(allocation_len: u32) -> tile_ir::Layout {
    tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([1, allocation_len]),
        &[0, 1],
    )
}

#[derive(Clone)]
struct TensorMeta {
    datatype: DataTypeEnum,
    shape: Vec<u32>,
    strides: Vec<u32>,
    offset: u32,
    allocation_len: u32,
}

impl TensorMeta {
    fn new(tensor: &TensorData) -> Option<Self> {
        Some(Self {
            datatype: tensor.datatype(),
            shape: tensor
                .layout()
                .shape()
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<_, _>>()
                .ok()?,
            strides: tensor
                .layout()
                .strides()
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?,
            offset: tensor.layout().offset().try_into().ok()?,
            allocation_len: layout_allocation_len(tensor.layout())?,
        })
    }

    fn as_nary_meta(&self) -> crate::nary_direct::TensorMeta {
        crate::nary_direct::TensorMeta {
            datatype: self.datatype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            allocation_len: self.allocation_len,
        }
    }
}

fn layout_allocation_len(layout: &crate::Layout) -> Option<u32> {
    let max_index = layout
        .shape()
        .iter()
        .zip(layout.strides())
        .try_fold(layout.offset(), |acc, (dim, stride)| {
            acc.checked_add(dim.saturating_sub(1).checked_mul(*stride)?)
        })?;
    max_index.checked_add(1)?.try_into().ok()
}
