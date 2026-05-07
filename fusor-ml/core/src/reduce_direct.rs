use phase_token_prototype as tile_ir;

use crate::{
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        kernel_backend,
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::apply_unary_function_chain,
    nary_wise::NaryScalar,
    reduce::{ReduceOp, ReduceOperation},
    tensor::{DataTypeEnum, TensorData},
};

const BLOCK: usize = 256;

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

    let dispatch_size = operation.dispatch_size(workgroup_shape, inputs);
    let cache_key = format!(
        "{}:tile-program:{:?}:dispatch={dispatch_size:?}:reduce={reduce_size}:stride={reduce_stride}:pre={:?}:post={:?}:{:?}:{:?}:{:?}:{:?}",
        operation.name(),
        workgroup_shape.shape(),
        operation.pre_element_wise,
        operation.post_element_wise,
        input.datatype(),
        input.layout(),
        output.datatype(),
        output.layout()
    );
    kernel_backend::dynamic_kernel_from_ir(
        &graph.device(),
        operation.name(),
        cache_key,
        || {
            build_reduce_tile_ir(
                operation,
                &input,
                &output,
                dispatch_size,
                reduce_size,
                reduce_stride,
            )
        },
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: output.buffer().clone(),
                read_only: false,
            },
        ],
        dispatch_size,
    )
}

fn build_reduce_tile_ir(
    operation: &ReduceOperation,
    input: &TensorData,
    output: &TensorData,
    dispatch_size: [u32; 3],
    reduce_size: u32,
    reduce_stride: u32,
) -> Option<tile_ir::KernelIr> {
    let input_meta = TensorMeta::new(input)?;
    let output_meta = TensorMeta::new(output)?;
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
    let initial = tile_literal(operation.function.initial_value);

    Some(tile_ir::tile::build(move |phase| {
        let input_layout = flat_layout(input_meta.allocation_len);
        let output_layout = flat_layout(output_meta.allocation_len);
        let input_storage =
            phase.storage_read_element_with_layout_offset::<2>(input_meta.element, input_layout, 0);
        let output_storage = phase.storage_write_element_with_layout_offset::<2>(
            output_meta.element,
            output_layout,
            0,
        );

        phase.program_grid::<BLOCK>(dispatch_size, |program| {
            let lane = program.arange();
            let group = linear_group(program, dispatch_size);
            let flat = group * BLOCK as u32 + lane.clone();
            let in_bounds = flat.lt(total_outputs);
            let dims =
                output_dims_from_flat(tile_ir::tile::Tile::from_index(flat.clone()), &output_shape);
            let base = layout_index(&input_meta, &dims);
            let k = tile_ir::tile::Tile::from_index(program.loop_index() * reduce_stride);
            let value_index = base + k;
            let value = program.load_erased(
                input_storage.at(0, value_index),
                in_bounds.clone(),
                zero_literal(input_meta.datatype),
            );
            let (value, value_ty) =
                apply_unary_function_chain(value, input_meta.datatype, &operation.pre_element_wise)
                    .expect("validated reduce pre_element_wise chain");
            let value = cast_tile(value, value_ty, reduce_dtype);
            let reduced = program.loop_fold(reduce_op, reduce_size, value, initial);
            let (reduced, reduced_ty) =
                apply_unary_function_chain(reduced, reduce_dtype, &operation.post_element_wise)
                    .expect("validated reduce post_element_wise chain");
            let reduced = cast_tile(reduced, reduced_ty, output_meta.datatype);
            let output_index = layout_index(&output_meta, &dims);
            program.store_erased(output_storage.at(0, output_index), reduced, in_bounds);
        });
    }))
}

fn output_dims_from_flat<const N: usize>(
    flat: tile_ir::tile::Tile<N>,
    shape: &[u32],
) -> Vec<tile_ir::tile::Tile<N>> {
    (0..shape.len())
        .map(|axis| {
            let divisor = shape[axis + 1..]
                .iter()
                .fold(1u32, |acc, dim| acc.saturating_mul(*dim));
            let quotient = if divisor == 1 {
                flat.clone()
            } else {
                flat.clone() / tile_u32(divisor)
            };
            let dim = shape[axis];
            if dim == 1 {
                tile_u32(0)
            } else {
                quotient % tile_u32(dim)
            }
        })
        .collect()
}

fn layout_index<const N: usize>(
    meta: &TensorMeta,
    coords: &[tile_ir::tile::Tile<N>],
) -> tile_ir::tile::Tile<N> {
    let mut index = tile_u32(meta.offset);
    for (coord, stride) in coords.iter().zip(&meta.strides) {
        if *stride != 0 {
            index = index + coord.clone() * tile_u32(*stride);
        }
    }
    index
}

fn linear_group<const N: usize>(
    program: &tile_ir::tile::TileBlock<'_, N>,
    dispatch_size: [u32; 3],
) -> tile_ir::tile::ScalarIndex {
    program.program_id(tile_ir::WorkgroupAxis::X)
        + program.program_id(tile_ir::WorkgroupAxis::Y) * dispatch_size[0]
        + program.program_id(tile_ir::WorkgroupAxis::Z)
            * dispatch_size[0].saturating_mul(dispatch_size[1])
}

fn flat_layout(allocation_len: u32) -> tile_ir::Layout {
    tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([1, allocation_len]),
        tile_ir::Strides::new([0, 1]),
    )
}

fn cast_tile<const N: usize>(
    value: tile_ir::tile::Tile<N>,
    source: DataTypeEnum,
    target: DataTypeEnum,
) -> tile_ir::tile::Tile<N> {
    if source == target {
        value
    } else {
        value.cast(tile_element(target))
    }
}

fn tile_reduce_op(op: ReduceOp) -> tile_ir::TileReduceOp {
    match op {
        ReduceOp::Sum => tile_ir::TileReduceOp::Sum,
        ReduceOp::Product => tile_ir::TileReduceOp::Product,
        ReduceOp::Max => tile_ir::TileReduceOp::Max,
        ReduceOp::Min => tile_ir::TileReduceOp::Min,
    }
}

fn tile_literal(value: NaryScalar) -> tile_ir::TileLiteral {
    match value {
        NaryScalar::F32(value) => tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(value)),
        NaryScalar::F16(value) => tile_ir::TileLiteral::F16(value.to_bits()),
        NaryScalar::U32(value) => tile_ir::TileLiteral::U32(value),
    }
}

fn tile_u32<const N: usize>(value: u32) -> tile_ir::tile::Tile<N> {
    tile_ir::tile::Tile::literal(tile_ir::TileLiteral::U32(value))
}

fn zero_literal(value: DataTypeEnum) -> tile_ir::TileLiteral {
    match value {
        DataTypeEnum::F32 => tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(0.0)),
        DataTypeEnum::F16 => tile_ir::TileLiteral::F16(half::f16::from_f32(0.0).to_bits()),
        DataTypeEnum::U32 => tile_ir::TileLiteral::U32(0),
    }
}

fn tile_element(value: DataTypeEnum) -> tile_ir::ElementType {
    match value {
        DataTypeEnum::F32 => tile_ir::ElementType::F32,
        DataTypeEnum::F16 => tile_ir::ElementType::F16,
        DataTypeEnum::U32 => tile_ir::ElementType::U32,
    }
}

#[derive(Clone)]
struct TensorMeta {
    datatype: DataTypeEnum,
    element: tile_ir::ElementType,
    strides: Vec<u32>,
    offset: u32,
    allocation_len: u32,
}

impl TensorMeta {
    fn new(tensor: &TensorData) -> Option<Self> {
        Some(Self {
            datatype: tensor.datatype(),
            element: tile_element(tensor.datatype()),
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
