use fusor_tile_ir as tile_ir;

use crate::{
    matmul::MatMulOperation,
    mir::{
        direct_kernel::DirectKernel, inputs::MirValue, kernel_backend, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::apply_unary_function_chain,
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::distribute_workgroups,
};

const BLOCK: usize = 256;

pub(crate) fn build_serial_matmul_direct_kernel(
    operation: &MatMulOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    _workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    let [input_a, input_b, output] = inputs else {
        return None;
    };
    let input_a = input_a.as_tensor()?.clone();
    let input_b = input_b.as_tensor()?.clone();
    let output = output.as_tensor()?.clone();
    if input_a.layout().rank() != input_b.layout().rank()
        || input_a.layout().rank() != output.layout().rank()
        || input_a.layout().rank() < 2
        || input_a.datatype() != input_b.datatype()
    {
        return None;
    }
    if (input_a.datatype() == DataTypeEnum::F16
        || input_b.datatype() == DataTypeEnum::F16
        || output.datatype() == DataTypeEnum::F16)
        && !graph.device().f16_supported()
    {
        return None;
    }

    let total_outputs = output
        .layout()
        .shape()
        .iter()
        .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
    let dispatch_size = distribute_workgroups(total_outputs.div_ceil(BLOCK as u32));
    let cache_key = operation.kernel_cache_key_with_dispatch(
        "matmul_serial_direct",
        Some(_workgroup_shape),
        dispatch_size,
        inputs,
    );
    let a_meta = TensorMeta::new(&input_a)?;
    let b_meta = TensorMeta::new(&input_b)?;
    let y_meta = TensorMeta::new(&output)?;
    if operation.pre_element_wise[0].input_datatype() != a_meta.datatype
        || operation.pre_element_wise[1].input_datatype() != b_meta.datatype
    {
        return None;
    }
    let a_product_dtype = operation.pre_element_wise[0].out_datatype();
    let b_product_dtype = operation.pre_element_wise[1].out_datatype();
    if a_product_dtype != b_product_dtype
        || operation.post_element_wise.input_datatype() != a_product_dtype
    {
        return None;
    }
    let out_shape = output
        .layout()
        .shape()
        .iter()
        .copied()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let rank = out_shape.len();
    let total_outputs = out_shape
        .iter()
        .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;
    let k: u32 = input_a
        .layout()
        .shape()
        .last()
        .copied()
        .and_then(|value| value.try_into().ok())?;
    let acc_dtype = match a_product_dtype {
        DataTypeEnum::U32 => DataTypeEnum::U32,
        DataTypeEnum::F32 | DataTypeEnum::F16 => DataTypeEnum::F32,
    };
    let result_dtype = a_product_dtype;

    let a_buffer = input_a.buffer().clone();
    let b_buffer = input_b.buffer().clone();
    let y_buffer = output.buffer().clone();
    let a_layout = flat_layout(a_meta.allocation_len);
    let b_layout = flat_layout(b_meta.allocation_len);
    let y_layout = flat_layout(y_meta.allocation_len);
    let a_meta_body = a_meta.clone();
    let b_meta_body = b_meta.clone();
    let y_meta_body = y_meta.clone();

    kernel_backend::run_kernel(
        &graph.device(),
        operation.name(),
        cache_key,
        dispatch_size,
        move |kb| {
            let a_storage = kb.read_element::<2>(
                a_meta_body.element,
                tile_ir::KernelTensorRef::new(a_buffer, a_layout),
            );
            let b_storage = kb.read_element::<2>(
                b_meta_body.element,
                tile_ir::KernelTensorRef::new(b_buffer, b_layout),
            );
            let y_storage = kb.write_element::<2>(
                y_meta_body.element,
                tile_ir::KernelTensorRef::new(y_buffer, y_layout),
            );

            kb.program()
                .program_grid::<BLOCK>(dispatch_size, |program| {
                    let lane = program.lane();
                    let group = linear_group(program, dispatch_size);
                    let flat = group * BLOCK as u32 + lane.clone();
                    let in_bounds = flat.lt(total_outputs);
                    let dims = output_dims_from_flat(
                        tile_ir::tile::Tile::from_index(flat.clone()),
                        &out_shape,
                    );
                    let sum = program.loop_fold(
                        tile_ir::TileReduceOp::Sum,
                        k,
                        zero_literal(acc_dtype),
                        |program, k_iter| {
                            let k_index = tile_ir::tile::Tile::from_index(k_iter);
                            let mut a_coords = dims[..rank - 1].to_vec();
                            a_coords.push(k_index.clone());
                            let mut b_coords = dims[..rank - 2].to_vec();
                            b_coords.push(k_index);
                            b_coords.push(dims[rank - 1].clone());

                            let a = program.load_literal(
                                a_storage.at((0, layout_index(&a_meta_body, &a_coords))),
                                in_bounds.clone(),
                                zero_literal(a_meta_body.datatype),
                            );
                            let b = program.load_literal(
                                b_storage.at((0, layout_index(&b_meta_body, &b_coords))),
                                in_bounds.clone(),
                                zero_literal(b_meta_body.datatype),
                            );
                            let (a, a_ty) = apply_unary_function_chain(
                                a,
                                a_meta_body.datatype,
                                &operation.pre_element_wise[0],
                            )
                            .expect("validated matmul pre_element_wise[0] chain");
                            let (b, b_ty) = apply_unary_function_chain(
                                b,
                                b_meta_body.datatype,
                                &operation.pre_element_wise[1],
                            )
                            .expect("validated matmul pre_element_wise[1] chain");
                            let a = cast_tile(a, a_ty, acc_dtype);
                            let b = cast_tile(b, b_ty, acc_dtype);
                            a * b
                        },
                    );
                    let sum = cast_tile(sum, acc_dtype, result_dtype);
                    let (sum, sum_ty) =
                        apply_unary_function_chain(sum, result_dtype, &operation.post_element_wise)
                            .expect("validated matmul post_element_wise chain");
                    let sum = cast_tile(sum, sum_ty, y_meta_body.datatype);
                    program.store(
                        y_storage.at((0, layout_index(&y_meta_body, &dims))),
                        sum,
                        in_bounds,
                    );
                });
            Some(())
        },
    )
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
        &[0, 1],
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
