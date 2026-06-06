use fusor_tile_ir as tile_ir;

use crate::{
    mir::{
        inputs::MirValue, kernel_backend, kernel_backend::DirectKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::{
        TensorMeta, ValueTile, apply_unary_function_chain, flat_layout, layout_index, linear_group,
        output_dims_from_flat,
    },
    nary_wise::NaryScalar,
    reduce::{ReduceOp, ReduceOperation},
    tensor::DataTypeEnum,
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
            let input_element = crate::nary_direct::datatype_element(input_meta_body.datatype);
            let input_storage = match input_meta_body.datatype {
                DataTypeEnum::F32 => {
                    crate::nary_direct::Storage2::F32(kb.read(input_element, input_tensor))
                }
                DataTypeEnum::F16 => {
                    crate::nary_direct::Storage2::F16(kb.read(input_element, input_tensor))
                }
                DataTypeEnum::U32 => {
                    crate::nary_direct::Storage2::U32(kb.read(input_element, input_tensor))
                }
            };
            let output_element = crate::nary_direct::datatype_element(output_meta_body.datatype);
            let output_storage = match output_meta_body.datatype {
                DataTypeEnum::F32 => {
                    crate::nary_direct::Storage2::F32(kb.write(output_element, output_tensor))
                }
                DataTypeEnum::F16 => {
                    crate::nary_direct::Storage2::F16(kb.write(output_element, output_tensor))
                }
                DataTypeEnum::U32 => {
                    crate::nary_direct::Storage2::U32(kb.write(output_element, output_tensor))
                }
            };

            let cooperative =
                crate::reduce::use_cooperative_reduce(total_outputs, reduce_size, BLOCK as u32);
            if cooperative {
                let chunks = reduce_size.div_ceil(BLOCK as u32);
                kb.program()
                    .program_grid(BLOCK as u32, dispatch_size, |program| {
                        let lane = program.lane();
                        let output_flat = linear_group(program, dispatch_size);
                        let in_bounds = output_flat.lt(total_outputs);
                        let dims = output_dims_from_flat_usize(output_flat.clone(), &output_shape);
                        let base = layout_index(&input_meta_body, &dims);
                        let reduce_binary = reduce_op.binary();

                        let value_at =
                            |program: &mut tile_ir::tile::TileBlock<'_>,
                             reduce_index: tile_ir::tile::Tile,
                             active: tile_ir::tile::Mask| {
                                let value_index = base.clone() + reduce_index * reduce_stride;
                                let value =
                                    input_storage.load(program, value_index, active.clone());
                                let (value, value_ty) = apply_unary_function_chain(
                                    value.into_f32(),
                                    input_meta_body.datatype,
                                    &pre_chain,
                                )
                                .expect("validated reduce pre_element_wise chain");
                                let value = ValueTile::F32(value)
                                    .cast_to(value_ty)
                                    .cast_to(reduce_dtype);
                                let identity = tile_ir::tile::Tile::literal(tile_literal_for(
                                    initial,
                                    reduce_dtype,
                                ));
                                match reduce_dtype {
                                    DataTypeEnum::F32 => tile_ir::tile::Tile::select(
                                        active,
                                        value.into_f32(),
                                        identity,
                                    ),
                                    DataTypeEnum::F16 => tile_ir::tile::Tile::select(
                                        active,
                                        value.into_f16(),
                                        identity,
                                    ),
                                    DataTypeEnum::U32 => tile_ir::tile::Tile::select(
                                        active,
                                        value.into_u32(),
                                        identity,
                                    ),
                                }
                            };

                        let initial_acc =
                            tile_ir::tile::Tile::literal(tile_literal_for(initial, reduce_dtype));
                        let [partial] = program.fold(
                            tile_ir::tile::range(chunks),
                            [initial_acc],
                            |program, loop_index, [acc]| {
                                let reduce_index = loop_index * BLOCK as u32 + lane.clone();
                                let active = in_bounds.clone().and(reduce_index.lt(reduce_size));
                                [
                                    acc.binary(
                                        reduce_binary,
                                        value_at(program, reduce_index, active),
                                    ),
                                ]
                            },
                        );
                        let reduced = program.group_reduce(reduce_op, BLOCK as u32, partial);

                        let reduced = match reduce_dtype {
                            DataTypeEnum::F32 => ValueTile::F32(reduced),
                            DataTypeEnum::F16 => ValueTile::F16(reduced),
                            DataTypeEnum::U32 => ValueTile::U32(reduced),
                        };
                        let (reduced, reduced_ty) = apply_unary_function_chain(
                            reduced.into_f32(),
                            reduce_dtype,
                            &post_chain,
                        )
                        .expect("validated reduce post_element_wise chain");
                        let reduced = ValueTile::F32(reduced)
                            .cast_to(reduced_ty)
                            .cast_to(output_meta_body.datatype);
                        let output_index = layout_index(&output_meta_body, &dims);
                        let store_mask = in_bounds.and(lane.eq(0u32));
                        output_storage.store(program, output_index, reduced, store_mask);
                    });
            } else {
                kb.program()
                    .program_grid(BLOCK as u32, dispatch_size, |program| {
                        let lane = program.lane();
                        let group = linear_group(program, dispatch_size);
                        let flat = group * BLOCK as u32 + lane.clone();
                        let in_bounds = flat.lt(total_outputs);
                        let dims = output_dims_from_flat_usize(flat.clone(), &output_shape);
                        let base = layout_index(&input_meta_body, &dims);
                        let value_at =
                            |program: &mut tile_ir::tile::TileBlock<'_>,
                             loop_index: tile_ir::tile::Tile| {
                                let value_index = base.clone() + loop_index * reduce_stride;
                                let value =
                                    input_storage.load(program, value_index, in_bounds.clone());
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

                        let reduce_binary = reduce_op.binary();
                        let reduced = match reduce_dtype {
                            DataTypeEnum::F32 => {
                                let [acc] = program.fold(
                                    tile_ir::tile::range(reduce_size),
                                    [tile_ir::tile::Tile::literal(tile_literal_for(
                                        initial,
                                        DataTypeEnum::F32,
                                    ))],
                                    |program, loop_index, [acc]| {
                                        [acc.binary(
                                            reduce_binary,
                                            value_at(program, loop_index).into_f32(),
                                        )]
                                    },
                                );
                                ValueTile::F32(acc)
                            }
                            DataTypeEnum::F16 => {
                                let [acc] = program.fold(
                                    tile_ir::tile::range(reduce_size),
                                    [tile_ir::tile::Tile::literal(tile_literal_for(
                                        initial,
                                        DataTypeEnum::F16,
                                    ))],
                                    |program, loop_index, [acc]| {
                                        [acc.binary(
                                            reduce_binary,
                                            value_at(program, loop_index).into_f16(),
                                        )]
                                    },
                                );
                                ValueTile::F16(acc)
                            }
                            DataTypeEnum::U32 => {
                                let [acc] = program.fold(
                                    tile_ir::tile::range(reduce_size),
                                    [tile_ir::tile::Tile::literal(tile_literal_for(
                                        initial,
                                        DataTypeEnum::U32,
                                    ))],
                                    |program, loop_index, [acc]| {
                                        [acc.binary(
                                            reduce_binary,
                                            value_at(program, loop_index).into_u32(),
                                        )]
                                    },
                                );
                                ValueTile::U32(acc)
                            }
                        };

                        let (reduced, reduced_ty) = apply_unary_function_chain(
                            reduced.into_f32(),
                            reduce_dtype,
                            &post_chain,
                        )
                        .expect("validated reduce post_element_wise chain");
                        let reduced = ValueTile::F32(reduced)
                            .cast_to(reduced_ty)
                            .cast_to(output_meta_body.datatype);
                        let output_index = layout_index(&output_meta_body, &dims);
                        output_storage.store(program, output_index, reduced, in_bounds);
                    });
            }
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
    flat: tile_ir::tile::Tile,
    shape: &[u32],
) -> Vec<tile_ir::tile::Tile> {
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
