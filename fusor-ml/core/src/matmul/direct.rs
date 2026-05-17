use fusor_tile_ir as tile_ir;

use crate::{
    matmul::MatMulOperation,
    mir::{
        inputs::MirValue, kernel_backend, kernel_backend::DirectKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::{
        ValueTile, apply_unary_function_chain, layout_index, linear_group, output_dims_from_flat,
    },
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::distribute_workgroups,
};

const BLOCK: usize = 256;

struct MatmulSerialDirectKernelVariant;

pub(crate) fn build_serial_matmul_direct_kernel(
    operation: &MatMulOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
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
    let dispatch_size = distribute_workgroups(
        total_outputs.div_ceil(BLOCK as u32),
        graph.device().limits().max_compute_workgroups_per_dimension,
    );
    let cache_key = operation.kernel_cache_key_with_dispatch(
        kernel_backend::KernelVariantKey::of::<MatmulSerialDirectKernelVariant>(),
        Some(workgroup_shape),
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
    let acc_dtype = match a_product_dtype {
        DataTypeEnum::U32 => DataTypeEnum::U32,
        DataTypeEnum::F32 | DataTypeEnum::F16 => DataTypeEnum::F32,
    };
    let result_dtype = a_product_dtype;
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

    let a_buffer = input_a.buffer().clone();
    let b_buffer = input_b.buffer().clone();
    let y_buffer = output.buffer().clone();
    let a_layout = flat_layout(a_meta.allocation_len);
    let b_layout = flat_layout(b_meta.allocation_len);
    let y_layout = flat_layout(y_meta.allocation_len);
    let a_meta_body = a_meta.clone();
    let b_meta_body = b_meta.clone();
    let y_meta_body = y_meta.clone();
    let pre_a = operation.pre_element_wise[0].clone();
    let pre_b = operation.pre_element_wise[1].clone();
    let post = operation.post_element_wise.clone();

    kernel_backend::run_kernel(
        graph.device().kernel_cache(),
        operation.name(),
        cache_key,
        dispatch_size,
        move |kb| {
            let a_tensor = tile_ir::KernelTensorRef::new(a_buffer.clone(), a_layout.clone());
            let b_tensor = tile_ir::KernelTensorRef::new(b_buffer.clone(), b_layout.clone());
            let y_tensor = tile_ir::KernelTensorRef::new(y_buffer.clone(), y_layout.clone());
            let a_storage = storage_read(kb, a_meta_body.datatype, a_tensor);
            let b_storage = storage_read(kb, b_meta_body.datatype, b_tensor);
            let y_storage = storage_write(kb, y_meta_body.datatype, y_tensor);

            kb.program()
                .program_grid::<BLOCK>(dispatch_size, |program| {
                    let lane = program.lane();
                    let group = linear_group(program, dispatch_size);
                    let flat = group * BLOCK as u32 + lane.clone();
                    let in_bounds = flat.lt(total_outputs);
                    let dims = output_dims_from_flat_u32(flat.clone(), &out_shape);

                    let value_at =
                        |program: &mut tile_ir::tile::TileBlock<'_>,
                         k_index: tile_ir::tile::Tile<tile_ir::U32>| {
                            let mut a_coords = dims[..rank - 1].to_vec();
                            a_coords.push(k_index.clone());
                            let mut b_coords = dims[..rank - 2].to_vec();
                            b_coords.push(k_index);
                            b_coords.push(dims[rank - 1].clone());

                            let a = a_storage.load(
                                program,
                                layout_index(&a_meta_body.as_nary_meta(), &a_coords),
                                in_bounds.clone(),
                            );
                            let b = b_storage.load(
                                program,
                                layout_index(&b_meta_body.as_nary_meta(), &b_coords),
                                in_bounds.clone(),
                            );
                            let (a, a_ty) = apply_unary_function_chain(
                                a.into_f32(),
                                a_meta_body.datatype,
                                &pre_a,
                            )
                            .expect("validated matmul pre_element_wise[0] chain");
                            let (b, b_ty) = apply_unary_function_chain(
                                b.into_f32(),
                                b_meta_body.datatype,
                                &pre_b,
                            )
                            .expect("validated matmul pre_element_wise[1] chain");
                            let a = ValueTile::F32(a).cast_to(a_ty).cast_to(acc_dtype);
                            let b = ValueTile::F32(b).cast_to(b_ty).cast_to(acc_dtype);
                            a.binary(tile_ir::TileBinaryOp::Mul, b)
                        };

                    let sum = match acc_dtype {
                        DataTypeEnum::F32 => ValueTile::F32(program.loop_fold(
                            tile_ir::TileReduceOp::Sum,
                            k,
                            tile_ir::TileLiteral::f32(0.0),
                            |program, k_index| value_at(program, k_index).into_f32(),
                        )),
                        DataTypeEnum::F16 => unreachable!("matmul accumulates f16 products in f32"),
                        DataTypeEnum::U32 => ValueTile::U32(program.loop_fold(
                            tile_ir::TileReduceOp::Sum,
                            k,
                            tile_ir::TileLiteral::U32(0),
                            |program, k_index| value_at(program, k_index).into_u32(),
                        )),
                    };
                    let sum = sum.cast_to(result_dtype);
                    let (sum, sum_ty) =
                        apply_unary_function_chain(sum.into_f32(), result_dtype, &post)
                            .expect("validated matmul post_element_wise chain");
                    let sum = ValueTile::F32(sum)
                        .cast_to(sum_ty)
                        .cast_to(y_meta_body.datatype);
                    y_storage.store(
                        program,
                        layout_index(&y_meta_body.as_nary_meta(), &dims),
                        sum,
                        in_bounds,
                    );
                });
            Some(())
        },
    )
}

fn storage_read<B>(
    kb: &mut tile_ir::KernelBuilder<B>,
    datatype: DataTypeEnum,
    tensor: tile_ir::KernelTensorRef<B>,
) -> crate::nary_direct::Storage2 {
    match datatype {
        DataTypeEnum::F32 => crate::nary_direct::Storage2::F32(kb.read::<tile_ir::F32, 2>(tensor)),
        DataTypeEnum::F16 => crate::nary_direct::Storage2::F16(kb.read::<tile_ir::F16, 2>(tensor)),
        DataTypeEnum::U32 => crate::nary_direct::Storage2::U32(kb.read::<tile_ir::U32, 2>(tensor)),
    }
}

fn storage_write<B>(
    kb: &mut tile_ir::KernelBuilder<B>,
    datatype: DataTypeEnum,
    tensor: tile_ir::KernelTensorRef<B>,
) -> crate::nary_direct::Storage2 {
    match datatype {
        DataTypeEnum::F32 => crate::nary_direct::Storage2::F32(kb.write::<tile_ir::F32, 2>(tensor)),
        DataTypeEnum::F16 => crate::nary_direct::Storage2::F16(kb.write::<tile_ir::F16, 2>(tensor)),
        DataTypeEnum::U32 => crate::nary_direct::Storage2::U32(kb.write::<tile_ir::U32, 2>(tensor)),
    }
}

fn output_dims_from_flat_u32(
    flat: tile_ir::tile::Tile<tile_ir::U32>,
    shape: &[u32],
) -> Vec<tile_ir::tile::Tile<tile_ir::U32>> {
    let shape = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    output_dims_from_flat(flat, &shape)
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
