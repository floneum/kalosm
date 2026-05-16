use std::sync::OnceLock;

use fusor_tile_ir as tile_ir;

use crate::{
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        kernel_backend,
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryOperation, NaryScalar, UnaryFunctionChain},
    tensor::{DataTypeEnum, TensorData},
};

const BLOCK: usize = 256;
const NARY_DIRECT_MODULE_CACHE_SIZE: usize = 1024;

fn nary_direct_module_cache() -> &'static kernel_backend::ModuleCache {
    static CACHE: OnceLock<kernel_backend::ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| kernel_backend::module_cache(NARY_DIRECT_MODULE_CACHE_SIZE))
}

pub(crate) fn build_nary_direct_kernel(
    operation: &NaryOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    build_nary_direct_kernel_with_output_index(operation, graph, workgroup_shape, inputs, None)
}

pub(crate) fn build_nary_direct_kernel_to_output(
    operation: &NaryOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
    output_index: usize,
) -> Option<DirectKernel> {
    build_nary_direct_kernel_with_output_index(
        operation,
        graph,
        workgroup_shape,
        inputs,
        Some(output_index),
    )
}

fn build_nary_direct_kernel_with_output_index(
    operation: &NaryOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
    forced_output_index: Option<usize>,
) -> Option<DirectKernel> {
    let output_index = forced_output_index.or_else(|| operation.output_tensor_index(inputs))?;
    let tensors = inputs
        .iter()
        .map(|input| input.as_tensor().cloned())
        .collect::<Option<Vec<_>>>()?;
    tensors.get(output_index)?;

    if tensors
        .iter()
        .any(|tensor| tensor.datatype() == DataTypeEnum::F16 && !graph.device().f16_supported())
    {
        return None;
    }

    let dispatch_size = operation.dispatch_size(workgroup_shape, inputs);
    let key_label = format!("nary_direct_out_{output_index}");
    let module_key = operation.kernel_module_key_with_dispatch(
        &key_label,
        Some(workgroup_shape),
        dispatch_size,
        inputs,
    );
    let cache_key = kernel_backend::hashed_cache_key("nary", module_key);
    let module = if let Some(module) = nary_direct_module_cache().write().get(&module_key) {
        module.clone()
    } else {
        let module = kernel_backend::cached_kernel_ir(&graph.device(), cache_key.clone(), || {
            build_nary_tile_ir(operation, &tensors, output_index, dispatch_size)
        })?;
        nary_direct_module_cache()
            .write()
            .get_or_insert(module_key, || module.clone())
            .clone()
    };

    let bindings = tensors
        .iter()
        .enumerate()
        .map(|(binding, tensor)| DirectKernelBinding::Storage {
            binding: binding as u32,
            buffer: tensor.buffer().clone(),
            read_only: binding != output_index,
        })
        .collect();

    let name = if std::env::var_os("FUSOR_TRACE_DECODE_NAMES").is_some() {
        operation.name()
    } else {
        cache_key.clone()
    };

    Some(kernel_backend::dynamic_kernel_from_module(
        name,
        cache_key,
        module,
        bindings,
        dispatch_size,
    ))
}

impl NaryOperation {
    pub(crate) fn output_tensor_index(&self, inputs: &[MirValue]) -> Option<usize> {
        inputs[..self.inputs.len()]
            .iter()
            .enumerate()
            .find_map(|(i, input)| {
                if self.expression.uses_custom_indexing_for_input(i) {
                    return None;
                }
                let data = input.as_tensor()?;
                (data.datatype() == self.output_datatype
                    && data.owned()
                    && !data.layout().allocation_overlaps())
                .then_some(i)
            })
            .or_else(|| inputs.len().checked_sub(1))
    }
}

fn build_nary_tile_ir(
    operation: &NaryOperation,
    tensors: &[TensorData],
    output_index: usize,
    dispatch_size: [u32; 3],
) -> Option<tile_ir::KernelIr> {
    let total_elements = operation
        .shape
        .iter()
        .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
    let tensor_metas = tensors
        .iter()
        .map(TensorMeta::new)
        .collect::<Option<Vec<_>>>()?;

    Some(tile_ir::tile::build(move |phase| {
        let storages = tensor_metas
            .iter()
            .enumerate()
            .map(|(binding, meta)| {
                let layout = tile_ir::Layout::strided(
                    tile_ir::MemoryLevel::Storage,
                    tile_ir::Shape::new([1, meta.allocation_len]),
                    &[0, 1],
                );
                if binding == output_index {
                    phase.storage_write_element_with_layout_offset::<2>(meta.element, layout, 0)
                } else {
                    phase.storage_read_element_with_layout_offset::<2>(meta.element, layout, 0)
                }
            })
            .collect::<Vec<_>>();

        phase.program_grid::<BLOCK>(dispatch_size, |program| {
            let lane = program.lane();
            let group = linear_group(program, dispatch_size);
            let flat_index = group * BLOCK as u32 + lane.clone();
            let in_bounds = flat_index.lt(total_elements);
            let flat_value = tile_ir::tile::Tile::from_index(flat_index.clone());
            let dims = output_dims_from_flat(flat_value, &operation.shape);
            let (value, value_ty) = eval_nary_expr(
                program,
                &operation.expression,
                &dims,
                &storages,
                &tensor_metas,
                in_bounds.clone(),
            );
            let value = cast_tile(value, value_ty, operation.output_datatype);
            let output_index_value = layout_index(&tensor_metas[output_index], &dims);
            program.store(
                storages[output_index].at((0, output_index_value)),
                value,
                in_bounds,
            );
        });
    }))
}

fn eval_nary_expr(
    program: &mut tile_ir::tile::TileBlock<'_>,
    expr: &NaryExpr,
    dims: &[tile_ir::tile::Tile],
    storages: &[tile_ir::tile::Storage<tile_ir::tile::RuntimeElement, 2>],
    metas: &[TensorMeta],
    mask: tile_ir::tile::Mask,
) -> (tile_ir::tile::Tile, DataTypeEnum) {
    match expr {
        NaryExpr::Op { children, function } => {
            let mut values = children
                .iter()
                .zip(&function.input_types)
                .map(|(child, expected)| {
                    let (value, ty) =
                        eval_nary_expr(program, child, dims, storages, metas, mask.clone());
                    (cast_tile(value, ty, *expected), *expected)
                })
                .collect::<Vec<_>>();
            (emit_function(function, &mut values), function.output_type)
        }
        NaryExpr::IndexedInput { input_idx, indices } => {
            let meta = &metas[*input_idx];
            let storage = &storages[*input_idx];
            let coords = indices
                .iter()
                .map(|index| {
                    let (value, ty) =
                        eval_nary_expr(program, index, dims, storages, metas, mask.clone());
                    cast_tile(value, ty, DataTypeEnum::U32)
                })
                .collect::<Vec<_>>();
            let index = layout_index(meta, &coords);
            let value = program.load(storage.at((0, index)), mask, zero_literal(meta.datatype));
            (value, meta.datatype)
        }
        NaryExpr::DimIndex(dim) => (dims[*dim].clone(), DataTypeEnum::U32),
        NaryExpr::Scalar(value) => (tile_literal(*value), value.datatype()),
    }
}

fn emit_function(
    function: &NaryFunction,
    values: &mut [(tile_ir::tile::Tile, DataTypeEnum)],
) -> tile_ir::tile::Tile {
    match function.op {
        NaryOp::Add => values[0].0.clone() + values[1].0.clone(),
        NaryOp::Sub => values[0].0.clone() - values[1].0.clone(),
        NaryOp::Mul => values[0].0.clone() * values[1].0.clone(),
        NaryOp::Div => values[0].0.clone() / values[1].0.clone(),
        NaryOp::Rem => values[0].0.clone() % values[1].0.clone(),
        NaryOp::Pow => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Pow, values[1].0.clone()),
        NaryOp::Min => values[0].0.clone().min(values[1].0.clone()),
        NaryOp::Max => values[0].0.clone().max(values[1].0.clone()),
        NaryOp::Neg => values[0].0.clone().unary(tile_ir::TileUnaryOp::Neg),
        NaryOp::Cast => values[0].0.clone().cast(tile_element(function.output_type)),
        NaryOp::Select => tile_ir::tile::Tile::select(
            values[0].0.clone(),
            values[1].0.clone(),
            values[2].0.clone(),
        ),
        NaryOp::Exp | NaryOp::ApproximateExp | NaryOp::LessApproximateExp => {
            values[0].0.clone().unary(tile_ir::TileUnaryOp::Exp)
        }
        NaryOp::Exp2 => values[0].0.clone().unary(tile_ir::TileUnaryOp::Exp2),
        NaryOp::Log => values[0].0.clone().unary(tile_ir::TileUnaryOp::Log),
        NaryOp::Log2 => values[0].0.clone().unary(tile_ir::TileUnaryOp::Log2),
        NaryOp::Sqrt => values[0].0.clone().unary(tile_ir::TileUnaryOp::Sqrt),
        NaryOp::Sin => values[0].0.clone().unary(tile_ir::TileUnaryOp::Sin),
        NaryOp::Cos => values[0].0.clone().unary(tile_ir::TileUnaryOp::Cos),
        NaryOp::Tan => values[0].0.clone().unary(tile_ir::TileUnaryOp::Tan),
        NaryOp::Tanh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Tanh),
        NaryOp::TanhExact => tanh_exact(values[0].0.clone()),
        NaryOp::Asin => values[0].0.clone().unary(tile_ir::TileUnaryOp::Asin),
        NaryOp::Acos => values[0].0.clone().unary(tile_ir::TileUnaryOp::Acos),
        NaryOp::Atan => values[0].0.clone().unary(tile_ir::TileUnaryOp::Atan),
        NaryOp::Sinh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Sinh),
        NaryOp::Cosh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Cosh),
        NaryOp::Asinh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Asinh),
        NaryOp::Acosh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Acosh),
        NaryOp::Atanh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Atanh),
        NaryOp::Abs => values[0].0.clone().unary(tile_ir::TileUnaryOp::Abs),
        NaryOp::Equal => compare_tile(
            tile_ir::TileCompareOp::Eq,
            &values[0],
            &values[1],
            function.output_type,
        ),
        NaryOp::Less => compare_tile(
            tile_ir::TileCompareOp::Lt,
            &values[0],
            &values[1],
            function.output_type,
        ),
        NaryOp::LessEqual => compare_tile(
            tile_ir::TileCompareOp::Le,
            &values[0],
            &values[1],
            function.output_type,
        ),
        NaryOp::Greater => compare_tile(
            tile_ir::TileCompareOp::Gt,
            &values[0],
            &values[1],
            function.output_type,
        ),
        NaryOp::GreaterEqual => compare_tile(
            tile_ir::TileCompareOp::Ge,
            &values[0],
            &values[1],
            function.output_type,
        ),
        NaryOp::AddConst(scalar) => values[0].0.clone() + tile_literal(scalar),
        NaryOp::SubConst(scalar) => values[0].0.clone() - tile_literal(scalar),
        NaryOp::RSubConst(scalar) => tile_literal(scalar) - values[0].0.clone(),
        NaryOp::MulConst(scalar) => values[0].0.clone() * tile_literal(scalar),
        NaryOp::DivConst(scalar) => values[0].0.clone() / tile_literal(scalar),
        NaryOp::RDivConst(scalar) => tile_literal(scalar) / values[0].0.clone(),
        NaryOp::RemConst(scalar) => values[0].0.clone() % tile_literal(scalar),
        NaryOp::RRemConst(scalar) => tile_literal(scalar) % values[0].0.clone(),
        NaryOp::PowConst(scalar) => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Pow, tile_literal(scalar)),
        NaryOp::MinConst(scalar) => values[0].0.clone().min(tile_literal(scalar)),
        NaryOp::MaxConst(scalar) => values[0].0.clone().max(tile_literal(scalar)),
        NaryOp::EqualConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Eq,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::LessConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Lt,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::LessEqualConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Le,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::GreaterConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Gt,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::GreaterEqualConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Ge,
            &values[0],
            scalar,
            function.output_type,
        ),
    }
}

/// Evaluate a tile-evaluatable `NaryExpr` (no DimIndex, only element-wise
/// IndexedInput leaves) over a vector of pre-loaded tile values. Used by the
/// resolver to re-emit captured tensor-level expressions at the tile-IR level
/// when fusing into kernels that materialize inputs in-register (e.g. the
/// qgemv paired epilogue).
pub(crate) fn eval_nary_expr_on_tiles(
    expr: &NaryExpr,
    inputs: &[(tile_ir::tile::Tile, DataTypeEnum)],
    output_dtype: DataTypeEnum,
) -> (tile_ir::tile::Tile, DataTypeEnum) {
    match expr {
        NaryExpr::Op { children, function } => {
            let mut values = children
                .iter()
                .zip(&function.input_types)
                .map(|(child, expected)| {
                    let (value, ty) = eval_nary_expr_on_tiles(child, inputs, output_dtype);
                    (cast_tile(value, ty, *expected), *expected)
                })
                .collect::<Vec<_>>();
            (emit_function(function, &mut values), function.output_type)
        }
        NaryExpr::IndexedInput { input_idx, .. } => {
            let (tile, ty) = inputs[*input_idx].clone();
            (tile, ty)
        }
        NaryExpr::Scalar(value) => (tile_literal(*value), value.datatype()),
        NaryExpr::DimIndex(_) => {
            panic!("eval_nary_expr_on_tiles called with a DimIndex leaf — not supported");
        }
    }
}

pub(crate) fn apply_unary_function_chain(
    mut value: tile_ir::tile::Tile,
    mut value_ty: DataTypeEnum,
    chain: &UnaryFunctionChain,
) -> Option<(tile_ir::tile::Tile, DataTypeEnum)> {
    if chain.input_datatype() != value_ty {
        return None;
    }

    for function in &chain.functions {
        if function.input_types.as_slice() != [value_ty] {
            return None;
        }
        let mut values = [(value, value_ty)];
        value = emit_function(function, &mut values);
        value_ty = function.output_type;
    }

    Some((value, value_ty))
}

fn tanh_exact(value: tile_ir::tile::Tile) -> tile_ir::tile::Tile {
    let exp_pos = value.clone().unary(tile_ir::TileUnaryOp::Exp);
    let exp_neg = value
        .unary(tile_ir::TileUnaryOp::Neg)
        .unary(tile_ir::TileUnaryOp::Exp);
    (exp_pos.clone() - exp_neg.clone()) / (exp_pos + exp_neg)
}

fn compare_tile(
    op: tile_ir::TileCompareOp,
    left: &(tile_ir::tile::Tile, DataTypeEnum),
    right: &(tile_ir::tile::Tile, DataTypeEnum),
    output: DataTypeEnum,
) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::compare(
        op,
        left.0.clone(),
        cast_tile(right.0.clone(), right.1, left.1),
        tile_element(output),
    )
}

fn compare_const(
    op: tile_ir::TileCompareOp,
    left: &(tile_ir::tile::Tile, DataTypeEnum),
    scalar: NaryScalar,
    output: DataTypeEnum,
) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::compare(
        op,
        left.0.clone(),
        cast_tile(tile_literal(scalar), scalar.datatype(), left.1),
        tile_element(output),
    )
}

fn output_dims_from_flat(flat: tile_ir::tile::Tile, shape: &[usize]) -> Vec<tile_ir::tile::Tile> {
    (0..shape.len())
        .map(|axis| {
            let dim = shape[axis] as u32;
            if dim == 1 {
                return tile_u32(0);
            }
            let divisor = shape[axis + 1..]
                .iter()
                .fold(1u32, |acc, dim| acc.saturating_mul(*dim as u32));
            let quotient = if divisor == 1 {
                flat.clone()
            } else {
                flat.clone() / tile_u32(divisor)
            };
            quotient % tile_u32(dim)
        })
        .collect()
}

fn layout_index(meta: &TensorMeta, coords: &[tile_ir::tile::Tile]) -> tile_ir::tile::Tile {
    let mut index = tile_u32(meta.offset);
    for (axis, (coord, stride)) in coords.iter().zip(&meta.strides).enumerate() {
        if *stride == 0 || meta.shape.get(axis).copied() == Some(1) {
            continue;
        }
        index = index + coord.clone() * tile_u32(*stride);
    }
    index
}

fn linear_group(
    program: &tile_ir::tile::TileBlock<'_>,
    dispatch_size: [u32; 3],
) -> tile_ir::tile::ScalarIndex {
    program.program_id(tile_ir::WorkgroupAxis::X)
        + program.program_id(tile_ir::WorkgroupAxis::Y) * dispatch_size[0]
        + program.program_id(tile_ir::WorkgroupAxis::Z)
            * dispatch_size[0].saturating_mul(dispatch_size[1])
}

fn cast_tile(
    value: tile_ir::tile::Tile,
    source: DataTypeEnum,
    target: DataTypeEnum,
) -> tile_ir::tile::Tile {
    if source == target {
        value
    } else {
        value.cast(tile_element(target))
    }
}

fn tile_literal(value: NaryScalar) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::literal(match value {
        NaryScalar::F32(value) => tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(value)),
        NaryScalar::F16(value) => tile_ir::TileLiteral::F16(value.to_bits()),
        NaryScalar::U32(value) => tile_ir::TileLiteral::U32(value),
    })
}

fn tile_u32(value: u32) -> tile_ir::tile::Tile {
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
    shape: Vec<u32>,
    strides: Vec<u32>,
    offset: u32,
    allocation_len: u32,
}

impl TensorMeta {
    fn new(tensor: &TensorData) -> Option<Self> {
        Some(Self {
            datatype: tensor.datatype(),
            element: tile_element(tensor.datatype()),
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
