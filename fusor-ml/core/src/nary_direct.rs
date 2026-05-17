use std::{hash::Hash, sync::Arc, sync::OnceLock};

use fusor_tile_ir as tile_ir;

use crate::{
    mir::{
        inputs::MirValue,
        kernel_backend::{self, DirectKernel, DirectKernelBinding},
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryOperation, NaryScalar, UnaryFunctionChain},
    tensor::{DataTypeEnum, TensorData},
};

const BLOCK: usize = 256;
const NARY_DIRECT_MODULE_CACHE_SIZE: usize = 1024;

struct NaryDirectKernelVariant;

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
    let variant =
        kernel_backend::KernelVariantKey::with_payload::<NaryDirectKernelVariant>(|state| {
            output_index.hash(state);
        });
    let module_key = operation.kernel_module_key_with_dispatch(
        variant,
        Some(workgroup_shape),
        dispatch_size,
        inputs,
    );
    let naga = kernel_backend::cached_hashed_naga(nary_direct_module_cache(), module_key, || {
        let ir = build_nary_tile_ir(operation, &tensors, output_index, dispatch_size)?;
        Some(Arc::new(ir.lower_to_naga().ok()?.module().clone()))
    })?;
    let cached = graph
        .device()
        .kernel_cache()
        .get_or_insert_kernel(module_key, || naga);

    let bindings = tensors
        .iter()
        .enumerate()
        .map(|(binding, tensor)| DirectKernelBinding {
            binding: binding as u32,
            buffer: tensor.buffer().clone(),
            read_only: binding != output_index,
        })
        .collect();

    let name = if std::env::var_os("FUSOR_TRACE_DECODE_NAMES").is_some() {
        operation.name()
    } else {
        format!("nary_direct_out_{output_index}")
    };

    Some(DirectKernel::from_cached(name, cached, bindings, dispatch_size))
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

#[derive(Clone)]
pub(crate) enum ValueTile {
    F32(tile_ir::tile::Tile<tile_ir::F32>),
    F16(tile_ir::tile::Tile<tile_ir::F16>),
    U32(tile_ir::tile::Tile<tile_ir::U32>),
    Bool(tile_ir::tile::Mask),
}

impl ValueTile {
    pub(crate) fn cast_to(self, target: DataTypeEnum) -> Self {
        match (self, target) {
            (Self::F32(v), DataTypeEnum::F32) => Self::F32(v),
            (Self::F32(v), DataTypeEnum::F16) => Self::F16(v.cast::<tile_ir::F16>()),
            (Self::F32(v), DataTypeEnum::U32) => Self::U32(v.cast::<tile_ir::U32>()),
            (Self::F16(v), DataTypeEnum::F32) => Self::F32(v.cast::<tile_ir::F32>()),
            (Self::F16(v), DataTypeEnum::F16) => Self::F16(v),
            (Self::F16(v), DataTypeEnum::U32) => Self::U32(v.cast::<tile_ir::U32>()),
            (Self::U32(v), DataTypeEnum::F32) => Self::F32(v.cast::<tile_ir::F32>()),
            (Self::U32(v), DataTypeEnum::F16) => Self::F16(v.cast::<tile_ir::F16>()),
            (Self::U32(v), DataTypeEnum::U32) => Self::U32(v),
            (Self::Bool(v), DataTypeEnum::F32) => Self::F32(bool_as_f32(v)),
            (Self::Bool(v), DataTypeEnum::F16) => Self::F16(bool_as_f32(v).cast::<tile_ir::F16>()),
            (Self::Bool(v), DataTypeEnum::U32) => Self::U32(bool_as_u32(v)),
        }
    }

    pub(crate) fn into_f32(self) -> tile_ir::tile::Tile {
        match self.cast_to(DataTypeEnum::F32) {
            Self::F32(v) => v,
            _ => unreachable!(),
        }
    }

    pub(crate) fn into_f16(self) -> tile_ir::tile::Tile<tile_ir::F16> {
        match self.cast_to(DataTypeEnum::F16) {
            Self::F16(v) => v,
            _ => unreachable!(),
        }
    }

    pub(crate) fn into_u32(self) -> tile_ir::tile::Tile<tile_ir::U32> {
        match self.cast_to(DataTypeEnum::U32) {
            Self::U32(v) => v,
            _ => unreachable!(),
        }
    }

    pub(crate) fn into_mask(self) -> tile_ir::tile::Mask {
        match self {
            Self::Bool(v) => v,
            Self::F32(v) => v.ne(0.0),
            Self::F16(v) => v.ne(tile_ir::tile::Tile::<tile_ir::F16>::literal_bits(0)),
            Self::U32(v) => v.ne(0u32),
        }
    }

    fn unary(self, op: tile_ir::TileUnaryOp) -> Self {
        match self {
            Self::F32(v) => Self::F32(v.unary(op)),
            Self::F16(v) => Self::F16(v.unary(op)),
            Self::U32(v) => Self::U32(v.unary(op)),
            Self::Bool(v) => Self::Bool(v.unary(op)),
        }
    }

    pub(crate) fn binary(self, op: tile_ir::TileBinaryOp, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => Self::F32(a.binary(op, b)),
            (Self::F16(a), Self::F16(b)) => Self::F16(a.binary(op, b)),
            (Self::U32(a), Self::U32(b)) => Self::U32(a.binary(op, b)),
            (Self::Bool(a), Self::Bool(b)) => Self::Bool(a.binary(op, b)),
            _ => panic!("nary direct binary op called with mismatched tile types"),
        }
    }

    fn compare(self, op: tile_ir::TileCompareOp, rhs: Self, output: DataTypeEnum) -> Self {
        let mask = match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => tile_ir::tile::Tile::compare_bool(op, a, b),
            (Self::F16(a), Self::F16(b)) => tile_ir::tile::Tile::compare_bool(op, a, b),
            (Self::U32(a), Self::U32(b)) => tile_ir::tile::Tile::compare_bool(op, a, b),
            (Self::Bool(a), Self::Bool(b)) => tile_ir::tile::Tile::compare_bool(op, a, b),
            _ => panic!("nary direct compare called with mismatched tile types"),
        };
        ValueTile::Bool(mask).cast_to(output)
    }
}

fn bool_as_f32(value: tile_ir::tile::Mask) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::select(value, 1.0.into(), 0.0.into())
}

fn bool_as_u32(value: tile_ir::tile::Mask) -> tile_ir::tile::Tile<tile_ir::U32> {
    tile_ir::tile::Tile::select(value, 1u32.into(), 0u32.into())
}

pub(crate) enum Storage2 {
    F32(tile_ir::tile::Storage<tile_ir::F32, 2>),
    F16(tile_ir::tile::Storage<tile_ir::F16, 2>),
    U32(tile_ir::tile::Storage<tile_ir::U32, 2>),
}

impl Storage2 {
    pub(crate) fn load(
        &self,
        program: &tile_ir::tile::TileBlock<'_>,
        index: tile_ir::tile::Tile<tile_ir::U32>,
        mask: tile_ir::tile::Mask,
    ) -> ValueTile {
        match self {
            Self::F32(storage) => ValueTile::F32(program.load(
                storage.at((0u32, index)),
                mask,
                zero_literal(DataTypeEnum::F32),
            )),
            Self::F16(storage) => ValueTile::F16(program.load(
                storage.at((0u32, index)),
                mask,
                zero_literal(DataTypeEnum::F16),
            )),
            Self::U32(storage) => ValueTile::U32(program.load(
                storage.at((0u32, index)),
                mask,
                zero_literal(DataTypeEnum::U32),
            )),
        }
    }

    pub(crate) fn store(
        &self,
        program: &mut tile_ir::tile::TileBlock<'_>,
        index: tile_ir::tile::Tile<tile_ir::U32>,
        value: ValueTile,
        mask: tile_ir::tile::Mask,
    ) {
        match self {
            Self::F32(storage) => {
                if let ValueTile::F32(value) = value.cast_to(DataTypeEnum::F32) {
                    program.store(storage.at((0u32, index)), value, mask);
                }
            }
            Self::F16(storage) => {
                if let ValueTile::F16(value) = value.cast_to(DataTypeEnum::F16) {
                    program.store(storage.at((0u32, index)), value, mask);
                }
            }
            Self::U32(storage) => {
                if let ValueTile::U32(value) = value.cast_to(DataTypeEnum::U32) {
                    program.store(storage.at((0u32, index)), value, mask);
                }
            }
        }
    }
}

pub(crate) fn declare_storage(
    phase: &mut tile_ir::tile::Program,
    meta: &TensorMeta,
    write: bool,
) -> Storage2 {
    let layout = tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([1, meta.allocation_len]),
        &[0, 1],
    );
    match (meta.datatype, write) {
        (DataTypeEnum::F32, true) => {
            Storage2::F32(phase.storage_write_with_layout_offset::<tile_ir::F32, 2>(layout, 0))
        }
        (DataTypeEnum::F32, false) => {
            Storage2::F32(phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(layout, 0))
        }
        (DataTypeEnum::F16, true) => {
            Storage2::F16(phase.storage_write_with_layout_offset::<tile_ir::F16, 2>(layout, 0))
        }
        (DataTypeEnum::F16, false) => {
            Storage2::F16(phase.storage_read_with_layout_offset::<tile_ir::F16, 2>(layout, 0))
        }
        (DataTypeEnum::U32, true) => {
            Storage2::U32(phase.storage_write_with_layout_offset::<tile_ir::U32, 2>(layout, 0))
        }
        (DataTypeEnum::U32, false) => {
            Storage2::U32(phase.storage_read_with_layout_offset::<tile_ir::U32, 2>(layout, 0))
        }
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
            .map(|(binding, meta)| declare_storage(phase, meta, binding == output_index))
            .collect::<Vec<_>>();

        phase.program_grid::<BLOCK>(dispatch_size, |program| {
            let lane = program.lane();
            let group = linear_group(program, dispatch_size);
            let flat_index = group * BLOCK as u32 + lane.clone();
            let in_bounds = flat_index.lt(total_elements);
            let dims = output_dims_from_flat(flat_index.clone(), &operation.shape);
            let (value, value_ty) = eval_nary_expr(
                program,
                &operation.expression,
                &dims,
                &storages,
                &tensor_metas,
                in_bounds.clone(),
            );
            let value = value.cast_to(operation.output_datatype);
            debug_assert_eq!(value_ty, operation.output_datatype);
            let output_index_value = layout_index(&tensor_metas[output_index], &dims);
            storages[output_index].store(program, output_index_value, value, in_bounds);
        });
    }))
}

fn eval_nary_expr(
    program: &mut tile_ir::tile::TileBlock<'_>,
    expr: &NaryExpr,
    dims: &[tile_ir::tile::Tile<tile_ir::U32>],
    storages: &[Storage2],
    metas: &[TensorMeta],
    mask: tile_ir::tile::Mask,
) -> (ValueTile, DataTypeEnum) {
    match expr {
        NaryExpr::Op { children, function } => {
            let mut values = children
                .iter()
                .zip(&function.input_types)
                .map(|(child, expected)| {
                    let (value, ty) =
                        eval_nary_expr(program, child, dims, storages, metas, mask.clone());
                    (value.cast_to(*expected), ty)
                })
                .collect::<Vec<_>>();
            (emit_function(function, &mut values), function.output_type)
        }
        NaryExpr::IndexedInput { input_idx, indices } => {
            let meta = &metas[*input_idx];
            let coords = indices
                .iter()
                .map(|index| {
                    let (value, _) =
                        eval_nary_expr(program, index, dims, storages, metas, mask.clone());
                    match value.cast_to(DataTypeEnum::U32) {
                        ValueTile::U32(value) => value,
                        _ => unreachable!(),
                    }
                })
                .collect::<Vec<_>>();
            let index = layout_index(meta, &coords);
            let value = storages[*input_idx].load(program, index, mask);
            (value, meta.datatype)
        }
        NaryExpr::DimIndex(dim) => (ValueTile::U32(dims[*dim].clone()), DataTypeEnum::U32),
        NaryExpr::Scalar(value) => (tile_literal(*value), value.datatype()),
    }
}

fn emit_function(function: &NaryFunction, values: &mut [(ValueTile, DataTypeEnum)]) -> ValueTile {
    match function.op {
        NaryOp::Add => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Add, values[1].0.clone()),
        NaryOp::Sub => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Sub, values[1].0.clone()),
        NaryOp::Mul => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Mul, values[1].0.clone()),
        NaryOp::Div => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Div, values[1].0.clone()),
        NaryOp::Rem => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Rem, values[1].0.clone()),
        NaryOp::Pow => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Pow, values[1].0.clone()),
        NaryOp::Min => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Min, values[1].0.clone()),
        NaryOp::Max => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Max, values[1].0.clone()),
        NaryOp::Neg => values[0].0.clone().unary(tile_ir::TileUnaryOp::Neg),
        NaryOp::Cast => values[0].0.clone().cast_to(function.output_type),
        NaryOp::Select => match values[1].0.clone().cast_to(function.output_type) {
            ValueTile::F32(a) => {
                if let ValueTile::F32(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::F32(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
            ValueTile::F16(a) => {
                if let ValueTile::F16(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::F16(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
            ValueTile::U32(a) => {
                if let ValueTile::U32(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::U32(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
            ValueTile::Bool(a) => {
                if let ValueTile::Bool(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::Bool(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
        },
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
        NaryOp::AddConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Add,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::SubConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Sub,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::RSubConst(scalar) => tile_literal(scalar)
            .cast_to(values[0].1)
            .binary(tile_ir::TileBinaryOp::Sub, values[0].0.clone()),
        NaryOp::MulConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Mul,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::DivConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Div,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::RDivConst(scalar) => tile_literal(scalar)
            .cast_to(values[0].1)
            .binary(tile_ir::TileBinaryOp::Div, values[0].0.clone()),
        NaryOp::RemConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Rem,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::RRemConst(scalar) => tile_literal(scalar)
            .cast_to(values[0].1)
            .binary(tile_ir::TileBinaryOp::Rem, values[0].0.clone()),
        NaryOp::PowConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Pow,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::MinConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Min,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::MaxConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Max,
            tile_literal(scalar).cast_to(values[0].1),
        ),
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

pub(crate) fn eval_nary_expr_on_tiles(
    expr: &NaryExpr,
    inputs: &[(tile_ir::tile::Tile, DataTypeEnum)],
) -> (tile_ir::tile::Tile, DataTypeEnum) {
    let inputs = inputs
        .iter()
        .map(|(tile, dtype)| (ValueTile::F32(tile.clone()).cast_to(*dtype), *dtype))
        .collect::<Vec<_>>();
    let (value, dtype) = eval_nary_expr_on_value_tiles(expr, &inputs);
    (value.into_f32(), dtype)
}

fn eval_nary_expr_on_value_tiles(
    expr: &NaryExpr,
    inputs: &[(ValueTile, DataTypeEnum)],
) -> (ValueTile, DataTypeEnum) {
    match expr {
        NaryExpr::Op { children, function } => {
            let mut values = children
                .iter()
                .zip(&function.input_types)
                .map(|(child, expected)| {
                    let (value, ty) = eval_nary_expr_on_value_tiles(child, inputs);
                    (value.cast_to(*expected), ty)
                })
                .collect::<Vec<_>>();
            (emit_function(function, &mut values), function.output_type)
        }
        NaryExpr::IndexedInput { input_idx, .. } => inputs[*input_idx].clone(),
        NaryExpr::Scalar(value) => (tile_literal(*value), value.datatype()),
        NaryExpr::DimIndex(_) => {
            panic!("eval_nary_expr_on_tiles called with a DimIndex leaf — not supported");
        }
    }
}

pub(crate) fn apply_unary_function_chain(
    value: tile_ir::tile::Tile,
    value_ty: DataTypeEnum,
    chain: &UnaryFunctionChain,
) -> Option<(tile_ir::tile::Tile, DataTypeEnum)> {
    if chain.input_datatype() != value_ty {
        return None;
    }

    let mut value = ValueTile::F32(value).cast_to(value_ty);
    let mut value_ty = value_ty;
    for function in &chain.functions {
        if function.input_types.as_slice() != [value_ty] {
            return None;
        }
        let mut values = [(value, value_ty)];
        value = emit_function(function, &mut values);
        value_ty = function.output_type;
    }
    Some((value.into_f32(), value_ty))
}

fn tanh_exact(value: ValueTile) -> ValueTile {
    let exp_pos = value.clone().unary(tile_ir::TileUnaryOp::Exp);
    let exp_neg = value
        .unary(tile_ir::TileUnaryOp::Neg)
        .unary(tile_ir::TileUnaryOp::Exp);
    exp_pos
        .clone()
        .binary(tile_ir::TileBinaryOp::Sub, exp_neg.clone())
        .binary(
            tile_ir::TileBinaryOp::Div,
            exp_pos.binary(tile_ir::TileBinaryOp::Add, exp_neg),
        )
}

fn compare_tile(
    op: tile_ir::TileCompareOp,
    left: &(ValueTile, DataTypeEnum),
    right: &(ValueTile, DataTypeEnum),
    output: DataTypeEnum,
) -> ValueTile {
    left.0
        .clone()
        .compare(op, right.0.clone().cast_to(left.1), output)
}

fn compare_const(
    op: tile_ir::TileCompareOp,
    left: &(ValueTile, DataTypeEnum),
    scalar: NaryScalar,
    output: DataTypeEnum,
) -> ValueTile {
    left.0
        .clone()
        .compare(op, tile_literal(scalar).cast_to(left.1), output)
}

pub(crate) fn output_dims_from_flat(
    flat: tile_ir::tile::Tile<tile_ir::U32>,
    shape: &[usize],
) -> Vec<tile_ir::tile::Tile<tile_ir::U32>> {
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

pub(crate) fn layout_index(
    meta: &TensorMeta,
    coords: &[tile_ir::tile::Tile<tile_ir::U32>],
) -> tile_ir::tile::Tile<tile_ir::U32> {
    let mut index = tile_u32(meta.offset);
    for (axis, (coord, stride)) in coords.iter().zip(&meta.strides).enumerate() {
        if *stride == 0 || meta.shape.get(axis).copied() == Some(1) {
            continue;
        }
        index = index + coord.clone() * tile_u32(*stride);
    }
    index
}

pub(crate) fn linear_group(
    program: &tile_ir::tile::TileBlock<'_>,
    dispatch_size: [u32; 3],
) -> tile_ir::tile::Tile<tile_ir::U32> {
    program.program_id(tile_ir::WorkgroupAxis::X)
        + program.program_id(tile_ir::WorkgroupAxis::Y) * dispatch_size[0]
        + program.program_id(tile_ir::WorkgroupAxis::Z)
            * dispatch_size[0].saturating_mul(dispatch_size[1])
}

pub(crate) fn tile_literal(value: NaryScalar) -> ValueTile {
    match value {
        NaryScalar::F32(value) => ValueTile::F32(tile_ir::tile::Tile::literal(
            tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(value)),
        )),
        NaryScalar::F16(value) => ValueTile::F16(tile_ir::tile::Tile::literal(
            tile_ir::TileLiteral::F16(value.to_bits()),
        )),
        NaryScalar::U32(value) => ValueTile::U32(tile_ir::tile::Tile::literal(
            tile_ir::TileLiteral::U32(value),
        )),
    }
}

pub(crate) fn tile_u32(value: u32) -> tile_ir::tile::Tile<tile_ir::U32> {
    tile_ir::tile::Tile::literal(tile_ir::TileLiteral::U32(value))
}

pub(crate) fn zero_literal(value: DataTypeEnum) -> tile_ir::TileLiteral {
    match value {
        DataTypeEnum::F32 => tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(0.0)),
        DataTypeEnum::F16 => tile_ir::TileLiteral::F16(half::f16::from_f32(0.0).to_bits()),
        DataTypeEnum::U32 => tile_ir::TileLiteral::U32(0),
    }
}

#[derive(Clone)]
pub(crate) struct TensorMeta {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) shape: Vec<u32>,
    pub(crate) strides: Vec<u32>,
    pub(crate) offset: u32,
    pub(crate) allocation_len: u32,
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
