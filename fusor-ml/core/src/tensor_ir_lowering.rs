use tensor_ir::{
    BinaryOp, DType, Dim as IrDim, ExprId, ReduceOp, Shape, Strides, TensorExprBuilder, TernaryOp,
    UnaryOp,
};

use crate::{
    DataTypeEnum, Layout, MatMulOperation, ReduceFunction, ReduceOperation, TensorData,
    composite::SoftmaxOperation,
    mir::{inputs::MirValue, operation::TensorIrLowering},
    nary_wise::{NaryExpr, NaryFunction, NaryFunctionKind, NaryOperation},
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
};

fn dtype(ty: DataTypeEnum) -> Result<DType, String> {
    match ty {
        DataTypeEnum::F32 => Ok(DType::F32),
        DataTypeEnum::F16 => Ok(DType::F16),
        DataTypeEnum::U32 => Ok(DType::U32),
    }
}

fn shape(dims: &[usize]) -> Result<Shape, String> {
    dims.iter()
        .map(|dim| {
            u32::try_from(*dim)
                .map(IrDim::Lit)
                .map_err(|_| format!("tensor dimension {dim} exceeds u32"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Shape)
}

fn strides(layout: &Layout) -> Result<Strides, String> {
    layout
        .strides()
        .iter()
        .map(|stride| {
            i64::try_from(*stride).map_err(|_| format!("tensor stride {stride} exceeds i64"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Strides)
}

fn tensor_inputs(inputs: &[MirValue], count: usize) -> Result<Vec<TensorData>, String> {
    inputs
        .iter()
        .take(count)
        .map(|input| match input {
            MirValue::Tensor(tensor) => Ok(tensor.clone()),
            other => Err(format!("tensor_ir expected tensor input, got {other:?}")),
        })
        .collect()
}

fn storage_shape(tensor: &TensorData) -> Result<Shape, String> {
    let elements = tensor.buffer().size()
        / u64::try_from(tensor.datatype().element_size()).map_err(|e| e.to_string())?;
    let elements = usize::try_from(elements)
        .map_err(|_| format!("tensor buffer length {elements} exceeds usize"))?;
    shape(&[elements])
}

fn input_shape_for_restrided_view(tensor: &TensorData) -> Result<Shape, String> {
    let logical_shape = shape(tensor.layout().shape())?;
    let row_major = Strides::row_major_for_shape(&logical_shape);
    let tensor_strides = strides(tensor.layout())?;
    if tensor.layout().offset() == 0 && row_major.as_ref() == Some(&tensor_strides) {
        Ok(logical_shape)
    } else {
        storage_shape(tensor)
    }
}

fn add_tensor_input(
    builder: &mut TensorExprBuilder,
    input_id: u32,
    tensor: &TensorData,
    index_space: &Shape,
) -> Result<ExprId, String> {
    let index_dims = index_space
        .0
        .iter()
        .map(|dim| match dim {
            IrDim::Lit(value) => *value as usize,
            IrDim::Sym(_) => 0,
        })
        .collect::<Vec<_>>();
    if tensor.layout().offset() == 0 && tensor.layout().shape() == index_dims.as_slice() {
        let row_major = Strides::row_major_for_shape(index_space);
        let tensor_strides = strides(tensor.layout())?;
        if row_major.as_ref() == Some(&tensor_strides) {
            let input = builder.input(
                input_id,
                shape(tensor.layout().shape())?,
                dtype(tensor.datatype())?,
            );
            return Ok(input);
        }
    }
    let input = builder.input(input_id, storage_shape(tensor)?, dtype(tensor.datatype())?);
    Ok(builder.restride_with_offset(
        input,
        index_space.clone(),
        strides(tensor.layout())?,
        i64::try_from(tensor.layout().offset())
            .map_err(|_| format!("tensor offset {} exceeds i64", tensor.layout().offset()))?,
    ))
}

pub(crate) fn nary(op: &NaryOperation, inputs: &[MirValue]) -> Result<TensorIrLowering, String> {
    if let Some(lowered) = try_index_select_nary(op, inputs)? {
        return Ok(lowered);
    }

    let mut tensors = tensor_inputs(inputs, op.inputs.len())?;
    let index_space = shape(&op.shape)?;
    let mut builder = TensorExprBuilder::new();
    let mut ir_inputs = tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| add_tensor_input(&mut builder, index as u32, tensor, &index_space))
        .collect::<Result<Vec<_>, _>>()?;
    let body = lower_nary_expr(
        &mut builder,
        &op.expression,
        &mut tensors,
        &mut ir_inputs,
        &index_space,
        &op.shape,
    )?;
    let root = builder.elementwise(index_space, &ir_inputs, body);
    let program = builder.build(root)?;

    Ok(TensorIrLowering {
        program,
        inputs: tensors,
        output_shape: op.shape.clone(),
        output_datatype: op.output_datatype,
    })
}

fn try_index_select_nary(
    op: &NaryOperation,
    inputs: &[MirValue],
) -> Result<Option<TensorIrLowering>, String> {
    let NaryExpr::IndexedInput { input_idx, indices } = &op.expression else {
        return Ok(None);
    };
    if *input_idx != 0 || op.inputs.len() < 2 {
        return Ok(None);
    }

    let mut select_axis = None;
    for (axis, index) in indices.iter().enumerate() {
        match index {
            NaryExpr::DimIndex(dim) if *dim == axis => {}
            NaryExpr::IndexedInput {
                input_idx: 1,
                indices: index_indices,
            } if index_indices.len() == 1 => {
                let NaryExpr::DimIndex(dim) = index_indices[0] else {
                    return Ok(None);
                };
                if dim != axis || select_axis.replace(axis).is_some() {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }
    }

    let Some(axis) = select_axis else {
        return Ok(None);
    };
    let tensors = tensor_inputs(inputs, 2)?;
    let input = &tensors[0];
    let indices = &tensors[1];
    if indices.datatype() != DataTypeEnum::U32 {
        return Err(format!(
            "tensor_ir index_select expected u32 indices, got {}",
            indices.datatype()
        ));
    }
    if indices.layout().rank() != 1 {
        return Err(format!(
            "tensor_ir index_select expected rank-1 indices, got rank {}",
            indices.layout().rank()
        ));
    }
    if input.layout().rank() != op.shape.len() {
        return Err(format!(
            "tensor_ir index_select input rank {} does not match output rank {}",
            input.layout().rank(),
            op.shape.len()
        ));
    }

    let mut builder = TensorExprBuilder::new();
    let input_shape = shape(input.layout().shape())?;
    let indices_shape = shape(indices.layout().shape())?;
    let input_id = add_tensor_input(&mut builder, 0, input, &input_shape)?;
    let indices_id = add_tensor_input(&mut builder, 1, indices, &indices_shape)?;
    let output_shape = shape(&op.shape)?;
    let root = builder.index_select(
        input_id,
        indices_id,
        output_shape,
        u32::try_from(axis).map_err(|_| format!("index_select axis {axis} exceeds u32"))?,
    );
    let program = builder.build(root)?;

    Ok(Some(TensorIrLowering {
        program,
        inputs: tensors,
        output_shape: op.shape.clone(),
        output_datatype: op.output_datatype,
    }))
}

fn lower_nary_expr(
    builder: &mut TensorExprBuilder,
    expr: &NaryExpr,
    tensors: &mut Vec<TensorData>,
    ir_inputs: &mut Vec<ExprId>,
    index_space: &Shape,
    output_shape: &[usize],
) -> Result<ExprId, String> {
    match expr {
        NaryExpr::IndexedInput { input_idx, indices } => {
            if NaryExpr::is_elementwise_indices(indices) {
                return Ok(builder.scalar_arg(*input_idx as u32));
            }
            let indices = indices
                .iter()
                .map(|index| {
                    lower_nary_expr(
                        builder,
                        index,
                        tensors,
                        ir_inputs,
                        index_space,
                        output_shape,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(builder.indexed_arg(*input_idx as u32, indices))
        }
        NaryExpr::DimIndex(dim) => Ok(builder.scalar_index(
            u32::try_from(*dim).map_err(|_| format!("dimension index {dim} exceeds u32"))?,
        )),
        NaryExpr::Op { children, function } => {
            let children = children
                .iter()
                .map(|child| {
                    lower_nary_expr(
                        builder,
                        child,
                        tensors,
                        ir_inputs,
                        index_space,
                        output_shape,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            lower_nary_function(
                builder,
                function,
                &children,
                tensors,
                ir_inputs,
                index_space,
                output_shape,
            )
        }
    }
}

fn lower_nary_function(
    builder: &mut TensorExprBuilder,
    function: &NaryFunction,
    children: &[ExprId],
    tensors: &mut Vec<TensorData>,
    ir_inputs: &mut Vec<ExprId>,
    index_space: &Shape,
    output_shape: &[usize],
) -> Result<ExprId, String> {
    match (&function.kind, children) {
        (NaryFunctionKind::Unary(op), [a]) => Ok(builder.scalar_unop(*op, *a)),
        (NaryFunctionKind::Binary(op), [a, b]) => Ok(builder.scalar_binop(*op, [*a, *b])),
        (NaryFunctionKind::Select { condition_type }, [condition, on_true, on_false]) => {
            let zero = match condition_type {
                DataTypeEnum::F32 | DataTypeEnum::F16 => builder.scalar_f32(0.0),
                DataTypeEnum::U32 => builder.scalar_u32(0),
            };
            let predicate = builder.scalar_binop(BinaryOp::Neq, [*condition, zero]);
            Ok(builder.scalar_ternop(TernaryOp::Select, [predicate, *on_true, *on_false]))
        }
        (
            NaryFunctionKind::BinaryConst {
                op,
                constant,
                input_first,
            },
            [a],
        ) => {
            let constant = add_splat_input(
                builder,
                tensors,
                ir_inputs,
                index_space,
                output_shape,
                constant,
                function.output_type,
            )?;
            lower_const_binop(builder, *op, *a, constant, *input_first)
        }
        (NaryFunctionKind::CompareConst { op, constant }, [a]) => {
            let constant = add_splat_input(
                builder,
                tensors,
                ir_inputs,
                index_space,
                output_shape,
                constant,
                function.output_type,
            )?;
            lower_const_compare(builder, *op, *a, constant)
        }
        (NaryFunctionKind::Cast(output_type), [a]) => {
            Ok(builder.scalar_unop(cast_op(*output_type)?, *a))
        }
        (NaryFunctionKind::Unsupported, [a]) => {
            lower_named_index_function(builder, function.name(), *a, output_shape)
        }
        (kind, children) => Err(format!(
            "tensor_ir lowering got invalid arity {} for nary function {} ({kind:?})",
            children.len(),
            function.name()
        )),
    }
}

fn lower_named_index_function(
    builder: &mut TensorExprBuilder,
    name: &str,
    input: ExprId,
    output_shape: &[usize],
) -> Result<ExprId, String> {
    let last_dim = output_shape
        .last()
        .copied()
        .ok_or_else(|| "index helper requires a non-scalar output shape".to_string())?;
    let half = u32::try_from(last_dim / 2)
        .map_err(|_| format!("last dimension {last_dim} exceeds u32"))?;
    let one = builder.scalar_u32(1);
    let two = builder.scalar_u32(2);
    let half_lit = builder.scalar_u32(half);

    Ok(match name {
        "div2" => builder.scalar_binop(BinaryOp::Div, [input, two]),
        "mod2" => builder.scalar_binop(BinaryOp::Mod, [input, two]),
        "mod_half" => builder.scalar_binop(BinaryOp::Mod, [input, half_lit]),
        "div_half" => builder.scalar_binop(BinaryOp::Div, [input, half_lit]),
        "neighbor_interleaved_idx" => {
            let parity = builder.scalar_binop(BinaryOp::Mod, [input, two]);
            let zero = builder.scalar_u32(0);
            let is_odd = builder.scalar_binop(BinaryOp::Neq, [parity, zero]);
            let prev = builder.scalar_binop(BinaryOp::Sub, [input, one]);
            let next = builder.scalar_binop(BinaryOp::Add, [input, one]);
            builder.scalar_ternop(TernaryOp::Select, [is_odd, prev, next])
        }
        "neighbor_half_idx" => {
            let side = builder.scalar_binop(BinaryOp::Div, [input, half_lit]);
            let zero = builder.scalar_u32(0);
            let is_upper_half = builder.scalar_binop(BinaryOp::Neq, [side, zero]);
            let prev = builder.scalar_binop(BinaryOp::Sub, [input, half_lit]);
            let next = builder.scalar_binop(BinaryOp::Add, [input, half_lit]);
            builder.scalar_ternop(TernaryOp::Select, [is_upper_half, prev, next])
        }
        _ => {
            return Err(format!(
                "tensor_ir lowering does not support nary function {name} yet"
            ));
        }
    })
}

fn add_splat_input(
    builder: &mut TensorExprBuilder,
    tensors: &mut Vec<TensorData>,
    ir_inputs: &mut Vec<ExprId>,
    index_space: &Shape,
    output_shape: &[usize],
    constant: &str,
    datatype: DataTypeEnum,
) -> Result<ExprId, String> {
    let value = parse_f32_literal(constant)?;
    let device = tensors
        .first()
        .ok_or_else(|| "constant nary op requires at least one tensor input".to_string())?
        .device()
        .clone();
    let tensor = match datatype {
        DataTypeEnum::F32 => TensorData::new_splat(&device, output_shape, value),
        DataTypeEnum::F16 => {
            TensorData::new_splat(&device, output_shape, half::f16::from_f32(value))
        }
        DataTypeEnum::U32 => TensorData::new_splat(&device, output_shape, value as u32),
    };
    let input_id = u32::try_from(tensors.len()).map_err(|_| "too many tensor_ir inputs")?;
    let ir_input = add_tensor_input(builder, input_id, &tensor, index_space)?;
    tensors.push(tensor);
    ir_inputs.push(ir_input);
    Ok(builder.scalar_arg(input_id))
}

fn cast_op(output_type: DataTypeEnum) -> Result<UnaryOp, String> {
    match output_type {
        DataTypeEnum::F32 => Ok(UnaryOp::CastF32),
        DataTypeEnum::F16 => Ok(UnaryOp::CastF16),
        DataTypeEnum::U32 => Ok(UnaryOp::CastU32),
    }
}

fn lower_const_binop(
    builder: &mut TensorExprBuilder,
    op: BinaryOp,
    input: ExprId,
    constant: ExprId,
    input_first: bool,
) -> Result<ExprId, String> {
    let args = if input_first {
        [input, constant]
    } else {
        [constant, input]
    };
    Ok(builder.scalar_binop(op, args))
}

fn lower_const_compare(
    builder: &mut TensorExprBuilder,
    op: BinaryOp,
    input: ExprId,
    constant: ExprId,
) -> Result<ExprId, String> {
    let compared = builder.scalar_binop(op, [input, constant]);
    Ok(builder.scalar_unop(UnaryOp::CastF32, compared))
}

fn parse_f32_literal(source: &str) -> Result<f32, String> {
    source
        .trim()
        .trim_end_matches('f')
        .parse::<f32>()
        .map_err(|e| format!("could not parse f32 literal {source:?}: {e}"))
}

pub(crate) fn reduce(
    op: &ReduceOperation,
    inputs: &[MirValue],
) -> Result<TensorIrLowering, String> {
    let input = reduce_input_view(op, inputs)?;
    let mut builder = TensorExprBuilder::new();
    let input_shape = shape(input.layout().shape())?;
    let input_id = add_tensor_input(&mut builder, 0, &input, &input_shape)?;
    if !op.pre_element_wise.functions.is_empty() || !op.post_element_wise.functions.is_empty() {
        return Err("tensor_ir reduce lowering does not support legacy fused unary chains".into());
    }
    let root = builder.reduce(input_id, op.axis as u32, reduce_op(&op.function)?);
    let program = builder.build(root)?;
    let output_shape: Box<[usize]> = op
        .shape
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| (axis != op.axis).then_some(*dim))
        .collect();

    Ok(TensorIrLowering {
        program,
        inputs: vec![input],
        output_shape,
        output_datatype: op.out_datatype(),
    })
}

fn reduce_input_view(op: &ReduceOperation, inputs: &[MirValue]) -> Result<TensorData, String> {
    let [
        MirValue::Tensor(trimmed),
        _,
        MirValue::Integer(reduction_len),
        MirValue::Integer(reduction_stride),
    ] = inputs
    else {
        return Err(
            "tensor_ir reduce expected [tensor, output, reduction_len, reduction_stride]"
                .to_string(),
        );
    };
    if op.axis > trimmed.layout().rank() {
        return Err(format!(
            "reduce axis {} is out of bounds for trimmed rank {}",
            op.axis,
            trimmed.layout().rank()
        ));
    }

    let mut full_shape = trimmed.layout().shape().to_vec();
    full_shape.insert(
        op.axis,
        usize::try_from(*reduction_len).map_err(|e| e.to_string())?,
    );
    let mut full_strides = trimmed.layout().strides().to_vec();
    full_strides.insert(
        op.axis,
        usize::try_from(*reduction_stride).map_err(|e| e.to_string())?,
    );
    let layout = Layout::from_parts(
        trimmed.layout().offset(),
        full_shape.into(),
        full_strides.into(),
    );
    Ok(TensorData::new_from_parts(
        trimmed.device(),
        trimmed.buffer().clone(),
        layout,
        trimmed.datatype(),
    ))
}

pub(crate) fn softmax(
    op: &SoftmaxOperation,
    inputs: &[MirValue],
) -> Result<TensorIrLowering, String> {
    let tensors = tensor_inputs(inputs, 1)?;
    let input = &tensors[0];
    let input_shape = shape(&op.shape)?;
    let mut builder = TensorExprBuilder::new();
    let input_id = add_tensor_input(&mut builder, 0, input, &input_shape)?;
    let root = builder.softmax(input_id, input_shape, op.axis as u32);
    let program = builder.build(root)?;

    Ok(TensorIrLowering {
        program,
        inputs: tensors,
        output_shape: op.shape.clone(),
        output_datatype: op.out_datatype(),
    })
}

fn reduce_op(function: &ReduceFunction) -> Result<ReduceOp, String> {
    Ok(function.op)
}

pub(crate) fn matmul(
    op: &MatMulOperation,
    inputs: &[MirValue],
) -> Result<TensorIrLowering, String> {
    let tensors = tensor_inputs(inputs, 2)?;
    if !op.pre_element_wise[0].functions.is_empty()
        || !op.pre_element_wise[1].functions.is_empty()
        || !op.post_element_wise.functions.is_empty()
    {
        return Err("tensor_ir matmul lowering does not support legacy fused unary chains".into());
    }

    let lhs = &tensors[0];
    let rhs = &tensors[1];
    let out_shape = op.out_shape.clone();
    let ir_dtype = dtype(op.datatype)?;
    let rank = out_shape.len();
    let m = out_shape[rank - 2];
    let n = out_shape[rank - 1];
    let k = op.first_shape[rank - 1];

    let mut index_dims = out_shape[..rank - 2].to_vec();
    index_dims.extend([m, n, k]);
    let index_space = shape(&index_dims)?;

    let mut builder = TensorExprBuilder::new();
    let lhs_id = builder.input(0, input_shape_for_restrided_view(lhs)?, ir_dtype);
    let rhs_id = builder.input(1, input_shape_for_restrided_view(rhs)?, ir_dtype);

    let lhs_strides = matmul_lhs_strides(lhs.layout(), &out_shape)?;
    let rhs_strides = matmul_rhs_strides(rhs.layout(), &out_shape)?;
    let lhs_id = builder.restride_with_offset(
        lhs_id,
        index_space.clone(),
        Strides(lhs_strides),
        i64::try_from(lhs.layout().offset())
            .map_err(|_| format!("tensor offset {} exceeds i64", lhs.layout().offset()))?,
    );
    let rhs_id = builder.restride_with_offset(
        rhs_id,
        index_space.clone(),
        Strides(rhs_strides),
        i64::try_from(rhs.layout().offset())
            .map_err(|_| format!("tensor offset {} exceeds i64", rhs.layout().offset()))?,
    );
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let root = builder.elementwise(index_space, &[lhs_id, rhs_id], body);
    let root = builder.reduce(root, rank as u32, ReduceOp::Add);
    let program = builder.build(root)?;

    Ok(TensorIrLowering {
        program,
        inputs: tensors,
        output_shape: out_shape,
        output_datatype: op.post_element_wise.out_datatype(),
    })
}

pub(crate) fn slice_assign(
    op: &SliceAssignOperation,
    inputs: &[MirValue],
) -> Result<TensorIrLowering, String> {
    let [MirValue::Tensor(input), MirValue::Tensor(value), _] = inputs else {
        return Err("tensor_ir slice_assign expected [input, value, output]".to_string());
    };
    if input.datatype() != value.datatype() {
        return Err(format!(
            "slice_assign input datatype {} does not match value datatype {}",
            input.datatype(),
            value.datatype()
        ));
    }
    if op.slices.len() != input.layout().rank() {
        return Err(format!(
            "slice_assign got {} slices for rank-{} input",
            op.slices.len(),
            input.layout().rank()
        ));
    }
    if value.layout().rank() != op.slices.len() {
        return Err(format!(
            "slice_assign value rank {} does not match slice rank {}",
            value.layout().rank(),
            op.slices.len()
        ));
    }
    for (axis, (slice, value_dim)) in op
        .slices
        .iter()
        .zip(value.layout().shape().iter())
        .enumerate()
    {
        let input_dim = input.layout().shape()[axis];
        if slice.start > slice.end || slice.end > input_dim {
            return Err(format!(
                "slice_assign slice {:?} is invalid for axis {axis} with dim {input_dim}",
                slice
            ));
        }
        let slice_len = slice.end - slice.start;
        if slice_len != *value_dim {
            return Err(format!(
                "slice_assign value axis {axis} has dim {value_dim}, expected {slice_len}"
            ));
        }
    }

    let output_shape = shape(input.layout().shape())?;
    let mut builder = TensorExprBuilder::new();
    let input_id = add_tensor_input(&mut builder, 0, input, &output_shape)?;
    let value_shape = shape(value.layout().shape())?;
    let value_id = add_tensor_input(&mut builder, 1, value, &value_shape)?;
    let slices = op
        .slices
        .iter()
        .map(|slice| {
            let start = u32::try_from(slice.start)
                .map_err(|_| format!("slice start {} exceeds u32", slice.start))?;
            let end = u32::try_from(slice.end)
                .map_err(|_| format!("slice end {} exceeds u32", slice.end))?;
            Ok((start, end))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let root = builder.slice_assign(input_id, value_id, output_shape, slices);
    let program = builder.build(root)?;

    Ok(TensorIrLowering {
        program,
        inputs: vec![input.clone(), value.clone()],
        output_shape: input.layout().shape().into(),
        output_datatype: input.datatype(),
    })
}

pub(crate) fn resize(
    op: &ResizeOperation,
    inputs: &[MirValue],
) -> Result<TensorIrLowering, String> {
    let [MirValue::Tensor(input), _] = inputs else {
        return Err("tensor_ir resize expected [input, output_slice]".to_string());
    };
    let current_elems = op.current_shape.iter().product::<usize>();
    let new_elems = op.new_shape.iter().product::<usize>();
    if op.fill_shape == op.new_shape && current_elems != new_elems {
        return Err(format!(
            "tensor_ir resize cannot change element count from {current_elems} to {new_elems}"
        ));
    }
    if op.fill_shape != op.new_shape {
        if op.current_shape.len() != op.new_shape.len() || op.fill_shape.len() != op.new_shape.len()
        {
            return Err(format!(
                "tensor_ir resize expected matching ranks, got current {}, fill {}, new {}",
                op.current_shape.len(),
                op.fill_shape.len(),
                op.new_shape.len()
            ));
        }
        for (axis, ((fill, current), new)) in op
            .fill_shape
            .iter()
            .zip(op.current_shape.iter())
            .zip(op.new_shape.iter())
            .enumerate()
        {
            if *fill > *current || *fill > *new {
                return Err(format!(
                    "tensor_ir resize fill dim {fill} on axis {axis} exceeds current {current} or new {new}"
                ));
            }
        }
    }

    let input_shape = shape(&op.current_shape)?;
    let output_shape = shape(&op.new_shape)?;
    let mut builder = TensorExprBuilder::new();
    let input_id = add_tensor_input(&mut builder, 0, input, &input_shape)?;
    let root = builder.resize(input_id, input_shape, output_shape);
    let program = builder.build(root)?;

    Ok(TensorIrLowering {
        program,
        inputs: vec![input.clone()],
        output_shape: op.new_shape.clone(),
        output_datatype: input.datatype(),
    })
}

fn matmul_lhs_strides(layout: &Layout, out_shape: &[usize]) -> Result<Vec<i64>, String> {
    let rank = layout.rank();
    let mut result = Vec::with_capacity(rank + 1);
    for axis in 0..rank - 2 {
        let stride = if layout.shape()[axis] == 1 && out_shape[axis] > 1 {
            0
        } else {
            layout.strides()[axis]
        };
        result.push(i64::try_from(stride).map_err(|e| e.to_string())?);
    }
    result.push(i64::try_from(layout.strides()[rank - 2]).map_err(|e| e.to_string())?);
    result.push(0);
    result.push(i64::try_from(layout.strides()[rank - 1]).map_err(|e| e.to_string())?);
    Ok(result)
}

fn matmul_rhs_strides(layout: &Layout, out_shape: &[usize]) -> Result<Vec<i64>, String> {
    let rank = layout.rank();
    let mut result = Vec::with_capacity(rank + 1);
    for axis in 0..rank - 2 {
        let stride = if layout.shape()[axis] == 1 && out_shape[axis] > 1 {
            0
        } else {
            layout.strides()[axis]
        };
        result.push(i64::try_from(stride).map_err(|e| e.to_string())?);
    }
    result.push(0);
    result.push(i64::try_from(layout.strides()[rank - 1]).map_err(|e| e.to_string())?);
    result.push(i64::try_from(layout.strides()[rank - 2]).map_err(|e| e.to_string())?);
    Ok(result)
}
