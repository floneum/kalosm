use tensor_ir::{
    BinaryOp, DType, Dim as IrDim, ExprId, ReduceOp, Shape, Strides, TensorExprBuilder, TernaryOp,
    UnaryOp,
};

use crate::{
    DataTypeEnum, Layout, MatMulOperation, ReduceFunction, ReduceOperation, TensorData,
    composite::SoftmaxOperation,
    mir::{inputs::MirValue, operation::TensorIrLowering},
    nary_wise::{NaryExpr, NaryFunction, NaryFunctionKind, NaryOperation},
};

fn dtype(ty: DataTypeEnum) -> Result<DType, String> {
    match ty {
        DataTypeEnum::F32 => Ok(DType::F32),
        DataTypeEnum::F16 => Err("tensor_ir lowering does not support f16 yet".to_string()),
        DataTypeEnum::U32 => Err("tensor_ir lowering does not support u32 tensors yet".to_string()),
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

fn add_tensor_input(
    builder: &mut TensorExprBuilder,
    input_id: u32,
    tensor: &TensorData,
    index_space: &Shape,
) -> Result<ExprId, String> {
    let input = builder.input(
        input_id,
        shape(tensor.layout().shape())?,
        dtype(tensor.datatype())?,
    );
    let index_dims = index_space
        .0
        .iter()
        .map(|dim| match dim {
            IrDim::Lit(value) => *value as usize,
            IrDim::Sym(_) => 0,
        })
        .collect::<Vec<_>>();
    if tensor.layout().shape() == index_dims.as_slice() {
        let row_major = Strides::row_major_for_shape(index_space);
        let tensor_strides = strides(tensor.layout())?;
        if row_major.as_ref() == Some(&tensor_strides) {
            return Ok(input);
        }
    }
    Ok(builder.restride_with_offset(
        input,
        index_space.clone(),
        strides(tensor.layout())?,
        i64::try_from(tensor.layout().offset())
            .map_err(|_| format!("tensor offset {} exceeds i64", tensor.layout().offset()))?,
    ))
}

pub(crate) fn nary(op: &NaryOperation, inputs: &[MirValue]) -> Result<TensorIrLowering, String> {
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
            if !NaryExpr::is_elementwise_indices(indices) {
                return Err("tensor_ir lowering does not support custom nary indexing yet".into());
            }
            Ok(builder.scalar_arg(*input_idx as u32))
        }
        NaryExpr::DimIndex(_) => {
            Err("tensor_ir lowering does not support dimension-index scalar expressions yet".into())
        }
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
            let constant = add_f32_splat_input(
                builder,
                tensors,
                ir_inputs,
                index_space,
                output_shape,
                constant,
            )?;
            lower_const_binop(builder, *op, *a, constant, *input_first)
        }
        (NaryFunctionKind::CompareConst { op, constant }, [a]) => {
            let constant = add_f32_splat_input(
                builder,
                tensors,
                ir_inputs,
                index_space,
                output_shape,
                constant,
            )?;
            lower_const_compare(builder, *op, *a, constant)
        }
        (NaryFunctionKind::Cast(output_type), [a]) => {
            Ok(builder.scalar_unop(cast_op(*output_type)?, *a))
        }
        (NaryFunctionKind::Unsupported, _) => Err(format!(
            "tensor_ir lowering does not support nary function {} yet",
            function.name()
        )),
        (kind, children) => Err(format!(
            "tensor_ir lowering got invalid arity {} for nary function {} ({kind:?})",
            children.len(),
            function.name()
        )),
    }
}

fn add_f32_splat_input(
    builder: &mut TensorExprBuilder,
    tensors: &mut Vec<TensorData>,
    ir_inputs: &mut Vec<ExprId>,
    index_space: &Shape,
    output_shape: &[usize],
    constant: &str,
) -> Result<ExprId, String> {
    let value = parse_f32_literal(constant)?;
    let device = tensors
        .first()
        .ok_or_else(|| "constant nary op requires at least one tensor input".to_string())?
        .device()
        .clone();
    let tensor = TensorData::new_splat(&device, output_shape, value);
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
    let input_id = builder.input(0, shape(input.layout().shape())?, dtype(input.datatype())?);
    let input_shape = shape(input.layout().shape())?;
    let row_major = Strides::row_major_for_shape(&input_shape);
    let input_strides = strides(input.layout())?;
    let input_id = if input.layout().offset() == 0 && row_major.as_ref() == Some(&input_strides) {
        input_id
    } else {
        builder.restride_with_offset(
            input_id,
            input_shape,
            input_strides,
            i64::try_from(input.layout().offset())
                .map_err(|_| format!("tensor offset {} exceeds i64", input.layout().offset()))?,
        )
    };
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
    if op.datatype != DataTypeEnum::F32 {
        return Err(format!(
            "tensor_ir matmul only supports f32, got {}",
            op.datatype
        ));
    }
    if !op.pre_element_wise[0].functions.is_empty()
        || !op.pre_element_wise[1].functions.is_empty()
        || !op.post_element_wise.functions.is_empty()
    {
        return Err("tensor_ir matmul lowering does not support legacy fused unary chains".into());
    }

    let lhs = &tensors[0];
    let rhs = &tensors[1];
    let out_shape = op.out_shape.clone();
    let rank = out_shape.len();
    let m = out_shape[rank - 2];
    let n = out_shape[rank - 1];
    let k = op.first_shape[rank - 1];

    let mut index_dims = out_shape[..rank - 2].to_vec();
    index_dims.extend([m, n, k]);
    let index_space = shape(&index_dims)?;

    let mut builder = TensorExprBuilder::new();
    let lhs_id = builder.input(0, shape(lhs.layout().shape())?, DType::F32);
    let rhs_id = builder.input(1, shape(rhs.layout().shape())?, DType::F32);

    let lhs_strides = matmul_lhs_strides(lhs.layout())?;
    let rhs_strides = matmul_rhs_strides(rhs.layout())?;
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

fn matmul_lhs_strides(layout: &Layout) -> Result<Vec<i64>, String> {
    let rank = layout.rank();
    let mut result = Vec::with_capacity(rank + 1);
    for axis in 0..rank - 2 {
        result.push(i64::try_from(layout.strides()[axis]).map_err(|e| e.to_string())?);
    }
    result.push(i64::try_from(layout.strides()[rank - 2]).map_err(|e| e.to_string())?);
    result.push(0);
    result.push(i64::try_from(layout.strides()[rank - 1]).map_err(|e| e.to_string())?);
    Ok(result)
}

fn matmul_rhs_strides(layout: &Layout) -> Result<Vec<i64>, String> {
    let rank = layout.rank();
    let mut result = Vec::with_capacity(rank + 1);
    for axis in 0..rank - 2 {
        result.push(i64::try_from(layout.strides()[axis]).map_err(|e| e.to_string())?);
    }
    result.push(0);
    result.push(i64::try_from(layout.strides()[rank - 1]).map_err(|e| e.to_string())?);
    result.push(i64::try_from(layout.strides()[rank - 2]).map_err(|e| e.to_string())?);
    Ok(result)
}
