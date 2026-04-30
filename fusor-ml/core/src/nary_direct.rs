use std::num::NonZeroU32;

use wgpu::naga::{
    AddressSpace, Arena, ArraySize, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, MathFunction, Module,
    Range as NagaRange, ResourceBinding, Scalar, ScalarKind, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner, UnaryOperator,
};

use crate::{
    TILE_SIZE,
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryOperation, NaryScalar},
    tensor::{DataTypeEnum, TensorData},
};

#[derive(Clone)]
struct TensorView {
    global: Handle<GlobalVariable>,
    datatype: DataTypeEnum,
    strides: Vec<u32>,
    offset: u32,
}

#[derive(Clone, Copy)]
struct ExprValue {
    handle: Handle<Expression>,
    datatype: DataTypeEnum,
}

struct NaryDirectBuilder<'a> {
    module: Module,
    u32_ty: Handle<Type>,
    inputs: Vec<TensorView>,
    output: TensorView,
    operation: &'a NaryOperation,
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
    let mut tensors = Vec::with_capacity(inputs.len());
    for input in inputs {
        tensors.push(input.as_tensor()?.clone());
    }
    tensors.get(output_index)?;

    if tensors
        .iter()
        .any(|tensor| tensor.datatype() == DataTypeEnum::F16 && !graph.device().f16_supported())
    {
        return None;
    }

    let dispatch_size = operation.dispatch_size(workgroup_shape, inputs);
    let cache_key = format!(
        "{}:direct:{:?}:out={output_index}:dispatch={dispatch_size:?}:expr={:?}:{}",
        operation.name(),
        workgroup_shape.shape(),
        operation.expression,
        tensors
            .iter()
            .map(|tensor| format!("{:?}:{:?}", tensor.datatype(), tensor.layout()))
            .collect::<Vec<_>>()
            .join("|")
    );
    let module = NaryDirectBuilder::new(operation, &tensors, output_index)?
        .build(*workgroup_shape, dispatch_size)?;
    let bindings = tensors
        .iter()
        .enumerate()
        .map(|(binding, tensor)| DirectKernelBinding::Storage {
            binding: binding as u32,
            buffer: tensor.buffer().clone(),
            read_only: binding != output_index,
        })
        .collect();

    Some(DirectKernel::new_with_cache_key(
        operation.name(),
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

impl<'a> NaryDirectBuilder<'a> {
    fn new(
        operation: &'a NaryOperation,
        tensors: &[TensorData],
        output_index: usize,
    ) -> Option<Self> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("f32".into()),
                inner: TypeInner::Scalar(Self::scalar(DataTypeEnum::F32)),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Self::scalar(DataTypeEnum::U32)),
            },
            Span::default(),
        );
        let f16_ty = tensors
            .iter()
            .any(|tensor| tensor.datatype() == DataTypeEnum::F16)
            .then(|| {
                module.types.insert(
                    Type {
                        name: Some("f16".into()),
                        inner: TypeInner::Scalar(Self::scalar(DataTypeEnum::F16)),
                    },
                    Span::default(),
                )
            });
        let mut views = Vec::with_capacity(tensors.len());
        for (binding, tensor) in tensors.iter().enumerate() {
            let layout = tensor.layout();
            let allocation_len = layout_allocation_len(layout)?;
            let array_ty = module.types.insert(
                Type {
                    name: Some(format!("Buffer{binding}")),
                    inner: TypeInner::Array {
                        base: match tensor.datatype() {
                            DataTypeEnum::F32 => f32_ty,
                            DataTypeEnum::F16 => f16_ty?,
                            DataTypeEnum::U32 => u32_ty,
                        },
                        size: ArraySize::Constant(NonZeroU32::new(allocation_len)?),
                        stride: tensor.datatype().element_size() as u32,
                    },
                },
                Span::default(),
            );
            let global = module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("buffer_{binding}")),
                    space: AddressSpace::Storage {
                        access: if binding == output_index {
                            StorageAccess::LOAD | StorageAccess::STORE
                        } else {
                            StorageAccess::LOAD
                        },
                    },
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: binding as u32,
                    }),
                    ty: array_ty,
                    init: None,
                },
                Span::default(),
            );
            views.push(TensorView {
                global,
                datatype: tensor.datatype(),
                strides: layout
                    .strides()
                    .iter()
                    .copied()
                    .map(u32::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?,
                offset: layout.offset().try_into().ok()?,
            });
        }

        Some(Self {
            module,
            u32_ty,
            output: views[output_index].clone(),
            inputs: views,
            operation,
        })
    }

    fn build(mut self, workgroup_shape: WorkgroupShape, dispatch_size: [u32; 3]) -> Option<Module> {
        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![
                FunctionArgument {
                    name: Some("local_invocation_index".into()),
                    ty: self.u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
                },
                FunctionArgument {
                    name: Some("workgroup_id".into()),
                    ty: self.module.types.insert(
                        Type {
                            name: Some("WorkgroupId".into()),
                            inner: TypeInner::Vector {
                                size: wgpu::naga::VectorSize::Tri,
                                scalar: Self::scalar(DataTypeEnum::U32),
                            },
                        },
                        Span::default(),
                    ),
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                },
            ],
            ..Function::default()
        };

        let mut body = Block::new();
        let workgroup_flat =
            self.workgroup_flat_index(&mut function.expressions, &mut body, dispatch_size);
        let workgroup_volume = workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z();
        let tile_span = workgroup_volume * TILE_SIZE;
        let group_base = self.mul_lit_u32(
            &mut function.expressions,
            &mut body,
            workgroup_flat,
            tile_span,
        );
        let local = self.append(&mut function.expressions, Expression::FunctionArgument(0));
        let local_base = self.mul_lit_u32(&mut function.expressions, &mut body, local, TILE_SIZE);
        let thread_base = self.binary(
            &mut function.expressions,
            &mut body,
            BinaryOperator::Add,
            group_base,
            local_base,
        );
        let total_elements = self
            .operation
            .shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;

        for local_tile in 0..TILE_SIZE {
            let flat = self.add_lit_u32(
                &mut function.expressions,
                &mut body,
                thread_base,
                local_tile,
            );
            let in_bounds = self.cmp_lit_u32(
                &mut function.expressions,
                &mut body,
                BinaryOperator::Less,
                flat,
                total_elements,
            );
            let mut accept = Block::new();
            let dims = self.output_dims_from_flat(&mut function.expressions, &mut accept, flat)?;
            let value = self.eval_expr(
                &mut function.expressions,
                &mut accept,
                &self.operation.expression,
                &dims,
            )?;
            let value = self.cast_value(
                &mut function.expressions,
                &mut accept,
                value,
                self.operation.output_datatype,
            );
            let output_index =
                self.layout_index(&mut function.expressions, &mut accept, &self.output, &dims);
            let output_ptr = self.storage_pointer(
                &mut function.expressions,
                &mut accept,
                &self.output,
                output_index,
            );
            accept.push(
                Statement::Store {
                    pointer: output_ptr,
                    value: value.handle,
                },
                Span::default(),
            );
            body.push(
                Statement::If {
                    condition: in_bounds,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
        }

        body.push(Statement::Return { value: None }, Span::default());
        function.body = body;
        self.module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [
                workgroup_shape.x(),
                workgroup_shape.y(),
                workgroup_shape.z(),
            ],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let mut capabilities = wgpu::naga::valid::Capabilities::empty();
        if self
            .inputs
            .iter()
            .any(|view| view.datatype == DataTypeEnum::F16)
            || self.operation.output_datatype == DataTypeEnum::F16
            || Self::expr_uses_f16(&self.operation.expression)
        {
            capabilities |= wgpu::naga::valid::Capabilities::SHADER_FLOAT16;
        }
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), capabilities)
            .validate(&self.module)
            .unwrap_or_else(|error| {
                panic!(
                    "direct nary Naga validation failed: {error:#?}\nentry: {:#?}",
                    self.module.entry_points.first()
                )
            });
        Some(self.module)
    }

    fn eval_expr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expr: &NaryExpr,
        dims: &[Handle<Expression>],
    ) -> Option<ExprValue> {
        match expr {
            NaryExpr::Op { children, function } => {
                let mut values = Vec::with_capacity(children.len());
                for (child, expected) in children.iter().zip(&function.input_types) {
                    let value = self.eval_expr(expressions, body, child, dims)?;
                    values.push(self.cast_value(expressions, body, value, *expected));
                }
                Some(self.emit_function(expressions, body, function, &values))
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                let view = self.inputs.get(*input_idx)?;
                let mut coords = Vec::with_capacity(indices.len());
                for index_expr in indices {
                    let value = self.eval_expr(expressions, body, index_expr, dims)?;
                    let value = self.cast_value(expressions, body, value, DataTypeEnum::U32);
                    coords.push(value.handle);
                }
                let index = self.layout_index(expressions, body, view, &coords);
                let pointer = self.storage_pointer(expressions, body, view, index);
                let value = self.emit(expressions, body, Expression::Load { pointer });
                Some(ExprValue {
                    handle: value,
                    datatype: view.datatype,
                })
            }
            NaryExpr::DimIndex(dim) => Some(ExprValue {
                handle: *dims.get(*dim)?,
                datatype: DataTypeEnum::U32,
            }),
            NaryExpr::Scalar(value) => Some(ExprValue {
                handle: self.scalar_literal(expressions, *value),
                datatype: value.datatype(),
            }),
        }
    }

    fn emit_function(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        function: &NaryFunction,
        values: &[ExprValue],
    ) -> ExprValue {
        let handle = match function.op {
            NaryOp::Add => self.binary(
                expressions,
                body,
                BinaryOperator::Add,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Sub => self.binary(
                expressions,
                body,
                BinaryOperator::Subtract,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Mul => self.binary(
                expressions,
                body,
                BinaryOperator::Multiply,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Div => self.binary(
                expressions,
                body,
                BinaryOperator::Divide,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Rem => self.binary(
                expressions,
                body,
                BinaryOperator::Modulo,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Pow => self.math2(
                expressions,
                body,
                MathFunction::Pow,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Min => self.math2(
                expressions,
                body,
                MathFunction::Min,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Max => self.math2(
                expressions,
                body,
                MathFunction::Max,
                values[0].handle,
                values[1].handle,
            ),
            NaryOp::Neg => self.emit(
                expressions,
                body,
                Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: values[0].handle,
                },
            ),
            NaryOp::Cast => {
                return self.cast_value(expressions, body, values[0], function.output_type);
            }
            NaryOp::Select => {
                let condition = self.numeric_not_zero(expressions, body, values[0]);
                self.emit(
                    expressions,
                    body,
                    Expression::Select {
                        condition,
                        accept: values[1].handle,
                        reject: values[2].handle,
                    },
                )
            }
            NaryOp::Exp => self.math1(expressions, body, MathFunction::Exp, values[0].handle),
            NaryOp::Exp2 => self.math1(expressions, body, MathFunction::Exp2, values[0].handle),
            NaryOp::Log => self.math1(expressions, body, MathFunction::Log, values[0].handle),
            NaryOp::Log2 => self.math1(expressions, body, MathFunction::Log2, values[0].handle),
            NaryOp::Sqrt => self.math1(expressions, body, MathFunction::Sqrt, values[0].handle),
            NaryOp::Sin => self.math1(expressions, body, MathFunction::Sin, values[0].handle),
            NaryOp::Cos => self.math1(expressions, body, MathFunction::Cos, values[0].handle),
            NaryOp::Tan => self.math1(expressions, body, MathFunction::Tan, values[0].handle),
            NaryOp::Tanh => self.math1(expressions, body, MathFunction::Tanh, values[0].handle),
            NaryOp::Asin => self.math1(expressions, body, MathFunction::Asin, values[0].handle),
            NaryOp::Acos => self.math1(expressions, body, MathFunction::Acos, values[0].handle),
            NaryOp::Atan => self.math1(expressions, body, MathFunction::Atan, values[0].handle),
            NaryOp::Sinh => self.math1(expressions, body, MathFunction::Sinh, values[0].handle),
            NaryOp::Cosh => self.math1(expressions, body, MathFunction::Cosh, values[0].handle),
            NaryOp::Asinh => self.math1(expressions, body, MathFunction::Asinh, values[0].handle),
            NaryOp::Acosh => self.math1(expressions, body, MathFunction::Acosh, values[0].handle),
            NaryOp::Atanh => self.math1(expressions, body, MathFunction::Atanh, values[0].handle),
            NaryOp::Abs => self.math1(expressions, body, MathFunction::Abs, values[0].handle),
            NaryOp::ApproximateExp | NaryOp::LessApproximateExp => {
                self.math1(expressions, body, MathFunction::Exp, values[0].handle)
            }
            NaryOp::TanhExact => self.tanh_exact(expressions, body, values[0].handle),
            NaryOp::Equal
            | NaryOp::Less
            | NaryOp::LessEqual
            | NaryOp::Greater
            | NaryOp::GreaterEqual => {
                let op = match function.op {
                    NaryOp::Equal => BinaryOperator::Equal,
                    NaryOp::Less => BinaryOperator::Less,
                    NaryOp::LessEqual => BinaryOperator::LessEqual,
                    NaryOp::Greater => BinaryOperator::Greater,
                    NaryOp::GreaterEqual => BinaryOperator::GreaterEqual,
                    _ => unreachable!(),
                };
                let condition =
                    self.binary(expressions, body, op, values[0].handle, values[1].handle);
                let one = self.one(expressions, function.output_type);
                let zero = self.zero(expressions, function.output_type);
                self.emit(
                    expressions,
                    body,
                    Expression::Select {
                        condition,
                        accept: one,
                        reject: zero,
                    },
                )
            }
            NaryOp::AddConst(scalar)
            | NaryOp::SubConst(scalar)
            | NaryOp::RSubConst(scalar)
            | NaryOp::MulConst(scalar)
            | NaryOp::DivConst(scalar)
            | NaryOp::RDivConst(scalar)
            | NaryOp::RemConst(scalar)
            | NaryOp::RRemConst(scalar) => {
                let scalar = self.scalar_literal(expressions, scalar);
                let (op, left, right) = match function.op {
                    NaryOp::AddConst(_) => (BinaryOperator::Add, values[0].handle, scalar),
                    NaryOp::SubConst(_) => (BinaryOperator::Subtract, values[0].handle, scalar),
                    NaryOp::RSubConst(_) => (BinaryOperator::Subtract, scalar, values[0].handle),
                    NaryOp::MulConst(_) => (BinaryOperator::Multiply, values[0].handle, scalar),
                    NaryOp::DivConst(_) => (BinaryOperator::Divide, values[0].handle, scalar),
                    NaryOp::RDivConst(_) => (BinaryOperator::Divide, scalar, values[0].handle),
                    NaryOp::RemConst(_) => (BinaryOperator::Modulo, values[0].handle, scalar),
                    NaryOp::RRemConst(_) => (BinaryOperator::Modulo, scalar, values[0].handle),
                    _ => unreachable!(),
                };
                self.binary(expressions, body, op, left, right)
            }
            NaryOp::PowConst(scalar) => {
                let scalar = self.scalar_literal(expressions, scalar);
                self.math2(
                    expressions,
                    body,
                    MathFunction::Pow,
                    values[0].handle,
                    scalar,
                )
            }
            NaryOp::MinConst(scalar) | NaryOp::MaxConst(scalar) => {
                let scalar = self.scalar_literal(expressions, scalar);
                let fun = if matches!(function.op, NaryOp::MinConst(_)) {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                self.math2(expressions, body, fun, values[0].handle, scalar)
            }
            NaryOp::EqualConst(scalar)
            | NaryOp::LessConst(scalar)
            | NaryOp::LessEqualConst(scalar)
            | NaryOp::GreaterConst(scalar)
            | NaryOp::GreaterEqualConst(scalar) => {
                let scalar = self.scalar_literal(expressions, scalar);
                let op = match function.op {
                    NaryOp::EqualConst(_) => BinaryOperator::Equal,
                    NaryOp::LessConst(_) => BinaryOperator::Less,
                    NaryOp::LessEqualConst(_) => BinaryOperator::LessEqual,
                    NaryOp::GreaterConst(_) => BinaryOperator::Greater,
                    NaryOp::GreaterEqualConst(_) => BinaryOperator::GreaterEqual,
                    _ => unreachable!(),
                };
                let condition = self.binary(expressions, body, op, values[0].handle, scalar);
                let one = self.one(expressions, function.output_type);
                let zero = self.zero(expressions, function.output_type);
                self.emit(
                    expressions,
                    body,
                    Expression::Select {
                        condition,
                        accept: one,
                        reject: zero,
                    },
                )
            }
        };
        ExprValue {
            handle,
            datatype: function.output_type,
        }
    }

    fn tanh_exact(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let exp_pos = self.math1(expressions, body, MathFunction::Exp, value);
        let neg = self.emit(
            expressions,
            body,
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr: value,
            },
        );
        let exp_neg = self.math1(expressions, body, MathFunction::Exp, neg);
        let numerator = self.binary(
            expressions,
            body,
            BinaryOperator::Subtract,
            exp_pos,
            exp_neg,
        );
        let denominator = self.binary(expressions, body, BinaryOperator::Add, exp_pos, exp_neg);
        self.binary(
            expressions,
            body,
            BinaryOperator::Divide,
            numerator,
            denominator,
        )
    }

    fn output_dims_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        flat: Handle<Expression>,
    ) -> Option<Vec<Handle<Expression>>> {
        let mut dims = Vec::with_capacity(self.operation.shape.len());
        for axis in 0..self.operation.shape.len() {
            let divisor = self.operation.shape[axis + 1..]
                .iter()
                .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
            let quotient = if divisor == 1 {
                flat
            } else {
                self.div_lit_u32(expressions, body, flat, divisor)
            };
            let dim = u32::try_from(self.operation.shape[axis]).ok()?;
            dims.push(if dim == 1 {
                self.u32(expressions, 0)
            } else {
                self.mod_lit_u32(expressions, body, quotient, dim)
            });
        }
        Some(dims)
    }

    fn layout_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        view: &TensorView,
        coords: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let mut index = self.u32(expressions, view.offset);
        for (coord, stride) in coords.iter().copied().zip(&view.strides) {
            if *stride == 0 {
                continue;
            }
            let term = self.mul_lit_u32(expressions, body, coord, *stride);
            index = self.binary(expressions, body, BinaryOperator::Add, index, term);
        }
        index
    }

    fn storage_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        view: &TensorView,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = self.append(expressions, Expression::GlobalVariable(view.global));
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn workgroup_flat_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dispatch_size: [u32; 3],
    ) -> Handle<Expression> {
        let wg = self.append(expressions, Expression::FunctionArgument(1));
        let x = self.emit(
            expressions,
            body,
            Expression::AccessIndex { base: wg, index: 0 },
        );
        let y = self.emit(
            expressions,
            body,
            Expression::AccessIndex { base: wg, index: 1 },
        );
        let z = self.emit(
            expressions,
            body,
            Expression::AccessIndex { base: wg, index: 2 },
        );
        let y_term = self.mul_lit_u32(expressions, body, y, dispatch_size[0]);
        let xy = self.binary(expressions, body, BinaryOperator::Add, x, y_term);
        let z_term = self.mul_lit_u32(
            expressions,
            body,
            z,
            dispatch_size[0].saturating_mul(dispatch_size[1]),
        );
        self.binary(expressions, body, BinaryOperator::Add, xy, z_term)
    }

    fn cast_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: ExprValue,
        target: DataTypeEnum,
    ) -> ExprValue {
        if value.datatype == target {
            return value;
        }
        let scalar = Self::scalar(target);
        let handle = self.emit(
            expressions,
            body,
            Expression::As {
                expr: value.handle,
                kind: scalar.kind,
                convert: Some(scalar.width),
            },
        );
        ExprValue {
            handle,
            datatype: target,
        }
    }

    fn numeric_not_zero(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: ExprValue,
    ) -> Handle<Expression> {
        let zero = self.zero(expressions, value.datatype);
        self.binary(
            expressions,
            body,
            BinaryOperator::NotEqual,
            value.handle,
            zero,
        )
    }

    fn math1(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    fn math2(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: None,
                arg3: None,
            },
        )
    }

    fn binary(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn add_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            value
        } else {
            let rhs = self.u32(expressions, literal);
            self.binary(expressions, body, BinaryOperator::Add, value, rhs)
        }
    }

    fn mul_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 1 {
            value
        } else {
            let rhs = self.u32(expressions, literal);
            self.binary(expressions, body, BinaryOperator::Multiply, value, rhs)
        }
    }

    fn div_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 1 {
            value
        } else {
            let rhs = self.u32(expressions, literal);
            self.binary(expressions, body, BinaryOperator::Divide, value, rhs)
        }
    }

    fn mod_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32(expressions, literal);
        self.binary(expressions, body, BinaryOperator::Modulo, value, rhs)
    }

    fn cmp_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32(expressions, literal);
        self.binary(expressions, body, op, value, rhs)
    }

    fn zero(
        &self,
        expressions: &mut Arena<Expression>,
        datatype: DataTypeEnum,
    ) -> Handle<Expression> {
        match datatype {
            DataTypeEnum::F32 => {
                expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default())
            }
            DataTypeEnum::F16 => expressions.append(
                Expression::Literal(Literal::F16(half::f16::from_f32(0.0))),
                Span::default(),
            ),
            DataTypeEnum::U32 => self.u32(expressions, 0),
        }
    }

    fn one(
        &self,
        expressions: &mut Arena<Expression>,
        datatype: DataTypeEnum,
    ) -> Handle<Expression> {
        match datatype {
            DataTypeEnum::F32 => {
                expressions.append(Expression::Literal(Literal::F32(1.0)), Span::default())
            }
            DataTypeEnum::F16 => expressions.append(
                Expression::Literal(Literal::F16(half::f16::from_f32(1.0))),
                Span::default(),
            ),
            DataTypeEnum::U32 => self.u32(expressions, 1),
        }
    }

    fn u32(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    fn scalar_literal(
        &self,
        expressions: &mut Arena<Expression>,
        value: NaryScalar,
    ) -> Handle<Expression> {
        expressions.append(
            Expression::Literal(match value {
                NaryScalar::F32(value) => Literal::F32(value),
                NaryScalar::F16(value) => Literal::F16(value),
                NaryScalar::U32(value) => Literal::U32(value),
            }),
            Span::default(),
        )
    }

    fn emit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expression: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expression, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, handle)),
            Span::default(),
        );
        handle
    }

    fn append(
        &self,
        expressions: &mut Arena<Expression>,
        expression: Expression,
    ) -> Handle<Expression> {
        expressions.append(expression, Span::default())
    }

    fn single_expression_range(
        expressions: &Arena<Expression>,
        handle: Handle<Expression>,
    ) -> NagaRange<Expression> {
        NagaRange::from_index_range(
            handle.index() as u32..handle.index() as u32 + 1,
            expressions,
        )
    }

    fn scalar(datatype: DataTypeEnum) -> Scalar {
        match datatype {
            DataTypeEnum::F32 => Scalar {
                kind: ScalarKind::Float,
                width: 4,
            },
            DataTypeEnum::F16 => Scalar {
                kind: ScalarKind::Float,
                width: 2,
            },
            DataTypeEnum::U32 => Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
        }
    }

    fn expr_uses_f16(expr: &NaryExpr) -> bool {
        match expr {
            NaryExpr::Op { children, function } => {
                function.output_type == DataTypeEnum::F16
                    || function
                        .input_types
                        .iter()
                        .any(|datatype| *datatype == DataTypeEnum::F16)
                    || children.iter().any(Self::expr_uses_f16)
            }
            NaryExpr::Scalar(value) => value.datatype() == DataTypeEnum::F16,
            NaryExpr::IndexedInput { indices, .. } => indices.iter().any(Self::expr_uses_f16),
            NaryExpr::DimIndex(_) => false,
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
