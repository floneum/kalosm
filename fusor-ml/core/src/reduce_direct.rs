use std::num::NonZeroU32;

use wgpu::naga::{
    AddressSpace, Arena, ArraySize, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, Range as NagaRange, ResourceBinding, Scalar, ScalarKind, ShaderStage,
    Span, Statement, StorageAccess, Type, TypeInner,
};

use crate::{
    Layout,
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_wise::NaryScalar,
    reduce::{ReduceOp, ReduceOperation},
    tensor::{DataTypeEnum, TensorData},
};

#[derive(Clone)]
struct TensorView {
    global: Handle<GlobalVariable>,
    datatype: DataTypeEnum,
    strides: Vec<u32>,
    offset: u32,
}

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
        "{}:direct:{:?}:dispatch={dispatch_size:?}:reduce={reduce_size}:stride={reduce_stride}:{:?}:{:?}:{:?}:{:?}",
        operation.name(),
        workgroup_shape.shape(),
        input.datatype(),
        input.layout(),
        output.datatype(),
        output.layout()
    );
    let module = ReduceDirectBuilder::new(operation, input.clone(), output.clone())?.build(
        *workgroup_shape,
        dispatch_size,
        reduce_size,
        reduce_stride,
    )?;

    Some(DirectKernel::new_with_cache_key(
        operation.name(),
        cache_key,
        module,
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
    ))
}

struct ReduceDirectBuilder<'a> {
    module: Module,
    u32_ty: Handle<Type>,
    value_ty: Handle<Type>,
    input: TensorView,
    output: TensorView,
    operation: &'a ReduceOperation,
}

impl<'a> ReduceDirectBuilder<'a> {
    fn new(operation: &'a ReduceOperation, input: TensorData, output: TensorData) -> Option<Self> {
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
        let f16_ty = (input.datatype() == DataTypeEnum::F16
            || output.datatype() == DataTypeEnum::F16
            || operation.function.datatype() == DataTypeEnum::F16)
            .then(|| {
                module.types.insert(
                    Type {
                        name: Some("f16".into()),
                        inner: TypeInner::Scalar(Self::scalar(DataTypeEnum::F16)),
                    },
                    Span::default(),
                )
            });

        let value_ty = match operation.function.datatype() {
            DataTypeEnum::F32 => f32_ty,
            DataTypeEnum::F16 => f16_ty?,
            DataTypeEnum::U32 => u32_ty,
        };

        let input = Self::add_tensor(&mut module, 0, &input, true, f32_ty, f16_ty, u32_ty)?;
        let output = Self::add_tensor(&mut module, 1, &output, false, f32_ty, f16_ty, u32_ty)?;

        Some(Self {
            module,
            u32_ty,
            value_ty,
            input,
            output,
            operation,
        })
    }

    fn add_tensor(
        module: &mut Module,
        binding: u32,
        tensor: &TensorData,
        read_only: bool,
        f32_ty: Handle<Type>,
        f16_ty: Option<Handle<Type>>,
        u32_ty: Handle<Type>,
    ) -> Option<TensorView> {
        let layout = tensor.layout();
        let base = match tensor.datatype() {
            DataTypeEnum::F32 => f32_ty,
            DataTypeEnum::F16 => f16_ty?,
            DataTypeEnum::U32 => u32_ty,
        };
        let array_ty = module.types.insert(
            Type {
                name: Some(format!("ReduceBuffer{binding}")),
                inner: TypeInner::Array {
                    base,
                    size: ArraySize::Constant(NonZeroU32::new(layout_allocation_len(layout)?)?),
                    stride: tensor.datatype().element_size() as u32,
                },
            },
            Span::default(),
        );
        let global = module.global_variables.append(
            GlobalVariable {
                name: Some(format!("reduce_buffer_{binding}")),
                space: AddressSpace::Storage {
                    access: if read_only {
                        StorageAccess::LOAD
                    } else {
                        StorageAccess::LOAD | StorageAccess::STORE
                    },
                },
                binding: Some(ResourceBinding { group: 0, binding }),
                ty: array_ty,
                init: None,
            },
            Span::default(),
        );
        Some(TensorView {
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
        })
    }

    fn build(
        mut self,
        workgroup_shape: WorkgroupShape,
        dispatch_size: [u32; 3],
        reduce_size: u32,
        reduce_stride: u32,
    ) -> Option<Module> {
        let workgroup_id_ty = self.module.types.insert(
            Type {
                name: Some("ReduceWorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: wgpu::naga::VectorSize::Tri,
                    scalar: Self::scalar(DataTypeEnum::U32),
                },
            },
            Span::default(),
        );
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
                    ty: workgroup_id_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                },
            ],
            ..Function::default()
        };

        let merged_local = function.local_variables.append(
            LocalVariable {
                name: Some("merged".into()),
                ty: self.value_ty,
                init: None,
            },
            Span::default(),
        );
        let k_local = function.local_variables.append(
            LocalVariable {
                name: Some("k".into()),
                ty: self.u32_ty,
                init: None,
            },
            Span::default(),
        );

        let mut body = Block::new();
        let wg_flat =
            self.workgroup_flat_index(&mut function.expressions, &mut body, dispatch_size);
        let group_base = self.mul_lit_u32(
            &mut function.expressions,
            &mut body,
            wg_flat,
            workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z(),
        );
        let local = self.append(&mut function.expressions, Expression::FunctionArgument(0));
        let output_flat = self.binary(
            &mut function.expressions,
            &mut body,
            BinaryOperator::Add,
            group_base,
            local,
        );
        let total_outputs = self
            .output
            .strides
            .iter()
            .zip(
                self.operation
                    .shape
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dim)| (i != self.operation.axis).then_some(*dim)),
            )
            .map(|(_, dim)| dim)
            .try_fold(1u32, |acc, dim| acc.checked_mul(dim.try_into().ok()?))?;
        let in_bounds = self.cmp_lit_u32(
            &mut function.expressions,
            &mut body,
            BinaryOperator::Less,
            output_flat,
            total_outputs,
        );

        let mut accept = Block::new();
        let dims =
            self.output_dims_from_flat(&mut function.expressions, &mut accept, output_flat)?;
        let input_base =
            self.layout_index(&mut function.expressions, &mut accept, &self.input, &dims);
        let output_index =
            self.layout_index(&mut function.expressions, &mut accept, &self.output, &dims);
        let initial = self.scalar_literal(
            &mut function.expressions,
            &mut accept,
            self.operation.function.initial_value,
        );
        let merged_ptr = self.local_pointer(&mut function.expressions, merged_local);
        accept.push(
            Statement::Store {
                pointer: merged_ptr,
                value: initial,
            },
            Span::default(),
        );
        let zero = self.u32(&mut function.expressions, 0);
        let k_ptr = self.local_pointer(&mut function.expressions, k_local);
        accept.push(
            Statement::Store {
                pointer: k_ptr,
                value: zero,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let k_ptr = self.local_pointer(&mut function.expressions, k_local);
        let k = self.emit(
            &mut function.expressions,
            &mut loop_body,
            Expression::Load { pointer: k_ptr },
        );
        let done = self.cmp_lit_u32(
            &mut function.expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            k,
            reduce_size,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let reduce_offset =
            self.mul_lit_u32(&mut function.expressions, &mut loop_body, k, reduce_stride);
        let input_index = self.binary(
            &mut function.expressions,
            &mut loop_body,
            BinaryOperator::Add,
            input_base,
            reduce_offset,
        );
        let input_ptr = self.storage_pointer(
            &mut function.expressions,
            &mut loop_body,
            &self.input,
            input_index,
        );
        let value = self.emit(
            &mut function.expressions,
            &mut loop_body,
            Expression::Load { pointer: input_ptr },
        );
        let value = self.cast_value(
            &mut function.expressions,
            &mut loop_body,
            value,
            self.input.datatype,
            self.operation.function.datatype(),
        );
        let merged_ptr = self.local_pointer(&mut function.expressions, merged_local);
        let merged = self.emit(
            &mut function.expressions,
            &mut loop_body,
            Expression::Load {
                pointer: merged_ptr,
            },
        );
        let reduced = self.reduce_value(&mut function.expressions, &mut loop_body, value, merged);
        let merged_ptr = self.local_pointer(&mut function.expressions, merged_local);
        loop_body.push(
            Statement::Store {
                pointer: merged_ptr,
                value: reduced,
            },
            Span::default(),
        );
        let next_k = self.add_lit_u32(&mut function.expressions, &mut loop_body, k, 1);
        let k_ptr = self.local_pointer(&mut function.expressions, k_local);
        loop_body.push(
            Statement::Store {
                pointer: k_ptr,
                value: next_k,
            },
            Span::default(),
        );
        accept.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        let merged_ptr = self.local_pointer(&mut function.expressions, merged_local);
        let output_value = self.emit(
            &mut function.expressions,
            &mut accept,
            Expression::Load {
                pointer: merged_ptr,
            },
        );
        let output_value = self.cast_value(
            &mut function.expressions,
            &mut accept,
            output_value,
            self.operation.function.datatype(),
            self.output.datatype,
        );
        let output_ptr = self.storage_pointer(
            &mut function.expressions,
            &mut accept,
            &self.output,
            output_index,
        );
        accept.push(
            Statement::Store {
                pointer: output_ptr,
                value: output_value,
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
        if self.input.datatype == DataTypeEnum::F16
            || self.output.datatype == DataTypeEnum::F16
            || self.operation.function.datatype() == DataTypeEnum::F16
        {
            capabilities |= wgpu::naga::valid::Capabilities::SHADER_FLOAT16;
        }
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), capabilities)
            .validate(&self.module)
            .unwrap_or_else(|error| panic!("direct reduce Naga validation failed: {error:#?}"));
        Some(self.module)
    }

    fn reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        match self.operation.function.op {
            ReduceOp::Sum => self.binary(expressions, body, BinaryOperator::Add, left, right),
            ReduceOp::Product => {
                self.binary(expressions, body, BinaryOperator::Multiply, left, right)
            }
            ReduceOp::Max => self.math2(expressions, body, MathFunction::Max, left, right),
            ReduceOp::Min => self.math2(expressions, body, MathFunction::Min, left, right),
        }
    }

    fn output_dims_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        flat: Handle<Expression>,
    ) -> Option<Vec<Handle<Expression>>> {
        let output_shape = self
            .operation
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| (i != self.operation.axis).then_some(*dim))
            .collect::<Vec<_>>();
        let mut dims = Vec::with_capacity(output_shape.len());
        for axis in 0..output_shape.len() {
            let divisor = output_shape[axis + 1..]
                .iter()
                .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
            let quotient = if divisor == 1 {
                flat
            } else {
                self.div_lit_u32(expressions, body, flat, divisor)
            };
            let dim = u32::try_from(output_shape[axis]).ok()?;
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

    fn local_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        self.append(expressions, Expression::LocalVariable(local))
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
        value: Handle<Expression>,
        source: DataTypeEnum,
        target: DataTypeEnum,
    ) -> Handle<Expression> {
        if source == target {
            return value;
        }
        let scalar = Self::scalar(target);
        self.emit(
            expressions,
            body,
            Expression::As {
                expr: value,
                kind: scalar.kind,
                convert: Some(scalar.width),
            },
        )
    }

    fn scalar_literal(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: NaryScalar,
    ) -> Handle<Expression> {
        match value {
            NaryScalar::F32(value) => {
                expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
            }
            NaryScalar::F16(value) => {
                let f32_value = expressions.append(
                    Expression::Literal(Literal::F32(value.to_f32())),
                    Span::default(),
                );
                self.emit(
                    expressions,
                    body,
                    Expression::As {
                        expr: f32_value,
                        kind: ScalarKind::Float,
                        convert: Some(2),
                    },
                )
            }
            NaryScalar::U32(value) => self.u32(expressions, value),
        }
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

    fn u32(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
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
}

fn layout_allocation_len(layout: &Layout) -> Option<u32> {
    let max_index = layout
        .shape()
        .iter()
        .zip(layout.strides())
        .try_fold(layout.offset(), |acc, (dim, stride)| {
            acc.checked_add(dim.saturating_sub(1).checked_mul(*stride)?)
        })?;
    max_index.checked_add(1)?.try_into().ok()
}
