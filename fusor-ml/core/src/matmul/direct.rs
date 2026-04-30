use wgpu::naga::{
    AddressSpace, Arena, BinaryOperator, Binding, Block, BuiltIn, EntryPoint, Expression, Function,
    FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable, Module, Range as NagaRange,
    ResourceBinding, Scalar, ScalarKind, ShaderStage, Span, Statement, StorageAccess, Type,
    TypeInner,
};

use crate::{
    Layout,
    matmul::MatMulOperation,
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding, direct_storage_array_size},
        inputs::MirValue,
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    tensor::{DataTypeEnum, TensorData},
};

#[derive(Clone)]
struct TensorView {
    global: Handle<GlobalVariable>,
    datatype: DataTypeEnum,
    shape: Vec<u32>,
    strides: Vec<u32>,
    offset: u32,
}

pub(crate) fn build_serial_matmul_direct_kernel(
    operation: &MatMulOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    if operation
        .pre_element_wise
        .iter()
        .any(|chain| !chain.functions.is_empty())
        || !operation.post_element_wise.functions.is_empty()
    {
        return None;
    }

    let [input_a, input_b, output] = inputs else {
        return None;
    };
    let input_a = input_a.as_tensor()?.clone();
    let input_b = input_b.as_tensor()?.clone();
    let output = output.as_tensor()?.clone();
    if input_a.layout().rank() != input_b.layout().rank()
        || input_a.layout().rank() != output.layout().rank()
        || input_a.layout().rank() < 2
    {
        return None;
    }
    if input_a.datatype() != input_b.datatype() {
        return None;
    }
    if output.datatype() != operation.post_element_wise.out_datatype() {
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
    let workgroup_volume = workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z();
    let dispatch_size = [total_outputs.div_ceil(workgroup_volume), 1, 1];
    let cache_key = format!(
        "{}:serial-direct:{:?}:dispatch={dispatch_size:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
        operation.name(),
        workgroup_shape.shape(),
        input_a.datatype(),
        input_a.layout(),
        input_b.datatype(),
        input_b.layout(),
        output.datatype(),
        output.layout()
    );
    let module = SerialMatMulDirectBuilder::new(operation, &input_a, &input_b, &output)?
        .build(*workgroup_shape, dispatch_size)?;

    Some(DirectKernel::new_with_cache_key(
        operation.name(),
        cache_key,
        module,
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input_a.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: input_b.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: output.buffer().clone(),
                read_only: false,
            },
        ],
        dispatch_size,
    ))
}

struct SerialMatMulDirectBuilder<'a> {
    module: Module,
    u32_ty: Handle<Type>,
    acc_ty: Handle<Type>,
    input_a: TensorView,
    input_b: TensorView,
    output: TensorView,
    operation: &'a MatMulOperation,
}

impl<'a> SerialMatMulDirectBuilder<'a> {
    fn new(
        operation: &'a MatMulOperation,
        input_a: &TensorData,
        input_b: &TensorData,
        output: &TensorData,
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
        let needs_f16 = input_a.datatype() == DataTypeEnum::F16
            || input_b.datatype() == DataTypeEnum::F16
            || output.datatype() == DataTypeEnum::F16;
        let f16_ty = needs_f16.then(|| {
            module.types.insert(
                Type {
                    name: Some("f16".into()),
                    inner: TypeInner::Scalar(Self::scalar(DataTypeEnum::F16)),
                },
                Span::default(),
            )
        });
        let acc_ty = match operation.matmul_dtype() {
            DataTypeEnum::U32 => u32_ty,
            DataTypeEnum::F32 | DataTypeEnum::F16 => f32_ty,
        };

        let input_a = Self::add_tensor(&mut module, 0, input_a, true, f32_ty, f16_ty, u32_ty)?;
        let input_b = Self::add_tensor(&mut module, 1, input_b, true, f32_ty, f16_ty, u32_ty)?;
        let output = Self::add_tensor(&mut module, 2, output, false, f32_ty, f16_ty, u32_ty)?;

        Some(Self {
            module,
            u32_ty,
            acc_ty,
            input_a,
            input_b,
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
        let base = match tensor.datatype() {
            DataTypeEnum::F32 => f32_ty,
            DataTypeEnum::F16 => f16_ty?,
            DataTypeEnum::U32 => u32_ty,
        };
        let array_ty = module.types.insert(
            Type {
                name: Some(format!("MatMulBuffer{binding}")),
                inner: TypeInner::Array {
                    base,
                    size: direct_storage_array_size(layout_allocation_len(tensor.layout())?),
                    stride: tensor.datatype().element_size() as u32,
                },
            },
            Span::default(),
        );
        let global = module.global_variables.append(
            GlobalVariable {
                name: Some(format!("matmul_buffer_{binding}")),
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
            shape: tensor
                .layout()
                .shape()
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
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
        })
    }

    fn build(mut self, workgroup_shape: WorkgroupShape, dispatch_size: [u32; 3]) -> Option<Module> {
        let workgroup_id_ty = self.module.types.insert(
            Type {
                name: Some("MatMulWorkgroupId".into()),
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

        let acc_local = function.local_variables.append(
            LocalVariable {
                name: Some("acc".into()),
                ty: self.acc_ty,
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
            .shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;
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
        let zero = self.zero(&mut function.expressions, self.operation.matmul_dtype());
        let zero = self.cast_value(
            &mut function.expressions,
            &mut accept,
            zero,
            self.operation.matmul_dtype(),
            self.acc_datatype(),
        );
        let acc_ptr = self.local_pointer(&mut function.expressions, acc_local);
        accept.push(
            Statement::Store {
                pointer: acc_ptr,
                value: zero,
            },
            Span::default(),
        );
        let k_ptr = self.local_pointer(&mut function.expressions, k_local);
        let zero_u32 = self.u32(&mut function.expressions, 0);
        accept.push(
            Statement::Store {
                pointer: k_ptr,
                value: zero_u32,
            },
            Span::default(),
        );

        let rank = self.output.shape.len();
        let k_size = *self.input_a.shape.get(rank - 1)?;
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
            k_size,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let mut a_coords = dims.clone();
        a_coords[rank - 1] = k;
        let mut b_coords = dims.clone();
        b_coords[rank - 2] = k;
        let a = self.load_tensor_value(
            &mut function.expressions,
            &mut loop_body,
            &self.input_a,
            &a_coords,
            self.acc_datatype(),
        );
        let b = self.load_tensor_value(
            &mut function.expressions,
            &mut loop_body,
            &self.input_b,
            &b_coords,
            self.acc_datatype(),
        );
        let product = self.binary(
            &mut function.expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            a,
            b,
        );
        let acc_ptr = self.local_pointer(&mut function.expressions, acc_local);
        let acc = self.emit(
            &mut function.expressions,
            &mut loop_body,
            Expression::Load { pointer: acc_ptr },
        );
        let acc_next = self.binary(
            &mut function.expressions,
            &mut loop_body,
            BinaryOperator::Add,
            acc,
            product,
        );
        let acc_ptr = self.local_pointer(&mut function.expressions, acc_local);
        loop_body.push(
            Statement::Store {
                pointer: acc_ptr,
                value: acc_next,
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

        let acc_ptr = self.local_pointer(&mut function.expressions, acc_local);
        let output_value = self.emit(
            &mut function.expressions,
            &mut accept,
            Expression::Load { pointer: acc_ptr },
        );
        let output_value = self.cast_value(
            &mut function.expressions,
            &mut accept,
            output_value,
            self.acc_datatype(),
            self.output.datatype,
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
        if self.input_a.datatype == DataTypeEnum::F16
            || self.input_b.datatype == DataTypeEnum::F16
            || self.output.datatype == DataTypeEnum::F16
        {
            capabilities |= wgpu::naga::valid::Capabilities::SHADER_FLOAT16;
        }
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), capabilities)
            .validate(&self.module)
            .unwrap_or_else(|error| panic!("direct matmul Naga validation failed: {error:#?}"));
        Some(self.module)
    }

    fn acc_datatype(&self) -> DataTypeEnum {
        match self.operation.matmul_dtype() {
            DataTypeEnum::U32 => DataTypeEnum::U32,
            DataTypeEnum::F32 | DataTypeEnum::F16 => DataTypeEnum::F32,
        }
    }

    fn load_tensor_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        view: &TensorView,
        coords: &[Handle<Expression>],
        target: DataTypeEnum,
    ) -> Handle<Expression> {
        let index = self.layout_index(expressions, body, view, coords);
        let pointer = self.storage_pointer(expressions, body, view, index);
        let value = self.emit(expressions, body, Expression::Load { pointer });
        self.cast_value(expressions, body, value, view.datatype, target)
    }

    fn output_dims_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        flat: Handle<Expression>,
    ) -> Option<Vec<Handle<Expression>>> {
        let mut dims = Vec::with_capacity(self.output.shape.len());
        for axis in 0..self.output.shape.len() {
            let divisor = self.output.shape[axis + 1..]
                .iter()
                .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;
            let quotient = if divisor == 1 {
                flat
            } else {
                self.div_lit_u32(expressions, body, flat, divisor)
            };
            let dim = self.output.shape[axis];
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

    fn zero(
        &self,
        expressions: &mut Arena<Expression>,
        datatype: DataTypeEnum,
    ) -> Handle<Expression> {
        expressions.append(
            Expression::Literal(match datatype {
                DataTypeEnum::F32 => Literal::F32(0.0),
                DataTypeEnum::F16 => Literal::F16(half::f16::from_f32(0.0)),
                DataTypeEnum::U32 => Literal::U32(0),
            }),
            Span::default(),
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
