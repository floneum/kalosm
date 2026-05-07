use std::num::NonZeroU32;

use wgpu::naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CollectiveOperation, EntryPoint, Expression, Function, FunctionArgument, GlobalVariable,
    Handle, Literal, LocalVariable, MathFunction, Module, Range, ResourceBinding, Scalar,
    ShaderStage, Span, Statement, StorageAccess, SubgroupOperation, Type, TypeInner, VectorSize,
};

use super::{RmsNormVec4Meta, VEC4_BLOCK, VEC4_SUBGROUP_WIDTH};

#[derive(Clone, Copy)]
struct RmsNormVec4Globals {
    input: Handle<GlobalVariable>,
    residual: Option<Handle<GlobalVariable>>,
    weight: Handle<GlobalVariable>,
    bias: Option<Handle<GlobalVariable>>,
    output: Handle<GlobalVariable>,
    scratch: Handle<GlobalVariable>,
}

#[derive(Clone, Copy)]
struct RmsNormVec4Locals {
    col: Handle<LocalVariable>,
    sum: Handle<LocalVariable>,
}

struct RmsNormVec4NagaBuilder {
    meta: RmsNormVec4Meta,
    has_residual: bool,
    has_bias: bool,
}

impl RmsNormVec4NagaBuilder {
    fn new(meta: RmsNormVec4Meta) -> Self {
        Self {
            meta,
            has_residual: meta.residual_offset_vec.is_some(),
            has_bias: meta.bias_offset_vec.is_some(),
        }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_vec4_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                },
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let storage_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: f32_vec4_ty,
                    size: ArraySize::Dynamic,
                    stride: 16,
                },
            },
            Span::default(),
        );
        let scratch_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(VEC4_SUBGROUP_WIDTH)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let input = Self::storage_global(&mut module, 0, storage_ty, true);
        let mut binding = 1;
        let residual = if self.has_residual {
            let residual = Self::storage_global(&mut module, binding, storage_ty, true);
            binding += 1;
            Some(residual)
        } else {
            None
        };
        let weight = Self::storage_global(&mut module, binding, storage_ty, true);
        binding += 1;
        let bias = if self.has_bias {
            let bias = Self::storage_global(&mut module, binding, storage_ty, true);
            binding += 1;
            Some(bias)
        } else {
            None
        };
        let output = Self::storage_global(&mut module, binding, storage_ty, false);
        let scratch = module.global_variables.append(
            GlobalVariable {
                name: None,
                space: AddressSpace::WorkGroup,
                binding: None,
                ty: scratch_ty,
                init: None,
            },
            Span::default(),
        );
        let globals = RmsNormVec4Globals {
            input,
            residual,
            weight,
            bias,
            output,
            scratch,
        };

        let mut function = Function {
            name: None,
            arguments: vec![
                FunctionArgument {
                    name: None,
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
                },
                FunctionArgument {
                    name: None,
                    ty: u32_vec3_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                },
                FunctionArgument {
                    name: None,
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::SubgroupId)),
                },
                FunctionArgument {
                    name: None,
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::SubgroupInvocationId)),
                },
            ],
            ..Function::default()
        };
        let locals = RmsNormVec4Locals {
            col: Self::local(&mut function, u32_ty),
            sum: Self::local(&mut function, f32_ty),
        };

        function.body = self.entry_body(
            &mut function.expressions,
            globals,
            locals,
            f32_ty,
            f32_vec4_ty,
        );
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [VEC4_BLOCK, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        Some(module)
    }

    fn entry_body(
        &self,
        expressions: &mut Arena<Expression>,
        globals: RmsNormVec4Globals,
        locals: RmsNormVec4Locals,
        f32_ty: Handle<Type>,
        f32_vec4_ty: Handle<Type>,
    ) -> Block {
        let mut body = Block::new();
        let local_index = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let subgroup_id = expressions.append(Expression::FunctionArgument(2), Span::default());
        let subgroup_lane = expressions.append(Expression::FunctionArgument(3), Span::default());
        let row = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );

        let first_subgroup = self.eq_lit(expressions, &mut body, subgroup_id, 0);
        let mut init_scratch = Block::new();
        let zero = self.f32_lit(expressions, 0.0);
        self.store_workgroup(
            expressions,
            &mut init_scratch,
            globals.scratch,
            subgroup_lane,
            zero,
        );
        body.push(
            Statement::If {
                condition: first_subgroup,
                accept: init_scratch,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        self.store_local(expressions, &mut body, locals.sum, zero);
        self.store_local(expressions, &mut body, locals.col, local_index);
        self.append_sum_loop(expressions, &mut body, globals, locals, row);

        let sum = self.load_local(expressions, &mut body, locals.sum);
        let subgroup_sum = self.subgroup_sum(expressions, &mut body, sum, f32_ty);
        let subgroup_lane_zero = self.eq_lit(expressions, &mut body, subgroup_lane, 0);
        let mut store_subgroup_sum = Block::new();
        self.store_workgroup(
            expressions,
            &mut store_subgroup_sum,
            globals.scratch,
            subgroup_id,
            subgroup_sum,
        );
        body.push(
            Statement::If {
                condition: subgroup_lane_zero,
                accept: store_subgroup_sum,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let scratch_sum =
            self.load_workgroup(expressions, &mut body, globals.scratch, subgroup_lane);
        let total_sum = self.subgroup_sum(expressions, &mut body, scratch_sum, f32_ty);
        let cols = self.f32_lit(expressions, self.meta.cols as f32);
        let mean = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Divide,
            total_sum,
            cols,
        );
        let eps = self.f32_lit(expressions, self.meta.eps);
        let mean_eps = self.bin(expressions, &mut body, BinaryOperator::Add, mean, eps);
        let scale = self.emit(
            expressions,
            &mut body,
            Expression::Math {
                fun: MathFunction::InverseSqrt,
                arg: mean_eps,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );

        self.store_local(expressions, &mut body, locals.col, local_index);
        self.append_store_loop(
            expressions,
            &mut body,
            globals,
            locals,
            row,
            scale,
            f32_vec4_ty,
        );

        body
    }

    fn append_sum_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: RmsNormVec4Globals,
        locals: RmsNormVec4Locals,
        row: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let col = self.load_local(expressions, &mut loop_body, locals.col);
        let done = self.ge_lit(expressions, &mut loop_body, col, self.meta.cols_vec);
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let value = self.load_input_vec4(expressions, &mut loop_body, globals, row, col);
        let dot = self.emit(
            expressions,
            &mut loop_body,
            Expression::Math {
                fun: MathFunction::Dot,
                arg: value,
                arg1: Some(value),
                arg2: None,
                arg3: None,
            },
        );
        let sum = self.load_local(expressions, &mut loop_body, locals.sum);
        let sum = self.bin(expressions, &mut loop_body, BinaryOperator::Add, sum, dot);
        self.store_local(expressions, &mut loop_body, locals.sum, sum);
        let next_col = self.add_lit(expressions, &mut loop_body, col, VEC4_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.col, next_col);

        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn append_store_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: RmsNormVec4Globals,
        locals: RmsNormVec4Locals,
        row: Handle<Expression>,
        scale: Handle<Expression>,
        f32_vec4_ty: Handle<Type>,
    ) {
        let mut loop_body = Block::new();
        let col = self.load_local(expressions, &mut loop_body, locals.col);
        let done = self.ge_lit(expressions, &mut loop_body, col, self.meta.cols_vec);
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let value = self.load_input_vec4(expressions, &mut loop_body, globals, row, col);
        let scale_vec = self.splat_vec4(expressions, &mut loop_body, f32_vec4_ty, scale);
        let normalized = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            value,
            scale_vec,
        );
        let weight_index = self.add_lit(
            expressions,
            &mut loop_body,
            col,
            self.meta.weight_offset_vec,
        );
        let weight = self.load_storage(expressions, &mut loop_body, globals.weight, weight_index);
        let mut output = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            normalized,
            weight,
        );
        if let (Some(bias), Some(bias_offset_vec)) = (globals.bias, self.meta.bias_offset_vec) {
            let bias_index = self.add_lit(expressions, &mut loop_body, col, bias_offset_vec);
            let bias_value = self.load_storage(expressions, &mut loop_body, bias, bias_index);
            output = self.bin(
                expressions,
                &mut loop_body,
                BinaryOperator::Add,
                output,
                bias_value,
            );
        }
        let output_index = self.matrix_index(
            expressions,
            &mut loop_body,
            self.meta.output_offset_vec,
            self.meta.output_row_stride_vec,
            row,
            col,
        );
        self.store_storage(
            expressions,
            &mut loop_body,
            globals.output,
            output_index,
            output,
        );
        let next_col = self.add_lit(expressions, &mut loop_body, col, VEC4_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.col, next_col);

        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn load_input_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: RmsNormVec4Globals,
        row: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Handle<Expression> {
        let input_index = self.matrix_index(
            expressions,
            body,
            self.meta.input_offset_vec,
            self.meta.input_row_stride_vec,
            row,
            col,
        );
        let mut value = self.load_storage(expressions, body, globals.input, input_index);
        if let (Some(residual), Some(residual_offset_vec)) =
            (globals.residual, self.meta.residual_offset_vec)
        {
            let residual_index = self.matrix_index(
                expressions,
                body,
                residual_offset_vec,
                self.meta.residual_row_stride_vec,
                row,
                col,
            );
            let residual_value = self.load_storage(expressions, body, residual, residual_index);
            value = self.bin(
                expressions,
                body,
                BinaryOperator::Add,
                value,
                residual_value,
            );
        }
        value
    }

    fn matrix_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        row_stride: u32,
        row: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = self.u32_lit(expressions, offset);
        let index = self.add_scaled_index(expressions, body, base, row, row_stride);
        self.bin(expressions, body, BinaryOperator::Add, index, col)
    }

    fn add_scaled_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        index: Handle<Expression>,
        component: Handle<Expression>,
        stride: u32,
    ) -> Handle<Expression> {
        if stride == 0 {
            return index;
        }
        let term = self.mul_lit(expressions, body, component, stride);
        self.bin(expressions, body, BinaryOperator::Add, index, term)
    }

    fn subgroup_sum(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        result_ty: Handle<Type>,
    ) -> Handle<Expression> {
        let result = expressions.append(
            Expression::SubgroupOperationResult { ty: result_ty },
            Span::default(),
        );
        body.push(
            Statement::SubgroupCollectiveOperation {
                op: SubgroupOperation::Add,
                collective_op: CollectiveOperation::Reduce,
                argument: value,
                result,
            },
            Span::default(),
        );
        result
    }

    fn splat_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        ty: Handle<Type>,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Compose {
                ty,
                components: vec![value, value, value, value],
            },
        )
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.storage_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.storage_ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn load_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.workgroup_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.workgroup_ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn storage_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn workgroup_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn eq_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Equal, value, rhs)
    }

    fn ge_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::GreaterEqual, value, rhs)
    }

    fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Add, value, rhs)
    }

    fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn emit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expression: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expression, Span::default());
        body.push(
            Statement::Emit(Range::new_from_bounds(handle, handle)),
            Span::default(),
        );
        handle
    }

    fn f32_lit(&self, expressions: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    fn u32_lit(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    fn storage_global(
        module: &mut Module,
        binding: u32,
        ty: Handle<Type>,
        read_only: bool,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: None,
                space: AddressSpace::Storage {
                    access: if read_only {
                        StorageAccess::LOAD
                    } else {
                        StorageAccess::LOAD | StorageAccess::STORE
                    },
                },
                binding: Some(ResourceBinding { group: 0, binding }),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn local(function: &mut Function, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }
}

pub(super) fn build_rms_norm_vec4_naga_module(meta: RmsNormVec4Meta) -> Option<Module> {
    RmsNormVec4NagaBuilder::new(meta).build()
}
