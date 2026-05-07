use std::num::NonZeroU32;

use wgpu::naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, Range, ResourceBinding, Scalar, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner,
};

use super::{TopKExactnessGlobals, TopKExactnessLocals, TopKExactnessMeta};
use crate::sampling::{MAX_F32, TOP_K_BLOCK};

impl super::TopKExactnessModuleBuilder {
    pub(super) fn new(meta: TopKExactnessMeta) -> Self {
        Self { meta }
    }

    pub(super) fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("TopKExactF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("TopKExactU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKExactF32Buffer".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let u32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKExactU32Buffer".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_ty = module.types.insert(
            Type {
                name: Some("TopKExactScratch".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = TopKExactnessGlobals {
            top_values: Self::storage_global(&mut module, "top_values", 0, f32_storage_ty, true),
            chunk_values: Self::storage_global(
                &mut module,
                "chunk_values",
                1,
                f32_storage_ty,
                true,
            ),
            flag: Self::storage_global(&mut module, "flag", 2, u32_storage_ty, false),
            scratch: Self::workgroup_global(&mut module, "scratch", scratch_ty),
        };

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let locals = TopKExactnessLocals {
            chunk: Self::local(&mut function, "chunk", u32_ty),
            inexact: Self::local(&mut function, "inexact", u32_ty),
        };
        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [TOP_K_BLOCK, 1, 1],
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
        globals: TopKExactnessGlobals,
        locals: TopKExactnessLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());
        let threshold_rank = self.u32_lit(expressions, self.meta.top_k - 1);
        let threshold_index = self.index1(
            expressions,
            &mut body,
            self.meta.top_values_offset,
            self.meta.top_values_stride,
            threshold_rank,
        );
        let threshold =
            self.load_storage(expressions, &mut body, globals.top_values, threshold_index);
        let threshold_finite = self.is_finite(expressions, &mut body, threshold);

        let zero = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.inexact, zero);
        self.store_local(expressions, &mut body, locals.chunk, lane);
        self.append_scan_loop(
            expressions,
            &mut body,
            globals,
            locals,
            threshold,
            threshold_finite,
        );

        let inexact = self.load_local(expressions, &mut body, locals.inexact);
        self.store_workgroup(expressions, &mut body, globals.scratch, lane, inexact);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut stride = TOP_K_BLOCK / 2;
        while stride > 0 {
            let participates = self.lt_lit(expressions, &mut body, lane, stride);
            let mut accept = Block::new();
            let rhs_index = self.add_lit(expressions, &mut accept, lane, stride);
            let lhs = self.load_workgroup(expressions, &mut accept, globals.scratch, lane);
            let rhs = self.load_workgroup(expressions, &mut accept, globals.scratch, rhs_index);
            let merged = self.bin(
                expressions,
                &mut accept,
                BinaryOperator::InclusiveOr,
                lhs,
                rhs,
            );
            self.store_workgroup(expressions, &mut accept, globals.scratch, lane, merged);
            body.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let lane_zero = self.bin(expressions, &mut body, BinaryOperator::Equal, lane, zero);
        let mut store_accept = Block::new();
        let root = self.load_workgroup(expressions, &mut store_accept, globals.scratch, zero);
        let exact = self.bin(
            expressions,
            &mut store_accept,
            BinaryOperator::Equal,
            root,
            zero,
        );
        let mut exact_accept = Block::new();
        let one = self.u32_lit(expressions, 1);
        self.store_storage(expressions, &mut exact_accept, globals.flag, zero, one);
        let mut exact_reject = Block::new();
        self.store_storage(expressions, &mut exact_reject, globals.flag, zero, zero);
        store_accept.push(
            Statement::If {
                condition: exact,
                accept: exact_accept,
                reject: exact_reject,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: lane_zero,
                accept: store_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn append_scan_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: TopKExactnessGlobals,
        locals: TopKExactnessLocals,
        threshold: Handle<Expression>,
        threshold_finite: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let chunk = self.load_local(expressions, &mut loop_body, locals.chunk);
        let chunks = self.u32_lit(expressions, self.meta.chunks);
        let done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            chunk,
            chunks,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let bound_rank = self.mul_lit(
            expressions,
            &mut loop_body,
            chunk,
            self.meta.output_per_chunk,
        );
        let bound_rank = self.add_lit(
            expressions,
            &mut loop_body,
            bound_rank,
            self.meta.candidate_count,
        );
        let bound_index = self.index1(
            expressions,
            &mut loop_body,
            self.meta.chunk_values_offset,
            self.meta.chunk_values_stride,
            bound_rank,
        );
        let bound = self.load_storage(
            expressions,
            &mut loop_body,
            globals.chunk_values,
            bound_index,
        );
        let bound_finite = self.is_finite(expressions, &mut loop_body, bound);
        let bound_ge_threshold = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            bound,
            threshold,
        );
        let finite_bound_inexact = self.and(
            expressions,
            &mut loop_body,
            bound_finite,
            bound_ge_threshold,
        );
        let finite_inexact = self.and(
            expressions,
            &mut loop_body,
            threshold_finite,
            finite_bound_inexact,
        );
        let false_lit = self.bool_lit(expressions, false);
        let threshold_not_finite = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Equal,
            threshold_finite,
            false_lit,
        );
        let nonfinite_inexact = self.and(
            expressions,
            &mut loop_body,
            threshold_not_finite,
            bound_finite,
        );
        let inexact = self.or(
            expressions,
            &mut loop_body,
            finite_inexact,
            nonfinite_inexact,
        );
        let mut inexact_accept = Block::new();
        let one = self.u32_lit(expressions, 1);
        self.store_local(expressions, &mut inexact_accept, locals.inexact, one);
        loop_body.push(
            Statement::If {
                condition: inexact,
                accept: inexact_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let next_chunk = self.add_lit(expressions, &mut loop_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.chunk, next_chunk);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn storage_global(
        module: &mut Module,
        name: &str,
        binding: u32,
        ty: Handle<Type>,
        read_only: bool,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
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

    fn workgroup_global(
        module: &mut Module,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn local(function: &mut Function, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn index1(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        stride: u32,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let scaled = if stride == 1 {
            index
        } else {
            self.mul_lit(expressions, body, index, stride)
        };
        self.add_lit(expressions, body, scaled, offset)
    }

    fn is_finite(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let self_equal = self.bin(expressions, body, BinaryOperator::Equal, value, value);
        let abs = self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Abs,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        let max = self.f32_lit(expressions, MAX_F32);
        let finite_magnitude = self.bin(expressions, body, BinaryOperator::LessEqual, abs, max);
        self.and(expressions, body, self_equal, finite_magnitude)
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
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
        let pointer = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn load_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
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
        let pointer = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        base: Expression,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(base, Span::default());
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

    fn lt_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Less, value, rhs)
    }

    fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            value
        } else {
            let rhs = self.u32_lit(expressions, literal);
            self.bin(expressions, body, BinaryOperator::Add, value, rhs)
        }
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

    fn and(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalAnd, left, right)
    }

    fn or(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalOr, left, right)
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

    fn bool_lit(&self, expressions: &mut Arena<Expression>, value: bool) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::Bool(value)), Span::default())
    }
}
