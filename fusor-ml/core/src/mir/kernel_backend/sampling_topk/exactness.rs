use std::num::NonZeroU32;

use wgpu::naga::{
    Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint, Expression,
    Function, FunctionArgument, Handle, MathFunction, Module, Scalar, ShaderStage, Span, Statement,
    Type, TypeInner,
};

use super::{TopKExactnessGlobals, TopKExactnessLocals, TopKExactnessMeta};
use crate::mir::kernel_backend::naga_helpers::{
    NagaBuilderExt, local, storage_global, workgroup_global,
};
use crate::sampling::{MAX_F32, TOP_K_BLOCK};

impl super::TopKExactnessModuleBuilder {
    pub(super) fn new(meta: TopKExactnessMeta) -> Self {
        Self { meta }
    }

    pub(super) fn build(self) -> Option<Module> {
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
        let f32_storage_ty = module.types.insert(
            Type {
                name: None,
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
                name: None,
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
                name: None,
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = TopKExactnessGlobals {
            top_values: storage_global(&mut module, 0, f32_storage_ty, true),
            chunk_values: storage_global(&mut module, 1, f32_storage_ty, true),
            flag: storage_global(&mut module, 2, u32_storage_ty, false),
            scratch: workgroup_global(&mut module, scratch_ty),
        };

        let mut function = Function {
            name: None,
            arguments: vec![FunctionArgument {
                name: None,
                ty: u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let locals = TopKExactnessLocals {
            chunk: local(&mut function, u32_ty),
            inexact: local(&mut function, u32_ty),
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
}
