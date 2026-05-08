use std::num::NonZeroU32;

use wgpu::naga::{
    Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint, Expression,
    Function, FunctionArgument, GlobalVariable, Handle, MathFunction, Module, Scalar, ScalarKind,
    ShaderStage, Span, Statement, Type, TypeInner, VectorSize,
};

use super::{TopKGlobals, TopKLocals};
use crate::mir::kernel_backend::naga_helpers::{
    NagaBuilderExt, local, storage_global, workgroup_global,
};
use crate::sampling::{MAX_F32, NEG_MAX_F32, TOP_K_BLOCK, TOP_K_CHUNK};

impl super::TopKModuleBuilder {
    pub(super) fn new(
        input_len: u32,
        output_per_chunk: u32,
        input_offset: u32,
        input_stride: u32,
        processors: bool,
    ) -> Self {
        Self {
            input_len,
            output_per_chunk,
            input_offset,
            input_stride,
            processors,
        }
    }

    pub(super) fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let bool_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::BOOL),
            },
            Span::default(),
        );
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
        let scratch_f32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_u32_ty = module.types.insert(
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

        let globals = TopKGlobals {
            input: storage_global(&mut module, 0, f32_storage_ty, true),
            output_ids: storage_global(&mut module, 1, u32_storage_ty, false),
            output_values: storage_global(&mut module, 2, f32_storage_ty, false),
            previous_tokens: self
                .processors
                .then(|| storage_global(&mut module, 3, u32_storage_ty, true)),
            processor_params: self
                .processors
                .then(|| storage_global(&mut module, 4, u32_storage_ty, true)),
            scratch_values: workgroup_global(&mut module, scratch_f32_ty),
            scratch_ids: workgroup_global(&mut module, scratch_u32_ty),
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
            ],
            ..Function::default()
        };
        let locals = TopKLocals {
            current_value: local(&mut function, f32_ty),
            current_id: local(&mut function, u32_ty),
            previous_index: local(&mut function, u32_ty),
            repeated: local(&mut function, bool_ty),
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
        globals: TopKGlobals,
        locals: TopKLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let chunk = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );
        let neg_max = self.f32_lit(expressions, NEG_MAX_F32);
        let invalid_id = self.u32_lit(expressions, u32::MAX);
        self.store_local(expressions, &mut body, locals.current_value, neg_max);
        self.store_local(expressions, &mut body, locals.current_id, invalid_id);

        let chunk_base = self.mul_lit(expressions, &mut body, chunk, TOP_K_CHUNK as u32);
        let token_id = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Add,
            chunk_base,
            lane,
        );
        let input_len = self.u32_lit(expressions, self.input_len);
        let token_valid = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            token_id,
            input_len,
        );
        let mut load_accept = Block::new();
        let input_index = if self.input_stride == 1 {
            self.add_lit(expressions, &mut load_accept, token_id, self.input_offset)
        } else {
            let scaled = self.mul_lit(expressions, &mut load_accept, token_id, self.input_stride);
            self.add_lit(expressions, &mut load_accept, scaled, self.input_offset)
        };
        let value = self.load_storage(expressions, &mut load_accept, globals.input, input_index);
        let raw_finite = self.is_finite(expressions, &mut load_accept, value);
        let mut finite_accept = Block::new();
        let value = self.apply_processors(
            expressions,
            &mut finite_accept,
            &globals,
            &locals,
            value,
            token_id,
        );
        let finite = self.is_finite(expressions, &mut finite_accept, value);
        let mut processed_finite_accept = Block::new();
        self.store_local(
            expressions,
            &mut processed_finite_accept,
            locals.current_value,
            value,
        );
        self.store_local(
            expressions,
            &mut processed_finite_accept,
            locals.current_id,
            token_id,
        );
        finite_accept.push(
            Statement::If {
                condition: finite,
                accept: processed_finite_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        load_accept.push(
            Statement::If {
                condition: raw_finite,
                accept: finite_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: token_valid,
                accept: load_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let current_value = self.load_local(expressions, &mut body, locals.current_value);
        let current_id = self.load_local(expressions, &mut body, locals.current_id);
        self.store_storage(
            expressions,
            &mut body,
            globals.scratch_values,
            lane,
            current_value,
        );
        self.store_storage(
            expressions,
            &mut body,
            globals.scratch_ids,
            lane,
            current_id,
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut size = 2;
        while size <= TOP_K_BLOCK {
            let mut stride = size / 2;
            while stride > 0 {
                self.append_bitonic_stage(expressions, &mut body, &globals, lane, size, stride);
                stride /= 2;
            }
            size *= 2;
        }

        let output_per_chunk = self.u32_lit(expressions, self.output_per_chunk);
        let writes_output = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            lane,
            output_per_chunk,
        );
        let mut write_accept = Block::new();
        let chunk_base = self.mul_lit(expressions, &mut write_accept, chunk, self.output_per_chunk);
        let output_index = self.bin(
            expressions,
            &mut write_accept,
            BinaryOperator::Add,
            chunk_base,
            lane,
        );
        let selected_value =
            self.load_storage(expressions, &mut write_accept, globals.scratch_values, lane);
        let selected_id =
            self.load_storage(expressions, &mut write_accept, globals.scratch_ids, lane);
        self.store_storage(
            expressions,
            &mut write_accept,
            globals.output_values,
            output_index,
            selected_value,
        );
        self.store_storage(
            expressions,
            &mut write_accept,
            globals.output_ids,
            output_index,
            selected_id,
        );
        body.push(
            Statement::If {
                condition: writes_output,
                accept: write_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn append_bitonic_stage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &TopKGlobals,
        lane: Handle<Expression>,
        size: u32,
        stride: u32,
    ) {
        let stride_lit = self.u32_lit(expressions, stride);
        let partner = self.bin(
            expressions,
            body,
            BinaryOperator::ExclusiveOr,
            lane,
            stride_lit,
        );
        let current_value = self.load_storage(expressions, body, globals.scratch_values, lane);
        let current_id = self.load_storage(expressions, body, globals.scratch_ids, lane);
        let partner_value = self.load_storage(expressions, body, globals.scratch_values, partner);
        let partner_id = self.load_storage(expressions, body, globals.scratch_ids, partner);

        let stride_lit = self.u32_lit(expressions, stride);
        let lane_stride_bits = self.bin(expressions, body, BinaryOperator::And, lane, stride_lit);
        let size_lit = self.u32_lit(expressions, size);
        let lane_size_bits = self.bin(expressions, body, BinaryOperator::And, lane, size_lit);
        let zero = self.u32_lit(expressions, 0);
        let lower_lane = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lane_stride_bits,
            zero,
        );
        let descending = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lane_size_bits,
            zero,
        );
        let want_better = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lower_lane,
            descending,
        );

        let partner_better = self.better_candidate(
            expressions,
            body,
            partner_value,
            partner_id,
            current_value,
            current_id,
        );
        let current_better = self.better_candidate(
            expressions,
            body,
            current_value,
            current_id,
            partner_value,
            partner_id,
        );
        let false_lit = self.bool_lit(expressions, false);
        let want_worse = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            want_better,
            false_lit,
        );
        let choose_better_partner = self.and(expressions, body, want_better, partner_better);
        let choose_worse_partner = self.and(expressions, body, want_worse, current_better);
        let choose_partner = self.or(
            expressions,
            body,
            choose_better_partner,
            choose_worse_partner,
        );

        let mut accept = Block::new();
        self.store_storage(
            expressions,
            &mut accept,
            globals.scratch_values,
            lane,
            partner_value,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.scratch_ids,
            lane,
            partner_id,
        );
        body.push(
            Statement::If {
                condition: choose_partner,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn apply_processors(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &TopKGlobals,
        locals: &TopKLocals,
        value: Handle<Expression>,
        token_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let (Some(previous_tokens), Some(processor_params)) =
            (globals.previous_tokens, globals.processor_params)
        else {
            return value;
        };

        self.store_local(expressions, body, locals.current_value, value);
        let zero_u32 = self.u32_lit(expressions, 0);
        self.store_local(expressions, body, locals.previous_index, zero_u32);
        let false_lit = self.bool_lit(expressions, false);
        self.store_local(expressions, body, locals.repeated, false_lit);

        let previous_len_index = self.u32_lit(expressions, 2);
        let previous_len =
            self.load_storage(expressions, body, processor_params, previous_len_index);
        let mut scan_body = Block::new();
        let previous_index = self.load_local(expressions, &mut scan_body, locals.previous_index);
        let done = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::GreaterEqual,
            previous_index,
            previous_len,
        );
        scan_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let previous_index = self.load_local(expressions, &mut scan_body, locals.previous_index);
        let previous_token =
            self.load_storage(expressions, &mut scan_body, previous_tokens, previous_index);
        let repeated = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::Equal,
            previous_token,
            token_id,
        );
        let mut repeated_accept = Block::new();
        let true_lit = self.bool_lit(expressions, true);
        self.store_local(expressions, &mut repeated_accept, locals.repeated, true_lit);
        repeated_accept.push(Statement::Break, Span::default());
        scan_body.push(
            Statement::If {
                condition: repeated,
                accept: repeated_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let previous_index = self.load_local(expressions, &mut scan_body, locals.previous_index);
        let next_previous_index = self.add_lit(expressions, &mut scan_body, previous_index, 1);
        self.store_local(
            expressions,
            &mut scan_body,
            locals.previous_index,
            next_previous_index,
        );
        body.push(
            Statement::Loop {
                body: scan_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        let repetition_penalty =
            self.load_processor_param_f32(expressions, body, processor_params, 1);
        let repeated = self.load_local(expressions, body, locals.repeated);
        let one = self.f32_lit(expressions, 1.0);
        let penalty_gt_one = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            repetition_penalty,
            one,
        );
        let should_apply_penalty = self.and(expressions, body, repeated, penalty_gt_one);
        let mut penalty_accept = Block::new();
        let current = self.load_local(expressions, &mut penalty_accept, locals.current_value);
        let zero = self.f32_lit(expressions, 0.0);
        let non_positive = self.bin(
            expressions,
            &mut penalty_accept,
            BinaryOperator::LessEqual,
            current,
            zero,
        );
        let mut non_positive_accept = Block::new();
        let current = self.load_local(expressions, &mut non_positive_accept, locals.current_value);
        let penalized = self.bin(
            expressions,
            &mut non_positive_accept,
            BinaryOperator::Multiply,
            current,
            repetition_penalty,
        );
        self.store_local(
            expressions,
            &mut non_positive_accept,
            locals.current_value,
            penalized,
        );
        let mut positive_accept = Block::new();
        let current = self.load_local(expressions, &mut positive_accept, locals.current_value);
        let penalized = self.bin(
            expressions,
            &mut positive_accept,
            BinaryOperator::Divide,
            current,
            repetition_penalty,
        );
        self.store_local(
            expressions,
            &mut positive_accept,
            locals.current_value,
            penalized,
        );
        penalty_accept.push(
            Statement::If {
                condition: non_positive,
                accept: non_positive_accept,
                reject: positive_accept,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: should_apply_penalty,
                accept: penalty_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let temperature = self.load_processor_param_f32(expressions, body, processor_params, 0);
        let zero = self.f32_lit(expressions, 0.0);
        let temp_nonzero = self.bin(
            expressions,
            body,
            BinaryOperator::NotEqual,
            temperature,
            zero,
        );
        let mut temperature_accept = Block::new();
        let current = self.load_local(expressions, &mut temperature_accept, locals.current_value);
        let adjusted = self.bin(
            expressions,
            &mut temperature_accept,
            BinaryOperator::Divide,
            current,
            temperature,
        );
        self.store_local(
            expressions,
            &mut temperature_accept,
            locals.current_value,
            adjusted,
        );
        body.push(
            Statement::If {
                condition: temp_nonzero,
                accept: temperature_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        self.load_local(expressions, body, locals.current_value)
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

    fn better_candidate(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        id: Handle<Expression>,
        best_value: Handle<Expression>,
        best_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let value_greater = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            value,
            best_value,
        );
        let value_equal = self.bin(expressions, body, BinaryOperator::Equal, value, best_value);
        let id_greater = self.bin(expressions, body, BinaryOperator::Greater, id, best_id);
        let equal_and_id = self.and(expressions, body, value_equal, id_greater);
        self.or(expressions, body, value_greater, equal_and_id)
    }

    fn load_processor_param_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: u32,
    ) -> Handle<Expression> {
        let index = self.u32_lit(expressions, index);
        let bits = self.load_storage(expressions, body, global, index);
        self.emit(
            expressions,
            body,
            Expression::As {
                expr: bits,
                kind: ScalarKind::Float,
                convert: None,
            },
        )
    }
}
