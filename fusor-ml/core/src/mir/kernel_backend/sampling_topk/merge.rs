use wgpu::naga::{
    Arena, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint, Expression, Function,
    FunctionArgument, Handle, MathFunction, Module, Scalar, ShaderStage, Span, Statement,
};

use super::{MergeTopKGlobals, MergeTopKLocals};
use crate::mir::kernel_backend::naga_helpers::{
    NagaBuilderExt, constant_array_type, dynamic_array_type, local, scalar_type, storage_global,
    workgroup_global,
};
use crate::sampling::{MAX_F32, NEG_MAX_F32, TOP_K_BLOCK};

impl super::MergeTopKModuleBuilder {
    pub(super) fn new(
        chunks: u32,
        chunk_len: u32,
        chunk_stride: u32,
        input_len: u32,
        k: u32,
    ) -> Self {
        Self {
            chunks,
            chunk_len,
            chunk_stride,
            input_len,
            k,
        }
    }

    pub(super) fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = scalar_type(&mut module, Scalar::F32);
        let u32_ty = scalar_type(&mut module, Scalar::U32);
        let f32_storage_ty = dynamic_array_type(&mut module, f32_ty, 4);
        let u32_storage_ty = dynamic_array_type(&mut module, u32_ty, 4);
        let chunk_positions_ty = constant_array_type(&mut module, u32_ty, self.chunks, 4)?;
        let scratch_f32_ty = constant_array_type(&mut module, f32_ty, TOP_K_BLOCK, 4)?;
        let scratch_u32_ty = constant_array_type(&mut module, u32_ty, TOP_K_BLOCK, 4)?;

        let globals = MergeTopKGlobals {
            input_ids: storage_global(&mut module, 0, u32_storage_ty, true),
            input_values: storage_global(&mut module, 1, f32_storage_ty, true),
            output_ids: storage_global(&mut module, 2, u32_storage_ty, false),
            output_values: storage_global(&mut module, 3, f32_storage_ty, false),
            chunk_positions: workgroup_global(&mut module, chunk_positions_ty),
            scratch_values: workgroup_global(&mut module, scratch_f32_ty),
            scratch_ids: workgroup_global(&mut module, scratch_u32_ty),
            scratch_chunks: workgroup_global(&mut module, scratch_u32_ty),
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
        let locals = MergeTopKLocals {
            rank: local(&mut function, u32_ty),
            scan_chunk: local(&mut function, u32_ty),
            local_best_value: local(&mut function, f32_ty),
            local_best_id: local(&mut function, u32_ty),
            local_best_chunk: local(&mut function, u32_ty),
            reduce_step: local(&mut function, u32_ty),
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
        globals: MergeTopKGlobals,
        locals: MergeTopKLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());

        self.store_local(expressions, &mut body, locals.scan_chunk, lane);
        let mut init_body = Block::new();
        let chunk = self.load_local(expressions, &mut init_body, locals.scan_chunk);
        let done = self.ge_lit(expressions, &mut init_body, chunk, self.chunks);
        init_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let zero = self.u32_lit(expressions, 0);
        self.store_storage(
            expressions,
            &mut init_body,
            globals.chunk_positions,
            chunk,
            zero,
        );
        let chunk = self.load_local(expressions, &mut init_body, locals.scan_chunk);
        let next = self.add_lit(expressions, &mut init_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut init_body, locals.scan_chunk, next);
        body.push(
            Statement::Loop {
                body: init_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let zero = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.rank, zero);
        let mut rank_body = Block::new();
        let rank = self.load_local(expressions, &mut rank_body, locals.rank);
        let done = self.ge_lit(expressions, &mut rank_body, rank, self.k);
        rank_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let neg_max = self.f32_lit(expressions, NEG_MAX_F32);
        let invalid = self.u32_lit(expressions, u32::MAX);
        self.store_local(
            expressions,
            &mut rank_body,
            locals.local_best_value,
            neg_max,
        );
        self.store_local(expressions, &mut rank_body, locals.local_best_id, invalid);
        self.store_local(
            expressions,
            &mut rank_body,
            locals.local_best_chunk,
            invalid,
        );
        self.store_local(expressions, &mut rank_body, locals.scan_chunk, lane);

        self.append_scan_chunks_loop(expressions, &mut rank_body, &globals, &locals);
        self.store_local_best_to_scratch(expressions, &mut rank_body, &globals, &locals, lane);
        self.append_reduce_loop(expressions, &mut rank_body, &globals, &locals, lane);
        self.store_rank_output(expressions, &mut rank_body, &globals, &locals, lane);

        let rank = self.load_local(expressions, &mut rank_body, locals.rank);
        let next_rank = self.add_lit(expressions, &mut rank_body, rank, 1);
        self.store_local(expressions, &mut rank_body, locals.rank, next_rank);
        body.push(
            Statement::Loop {
                body: rank_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        body
    }

    fn append_scan_chunks_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
    ) {
        let mut scan_body = Block::new();
        let chunk = self.load_local(expressions, &mut scan_body, locals.scan_chunk);
        let done = self.ge_lit(expressions, &mut scan_body, chunk, self.chunks);
        scan_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let position =
            self.load_storage(expressions, &mut scan_body, globals.chunk_positions, chunk);
        let chunk_len = self.u32_lit(expressions, self.chunk_len);
        let in_chunk = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::Less,
            position,
            chunk_len,
        );
        let mut candidate_accept = Block::new();
        let chunk_offset =
            self.mul_lit(expressions, &mut candidate_accept, chunk, self.chunk_stride);
        let index = self.bin(
            expressions,
            &mut candidate_accept,
            BinaryOperator::Add,
            chunk_offset,
            position,
        );
        let id = self.load_storage(expressions, &mut candidate_accept, globals.input_ids, index);
        let input_len = self.u32_lit(expressions, self.input_len);
        let valid_id = self.bin(
            expressions,
            &mut candidate_accept,
            BinaryOperator::Less,
            id,
            input_len,
        );
        let value = self.load_storage(
            expressions,
            &mut candidate_accept,
            globals.input_values,
            index,
        );
        let finite = self.is_finite(expressions, &mut candidate_accept, value);
        let valid = self.and(expressions, &mut candidate_accept, valid_id, finite);
        let best_value =
            self.load_local(expressions, &mut candidate_accept, locals.local_best_value);
        let best_id = self.load_local(expressions, &mut candidate_accept, locals.local_best_id);
        let better = self.better_candidate(
            expressions,
            &mut candidate_accept,
            value,
            id,
            best_value,
            best_id,
        );
        let should_update = self.and(expressions, &mut candidate_accept, valid, better);
        let mut update = Block::new();
        self.store_local(expressions, &mut update, locals.local_best_value, value);
        self.store_local(expressions, &mut update, locals.local_best_id, id);
        self.store_local(expressions, &mut update, locals.local_best_chunk, chunk);
        candidate_accept.push(
            Statement::If {
                condition: should_update,
                accept: update,
                reject: Block::new(),
            },
            Span::default(),
        );
        scan_body.push(
            Statement::If {
                condition: in_chunk,
                accept: candidate_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let chunk = self.load_local(expressions, &mut scan_body, locals.scan_chunk);
        let next = self.add_lit(expressions, &mut scan_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut scan_body, locals.scan_chunk, next);
        body.push(
            Statement::Loop {
                body: scan_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn store_local_best_to_scratch(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let value = self.load_local(expressions, body, locals.local_best_value);
        let id = self.load_local(expressions, body, locals.local_best_id);
        let chunk = self.load_local(expressions, body, locals.local_best_chunk);
        self.store_storage(expressions, body, globals.scratch_values, lane, value);
        self.store_storage(expressions, body, globals.scratch_ids, lane, id);
        self.store_storage(expressions, body, globals.scratch_chunks, lane, chunk);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn append_reduce_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let half_block = self.u32_lit(expressions, TOP_K_BLOCK / 2);
        self.store_local(expressions, body, locals.reduce_step, half_block);

        let mut reduce_body = Block::new();
        let step = self.load_local(expressions, &mut reduce_body, locals.reduce_step);
        let done = self.eq_lit(expressions, &mut reduce_body, step, 0);
        reduce_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let participates = self.bin(
            expressions,
            &mut reduce_body,
            BinaryOperator::Less,
            lane,
            step,
        );
        let mut accept = Block::new();
        let other_index = self.bin(expressions, &mut accept, BinaryOperator::Add, lane, step);
        let other_value = self.load_storage(
            expressions,
            &mut accept,
            globals.scratch_values,
            other_index,
        );
        let other_id =
            self.load_storage(expressions, &mut accept, globals.scratch_ids, other_index);
        let other_chunk = self.load_storage(
            expressions,
            &mut accept,
            globals.scratch_chunks,
            other_index,
        );
        let current_value =
            self.load_storage(expressions, &mut accept, globals.scratch_values, lane);
        let current_id = self.load_storage(expressions, &mut accept, globals.scratch_ids, lane);
        let better = self.better_candidate(
            expressions,
            &mut accept,
            other_value,
            other_id,
            current_value,
            current_id,
        );
        let mut better_accept = Block::new();
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_values,
            lane,
            other_value,
        );
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_ids,
            lane,
            other_id,
        );
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_chunks,
            lane,
            other_chunk,
        );
        accept.push(
            Statement::If {
                condition: better,
                accept: better_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        reduce_body.push(
            Statement::If {
                condition: participates,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        reduce_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let step = self.load_local(expressions, &mut reduce_body, locals.reduce_step);
        let two = self.u32_lit(expressions, 2);
        let next_step = self.bin(
            expressions,
            &mut reduce_body,
            BinaryOperator::Divide,
            step,
            two,
        );
        self.store_local(expressions, &mut reduce_body, locals.reduce_step, next_step);
        body.push(
            Statement::Loop {
                body: reduce_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn store_rank_output(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let lane_zero = self.eq_lit(expressions, body, lane, 0);
        let mut accept = Block::new();
        let zero = self.u32_lit(expressions, 0);
        let selected_value =
            self.load_storage(expressions, &mut accept, globals.scratch_values, zero);
        let zero = self.u32_lit(expressions, 0);
        let selected_id = self.load_storage(expressions, &mut accept, globals.scratch_ids, zero);
        let zero = self.u32_lit(expressions, 0);
        let selected_chunk =
            self.load_storage(expressions, &mut accept, globals.scratch_chunks, zero);
        let rank = self.load_local(expressions, &mut accept, locals.rank);
        self.store_storage(
            expressions,
            &mut accept,
            globals.output_values,
            rank,
            selected_value,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.output_ids,
            rank,
            selected_id,
        );

        let chunks = self.u32_lit(expressions, self.chunks);
        let valid_chunk = self.bin(
            expressions,
            &mut accept,
            BinaryOperator::Less,
            selected_chunk,
            chunks,
        );
        let mut advance = Block::new();
        let position = self.load_storage(
            expressions,
            &mut advance,
            globals.chunk_positions,
            selected_chunk,
        );
        let next_position = self.add_lit(expressions, &mut advance, position, 1);
        self.store_storage(
            expressions,
            &mut advance,
            globals.chunk_positions,
            selected_chunk,
            next_position,
        );
        accept.push(
            Statement::If {
                condition: valid_chunk,
                accept: advance,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: lane_zero,
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
}
