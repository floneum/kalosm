use wgpu::naga::{
    Arena, Barrier, BinaryOperator, Block, Expression, GlobalVariable, Handle, Span, Statement,
};

use super::*;
use crate::mir::kernel_backend::naga_helpers::NagaBuilderExt;

impl super::FlashDecodeSmallNagaBuilder {
    pub(super) fn append_tiled_max_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
        indices: FlashDecodeRowIndices,
        local: Handle<Expression>,
        active_kv_len: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let kv = self.load_local(expressions, &mut loop_body, locals.kv);
        self.append_break_if_kv_done(expressions, &mut loop_body, kv, active_kv_len);
        let score = self.score_for_kv(expressions, &mut loop_body, globals, locals, indices, kv);
        let current = self.load_workgroup(expressions, &mut loop_body, globals.reduce, local);
        let next = self.max_f32(expressions, &mut loop_body, current, score);
        self.store_workgroup(expressions, &mut loop_body, globals.reduce, local, next);
        let next_kv = self.add_lit(expressions, &mut loop_body, kv, self.meta.decode_block);
        self.store_local(expressions, &mut loop_body, locals.kv, next_kv);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    pub(super) fn append_tiled_sum_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
        indices: FlashDecodeRowIndices,
        local: Handle<Expression>,
        active_kv_len: Handle<Expression>,
        max_score: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let kv = self.load_local(expressions, &mut loop_body, locals.kv);
        self.append_break_if_kv_done(expressions, &mut loop_body, kv, active_kv_len);
        let score = self.score_for_kv(expressions, &mut loop_body, globals, locals, indices, kv);
        let shifted = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Subtract,
            score,
            max_score,
        );
        let prob = self.exp_f32(expressions, &mut loop_body, shifted);
        let current = self.load_workgroup(expressions, &mut loop_body, globals.reduce, local);
        let next = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            current,
            prob,
        );
        self.store_workgroup(expressions, &mut loop_body, globals.reduce, local, next);
        let next_kv = self.add_lit(expressions, &mut loop_body, kv, self.meta.decode_block);
        self.store_local(expressions, &mut loop_body, locals.kv, next_kv);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    pub(super) fn append_tiled_output_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
        indices: FlashDecodeRowIndices,
        local: Handle<Expression>,
        active_kv_len: Handle<Expression>,
        max_score: Handle<Expression>,
        denom: Handle<Expression>,
        out_valid: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let tile_base = self.load_local(expressions, &mut loop_body, locals.kv);
        self.append_break_if_kv_done(expressions, &mut loop_body, tile_base, active_kv_len);
        let kv = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            tile_base,
            local,
        );
        let kv_valid = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Less,
            kv,
            active_kv_len,
        );
        let mut prob_accept = Block::new();
        let score = self.score_for_kv(expressions, &mut prob_accept, globals, locals, indices, kv);
        let shifted = self.bin(
            expressions,
            &mut prob_accept,
            BinaryOperator::Subtract,
            score,
            max_score,
        );
        let prob = self.exp_f32(expressions, &mut prob_accept, shifted);
        let prob = self.bin(
            expressions,
            &mut prob_accept,
            BinaryOperator::Divide,
            prob,
            denom,
        );
        self.store_workgroup(expressions, &mut prob_accept, globals.probs, local, prob);
        let mut prob_reject = Block::new();
        let zero = self.f32_lit(expressions, 0.0);
        self.store_workgroup(expressions, &mut prob_reject, globals.probs, local, zero);
        loop_body.push(
            Statement::If {
                condition: kv_valid,
                accept: prob_accept,
                reject: prob_reject,
            },
            Span::default(),
        );
        loop_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let zero_u32 = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut loop_body, locals.item, zero_u32);
        let mut output_accept = Block::new();
        self.append_tiled_output_item_loop(
            expressions,
            &mut output_accept,
            globals,
            locals,
            indices,
            tile_base,
            local,
            active_kv_len,
        );
        loop_body.push(
            Statement::If {
                condition: out_valid,
                accept: output_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let next_tile = self.add_lit(
            expressions,
            &mut loop_body,
            tile_base,
            self.meta.decode_block,
        );
        self.store_local(expressions, &mut loop_body, locals.kv, next_tile);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    pub(super) fn append_tiled_output_item_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
        indices: FlashDecodeRowIndices,
        tile_base: Handle<Expression>,
        out_dim: Handle<Expression>,
        active_kv_len: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let item = self.load_local(expressions, &mut loop_body, locals.item);
        let block_done = self.ge_lit(expressions, &mut loop_body, item, self.meta.decode_block);
        let kv = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            tile_base,
            item,
        );
        let kv_done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            kv,
            active_kv_len,
        );
        let done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::LogicalOr,
            block_done,
            kv_done,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let prob = self.load_workgroup(expressions, &mut loop_body, globals.probs, item);
        let v_index = self.index4_dyn_last(
            expressions,
            &mut loop_body,
            self.meta.v_offset,
            self.meta.v_strides,
            indices.batch_idx,
            indices.kv_head_idx,
            kv,
            out_dim,
        );
        let v_value = self.load_storage(expressions, &mut loop_body, globals.v, v_index);
        let weighted = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            prob,
            v_value,
        );
        let acc = self.load_local(expressions, &mut loop_body, locals.acc);
        let acc = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            acc,
            weighted,
        );
        self.store_local(expressions, &mut loop_body, locals.acc, acc);
        let next_item = self.add_lit(expressions, &mut loop_body, item, 1);
        self.store_local(expressions, &mut loop_body, locals.item, next_item);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    pub(super) fn append_break_if_kv_done(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        kv: Handle<Expression>,
        active_kv_len: Handle<Expression>,
    ) {
        let done = self.bin(
            expressions,
            body,
            BinaryOperator::GreaterEqual,
            kv,
            active_kv_len,
        );
        body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
    }

    pub(super) fn append_output_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
        batch_idx: Handle<Expression>,
        head_idx: Handle<Expression>,
        kv_head_idx: Handle<Expression>,
        out_dim: Handle<Expression>,
        active_kv_len: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let kv = self.load_local(expressions, &mut loop_body, locals.kv);
        let done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            kv,
            active_kv_len,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let prob = self.load_workgroup(expressions, &mut loop_body, globals.probs, kv);
        let v_index = self.index4_dyn_last(
            expressions,
            &mut loop_body,
            self.meta.v_offset,
            self.meta.v_strides,
            batch_idx,
            kv_head_idx,
            kv,
            out_dim,
        );
        let v_value = self.load_storage(expressions, &mut loop_body, globals.v, v_index);
        let weighted = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            prob,
            v_value,
        );
        let acc = self.load_local(expressions, &mut loop_body, locals.acc);
        let acc = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            acc,
            weighted,
        );
        self.store_local(expressions, &mut loop_body, locals.acc, acc);
        let next_kv = self.add_lit(expressions, &mut loop_body, kv, 1);
        self.store_local(expressions, &mut loop_body, locals.kv, next_kv);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        let output_value = self.load_local(expressions, body, locals.acc);
        let q_idx = self.u32_lit(expressions, 0);
        let output_index = self.index4_dyn_last(
            expressions,
            body,
            self.meta.output_offset,
            self.meta.output_strides,
            batch_idx,
            head_idx,
            q_idx,
            out_dim,
        );
        self.store_storage(
            expressions,
            body,
            globals.output,
            output_index,
            output_value,
        );
    }

    pub(super) fn reduce_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        scratch: Handle<GlobalVariable>,
        local: Handle<Expression>,
        op: FlashReduceOp,
    ) {
        let mut stride = self.meta.decode_block / 2;
        while stride > 0 {
            let participates = self.lt_lit(expressions, body, local, stride);
            let mut accept = Block::new();
            let left = self.load_workgroup(expressions, &mut accept, scratch, local);
            let rhs_index = self.add_lit(expressions, &mut accept, local, stride);
            let right = self.load_workgroup(expressions, &mut accept, scratch, rhs_index);
            let reduced = match op {
                FlashReduceOp::Sum => {
                    self.bin(expressions, &mut accept, BinaryOperator::Add, left, right)
                }
                FlashReduceOp::Max => self.max_f32(expressions, &mut accept, left, right),
            };
            self.store_workgroup(expressions, &mut accept, scratch, local, reduced);
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
    }

    pub(super) fn index4_dyn_last(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        strides: [u32; 4],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
        i2: Handle<Expression>,
        i3: Handle<Expression>,
    ) -> Handle<Expression> {
        let index =
            self.index2_with_base(expressions, body, offset, [strides[0], strides[1]], i0, i1);
        let index = self.add_scaled_index(expressions, body, index, i2, strides[2]);
        self.add_scaled_index(expressions, body, index, i3, strides[3])
    }

    pub(super) fn index2_with_base(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        base: u32,
        strides: [u32; 2],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = self.u32_lit(expressions, base);
        let index = self.add_scaled_index(expressions, body, base, i0, strides[0]);
        self.add_scaled_index(expressions, body, index, i1, strides[1])
    }

    pub(super) fn add_scaled_index(
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
}
