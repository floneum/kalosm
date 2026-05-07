use wgpu::naga::{
    AddressSpace, Arena, Barrier, BinaryOperator, Block, Expression, Function, GlobalVariable,
    Handle, Literal, LocalVariable, MathFunction, Module, Range, ResourceBinding, Span, Statement,
    StorageAccess, Type,
};

use super::*;

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
        let score = self.score_for_kv(expressions, &mut loop_body, globals, indices, kv);
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
        let score = self.score_for_kv(expressions, &mut loop_body, globals, indices, kv);
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
        let score = self.score_for_kv(expressions, &mut prob_accept, globals, indices, kv);
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
        self.append_tiled_output_item_loop(
            expressions,
            &mut loop_body,
            globals,
            locals,
            indices,
            tile_base,
            local,
            active_kv_len,
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

    pub(super) fn index4_const_last(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        strides: [u32; 4],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
        i2: u32,
        i3: u32,
    ) -> Handle<Expression> {
        let base = offset + i2 * strides[2] + i3 * strides[3];
        self.index2_with_base(expressions, body, base, [strides[0], strides[1]], i0, i1)
    }

    pub(super) fn index4_const_last_dyn_i2(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        strides: [u32; 4],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
        i2: Handle<Expression>,
        i3: u32,
    ) -> Handle<Expression> {
        let base = offset + i3 * strides[3];
        let index =
            self.index2_with_base(expressions, body, base, [strides[0], strides[1]], i0, i1);
        self.add_scaled_index(expressions, body, index, i2, strides[2])
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

    pub(super) fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    pub(super) fn store_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(super) fn load_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    pub(super) fn store_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(super) fn ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    pub(super) fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    pub(super) fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(super) fn exp_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Exp,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(super) fn max_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Max,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(super) fn select(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Select {
                condition,
                accept,
                reject,
            },
        )
    }

    pub(super) fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    pub(super) fn lt_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Less, value, rhs)
    }

    pub(super) fn ge_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::GreaterEqual, value, rhs)
    }

    pub(super) fn div_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Divide, value, rhs)
    }

    pub(super) fn rem_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Modulo, value, rhs)
    }

    pub(super) fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Add, value, rhs)
    }

    pub(super) fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    pub(super) fn emit(
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

    pub(super) fn f32_lit(
        &self,
        expressions: &mut Arena<Expression>,
        value: f32,
    ) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    pub(super) fn u32_lit(
        &self,
        expressions: &mut Arena<Expression>,
        value: u32,
    ) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    pub(super) fn storage_global(
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

    pub(super) fn workgroup_global(
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

    pub(super) fn local(
        function: &mut Function,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }
}
