use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_gemv(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemvOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(&op.a)?;
        let x_layout = self.storage_layout(&op.x)?;
        let y_layout = self.storage_layout(&op.y)?;
        let partial_layout = self.tile_layout(op.partials)?;
        let [m, k] = Self::matrix_shape(a_layout)?;
        let [x_k, x_cols] = Self::matrix_shape(x_layout)?;
        let [y_m, y_cols] = Self::matrix_shape(y_layout)?;

        if k != x_k || x_cols != 1 || m != y_m || y_cols != 1 {
            return Err(LowerError::UnsupportedOperation("gemv shape mismatch"));
        }
        if op.rows_per_workgroup == 0 || op.rows_per_workgroup > 4 {
            return Err(LowerError::UnsupportedOperation(
                "gemv rows per workgroup must be between 1 and 4",
            ));
        }
        if m % op.rows_per_workgroup != 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemv rows per workgroup must divide M",
            ));
        }
        if partial_layout.memory_level() != MemoryLevel::Workgroup {
            return Err(LowerError::UnsupportedMemoryLevel(
                partial_layout.memory_level(),
            ));
        }
        if partial_layout.shape().rank() != 1
            || partial_layout.element_count().get()
                != self.workgroup_invocations * op.rows_per_workgroup
        {
            return Err(LowerError::UnsupportedOperation(
                "gemv partials must match the selected workgroup size",
            ));
        }
        if op.vector_width == 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemv vector width must be non-zero",
            ));
        }

        let mut body = Block::new();
        let workgroup_id = expressions.append(
            Expression::FunctionArgument(WORKGROUP_ID_ARG),
            Span::default(),
        );
        let row = expressions.append(
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
            Span::default(),
        );
        let mut row_emits = Vec::new();
        let row_base =
            self.mul_literal_u32_emitted(expressions, row, op.rows_per_workgroup, &mut row_emits);
        if row_emits.is_empty() {
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, row_base)),
                Span::default(),
            );
        } else {
            Self::push_emits(&mut body, row_emits);
        }

        let row_limit = expressions.append(Expression::Literal(Literal::U32(m)), Span::default());
        let row_done = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: row_base,
                right: row_limit,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row_done)),
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: row_done,
                accept: Block::from_vec(vec![Statement::Return { value: None }]),
                reject: Block::new(),
            },
            Span::default(),
        );

        for row_offset in 0..op.rows_per_workgroup {
            let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
            let sum_pointer = expressions.append(
                Expression::LocalVariable(self.gemv_sum_local(scratch, row_offset)?),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value: zero,
                },
                Span::default(),
            );
        }

        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let mut start_emits = Vec::new();
        let k_start = self.mul_literal_u32_emitted(
            expressions,
            local_invocation,
            op.vector_width,
            &mut start_emits,
        );
        Self::push_emits(&mut body, start_emits);
        let k_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_k), Span::default());
        body.push(
            Statement::Store {
                pointer: k_pointer,
                value: k_start,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k_index, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        loop_body.push(Statement::Emit(k_emit), Span::default());
        let k_limit = expressions.append(Expression::Literal(Literal::U32(k)), Span::default());
        let k_done = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: k_index,
                right: k_limit,
            },
            Span::default(),
        );
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, k_done)),
            Span::default(),
        );
        loop_body.push(
            Statement::If {
                condition: k_done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let needs_tail_checks = k % op.vector_width != 0;
        for lane in 0..op.vector_width {
            let mut lane_emits = Vec::new();
            let k_lane = self.add_literal_u32_emitted(expressions, k_index, lane, &mut lane_emits);
            Self::push_emits(&mut loop_body, lane_emits);
            let lane_body = self.lower_gemv_fma_lane(expressions, scratch, op, row_base, k_lane)?;
            if needs_tail_checks && lane != 0 {
                let k_limit =
                    expressions.append(Expression::Literal(Literal::U32(k)), Span::default());
                let in_bounds = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Less,
                        left: k_lane,
                        right: k_limit,
                    },
                    Span::default(),
                );
                loop_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, in_bounds)),
                    Span::default(),
                );
                loop_body.push(
                    Statement::If {
                        condition: in_bounds,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            } else {
                loop_body.push(Statement::Block(lane_body), Span::default());
            }
        }

        let stride = self
            .workgroup_invocations
            .checked_mul(op.vector_width)
            .ok_or(LowerError::UnsupportedOperation("gemv stride overflow"))?;
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    scratch.mma_k,
                    stride,
                )]),
                break_if: None,
            },
            Span::default(),
        );

        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        for row_offset in 0..op.rows_per_workgroup {
            let mut partial_index_emits = Vec::new();
            let partial_index = self.add_literal_u32_emitted(
                expressions,
                local_invocation,
                row_offset * self.workgroup_invocations,
                &mut partial_index_emits,
            );
            Self::push_emits(&mut body, partial_index_emits);
            let (partial_pointer, partial_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.partials, partial_index)?;
            Self::push_emits(&mut body, partial_pointer_emits);
            let sum_pointer = expressions.append(
                Expression::LocalVariable(self.gemv_sum_local(scratch, row_offset)?),
                Span::default(),
            );
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: partial_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut reduce_stride = self.workgroup_invocations / 2;
        while reduce_stride > 1 {
            let local_invocation = expressions.append(
                Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
                Span::default(),
            );
            let limit = expressions.append(
                Expression::Literal(Literal::U32(reduce_stride)),
                Span::default(),
            );
            let participates = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: local_invocation,
                    right: limit,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, participates)),
                Span::default(),
            );
            body.push(
                Statement::If {
                    condition: participates,
                    accept: self.lower_gemv_partial_add(
                        expressions,
                        op.partials,
                        local_invocation,
                        reduce_stride,
                        op.rows_per_workgroup,
                    )?,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            reduce_stride /= 2;
        }

        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let is_lane_zero = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Equal,
                left: local_invocation,
                right: zero,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, is_lane_zero)),
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: is_lane_zero,
                accept: self.lower_gemv_final_store(expressions, op, row_base)?,
                reject: Block::new(),
            },
            Span::default(),
        );

        Ok(Statement::Block(body))
    }

    pub(super) fn lower_gemv_fma_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemvOp,
        row_base: Handle<Expression>,
        k: Handle<Expression>,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let (x_index, x_index_emits) =
            self.storage_index_from_coords(expressions, &op.x, &[k, zero])?;
        Self::push_emits(&mut body, x_index_emits);
        let (x_pointer, x_pointer_emits) =
            self.storage_dynamic_pointer(expressions, &op.x, x_index)?;
        Self::push_emits(&mut body, x_pointer_emits);
        let x_value = expressions.append(Expression::Load { pointer: x_pointer }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, x_value)),
            Span::default(),
        );

        for row_offset in 0..op.rows_per_workgroup {
            let mut row_emits = Vec::new();
            let row =
                self.add_literal_u32_emitted(expressions, row_base, row_offset, &mut row_emits);
            Self::push_emits(&mut body, row_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, &op.a, &[row, k])?;
            Self::push_emits(&mut body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
            Self::push_emits(&mut body, a_pointer_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let sum_pointer = expressions.append(
                Expression::LocalVariable(self.gemv_sum_local(scratch, row_offset)?),
                Span::default(),
            );
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a_value,
                    arg1: Some(x_value),
                    arg2: Some(sum_value),
                    arg3: None,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
        }
        Ok(body)
    }

    pub(super) fn lower_gemv_partial_add(
        &self,
        expressions: &mut Arena<Expression>,
        partials: TileRef,
        local_invocation: Handle<Expression>,
        stride: u32,
        rows_per_workgroup: u32,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        for row_offset in 0..rows_per_workgroup {
            let base = row_offset * self.workgroup_invocations;
            let mut lhs_emits = Vec::new();
            let lhs_index =
                self.add_literal_u32_emitted(expressions, local_invocation, base, &mut lhs_emits);
            Self::push_emits(&mut body, lhs_emits);
            let mut rhs_emits = Vec::new();
            let rhs_index = self.add_literal_u32_emitted(
                expressions,
                local_invocation,
                base + stride,
                &mut rhs_emits,
            );
            Self::push_emits(&mut body, rhs_emits);
            let (lhs_pointer, lhs_pointer_emits) =
                self.tile_dynamic_pointer(expressions, partials, lhs_index)?;
            let (rhs_pointer, rhs_pointer_emits) =
                self.tile_dynamic_pointer(expressions, partials, rhs_index)?;
            Self::push_emits(&mut body, lhs_pointer_emits);
            Self::push_emits(&mut body, rhs_pointer_emits);

            let lhs_value = expressions.append(
                Expression::Load {
                    pointer: lhs_pointer,
                },
                Span::default(),
            );
            let rhs_value = expressions.append(
                Expression::Load {
                    pointer: rhs_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: lhs_value,
                    right: rhs_value,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::range_from(expressions, lhs_value, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: lhs_pointer,
                    value,
                },
                Span::default(),
            );
        }
        Ok(body)
    }

    pub(super) fn lower_gemv_final_store(
        &self,
        expressions: &mut Arena<Expression>,
        op: &GemvOp,
        row_base: Handle<Expression>,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        for row_offset in 0..op.rows_per_workgroup {
            let base = row_offset * self.workgroup_invocations;
            let partial_0_index =
                expressions.append(Expression::Literal(Literal::U32(base)), Span::default());
            let partial_1_index =
                expressions.append(Expression::Literal(Literal::U32(base + 1)), Span::default());
            let (partial_0, partial_0_emits) =
                self.tile_dynamic_pointer(expressions, op.partials, partial_0_index)?;
            let (partial_1, partial_1_emits) =
                self.tile_dynamic_pointer(expressions, op.partials, partial_1_index)?;
            Self::push_emits(&mut body, partial_0_emits);
            Self::push_emits(&mut body, partial_1_emits);
            let value_0 =
                expressions.append(Expression::Load { pointer: partial_0 }, Span::default());
            let value_1 =
                expressions.append(Expression::Load { pointer: partial_1 }, Span::default());
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: value_0,
                    right: value_1,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::range_from(expressions, value_0, value)),
                Span::default(),
            );

            let mut row_emits = Vec::new();
            let row =
                self.add_literal_u32_emitted(expressions, row_base, row_offset, &mut row_emits);
            Self::push_emits(&mut body, row_emits);
            let zero_col =
                expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            let (y_index, y_index_emits) =
                self.storage_index_from_coords(expressions, &op.y, &[row, zero_col])?;
            Self::push_emits(&mut body, y_index_emits);
            let (y_pointer, y_pointer_emits) =
                self.storage_dynamic_pointer(expressions, &op.y, y_index)?;
            Self::push_emits(&mut body, y_pointer_emits);
            body.push(
                Statement::Store {
                    pointer: y_pointer,
                    value,
                },
                Span::default(),
            );
        }
        Ok(body)
    }

    pub(super) fn gemv_sum_local(
        &self,
        scratch: ScratchLocals,
        row_offset: u32,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        match row_offset {
            0 => Ok(scratch.mma_sum),
            1 => Ok(scratch.mma_sum_1),
            2 => scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
                "missing gemv scratch local",
            )),
            3 => scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
                "missing gemv scratch local",
            )),
            _ => Err(LowerError::UnsupportedOperation(
                "gemv rows per workgroup must be between 1 and 4",
            )),
        }
    }

    pub(super) fn gemm_sum_local(
        &self,
        scratch: ScratchLocals,
        index: u32,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        match index {
            0 => Ok(scratch.mma_sum),
            1 => Ok(scratch.mma_sum_1),
            2 => scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            3 => scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            4 => scratch.mma_sum_4.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            5 => scratch.mma_sum_5.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            6 => scratch.mma_sum_6.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            7 => scratch.mma_sum_7.ok_or(LowerError::UnsupportedOperation(
                "missing gemm scratch local",
            )),
            _ => Err(LowerError::UnsupportedOperation(
                "gemm microtile width is too large",
            )),
        }
    }
}
