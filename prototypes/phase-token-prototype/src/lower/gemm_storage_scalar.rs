use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_storage_gemm_loop_to_storage(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        op: &GemmOp,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemm loop iteration count must be non-zero",
            ));
        }
        if PREFER_COOP_MATRIX_GEMM
            && Self::can_lower_storage_gemm_coop8(
                a_layout,
                b_layout,
                acc_layout,
                dst_layout,
                outer_iterations,
            )
        {
            return self.lower_storage_gemm_loop_to_storage_coop8(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
            );
        }
        if n % 8 == 0 && k_a % 4 == 0 {
            return self.lower_storage_gemm_loop_to_storage_widecol(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
                8,
            );
        }
        if n % 4 == 0 && k_a % 4 == 0 {
            return self.lower_storage_gemm_loop_to_storage_4col(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
            );
        }
        if n % 2 == 0 && k_a % 4 == 0 {
            return self.lower_storage_gemm_loop_to_storage_2col(
                expressions,
                scratch,
                a,
                b,
                dst,
                outer_iterations,
            );
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(flat_emit), Span::default());

        let cols = expressions.append(Expression::Literal(Literal::U32(n)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        let col = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: flat,
                right: cols,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, row, col)),
            Span::default(),
        );

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_pointer,
                value: zero,
            },
            Span::default(),
        );

        let mut outer_body = Block::new();
        let (inner_body, inner_iterations) = if k_a % 4 == 0 {
            let mut inner_body = Block::new();
            let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            inner_body.push(Statement::Emit(k_emit), Span::default());
            let mut emits = Vec::new();
            let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
            Self::push_emits(&mut inner_body, emits);

            let mut a_values = Vec::with_capacity(4);
            let mut b_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let mut lane_emits = Vec::new();
                let k = self.add_literal_u32_emitted(expressions, base_k, lane, &mut lane_emits);
                let (a_index, a_index_emits) =
                    self.storage_index_from_coords(expressions, a, &[row, k])?;
                let (b_index, b_index_emits) =
                    self.storage_index_from_coords(expressions, b, &[k, col])?;
                lane_emits.extend(a_index_emits);
                lane_emits.extend(b_index_emits);
                let (a_pointer, a_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, a, a_index)?;
                let (b_pointer, b_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, b, b_index)?;
                lane_emits.extend(a_pointer_emits);
                lane_emits.extend(b_pointer_emits);
                Self::push_emits(&mut inner_body, lane_emits);

                let a_value =
                    expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
                let b_value =
                    expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
                inner_body.push(
                    Statement::Emit(Self::range_from(expressions, a_value, b_value)),
                    Span::default(),
                );
                a_values.push(a_value);
                b_values.push(b_value);
            }

            let a_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: a_values,
                },
                Span::default(),
            );
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: b_values,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, a_vec, dot)),
                Span::default(),
            );

            let sum_pointer =
                expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
            let sum_value = expressions.append(
                Expression::Load {
                    pointer: sum_pointer,
                },
                Span::default(),
            );
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: sum_value,
                    right: dot,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, sum_value, value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (inner_body, k_a / 4)
        } else {
            let mut inner_body = Block::new();
            let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            inner_body.push(Statement::Emit(k_emit), Span::default());

            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            let (b_index, b_index_emits) =
                self.storage_index_from_coords(expressions, b, &[k, col])?;
            Self::push_emits(&mut inner_body, a_index_emits);
            Self::push_emits(&mut inner_body, b_index_emits);

            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            let (b_pointer, b_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b_index)?;
            Self::push_emits(&mut inner_body, a_pointer_emits);
            Self::push_emits(&mut inner_body, b_pointer_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b_value =
                expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
            let sum_pointer =
                expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
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
                    arg1: Some(b_value),
                    arg2: Some(sum_value),
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (inner_body, k_a)
        };

        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, inner_iterations, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
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

        let (dst_index, dst_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col])?;
        Self::push_emits(&mut body, dst_index_emits);
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, dst_pointer_emits);
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value: sum_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(
            expressions,
            scratch.linear_index,
            acc_layout.element_count(),
            body,
        ))
    }

    pub(super) fn lower_storage_gemm_loop_to_storage_2col(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        if k_a != k_b || m != m_dst || n != n_dst || n % 2 != 0 || k_a % 4 != 0 {
            return Err(LowerError::UnsupportedOperation(
                "2-column gemm shape mismatch",
            ));
        }
        let lanes = std::num::NonZeroU32::new(m * (n / 2))
            .ok_or(LowerError::UnsupportedOperation("empty 2-column gemm"))?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_pairs =
            expressions.append(Expression::Literal(Literal::U32(n / 2)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_pairs,
            },
            Span::default(),
        );
        let col_pair = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_pairs,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_pair, 2);
        let col1 = self.add_literal_u32(expressions, col0, 1);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col_pair)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col0)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col1)),
            Span::default(),
        );

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: zero,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: zero,
            },
            Span::default(),
        );

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut inner_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            let (b0_index, b0_index_emits) =
                self.storage_index_from_coords(expressions, b, &[k, col0])?;
            let (b1_index, b1_index_emits) =
                self.storage_index_from_coords(expressions, b, &[k, col1])?;
            lane_emits.extend(a_index_emits);
            lane_emits.extend(b0_index_emits);
            lane_emits.extend(b1_index_emits);

            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            let (b0_pointer, b0_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b0_index)?;
            let (b1_pointer, b1_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b1_index)?;
            lane_emits.extend(a_pointer_emits);
            lane_emits.extend(b0_pointer_emits);
            lane_emits.extend(b1_pointer_emits);
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b0_value = expressions.append(
                Expression::Load {
                    pointer: b0_pointer,
                },
                Span::default(),
            );
            let b1_value = expressions.append(
                Expression::Load {
                    pointer: b1_pointer,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b0_value)),
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b1_value)),
                Span::default(),
            );
            a_values.push(a_value);
            b0_values.push(b0_value);
            b1_values.push(b1_value);
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        let b0_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: b0_values,
            },
            Span::default(),
        );
        let b1_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: b1_values,
            },
            Span::default(),
        );
        let dot0 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a_vec,
                arg1: Some(b0_vec),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        let dot1 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a_vec,
                arg1: Some(b1_vec),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b0_vec)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b1_vec)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot0)),
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot1)),
            Span::default(),
        );

        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        let sum0_value = expressions.append(
            Expression::Load {
                pointer: sum0_pointer,
            },
            Span::default(),
        );
        let sum1_value = expressions.append(
            Expression::Load {
                pointer: sum1_pointer,
            },
            Span::default(),
        );
        let value0 = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: sum0_value,
                right: dot0,
            },
            Span::default(),
        );
        let value1 = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: sum1_value,
                right: dot1,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, value1)),
            Span::default(),
        );
        inner_body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: value0,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: value1,
            },
            Span::default(),
        );

        let mut outer_body = Block::new();
        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        let sum0_value = expressions.append(
            Expression::Load {
                pointer: sum0_pointer,
            },
            Span::default(),
        );
        let sum1_value = expressions.append(
            Expression::Load {
                pointer: sum1_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, sum1_value)),
            Span::default(),
        );

        let (dst0_index, dst0_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col0])?;
        let (dst1_index, dst1_index_emits) =
            self.storage_index_from_coords(expressions, dst, &[row, col1])?;
        Self::push_emits(&mut body, dst0_index_emits);
        Self::push_emits(&mut body, dst1_index_emits);
        let (dst0_pointer, dst0_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst0_index)?;
        let (dst1_pointer, dst1_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst1_index)?;
        Self::push_emits(&mut body, dst0_pointer_emits);
        Self::push_emits(&mut body, dst1_pointer_emits);
        body.push(
            Statement::Store {
                pointer: dst0_pointer,
                value: sum0_value,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: dst1_pointer,
                value: sum1_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }

    pub(super) fn lower_storage_gemm_loop_to_storage_4col(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        if k_a != k_b || m != m_dst || n != n_dst || n % 4 != 0 || k_a % 4 != 0 {
            return Err(LowerError::UnsupportedOperation(
                "4-column gemm shape mismatch",
            ));
        }
        let sum2 = scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;
        let sum3 = scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;
        let lanes = std::num::NonZeroU32::new(m * (n / 4))
            .ok_or(LowerError::UnsupportedOperation("empty 4-column gemm"))?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_quads =
            expressions.append(Expression::Literal(Literal::U32(n / 4)), Span::default());
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_quads,
            },
            Span::default(),
        );
        let col_quad = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_quads,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_quad, 4);
        let col1 = self.add_literal_u32(expressions, col0, 1);
        let col2 = self.add_literal_u32(expressions, col0, 2);
        let col3 = self.add_literal_u32(expressions, col0, 3);
        for value in [row, col_quad, col0, col1, col2, col3] {
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
        }

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        for sum in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3] {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut inner_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        let mut b2_values = Vec::with_capacity(4);
        let mut b3_values = Vec::with_capacity(4);
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            lane_emits.extend(a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            lane_emits.extend(a_pointer_emits);

            let mut b_pointers = Vec::with_capacity(4);
            for col in [col0, col1, col2, col3] {
                let (b_index, b_index_emits) =
                    self.storage_index_from_coords(expressions, b, &[k, col])?;
                lane_emits.extend(b_index_emits);
                let (b_pointer, b_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, b, b_index)?;
                lane_emits.extend(b_pointer_emits);
                b_pointers.push(b_pointer);
            }
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            let b0_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[0],
                },
                Span::default(),
            );
            let b1_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[1],
                },
                Span::default(),
            );
            let b2_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[2],
                },
                Span::default(),
            );
            let b3_value = expressions.append(
                Expression::Load {
                    pointer: b_pointers[3],
                },
                Span::default(),
            );
            for value in [a_value, b0_value, b1_value, b2_value, b3_value] {
                inner_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, value)),
                    Span::default(),
                );
            }
            a_values.push(a_value);
            b0_values.push(b0_value);
            b1_values.push(b1_value);
            b2_values.push(b2_value);
            b3_values.push(b3_value);
        }

        let a_vec = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a_values,
            },
            Span::default(),
        );
        inner_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );

        let mut dots = Vec::with_capacity(4);
        for components in [b0_values, b1_values, b2_values, b3_values] {
            let b_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components,
                },
                Span::default(),
            );
            let dot = expressions.append(
                Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a_vec,
                    arg1: Some(b_vec),
                    arg2: None,
                    arg3: None,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, b_vec, dot)),
                Span::default(),
            );
            dots.push(dot);
        }

        for (sum, dot) in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3]
            .into_iter()
            .zip(dots)
        {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            let current = expressions.append(Expression::Load { pointer }, Span::default());
            let next = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: current,
                    right: dot,
                },
                Span::default(),
            );
            inner_body.push(
                Statement::Emit(Self::range_from(expressions, current, next)),
                Span::default(),
            );
            inner_body.push(
                Statement::Store {
                    pointer,
                    value: next,
                },
                Span::default(),
            );
        }

        let mut outer_body = Block::new();
        outer_body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, inner_body),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        for (sum, col) in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3]
            .into_iter()
            .zip([col0, col1, col2, col3])
        {
            let sum_pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
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
            let (dst_index, dst_index_emits) =
                self.storage_index_from_coords(expressions, dst, &[row, col])?;
            Self::push_emits(&mut body, dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.storage_dynamic_pointer(expressions, dst, dst_index)?;
            Self::push_emits(&mut body, dst_pointer_emits);
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }
}
