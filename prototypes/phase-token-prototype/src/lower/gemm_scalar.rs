use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_gemm(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemmDescriptor,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("gemm shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
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

        let (acc_index, acc_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[row, col])?;
        Self::push_emits(&mut body, acc_index_emits);
        let (acc_pointer, acc_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc_index)?;
        Self::push_emits(&mut body, acc_pointer_emits);
        let acc_value = expressions.append(
            Expression::Load {
                pointer: acc_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc_value)),
            Span::default(),
        );
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_pointer,
                value: acc_value,
            },
            Span::default(),
        );

        let (k_body, k_iterations) = if k_a % 4 == 0 {
            let mut k_body = Block::new();
            let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            k_body.push(Statement::Emit(k_emit), Span::default());
            let mut emits = Vec::new();
            let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
            Self::push_emits(&mut k_body, emits);

            let mut a_values = Vec::with_capacity(4);
            let mut b_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let mut lane_emits = Vec::new();
                let k = self.add_literal_u32_emitted(expressions, base_k, lane, &mut lane_emits);
                let (a_index, a_index_emits) =
                    self.layout_index_expr(expressions, a_layout, &[row, k])?;
                let (b_index, b_index_emits) =
                    self.layout_index_expr(expressions, b_layout, &[k, col])?;
                lane_emits.extend(a_index_emits);
                lane_emits.extend(b_index_emits);
                let (a_pointer, a_pointer_emits) =
                    self.tile_dynamic_pointer(expressions, op.a, a_index)?;
                let (b_pointer, b_pointer_emits) =
                    self.tile_dynamic_pointer(expressions, op.b, b_index)?;
                lane_emits.extend(a_pointer_emits);
                lane_emits.extend(b_pointer_emits);
                Self::push_emits(&mut k_body, lane_emits);

                let a_value =
                    expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
                let b_value =
                    expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
                k_body.push(
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
            k_body.push(
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
            k_body.push(
                Statement::Emit(Self::range_from(expressions, sum_value, value)),
                Span::default(),
            );
            k_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (k_body, k_a / 4)
        } else {
            let mut k_body = Block::new();
            let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            k_body.push(Statement::Emit(k_emit), Span::default());

            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, k])?;
            let (b_index, b_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[k, col])?;
            Self::push_emits(&mut k_body, a_index_emits);
            Self::push_emits(&mut k_body, b_index_emits);

            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.a, a_index)?;
            let (b_pointer, b_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.b, b_index)?;
            Self::push_emits(&mut k_body, a_pointer_emits);
            Self::push_emits(&mut k_body, b_pointer_emits);

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
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
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
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            k_body.push(
                Statement::Store {
                    pointer: sum_pointer,
                    value,
                },
                Span::default(),
            );
            (k_body, k_a)
        };

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_iterations, k_body),
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
        body.push(
            Statement::Store {
                pointer: acc_pointer,
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

    #[allow(dead_code)]
    pub(super) fn lower_gemm_2col_microtile(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemmDescriptor,
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
    ) -> Result<Statement, LowerError> {
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [_, n] = Self::matrix_shape(b_layout)?;
        let lanes = std::num::NonZeroU32::new(m * (n / 2))
            .ok_or(LowerError::UnsupportedOperation("empty microtile"))?;

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

        let (acc0_index, acc0_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[row, col0])?;
        let (acc1_index, acc1_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[row, col1])?;
        Self::push_emits(&mut body, acc0_index_emits);
        Self::push_emits(&mut body, acc1_index_emits);
        let (acc0_pointer, acc0_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc0_index)?;
        let (acc1_pointer, acc1_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc1_index)?;
        Self::push_emits(&mut body, acc0_pointer_emits);
        Self::push_emits(&mut body, acc1_pointer_emits);
        let acc0_value = expressions.append(
            Expression::Load {
                pointer: acc0_pointer,
            },
            Span::default(),
        );
        let acc1_value = expressions.append(
            Expression::Load {
                pointer: acc1_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, acc0_value, acc1_value)),
            Span::default(),
        );
        let sum0_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum1_pointer = expressions.append(
            Expression::LocalVariable(scratch.mma_sum_1),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: acc0_value,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: acc1_value,
            },
            Span::default(),
        );

        let mut k_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());
        let mut emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 4, &mut emits);
        Self::push_emits(&mut k_body, emits);

        let mut a_values = Vec::with_capacity(4);
        let mut b0_values = Vec::with_capacity(4);
        let mut b1_values = Vec::with_capacity(4);
        for lane in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, k])?;
            let (b0_index, b0_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[k, col0])?;
            let (b1_index, b1_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[k, col1])?;
            lane_emits.extend(a_index_emits);
            lane_emits.extend(b0_index_emits);
            lane_emits.extend(b1_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.a, a_index)?;
            let (b0_pointer, b0_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.b, b0_index)?;
            let (b1_pointer, b1_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.b, b1_index)?;
            lane_emits.extend(a_pointer_emits);
            lane_emits.extend(b0_pointer_emits);
            lane_emits.extend(b1_pointer_emits);
            Self::push_emits(&mut k_body, lane_emits);

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
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            k_body.push(
                Statement::Emit(Self::single_expression_range(expressions, b0_value)),
                Span::default(),
            );
            k_body.push(
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
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_vec)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b0_vec)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b1_vec)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, dot0)),
            Span::default(),
        );
        k_body.push(
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
        k_body.push(
            Statement::Emit(Self::range_from(expressions, sum0_value, value1)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum0_pointer,
                value: value0,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum1_pointer,
                value: value1,
            },
            Span::default(),
        );

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a / 4, k_body),
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
        body.push(
            Statement::Store {
                pointer: acc0_pointer,
                value: sum0_value,
            },
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: acc1_pointer,
                value: sum1_value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, scratch.linear_index, lanes, body))
    }

    pub(super) fn lower_gemm_to_storage(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &GemmDescriptor,
        dst: &StorageView,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
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

        let mut k_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (a_index, a_index_emits) = self.layout_index_expr(expressions, a_layout, &[row, k])?;
        let (b_index, b_index_emits) = self.layout_index_expr(expressions, b_layout, &[k, col])?;
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (a_pointer, a_pointer_emits) = self.tile_dynamic_pointer(expressions, op.a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.tile_dynamic_pointer(expressions, op.b, b_index)?;
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b_value)),
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
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum_pointer,
                value,
            },
            Span::default(),
        );

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a, k_body),
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

    #[allow(dead_code)]
    pub(super) fn lower_storage_gemm_to_storage(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        op: &GemmDescriptor,
        dst: &StorageView,
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

        let mut k_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (a_index, a_index_emits) = self.storage_index_from_coords(expressions, a, &[row, k])?;
        let (b_index, b_index_emits) = self.storage_index_from_coords(expressions, b, &[k, col])?;
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (a_pointer, a_pointer_emits) = self.storage_dynamic_pointer(expressions, a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.storage_dynamic_pointer(expressions, b, b_index)?;
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        let sum_pointer =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum_value = expressions.append(
            Expression::Load {
                pointer: sum_pointer,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b_value)),
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
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: sum_pointer,
                value,
            },
            Span::default(),
        );

        body.push(
            self.counted_loop(expressions, scratch.mma_k, k_a, k_body),
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
}
