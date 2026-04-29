use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_storage_gemm_loop_to_storage_widecol(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
        columns: u32,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        if columns == 0
            || k_a != k_b
            || m != m_dst
            || n != n_dst
            || n % columns != 0
            || k_a % 4 != 0
        {
            return Err(LowerError::UnsupportedOperation(
                "wide-column gemm shape mismatch",
            ));
        }

        let lanes = std::num::NonZeroU32::new(m * (n / columns))
            .ok_or(LowerError::UnsupportedOperation("empty wide-column gemm"))?;
        let sum_locals = (0..columns)
            .map(|index| self.gemm_sum_local(scratch, index))
            .collect::<Result<Vec<_>, _>>()?;

        let mut body = Block::new();
        let (lane, lane_emit) = self.load_u32_local(expressions, scratch.linear_index);
        body.push(Statement::Emit(lane_emit), Span::default());

        let col_groups = expressions.append(
            Expression::Literal(Literal::U32(n / columns)),
            Span::default(),
        );
        let row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Divide,
                left: lane,
                right: col_groups,
            },
            Span::default(),
        );
        let col_group = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Modulo,
                left: lane,
                right: col_groups,
            },
            Span::default(),
        );
        let col0 = self.mul_literal_u32(expressions, col_group, columns);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, row)),
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, col_group)),
            Span::default(),
        );

        let mut cols = Vec::with_capacity(columns as usize);
        for offset in 0..columns {
            let col = self.add_literal_u32(expressions, col0, offset);
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, col)),
                Span::default(),
            );
            cols.push(col);
        }

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        for sum in &sum_locals {
            let pointer = expressions.append(Expression::LocalVariable(*sum), Span::default());
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
        let mut b_values_by_col: Vec<Vec<Handle<Expression>>> =
            (0..columns).map(|_| Vec::with_capacity(4)).collect();
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, a, &[row, k])?;
            lane_emits.extend(a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            lane_emits.extend(a_pointer_emits);

            let mut b_pointers = Vec::with_capacity(columns as usize);
            for col in &cols {
                let (b_index, b_index_emits) =
                    self.storage_index_from_coords(expressions, b, &[k, *col])?;
                lane_emits.extend(b_index_emits);
                let (b_pointer, b_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, b, b_index)?;
                lane_emits.extend(b_pointer_emits);
                b_pointers.push(b_pointer);
            }
            Self::push_emits(&mut inner_body, lane_emits);

            let a_value =
                expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
            inner_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            a_values.push(a_value);

            for (col_index, b_pointer) in b_pointers.into_iter().enumerate() {
                let b_value =
                    expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
                inner_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, b_value)),
                    Span::default(),
                );
                b_values_by_col[col_index].push(b_value);
            }
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

        for (sum, components) in sum_locals.iter().copied().zip(b_values_by_col) {
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

        for (sum, col) in sum_locals.into_iter().zip(cols) {
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
