use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_shared_gemm_loop_to_storage_4col(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a_load: &crate::CooperativeLoadOp,
        b_load: &crate::CooperativeLoadOp,
        op: &GemmOp,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Option<Statement>, LowerError> {
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
        if a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
        {
            return Ok(None);
        }
        if a_load.dst != op.a || b_load.dst != op.b {
            return Ok(None);
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "gemm loop iteration count must be non-zero",
            ));
        }
        if n % 4 != 0 || k_a % 4 != 0 {
            return Ok(None);
        }

        let lanes = std::num::NonZeroU32::new(m * (n / 4)).ok_or(
            LowerError::UnsupportedOperation("empty shared 4-column gemm"),
        )?;
        if lanes.get() > self.workgroup_invocations {
            return Ok(None);
        }

        let sum2 = scratch.mma_sum_2.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;
        let sum3 = scratch.mma_sum_3.ok_or(LowerError::UnsupportedOperation(
            "missing gemm scratch local",
        ))?;

        let mut body = Block::new();
        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let lane_limit = expressions.append(
            Expression::Literal(Literal::U32(lanes.get())),
            Span::default(),
        );
        let active = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Less,
                left: lane,
                right: lane_limit,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, active)),
            Span::default(),
        );

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
        let mut init_body = Block::new();
        for sum in [scratch.mma_sum, scratch.mma_sum_1, sum2, sum3] {
            let pointer = expressions.append(Expression::LocalVariable(sum), Span::default());
            init_body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::If {
                condition: active,
                accept: init_body,
                reject: Block::new(),
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
        let mut b2_values = Vec::with_capacity(4);
        let mut b3_values = Vec::with_capacity(4);
        for lane_index in 0..4 {
            let mut lane_emits = Vec::new();
            let k = self.add_literal_u32_emitted(expressions, base_k, lane_index, &mut lane_emits);
            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, k])?;
            lane_emits.extend(a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, op.a, a_index)?;
            lane_emits.extend(a_pointer_emits);

            let mut b_pointers = Vec::with_capacity(4);
            for col in [col0, col1, col2, col3] {
                let (b_index, b_index_emits) =
                    self.layout_index_expr(expressions, b_layout, &[k, col])?;
                lane_emits.extend(b_index_emits);
                let (b_pointer, b_pointer_emits) =
                    self.tile_dynamic_pointer(expressions, op.b, b_index)?;
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
            self.lower_cooperative_load(expressions, scratch.tile_index, a_load.dst, &a_load.src)?,
            Span::default(),
        );
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, b_load.dst, &b_load.src)?,
            Span::default(),
        );
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        outer_body.push(
            Statement::If {
                condition: active,
                accept: Block::from_vec(vec![self.counted_loop(
                    expressions,
                    scratch.mma_k,
                    k_a / 4,
                    inner_body,
                )]),
                reject: Block::new(),
            },
            Span::default(),
        );
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
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

        let mut store_body = Block::new();
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
            store_body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum_value)),
                Span::default(),
            );
            let (dst_index, dst_index_emits) =
                self.storage_index_from_coords(expressions, dst, &[row, col])?;
            Self::push_emits(&mut store_body, dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.storage_dynamic_pointer(expressions, dst, dst_index)?;
            Self::push_emits(&mut store_body, dst_pointer_emits);
            store_body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value: sum_value,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::If {
                condition: active,
                accept: store_body,
                reject: Block::new(),
            },
            Span::default(),
        );

        Ok(Some(Statement::Block(body)))
    }
}
