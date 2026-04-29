use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_mma(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &MmaOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("mma shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }

        let mut j_body = Block::new();
        let (i, i_emit) = self.load_u32_local(expressions, scratch.mma_i);
        let (j, j_emit) = self.load_u32_local(expressions, scratch.mma_j);
        j_body.push(Statement::Emit(i_emit), Span::default());
        j_body.push(Statement::Emit(j_emit), Span::default());

        let (acc_index, acc_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[i, j])?;
        Self::push_emits(&mut j_body, acc_index_emits);
        let (_, acc_offset) = self.storage_tile_and_offset(op.acc)?;
        let mut acc_owner_emits = Vec::new();
        let acc_owner_index =
            self.add_literal_u32_emitted(expressions, acc_index, acc_offset, &mut acc_owner_emits);
        Self::push_emits(&mut j_body, acc_owner_emits);
        let (acc_pointer, acc_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc_index)?;
        Self::push_emits(&mut j_body, acc_pointer_emits);
        let local_invocation = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let owns_acc_element = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Equal,
                left: local_invocation,
                right: acc_owner_index,
            },
            Span::default(),
        );
        j_body.push(
            Statement::Emit(Self::single_expression_range(expressions, owns_acc_element)),
            Span::default(),
        );

        let mut k_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (a_index, a_index_emits) = self.layout_index_expr(expressions, a_layout, &[i, k])?;
        let (b_index, b_index_emits) = self.layout_index_expr(expressions, b_layout, &[k, j])?;
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (a_pointer, a_pointer_emits) = self.tile_dynamic_pointer(expressions, op.a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.tile_dynamic_pointer(expressions, op.b, b_index)?;
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let acc_value = expressions.append(
            Expression::Load {
                pointer: acc_pointer,
            },
            Span::default(),
        );
        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc_value)),
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
        let product = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: a_value,
                right: b_value,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, product)),
            Span::default(),
        );
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: acc_value,
                right: product,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: acc_pointer,
                value,
            },
            Span::default(),
        );

        let k_loop = self.counted_loop(expressions, scratch.mma_k, k_a, k_body);
        j_body.push(
            Statement::If {
                condition: owns_acc_element,
                accept: Block::from_vec(vec![k_loop]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let j_loop = self.counted_loop(expressions, scratch.mma_j, n, j_body);
        let i_loop =
            self.counted_loop(expressions, scratch.mma_i, m, Block::from_vec(vec![j_loop]));

        Ok(i_loop)
    }

    pub(super) fn store_zero_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<Statement, LowerError> {
        self.lower_workgroup_tile_op(expressions, tile_index, tile, |this, expressions, index| {
            let (pointer, pointer_emits) = this.tile_index_pointer(expressions, index, tile)?;
            let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
            let mut body = Block::new();
            Self::push_emits(&mut body, pointer_emits);
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
            Ok(body)
        })
    }

    pub(super) fn lower_cooperative_load(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst: TileRef,
        src: &StorageView,
    ) -> Result<Statement, LowerError> {
        if let Some(statement) =
            self.try_lower_cooperative_load_vec4(expressions, tile_index, dst, src)?
        {
            return Ok(statement);
        }

        self.lower_workgroup_tile_op(expressions, tile_index, dst, |this, expressions, index| {
            let src_base = this.storage_base_expression(expressions, src)?;
            let dst_layout = this.tile_layout(dst)?;
            let (dst_pointer, dst_emits) = this.tile_index_pointer(expressions, index, dst)?;
            let (src_pointer, src_emits) = this.storage_index_pointer_from_tile_index_with_base(
                expressions,
                index,
                dst_layout,
                src,
                src_base,
            )?;
            let value = expressions.append(
                Expression::Load {
                    pointer: src_pointer,
                },
                Span::default(),
            );

            let mut body = Block::new();
            Self::push_emits(&mut body, dst_emits);
            for emit in src_emits {
                body.push(Statement::Emit(emit), Span::default());
            }
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value,
                },
                Span::default(),
            );
            Ok(body)
        })
    }

    pub(super) fn try_lower_cooperative_load_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst: TileRef,
        src: &StorageView,
    ) -> Result<Option<Statement>, LowerError> {
        let dst_layout = self.tile_layout(dst)?;
        let src_layout = self.storage_layout(src)?;
        if dst_layout.shape() != src_layout.shape()
            || dst_layout.shape().rank() != 2
            || dst_layout.strides().values()[1] != 1
            || src_layout.strides().values()[1] != 1
        {
            return Ok(None);
        }
        let rows = dst_layout.shape().dims()[0].get();
        let cols = dst_layout.shape().dims()[1].get();
        if rows == 0 || cols == 0 || cols % COOPERATIVE_LOAD_WIDTH != 0 {
            return Ok(None);
        }
        let groups_per_row = cols / COOPERATIVE_LOAD_WIDTH;
        let Some(groups) = std::num::NonZeroU32::new(rows * groups_per_row) else {
            return Ok(None);
        };

        let src_base = self.storage_base_expression(expressions, src)?;
        let (src_dynamic_base, base_emits) = self.storage_dynamic_base_index(expressions, src)?;
        let mut prelude = Block::new();
        Self::push_emits(&mut prelude, base_emits);

        let mut body = Block::new();
        let (group, group_emit) = self.load_u32_local(expressions, tile_index);
        body.push(Statement::Emit(group_emit), Span::default());
        let mut emits = Vec::new();
        let row = self.div_literal_u32_emitted(expressions, group, groups_per_row, &mut emits);
        let col_group =
            self.mod_literal_u32_emitted(expressions, group, groups_per_row, &mut emits);
        let col0 = self.mul_literal_u32_emitted(
            expressions,
            col_group,
            COOPERATIVE_LOAD_WIDTH,
            &mut emits,
        );
        Self::push_emits(&mut body, emits);

        for lane in 0..COOPERATIVE_LOAD_WIDTH {
            let mut lane_emits = Vec::new();
            let col = self.add_literal_u32_emitted(expressions, col0, lane, &mut lane_emits);
            let (src_index, src_index_emits) =
                self.layout_index_expr(expressions, src_layout, &[row, col])?;
            lane_emits.extend(src_index_emits);
            let src_index = self.add_optional_base_u32_emitted(
                expressions,
                src_index,
                src_dynamic_base,
                &mut lane_emits,
            );
            let src_pointer = expressions.append(
                Expression::Access {
                    base: src_base,
                    index: src_index,
                },
                Span::default(),
            );
            lane_emits.push(Self::single_expression_range(expressions, src_pointer));
            let (dst_index, dst_index_emits) =
                self.layout_index_expr(expressions, dst_layout, &[row, col])?;
            lane_emits.extend(dst_index_emits);
            let (dst_pointer, dst_pointer_emits) =
                self.tile_dynamic_pointer(expressions, dst, dst_index)?;
            lane_emits.extend(dst_pointer_emits);
            Self::push_emits(&mut body, lane_emits);

            let value = expressions.append(
                Expression::Load {
                    pointer: src_pointer,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value,
                },
                Span::default(),
            );
        }

        prelude.push(
            self.distributed_index_loop(expressions, tile_index, groups, body),
            Span::default(),
        );
        Ok(Some(Statement::Block(prelude)))
    }
}
